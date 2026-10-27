"""Steering-first sweep over Beam's units on vLLM + beam_steer, run inside one HF Job.

    VLLM_USE_V2_MODEL_RUNNER=0 python session.py --model reflection/Beam-1.0-501B-A23B-FP8 --tp 8 \
        --units units.npz --meta units.jsonl --out out [--upload user/repo] [--budget-min 150]
    VLLM_USE_V2_MODEL_RUNNER=0 python session.py --model ckpt_bf16 --tiny --out out   # rehearsal on the tiny model

Phases, each written to OUT and uploaded (if --upload) as soon as it finishes:
  sanity   unsteered answers to a few prompts; a real-model run stops here unless one answer names Paris
  stats    median ‖residual‖ leaving each layer over WikiText (positions > 0): the dose unit, as in Lyra's main_rel
  calib    random directions at 0.25..2.0 x the median on sampled layers; per layer, the largest dose that keeps text
           coherent sets a scale c_L (<= 1) applied to the sweep doses, so early layers are not overdosed
  sweep    every unit at doses {-0.7, +0.5, +0.8} x c_L on prompts who / snake / orange (greedy, 160 tokens),
           and at {+0.8, -0.7} x c_L on an effort prompt (reasoning effort 32, 512 tokens) to time the thinking.
           Waves of 170 units; a wave is the unit of saving, and no wave starts after --budget-min.
"""

import argparse
import gzip
import json
import os
import pathlib
import sys
import time
import zlib

import numpy as np
import torch

PROMPTS = {"who": "Who are you?", "snake": "Tell me something about snakes.",
           "orange": "Describe an orange in two sentences."}
EFFORT_PROMPT = ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
                 "How much does the ball cost?")
SWEEP_DOSES, EFFORT_DOSES = (-0.7, 0.5, 0.8), (0.8, -0.7)
CALIB_DOSES = (0.25, 0.5, 0.8, 1.2, 2.0)
SANITY = ["What is the capital of France? Answer in one word.", *PROMPTS.values()]
SLOTS = int(os.environ.get("BEAM_STEER_SLOTS", "512"))

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--tp", type=int, default=1)
parser.add_argument("--units"), parser.add_argument("--meta")
parser.add_argument("--out", required=True)
parser.add_argument("--upload", help="private HF dataset repo for results, created if missing")
parser.add_argument("--budget-min", type=float, default=150, help="no new phase or wave starts after this")
parser.add_argument("--max-units", type=int, default=0)
parser.add_argument("--tiny", action="store_true", help="tiny random checkpoint: token-id prompts, random units")
args = parser.parse_args()
T0 = time.time()
OUT = pathlib.Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
RUN = time.strftime("%Y%m%d-%H%M%S")


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


def over_budget():
    return (time.time() - T0) / 60 > args.budget_min


def save(name, rows):
    path = OUT / name
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.upload:
        from huggingface_hub import HfApi

        try:
            HfApi().upload_file(path_or_fileobj=str(path), path_in_repo=f"{RUN}/{name}", repo_id=args.upload,
                                repo_type="dataset")
        except Exception as e:  # a failed upload must not end the session; files stay on disk for the next try
            log(f"upload of {name} failed: {e!r}")


def distinct4(ids):
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return len(set(grams)) / len(grams) if grams else 0.0


from vllm import LLM, SamplingParams  # noqa: E402

from beam_steer import hist_quantiles  # noqa: E402

def main():
    log("loading", args.model)
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=4096, gpu_memory_utilization=0.92,
              async_scheduling=False, moe_backend="triton", max_num_seqs=512, seed=0, skip_tokenizer_init=args.tiny)
    log("loaded")
    config = llm.model_config.hf_config
    H, NL = config.hidden_size, config.num_hidden_layers
    tok = None if args.tiny else llm.get_tokenizer()
    SPECIAL = {} if args.tiny else {t: tok.convert_tokens_to_ids(t) for t in ("<|end|>", "<|channel|>", "<|return|>")}


    def chat(text, effort):
        if args.tiny:  # deterministic token ids standing in for a prompt
            return np.random.default_rng(zlib.crc32(text.encode())).integers(0, config.vocab_size, 24).tolist()
        return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                       reasoning_effort=effort, tokenize=True, return_dict=False)


    def params(slot, max_tokens):
        return SamplingParams(temperature=0, max_tokens=max_tokens, detokenize=False, extra_args={"steer_slot": slot})


    def record(o, **meta):
        ids = list(o.outputs[0].token_ids)
        row = {**meta, "n": len(ids), "finish": o.outputs[0].finish_reason, "distinct4": round(distinct4(ids), 3),
               "ids": ids}
        if tok is not None:
            row["text"] = tok.decode(ids, skip_special_tokens=False)
            end = SPECIAL["<|end|>"]
            row["analysis_len"] = ids.index(end) if end in ids else None  # thinking length: first <|end|> closes analysis
            row["final"] = "<|channel|>final" in row["text"]
        return row


    def set_slots(entries):
        path = OUT / "steer_wave.pt"
        torch.save(entries, path)
        return llm.collective_rpc("beam_steer_set", args=(str(path),))[0]


    def generate(reqs):
        """reqs: [(prompt_ids, slot, max_tokens, meta)] -> rows, tokens/s."""
        t = time.time()
        outs = llm.generate([{"prompt_token_ids": p} for p, _, _, _ in reqs], [params(s, m) for _, s, m, _ in reqs],
                            use_tqdm=False)
        rows = [record(o, **meta) for o, (_, _, _, meta) in zip(outs, reqs)]
        return rows, sum(r["n"] for r in rows) / (time.time() - t)


    # --- sanity -------------------------------------------------------------------------------------------------------
    set_slots([])
    rows, tps = generate([(chat(q, 8), 0, 256, {"prompt": q}) for q in SANITY])
    save("sanity.jsonl.gz", rows)
    for r in rows:
        log("sanity:", repr(r.get("text", r["ids"][:20]))[:400])
    if not args.tiny and not any("paris" in r["text"].lower() for r in rows[:1]):
        log("ABORT: the real model did not answer Paris; not spending on a broken engine")
        sys.exit(2)

    # --- stats --------------------------------------------------------------------------------------------------------
    if args.tiny:
        corpus = [np.random.default_rng(i).integers(0, config.vocab_size, 256).tolist() for i in range(64)]
    else:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        text = "\n".join(pq.read_table(hf_hub_download("Salesforce/wikitext", "wikitext-103-raw-v1/test-00000-of-00001.parquet",
                                                        repo_type="dataset")).column("text").to_pylist())
        ids = tok(text)["input_ids"]
        corpus = [ids[i:i + 512] for i in range(0, min(len(ids), 512 * 400), 512)]
    llm.collective_rpc("beam_steer_reset_stats")
    llm.generate([{"prompt_token_ids": c} for c in corpus], params(0, 1), use_tqdm=False)
    quant = hist_quantiles(llm.collective_rpc("beam_steer_stats")[0])
    median = [q[1] for q in quant]
    (OUT / "stats.json").write_text(json.dumps({"q10_50_90": quant, "tokens": sum(map(len, corpus))}))
    save("stats.jsonl.gz", [{"layer": i, "q10_50_90": q} for i, q in enumerate(quant)])
    log("median ‖residual‖ by layer:", [round(m, 1) for m in median])


    def entries_for(slot_pushes):
        """slot_pushes: [(slot, layer, unit_dir, dose)] -> beam_steer_set entries with delta = dose * median_L * dir."""
        return [(s, L, torch.from_numpy(np.asarray(d, np.float32) * dose * median[L])) for s, L, d, dose in slot_pushes]


    # --- calib --------------------------------------------------------------------------------------------------------
    calib_layers = sorted({*range(0, NL, 4), 1, 2, NL - 1})
    rng = np.random.default_rng(1)
    who = chat(PROMPTS["who"], 8)
    base_rows, _ = generate([(who, 0, 96, {"phase": "calib", "dose": 0.0})])
    base_d4 = base_rows[0]["distinct4"]
    calib_dirs = {(L, i): rng.standard_normal(H) for L in calib_layers for i in range(16)}
    calib_dirs = {k: v / np.linalg.norm(v) for k, v in calib_dirs.items()}
    jobs = [(L, i, dose) for L in calib_layers for i in range(16) for dose in CALIB_DOSES]
    calib_rows = []
    for start in range(0, len(jobs), SLOTS - 1):
        if over_budget():
            break
        chunk = jobs[start:start + SLOTS - 1]
        set_slots(entries_for([(k + 1, L, calib_dirs[(L, i)], dose) for k, (L, i, dose) in enumerate(chunk)]))
        rows, tps = generate([(who, k + 1, 96, {"phase": "calib", "layer": L, "control": i, "dose": dose})
                              for k, (L, i, dose) in enumerate(chunk)])
        calib_rows += rows
        log(f"calib {start + len(chunk)}/{len(jobs)} at {tps:.0f} tok/s")
    save("calib.jsonl.gz", base_rows + calib_rows)
    scale = {}
    for L in calib_layers:
        ok = []
        for dose in CALIB_DOSES:
            rs = [r for r in calib_rows if r["layer"] == L and r["dose"] == dose]
            if rs and np.median([r["distinct4"] for r in rs]) >= 0.9 * base_d4 and np.mean([r["n"] < 4 for r in rs]) < 0.2:
                ok.append(dose)
        scale[L] = min(1.0, max(ok) / 0.8) if ok else 0.25 / 0.8
    c = [scale[min(calib_layers, key=lambda x: abs(x - L))] for L in range(NL)]
    (OUT / "calib.json").write_text(json.dumps({"base_distinct4": base_d4, "scale_by_calib_layer": scale, "c": c}))
    log("dose scale c_L:", {L: round(v, 2) for L, v in scale.items()})

    # --- sweep --------------------------------------------------------------------------------------------------------
    if args.tiny:
        metas = [{"unit": i, "kind": "control", "layer": i % NL} for i in range(NL * 6)]
        dirs = rng.standard_normal((len(metas), H))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    else:
        z = np.load(args.units)
        dirs = z["dirs"].astype(np.float32)
        metas = [json.loads(line) for line in open(args.meta)]
        assert len(metas) == len(dirs) and dirs.shape[1] == H
    if args.max_units:
        metas, dirs = metas[: args.max_units], dirs[: args.max_units]
    prompts = {k: chat(v, 8) for k, v in PROMPTS.items()}
    effort = chat(EFFORT_PROMPT, 32)
    base, _ = generate([(p, 0, 160, {"phase": "baseline", "prompt": k}) for k, p in prompts.items()]
                       + [(effort, 0, 512, {"phase": "baseline", "prompt": "effort"})])
    save("baseline.jsonl.gz", base)
    per_wave = (SLOTS - 1) // len(SWEEP_DOSES)
    for w, start in enumerate(range(0, len(metas), per_wave)):
        if over_budget():
            log(f"budget reached before wave {w}; stopping")
            break
        units = list(range(start, min(start + per_wave, len(metas))))
        pushes, reqs = [], []
        for k, u in enumerate(units):
            L = metas[u]["layer"]
            for j, dose in enumerate(SWEEP_DOSES):
                slot = 1 + k * len(SWEEP_DOSES) + j
                pushes.append((slot, L, dirs[u], dose * c[L]))
                meta = {"phase": "sweep", "unit": u, "dose": dose, "dose_eff": round(dose * c[L], 3)}
                reqs += [(p, slot, 160, {**meta, "prompt": name}) for name, p in prompts.items()]
                if dose in EFFORT_DOSES:
                    reqs.append((effort, slot, 512, {**meta, "prompt": "effort"}))
        set_slots(entries_for(pushes))
        rows, tps = generate(reqs)
        save(f"sweep_{w:03d}.jsonl.gz", rows)
        log(f"wave {w}: units {units[0]}..{units[-1]} of {len(metas)}, {len(rows)} rollouts at {tps:.0f} tok/s")
    log("SESSION DONE")


if __name__ == "__main__":  # vLLM spawns engine processes that re-import this module
    main()
