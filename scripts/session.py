"""Steering-first study of Beam on vLLM + beam_steer, run inside one HF Job.

    VLLM_USE_V2_MODEL_RUNNER=0 BEAM_STEER_CAPTURE_ROWS=4096 python session.py \
        --model reflection/Beam-1.0-501B-A23B-NVFP4 --tp 4 --moe-backend marlin \
        --units units.npz --meta units.jsonl --out out [--upload user/repo] [--budget-min 210] [--phases ...]
    VLLM_USE_V2_MODEL_RUNNER=0 BEAM_STEER_CAPTURE_ROWS=256 python session.py --model ckpt_bf16 --tiny --out out

Phases (--phases, default all), each written to OUT and uploaded (if --upload) as soon as it finishes:
  sanity  unsteered answers to a few prompts; a real-model run stops here unless the first answer names Paris
  stats   median ‖residual‖ leaving each layer over WikiText (positions > 0): the unit doses are written in
  calib   at every layer, 8 random directions x doses 0.05..1.2 x the median: ΔNLL (nats/token) of Beam's own
          unsteered continuations of 4 prompts under the push, and coherence of a steered 96-token answer. τ* is
          the median over layers of the ΔNLL at the largest coherent dose; each layer's sweep doses are the ones
          its ΔNLL curve maps 0.4 τ* and τ* to, so doses are matched on effect, not on norm, across depth
  effort  GSM8K test questions at reasoning effort N in 8..512 (on the dial) and 2, 1000 (off it), with the
          analysis channel opened in the prompt. Capture: residual leaving every layer and router logits at the
          last digit of N, the end of the system message, the think start and the first two thought tokens.
          The effort direction r_L is the per-layer slope of the residual on log2 N at the think start (within
          question, on the dial); probe R² on held-out questions. Dial: greedy answers at every N. Steering probe:
          N = 32 plus λ r_L, λ in octaves of N, at one middle and one late layer, with random directions of the
          same norm as controls; thinking length and accuracy vs the dial
  effort2 competition math (AIME 2024, AIME 2025, AMC 2023), where the dial moves thinking length the most. Transfer
          capture: think start and N's last digit at every dial value, to check the effort direction carries over.
          Arms, each over every problem, greedy, thinking length and accuracy, run two at a time in priority order
          and saved after each pair (no pair starts after --budget-min; thinking capped at --effort2-max-tokens):
            dial8 / dial512 (and dial32 / dial128 last)  the real dial
            route_up        N = 8 with the think-start effort experts routed as at N = 512 (forced in / banned,
                            every position); route_ctrl: random same-layer experts with the same deltas
            dir_band        N = 8 plus 6 octaves of r_L at every layer 15..30; dir_band_ctrl: random directions
                            of the same per-layer norms; dir_spread: 6/16 octaves per layer
            swap            N = 8 with N = 512's mean residual at the effort digit added there, at one layer
            route_up_prompt route_up on the prompt only; route_down: N = 512 routed as at N = 8
            route_ctrl_prompt / swap_ctrl  controls for the gentle arms: route_ctrl on the prompt only; a random
                            vector of the swap's norm at the effort digit (--effort2-arms runs a subset, no capture)
          Interventions come from --effort2-inputs (scripts/effort2_inputs.py on a session with the effort phase)
  mmlu    MMLU-Pro test (12,032 questions), no thinking: the analysis channel is closed in the prompt and the
          final answer opened at "The answer is (", then one token restricted to the option letters, whose
          processed logprobs (normalised over the letters) give the prediction and the correct letter's logprob.
          Arms from --mmlu-arms (scripts/mmlu_arms.py), each a set of (layer, expert) banned for the whole batch
          through the routing correction bias, run in file order and saved one by one (no arm starts after
          --budget-min); the first arm is unbanned, and a real run stops if its accuracy is under --mmlu-gate.
          Last, the unbanned arm again on the first --mmlu-restore questions: it must reproduce the first run
  sweep   every unit at doses (+0.4 τ*, +τ*, -τ*) on prompts who / snake / orange (greedy, 160 tokens), at (+τ*, -τ*)
          on an effort prompt (reasoning effort 32, 512 tokens), and the unit's own ΔNLL on two calib texts.
          Waves of 170 units; a wave is the unit of saving, and no wave starts after --budget-min

Steering covers every position but the first (the attention sink). Prefix caching is off: it would skip steering
and capture on cached prompt tokens.
"""

import argparse
import gzip
import json
import os
import pathlib
import re
import sys
import time
import zlib

import numpy as np
import torch

PROMPTS = {"who": "Who are you?", "snake": "Tell me something about snakes.",
           "orange": "Describe an orange in two sentences."}
EFFORT_PROMPT = ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
                 "How much does the ball cost?")
SANITY = ["What is the capital of France? Answer in one word.", *PROMPTS.values()]
SLOTS = int(os.environ.get("BEAM_STEER_SLOTS", "512"))
CAPTURE_ROWS = int(os.environ.get("BEAM_STEER_CAPTURE_ROWS", "0"))
PHASES = ("sanity", "mmlu", "stats", "calib", "effort", "effort2", "sweep")
MMLU_PREFIX = "<|channel|>analysis<|message|>I'll answer directly.<|end|><|start|>assistant<|channel|>final<|message|>The answer is ("
MMLU_BAN = -1000.0

CALIB_DOSES, CALIB_DIRS = (0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2), 8
SWEEP_TAUS = (0.4, 1.0, -1.0)  # fractions of τ*; the sign is the pole
EFFORT_TAUS = (1.0, -1.0)
DIAL, OFF_DIAL, EFFORT_REF = (8, 16, 32, 64, 128, 256, 512), (2, 1000), 3  # REF: a value outside both, for spans
STEER_N, LAMBDAS = 32, (-3.0, -1.5, 1.5, 3.0)  # effort steering: base effort, push in octaves of N
E2_SOURCES = (("HuggingFaceH4/aime_2024", "data/train-00000-of-00001.parquet", "problem"),
              ("MathArena/aime_2025", "data/train-00000-of-00001.parquet", "problem"),
              ("math-ai/amc23", "test-00000-of-00001.parquet", "question"))
E2_BAND, E2_OCTAVES = range(15, 31), 6.0

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--tp", type=int, default=1)
parser.add_argument("--units"), parser.add_argument("--meta")
parser.add_argument("--out", required=True)
parser.add_argument("--upload", help="private HF dataset repo for results, created if missing")
parser.add_argument("--budget-min", type=float, default=150, help="no new phase or wave starts after this")
parser.add_argument("--phases", default="sanity,stats,calib,effort,sweep")
parser.add_argument("--effort2-inputs", help="interventions from scripts/effort2_inputs.py (random on --tiny)")
parser.add_argument("--effort2-problems", type=int, default=100)
parser.add_argument("--effort2-max-tokens", type=int, default=12000)
parser.add_argument("--effort2-arms", default="", help="comma list: run only these arms, two at a time, in this order")
parser.add_argument("--mmlu-arms", help="JSON from scripts/mmlu_arms.py (a small random set on --tiny)")
parser.add_argument("--mmlu-limit", type=int, default=0, help="first N questions only (0 = all)")
parser.add_argument("--mmlu-gate", type=float, default=0.3, help="stop a real run if the unbanned accuracy is lower")
parser.add_argument("--mmlu-restore", type=int, default=500)
parser.add_argument("--max-units", type=int, default=0)
parser.add_argument("--effort-questions", type=int, default=200, help="questions captured at every effort")
parser.add_argument("--effort-gen-questions", type=int, default=100, help="questions answered for the dial and probe")
parser.add_argument("--max-model-len", type=int, default=8192)
parser.add_argument("--moe-backend", default="triton", help="triton for FP8; marlin (W4A16) or auto (W4A4) for NVFP4")
parser.add_argument("--kv-cache-dtype", default="auto")
parser.add_argument("--tiny", action="store_true", help="tiny random checkpoint: token-id prompts, random units")
args = parser.parse_args()
T0 = time.time()
OUT = pathlib.Path(args.out).resolve()
OUT.mkdir(parents=True, exist_ok=True)
RUN = time.strftime("%Y%m%d-%H%M%S")


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


def over_budget():
    return (time.time() - T0) / 60 > args.budget_min


def upload(path):
    if args.upload:
        from huggingface_hub import HfApi

        try:
            HfApi().upload_file(path_or_fileobj=str(path), path_in_repo=f"{RUN}/{path.name}", repo_id=args.upload,
                                repo_type="dataset")
        except Exception as e:  # a failed upload must not end the session; files stay on disk for the next try
            log(f"upload of {path.name} failed: {e!r}")


def save(name, rows):
    path = OUT / name
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    upload(path)


def save_json(name, obj):
    path = OUT / name
    path.write_text(json.dumps(obj))
    upload(path)


def distinct4(ids):
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return len(set(grams)) / len(grams) if grams else 0.0


def last_number(text):
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text or "")
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def same_number(a, b):
    try:
        return a is not None and abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


from vllm import LLM, SamplingParams  # noqa: E402

from beam_steer import hist_quantiles  # noqa: E402


def main():
    phases = set(args.phases.split(","))
    assert phases <= set(PHASES), f"unknown phases {phases - set(PHASES)}"
    log("loading", args.model)
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=args.max_model_len,
              gpu_memory_utilization=0.92, async_scheduling=False, moe_backend=args.moe_backend, max_num_seqs=512,
              max_num_batched_tokens=4096, enable_prefix_caching=False, kv_cache_dtype=args.kv_cache_dtype, seed=0,
              skip_tokenizer_init=args.tiny,
              # the letters' logprobs after the allowed-token mask, i.e. normalised over the options (mmlu only)
              **({"logprobs_mode": "processed_logprobs"} if "mmlu" in phases else {}))
    log("loaded")
    config = llm.model_config.hf_config
    H, NL = config.hidden_size, config.num_hidden_layers
    tok = None if args.tiny else llm.get_tokenizer()
    END = None if args.tiny else tok.convert_tokens_to_ids("<|end|>")
    THINK = [1, 2, 3] if args.tiny else tok.encode("<|channel|>analysis<|message|>", add_special_tokens=False)
    rng = np.random.default_rng(1)

    def chat(text, effort):
        if args.tiny:  # deterministic token ids standing in for a prompt; position 5 stands in for the effort value
            ids = np.random.default_rng(zlib.crc32(text.encode())).integers(0, config.vocab_size, 24).tolist()
            ids[5] = effort % config.vocab_size
            return ids
        return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                       reasoning_effort=effort, tokenize=True, return_dict=False)

    def run(reqs):
        """reqs: dicts with prompt, slot, max_tokens, meta and optional extra (steering/capture extra_args) and
        prompt_logprobs. Returns vLLM outputs and tokens/s."""
        t = time.time()
        params = [SamplingParams(temperature=0, max_tokens=min(r["max_tokens"], args.max_model_len - len(r["prompt"])),
                                 detokenize=False,
                                 prompt_logprobs=r.get("prompt_logprobs"),
                                 extra_args={"steer_slot": r["slot"], "steer_from": 1, **r.get("extra", {})})
                  for r in reqs]
        outs = llm.generate([{"prompt_token_ids": r["prompt"]} for r in reqs], params, use_tqdm=False)
        return outs, sum(len(o.outputs[0].token_ids) for o in outs) / (time.time() - t)

    def record(o, meta, opened=False):
        """One generation as a row. opened: the prompt already opened the analysis channel."""
        ids = list(o.outputs[0].token_ids)
        row = {**meta, "n": len(ids), "finish": o.outputs[0].finish_reason, "distinct4": round(distinct4(ids), 3),
               "ids": ids}
        if tok is not None:
            row["text"] = tok.decode(ids, skip_special_tokens=False)
            row["analysis_len"] = ids.index(END) if END in ids else None  # thinking length: first <|end|>
            row["final"] = "<|channel|>final" in row["text"]
            if opened:
                row["answer"] = last_number(row["text"].split("<|channel|>final")[-1] if row["final"] else None)
        return row

    def generate(reqs, opened=False):
        outs, tps = run(reqs)
        return [record(o, r["meta"], opened) for o, r in zip(outs, reqs)], tps

    def set_slots(entries):
        path = OUT / "steer_wave.pt"
        torch.save(entries, path)
        return llm.collective_rpc("beam_steer_set", args=(str(path),))[0]

    def set_route(entries):
        path = OUT / "route_wave.pt"
        torch.save(entries, path)
        return llm.collective_rpc("beam_steer_set_route", args=(str(path),))[0]

    def entries_for(slot_pushes):
        """slot_pushes: [(slot, layer, unit_dir, dose)] -> beam_steer_set entries with delta = dose * median_L * dir."""
        return [(s, L, torch.from_numpy(np.asarray(d, np.float32) * dose * median[L])) for s, L, d, dose in slot_pushes]

    def nll_reqs(slot, meta):
        """Teacher-forced requests on the calib texts; their prompt logprobs give the push's ΔNLL."""
        return [{"prompt": ids, "slot": slot, "max_tokens": 1, "prompt_logprobs": 0, "meta": {**meta, "text": k}}
                for k, (ids, _) in texts.items()]

    def nll(o, k):
        ids, start = texts[k]
        return [o.prompt_logprobs[t][ids[t]].logprob for t in range(start, len(ids))]

    def dnll(outs, reqs):
        """Mean ΔNLL (nats/token, steered minus unsteered) over the calib texts of each request group, by meta id."""
        per = {}
        for o, r in zip(outs, reqs):
            k = r["meta"]["text"]
            per.setdefault(r["meta"]["id"], []).append(float(np.mean(np.array(base_lp[k]) - np.array(nll(o, k)))))
        return {i: float(np.mean(v)) for i, v in per.items()}

    # --- sanity -------------------------------------------------------------------------------------------------------
    set_slots([])
    if "sanity" in phases:
        rows, _ = generate([{"prompt": chat(q, 8), "slot": 0, "max_tokens": 256, "meta": {"prompt": q}} for q in SANITY])
        save("sanity.jsonl.gz", rows)
        for r in rows:
            log("sanity:", repr(r.get("text", r["ids"][:20]))[:400])
        if not args.tiny and not any("paris" in r["text"].lower() for r in rows[:1]):
            log("ABORT: the real model did not answer Paris; not spending on a broken engine")
            sys.exit(2)

    # --- mmlu ---------------------------------------------------------------------------------------------------------
    if "mmlu" in phases:
        run_mmlu(llm, config, tok, chat)
        if phases <= {"sanity", "mmlu"}:
            log("SESSION DONE")
            return

    # --- stats (always: every dose is written in units of the median) ------------------------------------------------
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
    llm.generate([{"prompt_token_ids": c} for c in corpus], SamplingParams(max_tokens=1, detokenize=False),
                 use_tqdm=False)
    quant = hist_quantiles(llm.collective_rpc("beam_steer_stats")[0])
    median = [q[1] for q in quant]
    save_json("stats.json", {"q10_50_90": quant, "tokens": sum(map(len, corpus))})
    log("median ‖residual‖ by layer:", [round(m, 1) for m in median])

    # Calib texts: Beam's own unsteered continuations, scored teacher-forced for ΔNLL (calib and sweep).
    seeds = {k: chat(v, 8) for k, v in PROMPTS.items()} | {"effort": chat(EFFORT_PROMPT, 32)}
    outs, _ = run([{"prompt": p, "slot": 0, "max_tokens": 160, "meta": {}} for p in seeds.values()])
    texts = {k: (p + list(o.outputs[0].token_ids), len(p)) for (k, p), o in zip(seeds.items(), outs)}
    reqs = nll_reqs(0, {"id": 0})
    base_lp = {r["meta"]["text"]: nll(o, r["meta"]["text"]) for o, r in zip(run(reqs)[0], reqs)}

    # --- calib --------------------------------------------------------------------------------------------------------
    who = chat(PROMPTS["who"], 8)
    base_rows, _ = generate([{"prompt": who, "slot": 0, "max_tokens": 96, "meta": {"phase": "calib", "dose": 0.0}}])
    base_d4 = base_rows[0]["distinct4"]
    calib_dirs = rng.standard_normal((NL, CALIB_DIRS, H))
    calib_dirs /= np.linalg.norm(calib_dirs, axis=-1, keepdims=True)
    curve = {L: (np.array(CALIB_DOSES), None) for L in range(NL)}  # layer -> (doses, ΔNLL by dose)
    tau = None
    if "calib" in phases:
        jobs = [(L, i, dose) for L in range(NL) for i in range(CALIB_DIRS) for dose in CALIB_DOSES]
        calib_rows, effect = [], {}
        for start in range(0, len(jobs), SLOTS - 1):
            chunk = jobs[start:start + SLOTS - 1]
            set_slots(entries_for([(k + 1, L, calib_dirs[L, i], dose) for k, (L, i, dose) in enumerate(chunk)]))
            gen = [{"prompt": who, "slot": k + 1, "max_tokens": 96,
                    "meta": {"phase": "calib", "layer": L, "control": i, "dose": dose}}
                   for k, (L, i, dose) in enumerate(chunk)]
            rows, tps = generate(gen)
            calib_rows += rows
            reqs = [q for k, job in enumerate(chunk) for q in nll_reqs(k + 1, {"id": job})]
            effect |= dnll(run(reqs)[0], reqs)
            log(f"calib {start + len(chunk)}/{len(jobs)} at {tps:.0f} tok/s")
        for r in calib_rows:
            r["dnll"] = round(effect[(r["layer"], r["control"], r["dose"])], 4)
        save("calib.jsonl.gz", base_rows + calib_rows)
        ok_tau = {}
        for L in range(NL):
            d = np.array([np.median([effect[(L, i, dose)] for i in range(CALIB_DIRS)]) for dose in CALIB_DOSES])
            # Monotone and strictly increasing, so it can be inverted by interpolation.
            d = np.maximum.accumulate(np.maximum(d, 1e-6)) * (1 + 1e-6 * np.arange(len(d)))
            curve[L] = (np.array(CALIB_DOSES), d)
            ok = []
            for j, dose in enumerate(CALIB_DOSES):
                rs = [r for r in calib_rows if r["layer"] == L and r["dose"] == dose]
                if np.median([r["distinct4"] for r in rs]) >= 0.9 * base_d4 and np.mean([r["n"] < 4 for r in rs]) < 0.2:
                    ok.append(j)
            ok_tau[L] = float(d[max(ok)]) if ok else float(d[0])
        tau = float(np.median(list(ok_tau.values())))
        save_json("calib.json", {"base_distinct4": base_d4, "tau_star": tau, "ok_tau_by_layer": ok_tau,
                                 "dnll_by_layer": {L: curve[L][1].tolist() for L in range(NL)},
                                 "doses": list(CALIB_DOSES)})
        log(f"τ* = {tau:.3f} nats/token; coherent ΔNLL by layer:", {L: round(v, 2) for L, v in ok_tau.items()})

    def dose_at(L, target):
        """Dose (x median) whose median ΔNLL at layer L is target, interpolated on log scales; clamped to the grid."""
        doses, d = curve[L]
        if d is None:
            return float(doses[2])
        return float(np.exp(np.interp(np.log(target), np.log(d), np.log(doses))))

    # --- effort -------------------------------------------------------------------------------------------------------
    if "effort" in phases and not over_budget():
        try:
            effort_phase(llm, chat, run, generate, set_slots, median, THINK, END, config)
        except Exception:  # the sweep does not depend on it; an engine-level failure will surface there anyway
            import traceback

            log("effort phase failed:\n" + traceback.format_exc())
            set_slots([])

    # --- effort2 ------------------------------------------------------------------------------------------------------
    if "effort2" in phases and not over_budget():
        effort2_phase(llm, chat, generate, set_slots, set_route, median, THINK, config)
        set_slots([]), set_route([])

    # --- sweep --------------------------------------------------------------------------------------------------------
    if "sweep" not in phases:
        log("SESSION DONE")
        return
    if tau is None:
        log("sweep needs calib in the same session; stopping")
        return
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
    base, _ = generate([{"prompt": p, "slot": 0, "max_tokens": 160, "meta": {"phase": "baseline", "prompt": k}}
                        for k, p in prompts.items()]
                       + [{"prompt": effort, "slot": 0, "max_tokens": 512, "meta": {"phase": "baseline", "prompt": "effort"}}])
    save("baseline.jsonl.gz", base)
    sweep_texts = ("who", "snake")
    per_wave = (SLOTS - 1) // len(SWEEP_TAUS)
    for w, start in enumerate(range(0, len(metas), per_wave)):
        if over_budget():
            log(f"budget reached before wave {w}; stopping")
            break
        units = list(range(start, min(start + per_wave, len(metas))))
        pushes, reqs, nreqs = [], [], []
        for k, u in enumerate(units):
            L = metas[u]["layer"]
            for j, frac in enumerate(SWEEP_TAUS):
                slot = 1 + k * len(SWEEP_TAUS) + j
                dose = np.sign(frac) * dose_at(L, abs(frac) * tau)
                pushes.append((slot, L, dirs[u], dose))
                meta = {"phase": "sweep", "unit": u, "dose": frac, "dose_eff": round(float(dose), 4)}
                reqs += [{"prompt": p, "slot": slot, "max_tokens": 160, "meta": {**meta, "prompt": name}}
                         for name, p in prompts.items()]
                if frac in EFFORT_TAUS:
                    reqs.append({"prompt": effort, "slot": slot, "max_tokens": 512, "meta": {**meta, "prompt": "effort"}})
                nreqs += [q for q in nll_reqs(slot, {"id": (u, frac)}) if q["meta"]["text"] in sweep_texts]
        set_slots(entries_for(pushes))
        rows, tps = generate(reqs)
        effect = dnll(run(nreqs)[0], nreqs)
        for r in rows:
            r["dnll"] = round(effect[(r["unit"], r["dose"])], 4)
        save(f"sweep_{w:03d}.jsonl.gz", rows)
        log(f"wave {w}: units {units[0]}..{units[-1]} of {len(metas)}, {len(rows)} rollouts at {tps:.0f} tok/s")
    log("SESSION DONE")


def effort_phase(llm, chat, run, generate, set_slots, median, THINK, END, config):
    H, NL = config.hidden_size, config.num_hidden_layers
    if args.tiny:
        questions = [(f"tiny question {i}", "0") for i in range(max(args.effort_questions, args.effort_gen_questions))]
    else:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        table = pq.read_table(hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset"))
        questions = [(q + "\nGive the final answer as a single number.", a.split("####")[-1].strip().replace(",", ""))
                     for q, a in zip(table.column("question").to_pylist(), table.column("answer").to_pylist())]

    def prompt(i, n):
        return chat(questions[i][0], n) + THINK

    def positions(i, n):
        """Captured positions: last digit of N, end of system message, think start, first two thought tokens."""
        ids, ref = prompt(i, n), prompt(i, EFFORT_REF)
        suffix = next(s for s in range(min(len(ids), len(ref))) if ids[-1 - s] != ref[-1 - s])
        n_last = len(ids) - suffix - 1
        sys_end = 10 if args.tiny else ids.index(END)
        return [n_last, sys_end, len(ids) - 1, len(ids), len(ids) + 1]

    KINDS = ("n_last", "sys_end", "think_start", "thought1", "thought2")
    efforts = (*DIAL, *OFF_DIAL)

    # Capture, in chunks that fit the capture buffer.
    if CAPTURE_ROWS <= 0:
        log("effort: BEAM_STEER_CAPTURE_ROWS is 0, skipping capture")
    else:
        set_slots([])
        per_chunk = max(1, CAPTURE_ROWS // (len(efforts) * len(KINDS)))
        for c, start in enumerate(range(0, args.effort_questions, per_chunk)):
            qs = range(start, min(start + per_chunk, args.effort_questions))
            reqs = [{"prompt": prompt(i, n), "slot": 0, "max_tokens": 3, "meta": {},
                     "extra": {"capture": positions(i, n), "capture_key": [i, n]}} for i in qs for n in efforts]
            run(reqs)
            path = OUT / f"effort_capture_{c:02d}.pt"
            rows, dropped = llm.collective_rpc("beam_steer_capture_dump", args=(str(path),))[0]
            log(f"effort capture chunk {c}: {rows} rows, {dropped} dropped")
            upload(path)

    # The effort direction per layer, from the think-start residuals on the dial (within-question centred).
    directions = effort_directions(KINDS.index("think_start"), positions)
    if directions is not None:
        torch.save({"r": torch.from_numpy(directions["r"]), "r2": directions["r2"]}, OUT / "effort_direction.pt")
        upload(OUT / "effort_direction.pt")
        log("effort direction: held-out R² of log2 N by layer:", [round(x, 2) for x in directions["r2"]])

    # The dial: greedy answers at every effort.
    gq = range(args.effort_gen_questions)
    max_tokens = args.max_model_len - max(len(prompt(i, 1000)) for i in gq) - 8
    rows, tps = generate([{"prompt": prompt(i, n), "slot": 0, "max_tokens": max_tokens,
                           "meta": {"phase": "dial", "q": i, "effort": n, "gold": questions[i][1]}}
                          for i in gq for n in efforts], opened=True)
    for r in rows:
        r["correct"] = same_number(r.get("answer"), r["gold"])
    save("effort_dial.jsonl.gz", rows)
    log(f"dial at {tps:.0f} tok/s:", {n: (round(float(np.mean([r["n"] for r in rows if r["effort"] == n]))),
                                         round(float(np.mean([r["correct"] for r in rows if r["effort"] == n])), 2))
                                     for n in efforts})
    if directions is None or over_budget():
        return

    # Steering probe: N = 32 plus λ r_L at a middle and a late layer; random directions of the same norm as controls.
    layers = (NL // 2, round(0.7 * NL))
    rng = np.random.default_rng(7)
    arms = []
    for L in layers:
        r = directions["r"][L]
        for lam in LAMBDAS:
            arms.append({"layer": L, "kind": "effort", "lambda": lam, "delta": lam * r})
        for lam in (min(LAMBDAS), max(LAMBDAS)):
            u = rng.standard_normal(H)
            arms.append({"layer": L, "kind": "control", "lambda": lam,
                         "delta": u / np.linalg.norm(u) * abs(lam) * np.linalg.norm(r)})
    set_slots([(k + 1, a["layer"], torch.from_numpy(a["delta"].astype(np.float32))) for k, a in enumerate(arms)])
    meta = [{"phase": "effort_steer", "layer": a["layer"], "kind": a["kind"], "lambda": a["lambda"],
             "dose_eff": round(float(np.linalg.norm(a["delta"]) / median[a["layer"]]), 4)} for a in arms]
    rows, tps = generate([{"prompt": prompt(i, STEER_N), "slot": k + 1, "max_tokens": max_tokens,
                           "meta": {**meta[k], "q": i, "effort": STEER_N, "gold": questions[i][1]}}
                          for k in range(len(arms)) for i in gq], opened=True)
    for r in rows:
        r["correct"] = same_number(r.get("answer"), r["gold"])
    save("effort_steer.jsonl.gz", rows)
    for m in meta:
        rs = [r for r in rows if all(r[k] == m[k] for k in ("layer", "kind", "lambda"))]
        log(f"effort steer L{m['layer']} {m['kind']} λ {m['lambda']:+} (dose {m['dose_eff']}): "
            f"tokens {np.mean([r['n'] for r in rs]):.0f}, correct {np.mean([r['correct'] for r in rs]):.2f}")
    set_slots([])


def effort2_phase(llm, chat, generate, set_slots, set_route, median, THINK, config):
    H, NL = config.hidden_size, config.num_hidden_layers
    rng = np.random.default_rng(11)
    if args.tiny:
        problems = [(f"tiny problem {i}", "0", "tiny") for i in range(args.effort2_problems)]
        moe = [i for i, t in enumerate(config.mlp_layer_types) if t == "sparse"]
        route_up = [(L, int(e), 8.0 if j % 2 else -30.0) for j, (L, e) in
                    enumerate((L, e) for L in moe for e in rng.choice(config.num_experts, 2, replace=False))]
        inputs = {"r": torch.from_numpy(rng.standard_normal((NL, H)).astype(np.float32) * 0.02),
                  "route_up": route_up, "route_down": [(L, e, -30.0 if d > 0 else 8.0) for L, e, d in route_up],
                  "route_ctrl": [(L, (e + 1) % config.num_experts, d) for L, e, d in route_up],
                  "swap": {"layer": 2, "delta": torch.from_numpy(rng.standard_normal(H).astype(np.float32))}}
        band = range(1, NL - 1)
    else:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        problems = []
        for repo, path, col in E2_SOURCES:
            t = pq.read_table(hf_hub_download(repo, path, repo_type="dataset"))
            for q, a in zip(t.column(col).to_pylist(), t.column("answer").to_pylist()):
                a = float(a) if not isinstance(a, str) else float(a.strip())
                problems.append((q + "\nGive the final answer as a single number.", f"{a:g}", repo.split("/")[1]))
        problems = problems[: args.effort2_problems]
        inputs = torch.load(args.effort2_inputs, weights_only=True)
        band = E2_BAND
    r = inputs["r"].float()

    def prompt(i, n):
        return chat(problems[i][0], n) + THINK

    def digit_pos(i, n):  # position of N's last digit: where N's prompt stops matching a reference effort's prompt
        ids, ref = prompt(i, n), prompt(i, EFFORT_REF)
        return len(ids) - next(s for s in range(min(len(ids), len(ref))) if ids[-1 - s] != ref[-1 - s]) - 1

    P = range(len(problems))
    max_tokens = min(args.effort2_max_tokens, args.max_model_len - max(len(prompt(i, 512)) for i in P) - 8)

    # Transfer capture: does Job 1's GSM8K effort direction read effort on these prompts?
    set_slots([]), set_route([])
    if CAPTURE_ROWS > 0 and not args.effort2_arms:
        per_chunk = max(1, CAPTURE_ROWS // (len(DIAL) * 2))
        for c, start in enumerate(range(0, len(problems), per_chunk)):
            qs = range(start, min(start + per_chunk, len(problems)))
            generate([{"prompt": prompt(i, n), "slot": 0, "max_tokens": 1, "meta": {},
                       "extra": {"capture": [digit_pos(i, n), len(prompt(i, n)) - 1], "capture_key": [i, n]}}
                      for i in qs for n in DIAL])
            path = OUT / f"effort2_capture_{c:02d}.pt"
            rows, dropped = llm.collective_rpc("beam_steer_capture_dump", args=(str(path),))[0]
            log(f"effort2 transfer capture chunk {c}: {rows} rows, {dropped} dropped")
            upload(path)

    def unit_like(v):  # a random direction with v's norm
        u = rng.standard_normal(H).astype(np.float32)
        return torch.from_numpy(u / np.linalg.norm(u)) * float(v.norm())

    arms = [  # name, effort, residual pushes [(layer, delta)], router deltas, request window, pair (run order)
        ("dial8", 8, [], [], None, 1), ("dial512", 512, [], [], None, 1),
        ("route_up", 8, [], inputs["route_up"], None, 2), ("route_ctrl", 8, [], inputs["route_ctrl"], None, 2),
        ("dir_band", 8, [(L, E2_OCTAVES * r[L]) for L in band], [], None, 3),
        ("dir_band_ctrl", 8, [(L, unit_like(E2_OCTAVES * r[L])) for L in band], [], None, 3),
        ("swap", 8, [(inputs["swap"]["layer"], inputs["swap"]["delta"].float())], [], "digit", 4),
        ("route_up_prompt", 8, [], inputs["route_up"], "prompt", 4),
        ("route_down", 512, [], inputs["route_down"], None, 5),
        ("dir_spread", 8, [(L, E2_OCTAVES / len(band) * r[L]) for L in band], [], None, 5),
        ("dial32", 32, [], [], None, 6), ("dial128", 128, [], [], None, 6),
        ("route_ctrl_prompt", 8, [], inputs["route_ctrl"], "prompt", 7),
        ("swap_ctrl", 8, [(inputs["swap"]["layer"], unit_like(inputs["swap"]["delta"].float()))], [], "digit", 7)]
    if args.effort2_arms:
        names = args.effort2_arms.split(",")
        assert set(names) <= {a[0] for a in arms}, f"unknown arms {set(names) - {a[0] for a in arms}}"
        arms = [(*a[:5], names.index(a[0]) // 2) for a in arms if a[0] in names]
    for batch in sorted({a[5] for a in arms}):
        if over_budget():
            log(f"effort2: budget reached before pair {batch}")
            break
        chosen = [a for a in arms if a[5] == batch]
        resid, route, reqs = [], [], []
        for k, (name, n, pushes, deltas, window, _) in enumerate(chosen):
            slot = k + 1
            resid += [(slot, L, d) for L, d in pushes]
            route += [(slot, L, e, d) for L, e, d in deltas]
            dose = max((float(d.norm()) / median[L] for L, d in pushes), default=0.0)
            for i in P:
                ids = prompt(i, n)
                extra = ({"steer_until": len(ids)} if window == "prompt" else
                         {"steer_from": digit_pos(i, n), "steer_until": digit_pos(i, n) + 1} if window == "digit" else {})
                reqs.append({"prompt": ids, "slot": slot if (pushes or deltas) else 0, "max_tokens": max_tokens,
                             "extra": extra, "meta": {"phase": "effort2", "arm": name, "q": i, "source": problems[i][2],
                                                      "effort": n, "gold": problems[i][1], "dose_eff": round(dose, 4)}})
        set_slots(resid), set_route(route)
        rows, tps = generate(reqs, opened=True)
        for row in rows:
            row["correct"] = same_number(row.get("answer"), row["gold"])
        save(f"effort2_pair{batch}.jsonl.gz", rows)
        for name, *_ in chosen:
            rs = [x for x in rows if x["arm"] == name]
            think = [x["n"] if x.get("analysis_len") is None else x["analysis_len"] for x in rs]
            log(f"effort2 {name:16s} thinking median {np.median(think):6.0f} mean {np.mean(think):6.0f}  "
                f"correct {np.mean([x['correct'] for x in rs]):.2f}  hit max_tokens {np.mean([x['finish'] == 'length' for x in rs]):.2f}")
        log(f"effort2 pair {batch} at {tps:.0f} tok/s")


def effort_directions(kind_index, positions):
    """Per-layer slope of the residual on log2 N (on the dial), within-question centred, from the capture files.
    Held-out R²: questions split 80/20; the test residuals' projection on r_L, mapped to log2 N by a 1-d fit on train."""
    files = sorted(OUT.glob("effort_capture_*.pt"))
    if not files:
        return None
    xs, keys = [], []
    for f in files:
        cap = torch.load(f, weights_only=True)
        rows = [(row, tuple(key)) for row, (key, pos, _, _) in enumerate(cap["index"])
                if key[1] in DIAL and pos == positions(*key)[kind_index]]
        if rows:
            xs.append(cap["resid"][:, [r for r, _ in rows]].float().numpy())
            keys += [k for _, k in rows]
    if not keys:
        return None
    X = np.concatenate(xs, 1)  # [layers, samples, hidden]
    q = np.array([i for i, _ in keys])
    y = np.log2([n for _, n in keys])
    for i in np.unique(q):  # centre within question: the dial is the only thing that varies
        m = q == i
        X[:, m] -= X[:, m].mean(1, keepdims=True)
        y[m] -= y[m].mean()
    qs = np.unique(q)
    train = np.isin(q, qs[: int(0.8 * len(qs))])
    r = np.einsum("lsh,s->lh", X[:, train], y[train]) / (y[train] ** 2).sum()  # residual change per octave of N
    r2 = []
    for L in range(X.shape[0]):
        norm = np.linalg.norm(r[L])
        if not (~train).any() or not np.isfinite(norm) or norm == 0:  # e.g. N out of every window at this layer
            r2.append(float("nan"))
            continue
        p_train, p_test = X[L, train] @ (r[L] / norm), X[L, ~train] @ (r[L] / norm)
        var = float(((p_train - p_train.mean()) ** 2).sum())
        slope = float(((p_train - p_train.mean()) * (y[train] - y[train].mean())).sum()) / var if var > 0 else 0.0
        pred = y[train].mean() + slope * (p_test - p_train.mean())
        r2.append(float(1 - ((y[~train] - pred) ** 2).sum() / max(((y[~train] - y[~train].mean()) ** 2).sum(), 1e-12)))
    return {"r": r, "r2": r2}



def mmlu_questions(config, tok, chat):
    """[{id, category, answer, n, prompt}] and the option letters' token ids."""
    if args.tiny:  # random token prompts standing in for questions
        g, cats = np.random.default_rng(7), ("math", "health", "law", "biology", "history", "other")
        qs = [{"id": i, "category": cats[i % len(cats)], "answer": int(g.integers(0, 10)), "n": 10,
               "prompt": g.integers(0, config.vocab_size, int(g.integers(60, 200))).tolist()}
              for i in range(args.mmlu_limit or 120)]
        return qs, list(range(32, 42))
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    letters = [tok.convert_tokens_to_ids(c) for c in "ABCDEFGHIJ"]
    assert len(set(letters)) == 10 and all(tok.convert_ids_to_tokens(i) == c for i, c in zip(letters, "ABCDEFGHIJ"))
    prefix = tok.encode(MMLU_PREFIX, add_special_tokens=False)
    rows = pq.read_table(hf_hub_download("TIGER-Lab/MMLU-Pro", "data/test-00000-of-00001.parquet",
                                         repo_type="dataset")).to_pylist()
    qs = []
    for r in rows[: args.mmlu_limit or None]:
        text = (r["question"] + "\n" + "\n".join(f"({'ABCDEFGHIJ'[i]}) {o}" for i, o in enumerate(r["options"]))
                + "\nAnswer with the letter only.")
        qs.append({"id": r["question_id"], "category": r["category"], "answer": r["answer_index"],
                   "n": len(r["options"]), "prompt": chat(text, 8) + prefix})
    return qs, letters


def run_mmlu(llm, config, tok, chat):
    qs, letters = mmlu_questions(config, tok, chat)
    moe = [i for i, t in enumerate(config.mlp_layer_types) if t == "sparse"]
    if args.mmlu_arms:
        arms = json.loads(pathlib.Path(args.mmlu_arms).read_text())["arms"]
    else:  # tiny: unbanned, half of every group at every MoE layer, one expert per layer, and its random control
        assert args.tiny, "a real run needs --mmlu-arms"
        g, per = np.random.default_rng(3), config.num_experts // config.n_group
        half = [[L, e] for L in moe for k in range(config.n_group) for e in range(k * per, k * per + per // 2)]
        arms = [{"name": "base", "bans": []}, {"name": "half", "bans": half},
                {"name": "one", "bans": [[L, 0] for L in moe]},
                {"name": "one_ctrl", "bans": [[L, int(g.integers(1, config.num_experts))] for L in moe]}]
    assert arms[0]["bans"] == [], "the first arm must be unbanned (gate and restore check)"
    # Every routing group must keep enough experts for the grouped top-k to pick top_k unbanned ones.
    need = -(-config.num_experts_per_tok // config.topk_group)
    per = config.num_experts // config.n_group
    for arm in arms:
        banned = {}
        for L, e in arm["bans"]:
            assert L in moe and 0 <= e < config.num_experts, (arm["name"], L, e)
            banned.setdefault((L, e // per), set()).add(e)
        worst = min((per - len(v) for v in banned.values()), default=per)
        assert worst >= need, f"arm {arm['name']}: a group keeps {worst} experts, top-k needs {need}"
    log(f"mmlu: {len(qs)} questions, {sum(len(q['prompt']) for q in qs)} prompt tokens per arm, arms",
        [(a["name"], len(a["bans"])) for a in arms])

    def score(subset):
        params = [SamplingParams(temperature=0, max_tokens=1, detokenize=False, logprobs=10,
                                 allowed_token_ids=letters[: q["n"]], extra_args={"steer_slot": 0}) for q in subset]
        t = time.time()
        outs = llm.generate([{"prompt_token_ids": q["prompt"]} for q in subset], params, use_tqdm=False)
        tps = sum(len(q["prompt"]) for q in subset) / (time.time() - t)
        rows = []
        for q, o in zip(subset, outs):
            lp = o.outputs[0].logprobs[0]
            lps = [lp[t].logprob if t in lp else float("-inf") for t in letters[: q["n"]]]
            rows.append({"id": q["id"], "category": q["category"], "answer": q["answer"], "pred": int(np.argmax(lps)),
                         "sampled": letters.index(o.outputs[0].token_ids[0]), "lp": [round(x, 4) for x in lps],
                         "mass": round(float(np.exp(lps).sum()), 4)})
        return rows, tps

    def summary(rows):
        acc = float(np.mean([r["pred"] == r["answer"] for r in rows]))
        by = {}
        for r in rows:
            by.setdefault(r["category"], []).append(r)
        return {"acc": round(acc, 4), "lp_correct": round(float(np.mean([r["lp"][r["answer"]] for r in rows])), 4),
                "mass_min": min(r["mass"] for r in rows), "sampled_is_pred": float(np.mean([r["sampled"] == r["pred"]
                                                                                             for r in rows])),
                "by_category": {c: {"n": len(v), "acc": round(float(np.mean([r["pred"] == r["answer"] for r in v])), 4),
                                    "lp_correct": round(float(np.mean([r["lp"][r["answer"]] for r in v])), 4)}
                                for c, v in sorted(by.items())}}

    results, base_rows, last_rows = {}, None, None
    for arm in arms:
        if over_budget():
            log(f"mmlu: budget reached, arm {arm['name']} and later not run")
            break
        bans = [(L, e, MMLU_BAN) for L, e in arm["bans"]]
        if bans:
            path = OUT / "mmlu_bias.pt"
            torch.save(bans, path)
            set_counts = llm.collective_rpc("beam_steer_set_bias", args=(str(path),))
            assert all(c == len(bans) for c in set_counts), set_counts
        try:
            rows, tps = score(qs)
        finally:
            if bans:
                llm.collective_rpc("beam_steer_reset_bias")
        save(f"mmlu_{arm['name']}.jsonl.gz", rows)
        last_rows = rows if bans else last_rows
        results[arm["name"]] = {"bans": len(bans), "tok_per_s": round(tps), **summary(rows)}
        save_json("mmlu_summary.json", results)
        r = results[arm["name"]]
        log(f"mmlu {arm['name']:14s} bans {len(bans):5d}  acc {r['acc']:.3f}  lp(correct) {r['lp_correct']:.3f}  "
            f"letter mass min {r['mass_min']:.4f}  sampled==argmax {r['sampled_is_pred']:.3f}  at {tps:.0f} tok/s")
        if base_rows is None:
            base_rows = rows
            if not args.tiny and r["acc"] < args.mmlu_gate:
                log(f"ABORT: unbanned accuracy {r['acc']:.3f} < gate {args.mmlu_gate}; not spending on the arms")
                sys.exit(3)
    n = min(args.mmlu_restore, len(qs))
    again, _ = score(qs[:n])
    # A smaller batch than the first run, so it differs by engine noise (on Beam with TP 4: mean 0.12 nats, 7% of
    # predictions); a bias left banned would make it look like the last banned arm instead, so that is the test.
    def mean_diff(xs, ys):
        return float(np.mean([abs(a - b) for x, y in zip(xs, ys) for a, b in zip(x["lp"], y["lp"])
                              if np.isfinite(a) and np.isfinite(b)]))
    to_base = mean_diff(again, base_rows[:n])
    to_last = mean_diff(again, last_rows[:n]) if last_rows else float("nan")
    same = float(np.mean([x["pred"] == y["pred"] for x, y in zip(again, base_rows[:n])]))
    acc = float(np.mean([r["pred"] == r["answer"] for r in again]))
    save_json("mmlu_restore.json", {"n": n, "mean_abs_lp_diff_to_base": to_base, "mean_abs_lp_diff_to_last_ban": to_last,
                                    "pred_agreement": same, "acc": acc,
                                    "acc_base_same_questions": float(np.mean([r["pred"] == r["answer"]
                                                                              for r in base_rows[:n]]))})
    save(f"mmlu_restore.jsonl.gz", again)
    log(f"mmlu restore check: unbanned again on {n} questions, mean |Δ letter logprob| to base {to_base:.3f}, "
        f"to the last banned arm {to_last:.3f}, same prediction as base {same:.3f}  "
        f"{'ok' if not to_base > 0.5 * to_last else 'BIAS LEAK?'}")


if __name__ == "__main__":  # vLLM spawns engine processes that re-import this module
    main()
