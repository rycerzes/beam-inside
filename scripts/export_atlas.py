# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pyarrow"]
# ///
"""Build the public dataset (Parquet tables, explorer JSON) from the local run files.

    uv run scripts/export_atlas.py OUT_DIR [--runs data/runs] [--no-explorer]

Inputs, all under data/ (gitignored; the private run repo and this session's scratch files hold the originals):
  runs/20261027-145310   main session: calibration, effort dial/steer/analysis, the 12,920-unit sweep and its
                         effect-matched ranking (rank_sweep.py, effect_rank.py)
  runs/20261028-044229, runs/20261028-075311   competition-math effort arms and their gentle-arm controls
  judge/                 blind three-way test: key, DeepSeek verdicts with token labels
  selflabel/             Beam's pole labels and DeepSeek's verdicts with those labels
  mmlu/                  ban arms and the MMLU-Pro run

Writes OUT_DIR as a dataset repo: one Parquet file per table (configs in README.md), summary.json with the headline
numbers, and explorer/ (per-layer JSON the Space loads). Steered text is kept whole in Parquet; token ids are dropped
(the text is their decoding). Benchmark problem text is never copied: rows point at the source dataset and row.
Activation captures are converted separately (export_captures.py).
"""

import argparse
import collections
import gzip
import json
import pathlib
import re

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

parser = argparse.ArgumentParser()
parser.add_argument("out", type=pathlib.Path)
parser.add_argument("--runs", type=pathlib.Path, default=pathlib.Path("data/runs"))
parser.add_argument("--no-explorer", action="store_true")
args = parser.parse_args()
MAIN, E2, E2C = (args.runs / r for r in ("20261027-145310", "20261028-044229", "20261028-075311"))
DATA = pathlib.Path("data")
# Judge-verified examples for the explorer's front page (picked from the 671 significant, coherent poles the judge
# matched with confidence 3, for spread across subjects and kinds of unit).
GALLERY = ["L26_E174_+", "L28_E155_+", "L26_E66_+", "L26_E122_+", "L31_E193_+", "L26_E102_+", "L43_E94_+", "L40_E209_+"]
PROMPTS = {"who": "Who are you?", "snake": "Tell me something about snakes.",  # session.py PROMPTS, effort 8
           "orange": "Describe an orange in two sentences.",
           "effort": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
                     "How much does the ball cost?"}  # EFFORT_PROMPT, effort 32, 512 tokens

src = (pathlib.Path(__file__).parent / "judge_items.py").read_text()  # the judge's own token decoding
helpers = {}
exec(src[src.index("def _byte_decoder"):src.index("def stems")], helpers)
readable = helpers["readable"]


def jsonl(path):
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt") as f:
        return [json.loads(line) for line in f]


def write(name, rows):
    path = args.out / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    print(f"{name}: {len(rows)} rows, {path.stat().st_size / 1e6:.1f} MB")


def tokens(lens):
    return [readable(t) for t in lens or []]


SPECIAL = re.compile(r"<\|[a-z_]+\|>")
TERMINATORS = ("<|end|>", "<|return|>", "<|call|>")
LOOP_DISTINCT4 = 0.5  # unsteered answers sit at 0.72-1.0; a repeating loop drives the distinct-4-gram share far lower


def harmony(text, finish, distinct4):
    """Parse Beam's harmony output into (analysis, final, status, broken).

    Messages are `<|start|>assistant<|channel|>NAME<|message|>BODY<|end|>`; the first one has no `<|start|>` (the prompt
    ends with it) and, when the prompt already opened the analysis channel, no header either. `final` is the body of the
    last message on the final channel; every analysis body is thinking. A message on any other channel name ("final?xml",
    "inal", a tool call), a special token inside a body, or a tool-call terminator is a format break. Status, first match:
      format    the push broke the reply format (see above)
      loop      ran to the token budget repeating itself (distinct-4-gram share below LOOP_DISTINCT4)
      budget    hit the token budget while answering (the answer was still going)
      thinking  hit the token budget before answering
      finished  stopped with an answer
      no-answer stopped without one
    """
    analysis, final, other, broken = [], None, [], False
    for i, seg in enumerate(text.split("<|start|>")):
        if i:  # tokenizer spacing variants (" assistant", " final") are the same message
            seg = seg.lstrip().removeprefix("assistant").lstrip()
        if seg.startswith("<|channel|>") and "<|message|>" in seg:
            header, body = seg[len("<|channel|>"):].split("<|message|>", 1)
            channel = SPECIAL.sub(" ", header).strip()
            if channel not in ("analysis", "final"):  # "final?xml", "inal", "commentary to=...", "final <|constrain|>"
                channel, broken = "", True
        elif i == 0:  # the prompt already opened the analysis channel
            channel, body = "analysis", seg
        else:
            channel, body, broken = "", seg, True
        if "<|call|>" in body:
            broken = True
        for t in TERMINATORS:
            body = body.split(t, 1)[0]
        if SPECIAL.search(body):
            broken = True
        body = SPECIAL.sub(lambda m: "⟨" + m.group(0)[2:-2] + "⟩", body).strip()
        if channel == "analysis":
            analysis.append(body)
        elif channel == "final":
            final = body
        elif body:
            other.append(body)
    if final is None and broken and other:
        final = other[-1]  # show what Beam wrote instead of an answer, with its special tokens marked
    final = final or ""
    if broken:
        status = "format"
    elif finish == "length" and distinct4 is not None and distinct4 < LOOP_DISTINCT4:
        status = "loop"
    elif finish == "length":
        status = "budget" if final else "thinking"
    else:
        status = "finished" if final else "no-answer"
    return "\n\n".join(a for a in analysis if a), final, status, broken


config = json.loads(pathlib.Path("meta/config.json").read_text())
MOE = [i for i, t in enumerate(config["mlp_layer_types"]) if t == "sparse"]

# ---- units and their steered outputs ----------------------------------------------------------------------------
units = {u["unit"]: u for u in jsonl(DATA / "sweep/units.jsonl")}
write("steering/units", [{"unit": u["unit"], "kind": u["kind"], "layer": u["layer"], "expert": u.get("expert"),
                          "dir": u.get("dir"), "sign": u.get("sign"), "neuron": u.get("neuron"),
                          "control": u.get("control"), "lens_z": u.get("z"), "lens_tokens": tokens(u.get("lens"))}
                         for u in units.values()])
outputs = []
for path in sorted(MAIN.glob("sweep_*.jsonl.gz")):
    for r in jsonl(path):
        analysis, final, status, _ = harmony(r["text"], r["finish"], r["distinct4"])
        outputs.append({"unit": r["unit"], "prompt": r["prompt"], "dose": r["dose"], "dose_eff": r["dose_eff"],
                        "dnll": r.get("dnll"), "n_tokens": r["n"], "finish": r["finish"], "distinct4": r["distinct4"],
                        "analysis_len": r.get("analysis_len"), "status": status, "analysis": analysis, "final": final,
                        "raw": r["text"]})
outputs.sort(key=lambda r: (r["unit"], r["prompt"], r["dose"]))
write("steering/outputs", outputs)
baseline = {}
for r in jsonl(MAIN / "baseline.jsonl.gz"):
    a, f, st, _ = harmony(r["text"], r["finish"], r.get("distinct4"))
    baseline[r.get("prompt", "effort")] = {"analysis": a, "final": f, "status": st, "n_tokens": r["n"]}
write("steering/baseline", [{"prompt": p, **b} for p, b in baseline.items()])

# ---- poles: one row per expert write direction and sign ----------------------------------------------------------
ranked = jsonl(MAIN / "ranked_matched.jsonl")
key = {k["id"]: k for k in jsonl(DATA / "judge/full_key.jsonl")}
pid = lambda L, e, p: f"L{L}_E{e}_{p}"  # noqa: E731
verdicts = jsonl(DATA / "judge/verdicts.jsonl")  # one per key item, same order
beam_verdicts = {v["id"]: {"correct": v["choice"] == key[v["id"]]["answer"], "confidence": v["confidence"]}
                 for v in jsonl(DATA / "selflabel/deepseek_beamlabel.jsonl")}
# 213 poles were drawn into both strata; a pole's columns come from its significant-stratum item when it has one,
# and judge/items keeps every item (summary rates are per item).
judge_tok, judge_beam = {}, {}
for k, v in sorted(zip(key.values(), verdicts), key=lambda kv: kv[0]["stratum"] != "sig", reverse=True):
    i = pid(k["layer"], k["expert"], k["pole"])
    judge_tok[i], judge_beam[i] = v, beam_verdicts.get(k["id"])
labels = {r["id"]: r for r in jsonl(DATA / "selflabel/labels.jsonl")}
# The study's pruning-emulation arm is left out of the public tables: only the subject bans, their controls and base.
SKIP_ARMS = {"reap"}
arms = [a for a in json.loads((DATA / "mmlu/arms.json").read_text())["arms"] if a["name"] not in SKIP_ARMS]
banned_in = collections.defaultdict(list)
for arm in arms:
    for L, e in arm["bans"]:
        banned_in[(L, e)].append(arm["name"])
effort_rd = {}
analysis = json.loads((MAIN / "effort_analysis.json").read_text())
for t in analysis["experts"]["think_start"]["top"]:
    effort_rd[(t["layer"], t["expert"])] = t["rd"]

# ranked_matched.jsonl carries the unit's lens (the tokens its + push promotes) on both of its pole rows, and the
# judge items and Beam's labels took the label from there. So only + poles were labelled and judged with their own
# tokens; a - push promotes the opposite side of the same weight direction, read here from the atlas. For - poles
# the label-based columns are left empty, and judge/items keeps their items flagged as a label/pole mismatch.
atlas = {}
for path in (DATA / "atlas").glob("experts_L*.jsonl"):
    for e in jsonl(path):
        for q in e["poles"]:
            atlas[(e["layer"], e["expert"], q["dir"], q["sign"])] = q
FLIP = {"+": "-", "-": "+"}
poles = []
for r in ranked:
    if r["kind"] != "expert_dir":
        continue
    i = pid(r["layer"], r["expert"], r["pole"])
    lab, jt, jb = labels.get(i, {}), judge_tok.get(i), judge_beam.get(i)
    lens, lens_z = r["lens"], r["z"]
    if r["pole"] == "-":
        opp = atlas[(r["layer"], r["expert"], r["dir"], FLIP[r["sign"]])]
        lens, lens_z, lab, jt, jb = opp["tokens"], opp["z_mean"], {}, None, None
    sig = r["null"] != "untestable" and (r["pct_matched"] or 0) >= 0.95
    poles.append({
        "id": i, "unit": r["unit"], "layer": r["layer"], "expert": r["expert"], "dir": r["dir"], "pole": r["pole"],
        "lens_tokens": tokens(lens), "lens_z": lens_z,
        "beam_name": lab.get("name"), "beam_description": lab.get("description"), "beam_subject": lab.get("subject"),
        "beam_coherent": lab.get("coherent"),
        "dnll": r["dnll"], "theme_score": r["score"], "theme_words": r["theme"], "null": r["null"],
        "pct_vs_matched_controls": r["pct_matched"], "significant": sig,
        "judge_stratum": None if jt is None else jt["stratum"],
        "judge_correct_token_label": None if jt is None else jt["correct"],
        "judge_confidence_token_label": None if jt is None else jt["confidence"],
        "judge_correct_beam_label": None if jb is None else jb["correct"],
        "judge_confidence_beam_label": None if jb is None else jb["confidence"],
        "banned_in": banned_in.get((r["layer"], r["expert"]), []),
        "effort_rd_think_start": effort_rd.get((r["layer"], r["expert"])),
    })
poles.sort(key=lambda p: (p["layer"], p["expert"], p["pole"]))
write("experts/poles", poles)

# ---- blind three-way test ----------------------------------------------------------------------------------------
write("judge/items", [{"item": k["id"], "pole_id": pid(k["layer"], k["expert"], k["pole"]), "stratum": k["stratum"],
                       "label_matches_pole": k["pole"] == "+",
                       "answer": k["answer"], "distractors": [pid(*d) for d in k["distractors"]],
                       "layer": k["layer"], "correct_token_label": v["correct"], "confidence_token_label": v["confidence"],
                       "correct_beam_label": (beam_verdicts.get(k["id"]) or {}).get("correct"),
                       "confidence_beam_label": (beam_verdicts.get(k["id"]) or {}).get("confidence")}
                      for k, v in zip(key.values(), verdicts)])

# ---- MMLU-Pro bans -----------------------------------------------------------------------------------------------
write("bans/arms", [{"arm": a["name"], "n_banned": len(a["bans"]), "banned": [f"L{L}_E{e}" for L, e in a["bans"]]}
                    for a in arms])
mmlu_run = DATA / "mmlu/runs/20261029-113719"
mmlu = []
for path in sorted(mmlu_run.glob("mmlu_*.jsonl.gz")):
    arm = path.name[len("mmlu_"):-len(".jsonl.gz")]
    if arm in SKIP_ARMS:
        continue
    for r in jsonl(path):
        mmlu.append({"arm": arm, "question_id": r["id"], "category": r["category"], "answer": r["answer"],
                     "pred": r["pred"], "correct": r["pred"] == r["answer"], "letter_logprobs": r["lp"]})
write("bans/mmlu_pro", mmlu)
score = json.loads((DATA / "mmlu/score.json").read_text())

# ---- effort ------------------------------------------------------------------------------------------------------
def effort_rows(path, extra=()):
    out = []
    for r in jsonl(path):
        a, f, st, _ = harmony(r["text"], r["finish"], r.get("distinct4"))
        # requests that hit the token cap never closed the analysis channel: all their tokens were thinking
        think = r["analysis_len"] if r.get("analysis_len") is not None else r["n"]
        out.append({**{k: r.get(k) for k in ("q", "effort", "gold", "n", "finish", "distinct4", "analysis_len",
                                               "correct", "answer", *extra)}, "think_tokens": think,
                    "status": st, "analysis": a, "final": f})
    return out


write("effort/gsm8k_dial", effort_rows(MAIN / "effort_dial.jsonl.gz"))
write("effort/gsm8k_steer", effort_rows(MAIN / "effort_steer.jsonl.gz", ("layer", "kind", "lambda", "dose_eff")))
e2 = []
for path in sorted(E2.glob("effort2_pair*.jsonl.gz")) + sorted(E2C.glob("effort2_pair*.jsonl.gz")):
    e2 += effort_rows(path, ("arm", "source", "dose_eff"))
SOURCES = {"aime_2024": "HuggingFaceH4/aime_2024", "aime_2025": "MathArena/aime_2025", "amc23": "math-ai/amc23"}
offsets = {}
for s in SOURCES:  # problems were concatenated in this order; q is the index into that list
    qs = sorted({r["q"] for r in e2 if r["source"] == s})
    offsets[s] = qs[0]
for r in e2:
    r["source_dataset"], r["source_row"] = SOURCES[r["source"]], r["q"] - offsets[r["source"]]
write("effort/competition_math", e2)
(args.out / "effort").mkdir(parents=True, exist_ok=True)
(args.out / "effort/analysis.json").write_text(json.dumps(analysis))
(args.out / "effort/direction_note.txt").write_text(
    "effort_direction.pt (session v2) is the per-layer unit direction of log2 N; kept in the private run repo.\n")

# ---- calibration -------------------------------------------------------------------------------------------------
stats = json.loads((MAIN / "stats.json").read_text())
calib = json.loads((MAIN / "calib.json").read_text())
write("calibration/resid_norms", [{"layer": L, "q10": q[0], "q50": q[1], "q90": q[2]}
                                  for L, q in enumerate(stats["q10_50_90"])])
write("calibration/dose_response", [{k: r.get(k) for k in ("layer", "control", "dose", "dnll", "distinct4", "n",
                                                           "finish", "analysis_len")}
                                     for r in jsonl(MAIN / "calib.jsonl.gz")])
(args.out / "calibration/calib.json").write_text(json.dumps(calib))

# ---- summary -----------------------------------------------------------------------------------------------------
def band(L):
    return f"L{8 * (L // 8)}-{min(8 * (L // 8) + 7, config['num_hidden_layers'] - 1)}"


def rate(xs):
    xs = [x for x in xs if x is not None]
    return {"n": len(xs), "rate": round(float(np.mean(xs)), 4) if xs else None}


testable = [p for p in poles if p["null"] != "untestable"]
by_band = collections.defaultdict(list)
for p in poles:
    by_band[band(p["layer"])].append(p)
depth = {b: {"poles": len(ps), "testable": sum(p["null"] != "untestable" for p in ps),
             "significant": rate([p["significant"] for p in ps if p["null"] != "untestable"]),
             "beam_coherent": rate([p["beam_coherent"] for p in ps if p["beam_name"]]),
             "judge_sig_token": rate([v["correct"] for k, v in zip(key.values(), verdicts)
                                      if k["stratum"] == "sig" and k["pole"] == "+" and band(k["layer"]) == b]),
             "judge_random_token": rate([v["correct"] for k, v in zip(key.values(), verdicts)
                                         if k["stratum"] == "random" and k["pole"] == "+" and band(k["layer"]) == b])}
         for b, ps in sorted(by_band.items(), key=lambda kv: int(kv[0][1:].split("-")[0]))}
e2_acc = collections.defaultdict(list)
e2_think = collections.defaultdict(list)
for r in e2:
    e2_acc[r["arm"]].append(bool(r["correct"]))
    e2_think[r["arm"]].append(r["think_tokens"])
summary = {
    "model": "reflection/Beam-1.0-501B-A23B (NVFP4 variant, vLLM, 4x RTX PRO 6000)",
    "moe_layers": MOE, "experts_per_layer": 256,
    "sweep": {"units": len(units), "expert_poles": len(poles), "testable": len(testable),
              "significant": sum(p["significant"] for p in poles),
              "expected_by_chance": round(0.05 * len(testable))},
    # label-based results on + poles, whose label describes the pushed direction (see the note above poles)
    "judge": {s: rate([v["correct"] for k, v in zip(key.values(), verdicts) if k["stratum"] == s and k["pole"] == "+"])
              for s in ("sig", "random")},
    "judge_beam_label": {s: rate([beam_verdicts[k["id"]]["correct"] for k in key.values()
                                  if k["stratum"] == s and k["pole"] == "+" and k["id"] in beam_verdicts])
                         for s in ("sig", "random")},
    "judge_confident_sig": rate([v["correct"] for k, v in zip(key.values(), verdicts)
                                 if k["stratum"] == "sig" and k["pole"] == "+" and v["confidence"] == 3]),
    "judge_mismatched_label": {s: rate([v["correct"] for k, v in zip(key.values(), verdicts)
                                        if k["stratum"] == s and k["pole"] == "-"]) for s in ("sig", "random")},
    "self_label": {"labelled": sum(p["beam_name"] is not None for p in poles),
                   "coherent": sum(bool(p["beam_coherent"]) for p in poles),
                   "sig_if_coherent": rate([p["significant"] for p in testable if p["beam_coherent"] is True]),
                   "sig_if_incoherent": rate([p["significant"] for p in testable if p["beam_coherent"] is False])},
    "depth": depth,
    "bans": score["bans"],
    "effort": {"readout_r2": {k: v["r2"] for k, v in analysis["readout"].items()},
               "off_dial": analysis["off_dial"],
               "competition_math": {a: {"n": len(v), "acc": round(float(np.mean(v)), 3),
                                        "median_think": float(np.median(e2_think[a]))} for a, v in e2_acc.items()}},
}
(args.out / "summary.json").write_text(json.dumps(summary, indent=1))
(args.out / "README.md").write_text((pathlib.Path(__file__).parent / "atlas_card.md").read_text())
print("summary.json and README.md written")

# ---- explorer ----------------------------------------------------------------------------------------------------
if not args.no_explorer:
    ex = args.out / "explorer"
    ex.mkdir(parents=True, exist_ok=True)
    clip = lambda s, n: s if len(s) <= n else s[:n] + "…"  # noqa: E731
    by_unit = collections.defaultdict(list)
    for o in outputs:
        by_unit[o["unit"]].append(o)
    index, gallery = [], {}
    for L in MOE:
        layer = {}
        for p in (p for p in poles if p["layer"] == L):
            e = layer.setdefault(p["expert"], {"expert": p["expert"],
                                               "banned_in": p["banned_in"], "effort_rd": p["effort_rd_think_start"],
                                               "poles": {}})
            doses = (0.4, 1.0) if p["pole"] == "+" else (-1.0,)
            e["poles"][p["pole"]] = {
                "name": p["beam_name"], "desc": p["beam_description"], "subject": p["beam_subject"],
                "coherent": p["beam_coherent"], "tokens": p["lens_tokens"][:12], "sig": p["significant"],
                "pct": p["pct_vs_matched_controls"], "null": p["null"], "dnll": p["dnll"], "theme": p["theme_words"],
                "judge": p["judge_correct_token_label"], "judge_beam": p["judge_correct_beam_label"],
                "out": [{"prompt": o["prompt"], "dose": o["dose"], "analysis": clip(o["analysis"], 360),
                         "final": clip(o["final"], 480), "finish": o["finish"], "status": o["status"],
                         **({"raw": clip(o["raw"], 600)} if o["status"] in ("format", "loop") else {})}
                        for o in by_unit[p["unit"]] if o["dose"] in doses or (o["prompt"] == "effort" and
                                                                           (o["dose"] > 0) == (p["pole"] == "+"))]}
            if p["id"] in GALLERY:
                gallery[p["id"]] = {"id": p["id"], "layer": L, "pole": p["pole"], **e}
            verified = (p["judge_stratum"] == "sig" and p["judge_correct_token_label"]
                        and p["judge_confidence_token_label"] == 3)
            index.append([p["id"], p["beam_name"] or "", p["beam_subject"] or "", int(bool(p["beam_coherent"])),
                          int(p["significant"]), -1 if p["judge_correct_token_label"] is None else
                          int(p["judge_correct_token_label"]), int(bool(verified)),
                          " ".join(t.strip() for t in p["lens_tokens"][:6])])
        (ex / f"L{L}.json").write_text(json.dumps(sorted(layer.values(), key=lambda e: e["expert"]),
                                                  ensure_ascii=False, separators=(",", ":")))
    (ex / "index.json").write_text(json.dumps({"fields": ["id", "name", "subject", "coherent", "sig", "judge",
                                                          "verified", "tokens"], "rows": index},
                                              ensure_ascii=False, separators=(",", ":")))
    (ex / "baseline.json").write_text(json.dumps(baseline,
                                                 ensure_ascii=False))
    (ex / "prompts.json").write_text(json.dumps(PROMPTS))
    (ex / "gallery.json").write_text(json.dumps([gallery[i] for i in GALLERY], ensure_ascii=False))
    (ex / "summary.json").write_text(json.dumps(summary, separators=(",", ":")))
    size = sum(f.stat().st_size for f in ex.iterdir())
    print(f"explorer: {len(list(ex.iterdir()))} files, {size / 1e6:.1f} MB")
