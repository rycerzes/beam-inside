# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Re-rank a v2 sweep against effect-matched controls (after rank_sweep.py has written RUN/ranked.jsonl).

    uv run scripts/effect_rank.py RUN_DIR [--meta data/sweep/units.jsonl]   # -> RUN_DIR/ranked_matched.jsonl

rank_sweep.py compares each unit with the controls of its 8-layer band at the same dose in units of the residual
norm. Expert directions and random directions of the same norm move Beam by very different amounts (ΔNLL), so the
v2 session records each unit's own ΔNLL at its strongest dose of each pole. Here a unit pole's null is the control
poles within ±8 layers whose ΔNLL is within 1.5x of the unit's (at least 20 of them):
  matched     enough controls in that window
  under       the unit moves Beam less than all but a few nearby controls: compared with the 20 weakest of them,
              which is conservative, since control theme scores rise with ΔNLL over this range
  untestable  the unit moves Beam more than nearly every nearby control (late-layer collapse); no fair null
Lens agreement: share of significant poles whose theme shares a 4-letter stem with one of the unit's lens tokens,
against the same share with lens labels shuffled among units.
"""

import argparse
import collections
import gzip
import json
import math
import pathlib
import re

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("run", type=pathlib.Path)
parser.add_argument("--meta", type=pathlib.Path, default=pathlib.Path("data/sweep/units.jsonl"))
args = parser.parse_args()
WINDOW, MIN_CONTROLS, RATIO = 8, 20, 1.5

ranked = [json.loads(line) for line in open(args.run / "ranked.jsonl")]
dnll = {}
for path in sorted(args.run.glob("sweep_*.jsonl.gz")):
    for r in map(json.loads, gzip.open(path, "rt")):
        dnll[(r["unit"], r["dose"])] = r["dnll"]
for r in ranked:
    r["dnll"] = dnll[(r["unit"], max(d for (u, d) in [(r["unit"], 1.0), (r["unit"], -1.0)] if (u, d) in dnll and
                                     (d > 0) == (r["pole"] == "+")))]
controls = [r for r in ranked if r["kind"] == "control"]
c_layer = np.array([c["layer"] for c in controls])
c_dnll = np.maximum(np.array([c["dnll"] for c in controls]), 1e-4)
c_score = np.array([c["score"] for c in controls])

for r in ranked:
    if r["kind"] == "control":
        continue
    near = np.abs(c_layer - r["layer"]) <= WINDOW
    d = max(r["dnll"], 1e-4)
    window = near & (np.abs(np.log(c_dnll / d)) <= math.log(RATIO))
    if window.sum() >= MIN_CONTROLS:
        null, r["null"] = c_score[window], "matched"
    elif (c_dnll[near] > d).sum() >= len(c_dnll[near]) - MIN_CONTROLS // 4:
        idx = np.flatnonzero(near)[np.argsort(c_dnll[near])[:MIN_CONTROLS]]
        null, r["null"] = c_score[idx], "under"
    else:
        null, r["null"] = None, "untestable"
    r["pct_matched"] = None if null is None else round(float(np.mean(null < r["score"])), 3)

WORD = re.compile(r"[a-z]{4,}")


def stems(tokens):
    out = set()
    for t in tokens or []:
        t = t.replace("Ġ", " ").replace("Ċ", " ").lower()
        out |= {w[:4] for w in WORD.findall(t)}
    return out


def agrees(r, lens):
    return bool(stems(lens) & {w[:4] for w in r["theme"] if len(w) >= 4})


units = [r for r in ranked if r["kind"] != "control"]
sig = [r for r in units if (r["pct_matched"] or 0) >= 0.95]
rng = np.random.default_rng(0)
shuffled = [u["lens"] for u in units]
rng.shuffle(shuffled)
lens_of = {id(u): s for u, s in zip(units, shuffled)}
print(f"{len(units)} unit poles; null: " + ", ".join(f"{k} {v}" for k, v in collections.Counter(r['null'] for r in units).items()))
print(f"significant vs effect-matched null (pct >= 0.95): {len(sig)} ({len(sig) / len(units):.1%}; "
      f"5% expected among testable: {0.05 * sum(r['null'] != 'untestable' for r in units):.0f})")
print(f"lens agreement among significant: {np.mean([agrees(r, r['lens']) for r in sig]):.0%} "
      f"vs shuffled lens {np.mean([agrees(r, lens_of[id(r)]) for r in sig]):.0%}")
print("band     testable  hit% (matched null)  hit% (band null, rank_sweep)   + pole / - pole")
for b in range(7):
    us = [r for r in units if r["layer"] // 8 == b and r["kind"] == "expert_dir"]
    if not us:
        continue
    test = [r for r in us if r["null"] != "untestable"]
    hit = lambda rs, k: np.mean([(r[k] or 0) >= 0.95 for r in rs]) if rs else float("nan")  # noqa: E731
    print(f"L{8 * b:2d}-{8 * b + 7:2d}  {len(test) / len(us):6.0%}  {hit(test, 'pct_matched'):10.1%}  {hit(us, 'pct'):18.1%}"
          f"   {hit([r for r in test if r['pole'] == '+'], 'pct_matched'):6.1%} / {hit([r for r in test if r['pole'] == '-'], 'pct_matched'):5.1%}")
with open(args.run / "ranked_matched.jsonl", "w") as f:
    for r in sorted(ranked, key=lambda r: (r["kind"] == "control", -(r.get("pct_matched") or 0), -r["score"])):
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
