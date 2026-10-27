# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Rank swept units against random-direction controls, after Lyra's neuron-explorer `rank.py`.

    uv run scripts/rank_sweep.py RUN_DIR [--meta data/sweep/units.jsonl]   # -> RUN_DIR/ranked.jsonl

Per unit and pole (positive doses +0.5/+0.8, negative dose -0.7), the theme is the set of words that show up in the
steered rollouts of at least two of the three unrelated prompts (who / snake / orange) and in none of the unsteered
baselines. A concept the unit writes should surface whatever the prompt; a generic disruption should not repeat.
Each theme word counts log(N_poles / poles with that word), so words every unit produces count for little.
Units are scored against the controls of nearby layers (same band of 8 layers): `pct` is the share of those
controls that score below the unit. Effort: thinking length (tokens before the analysis channel's <|end|>) and
correctness on the effort prompt, vs the unsteered baseline.
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
WORD = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
TOPICAL = ("who", "snake", "orange")


def words(text):
    text = re.sub(r"<\|[^|]*\|>", " ", text or "")
    return {w.lower() for w in WORD.findall(text)}


def load(pattern):
    for path in sorted(args.run.glob(pattern)):
        yield from map(json.loads, gzip.open(path, "rt"))


metas = {m["unit"]: m for m in map(json.loads, open(args.meta))}
base = {r["prompt"]: r for r in load("baseline.jsonl.gz")}
base_words = set().union(*(words(base[p].get("text")) for p in TOPICAL if p in base))

by_pole = collections.defaultdict(lambda: collections.defaultdict(set))  # (unit, pole) -> prompt -> words
effort = collections.defaultdict(dict)  # unit -> dose -> row
for r in load("sweep_*.jsonl.gz"):
    if r["prompt"] == "effort":
        effort[r["unit"]][r["dose"]] = r
        continue
    pole = "+" if r["dose"] > 0 else "-"
    by_pole[(r["unit"], pole)][r["prompt"]] |= words(r.get("text")) - base_words

themes = {}
for key, per_prompt in by_pole.items():
    counts = collections.Counter(w for ws in per_prompt.values() for w in ws)
    themes[key] = {w for w, n in counts.items() if n >= 2}
df = collections.Counter(w for t in themes.values() for w in t)
n_poles = max(1, len(themes))
score = {k: sum(math.log(n_poles / df[w]) for w in t) for k, t in themes.items()}

band = lambda layer: layer // 8  # noqa: E731
controls = collections.defaultdict(list)
for (u, pole), s in score.items():
    if metas[u]["kind"] == "control":
        controls[band(metas[u]["layer"])].append(s)


def correct(row):
    text = (row or {}).get("text") or ""
    return bool(re.search(r"\$?0?\.05\b|5\s*cents|five cents", text.split("final")[-1]))


base_effort = base.get("effort", {})
out = []
for (u, pole), s in score.items():
    m = metas[u]
    ctrl = controls[band(m["layer"])]
    top = sorted(themes[(u, pole)], key=lambda w: -math.log(n_poles / df[w]))[:12]
    row = {**{k: v for k, v in m.items() if k != "lens"}, "pole": pole, "score": round(s, 2),
           "pct": round(float(np.mean([c < s for c in ctrl])), 3) if ctrl else None, "theme": top, "lens": m.get("lens")}
    e = effort.get(u, {}).get(0.8 if pole == "+" else -0.7)
    if e is not None and base_effort:
        row["effort"] = {"analysis_len": e.get("analysis_len"), "base_analysis_len": base_effort.get("analysis_len"),
                         "final": e.get("final"), "correct": correct(e), "base_correct": correct(base_effort)}
    out.append(row)
out.sort(key=lambda r: (r["kind"] == "control", -(r["pct"] or 0), -r["score"]))
with open(args.run / "ranked.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
real = [r for r in out if r["kind"] != "control"]
print(f"{len(real)} unit poles, {len(out) - len(real)} control poles -> {args.run}/ranked.jsonl")
print(f"above every control in their band: {sum((r['pct'] or 0) == 1.0 for r in real)}; "
      f"above p95: {sum((r['pct'] or 0) >= 0.95 for r in real)} (5% expected by chance: {0.05 * len(real):.0f})")
for r in real[:10]:
    print(f"  L{r['layer']:2d} {r['kind']:12s} {r.get('expert', r.get('neuron'))} {r['pole']}  pct {r['pct']}  "
          f"theme {' '.join(r['theme'][:8])}")
