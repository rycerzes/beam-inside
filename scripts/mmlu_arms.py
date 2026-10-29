# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Expert-ban arms for session.py's mmlu phase.

    uv run scripts/mmlu_arms.py REAP_SALIENCY.json BEAM_LABELS.jsonl [--subjects math,health,law] [--out data/mmlu/arms.json]

Arms, in run order (the session saves each before starting the next):
  base             nothing banned
  <subject>        every expert with a coherent pole Beam labelled with that subject (self_label.py)
  <subject>_ctrl   the same number of experts per layer, none labelled with any tested subject, each matched to a
                   banned expert of the same layer on REAP's routing count (nearest log count, greedy, no reuse):
                   how often the router picks an expert, from REAP's ~925k calibration tokens
  reap             every expert REAP pruned (the 128 per layer not in `kept`): the pruned model, emulated

Each arm keeps at least ceil(top_k / topk_group) = 2 experts in every routing group (checked here and in session).
"""

import argparse
import json
import math
import pathlib

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("reap", type=pathlib.Path)
parser.add_argument("labels", type=pathlib.Path)
parser.add_argument("--subjects", default="math,health,law")
parser.add_argument("--experts", type=int, default=256)
parser.add_argument("--groups", type=int, default=8)
parser.add_argument("--need", type=int, default=2, help="experts every routing group must keep")
parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/mmlu/arms.json"))
args = parser.parse_args()

reap = json.load(open(args.reap))["layers"]
moe = sorted(int(L) for L, v in reap.items() if v.get("type") != "dense")
count = {L: np.asarray(reap[str(L)]["count"], float) for L in moe}
subjects = args.subjects.split(",")

labelled = {s: set() for s in subjects}
for r in map(json.loads, open(args.labels)):
    if r["name"] and r.get("coherent") is True and r.get("subject") in labelled:
        L, E, _ = r["id"].split("_")
        labelled[r["subject"]].add((int(L[1:]), int(E[1:])))
any_subject = set().union(*labelled.values())


def control(target):
    """Same count per layer, outside every tested subject, matched on log routing count."""
    out, gap = [], []
    for L in moe:
        mine = sorted(e for l, e in target if l == L)
        if not mine:
            continue
        pool = [e for e in range(args.experts) if (L, e) not in any_subject]
        logc = np.log1p(count[L])
        for e in sorted(mine, key=lambda e: -logc[e]):  # most-used first, while the pool is fullest
            best = min(pool, key=lambda c: abs(logc[c] - logc[e]))
            pool.remove(best)
            out.append((L, best))
            gap.append(abs(logc[best] - logc[e]))
    return sorted(out), float(np.mean(gap)) if gap else 0.0


def check(name, bans):
    per = args.experts // args.groups
    kept = {}
    for L, e in bans:
        kept[(L, e // per)] = kept.get((L, e // per), per) - 1
    worst = min(kept.values(), default=per)
    assert worst >= args.need, f"{name}: a routing group keeps {worst} experts"
    return worst


arms = [{"name": "base", "bans": []}]
for s in subjects:
    target = sorted(labelled[s])
    ctrl, gap = control(target)
    for name, bans in ((s, target), (f"{s}_ctrl", ctrl)):
        arms.append({"name": name, "bans": [list(b) for b in bans]})
    tc = np.mean([math.log1p(count[L][e]) for L, e in target])
    cc = np.mean([math.log1p(count[L][e]) for L, e in ctrl])
    print(f"{s:8s} {len(target):4d} experts over {len({L for L, _ in target}):2d} layers; control mean log count "
          f"{cc:.2f} vs {tc:.2f} (mean |gap| {gap:.3f})")
arms.append({"name": "reap", "bans": [[L, e] for L in moe for e in range(args.experts) if e not in set(reap[str(L)]["kept"])]})
for a in arms:
    a["min_group_kept"] = check(a["name"], a["bans"])
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps({"arms": arms, "source": {"reap": str(args.reap), "labels": str(args.labels)}}))
print("arms:", [(a["name"], len(a["bans"]), a["min_group_kept"]) for a in arms], "->", args.out)
