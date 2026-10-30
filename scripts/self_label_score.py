# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Score Beam's own expert labels (self_label.py) against the blind three-way test and the steering sweep.

    uv run scripts/self_label_score.py KEY.jsonl TOKEN_ANSWERS.jsonl BEAM_ANSWERS.jsonl LABELS.jsonl [--out OUT.json]

KEY and TOKEN_ANSWERS are the judge set and a judge's answers with token labels (judge_items.py, judge_router.py);
BEAM_ANSWERS are the same judge's answers with Beam's labels (label_items.py, judge_router.py --label-kind
description); LABELS are Beam's pole labels. Reports:

1. Accuracy with token labels vs Beam's labels on the same items (chance 1/3), by stratum and 8-layer band, with an
   exact McNemar test on the paired items.
2. Beam's coherence flag: does "coherent" predict that the pole is steering-significant, and that the judge picks its
   steered output (Fisher exact, one-sided)? Poles Beam could not label (thinking ran to the cap) are reported apart.
3. Coherence rate by 8-layer band over every labelled pole, and the subject counts.

--plus-only keeps + poles only. ranked_matched.jsonl carries the unit's lens (what its + push promotes) on both pole
rows, so every - pole was judged with, and labelled from, the + side's tokens: only + poles test a label against the
direction that was pushed.
"""

import argparse
import collections
import json
import math
import pathlib

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("key", type=pathlib.Path)
parser.add_argument("token_answers", type=pathlib.Path)
parser.add_argument("beam_answers", type=pathlib.Path)
parser.add_argument("labels", type=pathlib.Path)
parser.add_argument("--out", type=pathlib.Path)
parser.add_argument("--plus-only", action="store_true", help="+ poles only (see above)")
args = parser.parse_args()


def pole_id(k):
    return f"L{k['layer']}_E{k['expert']}_{k['pole']}"


key = {k["id"]: k for k in map(json.loads, open(args.key)) if not args.plus_only or k["pole"] == "+"}
tok = {r["id"]: r for r in map(json.loads, open(args.token_answers)) if r.get("choice")}
beam = {r["id"]: r for r in map(json.loads, open(args.beam_answers)) if r.get("choice")}
labels = {r["id"]: r for r in map(json.loads, open(args.labels)) if not args.plus_only or r["id"].endswith("_+")}
done = {i: r for i, r in labels.items() if r["name"]}


def ci(p, n):
    h = 1.96 * math.sqrt(p * (1 - p) / n) if n else 0
    return [round(p - h, 3), round(p + h, 3)]


def acc(ans, ids):
    ids = [i for i in ids if i in ans]
    if not ids:
        return None
    p = float(np.mean([ans[i]["choice"] == key[i]["answer"] for i in ids]))
    return {"n": len(ids), "acc": round(p, 3), "ci95": ci(p, len(ids))}


def mcnemar(ids):
    b = sum(1 for i in ids if tok[i]["choice"] == key[i]["answer"] and beam[i]["choice"] != key[i]["answer"])
    c = sum(1 for i in ids if tok[i]["choice"] != key[i]["answer"] and beam[i]["choice"] == key[i]["answer"])
    p = min(1.0, 2 * sum(math.comb(b + c, j) for j in range(min(b, c) + 1)) / 2 ** (b + c)) if b + c else 1.0
    return {"token_only_right": b, "beam_only_right": c, "p_exact": round(p, 4)}


def fisher_greater(a, b, c, d):
    """One-sided Fisher exact p for [[a, b], [c, d]], alternative: row 1 has the higher rate of column 1."""
    n1, n2, k = a + b, c + d, a + c
    total = math.comb(n1 + n2, k)
    return sum(math.comb(n1, x) * math.comb(n2, k - x) for x in range(a, min(n1, k) + 1)) / total


report = {}
paired = [i for i in key if i in tok and i in beam]
unlabelled = [i for i in key if pole_id(key[i]) not in done]
report["items"] = {"total": len(key), "paired": len(paired), "pole_not_labelled_by_beam": len(unlabelled)}
print(f"{len(paired)} of {len(key)} items judged with both labels; {len(unlabelled)} items whose pole Beam could not label")

print("\n1. three-way accuracy, same judge, same items (chance 33%)")
report["accuracy"] = {}
for s in ("sig", "random"):
    ids = [i for i in paired if key[i]["stratum"] == s]
    t, b, m = acc(tok, ids), acc(beam, ids), mcnemar(ids)
    report["accuracy"][s] = {"token": t, "beam": b, "mcnemar": m}
    print(f"  {s:6s} n={len(ids)}  token {t['acc']:.1%} {t['ci95']}  Beam {b['acc']:.1%} {b['ci95']}  "
          f"token-only {m['token_only_right']} / Beam-only {m['beam_only_right']}  p={m['p_exact']}")
report["accuracy_by_band"] = {}
for band in range(7):
    row = {}
    for s in ("sig", "random"):
        ids = [i for i in paired if key[i]["stratum"] == s and key[i]["layer"] // 8 == band]
        row[s] = {"token": acc(tok, ids), "beam": acc(beam, ids)}
    report["accuracy_by_band"][f"L{8 * band}-{8 * band + 7}"] = row
    fmt = lambda r: f"{r['acc']:.0%} (n={r['n']})" if r else "-"
    print(f"  L{8 * band:2d}-{8 * band + 7:2d}  sig token {fmt(row['sig']['token']):13s} Beam {fmt(row['sig']['beam']):13s}"
          f"  random token {fmt(row['random']['token']):13s} Beam {fmt(row['random']['beam'])}")

print("\n2. Beam's coherence flag on judge-set poles")
poles = {}
for i in key:
    p = pole_id(key[i])
    poles.setdefault(p, {"sig": False, "correct": [], "layer": key[i]["layer"]})
    poles[p]["sig"] |= key[i]["stratum"] == "sig"
    if i in beam:
        poles[p]["correct"].append(beam[i]["choice"] == key[i]["answer"])
coh = {p: done[p]["coherent"] is True for p in poles if p in done}
report["coherence"] = {}
for name, val in (("steering-significant", lambda p: poles[p]["sig"]),
                  ("judge picks it (Beam label)", lambda p: bool(poles[p]["correct"]) and np.mean(poles[p]["correct"]) > .5)):
    a = sum(1 for p in coh if coh[p] and val(p))
    b = sum(1 for p in coh if coh[p] and not val(p))
    c = sum(1 for p in coh if not coh[p] and val(p))
    d = sum(1 for p in coh if not coh[p] and not val(p))
    p = fisher_greater(a, b, c, d)
    report["coherence"][name] = {"coherent": [a, a + b], "incoherent": [c, c + d], "fisher_one_sided_p": p}
    print(f"  {name:28s} coherent {a}/{a + b} ({a / max(a + b, 1):.0%})  incoherent {c}/{c + d} ({c / max(c + d, 1):.0%})"
          f"  p={p:.2g}")
stuck = [p for p in poles if p not in done]
print(f"  Beam could not label {len(stuck)} judge-set poles (thinking hit the cap); "
      f"{sum(poles[p]['sig'] for p in stuck)} of them steering-significant")
report["coherence"]["unlabelled_poles"] = {"n": len(stuck), "sig": sum(poles[p]["sig"] for p in stuck)}

print("\n3. all labelled poles: coherence by depth, subjects")
band_coh = collections.defaultdict(list)
for p, r in done.items():
    band_coh[int(p.split("_")[0][1:]) // 8].append(r["coherent"] is True)
report["coherence_by_band"] = {f"L{8 * b}-{8 * b + 7}": {"n": len(v), "coherent": round(float(np.mean(v)), 3)}
                               for b, v in sorted(band_coh.items())}
for b, v in report["coherence_by_band"].items():
    print(f"  {b:7s} n={v['n']:5d} coherent {v['coherent']:.0%}")
subjects = collections.Counter(r.get("subject") or "unparsed" for r in done.values())
report["subjects"] = dict(subjects.most_common())
print("  subjects:", dict(subjects.most_common()))
failed = {i: r for i, r in labels.items() if not r["name"]}
report["labels"] = {"labelled": len(done), "failed": len(failed)}
print(f"  {len(done)} poles labelled, {len(failed)} not (still failing after retries)")
if args.out:
    args.out.write_text(json.dumps(report, indent=1))
