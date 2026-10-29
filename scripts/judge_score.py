# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Score a judge's answers to judge_items.py items against the key.

    uv run scripts/judge_score.py KEY.jsonl ANSWERS.jsonl [--out verdicts.jsonl]

Accuracy (chance 1/3) with a normal 95% interval, by stratum, by 8-layer band and by the judge's confidence; with
--out, one verdict per expert pole (layer, expert, pole, correct, confidence) for the explorer.
"""

import argparse
import collections
import json
import pathlib

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("key", type=pathlib.Path)
parser.add_argument("answers", type=pathlib.Path)
parser.add_argument("--out", type=pathlib.Path)
args = parser.parse_args()

key = {r["id"]: r for r in map(json.loads, open(args.key))}
ans = {r["id"]: r for r in map(json.loads, open(args.answers)) if r.get("choice")}
ids = [i for i in key if i in ans]
print(f"{len(ids)} of {len(key)} items answered")


def acc(sub):
    if not sub:
        return "-"
    p = np.mean([ans[i]["choice"] == key[i]["answer"] for i in sub])
    h = 1.96 * np.sqrt(p * (1 - p) / len(sub))
    return f"{p:5.1%} [{p - h:.0%}, {p + h:.0%}] n={len(sub)}"


strata = sorted({key[i]["stratum"] for i in ids})
for s in strata:
    sub = [i for i in ids if key[i]["stratum"] == s]
    print(f"{s:7s} {acc(sub)}")
    for c in (3, 2, 1):
        print(f"          confidence {c}: {acc([i for i in sub if ans[i]['confidence'] == c])}")
print("by band:  " + "  ".join(f"[{s}]" for s in strata))
for b in range(7):
    print(f"  L{8 * b:2d}-{8 * b + 7:2d}  " + "   ".join(acc([i for i in ids if key[i]["stratum"] == s and key[i]["layer"] // 8 == b])
                                              for s in strata))
if args.out:
    with open(args.out, "w") as f:
        for i in ids:
            k = key[i]
            f.write(json.dumps({"layer": k["layer"], "expert": k["expert"], "pole": k["pole"], "stratum": k["stratum"],
                                "correct": ans[i]["choice"] == k["answer"], "confidence": ans[i]["confidence"]}) + "\n")
