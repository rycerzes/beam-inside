# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Score the mmlu phase: each subject ban against its count-matched control, and the REAP emulation against base.

    uv run scripts/mmlu_score.py RUN_DIR [--crosscheck data/reap/crosscheck_beam.json]
        [--peterhdd data/reap/mmlu_official-nvfp4.txt data/reap/mmlu_reap-bf16.txt] [--out OUT.json]

RUN_DIR holds the session's mmlu_<arm>.jsonl.gz. Every arm scored the same questions, so comparisons are paired per
question: accuracy difference with an exact McNemar test, and the mean difference in the correct letter's logprob
with a 95% bootstrap interval (2,000 resamples of questions). For a ban, "own" is its subject's MMLU-Pro category
and "rest" the other categories pooled; the specificity test is whether own drops more than rest (difference of the
two mean logprob changes, bootstrap). With --crosscheck and --peterhdd, REAP's per-category accuracy drop here is
set against the share of Beam-labelled experts REAP kept per subject and against peterhdd's 30-question drops.
"""

import argparse
import gzip
import json
import math
import pathlib

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("run", type=pathlib.Path)
parser.add_argument("--crosscheck", type=pathlib.Path)
parser.add_argument("--peterhdd", type=pathlib.Path, nargs=2)
parser.add_argument("--out", type=pathlib.Path)
args = parser.parse_args()
rng = np.random.default_rng(0)

arms = {p.name[len("mmlu_"):-len(".jsonl.gz")]: {r["id"]: r for r in map(json.loads, gzip.open(p, "rt"))}
        for p in sorted(args.run.glob("mmlu_*.jsonl.gz"))}
ids = sorted(set.intersection(*(set(a) for a in arms.values())))
base = arms["base"]
cat = np.array([base[i]["category"] for i in ids])
print(f"arms {list(arms)}; {len(ids)} questions in every arm")


def vec(arm):
    a = arms[arm]
    correct = np.array([a[i]["pred"] == a[i]["answer"] for i in ids])
    lp = np.array([a[i]["lp"][a[i]["answer"]] for i in ids])
    return correct, lp


def mcnemar(x, y):
    b, c = int(np.sum(x & ~y)), int(np.sum(~x & y))
    return min(1.0, 2 * sum(math.comb(b + c, k) for k in range(min(b, c) + 1)) / 2 ** (b + c)) if b + c else 1.0


def boot(d, n=2000):
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def compare(a, b, mask):
    (ca, la), (cb, lb) = vec(a), vec(b)
    d = la[mask] - lb[mask]
    return {"n": int(mask.sum()), "acc_a": round(float(ca[mask].mean()), 4), "acc_b": round(float(cb[mask].mean()), 4),
            "acc_diff": round(float(ca[mask].mean() - cb[mask].mean()), 4), "mcnemar_p": mcnemar(ca[mask], cb[mask]),
            "lp_diff": round(float(d.mean()), 4), "lp_diff_ci95": boot(d)}


report = {"questions": len(ids), "bans": {}}
subjects = [a for a in arms if a not in ("base", "reap") and not a.endswith("_ctrl") and f"{a}_ctrl" in arms]
for s in subjects:
    own = cat == s
    rest = ~own
    by_cat = {c: compare(s, f"{s}_ctrl", cat == c) for c in sorted(set(cat))}
    _, l_ban = vec(s)
    _, l_ctrl = vec(f"{s}_ctrl")
    d = l_ban - l_ctrl
    # specificity: own-category logprob drop minus the rest's, resampled within each group
    spec = [d[own][rng.integers(0, own.sum(), own.sum())].mean() - d[rest][rng.integers(0, rest.sum(), rest.sum())].mean()
            for _ in range(2000)]
    report["bans"][s] = {"own": compare(s, f"{s}_ctrl", own), "rest": compare(s, f"{s}_ctrl", rest),
                         "ctrl_vs_base_all": compare(f"{s}_ctrl", "base", np.ones(len(ids), bool)),
                         "specificity_lp": round(float(d[own].mean() - d[rest].mean()), 4),
                         "specificity_ci95": [round(float(np.percentile(spec, 2.5)), 4),
                                              round(float(np.percentile(spec, 97.5)), 4)],
                         "by_category": by_cat}
    r = report["bans"][s]
    print(f"\n{s} ban vs its control (paired, n own {r['own']['n']}, rest {r['rest']['n']}):")
    print(f"  own   acc {r['own']['acc_diff'] * 100:+5.1f} pts (McNemar p={r['own']['mcnemar_p']:.2g}), "
          f"lp {r['own']['lp_diff']:+.3f} {r['own']['lp_diff_ci95']}")
    print(f"  rest  acc {r['rest']['acc_diff'] * 100:+5.1f} pts (p={r['rest']['mcnemar_p']:.2g}), "
          f"lp {r['rest']['lp_diff']:+.3f} {r['rest']['lp_diff_ci95']}")
    print(f"  own minus rest (lp): {r['specificity_lp']:+.3f} {r['specificity_ci95']}; control vs base overall "
          f"acc {r['ctrl_vs_base_all']['acc_diff'] * 100:+.1f} pts")
    worst = sorted(by_cat.items(), key=lambda kv: kv[1]["acc_diff"])[:5]
    print("  largest drops:", ", ".join(f"{c} {v['acc_diff'] * 100:+.1f} (p={v['mcnemar_p']:.1g})" for c, v in worst))

if "reap" in arms:
    by_cat = {c: compare("reap", "base", cat == c) for c in sorted(set(cat))}
    report["reap"] = {"all": compare("reap", "base", np.ones(len(ids), bool)), "by_category": by_cat}
    r = report["reap"]["all"]
    print(f"\nreap emulation vs base: acc {r['acc_a']:.3f} vs {r['acc_b']:.3f} ({r['acc_diff'] * 100:+.1f} pts, "
          f"p={r['mcnemar_p']:.2g}), lp {r['lp_diff']:+.3f}")
    rows = sorted(by_cat.items(), key=lambda kv: kv[1]["acc_diff"])
    print("  by category:", ", ".join(f"{c} {v['acc_diff'] * 100:+.1f}" for c, v in rows))
    rank = lambda x: np.argsort(np.argsort(x))
    if args.crosscheck:
        kept = json.loads(args.crosscheck.read_text())["beam_subjects"]
        cs = [c for c in by_cat if c in kept]
        rho = float(np.corrcoef(rank([kept[c]["kept"] for c in cs]), rank([by_cat[c]["acc_diff"] for c in cs]))[0, 1])
        report["reap"]["spearman_kept_vs_drop"] = {"rho": round(rho, 3), "n": len(cs)}
        print(f"  Spearman(share of Beam-labelled experts REAP kept, accuracy change) over {len(cs)} subjects: {rho:+.2f}"
              " (positive = subjects that lost more experts lost more accuracy)")
    if args.peterhdd:
        def acc(path):
            out = {}
            for line in open(path):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) > 7 and parts[1].startswith("- "):
                    out[parts[1][2:].replace("_", " ")] = float(parts[7])
            return out
        a, b = acc(args.peterhdd[0]), acc(args.peterhdd[1])
        cs = [c for c in by_cat if c in a and c in b]
        theirs = [b[c] - a[c] for c in cs]
        rho = float(np.corrcoef(rank(theirs), rank([by_cat[c]["acc_diff"] for c in cs]))[0, 1])
        report["reap"]["spearman_vs_peterhdd"] = {"rho": round(rho, 3), "n": len(cs)}
        print(f"  Spearman(our per-category drop, peterhdd's REAP-BF16 drop at 30 q/category) over {len(cs)}: {rho:+.2f}")
if args.out:
    args.out.write_text(json.dumps(report, indent=1))
