# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Build three-way label tests (after arXiv 2501.08319) from a ranked v2 sweep.

    uv run scripts/judge_items.py RUN_DIR OUT_DIR [--sig 100] [--random 100] [--seed 0]
    -> OUT_DIR/items.jsonl (what a judge sees) and OUT_DIR/key.jsonl (answers; never shown to a judge)

An item is one expert pole. The judge gets its vocabulary label (top lens tokens, from the weights alone) and three
options, each the steered answers of one pole to the three sweep prompts (who / snake / orange, strongest dose of
the pole). One option is the labelled pole; the other two are expert poles from the same 8-layer band whose ΔNLL is
within 1.5x (effect-matched distractors; widened to 2x, then 3x, when fewer than two qualify, and the item is
skipped if none of these finds two) and whose labels share no 4-letter stem with the target's. Chance is 1/3.
Strata: `sig` poles significant against the effect-matched null (ranked_matched.jsonl, pct_matched >= 0.95) and
`random` testable expert poles.
"""

import argparse
import gzip
import json
import math
import pathlib
import re

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("run", type=pathlib.Path)
parser.add_argument("out", type=pathlib.Path)
parser.add_argument("--sig", type=int, default=100)
parser.add_argument("--random", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
rng = np.random.default_rng(args.seed)
PROMPTS = ("who", "snake", "orange")


def _byte_decoder():
    """GPT-2 byte-level BPE: printable stand-in characters -> raw bytes."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


DECODE = _byte_decoder()


def readable(token):
    try:
        return bytes(DECODE[c] for c in token).decode("utf-8", errors="replace")
    except KeyError:
        return token


def label(lens, k=12):
    return ", ".join(repr(readable(t).strip() or readable(t)) for t in lens[:k])


def stems(lens):
    return {w[:4] for t in lens for w in re.findall(r"[a-z]{4,}", readable(t).lower())}


def clean(text, limit=420):
    text = re.sub(r"<\|channel\|>(analysis|final)<\|message\|>", lambda m: f"[{m.group(1)}] ", text or "")
    text = re.sub(r"<\|[^|]*\|>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + ("…" if len(text) > limit else "")


ranked = [json.loads(line) for line in open(args.run / "ranked_matched.jsonl")]
poles = [r for r in ranked if r["kind"] == "expert_dir" and r.get("null") != "untestable"]
texts = {}  # (unit, pole) -> {prompt: text}, strongest dose of the pole
for path in sorted(args.run.glob("sweep_*.jsonl.gz")):
    for r in map(json.loads, gzip.open(path, "rt")):
        if r["prompt"] in PROMPTS and abs(r["dose"]) == 1.0:
            texts.setdefault((r["unit"], "+" if r["dose"] > 0 else "-"), {})[r["prompt"]] = r.get("text")
poles = [p for p in poles if len(texts.get((p["unit"], p["pole"]), {})) == len(PROMPTS)]

sig = [p for p in poles if (p["pct_matched"] or 0) >= 0.95]
targets = [("sig", p) for p in rng.choice(sig, size=min(args.sig, len(sig)), replace=False)]
targets += [("random", p) for p in rng.choice(poles, size=args.random, replace=False)]

args.out.mkdir(parents=True, exist_ok=True)
skipped, i = 0, -1
with open(args.out / "items.jsonl", "w") as fi, open(args.out / "key.jsonl", "w") as fk:
    for stratum, t in targets:
        own = stems(t["lens"])
        for ratio in (1.5, 2.0, 3.0):
            pool = [p for p in poles if p["layer"] // 8 == t["layer"] // 8 and p["unit"] != t["unit"]
                    and abs(math.log(max(p["dnll"], 1e-4) / max(t["dnll"], 1e-4))) <= math.log(ratio)
                    and not (stems(p["lens"]) & own)]
            if len(pool) >= 2:
                break
        if len(pool) < 2:
            skipped += 1
            continue
        i += 1
        distractors = list(rng.choice(pool, size=2, replace=False))
        options = [t, *distractors]
        order = rng.permutation(3)
        letters = "ABC"
        item = {"id": i, "label": label(t["lens"]), "options": {}}
        for slot, j in enumerate(order):
            o = options[j]
            item["options"][letters[slot]] = {k: clean(texts[(o["unit"], o["pole"])][k]) for k in PROMPTS}
        answer = letters[list(order).index(0)]
        fi.write(json.dumps(item, ensure_ascii=False) + "\n")
        fk.write(json.dumps({"id": i, "answer": answer, "stratum": stratum, "layer": t["layer"], "expert": t.get("expert"),
                             "pole": t["pole"], "dnll": t["dnll"], "pct_matched": t["pct_matched"], "match_ratio": ratio,
                             "distractors": [(d["layer"], d.get("expert"), d["pole"]) for d in distractors]}) + "\n")
print(f"{i + 1} items from {len(targets)} targets ({sum(s == 'sig' for s, _ in targets)} sig, "
      f"{sum(s == 'random' for s, _ in targets)} random), {skipped} skipped for lack of matched distractors"
      f" -> {args.out}/items.jsonl, answers in key.jsonl")
