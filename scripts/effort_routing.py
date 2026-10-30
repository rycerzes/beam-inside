# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "torch", "pyarrow"]
# ///
"""How often Beam selects each expert at each reasoning-effort setting, from the effort captures, for the explorer.

    uv run scripts/effort_routing.py RUN_DIR OUT.json [--atlas data/export/atlas]

RUN_DIR holds effort_capture_*.pt (200 GSM8K questions x efforts 2-1000, router logits at five positions). Selection
is recomputed the way Beam routes (as in effort_analysis.py): sigmoid(logit) + e_score_correction_bias, the top 4 of
8 groups of 32 by their top-2 sum, then the top 8 experts. For the effort digit (n_last) and the start of thinking
(think_start), OUT holds per effort the share of questions that selected each expert, as integers in thousandths
[49 MoE layers][256], plus per effort the GSM8K accuracy and median thinking tokens (from the atlas's
effort/gsm8k_dial) and the competition-math dial rows where they exist.
"""

import argparse
import json
import pathlib

import numpy as np
import pyarrow.parquet as pq
import torch

parser = argparse.ArgumentParser()
parser.add_argument("run", type=pathlib.Path)
parser.add_argument("out", type=pathlib.Path)
parser.add_argument("--atlas", type=pathlib.Path, default=pathlib.Path("data/export/atlas"))
args = parser.parse_args()
BF16 = pathlib.Path("data/bf16")
KINDS = ("n_last", "sys_end", "think_start", "thought1", "thought2")
config = json.loads(pathlib.Path("meta/config.json").read_text())
MOE = [i for i, t in enumerate(config["mlp_layer_types"]) if t == "sparse"]
G, TOPG, K = config["n_group"], config["topk_group"], config["num_experts_per_tok"]

caps = [torch.load(p, weights_only=True, mmap=True) for p in sorted(args.run.glob("effort_capture_*.pt"))]
rows = {}
for f, cap in enumerate(caps):
    for r, (key, pos, _, _) in enumerate(cap["index"]):
        rows.setdefault(tuple(key), []).append((f, r, pos))
keys = sorted(k for k, v in rows.items() if len(v) == len(KINDS))
efforts = sorted({k[1] for k in keys})
n = np.array([k[1] for k in keys])


def select(kind):
    """[MoE layers, len(keys), 256] bool: Beam's top-8 choice at this position."""
    k = KINDS.index(kind)
    logits = []
    for key in keys:
        f, r, _ = sorted(rows[key], key=lambda x: x[2])[k]
        logits.append(caps[f]["router"][:, r].float())
    logits = torch.stack(logits, 1).numpy()  # [52, S, E]
    out = np.zeros((len(MOE), len(keys), logits.shape[-1]), dtype=bool)
    for i, L in enumerate(MOE):
        bias = np.load(BF16 / f"model.layers.{L}.mlp.gate.e_score_correction_bias.npy")
        s = 1 / (1 + np.exp(-logits[L].astype(np.float64))) + bias
        gscore = np.sort(s.reshape(len(keys), G, -1), -1)[..., -2:].sum(-1)
        keep = np.argsort(-gscore, -1)[:, :TOPG]
        mask = np.zeros_like(gscore, dtype=bool)
        np.put_along_axis(mask, keep, True, -1)
        s = np.where(np.repeat(mask, s.shape[1] // G, -1), s, -np.inf)
        np.put_along_axis(out[i], np.argsort(-s, -1)[:, :K], True, -1)
    return out


result = {"layers": MOE, "efforts": efforts, "questions": len(keys) // len(efforts), "rates": {}}
for kind in ("n_last", "think_start"):
    sel = select(kind)
    result["rates"][kind] = {str(e): np.rint(1000 * sel[:, n == e].mean(1)).astype(int).tolist() for e in efforts}
    lo, hi = sel[:, n == 8].mean(1), sel[:, n == 512].mean(1)
    print(f"{kind}: experts whose selection rate moves by >= 0.5 between effort 8 and 512: {(abs(hi - lo) >= 0.5).sum()}")

dial = pq.read_table(args.atlas / "effort/gsm8k_dial.parquet").to_pylist()
gsm = {}
for e in efforts:
    rs = [r for r in dial if r["effort"] == e]
    ok = [str(r["final"]).strip().replace(",", "").rstrip(".") == str(r["gold"]).strip() for r in rs]
    gsm[str(e)] = {"n": len(rs), "acc": round(float(np.mean(ok)), 3) if rs else None,
                   "median_think": float(np.median([r["think_tokens"] for r in rs])) if rs else None}
result["gsm8k"] = gsm
math = pq.read_table(args.atlas / "effort/competition_math.parquet").to_pylist()
result["competition_math"] = {}
for e in (8, 32, 128, 512):
    rs = [r for r in math if r["arm"] == f"dial{e}"]
    result["competition_math"][str(e)] = {"n": len(rs), "acc": round(float(np.mean([bool(r["correct"]) for r in rs])), 3),
                                          "median_think": float(np.median([r["think_tokens"] for r in rs]))}
args.out.write_text(json.dumps(result, separators=(",", ":")))
print(f"{len(keys)} requests, efforts {efforts}; gsm8k {gsm}; -> {args.out} ({args.out.stat().st_size / 1e3:.0f} kB)")
