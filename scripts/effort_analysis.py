# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "torch"]
# ///
"""Offline analysis of a v2 session's effort captures (RUN/effort_capture_*.pt).

    uv run scripts/effort_analysis.py RUN_DIR   # -> RUN_DIR/effort_analysis.json

Each (question, effort N) request captured 5 positions, in prompt order: the last digit of N, the end of the system
message, the think start (<|message|> after <|channel|>analysis), and the first two thought tokens.

1. Readout: per position and layer, held-out R² of log2 N from one direction (within-question centred slope on the
   dial values 8..512, questions split 80/20), as in session.py.
2. Off the dial: where N = 2 and N = 1000 land on that scale. Each is centred by its question's dial mean and mapped
   through the train fit; reported as the N the direction reads (2^prediction), median over questions.
3. Effort experts: Beam's own selection (sigmoid scores + e_score_correction_bias, 8 groups of 32 ranked by their
   top-2 sum, top 4 groups, top 8 experts) recomputed from the captured router logits. Per position, layer and
   expert: selection rate at N = 8 and N = 512 and the risk difference; the null is the largest |risk difference|
   over all experts with N labels shuffled within question (20 shuffles), so an expert above its 95th percentile is
   unlikely to be chance even after looking at all 12,544.
"""

import json
import pathlib
import sys

import numpy as np
import torch

RUN = pathlib.Path(sys.argv[1])
BF16 = pathlib.Path("data/bf16")
KINDS = ("n_last", "sys_end", "think_start", "thought1", "thought2")
DIAL, LOW, HIGH = (8, 16, 32, 64, 128, 256, 512), 8, 512
config = json.loads(pathlib.Path("meta/config.json").read_text())
MOE = [i for i, t in enumerate(config["mlp_layer_types"]) if t == "sparse"]

# Rows by (kind, question, N).
caps = [torch.load(p, weights_only=True, mmap=True) for p in sorted(RUN.glob("effort_capture_*.pt"))]  # 2 GB each
rows = {}  # (question, N) -> [(file, row, position)]
for f, cap in enumerate(caps):
    for r, (key, pos, _, _) in enumerate(cap["index"]):
        rows.setdefault(tuple(key), []).append((f, r, pos))
keys = sorted(k for k, v in rows.items() if len(v) == len(KINDS))
print(f"{len(caps)} capture files, {len(keys)} complete (question, N) requests, "
      f"{len(rows) - len(keys)} incomplete; dropped {sum(c['dropped'] for c in caps)}")


def gather(kind, what):
    """[layers, len(keys), d] float32 for one position kind; what = resid | router."""
    k = KINDS.index(kind)
    out = []
    for key in keys:
        f, r, _ = sorted(rows[key], key=lambda x: x[2])[k]
        out.append(caps[f][what][:, r].float())
    return torch.stack(out, 1).numpy()


q = np.array([k[0] for k in keys])
n = np.array([k[1] for k in keys])
on = np.isin(n, DIAL)
questions = np.unique(q)
train_q = questions[: int(0.8 * len(questions))]
result = {"readout": {}, "off_dial": {}, "experts": {}}

for kind in KINDS:
    Xc = gather(kind, "resid")  # [L, S, H], centred in place below
    y = np.log2(n.astype(float))
    yc = y.copy()
    for i in questions:  # centre every row by its question's dial mean
        m, d = q == i, (q == i) & on
        Xc[:, m] -= Xc[:, d].mean(1, keepdims=True)
        yc[m] -= y[d].mean()
    tr, te = on & np.isin(q, train_q), on & ~np.isin(q, train_q)
    r2, reads = [], {}
    for L in range(Xc.shape[0]):
        r = Xc[L, tr].T @ yc[tr] / (yc[tr] ** 2).sum()
        u = r / (np.linalg.norm(r) + 1e-12)
        p = Xc[L] @ u
        b = np.polyfit(p[tr], yc[tr], 1)
        pred = np.polyval(b, p)
        r2.append(float(1 - ((yc[te] - pred[te]) ** 2).sum() / ((yc[te] - yc[te].mean()) ** 2).sum()))
        reads[L] = {int(v): float(2 ** np.median(pred[n == v] + np.log2(DIAL).mean())) for v in (2, 8, 512, 1000)}
    best = int(np.nanargmax(r2))
    result["readout"][kind] = {"r2": r2, "best_layer": best}
    result["off_dial"][kind] = {L: reads[L] for L in (best, 26, 36, Xc.shape[0] - 1)}
    print(f"{kind:12s} best R² {r2[best]:.3f} at L{best}; R² at L10/20/30/40/51: "
          f"{[round(r2[L], 2) for L in (10, 20, 30, 40, 51)]}; reads N at L{best}: "
          + ", ".join(f"{v}->{reads[best][v]:.0f}" for v in (2, 8, 512, 1000)))
    del Xc

rng = np.random.default_rng(0)
G = config["n_group"]
for kind in ("n_last", "think_start", "thought1"):
    logits = gather(kind, "router")  # [L, S, E]
    sel = np.zeros(logits.shape, dtype=bool)
    for L in MOE:
        bias = np.load(BF16 / f"model.layers.{L}.mlp.gate.e_score_correction_bias.npy")
        s = 1 / (1 + np.exp(-logits[L].astype(np.float64))) + bias
        grouped = s.reshape(len(keys), G, -1)
        gscore = np.sort(grouped, -1)[..., -2:].sum(-1)
        keep = np.argsort(-gscore, -1)[:, : config["topk_group"]]
        mask = np.zeros_like(gscore, dtype=bool)
        np.put_along_axis(mask, keep, True, -1)
        s = np.where(np.repeat(mask, s.shape[1] // G, -1), s, -np.inf)
        top = np.argsort(-s, -1)[:, : config["num_experts_per_tok"]]
        np.put_along_axis(sel[L], top, True, -1)

    def risk_diff(labels):
        lo, hi = labels == LOW, labels == HIGH
        return sel[:, hi].mean(1) - sel[:, lo].mean(1)  # [L, E]

    rd = risk_diff(n)
    null = []
    for _ in range(20):
        perm = n.copy()
        for i in questions:
            m = np.flatnonzero(q == i)
            perm[m] = rng.permutation(n[m])
        null.append(np.abs(risk_diff(perm)[MOE]).max())
    thr = float(np.percentile(null, 95))
    top = sorted(((abs(rd[L, e]), L, e) for L in MOE for e in range(rd.shape[1])), reverse=True)
    hits = [(L, e) for v, L, e in top if v > thr]
    result["experts"][kind] = {"threshold": thr, "n_significant": len(hits), "top": [
        {"layer": L, "expert": e, "rd": round(float(rd[L, e]), 3), "rate_n8": round(float(sel[L, n == LOW, e].mean()), 3),
         "rate_n512": round(float(sel[L, n == HIGH, e].mean()), 3)} for _, L, e in top[:40]]}
    print(f"{kind:12s} effort experts: {len(hits)} with |risk diff 8 vs 512| > {thr:.3f} (shuffled-N max, p95); top: "
          + "; ".join(f"L{t['layer']} e{t['expert']} {t['rate_n8']:.2f}->{t['rate_n512']:.2f}"
                      for t in result["experts"][kind]["top"][:8]))
    del logits, sel
(RUN / "effort_analysis.json").write_text(json.dumps(result))
