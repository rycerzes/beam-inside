# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "torch", "tokenizers"]
# ///
"""Weight-only router analysis for Beam: load-balancing bias, router geometry, and a router lens on token embeddings.

    uv run scripts/atlas_router.py            # reads data/bf16, writes data/atlas/router.json and prints a summary

Beam's router (transformers BeamTopKRouter) scores experts with sigmoid(W_r · x), where x is the residual after
pre_feedforward_layernorm. The correction bias is added only when choosing experts, and is learned to balance load,
so a large bias marks an expert training had to push up (under-used) and a negative one an expert it held back.
The read direction of expert e is therefore w_pre ⊙ W_r[e] applied to the RMS-normalised residual.
"""

import json
import pathlib

import numpy as np
import torch
from tokenizers import Tokenizer

D = pathlib.Path("data/bf16")
OUT = pathlib.Path("data/atlas")
OUT.mkdir(parents=True, exist_ok=True)
SPARSE = range(3, 52)
GROUPS, PER_GROUP = 8, 32


def load(name):
    a = np.load(D / f"{name}.npy")
    if a.dtype == np.uint16:  # BF16 stored raw
        a = (a.astype(np.uint32) << 16).view(np.float32)
    return torch.from_numpy(a)


tok = Tokenizer.from_file("meta/tokenizer.json")
embed = load("model.embed_tokens.weight")
embed_n = embed / embed.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()  # what RMSNorm sees, before its weight

summary = {"layers": {}}
for L in SPARSE:
    W = load(f"model.layers.{L}.mlp.gate.weight")  # [256, 5120]
    b = load(f"model.layers.{L}.mlp.gate.e_score_correction_bias")
    w_pre = load(f"model.layers.{L}.pre_feedforward_layernorm.weight")
    V = W * w_pre  # read directions on the normalised residual
    Vn = V / V.norm(dim=-1, keepdim=True)
    cos = Vn @ Vn.T
    off = ~torch.eye(256, dtype=torch.bool)
    group = torch.arange(256) // PER_GROUP
    same = (group[:, None] == group[None, :]) & off
    sv = torch.linalg.svdvals(V)
    eff_rank = float((sv.sum() ** 2) / (sv ** 2).sum())
    # Router lens: which tokens' embeddings, fed straight to this router, make each expert beat the others. Selection
    # ranks sigmoid(v_e . x) + b_e and b is nearly uniform, so only v_e minus the mean row matters for who is picked.
    scores = embed_n @ (V - V.mean(0)).T  # [vocab, 256]
    top = scores.topk(12, dim=0).indices.T  # [256, 12]
    summary["layers"][L] = {
        "bias_dev": (b.double() - b.double().median()).tolist(),
        "bias": {"mean": float(b.mean()), "std": float(b.std()), "min": float(b.min()), "max": float(b.max()),
                 "argmin": int(b.argmin()), "argmax": int(b.argmax()), "values": b.tolist()},
        "cos_within_group": float(cos[same].mean()), "cos_between_group": float(cos[off & ~same].mean()),
        "cos_max": float(cos[off].max()), "router_norm_mean": float(V.norm(dim=-1).mean()),
        "eff_rank": eff_rank,
        "lens_top_tokens": [[tok.id_to_token(int(t)) for t in row] for row in top],
    }
    s = summary["layers"][L]
    print(f"L{L:2d} bias mean {s['bias']['mean']:+.3f} std {s['bias']['std']:.3f} "
          f"[{s['bias']['min']:+.3f} e{s['bias']['argmin']}, {s['bias']['max']:+.3f} e{s['bias']['argmax']}] "
          f"cos in/out group {s['cos_within_group']:+.3f}/{s['cos_between_group']:+.3f} max {s['cos_max']:.2f} "
          f"eff_rank {eff_rank:.0f}")

json.dump(summary, open(OUT / "router.json", "w"))
for L in (3, 10, 27, 40, 51):
    dev = summary["layers"][L]["bias_dev"]
    held = sorted(range(256), key=lambda e: dev[e])[:4]
    print(f"\n## layer {L}: router lens, first 8 experts, then the 4 most held-back experts")
    for e in list(range(8)) + held:
        print(f"  e{e:3d} bias_dev {dev[e]:+.5f}  {' '.join(summary['layers'][L]['lens_top_tokens'][e])}")
