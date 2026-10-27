# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Pick the units for the steering sweep and turn each into a unit-norm direction in the residual stream.

    uv run scripts/sweep_units.py [--per-layer 24] [--dense-per-layer 16] [--controls-per-layer 8]
    -> data/sweep/units.npz (dirs float16 [N, 5120], layer int16 [N]) and data/sweep/units.jsonl (one row per unit)

Units, in the order the session runs them (layers interleaved, so a sweep cut short still covers every depth):
  expert_dir   pole (sign * k-th left singular vector of an expert's W_down), the strongest lens pole per expert,
               top `per-layer` experts by lens z per MoE layer
  dense_neuron pole of a dense-layer neuron (sign * W_down[:, n]), top `dense-per-layer` by lens z per dense layer
  control      Gaussian random direction, `controls-per-layer` per layer (Lyra's random-direction controls)

A unit's push is added to the residual leaving its layer, after the MLP's post-norm, so its direction there is the
post-norm gain times the write direction. Normalisation of the MLP output is ignored, which is exact for the
direction of a single small push and approximate for large ones.
"""

import argparse
import json
import pathlib

import numpy as np

ATLAS, EXPERTS, BF16 = pathlib.Path("data/atlas"), pathlib.Path("data/experts"), pathlib.Path("data/bf16")
OUT = pathlib.Path("data/sweep")
DENSE, MOE = range(0, 3), range(3, 52)

parser = argparse.ArgumentParser()
parser.add_argument("--per-layer", type=int, default=24)
parser.add_argument("--dense-per-layer", type=int, default=16)
parser.add_argument("--controls-per-layer", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
rng = np.random.default_rng(args.seed)


def bf16(path):
    bits = np.load(path)
    return (bits.astype(np.uint32) << 16).view(np.float32) if bits.dtype == np.uint16 else bits.astype(np.float32)


def gain(layer):
    return bf16(BF16 / f"model.layers.{layer}.post_feedforward_layernorm.weight.npy")


def unit(v):
    return v / np.linalg.norm(v)


per_layer = {}  # layer -> list of (meta, direction)
for L in MOE:
    g = gain(L)
    U = np.load(EXPERTS / f"L{L}.npz")["U"]  # [256, K, 5120], scaled by singular values
    best = []
    for row in map(json.loads, open(ATLAS / f"experts_L{L}.jsonl")):
        pole = max(row["poles"], key=lambda p: p["z_mean"])
        best.append((pole["z_mean"], row["expert"], pole))
    best.sort(reverse=True)
    per_layer[L] = []
    for z, e, pole in best[: args.per_layer]:
        sign = 1.0 if pole["sign"] == "+" else -1.0
        d = unit(g * sign * U[e, pole["dir"]])
        per_layer[L].append(({"kind": "expert_dir", "layer": L, "expert": e, "dir": pole["dir"], "sign": pole["sign"],
                              "z": round(z, 2), "lens": pole["tokens"]}, d))

for L in DENSE:
    g = gain(L)
    down = bf16(BF16 / f"model.layers.{L}.mlp.down_proj.weight.npy")  # [5120, 20480]
    rows = [json.loads(line) for line in open(ATLAS / f"dense_L{L}.jsonl")]
    poles = sorted(((r["up_z"], r["neuron"], "+", r["up"]) for r in rows), reverse=True)[: args.dense_per_layer]
    per_layer[L] = []
    for z, n, sign, tokens in poles:
        d = unit(g * down[:, n])
        per_layer[L].append(({"kind": "dense_neuron", "layer": L, "neuron": n, "sign": sign, "z": round(z, 2),
                              "lens": tokens}, d))

for L in [*DENSE, *MOE]:
    for i in range(args.controls_per_layer):
        per_layer[L].append(({"kind": "control", "layer": L, "control": i}, unit(rng.standard_normal(5120))))

# Interleave: each round takes one unit from every layer, controls spread through the rounds.
for L in per_layer:
    order = rng.permutation(len(per_layer[L]))
    per_layer[L] = [per_layer[L][i] for i in order]
metas, dirs = [], []
for r in range(max(map(len, per_layer.values()))):
    for L in sorted(per_layer):
        if r < len(per_layer[L]):
            meta, d = per_layer[L][r]
            metas.append({"unit": len(metas), **meta})
            dirs.append(d)

OUT.mkdir(parents=True, exist_ok=True)
np.savez(OUT / "units.npz", dirs=np.stack(dirs).astype(np.float16),
         layer=np.array([m["layer"] for m in metas], np.int16))
with open(OUT / "units.jsonl", "w") as f:
    for m in metas:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")
kinds = {k: sum(m["kind"] == k for m in metas) for k in ("expert_dir", "dense_neuron", "control")}
print(f"{len(metas)} units {kinds} -> {OUT}/units.npz ({(OUT / 'units.npz').stat().st_size / 1e6:.1f} MB)")
