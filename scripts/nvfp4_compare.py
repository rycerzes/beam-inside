# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "numpy", "torch", "huggingface_hub"]
# ///
"""Compare routed-expert down_proj weights between the FP8 and NVFP4 checkpoints, for a sample of experts.

    uv run scripts/nvfp4_compare.py [layers...]   # default 10 27 46, experts 0-15 of each

Per expert: elementwise relative error and correlation of the dequantised NVFP4 weight against FP8 (a wrong nibble
order or scale layout shows up here as correlation far below 1), and the |cosine| between matching top-k left
singular vectors, i.e. the write directions the atlas labels and the sweep steers along. Both sides are FP8- or
FP4-rounded versions of the same BF16 weight, so neither is ground truth; FP8 is much closer to it.

NVFP4 (ModelOpt): weight uint8 [out, in/2], two E2M1 values per byte, low nibble first; weight_scale F8_E4M3
[out, in/16], one scale per group of 16 inputs; weight_scale_2 F32 scalar. w = e2m1 * weight_scale * weight_scale_2.
"""

import concurrent.futures as cf
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ckpt import Checkpoint  # noqa: E402

E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
K, EXPERTS = 8, range(16)
fp8, fp4 = Checkpoint("reflection/Beam-1.0-501B-A23B-FP8"), Checkpoint("reflection/Beam-1.0-501B-A23B-NVFP4")


def tensor(ck, name, dtype):
    _, shape, data = ck.raw(name)
    return torch.frombuffer(bytearray(data), dtype=dtype).view(shape)


def load_fp8(name):
    return tensor(fp8, name, torch.float8_e4m3fn).float() * tensor(fp8, name + "_scale", torch.bfloat16).float().view(-1, 1)


def load_fp4(name):
    packed = tensor(fp4, name, torch.uint8)
    vals = torch.stack([E2M1[(packed & 0xF).long()], E2M1[(packed >> 4).long()]], -1).view(packed.shape[0], -1)
    scale = tensor(fp4, name + "_scale", torch.float8_e4m3fn).float()
    scale2 = tensor(fp4, name + "_scale_2", torch.float32).float().reshape(())
    return vals * scale.repeat_interleave(16, dim=1) * scale2


def compare(L, e):
    name = f"model.layers.{L}.mlp.experts.{e}.down_proj.weight"
    a, b = load_fp8(name), load_fp4(name)
    rel = float((a - b).norm() / a.norm())
    corr = float(torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1])
    ua, sa, _ = torch.linalg.svd(a, full_matrices=False)
    ub, _, _ = torch.linalg.svd(b, full_matrices=False)
    cos = (ua[:, :K] * ub[:, :K]).sum(0).abs().numpy()
    # Subspace overlap of the top-K write directions (1 = same subspace), robust to near-degenerate singular values.
    overlap = float(torch.linalg.matrix_norm(ua[:, :K].T @ ub[:, :K]) ** 2 / K)
    energy1 = float(sa[0] ** 2 / (sa ** 2).sum())
    return L, e, rel, corr, cos, overlap, energy1


layers = list(map(int, sys.argv[1:])) or [10, 27, 46]
name = f"model.layers.{layers[0]}.mlp.experts.0.down_proj.weight"
print("dtypes:", fp4.info(name)[:2], fp4.info(name + "_scale")[:2], fp4.info(name + "_scale_2")[:2])
rows = []
with cf.ThreadPoolExecutor(8) as pool:
    for r in pool.map(lambda le: compare(*le), [(L, e) for L in layers for e in EXPERTS]):
        rows.append(r)
        L, e, rel, corr, cos, overlap, energy1 = r
        print(f"L{L:2d} e{e:3d}  rel err {rel:.3f}  corr {corr:.4f}  |cos| top1..4 {np.round(cos[:4], 3)}  "
              f"top{K} subspace overlap {overlap:.3f}  top1 energy {energy1:.3f}", flush=True)
rel = np.array([r[2] for r in rows]); cos = np.stack([r[4] for r in rows]); ov = np.array([r[5] for r in rows])
print(f"\n{len(rows)} experts: rel err median {np.median(rel):.3f}; |cos| median by rank {np.round(np.median(cos, 0), 3)}; "
      f"share of top-1 directions with |cos| > 0.9: {np.mean(cos[:, 0] > 0.9):.0%}; subspace overlap median {np.median(ov):.3f}")
