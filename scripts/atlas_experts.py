# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "numpy", "torch", "huggingface_hub"]
# ///
"""Stream every routed expert's down_proj from the FP8 checkpoint and keep only its top output directions.

    WORKERS=32 uv run scripts/atlas_experts.py 46 51   # layers to process; writes data/experts/L{n}.npz, skips done layers

Per expert we keep the top-K left singular vectors of W_down (5120 x 2560), scaled by their singular values, and the
Frobenius norm. That is the subspace the expert can write into the residual stream, about 10 MB per layer instead of
3.4 GB of FP8 weights. FP8 weights are per-output-channel scaled (compressed-tensors, strategy "channel").
"""

import concurrent.futures as cf
import os
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ckpt import Checkpoint  # noqa: E402

K, EXPERTS = 8, 256
WORKERS = int(os.environ.get("WORKERS", 8))
OUT = pathlib.Path("data/experts")
OUT.mkdir(parents=True, exist_ok=True)
ck = Checkpoint("reflection/Beam-1.0-501B-A23B-FP8")


def fetch(L, e):
    name = f"model.layers.{L}.mlp.experts.{e}.down_proj.weight"
    _, shape, w = ck.raw(name)
    _, _, s = ck.raw(name + "_scale")
    w = torch.frombuffer(bytearray(w), dtype=torch.float8_e4m3fn).view(shape).float()
    s = torch.frombuffer(bytearray(s), dtype=torch.bfloat16).view(-1, 1).float()
    return e, w * s


for L in map(int, sys.argv[1:]):
    path = OUT / f"L{L}.npz"
    if path.exists():
        continue
    t0, nbytes = time.time(), 0
    U = np.zeros((EXPERTS, K, 5120), np.float32)
    S = np.zeros((EXPERTS, K), np.float32)
    fro = np.zeros(EXPERTS, np.float32)
    with cf.ThreadPoolExecutor(WORKERS) as pool:
        for e, W in pool.map(lambda e: fetch(L, e), range(EXPERTS)):
            nbytes += W.numel()
            u, s, _ = torch.svd_lowrank(W, q=K + 8, niter=4)
            U[e], S[e], fro[e] = (u[:, :K] * s[:K]).T.numpy(), s[:K].numpy(), float(W.norm())
    np.savez(path, U=U, S=S, fro=fro)
    dt = time.time() - t0
    print(f"L{L}: {nbytes / 1e9:.1f} GB in {dt:.0f}s ({nbytes / 1e6 / dt:.0f} MB/s)  "
          f"top1 energy share median {np.median(S[:, 0] ** 2 / fro ** 2):.3f}", flush=True)
