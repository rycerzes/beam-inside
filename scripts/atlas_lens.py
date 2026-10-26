# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "torch", "tokenizers"]
# ///
"""Label each expert by the tokens its dominant output directions promote (logit lens on weights).

    uv run scripts/atlas_lens.py 46 51           # reads data/experts/L{n}.npz, writes data/atlas/experts_L{n}.jsonl

An expert's write reaches the logits through post_feedforward_layernorm (weight w_post), the depth-scaled residual
(a positive scale), the final norm (w_final) and lm_head, so direction u gives logits lm_head · (w_final ⊙ w_post ⊙ u).
Two corrections keep a few tokens from topping every list: lm_head rows are centred (the mean row adds the same to
every token), and each token's logit is standardised across all directions in the layer, so a token ranks high only
where this expert promotes it more than other experts do. Singular vectors have no sign, so both poles are kept.
"""

import json
import pathlib
import sys

import numpy as np
import torch
from tokenizers import Tokenizer

D, E, OUT = pathlib.Path("data/bf16"), pathlib.Path("data/experts"), pathlib.Path("data/atlas")
TOP, DIRS = 12, 3
dev = "mps" if torch.backends.mps.is_available() else "cpu"


def load(name):
    a = np.load(D / f"{name}.npy")
    if a.dtype == np.uint16:
        a = (a.astype(np.uint32) << 16).view(np.float32)
    return torch.from_numpy(a)


tok = Tokenizer.from_file("meta/tokenizer.json")
unembed = load("lm_head.weight")
unembed = (unembed - unembed.mean(0)).to(dev, torch.float16)
w_final = load("model.norm.weight")

for L in map(int, sys.argv[1:]):
    z = np.load(E / f"L{L}.npz")
    U = torch.from_numpy(z["U"][:, :DIRS])  # [256, DIRS, 5120], scaled by singular values
    dirs = U * (w_final * load(f"model.layers.{L}.post_feedforward_layernorm.weight"))
    dirs = torch.cat([dirs, -dirs], 1).reshape(-1, 5120)  # both poles: [256 * 2 * DIRS, 5120]
    logits = (dirs.to(dev, torch.float16) @ unembed.T).float()  # [n_dirs, vocab]
    logits = (logits - logits.mean(0)) / logits.std(0).clamp_min(1e-3)  # per-token standardisation across directions
    top = logits.topk(TOP, dim=-1)
    idx, val = top.indices.view(256, 2 * DIRS, TOP).cpu(), top.values.view(256, 2 * DIRS, TOP).cpu()
    rows = []
    for e in range(256):
        poles = [{"dir": d % DIRS, "sign": "+" if d < DIRS else "-", "z_mean": round(float(val[e, d].mean()), 2),
                  "tokens": [tok.id_to_token(int(t)) for t in idx[e, d]]} for d in range(2 * DIRS)]
        poles.sort(key=lambda p: -p["z_mean"])
        rows.append({"layer": L, "expert": e, "fro": float(z["fro"][e]), "poles": poles})
    with open(OUT / f"experts_L{L}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n## layer {L}: strongest pole of the first 24 experts (token z-score across the layer's directions)")
    for r in rows[:24]:
        p = r["poles"][0]
        print(f"  e{r['expert']:3d} d{p['dir']}{p['sign']} z{p['z_mean']:5.1f}  {' '.join(p['tokens'])}")
