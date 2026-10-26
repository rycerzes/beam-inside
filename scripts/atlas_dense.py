# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "torch", "tokenizers"]
# ///
"""Weight-only labels for the 61,440 neurons of Beam's dense MLP layers 0-2.

    uv run scripts/atlas_dense.py             # reads data/bf16, writes data/atlas/dense_L{0,1,2}.jsonl

Output side (what a neuron writes): Beam normalises every sublayer output (post_feedforward_layernorm) and the final
residual is, up to one positive scale, embedding + the sum of all 104 normalised sublayer outputs (depth-scaled
residual adds). A neuron's direct push on the logits is therefore lm_head · (w_final ⊙ w_post ⊙ W_down[:, n]),
ignoring the positive RMS scales. Both poles are kept: positive activation promotes `up`, negative promotes `down`.
SiLU-gated neurons can fire negatively, weakly.

Input side (what turns a neuron on): gate_proj[n] · (w_pre ⊙ normalised embedding), the response to a token's own
embedding. Only meaningful in layer 0, where the residual is still mostly the embedding.
"""

import json
import pathlib

import numpy as np
import torch
from tokenizers import Tokenizer

D = pathlib.Path("data/bf16")
OUT = pathlib.Path("data/atlas")
OUT.mkdir(parents=True, exist_ok=True)
K = 10
dev = "mps" if torch.backends.mps.is_available() else "cpu"


def load(name):
    a = np.load(D / f"{name}.npy")
    if a.dtype == np.uint16:
        a = (a.astype(np.uint32) << 16).view(np.float32)
    return torch.from_numpy(a)


tok = Tokenizer.from_file("meta/tokenizer.json")
decode = lambda ids: [tok.id_to_token(int(i)) for i in ids]
w_final = load("model.norm.weight")
unembed = load("lm_head.weight").to(dev, torch.float16)  # [vocab, 5120]
embed = load("model.embed_tokens.weight")
embed_n = (embed / embed.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()).to(dev, torch.float16)

for L in range(3):
    w_post = load(f"model.layers.{L}.post_feedforward_layernorm.weight")
    w_pre = load(f"model.layers.{L}.pre_feedforward_layernorm.weight")
    down = load(f"model.layers.{L}.mlp.down_proj.weight")  # [5120, 20480]
    gate = load(f"model.layers.{L}.mlp.gate_proj.weight")  # [20480, 5120]
    write = (down * (w_final * w_post)[:, None]).T.contiguous()  # [20480, 5120] per-neuron logit direction
    read = (gate * w_pre).to(dev, torch.float16)
    rows = []
    for s in range(0, write.shape[0], 512):
        logits = write[s:s + 512].to(dev, torch.float16) @ unembed.T  # [512, vocab]
        up, dn = logits.topk(K, dim=-1), (-logits).topk(K, dim=-1)
        z = (logits - logits.mean(-1, keepdim=True)) / logits.std(-1, keepdim=True)
        acts = read[s:s + 512] @ embed_n.T if L == 0 else None
        trig = acts.topk(K, dim=-1).indices.cpu() if acts is not None else None
        for i in range(logits.shape[0]):
            rows.append({"layer": L, "neuron": s + i,
                         "up": decode(up.indices[i].cpu()), "up_z": round(float(z[i, up.indices[i]].max()), 1),
                         "down": decode(dn.indices[i].cpu()), "down_z": round(float(-z[i, dn.indices[i]].min()), 1),
                         **({"trigger": decode(trig[i])} if trig is not None else {})})
    with open(OUT / f"dense_L{L}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    peaked = sorted(rows, key=lambda r: -max(r["up_z"], r["down_z"]))
    print(f"\n## layer {L}: 15 most peaked neurons (logit z-score of the strongest token)")
    for r in peaked[:15]:
        print(f"  n{r['neuron']:5d} z{r['up_z']:5.1f}/{r['down_z']:5.1f} up {' '.join(r['up'][:6])} | down {' '.join(r['down'][:4])}"
              + (f" | trig {' '.join(r['trigger'][:4])}" if 'trigger' in r else ""))
