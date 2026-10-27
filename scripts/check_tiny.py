# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "huggingface_hub", "numpy", "torch", "transformers==5.21.0", "safetensors"]
# ///
"""Check StreamedBeam against transformers' BeamForCausalLM on a tiny random Beam, through an FP8 checkpoint on disk.

    uv run scripts/check_tiny.py [workdir]

The tiny model has every structural feature of the real one (dense then MoE layers, sliding and NoPE full attention,
group-limited routing, nonzero gate and correction biases, softcap). Its projection weights are first rounded through
FP8 with BF16 per-channel scales and its other tensors through BF16, then written in the real checkpoint's naming
(per-expert tensors, `_scale` siblings, two shards). The streamed path therefore reads what it would read from the
Hub, and in float32 must reproduce the reference logits up to float rounding.
"""

import json
import pathlib
import re
import sys
import tempfile

import torch
from safetensors.torch import save_file
from transformers import BeamConfig, BeamForCausalLM

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ckpt import LocalCheckpoint  # noqa: E402
from streamed import StreamedBeam  # noqa: E402

torch.manual_seed(0)
work = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp())
work.mkdir(parents=True, exist_ok=True)
config = BeamConfig(vocab_size=1000, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                    num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                    num_experts=16, num_experts_per_tok=4, n_group=4, topk_group=2, sliding_window=8,
                    layer_types=["sliding_attention"] * 5 + ["full_attention", "sliding_attention", "full_attention"],
                    mlp_layer_types=["dense"] * 3 + ["sparse"] * 5,
                    rope_parameters={"sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                                     "full_attention": None},
                    routed_scaling_factor=2.5, final_logit_softcapping=30.0, tie_word_embeddings=False)
config._attn_implementation = "sdpa"
model = BeamForCausalLM(config).float().eval()
FP8_LINEAR = re.compile(r"model\.layers\.\d+\.(self_attn\.(q|k|v|o|gate)_proj|mlp\.(gate|up|down)_proj)\.weight")


def fp8(w):
    """Per-output-channel FP8 rounding as in the real checkpoint: returns (fp8 weight, bf16 scale, dequantised)."""
    s = (w.abs().amax(-1, keepdim=True) / 448).clamp_min(1e-12).to(torch.bfloat16)
    q = (w / s.float()).to(torch.float8_e4m3fn)
    return q, s, q.float() * s.float()


tensors = {}
with torch.no_grad():
    for name, p in list(model.state_dict().items()):
        if name.endswith(("norm.weight", "gate_bias")):  # make norms and gate biases non-trivial
            p.copy_(1 + 0.2 * torch.randn_like(p) if name.endswith("norm.weight") else torch.randn_like(p))
        if name.endswith("e_score_correction_bias"):  # nearly constant, like the real ones
            p.copy_(20 + 0.05 * torch.randn_like(p))
        if name.endswith("experts.gate_up_proj"):
            base, inter = name[: -len("gate_up_proj")], config.moe_intermediate_size
            for e in range(config.num_experts):
                for part, rows in (("gate", slice(0, inter)), ("up", slice(inter, 2 * inter))):
                    q, s, d = fp8(p[e, rows])
                    p[e, rows] = d
                    tensors[f"{base}{e}.{part}_proj.weight"], tensors[f"{base}{e}.{part}_proj.weight_scale"] = q, s
        elif name.endswith("experts.down_proj"):
            base = name[: -len("down_proj")]
            for e in range(config.num_experts):
                q, s, d = fp8(p[e])
                p[e] = d
                tensors[f"{base}{e}.down_proj.weight"], tensors[f"{base}{e}.down_proj.weight_scale"] = q, s
        elif FP8_LINEAR.fullmatch(name):
            q, s, d = fp8(p)
            p.copy_(d)
            tensors[name], tensors[name + "_scale"] = q, s
        elif name.startswith("model.layers.") and name.endswith(("mlp.gate.weight", "e_score_correction_bias")):
            tensors[name] = p.float().contiguous()  # router stays FP32
        else:
            p.copy_(p.to(torch.bfloat16).float())
            tensors[name] = p.to(torch.bfloat16).contiguous()

# Two shards, split mid-model, with the Hub's index format.
weight_map, shards = {}, {"model-00001-of-00002.safetensors": {}, "model-00002-of-00002.safetensors": {}}
for name, t in tensors.items():
    m = re.match(r"model\.layers\.(\d+)\.", name)
    shard = "model-00001-of-00002.safetensors" if m and int(m[1]) < 4 else "model-00002-of-00002.safetensors"
    shards[shard][name], weight_map[name] = t, shard
for shard, ts in shards.items():
    save_file(ts, work / shard)
(work / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

seqs = [torch.randint(0, config.vocab_size, (n,)) for n in (5, 20, 37)]  # 20 and 37 exceed the sliding window
steer = torch.randn(config.hidden_size)
STEER_LAYER = 4


def reference(ids, add=None):
    hook = None
    if add is not None:
        hook = model.model.layers[STEER_LAYER].register_forward_hook(lambda m, a, out: out + add)
    with torch.no_grad():
        out = model(ids[None]).logits[0]
    if hook:
        hook.remove()
    return out


def streamed(beam, add=None):
    hook = (lambda L, i, h: h + add.to(h.dtype) if L == STEER_LAYER else None) if add is not None else None
    hidden = beam.run(seqs, after_layer=hook)
    return [torch.cat([z for _, z in beam.logits(x)]) for x in hidden]


ok = True
for dtype, tol in ((torch.float32, 1e-3), (torch.bfloat16, None)):
    beam = StreamedBeam(LocalCheckpoint(work), config, device="cpu", dtype=dtype, workers=4)
    for label, add in (("plain", None), ("steered", steer)):
        for ids, got in zip(seqs, streamed(beam, add)):
            ref = reference(ids, add)
            diff = (got - ref).abs().max().item()
            agree = (got.argmax(-1) == ref.argmax(-1)).float().mean().item()
            passed = tol is None or diff < tol
            ok &= passed
            print(f"{str(dtype)[6:]:8s} {label:7s} T={len(ids):2d}  max|Δlogit| {diff:.2e}  top1 agree {agree:.0%}"
                  f"  {'ok' if passed else 'FAIL'}{'' if tol else ' (bf16, informational)'}")
print("ALL OK" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
