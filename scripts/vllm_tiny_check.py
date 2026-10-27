"""Check the beam_steer vLLM plugin against transformers on a tiny random Beam in the real FP8 checkpoint format.

    python vllm_tiny_check.py ref OUT       # transformers: build checkpoint, reference logprobs and residual norms
    python vllm_tiny_check.py vllm OUT [--eager]   # vLLM + beam_steer: same quantities, written to OUT/vllm_*.json
    python vllm_tiny_check.py compare OUT   # verdict

The tiny model takes the real config.json (meta/config.json, or OUT/real_config.json) and shrinks only the sizes,
so every feature flag matches Beam. Its projections go through per-channel FP8 with BF16 scales and are saved under
the Hub's names with the Hub's compressed-tensors quantization_config; the reference runs the dequantised weights
in float32. vLLM also quantises activations to FP8 per token, so agreement is close, not exact.

Steering slots: 1 pushes the residual leaving dense layer 1, 2 the one leaving MoE layer 5, 3 both layers 2 and 6.
Each push is 0.8 x the reference median residual norm leaving that layer, along a fixed random direction.
"""

import json
import pathlib
import re
import sys

import numpy as np
import torch

SEED, VOCAB, DOSE = 0, 1024, 0.8
PROMPT_LENS = (6, 40, 120)  # 40 and 120 exceed the sliding window
STEERS = {1: [1], 2: [5], 3: [2, 6]}  # slot -> layers pushed
QUANT = {
    "config_groups": {"FP8_DYNAMIC": {
        "format": "float-quantized",
        "input_activations": {"actorder": None, "block_structure": None, "dynamic": True, "group_size": None,
                              "num_bits": 8, "observer": None, "observer_kwargs": {}, "scale_dtype": None,
                              "strategy": "token", "symmetric": True, "type": "float", "zp_dtype": None},
        "output_activations": None, "targets": ["Linear"],
        "weights": {"actorder": None, "block_structure": None, "dynamic": False, "group_size": None, "num_bits": 8,
                    "observer": "memoryless_minmax", "observer_kwargs": {}, "scale_dtype": None,
                    "strategy": "channel", "symmetric": True, "type": "float", "zp_dtype": None}}},
    "format": "float-quantized", "global_compression_ratio": None,
    "ignore": ["lm_head", "model.embed_tokens", "re:.*mlp\\.gate$"],
    "kv_cache_scheme": None, "quant_method": "compressed-tensors", "quantization_status": "compressed",
    "transform_config": {}, "version": "0.18.0",
}
TINY = dict(vocab_size=VOCAB, hidden_size=512, intermediate_size=1024, moe_intermediate_size=256,
            num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2, head_dim=128, num_experts=16,
            num_experts_per_tok=4, n_group=4, topk_group=2, sliding_window=16, max_position_embeddings=4096,
            layer_types=["sliding_attention"] * 5 + ["full_attention", "sliding_attention", "full_attention"],
            mlp_layer_types=["dense"] * 3 + ["sparse"] * 5)
FP8_LINEAR = re.compile(r"model\.layers\.\d+\.(self_attn\.(q|k|v|o|gate)_proj|mlp\.(gate|up|down)_proj)\.weight")


def prompts():
    g = torch.Generator().manual_seed(SEED + 1)
    return [torch.randint(0, VOCAB, (n,), generator=g).tolist() for n in PROMPT_LENS]


def fp8(w):
    s = (w.abs().amax(-1, keepdim=True) / 448).clamp_min(1e-12).to(torch.bfloat16)
    q = (w / s.float()).to(torch.float8_e4m3fn)
    return q, s, q.float() * s.float()


def ref(out):
    from safetensors.torch import save_file
    from transformers import BeamConfig, BeamForCausalLM

    torch.manual_seed(SEED)
    real = json.loads((out / "real_config.json").read_text())
    cfg = {k: v for k, v in real.items() if k not in ("quantization_config", "auto_map")} | TINY
    config = BeamConfig(**cfg)
    config._attn_implementation = "eager"
    model = BeamForCausalLM(config).float().eval()

    tensors = {}
    with torch.no_grad():
        for name, p in list(model.state_dict().items()):
            if name.endswith("norm.weight"):
                p.copy_(1 + 0.2 * torch.randn_like(p))
            if name.endswith("e_score_correction_bias"):
                p.copy_(20 + 0.05 * torch.randn_like(p))
            if name.endswith("mlp.gate.weight"):
                p.mul_(20)  # spread router scores so expert choice is not a near-tie
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
                tensors[name] = p.float().contiguous()
            else:
                p.copy_(p.to(torch.bfloat16).float())
                tensors[name] = p.to(torch.bfloat16).contiguous()

    ckpt = out / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    save_file(tensors, ckpt / "model.safetensors")
    (ckpt / "config.json").write_text(json.dumps(cfg | {"quantization_config": QUANT, "dtype": "bfloat16"}, indent=1))

    ps = prompts()
    layers = model.model.layers

    def run(ids, pushes=()):
        norms, hooks = {}, []
        for i, layer in enumerate(layers):
            def hook(m, a, o, i=i):
                for li, delta in pushes:
                    if li == i:
                        o = o + delta
                norms[i] = o[0, 1:].norm(dim=-1)
                return o
            hooks.append(layer.register_forward_hook(hook))
        with torch.no_grad():
            logits = model(torch.tensor([ids])).logits[0]
        for h in hooks:
            h.remove()
        lp = torch.log_softmax(logits.float(), -1)[:-1].gather(1, torch.tensor(ids[1:])[:, None])[:, 0]
        return lp.tolist(), norms

    plain = [run(ids) for ids in ps]
    median = [torch.cat([n[i] for _, n in plain]).median().item() for i in range(len(layers))]
    g = torch.Generator().manual_seed(SEED + 2)
    entries, pushes = [], {}
    for slot, ls in STEERS.items():
        pushes[slot] = []
        for li in ls:
            u = torch.randn(config.hidden_size, generator=g)
            delta = DOSE * median[li] * u / u.norm()
            entries.append((slot, li, delta))
            pushes[slot].append((li, delta))
    torch.save(entries, out / "steer.pt")
    result = {"median_norm": median, "logprobs": {0: [lp for lp, _ in plain]}}
    for slot in STEERS:
        result["logprobs"][slot] = [run(ids, pushes[slot])[0] for ids in ps]
    (out / "ref.json").write_text(json.dumps({"prompts": ps, **result}))
    print("ref: median residual norm per layer", [round(m, 3) for m in median])


def vllm_run(out, eager):
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from vllm import LLM, SamplingParams

    from beam_steer import hist_quantiles

    llm = LLM(model=str(out / "ckpt"), skip_tokenizer_init=True, dtype="bfloat16", max_model_len=512,
              gpu_memory_utilization=0.3, enforce_eager=eager, async_scheduling=False, moe_backend="triton",
              max_num_seqs=64, seed=SEED)
    print("steered rows loaded:", llm.collective_rpc("beam_steer_set", args=(str(out / "steer.pt"),)))
    ps = prompts()

    def params(slot, **kw):
        return SamplingParams(temperature=0, detokenize=False, extra_args={"steer_slot": slot}, **kw)

    # Residual norms on an unsteered batch.
    llm.collective_rpc("beam_steer_reset_stats")
    llm.generate([{"prompt_token_ids": ids} for ids in ps], params(0, max_tokens=1), use_tqdm=False)
    median = [q[1] for q in hist_quantiles(llm.collective_rpc("beam_steer_stats")[0])]

    # Prompt logprobs and greedy continuations for every (slot, prompt) in one mixed batch.
    slots = [0, *STEERS]
    reqs = [(s, ids) for s in slots for ids in ps]
    outs = llm.generate([{"prompt_token_ids": ids} for _, ids in reqs],
                        [params(s, max_tokens=24, prompt_logprobs=0) for s, _ in reqs], use_tqdm=False)
    logprobs, greedy = {s: [] for s in slots}, {s: [] for s in slots}
    for (s, ids), o in zip(reqs, outs):
        logprobs[s].append([o.prompt_logprobs[t][ids[t]].logprob for t in range(1, len(ids))])
        greedy[s].append(list(o.outputs[0].token_ids))
    tag = "eager" if eager else "graphs"
    (out / f"vllm_{tag}.json").write_text(json.dumps(
        {"median_norm": median, "logprobs": logprobs, "greedy": greedy,
         "device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability()}))
    print(f"vllm {tag}: median residual norm per layer", [round(m, 3) for m in median])


def compare(out):
    ref = json.loads((out / "ref.json").read_text())
    ok = True
    for tag in ("eager", "graphs"):
        path = out / f"vllm_{tag}.json"
        if not path.exists():
            print(f"{tag}: missing")
            ok = False
            continue
        v = json.loads(path.read_text())
        print(f"== {tag} on {v['device']} sm{v['capability'][0]}{v['capability'][1]}")
        ratio = np.array(v["median_norm"]) / np.array(ref["median_norm"])
        norms_ok = bool(np.all(np.abs(ratio - 1) < 0.15))  # histogram bins are 9% wide
        ok &= norms_ok
        print(f"  median residual norm vLLM/ref per layer: {np.round(ratio, 3).tolist()}  {'ok' if norms_ok else 'FAIL'}")
        plain = [np.array(x) for x in ref["logprobs"]["0"]]
        for slot in ["0", *map(str, STEERS)]:
            got = [np.array(x) for x in v["logprobs"][slot]]
            want = [np.array(x) for x in ref["logprobs"][slot]]
            match = float(np.mean(np.concatenate([np.abs(a - b) for a, b in zip(got, want)])))
            line = f"  slot {slot}: mean |Δlogprob| vs matching ref {match:.4f}"
            passed = match < 0.1
            if slot != "0":
                off = float(np.mean(np.concatenate([np.abs(a - b) for a, b in zip(got, plain)])))
                line += f", vs unsteered ref {off:.4f}"
                passed &= off > 5 * match
            ok &= passed
            print(line + ("  ok" if passed else "  FAIL"))
    paths = [out / f"vllm_{t}.json" for t in ("eager", "graphs")]
    if all(p.exists() for p in paths):
        e, g = (json.loads(p.read_text())["greedy"] for p in paths)
        same = np.mean([a == b for s in e for a, b in zip(e[s], g[s])])
        diff = np.mean([a != b for s in e if s != "0" for a, b in zip(e[s], e["0"])])
        print(f"greedy continuations: eager == graphs for {same:.0%}; steered != unsteered for {diff:.0%}")
    print("ALL OK" if ok else "MISMATCH")
    return ok


if __name__ == "__main__":
    cmd, out = sys.argv[1], pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    if cmd == "ref":
        ref(out)
    elif cmd == "vllm":
        vllm_run(out, eager="--eager" in sys.argv)
    else:
        sys.exit(0 if compare(out) else 1)
