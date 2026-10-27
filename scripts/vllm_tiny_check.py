"""Check the beam_steer vLLM plugin against transformers on a tiny random Beam in the real FP8 checkpoint format.

    python vllm_tiny_check.py ref OUT       # transformers: build checkpoint, reference logprobs and residual norms
    python vllm_tiny_check.py vllm OUT fp8|bf16|nvfp4 [--eager] [--moe BACKEND]   # vLLM + beam_steer -> OUT/vllm_<tag>.json
    python vllm_tiny_check.py compare OUT   # verdict

The tiny model takes the real config.json (meta/config.json, or OUT/real_config.json) and shrinks only the sizes,
so every feature flag matches Beam. Its projections go through per-channel FP8 with BF16 scales and are saved under
the Hub's names with the Hub's compressed-tensors quantization_config; the reference runs the dequantised weights
(rounded to BF16) in float32. vLLM also quantises activations to FP8 per token, so on that checkpoint agreement is
close, not exact. A second, BF16 checkpoint holds the same dequantised weights without quantization, so vLLM's
result there is held to a tight tolerance on every slot.

A third checkpoint follows the Hub's NVFP4 release (ModelOpt): only routed experts are quantised, to E2M1 values in
groups of 16 with FP8 group scales, an FP32 per-tensor scale and a static FP32 input scale calibrated on the
reference activations; everything else is BF16 (routers FP32). It has its own reference, whose experts hold the
dequantised NVFP4 weights. Through Marlin (W4A16, BF16 activations) vLLM is held to that reference's BF16 floor;
the default Blackwell kernel also quantises activations to FP4 (W4A4) and is reported only.

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
    return q, s, (q.float() * s.float()).to(torch.bfloat16).float()


def _hub_named(state, config):
    """transformers state dict -> the Hub's tensor names (experts split per expert, gate/up separate)."""
    named = {}
    for name, t in state.items():
        if name.endswith("experts.gate_up_proj"):
            base, inter = name[: -len("gate_up_proj")], config.moe_intermediate_size
            for e in range(config.num_experts):
                named[f"{base}{e}.gate_proj.weight"], named[f"{base}{e}.up_proj.weight"] = t[e, :inter], t[e, inter:]
        elif name.endswith("experts.down_proj"):
            base = name[: -len("down_proj")]
            for e in range(config.num_experts):
                named[f"{base}{e}.down_proj.weight"] = t[e]
        else:
            named[name] = t
    return named


E2M1_GRID = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])


def nvfp4(w, scale2):
    """ModelOpt NVFP4 of w [out, in]: (packed uint8 [out, in/2], FP8 group scales [out, in/16], dequantised fp32)."""
    out_dim, in_dim = w.shape
    groups = w.reshape(out_dim, in_dim // 16, 16)
    scale = (groups.abs().amax(-1) / 6 / scale2).to(torch.float8_e4m3fn)
    denom = (scale.float() * scale2).clamp_min(1e-30)[..., None]
    x = (groups / denom).clamp(-6, 6)
    idx = (x.abs()[..., None] - E2M1_GRID).abs().argmin(-1)
    codes = (idx | ((x < 0).long() << 3)).reshape(out_dim, in_dim)
    deq = (E2M1_GRID[idx] * torch.where(x < 0, -1.0, 1.0) * denom).reshape(out_dim, in_dim)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    return packed, scale, deq


def ref_nvfp4(out):
    """NVFP4 checkpoint and its reference, built from the reference model ref() saved."""
    from safetensors.torch import save_file

    model, pushes, ps = _load_ref_model(out), _pushes(out), prompts()
    config = model.config
    inter = config.moe_intermediate_size
    # Calibrate static input scales on the unsteered prompts: MoE input, and every expert's intermediate activation.
    moe_in = {}
    hooks = [layer.mlp.register_forward_pre_hook(lambda m, a, i=i: moe_in.setdefault(i, []).append(a[0].reshape(-1, a[0].shape[-1])))
             for i, layer in enumerate(model.model.layers) if config.mlp_layer_types[i] == "sparse"]
    for ids in ps:
        _run(model, ids)
    for h in hooks:
        h.remove()
    tensors = {}
    with torch.no_grad():
        for i, layer in enumerate(model.model.layers):
            if config.mlp_layer_types[i] != "sparse":
                continue
            x = torch.cat(moe_in[i])
            experts = layer.mlp.experts
            act = torch.nn.functional.silu(torch.einsum("th,eoh->eto", x, experts.gate_up_proj[:, :inter]))
            act = act * torch.einsum("th,eoh->eto", x, experts.gate_up_proj[:, inter:])
            in_scale = {"gate": x.abs().max() / (6 * 448), "down": act.abs().max() / (6 * 448)}
            for e in range(config.num_experts):
                base = f"model.layers.{i}.mlp.experts.{e}."
                gu = experts.gate_up_proj[e]
                s2 = gu.abs().max() / (6 * 448)  # gate and up share one per-tensor scale: vLLM fuses them
                for part, rows in (("gate", slice(0, inter)), ("up", slice(inter, 2 * inter))):
                    q, sc, deq = nvfp4(gu[rows], s2)
                    gu[rows] = deq
                    tensors |= {base + f"{part}_proj.weight": q, base + f"{part}_proj.weight_scale": sc,
                                base + f"{part}_proj.weight_scale_2": s2.float().reshape(()),
                                base + f"{part}_proj.input_scale": in_scale["gate"].float().reshape(())}
                d = experts.down_proj[e]
                s2 = d.abs().max() / (6 * 448)
                q, sc, deq = nvfp4(d, s2)
                d.copy_(deq)
                tensors |= {base + "down_proj.weight": q, base + "down_proj.weight_scale": sc,
                            base + "down_proj.weight_scale_2": s2.float().reshape(()),
                            base + "down_proj.input_scale": in_scale["down"].float().reshape(())}
    router = ("mlp.gate.weight", "e_score_correction_bias")
    for n, t in _hub_named(model.state_dict(), config).items():
        if ".mlp.experts." not in n:
            tensors[n] = t.contiguous() if n.endswith(router) else t.to(torch.bfloat16).contiguous()
    ignore = ["lm_head", "model.embed_tokens", *[f"model.layers.{i}*" for i in range(3)],
              *[f"model.layers.{i}.{m}" for i in range(3, config.num_hidden_layers) for m in ("mlp.gate", "self_attn*")]]
    quant = {"config_groups": {"group_0": {
                 "input_activations": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
                 "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16}, "targets": ["Linear"]}},
             "ignore": ignore, "quant_algo": "NVFP4", "producer": {"name": "modelopt", "version": "0.47.0"},
             "quant_method": "modelopt"}
    ckpt = out / "ckpt_nvfp4"
    ckpt.mkdir(parents=True, exist_ok=True)
    save_file({n: t.clone() for n, t in tensors.items()}, ckpt / "model.safetensors")  # scalar scales share storage
    cfg = json.loads((out / "ckpt_bf16" / "config.json").read_text())
    (ckpt / "config.json").write_text(json.dumps(cfg | {"quantization_config": quant}, indent=1))
    (ckpt / "hf_quant_config.json").write_text(json.dumps({"producer": quant["producer"], "quantization": {
        "quant_algo": "NVFP4", "kv_cache_quant_algo": None, "group_size": 16, "exclude_modules": ignore}}, indent=1))
    torch.save(model.state_dict(), out / "ref_model_nvfp4.pt")

    plain = [_run(model, ids) for ids in ps]
    result = {"median_norm": [torch.cat([n[i] for _, n in plain]).median().item() for i in range(config.num_hidden_layers)],
              "logprobs": {0: [lp for lp, _ in plain]}}
    for slot in STEERS:
        result["logprobs"][slot] = [_run(model, ids, pushes[slot])[0] for ids in ps]
    model_bf16 = type(model)(config).eval().to(torch.bfloat16)
    model_bf16.load_state_dict(model.state_dict())
    for name, mod in model_bf16.named_modules():
        if name.endswith("mlp.gate"):
            mod.float()
    result["logprobs_bf16"] = {s_: [_run(model_bf16, ids, pushes.get(s_, []))[0] for ids in ps] for s_ in [0, *STEERS]}
    ref = json.loads((out / "ref.json").read_text())
    (out / "ref.json").write_text(json.dumps(ref | {"nvfp4": result}))
    print("ref nvfp4: median residual norm per layer", [round(m, 3) for m in result["median_norm"]])


def _run(model, ids, pushes=()):
    """Reference logprobs of ids[1:] and per-layer residual norms (positions > 0), with pushes [(layer, delta)]."""
    norms, hooks = {}, []
    for i, layer in enumerate(model.model.layers):
        def hook(m, a, o, i=i):
            for li, delta in pushes:
                if li == i:
                    o = o + delta.to(o.dtype)
            norms[i] = o[0, 1:].float().norm(dim=-1)
            return o
        hooks.append(layer.register_forward_hook(hook))
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0]
    for h in hooks:
        h.remove()
    lp = torch.log_softmax(logits.float(), -1)[:-1].gather(1, torch.tensor(ids[1:])[:, None])[:, 0]
    return lp.tolist(), norms


def _pushes(out):
    pushes = {}
    for slot, li, delta in torch.load(out / "steer.pt", weights_only=True):
        pushes.setdefault(slot, []).append((li, delta))
    return pushes


def _load_ref_model(out, name="ref_model.pt"):
    from transformers import BeamConfig, BeamForCausalLM

    saved = torch.load(out / name, weights_only=True)
    config = BeamConfig(**json.loads((out / "ckpt_bf16" / "config.json").read_text()))
    config._attn_implementation = "eager"
    model = BeamForCausalLM(config).float().eval()
    model.load_state_dict(saved)
    return model


def teacher(out):
    """Teacher-force vLLM's greedy continuations through the reference and compare decode-step logprobs."""
    models = {"": _load_ref_model(out)}
    if (out / "ref_model_nvfp4.pt").exists():
        models["nvfp4"] = _load_ref_model(out, "ref_model_nvfp4.pt")
    pushes, ps = _pushes(out), prompts()
    for tag in RUNS:
        path = out / f"vllm_{tag}.json"
        if not path.exists():
            continue
        model = models["nvfp4" if tag.startswith("nvfp4") else ""]
        v = json.loads(path.read_text())
        line = []
        for slot in ["0", *map(str, STEERS)]:
            diffs = []
            for ids, gen, glp in zip(ps, v["greedy"][slot], v["greedy_logprobs"][slot]):
                lp, _ = _run(model, ids + gen, pushes.get(int(slot), []))
                diffs += [abs(a - b) for a, b in zip(lp[len(ids) - 1:], glp)]
            line.append(f"slot {slot} {np.mean(diffs):.4f}")
        print(f"decode |Δlogprob| vs reference, {tag}: " + ", ".join(line))


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

    for kind, quant in (("fp8", {"quantization_config": QUANT}), ("bf16", {})):
        ckpt = out / f"ckpt_{kind}"
        ckpt.mkdir(parents=True, exist_ok=True)
        if kind == "fp8":
            save_file(tensors, ckpt / "model.safetensors")
        else:
            router = ("mlp.gate.weight", "e_score_correction_bias")  # FP32 on the Hub, as in the FP8 checkpoint
            save_file({n: t.contiguous() if n.endswith(router) else t.to(torch.bfloat16).contiguous()
                       for n, t in _hub_named(model.state_dict(), config).items()}, ckpt / "model.safetensors")
        (ckpt / "config.json").write_text(json.dumps(cfg | quant | {"dtype": "bfloat16"}, indent=1))

    torch.save(model.state_dict(), out / "ref_model.pt")
    ps = prompts()
    plain = [_run(model, ids) for ids in ps]
    median = [torch.cat([n[i] for _, n in plain]).median().item() for i in range(config.num_hidden_layers)]
    g = torch.Generator().manual_seed(SEED + 2)
    entries = []
    for slot, ls in STEERS.items():
        for li in ls:
            u = torch.randn(config.hidden_size, generator=g)
            entries.append((slot, li, DOSE * median[li] * u / u.norm()))
    torch.save(entries, out / "steer.pt")
    pushes = _pushes(out)
    result = {"median_norm": median, "logprobs": {0: [lp for lp, _ in plain]}}
    for slot in STEERS:
        result["logprobs"][slot] = [_run(model, ids, pushes[slot])[0] for ids in ps]
    # The same reference run in BF16: the rounding floor a BF16 engine can be held to.
    model_bf16 = BeamForCausalLM(config).eval().to(torch.bfloat16)
    model_bf16.load_state_dict(model.state_dict())
    for name, mod in model_bf16.named_modules():
        if name.endswith("mlp.gate"):
            mod.float()  # the router stays FP32, as in the checkpoint
    result["logprobs_bf16"] = {s_: [_run(model_bf16, ids, pushes.get(s_, []))[0] for ids in ps] for s_ in [0, *STEERS]}
    (out / "ref.json").write_text(json.dumps({"prompts": ps, **result}))
    print("ref: median residual norm per layer", [round(m, 3) for m in median])


def vllm_run(out, kind, eager, moe):
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from vllm import LLM, SamplingParams

    from beam_steer import hist_quantiles

    llm = LLM(model=str(out / f"ckpt_{kind}"), skip_tokenizer_init=True, dtype="bfloat16", max_model_len=512,
              gpu_memory_utilization=0.3, enforce_eager=eager, async_scheduling=False, moe_backend=moe or "triton",
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
                        [params(s, max_tokens=24, prompt_logprobs=0, logprobs=0) for s, _ in reqs], use_tqdm=False)
    logprobs, greedy, greedy_lp = ({s: [] for s in slots} for _ in range(3))
    for (s, ids), o in zip(reqs, outs):
        logprobs[s].append([o.prompt_logprobs[t][ids[t]].logprob for t in range(1, len(ids))])
        gen = list(o.outputs[0].token_ids)
        greedy[s].append(gen)
        greedy_lp[s].append([d[t].logprob for t, d in zip(gen, o.outputs[0].logprobs)])
    # The same unsteered requests on their own: any leak from steered neighbours shows up as a difference.
    alone = llm.generate([{"prompt_token_ids": ids} for ids in ps], [params(0, max_tokens=1, prompt_logprobs=0)] * len(ps),
                         use_tqdm=False)
    alone = [[o.prompt_logprobs[t][ids[t]].logprob for t in range(1, len(ids))] for ids, o in zip(ps, alone)]
    tag = f"{kind}{'-' + moe if moe else ''}_{'eager' if eager else 'graphs'}"
    (out / f"vllm_{tag}.json").write_text(json.dumps(
        {"median_norm": median, "logprobs": logprobs, "greedy": greedy, "greedy_logprobs": greedy_lp, "alone": alone,
         "device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability()}))
    print(f"vllm {tag}: median residual norm per layer", [round(m, 3) for m in median])


# Held to the BF16 floor (the reference itself run in BF16 vs FP32) on mean |Δlogprob|; False = report only.
RUNS = {"bf16_eager": True, "bf16_graphs": True, "fp8_graphs": False,
        "nvfp4-marlin_graphs": True, "nvfp4-auto_graphs": False}


def compare(out):
    refs = json.loads((out / "ref.json").read_text())

    def mad(xs, ys):
        return float(np.mean(np.concatenate([np.abs(np.array(a) - np.array(b)) for a, b in zip(xs, ys)])))

    ok = True
    for tag, tol in RUNS.items():
        path = out / f"vllm_{tag}.json"
        if not path.exists():
            print(f"{tag}: not run")
            continue
        ref = refs["nvfp4"] if tag.startswith("nvfp4") else refs
        plain = [np.array(x) for x in ref["logprobs"]["0"]]
        floor = {s_: mad(ref["logprobs_bf16"][s_], ref["logprobs"][s_]) for s_ in ref["logprobs"]}
        v = json.loads(path.read_text())
        print(f"== {tag} on {v['device']} sm{v['capability'][0]}{v['capability'][1]}"
              + ("" if tol else "  (quantised activations: informational)"))
        ratio = np.array(v["median_norm"]) / np.array(ref["median_norm"])
        norms_ok = bool(np.all(np.abs(ratio - 1) < 0.15))  # histogram bins are 9% wide
        ok &= norms_ok
        print(f"  median residual norm vLLM/ref per layer: {np.round(ratio, 3).tolist()}  {'ok' if norms_ok else 'FAIL'}")
        for slot in ["0", *map(str, STEERS)]:
            got, want = v["logprobs"][slot], ref["logprobs"][slot]
            match = mad(got, want)
            per_prompt = [round(mad([a], [b]), 4) for a, b in zip(got, want)]
            line = f"  slot {slot}: |Δ| vs its ref {match:.4f} (BF16 floor {floor[slot]:.4f}; per prompt {per_prompt})"
            passed = not tol or match < floor[slot]
            if slot != "0":
                off = mad(got, plain)
                line += f", vs unsteered ref {off:.4f}"
                passed &= off > 5 * match if tol else off > 2 * match
            ok &= passed
            print(line + ("  ok" if passed else "  FAIL"))
        leak = max(float(np.max(np.abs(np.array(a) - np.array(b)))) for a, b in zip(v["alone"], v["logprobs"]["0"]))
        leak_ok = not tol or leak < 1e-3
        ok &= leak_ok
        print(f"  unsteered alone vs in mixed batch: max |Δ| {leak:.4f}  {'ok' if leak_ok else 'FAIL'}")
    paths = [out / f"vllm_bf16_{m}.json" for m in ("eager", "graphs")]
    if all(p.exists() for p in paths):
        e, g = (json.loads(p.read_text())["greedy"] for p in paths)
        same = np.mean([a == b for s in e for a, b in zip(e[s], g[s])])
        diff = np.mean([a != b for s in e if s != "0" for a, b in zip(e[s], e["0"])])
        print(f"bf16 greedy continuations: eager == graphs for {same:.0%}; steered != unsteered for {diff:.0%}")
    print("ALL OK" if ok else "MISMATCH")
    return ok


if __name__ == "__main__":
    cmd, out = sys.argv[1], pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    if cmd == "ref":
        ref(out)
        ref_nvfp4(out)
    elif cmd == "vllm":
        moe = sys.argv[sys.argv.index("--moe") + 1] if "--moe" in sys.argv else None
        vllm_run(out, sys.argv[3], eager="--eager" in sys.argv, moe=moe)
    else:
        if cmd == "teacher":
            teacher(out)
        else:
            sys.exit(0 if compare(out) else 1)
