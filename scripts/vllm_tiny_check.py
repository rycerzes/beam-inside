"""Check the beam_steer vLLM plugin against transformers on a tiny random Beam in the real FP8 checkpoint format.

    python vllm_tiny_check.py ref OUT       # transformers: build checkpoint, reference logprobs and residual norms
    python vllm_tiny_check.py vllm OUT fp8|bf16|nvfp4 [--eager] [--moe BACKEND]   # vLLM + beam_steer -> OUT/vllm_<tag>.json
    python vllm_tiny_check.py teacher OUT   # decode steps of vLLM's greedy continuations through the reference
    python vllm_tiny_check.py capture OUT   # captured residuals and router logits vs the reference
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
Routing: slot 4 adds router logit deltas only: at every MoE layer it forces the two least-used experts in (+8) and
bans the two most-used (-30), applied inside the reference router before its grouped top-k.
Bias ban (beam_steer_set_bias, the whole batch): at every MoE layer, the two most-used experts of each routing
group get correction bias -1000, as REAP's per-group pruning does; held to the reference with the same bias, then
the bias is restored and the unsteered prompts must match their earlier logprobs exactly.
Variants: 2w is slot 2 steered on positions 20-59 only, 3s is slot 3 skipping a set of token ids; each is held to its
own masked reference. Capture: residuals leaving every layer and MoE router logits at five positions of each prompt
(first, sixth, last, and the first two fed-back generated tokens), unsteered and under slot 3, with the prefill
chunked across steps, held to the reference's BF16 floor.
"""

import json
import os
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


SKIP_IDS = sorted({ids[t] for ids in prompts() for t in range(2, min(40, len(ids)), 4)})
VARIANTS = {"2w": (2, {"steer_from": 20, "steer_until": 60}), "3s": (3, {"steer_skip_ids": SKIP_IDS})}
ROUTE_SLOT = 4
BAN_DELTA = -1000.0
CASES = {"0": (0, {}), **{str(s): (s, {}) for s in STEERS}, "4r": (ROUTE_SLOT, {}), **VARIANTS}  # key -> (slot, extra_args)
CAPTURE_SLOTS = (0, 3, ROUTE_SLOT)


def capture_positions(ids):
    return sorted({0, 5, len(ids) - 1, len(ids), len(ids) + 1})  # 5 is the last token of the 6-token prompt


def _mask(ids, kw):
    """Positions of ids that a request with steering extra_args kw steers."""
    pos = torch.arange(len(ids))
    m = pos >= kw.get("steer_from", 0)
    if kw.get("steer_until") is not None:
        m &= pos < kw["steer_until"]
    if kw.get("steer_skip_ids"):
        m &= ~torch.isin(torch.tensor(ids), torch.tensor(kw["steer_skip_ids"]))
    return m


def _case_logprobs(model, ids, pushes, key, routes):
    slot, kw = CASES[key]
    return _run(model, ids, pushes.get(slot, []), _mask(ids, kw), routes=routes.get(slot, []))[0]


def _bf16(model):
    """The reference in BF16 with the router kept FP32, as in the checkpoint: the floor a BF16 engine is held to."""
    copy = type(model)(model.config).eval().to(torch.bfloat16)
    copy.load_state_dict(model.state_dict())
    for name, mod in copy.named_modules():
        if name.endswith("mlp.gate"):
            mod.float()
    return copy


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
              **_case_refs(model, pushes, ps, _routes(out), _bans(out))}
    ref = json.loads((out / "ref.json").read_text())
    (out / "ref.json").write_text(json.dumps(ref | {"nvfp4": result}))
    print("ref nvfp4: median residual norm per layer", [round(m, 3) for m in result["median_norm"]])


def _routed(router, deltas):
    """transformers' BeamTopKRouter.forward with logit deltas [T, experts] added before the grouped top-k."""
    def forward(hidden_states):
        hidden_states = hidden_states.view(-1, router.hidden_dim)
        router_logits = torch.nn.functional.linear(hidden_states.float(), router.weight.float()) + deltas
        scores = router_logits.sigmoid()
        choice = scores + router.e_score_correction_bias
        group_scores = choice.view(-1, router.num_group, router.num_experts // router.num_group).topk(2, dim=-1)[0].sum(-1)
        group_idx = torch.topk(group_scores, k=router.topk_group, dim=-1)[1]
        group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(-1, router.num_group, router.num_experts // router.num_group)
        choice = choice.masked_fill(~score_mask.reshape(-1, router.num_experts).bool(), float("-inf"))
        topk_indices = torch.topk(choice, k=router.top_k, dim=-1)[1]
        topk_weights = scores.gather(1, topk_indices)
        if router.norm_topk_prob:
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        return router_logits, topk_weights, topk_indices
    return forward


def _run(model, ids, pushes=(), mask=None, keep=False, routes=()):
    """Reference logprobs of ids[1:] and per-layer residual norms (positions > 0), with pushes [(layer, delta)] and
    router logit deltas [(layer, expert, delta)] applied at the positions where mask is true (all if None). keep
    also returns the residual leaving every layer [layers, T, hidden] and the router logits as the router saw them
    {MoE layer: [T, experts]}, both float32."""
    norms, resid, moe_in, hooks = {}, {}, {}, []
    scale = None if mask is None else mask[None, :, None]
    on = torch.ones(len(ids)) if mask is None else mask.float()
    route_delta = {}
    for li, expert, delta in routes:
        route_delta.setdefault(li, torch.zeros(len(ids), model.config.num_experts))[:, expert] += float(delta) * on
    for li, d in route_delta.items():
        model.model.layers[li].mlp.gate.forward = _routed(model.model.layers[li].mlp.gate, d)
    for i, layer in enumerate(model.model.layers):
        def hook(m, a, o, i=i):
            for li, delta in pushes:
                if li == i:
                    d = delta.to(o.dtype)
                    o = o + (d if scale is None else d * scale.to(o.dtype))
            norms[i] = o[0, 1:].float().norm(dim=-1)
            if keep:
                resid[i] = o[0].float()
            return o
        hooks.append(layer.register_forward_hook(hook))
        if keep and model.config.mlp_layer_types[i] == "sparse":
            hooks.append(layer.mlp.register_forward_pre_hook(
                lambda m, a, i=i: moe_in.__setitem__(i, a[0].reshape(-1, a[0].shape[-1]))))
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0]
        router = {i: torch.nn.functional.linear(x.float(), model.model.layers[i].mlp.gate.weight.float())
                  + route_delta.get(i, 0.0) for i, x in moe_in.items()}
    for h in hooks:
        h.remove()
    for li in route_delta:
        del model.model.layers[li].mlp.gate.forward  # back to the class's forward
    lp = torch.log_softmax(logits.float(), -1)[:-1].gather(1, torch.tensor(ids[1:])[:, None])[:, 0]
    if keep:
        return lp.tolist(), norms, torch.stack([resid[i] for i in range(len(resid))]), router
    return lp.tolist(), norms


def _ban_logprobs(model, ps, bans):
    """Unsteered reference logprobs with the bias ban [(layer, expert, delta)] applied, then the bias restored."""
    saved = {L: model.model.layers[L].mlp.gate.e_score_correction_bias.detach().clone() for L, _, _ in bans}
    with torch.no_grad():
        for L, e, d in bans:
            model.model.layers[L].mlp.gate.e_score_correction_bias[e] += d
    try:
        return [_run(model, ids)[0] for ids in ps]
    finally:
        with torch.no_grad():
            for L, b in saved.items():
                model.model.layers[L].mlp.gate.e_score_correction_bias.copy_(b)


def _case_refs(model, pushes, ps, routes, bans):
    """Reference prompt logprobs for every case and the bias ban, in float32 and in BF16 (the floor)."""
    model_bf16 = _bf16(model)
    return {"logprobs": {k: [_case_logprobs(model, ids, pushes, k, routes) for ids in ps] for k in CASES},
            "logprobs_bf16": {k: [_case_logprobs(model_bf16, ids, pushes, k, routes) for ids in ps] for k in CASES},
            "ban": _ban_logprobs(model, ps, bans), "ban_bf16": _ban_logprobs(model_bf16, ps, bans)}


def _bans(out):
    return [(L, e, d) for L, e, d in torch.load(out / "ban.pt", weights_only=True)]


def _routes(out):
    routes = {}
    for slot, layer, expert, delta in torch.load(out / "steer_route.pt", weights_only=True):
        routes.setdefault(slot, []).append((layer, expert, delta))
    return routes


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
    pushes, ps, routes = _pushes(out), prompts(), _routes(out)
    for tag in RUNS:
        path = out / f"vllm_{tag}.json"
        if not path.exists():
            continue
        model = models["nvfp4" if tag.startswith("nvfp4") else ""]
        v = json.loads(path.read_text())
        line = []
        for key in CASES:
            diffs = []
            for ids, gen, glp in zip(ps, v["greedy"][key], v["greedy_logprobs"][key]):
                lp = _case_logprobs(model, ids + gen, pushes, key, routes)
                diffs += [abs(a - b) for a, b in zip(lp[len(ids) - 1:], glp)]
            line.append(f"{key} {np.mean(diffs):.4f}")
        print(f"decode |Δlogprob| vs reference, {tag}: " + ", ".join(line))


def _topk(logits, gate, config):
    """Experts Beam's grouped sigmoid router picks from router logits [experts] (noaux_tc: groups ranked by the
    sum of their top-2 biased scores)."""
    biased = (torch.sigmoid(logits) + gate.e_score_correction_bias.float()).view(config.n_group, -1)
    keep = biased.topk(2, -1).values.sum(-1).topk(config.topk_group).indices
    masked = torch.full_like(biased, -float("inf"))
    masked[keep] = biased[keep]
    return set(masked.flatten().topk(config.num_experts_per_tok).indices.tolist())


def capture(out):
    """Captured residuals and router logits vs the reference, teacher-forced on vLLM's greedy tokens."""
    models = {"": _load_ref_model(out)}
    if (out / "ref_model_nvfp4.pt").exists():
        models["nvfp4"] = _load_ref_model(out, "ref_model_nvfp4.pt")
    pushes, ps, ok, routes = _pushes(out), prompts(), True, _routes(out)
    checked = 0

    def rel(a, b):  # per-layer relative error, rows [layers, d] or [d]
        return ((a - b).norm(dim=-1) / b.norm(dim=-1)).reshape(-1)

    for tag, tol in RUNS.items():
        path = out / f"capture_{tag}.pt"
        if not path.exists():
            continue
        checked += tol
        model = models["nvfp4" if tag.startswith("nvfp4") else ""]
        model_bf16, config = _bf16(model), model.config
        cap = torch.load(path, weights_only=True)
        gens = json.loads((out / f"vllm_{tag}.json").read_text())["capture_gen"]
        rows = {(key[0], key[1], pos): (r, slot) for r, (key, pos, _, slot) in enumerate(cap["index"])}
        expected = sum(len(capture_positions(ids)) for ids in ps) * len(CAPTURE_SLOTS)
        err, floor, shifted, r_err, r_floor, agree, wrong_slot = ([] for _ in range(7))
        for s_ in CAPTURE_SLOTS:
            for i, ids in enumerate(ps):
                full = ids + gens[f"{s_}/{i}"][:2]
                _, _, want, router = _run(model, full, pushes.get(s_, []), keep=True, routes=routes.get(s_, []))
                _, _, want_bf16, router_bf16 = _run(model_bf16, full, pushes.get(s_, []), keep=True,
                                                    routes=routes.get(s_, []))
                for pos in capture_positions(ids):
                    if (s_, i, pos) not in rows:
                        continue
                    r, slot = rows[(s_, i, pos)]
                    wrong_slot.append(slot != s_)
                    got = cap["resid"][:, r].float()
                    err.append(rel(got, want[:, pos]))
                    floor.append(rel(want_bf16[:, pos], want[:, pos]))
                    shifted.append(rel(got, want[:, pos - 1 if pos else 1]))
                    for li, logits in router.items():
                        r_err.append(rel(cap["router"][li, r], logits[pos]))
                        r_floor.append(rel(router_bf16[li][pos], logits[pos]))
                        gate = model.model.layers[li].mlp.gate
                        agree.append(_topk(cap["router"][li, r], gate, config) == _topk(logits[pos], gate, config))
        err, floor, shifted = (float(torch.cat(x).mean()) for x in (err, floor, shifted))
        r_err, r_floor = (float(torch.cat(x).mean()) for x in (r_err, r_floor))
        count_ok = len(cap["index"]) == expected and cap["dropped"] == 0 and not any(wrong_slot)
        resid_ok = not tol or (err < 2 * floor and shifted > 5 * err)
        router_ok = not tol or r_err < 2 * r_floor
        ok &= count_ok and resid_ok and router_ok
        print(f"== capture {tag}: {len(cap['index'])}/{expected} rows, dropped {cap['dropped']}, "
              f"slot mislabelled {sum(wrong_slot)}  {'ok' if count_ok else 'FAIL'}")
        print(f"  residual rel err {err:.4f} (BF16 floor {floor:.4f}; vs neighbouring position {shifted:.3f})  "
              f"{'ok' if resid_ok else 'FAIL'}")
        print(f"  router logits rel err {r_err:.4f} (BF16 floor {r_floor:.4f}); same top-k experts {np.mean(agree):.0%}  "
              f"{'ok' if router_ok else 'FAIL'}")
    ok &= checked > 0
    print("CAPTURE OK" if ok else "CAPTURE MISMATCH")
    return ok


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
    # Routing slot: at every MoE layer, force the two least-used experts in and ban the two most-used.
    use = {}
    for ids in ps:
        for li, logits in _run(model, ids, keep=True)[3].items():
            for row in logits:
                for e in _topk(row, model.model.layers[li].mlp.gate, config):
                    use[(li, e)] = use.get((li, e), 0) + 1
    moe = [i for i, t in enumerate(config.mlp_layer_types) if t == "sparse"]
    by_use = {L: sorted(range(config.num_experts), key=lambda e: use.get((L, e), 0)) for L in moe}
    torch.save([(ROUTE_SLOT, L, e, 8.0) for L in moe for e in by_use[L][:2]]
               + [(ROUTE_SLOT, L, e, -30.0) for L in moe for e in by_use[L][-2:]], out / "steer_route.pt")
    # Bias ban: the two most-used experts of every routing group, at every MoE layer (half the experts, as REAP).
    per = config.num_experts // config.n_group
    torch.save([(L, e, BAN_DELTA) for L in moe for g in range(config.n_group)
                for e in sorted(range(g * per, (g + 1) * per), key=lambda e: use.get((L, e), 0))[-per // 2:]],
               out / "ban.pt")
    result = {"median_norm": median, **_case_refs(model, pushes, ps, _routes(out), _bans(out))}
    (out / "ref.json").write_text(json.dumps({"prompts": ps, **result}))
    print("ref: median residual norm per layer", [round(m, 3) for m in median])


def vllm_run(out, kind, eager, moe):
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    os.environ["BEAM_STEER_CAPTURE_ROWS"] = "256"
    from vllm import LLM, SamplingParams

    from beam_steer import hist_quantiles

    # max_num_batched_tokens 96 chunks the 120-token prompt across steps; prefix caching would skip steering and
    # capture on cached prompt tokens.
    llm = LLM(model=str(out / f"ckpt_{kind}"), skip_tokenizer_init=True, dtype="bfloat16", max_model_len=512,
              gpu_memory_utilization=0.3, enforce_eager=eager, async_scheduling=False, moe_backend=moe or "triton",
              max_num_seqs=64, max_num_batched_tokens=96, enable_prefix_caching=False, seed=SEED)
    print("steered rows loaded:", llm.collective_rpc("beam_steer_set", args=(str(out / "steer.pt"),)))
    print("router deltas loaded:", llm.collective_rpc("beam_steer_set_route", args=(str(out / "steer_route.pt"),)))
    ps = prompts()

    def params(slot, extra=None, **kw):
        return SamplingParams(temperature=0, detokenize=False, extra_args={"steer_slot": slot, **(extra or {})}, **kw)

    # Residual norms on an unsteered batch.
    llm.collective_rpc("beam_steer_reset_stats")
    llm.generate([{"prompt_token_ids": ids} for ids in ps], params(0, max_tokens=1), use_tqdm=False)
    median = [q[1] for q in hist_quantiles(llm.collective_rpc("beam_steer_stats")[0])]

    # Prompt logprobs and greedy continuations for every (case, prompt), plus the capture requests, in one mixed batch.
    reqs = [(k, ids, params(CASES[k][0], CASES[k][1], max_tokens=24, prompt_logprobs=0, logprobs=0))
            for k in CASES for ids in ps]
    reqs += [(f"cap{s}/{i}", ids, params(s, {"capture": capture_positions(ids), "capture_key": [s, i]}, max_tokens=3))
             for s in CAPTURE_SLOTS for i, ids in enumerate(ps)]
    outs = llm.generate([{"prompt_token_ids": ids} for _, ids, _ in reqs], [p for _, _, p in reqs], use_tqdm=False)
    logprobs, greedy, greedy_lp = ({k: [] for k in CASES} for _ in range(3))
    capture_gen = {}
    for (k, ids, _), o in zip(reqs, outs):
        gen = list(o.outputs[0].token_ids)
        if k.startswith("cap"):
            capture_gen[k[3:]] = gen
            continue
        logprobs[k].append([o.prompt_logprobs[t][ids[t]].logprob for t in range(1, len(ids))])
        greedy[k].append(gen)
        greedy_lp[k].append([d[t].logprob for t, d in zip(gen, o.outputs[0].logprobs)])
    # The same unsteered requests on their own: any leak from steered neighbours shows up as a difference.
    alone = llm.generate([{"prompt_token_ids": ids} for ids in ps], [params(0, max_tokens=1, prompt_logprobs=0)] * len(ps),
                         use_tqdm=False)
    alone = [[o.prompt_logprobs[t][ids[t]].logprob for t in range(1, len(ids))] for ids, o in zip(ps, alone)]

    def plain_logprobs():
        outs = llm.generate([{"prompt_token_ids": ids} for ids in ps], [params(0, max_tokens=1, prompt_logprobs=0)] * len(ps),
                            use_tqdm=False)
        return [[o.prompt_logprobs[t][ids[t]].logprob for t in range(1, len(ids))] for ids, o in zip(ps, outs)]

    # The bias ban, for the whole batch; then the bias restored, and the same prompts again.
    print("bias entries banned:", llm.collective_rpc("beam_steer_set_bias", args=(str(out / "ban.pt"),)))
    banned = plain_logprobs()
    print("bias layers restored:", llm.collective_rpc("beam_steer_reset_bias"))
    after_reset = plain_logprobs()
    tag = f"{kind}{'-' + moe if moe else ''}_{'eager' if eager else 'graphs'}"
    print("captured rows, dropped:", llm.collective_rpc("beam_steer_capture_dump", args=(str(out / f"capture_{tag}.pt"),)))
    (out / f"vllm_{tag}.json").write_text(json.dumps(
        {"median_norm": median, "logprobs": logprobs, "greedy": greedy, "greedy_logprobs": greedy_lp, "alone": alone,
         "capture_gen": capture_gen, "ban": banned, "after_reset": after_reset,
         "device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability()}))
    print(f"vllm {tag}: median residual norm per layer", [round(m, 3) for m in median])


# Held to the BF16 floor (the reference itself run in BF16 vs FP32) on mean |Δlogprob|; False = report only.
RUNS = {"bf16_eager": True, "bf16_graphs": True, "fp8_graphs": False,
        "nvfp4-marlin_graphs": True, "nvfp4-auto_graphs": False}


def compare(out):
    refs = json.loads((out / "ref.json").read_text())

    def mad(xs, ys):
        return float(np.mean(np.concatenate([np.abs(np.array(a) - np.array(b)) for a, b in zip(xs, ys)])))

    ok, ran = True, 0
    for tag, tol in RUNS.items():
        path = out / f"vllm_{tag}.json"
        if not path.exists():
            print(f"{tag}: not run")
            continue
        ran += tol
        ref = refs["nvfp4"] if tag.startswith("nvfp4") else refs
        floor = {s_: mad(ref["logprobs_bf16"][s_], ref["logprobs"][s_]) for s_ in ref["logprobs"]}
        v = json.loads(path.read_text())
        print(f"== {tag} on {v['device']} sm{v['capability'][0]}{v['capability'][1]}"
              + ("" if tol else "  (quantised activations: informational)"))
        ratio = np.array(v["median_norm"]) / np.array(ref["median_norm"])
        norms_ok = bool(np.all(np.abs(ratio - 1) < 0.15))  # histogram bins are 9% wide
        ok &= norms_ok
        print(f"  median residual norm vLLM/ref per layer: {np.round(ratio, 3).tolist()}  {'ok' if norms_ok else 'FAIL'}")
        for key, (slot, kw) in CASES.items():
            got, want = v["logprobs"][key], ref["logprobs"][key]
            match = mad(got, want)
            per_prompt = [round(mad([a], [b]), 4) for a, b in zip(got, want)]
            line = f"  {key}: |Δ| vs its ref {match:.4f} (BF16 floor {floor[key]:.4f}; per prompt {per_prompt})"
            passed = not tol or match < floor[key]
            # Off-target refs: unsteered for a slot; unsteered and the slot steered everywhere for a variant.
            for other in ([] if key == "0" else ["0"] + ([str(slot)] if kw else [])):
                if not kw:
                    off = mad(got, [np.array(x) for x in ref["logprobs"][other]])
                    line += f", vs {other} ref {off:.4f}"
                    passed &= off > 5 * match if tol else off > 2 * match
                    continue
                # A variant differs from its off-target refs on part of the tokens only: on the tokens where the
                # two refs differ by > 0.2 nats, vLLM's mean |Δ| to its own ref must be well under that to the other
                # (the BF16 reference itself, as a stand-in for vLLM, scores <= 0.47 on the tiny model).
                g, w, o = (np.concatenate([np.array(x) for x in xs]) for xs in (got, want, ref["logprobs"][other]))
                sel = np.abs(w - o) > 0.2
                ratio = float(np.mean(np.abs(g - w)[sel]) / np.mean(np.abs(g - o)[sel])) if sel.any() else 1.0
                line += f", own/{other} |Δ| {ratio:.2f} on {int(sel.sum())} tokens"
                passed &= sel.sum() >= 10 and ratio < (0.6 if tol else 0.9)
            ok &= passed
            print(line + ("  ok" if passed else "  FAIL"))
        leak = max(float(np.max(np.abs(np.array(a) - np.array(b)))) for a, b in zip(v["alone"], v["logprobs"]["0"]))
        leak_ok = not tol or leak < 1e-3
        ok &= leak_ok
        print(f"  unsteered alone vs in mixed batch: max |Δ| {leak:.4f}  {'ok' if leak_ok else 'FAIL'}")
        if "ban" in v and "ban" in ref:
            match, floor_ban = mad(v["ban"], ref["ban"]), mad(ref["ban_bf16"], ref["ban"])
            effect = mad(v["ban"], ref["logprobs"]["0"])
            restored = max(float(np.max(np.abs(np.array(a) - np.array(b)))) for a, b in zip(v["after_reset"], v["alone"]))
            ban_ok = (not tol or match < floor_ban) and effect > 5 * match and restored < 1e-3
            ok &= ban_ok
            print(f"  bias ban: |Δ| vs its ref {match:.4f} (BF16 floor {floor_ban:.4f}), vs unbanned ref {effect:.4f}; "
                  f"after reset vs before: max |Δ| {restored:.4f}  {'ok' if ban_ok else 'FAIL'}")
        else:
            ok = False
            print("  bias ban: missing from the run or the reference  FAIL")
    paths = [out / f"vllm_bf16_{m}.json" for m in ("eager", "graphs")]
    if all(p.exists() for p in paths):
        e, g = (json.loads(p.read_text())["greedy"] for p in paths)
        same = np.mean([a == b for s in e for a, b in zip(e[s], g[s])])
        diff = np.mean([a != b for s in e if s != "0" for a, b in zip(e[s], e["0"])])
        print(f"bf16 greedy continuations: eager == graphs for {same:.0%}; steered != unsteered for {diff:.0%}")
    ok &= ran > 0  # a held-to-tolerance run must exist: a crashed run is not a pass
    print("ALL OK" if ok else ("MISMATCH" if ran else "MISMATCH: no held-to-tolerance run produced output"))
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
        elif cmd == "capture":
            sys.exit(0 if capture(out) else 1)
        else:
            sys.exit(0 if compare(out) else 1)
