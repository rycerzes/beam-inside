"""Per-request residual-stream steering, capture and residual-norm stats for vLLM's Beam, as a vLLM general plugin.

Each request picks a steering slot through `SamplingParams(extra_args={"steer_slot": s, ...})`; slot 0 is unsteered.
A slot holds one vector per layer, added to the residual stream leaving that layer (Lyra's `mlp_out` hook point:
after the MLP's post-norm, so the push is not renormalised away). Beam's MLP boundary is a fused
post-norm + depth-scaled residual add + next pre-norm,

    new_residual = a * post_norm(mlp_out) + b * residual,

so adding delta / b to `residual` just before it yields new_residual + delta, with the next layer's normalised
input computed from the steered residual.

A slot can also steer routing: per MoE layer, a logit delta per expert, added to the router's logits before
Beam's grouped top-k (a large positive delta forces an expert in, a large negative one bans it, after SteerMoE).
Beam chooses experts by sigmoid(logit) + e_score_correction_bias, and the bias is ~18-28 with a spread of at most
0.02 across a layer's experts, so a logit ban only holds when every kept candidate scores above that spread. An
exact ban instead lowers the banned experts' correction bias (`beam_steer_set_bias`): the bias enters only the
choice, never the routing weights, so this is what slicing the experts out of the router does (as REAP does). The
bias is shared by every request, so a bias ban applies to the whole batch until `beam_steer_reset_bias`.

Per-request keys in `extra_args`:
  steer_slot       slot whose vectors and router deltas apply (default 0, none)
  steer_from       first absolute position steered (default 0)
  steer_until      absolute position the steering stops before (default none)
  steer_skip_ids   token ids never steered, e.g. the chat delimiters
  capture          absolute positions whose residual leaving every layer, and router logits at every MoE layer
                   (after any router delta), are copied out (needs BEAM_STEER_CAPTURE_ROWS > 0)
  capture_key      any JSON value, returned with each captured row

Positions refer to the token fed in, so the first generated token is captured at position len(prompt). Prefix
caching skips the forward for cached prompt tokens, so they are neither steered nor captured: run with
enable_prefix_caching=False.

Buffers are persistent and indexed in-graph, so CUDA graphs stay valid; the model runner writes each step's
per-token slots and capture indices before the forward. Workers gain RPCs for `LLM.collective_rpc`:
`beam_steer_set(path)`, `beam_steer_set_route(path)`, `beam_steer_set_bias(path)`, `beam_steer_reset_bias()`,
`beam_steer_stats()`, `beam_steer_reset_stats()` and `beam_steer_capture_dump(path)`.

Environment: BEAM_STEER_SLOTS (default 512), BEAM_STEER_CAPTURE_ROWS (rows held between dumps, default 0 = capture
off; each row costs layers x hidden x 2 bytes plus layers x experts x 4), BEAM_STEER_CAPTURE_PER_STEP (default 1024).

Written against vLLM main at 5d35ce3 (vllm/vllm-openai:nightly-5d35ce3934a7804b298559d2d99eec33792e1eb9).
Requires VLLM_USE_V2_MODEL_RUNNER=0 and async_scheduling=False.
"""

import os

import numpy as np
import torch

# Residual-norm histogram: 8 bins per octave from 2^-8 to 2^24.
HIST_BINS, HIST_LO, HIST_PER_OCTAVE = 256, -8, 8


class _State:
    table = None  # [num_layers, slots, hidden], delta / b in model dtype
    route = None  # [num_layers, slots, experts] float32, added to router logits (MoE layers)
    slots = None  # [max_tokens] int64, per-token slot for the current step
    valid = None  # [max_tokens] float32, 1 for real tokens, 0 for padding
    hist = None  # [num_layers, HIST_BINS] float32 counts of ‖residual‖ leaving each layer, positions > 0
    # Capture: rows [0, rows) hold captured tokens, the last row takes the padding entries of cap_src / cap_dst.
    resid = None  # [num_layers, rows + 1, hidden] model dtype, residual leaving each layer
    router = None  # [num_layers, rows + 1, experts] float32 router logits (zero for dense layers)
    cap_src = None  # [per_step] int64, token index in the step's batch
    cap_dst = None  # [per_step] int64, row in resid / router
    cap_rows = 0  # rows filled since the last dump
    cap_index = []  # per filled row: (capture_key, position, token_id, slot steering that token)
    cap_dropped = 0  # captures lost to a full buffer
    bias_orig = {}  # MoE layer -> its correction bias before beam_steer_set_bias


STATE = _State()


def register():
    if getattr(register, "done", False):
        return
    register.done = True
    _patch_model()
    _patch_runner()
    _patch_worker()


def _patch_model():
    from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
    from vllm.models.beam.nvidia import model as beam
    from vllm.models.beam.nvidia.layernorm import beam_rms_norm
    from vllm.models.beam.nvidia.moe import BeamSparseMoeBlock

    init = beam.BeamModel.__init__

    def model_init(self, *, vllm_config, prefix=""):
        init(self, vllm_config=vllm_config, prefix=prefix)
        config, sched = self.config, vllm_config.scheduler_config
        device = torch.device("cuda", torch.cuda.current_device())
        dtype = vllm_config.model_config.dtype
        max_tokens = max(sched.max_num_batched_tokens, sched.max_num_seqs) + 1024
        slots = int(os.environ.get("BEAM_STEER_SLOTS", "512"))
        STATE.table = torch.zeros(config.num_hidden_layers, slots, config.hidden_size, dtype=dtype, device=device)
        STATE.route = torch.zeros(config.num_hidden_layers, slots, config.num_experts, dtype=torch.float32, device=device)
        for layer in self.layers:
            if isinstance(getattr(layer, "mlp", None), BeamSparseMoeBlock):
                _wrap_gate(layer.mlp.gate, layer.layer_id)
        STATE.slots = torch.zeros(max_tokens, dtype=torch.int64, device=device)
        STATE.valid = torch.zeros(max_tokens, dtype=torch.float32, device=device)
        STATE.hist = torch.zeros(config.num_hidden_layers, HIST_BINS, dtype=torch.float32, device=device)
        rows = int(os.environ.get("BEAM_STEER_CAPTURE_ROWS", "0"))
        if rows > 0:
            per_step = int(os.environ.get("BEAM_STEER_CAPTURE_PER_STEP", "1024"))
            STATE.resid = torch.zeros(config.num_hidden_layers, rows + 1, config.hidden_size, dtype=dtype, device=device)
            STATE.router = torch.zeros(config.num_hidden_layers, rows + 1, config.num_experts, dtype=torch.float32,
                                       device=device)
            STATE.cap_src = torch.zeros(per_step, dtype=torch.int64, device=device)
            STATE.cap_dst = torch.full((per_step,), rows, dtype=torch.int64, device=device)

    def layer_forward(self, positions, hidden_states, normalized_states=None, next_norm_weight=None):
        # Upstream BeamDecoderLayer.forward, plus the steering add before the MLP boundary, capture and norm stats.
        x = (
            beam_rms_norm(hidden_states, self.input_layernorm.weight, self.rms_norm_eps)
            if normalized_states is None
            else normalized_states
        )
        x = self.self_attn(positions, x)
        hidden_states, x = self.attention_boundary(
            x,
            hidden_states,
            self.post_attention_layernorm.weight,
            self.rms_norm_eps,
            self.pre_feedforward_layernorm.weight,
            self.rms_norm_eps,
            self.attn_scales,
        )
        assert x is not None
        if STATE.resid is not None and isinstance(self.mlp, BeamSparseMoeBlock):
            # The router's own input and FP32 weight; selection (bias, groups, top-k) is left to the reader.
            logits = torch.nn.functional.linear(x.index_select(0, STATE.cap_src).float(), self.mlp.gate.weight.float())
            logits = logits + STATE.route[self.layer_id].index_select(0, STATE.slots.index_select(0, STATE.cap_src))
            STATE.router[self.layer_id].index_copy_(0, STATE.cap_dst, logits)
        x = self.mlp(x)
        boundary = self.mlp_boundary
        if isinstance(x, UnfinalizedMoEOutput):
            assert self.finalize_boundary is not None
            boundary = self.finalize_boundary
        n = hidden_states.shape[0]
        hidden_states = hidden_states + STATE.table[self.layer_id].index_select(0, STATE.slots[:n])
        hidden_states, normalized = boundary(
            x,
            hidden_states,
            self.post_feedforward_layernorm.weight,
            self.rms_norm_eps,
            next_norm_weight,
            self.rms_norm_eps if next_norm_weight is not None else None,
            self.mlp_scales,
        )
        if STATE.resid is not None:
            STATE.resid[self.layer_id].index_copy_(0, STATE.cap_dst, hidden_states.index_select(0, STATE.cap_src))
        norm = hidden_states.float().norm(dim=-1).clamp_min(2.0**HIST_LO)
        bins = ((torch.log2(norm) - HIST_LO) * HIST_PER_OCTAVE).long().clamp(0, HIST_BINS - 1)
        weight = STATE.valid[:n] * (positions[:n] > 0).float()
        STATE.hist[self.layer_id].index_add_(0, bins, weight)
        return hidden_states, normalized

    beam.BeamModel.__init__ = model_init
    beam.BeamDecoderLayer.forward = layer_forward


def _wrap_gate(gate, layer_id):
    """Add each token's slot's router delta to the gate's logits. The MoE runner calls the gate and feeds its
    logits to the grouped top-k, so the delta takes part in selection and in the routing weights."""
    forward = gate.forward

    def gate_forward(x):
        out = forward(x)
        logits, rest = (out[0], out[1:]) if isinstance(out, tuple) else (out, None)
        logits = logits + STATE.route[layer_id].index_select(0, STATE.slots[: x.shape[0]]).to(logits.dtype)
        return (logits, *rest) if rest is not None else logits

    gate.forward = gate_forward


def _patch_runner():
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    prepare = GPUModelRunner._prepare_inputs

    def prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        out = prepare(self, scheduler_output, num_scheduled_tokens)
        if STATE.slots is None:
            return out
        num_reqs = self.input_batch.num_reqs
        counts = np.asarray(num_scheduled_tokens[:num_reqs], dtype=np.int64)
        n = int(counts.sum())
        starts = np.cumsum(counts) - counts
        req_of = np.repeat(np.arange(num_reqs), counts)
        pos = self.input_batch.num_computed_tokens_cpu[:num_reqs][req_of] + (np.arange(n) - starts[req_of])
        token_ids = self.input_ids.cpu[:n].numpy()
        slots = np.zeros(STATE.slots.shape[0], dtype=np.int64)
        cap_src, cap_dst = [], []
        per_step = 0 if STATE.cap_src is None else STATE.cap_src.shape[0]
        for r, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            params = self.requests[req_id].sampling_params
            extra = (params and params.extra_args) or {}
            lo, hi = starts[r], starts[r] + counts[r]
            p = pos[lo:hi]
            slot = int(extra.get("steer_slot", 0))
            if slot:
                on = p >= int(extra.get("steer_from", 0))
                if extra.get("steer_until") is not None:
                    on &= p < int(extra["steer_until"])
                if extra.get("steer_skip_ids"):
                    on &= ~np.isin(token_ids[lo:hi], np.asarray(extra["steer_skip_ids"], dtype=np.int64))
                slots[lo:hi] = slot * on
            if extra.get("capture") and STATE.resid is not None:
                for i in np.flatnonzero(np.isin(p, np.asarray(extra["capture"], dtype=np.int64))):
                    if STATE.cap_rows >= STATE.resid.shape[1] - 1 or len(cap_src) >= per_step:
                        STATE.cap_dropped += 1
                        continue
                    cap_src.append(lo + i)
                    cap_dst.append(STATE.cap_rows)
                    STATE.cap_index.append((extra.get("capture_key", req_id), int(p[i]), int(token_ids[lo + i]),
                                            int(slots[lo + i])))
                    STATE.cap_rows += 1
        valid = np.zeros(STATE.valid.shape[0], dtype=np.float32)
        valid[:n] = 1
        # Blocking copies on the forward's stream: ordered after the previous step's kernels.
        STATE.slots.copy_(torch.from_numpy(slots))
        STATE.valid.copy_(torch.from_numpy(valid))
        if STATE.cap_src is not None:
            src = np.zeros(per_step, dtype=np.int64)
            dst = np.full(per_step, STATE.resid.shape[1] - 1, dtype=np.int64)
            src[: len(cap_src)], dst[: len(cap_dst)] = cap_src, cap_dst
            STATE.cap_src.copy_(torch.from_numpy(src))
            STATE.cap_dst.copy_(torch.from_numpy(dst))
        return out

    GPUModelRunner._prepare_inputs = prepare_inputs


def _patch_worker():
    from vllm.v1.worker.gpu_worker import Worker

    def beam_steer_set(self, path):
        """Load [(slot, layer, delta)] from a torch file; delta is the push added to the residual leaving `layer`."""
        layers = self.model_runner.get_model().model.layers
        STATE.table.zero_()
        for slot, layer, delta in torch.load(path, weights_only=True):
            assert 0 < slot < STATE.table.shape[1], f"slot {slot} out of range (0 is reserved for unsteered)"
            b = layers[layer].mlp_scales[1]
            STATE.table[layer, slot] += (delta.float() / b).to(STATE.table.dtype).to(STATE.table.device)
        torch.cuda.synchronize()
        return int((STATE.table.abs().sum(-1) > 0).sum())

    def beam_steer_set_route(self, path):
        """Load [(slot, layer, expert, delta)] from a torch file; delta is added to that expert's router logit."""
        STATE.route.zero_()
        for slot, layer, expert, delta in torch.load(path, weights_only=True):
            assert 0 < slot < STATE.route.shape[1], f"slot {slot} out of range (0 is reserved for unsteered)"
            STATE.route[layer, slot, expert] += float(delta)
        torch.cuda.synchronize()
        return int((STATE.route != 0).sum())

    def beam_steer_set_bias(self, path):
        """Load [(layer, expert, delta)] from a torch file; delta is added in place to that expert's routing
        correction bias for every request until beam_steer_reset_bias (a set first restores the original bias).
        In place, so the router (which holds the same tensor) and captured CUDA graphs see it."""
        beam_steer_reset_bias(self)
        layers = self.model_runner.get_model().model.layers
        n = 0
        for layer, expert, delta in torch.load(path, weights_only=True):
            bias = layers[layer].mlp.gate.e_score_correction_bias
            if layer not in STATE.bias_orig:
                STATE.bias_orig[layer] = bias.detach().clone()
            bias.data[expert] += float(delta)
            n += 1
        torch.cuda.synchronize()
        return n

    def beam_steer_reset_bias(self):
        """Restore every correction bias changed by beam_steer_set_bias; returns the layers restored."""
        layers = self.model_runner.get_model().model.layers
        for layer, orig in STATE.bias_orig.items():
            layers[layer].mlp.gate.e_score_correction_bias.data.copy_(orig)
        n = len(STATE.bias_orig)
        STATE.bias_orig = {}
        torch.cuda.synchronize()
        return n

    def beam_steer_stats(self):
        return STATE.hist.cpu().tolist()

    def beam_steer_reset_stats(self):
        STATE.hist.zero_()

    def beam_steer_capture_dump(self, path):
        """Save the rows captured since the last dump to `path` (first rank only) and start over.

        The file holds resid [layers, rows, hidden], router [layers, rows, experts], index [(capture_key, position,
        token_id, slot)] and dropped. Every rank sees the same batches, so every rank resets its own counters.
        """
        rows, dropped = STATE.cap_rows, STATE.cap_dropped
        torch.cuda.synchronize()
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            torch.save({"resid": STATE.resid[:, :rows].cpu(), "router": STATE.router[:, :rows].cpu(),
                        "index": STATE.cap_index, "dropped": dropped}, path)
        STATE.cap_rows, STATE.cap_index, STATE.cap_dropped = 0, [], 0
        return rows, dropped

    Worker.beam_steer_set = beam_steer_set
    Worker.beam_steer_set_route = beam_steer_set_route
    Worker.beam_steer_set_bias = beam_steer_set_bias
    Worker.beam_steer_reset_bias = beam_steer_reset_bias
    Worker.beam_steer_stats = beam_steer_stats
    Worker.beam_steer_reset_stats = beam_steer_reset_stats
    Worker.beam_steer_capture_dump = beam_steer_capture_dump


def hist_quantiles(hist, qs=(0.1, 0.5, 0.9)):
    """Per-layer quantiles of ‖residual‖ from a beam_steer_stats() histogram (bin centres)."""
    out = []
    centres = 2.0 ** (HIST_LO + (np.arange(HIST_BINS) + 0.5) / HIST_PER_OCTAVE)
    for row in np.asarray(hist):
        total = row.sum()
        if total == 0:
            out.append([float("nan")] * len(qs))
            continue
        cdf = np.cumsum(row) / total
        out.append([float(centres[min(np.searchsorted(cdf, q), HIST_BINS - 1)]) for q in qs])
    return out
