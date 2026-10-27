"""Per-request residual-stream steering and residual-norm stats for vLLM's Beam, as a vLLM general plugin.

Each request picks a steering slot through `SamplingParams(extra_args={"steer_slot": s})`; slot 0 is unsteered.
A slot holds one vector per layer, added to the residual stream leaving that layer (Lyra's `mlp_out` hook point:
after the MLP's post-norm, so the push is not renormalised away). Beam's MLP boundary is a fused
post-norm + depth-scaled residual add + next pre-norm,

    new_residual = a * post_norm(mlp_out) + b * residual,

so adding delta / b to `residual` just before it yields new_residual + delta, with the next layer's normalised
input computed from the steered residual.

Buffers are persistent and indexed in-graph, so CUDA graphs stay valid; the model runner writes each step's
per-token slots before the forward. Workers gain three RPCs for `LLM.collective_rpc`:
`beam_steer_set(path)`, `beam_steer_stats()` and `beam_steer_reset_stats()`.

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
    slots = None  # [max_tokens] int64, per-token slot for the current step
    valid = None  # [max_tokens] float32, 1 for real tokens, 0 for padding
    hist = None  # [num_layers, HIST_BINS] float32 counts of ‖residual‖ leaving each layer, positions > 0


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

    init = beam.BeamModel.__init__

    def model_init(self, *, vllm_config, prefix=""):
        init(self, vllm_config=vllm_config, prefix=prefix)
        config, sched = self.config, vllm_config.scheduler_config
        device = torch.device("cuda", torch.cuda.current_device())
        max_tokens = max(sched.max_num_batched_tokens, sched.max_num_seqs) + 1024
        slots = int(os.environ.get("BEAM_STEER_SLOTS", "512"))
        STATE.table = torch.zeros(config.num_hidden_layers, slots, config.hidden_size,
                                  dtype=vllm_config.model_config.dtype, device=device)
        STATE.slots = torch.zeros(max_tokens, dtype=torch.int64, device=device)
        STATE.valid = torch.zeros(max_tokens, dtype=torch.float32, device=device)
        STATE.hist = torch.zeros(config.num_hidden_layers, HIST_BINS, dtype=torch.float32, device=device)

    def layer_forward(self, positions, hidden_states, normalized_states=None, next_norm_weight=None):
        # Upstream BeamDecoderLayer.forward, plus the steering add before the MLP boundary and the norm stats.
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
        norm = hidden_states.float().norm(dim=-1).clamp_min(2.0**HIST_LO)
        bins = ((torch.log2(norm) - HIST_LO) * HIST_PER_OCTAVE).long().clamp(0, HIST_BINS - 1)
        weight = STATE.valid[:n] * (positions[:n] > 0).float()
        STATE.hist[self.layer_id].index_add_(0, bins, weight)
        return hidden_states, normalized

    beam.BeamModel.__init__ = model_init
    beam.BeamDecoderLayer.forward = layer_forward


def _patch_runner():
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    prepare = GPUModelRunner._prepare_inputs

    def prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        out = prepare(self, scheduler_output, num_scheduled_tokens)
        if STATE.slots is not None:
            num_reqs = self.input_batch.num_reqs
            per_req = []
            for req_id in self.input_batch.req_ids[:num_reqs]:
                params = self.requests[req_id].sampling_params
                per_req.append(int(((params and params.extra_args) or {}).get("steer_slot", 0)))
            slots = torch.zeros(STATE.slots.shape[0], dtype=torch.int64)
            valid = torch.zeros(STATE.valid.shape[0], dtype=torch.float32)
            n = int(num_scheduled_tokens[:num_reqs].sum())
            slots[:n] = torch.from_numpy(np.repeat(np.asarray(per_req, dtype=np.int64), num_scheduled_tokens[:num_reqs]))
            valid[:n] = 1
            # Blocking copies on the forward's stream: ordered after the previous step's kernels.
            STATE.slots.copy_(slots)
            STATE.valid.copy_(valid)
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

    def beam_steer_stats(self):
        return STATE.hist.cpu().tolist()

    def beam_steer_reset_stats(self):
        STATE.hist.zero_()

    Worker.beam_steer_set = beam_steer_set
    Worker.beam_steer_stats = beam_steer_stats
    Worker.beam_steer_reset_stats = beam_steer_reset_stats


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
