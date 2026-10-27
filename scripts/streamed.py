"""Run Beam one decoder layer at a time, streaming each layer's weights from the checkpoint and dropping them after.

Only one layer (about 10 GB in FP8, 20 GB once dequantised to BF16) is ever resident, so a single 48 GB GPU can push a
batch of sequences through the full 501B model; download speed, not compute, sets the pace. This is prefill only:
generation would re-stream every weight per token.

The layer math is transformers' own BeamDecoderLayer; this module only replaces BeamModel's loop. It loads each layer
from the checkpoint's per-expert FP8 tensors (per-output-channel scales) into transformers' fused expert layout, and
fetches layer L+1 in a background thread while layer L computes.

    beam = StreamedBeam(Checkpoint("reflection/Beam-1.0-501B-A23B-FP8"), config, device="cuda")
    out = beam.run([ids1, ids2], after_layer=hook)   # hook(L, i, h) may return a modified h (steering)
"""

import concurrent.futures as cf
import re

import torch
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.beam.modeling_beam import BeamDecoderLayer, BeamRotaryEmbedding, compute_attention_scaling

DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn}
EXPERT = re.compile(r"mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight")


class StreamedBeam:
    def __init__(self, ckpt, config, device="cuda", dtype=torch.bfloat16, workers=32):
        self.ck, self.config, self.device, self.dtype = ckpt, config, torch.device(device), dtype
        config._attn_implementation = "sdpa"
        self.pool = cf.ThreadPoolExecutor(workers)
        self.rotary = BeamRotaryEmbedding(config, device=self.device)
        self.embed = self.tensor("model.embed_tokens.weight")

    def fetch(self, names):
        """Raw bytes of the given tensors and their FP8 scales, fetched in parallel."""
        names = names + [n + "_scale" for n in names if n + "_scale" in self.ck.weight_map]
        return dict(zip(names, self.pool.map(self.ck.raw, names)))

    def materialise(self, raw, name):
        dtype, shape, data = raw[name]
        t = torch.frombuffer(bytearray(data), dtype=DTYPES[dtype]).view(shape).to(self.device)
        if name + "_scale" in raw:  # FP8 with one BF16 scale per output channel
            _, sshape, sdata = raw[name + "_scale"]
            s = torch.frombuffer(bytearray(sdata), dtype=torch.bfloat16).view(sshape).to(self.device)
            return (t.float() * s.float()).to(self.dtype)
        return t if t.dtype == torch.float32 else t.to(self.dtype)  # router weight and bias stay FP32, as trained

    def tensor(self, name):
        return self.materialise(self.fetch([name]), name)

    def layer_names(self, L):
        prefix = f"model.layers.{L}."
        return [n for n in self.ck.weight_map if n.startswith(prefix) and not n.endswith("_scale")]

    def build(self, L, raw):
        """A BeamDecoderLayer holding layer L's weights, per-expert tensors packed into the fused layout."""
        prefix, state, experts = f"model.layers.{L}.", {}, {}
        for name in list(raw):
            if name.endswith("_scale") or not name.startswith(prefix):
                continue
            key = name[len(prefix):]
            if m := EXPERT.fullmatch(key):
                experts[int(m[1]), m[2]] = name
            else:
                state[key] = self.materialise(raw, name)
        if experts:
            n_exp, inter, hidden = self.config.num_experts, self.config.moe_intermediate_size, self.config.hidden_size
            gate_up = torch.empty(n_exp, 2 * inter, hidden, dtype=self.dtype, device=self.device)
            down = torch.empty(n_exp, hidden, inter, dtype=self.dtype, device=self.device)
            for e in range(n_exp):  # BeamExperts splits gate_up_proj with chunk(2): gate rows first, then up
                gate_up[e, :inter] = self.materialise(raw, experts[e, "gate"])
                gate_up[e, inter:] = self.materialise(raw, experts[e, "up"])
                down[e] = self.materialise(raw, experts[e, "down"])
            state["mlp.experts.gate_up_proj"], state["mlp.experts.down_proj"] = gate_up, down
        with torch.device("meta"):
            layer = BeamDecoderLayer(self.config, L)
        layer.load_state_dict(state, strict=True, assign=True)
        return layer.eval()

    def inputs(self, ids):
        """Per-layer-type masks, rotary embeddings and NoPE query scales for one unpadded sequence."""
        pos = torch.arange(len(ids), device=self.device)[None]
        x = torch.empty(1, len(ids), 1, device=self.device, dtype=self.dtype)
        kw = dict(config=self.config, inputs_embeds=x, attention_mask=None, past_key_values=None, position_ids=pos)
        out = {}
        for kind in set(self.config.layer_types):
            mask = (create_causal_mask if kind == "full_attention" else create_sliding_window_causal_mask)(**kw)
            if self.config.rope_parameters[kind] is None:
                out[kind] = dict(attention_mask=mask, position_embeddings=None, attention_scaling=compute_attention_scaling(pos))
            else:
                out[kind] = dict(attention_mask=mask, position_embeddings=self.rotary(x, pos, kind), attention_scaling=None)
            out[kind]["position_ids"] = pos
        return out

    @torch.no_grad()
    def run(self, seqs, after_layer=None, layers=None):
        """Push token-id sequences through layers 0..n-1 and return the final hidden states (after model.norm).

        after_layer(L, i, h) sees sequence i's residual after layer L and may return a replacement (for steering).
        """
        layers = range(self.config.num_hidden_layers) if layers is None else layers
        h = [self.embed[torch.as_tensor(s, device=self.device)][None] for s in seqs]
        extra = [self.inputs(s) for s in seqs]
        pending = self.pool.submit(self.fetch, self.layer_names(layers[0]))
        for j, L in enumerate(layers):
            raw = pending.result()
            if j + 1 < len(layers):  # prefetch the next layer's bytes while this one computes
                pending = self.pool.submit(self.fetch, self.layer_names(layers[j + 1]))
            layer = self.build(L, raw)
            del raw
            kind = self.config.layer_types[L]
            for i in range(len(seqs)):
                h[i] = layer(h[i], **extra[i][kind])
                if after_layer is not None and (new := after_layer(L, i, h[i])) is not None:
                    h[i] = new
            del layer
        norm = self.tensor("model.norm.weight")
        return [(x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.config.rms_norm_eps)
                 * norm.float()).to(self.dtype) for x in h]

    @torch.no_grad()
    def logits(self, hidden, chunk=512):
        """Softcapped float32 logits for final hidden states [1, T, H], computed in chunks of positions."""
        if not hasattr(self, "lm_head"):
            self.lm_head = self.tensor("lm_head.weight")
        cap = self.config.final_logit_softcapping
        for s in range(0, hidden.shape[1], chunk):
            z = (hidden[0, s:s + chunk] @ self.lm_head.T).float()
            yield s, torch.tanh(z / cap) * cap if cap else z
