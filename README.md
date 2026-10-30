---
title: Inside Beam
emoji: 🔦
colorFrom: indigo
colorTo: gray
sdk: static
app_file: explorer/index.html
pinned: false
license: apache-2.0
short_description: Beam names its experts; banning them removes that knowledge
models:
  - reflection/Beam-1.0-501B-A23B
  - reflection/Beam-1.0-501B-A23B-NVFP4
datasets:
  - rycerzes/beam-expert-atlas
tags:
  - interpretability
  - mixture-of-experts
  - steering
  - vllm
---

# Inside Beam

A map of all 12,544 routed experts in [Beam-1.0-501B-A23B](https://huggingface.co/reflection/Beam-1.0-501B-A23B),
built by pushing each one inside the running model and checking what changes against random controls, then labelled
by Beam itself. Banning the experts Beam labelled *law* drops MMLU-Pro law by 13 points and leaves every other
subject where it was.

- **Explorer:** this Space (search any expert, read Beam's label for it and what Beam says when it is pushed)
- **Data:** [rycerzes/beam-expert-atlas](https://huggingface.co/datasets/rycerzes/beam-expert-atlas)
- **Write-up:** [rycerz.es/blog/inside-beam](https://rycerz.es/blog/inside-beam)
- **Research log:** [research_log.md](research_log.md), dated entries including what failed and what we corrected
- **Code:** this repo, mirrored at [github.com/rycerzes/beam-inside](https://github.com/rycerzes/beam-inside); the
  vLLM plugin is [`steer/`](steer/)

Built for Reflection's [Build with Beam](https://huggingface.co/reflection/Beam-1.0-501B-A23B/discussions/5) contest.

## What we found

Every number below is against a matched control. The details, and what did not work, are in the write-up.

**1. Beam's own labels for its experts are causally right.** We banned the experts Beam labelled with a subject
(exactly, through the router's correction bias) and scored all 12,032 MMLU-Pro questions, paired against a ban of
the same number of random experts with matched routing load. Base accuracy is 62.3% (no thinking, letter scoring).

| Experts banned | Their own subject | Every other subject | Random ban of the same size |
|---|---|---|---|
| law (143) | **−13.4 pts** (p = 1e-21) | +0.2 (ns) | −0.4 |
| health (165) | **−6.5 pts** (p = 4e-8) | −0.1 (ns) | −0.2 |
| math (588) | **−8.7 pts** (p = 3e-14) | physics −9.2, chemistry −7.9, economics −6.2, business −4.8; verbal subjects ±1 | −0.3 |

The "math" experts carry quantitative work across every number-heavy subject, not math as a topic.

**2. Steering finds real units, mostly in the middle of the network.** We pushed each expert's main write directions
into the residual stream of the live model at calibrated doses (12,920 units, 832 of them random-direction
controls) and asked whether the outputs drift toward a theme more than controls that move the model by the same
amount. 2,092 expert directions pass, against 1,128 expected by chance; they concentrate in layers 24–39. A blind
judge given three steered outputs picks the right one from the label (the tokens the pushed direction promotes) 70%
of the time for those directions (chance 33%, random directions 49%), and 98.7% of the time on the 670 items it
marked as clear matches. Given a label from the other side of the same direction, it is at chance (35%).

**3. Beam knows which of its experts are readable.** Beam named 10,926 expert directions from their top tokens and
flagged whether each label was coherent. Coherent ones pass the steering test 3.5x as often (19.5% vs 5.6%), although
the flag never saw any steering. Coherence rises with depth, from 29% in layers 0–7 to 90% in layers 32–39. With
Beam's labels in place of raw tokens, the judge scores 71% (vs 49% for random directions).

**4. The reasoning-effort setting is a number Beam reads early and clamps.** A single direction reads the effort N
from the residual at N's last digit with R² = 0.999 (layer 12). Effort values outside the trained 8–512 range read as
clamped (N = 2 reads as ~13). On 100 AIME 2024/2025 and AMC 2023 problems at effort 8, adding the difference
between the digit's mean representation at N = 512 and at N = 8 lengthens thinking 1.2x relative to a random vector
of the same norm (p = 5e-8). Low effort is fragile: re-routing 37 "effort experts" on the prompt (19 forced in, 18
banned) or 37 random experts both lift effort-8 accuracy from 80% to 91–94%, the same as raising the dial to 32, so
that gain is not effort-specific.

**5. REAP pruning, emulated.** Banning the 128 experts per layer that the community
[REAP-255B](https://huggingface.co/mrfakename/Beam-1.0-REAP-255B-A23B) prune removed costs 15 points overall: health
−32, history −23, philosophy −21, law −20, engineering only −5. This measures what a pruning plan costs on the full
model before anyone builds the pruned checkpoint.

## How Beam was used

- **As the subject.** Every intervention runs inside the real 501B model: the NVFP4 release in vLLM on 4x RTX PRO
  6000 (HF Jobs), with our plugin adding per-request residual pushes, router-logit changes, exact expert bans and
  activation capture. About 7 hours on 4 GPUs in total.
- **As the labeller.** Beam named 10,926 of its own expert directions (name, description, subject, coherent or not)
  through HF Inference Providers, and those labels picked the experts for the ban experiments.
- **As the measuring stick.** Its effort dial, its default identity line and its answers on MMLU-Pro and competition
  math are the outcomes.

## Tools you can use

- **[`beam-steer`](steer/), a vLLM plugin for Beam.** Per-request residual steering, per-request routing changes,
  exact expert bans for the whole batch, and capture of every layer's residual and router logits at chosen
  positions. Checked against a transformers reference on a tiny Beam built from the real config (below).
- **A pruning check.** Ban any set of experts and score it on MMLU-Pro in about 6 minutes per arm on 4 GPUs, before
  building a pruned checkpoint. Arms are lists of `[layer, expert]` ([`scripts/mmlu_arms.py`](scripts/mmlu_arms.py)).
- **The atlas.** Per expert direction: top tokens, Beam's label and subject, steering significance, judge verdict,
  whether REAP kept it, and Beam's steered outputs ([dataset](https://huggingface.co/datasets/rycerzes/beam-expert-atlas)).
- **Effort activations.** Residuals at every layer and router logits for 200 GSM8K questions at 9 effort settings
  (5 positions each), so effort probes can be trained without running Beam.

## The plugin

Install it into the environment vLLM runs in:

```bash
uv pip install "beam-steer @ git+https://github.com/rycerzes/beam-inside#subdirectory=steer"
# inside the vllm/vllm-openai image, which has pip:
pip install "beam-steer @ git+https://github.com/rycerzes/beam-inside#subdirectory=steer"
```

vLLM finds it through the `vllm.general_plugins` entry point. Beam support landed in vLLM main after 0.32.0, so the
plugin is written and tested against `vllm/vllm-openai:nightly-5d35ce3934a7804b298559d2d99eec33792e1eb9`; later
versions are untested. It needs `VLLM_USE_V2_MODEL_RUNNER=0`, `async_scheduling=False` and
`enable_prefix_caching=False` (cached prompt tokens skip the forward, so they would be neither steered nor captured).

```python
import os
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
import torch
from vllm import LLM, SamplingParams

llm = LLM("reflection/Beam-1.0-501B-A23B-NVFP4", tensor_parallel_size=4, moe_backend="marlin",
          async_scheduling=False, enable_prefix_caching=False)

# Steering: slot 1 adds `delta` to the residual leaving layer 30 (slot 0 is always unsteered).
delta = torch.randn(5120) * 2.0
torch.save([(1, 30, delta)], "/tmp/steer.pt")
llm.collective_rpc("beam_steer_set", args=("/tmp/steer.pt",))
out = llm.generate(["..."], SamplingParams(max_tokens=200, extra_args={"steer_slot": 1}))

# Exact expert bans, for every request until reset: lower the routing correction bias.
torch.save([(30, 222, -1000.0), (31, 17, -1000.0)], "/tmp/bans.pt")
llm.collective_rpc("beam_steer_set_bias", args=("/tmp/bans.pt",))
llm.collective_rpc("beam_steer_reset_bias")
```

| RPC / request key | What it does |
|---|---|
| `beam_steer_set(path)` | load `[(slot, layer, delta)]`: residual pushes per slot |
| `beam_steer_set_route(path)` | load `[(slot, layer, expert, delta)]`: router-logit changes per slot (force or discourage experts) |
| `beam_steer_set_bias(path)` / `beam_steer_reset_bias()` | `[(layer, expert, delta)]` added to the correction bias for the whole batch; −1000 is an exact ban |
| `beam_steer_capture_dump(path)` | save captured residuals `[layers, rows, hidden]`, router logits and an index (needs `BEAM_STEER_CAPTURE_ROWS`) |
| `extra_args`: `steer_slot`, `steer_from`, `steer_until`, `steer_skip_ids`, `capture`, `capture_key` | per request: which slot, which positions, what to capture |

Why bias bans, not logit bans: Beam picks experts by `sigmoid(logit) + e_score_correction_bias`, and that bias is
18–28 with a spread of at most 0.02 within a layer, so a −30 logit does not guarantee a ban. The bias only enters the
choice, never the routing weights, which is what slicing an expert out of the router (as REAP does) amounts to.

**Test.** [`scripts/vllm_tiny_check.py`](scripts/vllm_tiny_check.py) builds a tiny random Beam from the real
`config.json` in the Hub's FP8, BF16 and NVFP4 formats, runs it through transformers and through vLLM with the plugin,
and holds every case (steering slots, windows, skip ids, routing, bias bans and their exact restore, capture) to the
reference within the BF16 floor, in eager mode and with CUDA graphs. `uv run scripts/launch_tiny.py gpu` runs it on
one RTX PRO 6000 for about $0.30.

**Related tools** (as of 2026-10-29): upstream vLLM has no steering API. vllm-lens (UK AISI) reads activations in
vLLM; EasySteer steers residuals and routing but is pinned to vLLM 0.29, before Beam; the RhizoNymph steering fork
has no Beam support. SteerMoE (routing-logit steering) and MoTE (expert switching on DeepSeek-R1) are methods this
builds on.

## Reproduce

Everything ran from this repo on HF Jobs; [`scripts/watch_job.py`](scripts/watch_job.py) relays progress and spend.

| Step | Command | Hardware, time | Cost |
|---|---|---|---|
| Expert write directions from the FP8 weights (range reads, no full download) | `uv run scripts/atlas_experts.py` → `atlas_lens.py` | CPU | — |
| Pick units, directions and controls | `uv run scripts/sweep_units.py` | CPU | — |
| Plugin check on a tiny Beam | `uv run scripts/launch_tiny.py gpu` | 1x RTX PRO 6000, 10 min | $0.30 |
| Calibration, effort capture, full sweep | `uv run scripts/launch_session.py real --repo USER/runs --hours 3.5` | 4x RTX PRO 6000, 3.1 h | $34.28 |
| Rank against effect-matched controls | `uv run scripts/rank_sweep.py RUN` → `effect_rank.py RUN` | CPU | — |
| Effort analysis and competition-math arms | `effort_analysis.py`, `effort2_inputs.py`, session phase `effort2` | 4x RTX PRO 6000, 2.5 h incl. a cancelled first attempt | $28.05 |
| Blind three-way judge (DeepSeek-V4-Flash) | `judge_items.py` → `judge_router.py` → `judge_score.py` | HF Inference Providers | $0.88 |
| Beam labels its experts | `pole_rows.py` → `self_label.py` → `self_label_score.py` | HF Inference Providers (Baseten) | $38.07 |
| REAP cross-check, ban arms, MMLU-Pro | `reap_crosscheck.py`, `mmlu_arms.py`, session phase `mmlu`, `mmlu_score.py` | 4x RTX PRO 6000, 1 h | $10.87 |
| Public dataset and explorer files | `uv run scripts/export_atlas.py OUT` and `export_captures.py` | CPU | — |

Total spend about $120: ~$81 of GPU time (including an earlier pilot sweep, rehearsals and one cancelled run), $38
of Beam inference and under $1 of judge inference. Run every `uv` command with `UV_NO_CONFIG=1` if your machine has
a global index configured.

<details><summary>What each script does</summary>

| Script | Purpose |
|---|---|
| `fetch_tensors.py`, `ckpt.py` | read individual tensors from a Hub safetensors checkpoint with HTTP range requests |
| `atlas_experts.py`, `atlas_lens.py`, `atlas_dense.py`, `atlas_router.py` | weight-only atlas: experts' top write directions, their logit-lens tokens, dense neurons, router geometry |
| `streamed.py`, `check_tiny.py`, `speedtest.py` | layer-streamed Beam forward (prefill on one GPU) and its tiny-model check; Hub read speed |
| `nvfp4_compare.py` | FP8 vs NVFP4 expert weights: write directions transfer (\|cos\| ≥ 0.997) |
| `sweep_units.py` | units for the sweep: expert directions, dense neurons, random controls |
| `session.py` | the GPU session: sanity, residual stats, calibration, effort, sweep, effort2, mmlu phases |
| `launch_session.py`, `launch_tiny.py`, `watch_job.py` | HF Jobs launchers and spend watcher |
| `vllm_tiny_check.py` | plugin correctness against transformers on a tiny Beam |
| `rank_sweep.py`, `effect_rank.py` | theme scores vs band controls, then vs effect-matched controls |
| `effort_analysis.py`, `effort2_inputs.py` | effort readout, off-dial reads, effort experts; competition-math arm inputs |
| `judge_items.py`, `judge_router.py`, `judge_score.py`, `label_items.py` | blind three-way label test |
| `pole_rows.py`, `self_label.py`, `self_label_score.py` | Beam labels its own expert directions; scoring |
| `reap_crosscheck.py`, `mmlu_arms.py`, `mmlu_score.py` | REAP kept-expert analysis, ban arms, paired MMLU-Pro scoring |
| `export_atlas.py`, `export_captures.py` | public dataset tables, explorer JSON, activation captures |

</details>

## Limitations

- One greedy sample per prompt and problem; the competition-math arms are 100 problems.
- Early layers (0–23) are underdosed in effect terms: their few "significant" directions are at chance with the judge,
  so treat them as false positives of the theme metric. Late layers (48+) collapse into repetition at the doses
  needed, and 46% of their directions have no fair control.
- An earlier pilot sweep (not used for the numbers above, not in the dataset) ran with vLLM's prefix caching on, so
  it steered only from the assistant turn onwards.
- 3 of 79 sweep waves were lost to the job's time limit: 478 experts have no steering data.
- Beam's labels describe what an expert writes, read from its top tokens, not when the router uses it.
- Labels and the judge test cover the + side of each direction (the side the sweep's lens tokens describe). A first
  pass also labelled and judged − pushes, but with the + side's tokens; those results are excluded (they sit at
  chance, as a mismatched label should), and about half of the labelling spend went to them.
- MMLU-Pro is scored without thinking, so absolute accuracy (62.3%) is below Beam's reported score with thinking.
- The REAP emulation bans REAP's pruned experts in the NVFP4 model; we did not run the REAP checkpoint itself.
- None of this was pre-registered.

## Credits

Method: Lyra's [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer) (steering-first labelling with
random controls), SteerMoE, and Li et al. (arXiv 2511.08579) on self-explanation. REAP kept-expert lists from
[mrfakename/Beam-1.0-REAP-255B-A23B](https://huggingface.co/mrfakename/Beam-1.0-REAP-255B-A23B); per-subject REAP
numbers from peterhdd's thread entry. Static weight analysis of Beam:
[Barneyjm/beam-teardown](https://github.com/Barneyjm/beam-teardown).

Code is Apache-2.0. `meta/config.json` and `meta/chat_template.jinja` are Reflection's, from the Beam model repo
(Apache-2.0).
