# Research log: Inside Beam — steering, labelling and banning the experts of a 501B MoE

## Framing (2026-10-26)

**Question.** What do the 12,544 routed experts of Beam-1.0-501B-A23B write, and can the model's own behaviour
(steered generations, expert bans) confirm a label for each one? A second thread: how is the integer reasoning-effort
dial represented inside the model, and can it be moved from inside?

**Why.** The entry started as a harness study of Beam's ~3% on Terminal-Bench 4.0 (first entry below). That turned
out to be a real result rather than a harness bug, and a diagnosis alone looked unlikely to place. A census of ~30
contest entries found benches, effort sweeps, quantizations, distillations and safety audits, all calling Beam from
outside. None looked inside the model. That lane was empty.

**Model facts that shaped the design.** 52 layers: 3 dense SwiGLU (intermediate 20,480), then 49 MoE layers of 256
experts, top-8, no shared expert (moe_intermediate 2,560, hidden 5,120). Router: sigmoid scores plus an
`e_score_correction_bias` used for selection only; grouped top-k (8 groups of 32, top 4 groups by the sum of their
top-2). Sandwich norms and a depth-scaled residual, `h/sqrt(k) + res*sqrt((k-1)/k)`. Sliding window 512 on most
layers, NoPE global layers, logit softcap 30, Qwen vocab (151,936), Harmony format with analysis/final channels.
Effort is a line in the system prompt, "Reasoning effort: N", trained labels 8/32/64/128/256/512, default 512.

**Design, as it ended up.** After Lyra's neuron-explorer: the unit is an expert's top write direction (left singular
vector of W_down, both poles), pushed into the residual stream of the live model, with greedy rollouts on fixed
prompts and random-direction controls at the same depth. Labels come first from a weights-only logit lens, later from
Beam itself; a blind three-way judge tests whether a label predicts its unit's steered output; exact expert bans on
MMLU-Pro test whether a label predicts what the model loses without the expert.

**Infra.** Upstream vLLM main (Beam merged after 0.32.0, nightly image at commit 5d35ce3) plus our own general plugin;
the NVFP4 checkpoint on 4x RTX PRO 6000 (HF Jobs `rtx-pro-6000x4`, $11/h) with the Marlin W4A16 MoE kernel. Beam
inference for labels through HF Inference Providers (Baseten). Every paid run quoted first, capped, and costed from
the billing API afterwards; nothing published without an explicit OK.

## 2026-10-26 — Terminal-Bench 4.0: why Beam scores ~3% with mini-swe-agent

**Wire probes first.** Through a logging proxy: mini-swe-agent 2.4.6 stores `message.model_dump()` and replays
`reasoning_content` on every turn (checked with litellm 1.100.5, 1.103.4, 1.104.2, 1.106.4). That matters because
Baseten, the only live provider on the HF router, serves the template with `allow_missing_thinking` false and returns
400 when a prior assistant turn lacks its reasoning; a harness that drops it would crash every task at its second
call. String vs dict tool arguments made no visible difference (8 vs 8 repeats); the first single sample that
suggested otherwise was noise.

**Effort unset = max, not medium.** mini-swe-agent sends no `reasoning_effort` and no `max_tokens`. On one AIME
problem, median reasoning tokens: unset 22.7k, max 19.9k, medium 13.9k, low 2.4k. The HF blog says the default is
medium; the template and docs say 512. The template also renders every prior turn's reasoning in full, so at max
effort about half of each final context is old reasoning (46–58% in the first batch).

**Pre-registered slice.** 12 single-container CPU tasks on HF Sandbox (`cpu-upgrade`), arms unset/low/high, 3 repeats,
2 h agent timeout. Deviations logged before analysis: `risk-scorer-replay` excluded (its image runs as `USER nobody`
and Harbor installs the agent with apt as the image user); the high arm dropped for scope before any low or high
trial ran; the low arm run at concurrency 6 after the unset arm's throttling; a classifier fix so a verifier timeout
is not counted as a provider error.

**Results (11 tasks × 3).** Unset: 2/32 passed (6.2%), both on one task; 21 false submits (agent declared done, hidden
tests failed), 7 trials ended by Baseten 503/429 at concurrency 12 (22%), median 38 steps, 11.5 min, $0.23 per
trial. Low: 0/33, 31 false submits, median 13 steps and 0.4 min, $0.013 per trial. Blind claimed/flagged labels:
unset 17/4, low 20/11. At max effort Beam mostly believes it has finished; at low effort a third of its failures are
knowing give-ups ("I'll submit current result; even though separation errors remain"). The Wilcoxon test had nothing
to work with (one task differs between arms), so effort is reported descriptively. No length cutoffs, format errors
or refusals. Side bugs: huggingface_hub job names over 100 chars for digest-pinned images; Harbor's hf-sandbox
installs the agent as the image user; Harbor's podman backend breaks on the compose banner; litellm 1.106.4 fails to
import without `mcp`.

**Cost.** $8.79 Baseten + $0.71 HF Jobs. An earlier, unbilled estimate of $100–150 for the full grid caused alarm;
from here every cost is quoted from billing and every spend paused for an OK.

**Why we pivoted.** The 3% is real on our slice, so the entry would be "Beam fails agent tasks", a complaint rather
than a tool; a harness-fix follow-up was proposed and rejected as unlikely to win. The census showed the crowded
lanes (single-turn effort sweeps, cost per token, 2-bit quants, distillation into small models, safety benches) and
the empty one: nothing inside the model.

## 2026-10-26 — a weight-only atlas from range reads

**Fetching without downloading.** A small `Checkpoint` class reads the safetensors header and single tensors by HTTP
range request (38,366 tensors). Routers, norms, dense layers, embeddings and lm_head came to
5.26 GB. Expert `down_proj` weights were streamed from the FP8 checkpoint (per-output-channel BF16 scales) and reduced
to the top 8 left singular vectors per expert (`svd_lowrank`), ~40 MB per layer instead of 3.4 GB. Locally the link
capped at 11–12 MB/s whether 8 or 32 workers, ~4.7 min per layer, ~3.7 h for all layers, free.

**Expert output lens.** Direction × (final norm ⊙ post-FF norm gain) against a centred lm_head, each token
standardized across the layer's directions. L46 reads as clean multilingual concepts: e0 frame (帧, フレ, рам), e9 say
(說, nói, sagt, decir), e11 print (печ, 打印), e16 the "that" complementizer across languages, e22 hematology. L51:
adult, process (процесс, proceso), partner, end (конце, 結束), freeze, uppercase constants, brackets, C macro
suffixes; some poles whitespace noise. Experts are high-rank (top-1 singular energy share median 0.016 at L46, 0.025
at L51), so the lens shows dominant writes only.

**Router.** The raw router lens was dominated by a shared direction (mean cosine 0.4–0.7 between experts); centring
the rows fixed it. Effective rank 78–148; within-group cosine barely above between-group. The correction bias is a
near-constant per layer (+28.4 at L3, ~18.4–19 in the L30s, +24.8 at L51) with per-expert deviations of 1e-5 to
1e-2. Hypothesis, never tested: strongly negative deviations mark experts that were overloaded in training (L46 e23
−0.021, L51 e19 −0.017). The bias magnitude came back on 10-29 as the reason a logit ban is not a ban. **What
failed:** the dense-neuron output lens was dominated by artifact tokens ("(", curly quotes, newline, nbsp) and
under-trained tokens; early dense layers are not output-predictive. Left unfixed; the dense layers got 48 neurons in
the sweep and nothing more.

## 2026-10-27 — StreamedBeam, and why it is mostly sunk

To avoid a $40/h h200x8 session, we built a layer-streamed forward: transformers' own `BeamDecoderLayer`, one layer
resident at a time, FP8 weights fetched by range read, the next layer prefetched while the current one computes,
with a hook after any layer for capture or steering. Checked against a tiny FP8 model. A speed test on HF Jobs
(`cpu-basic`, 3.6 min, ~$0.001) measured range reads at 220–260 MB/s, best at 32 workers; the full FP8 model is 503 GB,
so one pass is ~35 min, ~$1–1.50 on an L40S.

**The flaw.** Every pass streams all 503 GB, so each generated token costs ~35 min. Fine for prefill captures,
useless for steered generation, which is what the references (Goodfire's "already clean" essay, Lyra's explorer)
actually do. On re-reading them, the weights-only lens was the weaker method and a prefill-only effort probe never
produces text. We dropped StreamedBeam and moved to steering the full model in vLLM.

## 2026-10-27 — a steering plugin for vLLM's Beam

**Where to push.** Beam support exists only on vLLM main (after 0.32.0); the RhizoNymph steering fork has no Beam.
vLLM's decoder layer fuses post-norm, the depth-scaled residual add and the next pre-norm into one `mlp_boundary`:
`new_residual = a * post_norm(mlp_out) + b * residual`. Adding `delta / b` to `residual` just before it yields
`new_residual + delta`, the analogue of Lyra's `mlp_out` hook: the push is not renormalised away, and the next
layer's normalised input is computed from the steered residual.

**Plugin v1.** A vLLM general plugin (entry point, no fork): a persistent `[layers, slots, hidden]` table indexed per
token inside the graph, so CUDA graphs stay valid; the model runner writes each step's per-token slot from
`SamplingParams.extra_args["steer_slot"]`; slot 0 is unsteered; a log-binned histogram of residual norms gives the
dose unit. Needs `VLLM_USE_V2_MODEL_RUNNER=0` and `async_scheduling=False`.

**Tiny-model check.** A random Beam built from the real config (hidden 512, 8 layers with 3 dense, 16 experts top-4)
in FP8 and BF16, held to a transformers reference on 1x RTX PRO 6000 (sm120). Steered requests matched their own
reference to 0.016–0.028 nats vs 0.28–0.42 from the unsteered one; alone vs mixed batch 0.0000; teacher-forced decode
0.012–0.032; norm stats within 4%. The first two runs printed MISMATCH: our fixed 0.03 threshold was tighter than the
BF16 floor itself (the reference in BF16 differs from FP32 by 0.10–0.16), so tolerances moved to the BF16 floor.
Spend $0.78. Gotchas: vLLM spawns engine processes that re-import the script, so the first session rehearsal died with "Engine
core initialization failed" until the body went under `main()`. HF Jobs overrides the image entrypoint
(`vllm serve`), and job names must be explicit.

## 2026-10-27 — NVFP4 vs FP8: the directions transfer, the W4A4 kernel does not

We first argued for FP8 on 8 GPUs (fidelity; atlas built from FP8). On a challenge, we measured instead. In the NVFP4
release only routed experts are FP4 (ModelOpt: packed E2M1, F8 group-16 scales, a scalar scale); dense layers,
attention and routers stay BF16; ~294 GB, so it fits 4 GPUs. Dequantizing 48 expert `down_proj` weights (16 each at
L10, L27, L46): ~9% elementwise difference (correlation 0.996) but top-1..8 write directions |cos| 0.997–1.000 and
top-8 subspace overlap 0.998. The atlas and units transfer with no rebuild.

**Kernel choice matters for steering.** A tiny NVFP4 checkpoint and rehearsal: Marlin (W4A16) matched the NVFP4
reference at 0.014–0.038 against a BF16 floor of 0.11–0.15; its alone-vs-mixed isolation was 0.008, batch-shape
numerics rather than a leak (a leak looks like 0.3–0.4). The default "auto" kernel (W4A4) moved unsteered rows by up
to 0.45 nats depending on what else shared the batch. Cause unresolved; W4A4 is unusable for mixed-slot batches, so
every real run used Marlin. GPU spend to date $1.33.

## 2026-10-27 — v1 sweep: 1,640 units on the real model

**Units.** 24 experts per MoE layer chosen by their strongest lens pole (1,176), 48 dense neurons, 416 Gaussian
controls (8 per layer). Doses −0.7/+0.5/+0.8 × c_L × the layer's median residual norm on who/snake/orange (160
greedy tokens), and +0.8/−0.7 on an effort prompt at effort 32. A sanity gate aborts unless the first answer says
Paris. Calibration by coherence (distinct-4-gram) on random directions: Beam breaks at 0.5× in L2–24 (c = 0.31), 0.62
in L28–44, 1.0 from L48.

**Run.** Job 6ae0672e: 22 min, $4.22, load 5.5 min, ~3.3k tok/s. 374 of 2,448 poles above their band's control p95,
against 122 by chance. By depth: 0% in L0–23 (doses scaled down there), 32% L24–31, 28% L32–39, 16% L40–47, 41%
L48–51. The lens pole is significant 30% of the time, the opposite pole 2%; 76% of significant poles share an
English stem with their own lens tokens vs 1% shuffled. L28 e222 (church) answers "I am a Christian…"; L27 e173
(distance) reasons about distance. Expert pushes dropped Beam's identity in 0.5–2.1% of rollouts, controls 0%. We
framed that as Beam treating identity as a "tier-1 rule"; on 10-29 that phrase turned out to be unsourced.

On the effort prompt, concept pushes shorten thinking (baseline 194 tokens) about as much as controls do; the extremes are
concept intrusions and format breaks, not effort control. A dedicated effort probe was still needed.

## 2026-10-27 — prefix caching, and plugin v2

**The v1 run steered less than we thought.** vLLM enables prefix caching by default (block 16). Our prompts were
48–51 tokens and the unsteered sanity phase had already cached them clean, so every steered request reused the cached
prompt KV: steering reached only the last 1–3 prompt tokens plus the generation, and position 0 was never steered.
Units and controls were treated alike, so v1 still reads as "steering from the assistant turn onward" (a CAA-like
setup), and it is disclosed as such. It also killed a hypothesis: early-layer fragility could not be an attention-sink
artifact, because the sink was never pushed. Sessions now pass `enable_prefix_caching=False`.

**Plugin v2.** Per-request `steer_from` / `steer_until` / `steer_skip_ids`; capture of the residual leaving every
layer and the router logits at every MoE layer, at named positions, across chunked prefill and on generated tokens.
Rehearsal on BF16 eager, BF16 graphs and NVFP4 Marlin: steering ok on every slot, window and skip case; isolation
0.0000; captured residuals 0.023–0.026 relative error and router logits 0.024–0.026, against BF16 noise of ~0.16 and
~0.19. One FAIL (28/30 captures) was our test listing the same position twice. $0.32; GPU to date $5.87.

**Literature that changed Job 1.** KL/ΔNLL-calibrated doses instead of norm fractions; effect-matched controls; the
three-way output-centric label test (arXiv 2501.08319, chance 1/3); "Beyond the Trace" (2608.17638) reading gpt-oss's
effort setting from routing; "The Expert Strikes Back" (2604.02178) labelling experts by logit attribution on up to
8 layers of ≤30B models, with no steering. Beam's 8/256 routing is sparser than anything they tested.

## 2026-10-27 — Job 1: calibration, effort capture, every expert

**Design.** Session v2 on 4x RTX PRO 6000, NVFP4/Marlin, 3.5 h cap ($38.50). Calibration at every layer: 8 random
directions × doses 0.05–1.2 × median norm, measuring ΔNLL of Beam's own unsteered continuations; τ* is the median
over layers of the ΔNLL at the largest coherent dose, and each layer's sweep doses map to 0.4τ* and τ*. Effort: GSM8K
at N = 8…512 plus 2 and 1000, capturing five positions (last digit of N, end of system message, think start, first
two thought tokens). Sweep: all 12,544 experts (top direction, both poles), 48 dense neurons, 832 controls.

**Run.** 187 min, $34.28 (GPU total $40.71). 76 of 79 waves before the time limit: 12,920 units, 478 experts with no
steering data.

**The calibration is a cliff.** Below a threshold Beam barely reacts (<0.05 nats/token); just past it, output collapses
(1.5–3.5 nats, repetition). Median ΔNLL at L2: 0.02 at 0.2×, 1.57 at 0.35×. L12: 0.42 at 0.35×, 0.89 at 0.5×. L30:
0.37 at 0.5×, 1.57 at 0.8×. L48: 0.14 at 0.8×, 0.63 at 1.2×. The cliff moves deeper with depth (~0.3× in L2–24,
0.5–0.8× in L28–44, ≥1.2× from L48), the same pattern as v1, so not a caching artifact. τ* = 0.38.

**Expert directions are not random directions.** At equal norm, expert pushes moved L0–19 5–40× less than random ones
(ΔNLL 0.01–0.09 vs 0.2–0.4), about equally at L24–31 (~0.35 vs ~0.42), and L48–51 6–10× more (1.3–1.8 vs 0.2–0.3,
collapse into repetition). Early experts were underdosed in effect terms; late ones were overdosed.

**First ranking (band controls).** 1,324 hits vs 1,211 by chance overall, diluted by empty early layers: 17.5% of
experts in L24–31 (38% of the 24 per layer with the strongest lens), 7.8% L32–39, 4.7% L40–47, 3.7% L48+ (v1 had 41%
there with gentler doses), ~0–1.6% L0–23. L29 e25 answers "Who are you?" in Vietnamese; L26 e174 (COVID) becomes a
COVID-19 assistant and frames the orange as COVID safety; L27 e187 (error) hallucinates "Error: An error occurred.
Please check the traceback".

**Effort, first pass.** One direction reads log₂ N with R² 0.98 at L19–20. GSM8K was the wrong task: the dial saturates
at N = 64 (thinking 108 → 272 tokens). A ±3-octave push along the direction at L26/L36 did nothing, but it was only
0.12× the median norm. Expert pushes in the sweep's effort prompt swing thinking 72–512 tokens (baseline 183), so
length is steerable in general; whether the effort direction does it specifically stayed open.

## 2026-10-27 — free analysis: effect-matched null, effort readout, effort experts

**Why re-rank.** Control theme scores rise with ΔNLL, and only 34% of experts' ΔNLL values fall inside the controls'
5–95% range (experts 0.01–0.99, controls 0.16–0.68). Comparing an expert with norm-matched controls is unfair in both
directions. `effect_rank.py`: a pole's null is the controls within ±8 layers whose ΔNLL is within 1.5×, falling back
to the weakest controls for weak experts; late-layer experts stronger than every nearby control are "untestable".

**Result.** 2,092 significant poles vs 1,132 by chance. L0–7 0.1%, L8–15 3.2%, L16–23 5.2%, L24–31 16.6%, **L32–39
20.4%** (7.8% under the band null), L40–47 7.5%, L48–51 4.8% with about half testable. 33% of hits share a stem with
their lens label vs 2% shuffled (as reported then; split by pole on 10-30, + poles 42% vs 2%, − poles 1%, because − poles
were compared with the + side's tokens). Concept experts sit in L24–39; L0–7 is untestable at these doses rather than shown
empty.

**Effort readout by position.** Held-out R² of log₂ N: last digit of N 0.999 (L12; that is mostly the token itself),
end of system message 0.985 (L33), think start 0.981 (L20), first thought tokens 0.50–0.64: the signal fades once
thinking begins. **Off the dial, values clamp.** N = 2 reads as ~12–14, N = 1000 as ~300–450, matching behaviour
(N = 2 thought 170 tokens, between N = 8's 108 and N = 16's 167). Beam reads the number as a position on a learned
scale, not as arithmetic.

**Effort experts.** At the think start the token is identical across N, so routing differences come from the setting
alone. Recomputing Beam's grouped selection from captured router logits, 192 experts change selection rate between
N = 8 and 512 beyond the shuffled-label p95; many are all-or-nothing switches in L15–22 (L21 e62 0% → 100%, L18 e97
100% → 0%, L22 e78 99% → 1%). Their lens labels are noise (code punctuation, formatting), and pushes along their write
directions in the sweep had no consistent sign on thinking length. A push along an expert's direction is not the same
as switching the expert on in the router, which led to routing control.

## 2026-10-27/28 — judge pilot, plugin v3, Job 2 inputs

**Judge pilot.** 200 three-way items (100 significant, 100 random testable poles; two effect-matched distractors from
the same band with no shared stem; chance 33%). Reference judges: Sonnet 5.5 subagents, 4 × 50 items. Significant
78% [70–86], random 46% [36–56]; confidence-3 answers 100% / 93% correct; by depth L0–23 42%, L24–39 72%, L40–51 62%.
Qwen3.8-Flash-Next thinks by default and starved: at a 400-token cap 135/200 came back empty, at 4,096 still 74; the
empty ones were the ambiguous items (Sonnet 36% on them). Cost $0.295, over our ~$0.25 note. DeepSeek-V4-Flash on
DeepInfra: 200/200 parsed, no thinking, significant 69% [60–78], random 50%, confidence-3 100% / 86%, 58% agreement
with Sonnet, $0.014. Caching barely applied (256 tokens; shared prefix too short), so price and no-thinking decided
it: DeepSeek became the open judge. (These are the numbers as reported then. Every − pole item in this set carried
its unit's + label; see the 10-30 correction.)

**Plugin v3.** Per-slot router logit deltas added before Beam's grouped top-k, after SteerMoE. Tiny case 4r (force the
2 least-used experts +8, ban the 2 most-used −30, every MoE layer): match 0.030 / 0.030 / 0.037 (BF16 eager, graphs,
NVFP4 Marlin) against a ~0.15 floor, effect 0.226, captured router logits within 0.028 with deltas applied. $0.37.

**Job 2 inputs.** Public dial data from another entry showed the dial moving thinking 2.4× on AIME/Olympiad with flat
accuracy, so the task became AIME 2024 + 2025 + AMC 2023 (100 problems). From Job 1's captures: 37 effort experts
(19 forced in, 18 banned) in L7–45, mostly L15–29, a route control of random same-layer experts with the same deltas,
the effort direction per layer, and a "swap" vector (mean residual at N's last digit at 512 minus at 8, at the layer
where the readout peaks). Tiny rehearsal of every arm: each changes outputs vs its base. $0.275; GPU total $41.35.

## 2026-10-28 — Job 2: effort arms on competition math

**A cancelled first attempt.** Batch 1 was one generate call of 700 requests, saved only when all finished, inside a
job hard-killed at 120 min. AIME thinking runs to 16k tokens, so only ~20–45 sequences fit at once; any arm that loops
runs to the cap. At 50 min nothing had been saved and finishing looked like a coin flip. Cancelled at 51.4 min
(~$9.4), keeping only the 1,400-row transfer capture already uploaded. **Lesson: never put a long run's results behind
one blocking call.** Rerun (job 6ae17d2a): arms in six saved pairs in priority order, thinking cap 12k.

**Results (87 min; Job 2 total $25.67 with the cancelled attempt; GPU total $67.01).** Greedy, 100 problems, paired
against N = 8:

| Arm | Accuracy | Paired p (won/lost) | Thinking vs N = 8 |
|---|---|---|---|
| dial N = 8 | 80% (median think 840) | — | 1.00× |
| dial N = 32 / 128 / 512 | 93% / 93% / 92% | 0.001 / 0.001 / 0.002 | 1.78× / 2.37× / 2.55× |
| effort experts routed as at 512, prompt only | 91% | 0.007 (13/2) | 1.80× |
| digit swap, one layer, effort-digit position only | 90% | 0.013 (12/2) | 1.20× |
| effort direction spread over L15–30 | 85% | 0.18 (7/2) | 1.10× |
| routing at every token (up, random ctrl, down at 512) | 7%, 9%, 31% | broken | 5.8–8.1× |
| direction × 6 octaves at each of 16 layers (and ctrl) | 0% | broken | 14× |

Accuracy saturates by N = 32; N = 512 spends 2.55× the thinking for nothing more. AIME alone: dial 8 at 67/70% (2024 /
2025) → prompt routing 90/83%, swap 80/87%. Blunt arms break Beam and their random controls break it equally, which
fits the calibration cliff. **The gap:** neither gentle arm had a matched control (the routing control ran at every
token and broke Beam). Proposed: random experts on the prompt only, and a random vector of the swap's norm at the digit.

## 2026-10-28 — gentle-arm controls: low effort is a fragile state

Job 6ae1a9db, 13 min, $2.38 (GPU total $69.40). Random experts routed on the prompt only: 94% (+14 vs N = 8, p =
0.001), thinking 1.74×, indistinguishable from the effort experts (91%, p = 0.45 between them). **The effort experts
are markers, not levers**: almost any disturbance of the prompt's routing knocks Beam out of its low-effort mode into
roughly N = 32 behaviour. That is a finding about how the dial was trained, not an effort switch. Random vector at the
digit: 85%, thinking 1.00× (p = 0.79). The swap lengthens thinking 1.20× relative to that control (Wilcoxon p = 5e-8),
so the digit representation is a specific causal lever on thinking length, but a weak one next to the dial's 2.55×;
its accuracy edge over control (90 vs 85%, p = 0.27) is mostly the generic disruption. One greedy sample, n = 100.
Without these $2.38 the headline would have been "we found Beam's effort switch", which is wrong.

## 2026-10-28 — full judge scoring

4,091 items (2,092 significant + 2,000 random poles), DeepSeek, 0 unparsed, $0.292 (judge inference total $0.60).
As reported then: significant 62.1% [60–64], random 39.9%, chance 33%; the 681 items the judge marked confidence 3
are 98.5% correct. By band (significant): L8–15 28.9%, L16–23 37.3%, L24–31 65.5%, L32–39 63.0%, L40–47 85.9%, L48+
68.6%. **The early-layer "hits" are at chance with the judge**, so the few significant poles in L8–23 are best read
as false positives of the theme metric. This agrees with the expert-vs-random asymmetry: we never found a dose where
early experts say anything. **Corrected on 10-30:** the pooled figures mix valid + pole items with − pole items that
carried the wrong label; on + poles alone the judge is at 69.9% vs 48.8% (see the 10-30 correction). The early-layer
conclusion survives it (L8–15 27% on + poles).

## 2026-10-29 — corrections from a prior-art and rules check

A read-only research pass (four agents plus our own checks) changed several claims. **Identity:** "identity is a
tier-1 rule" is not in the safety report or the card; the chat template carries a default identity line ("You are
Beam, an AI assistant trained by Reflection."), so the 0.5–2.1% identity drops are framed against that line. **Prior
art:** MoTE already steered experts on DeepSeek-R1, so no "first expert steering at scale" claim; "The Expert Strikes
Back" labels ~1.1k experts by logit attribution without steering; Li et al. (2511.08579) find self-explanation beats
other-model explanation, which motivated asking Beam to label itself. **Free data:** the community REAP-255B prune
ships `reap_saliency.json` (kept 128 original expert ids per layer, routing counts), so a pruning cross-check costs
nothing. **Tooling:** upstream vLLM has no steering API; vllm-lens reads activations only; EasySteer has router
steering but is pinned to vLLM 0.29, before Beam.

## 2026-10-29 — Beam labels its own experts: pilot

**Setup.** Beam via the HF router at $0.40/M in, $1.20/M out, effort low, sees only a pole's 12 decoded lens tokens
and returns name, description and a "coherent" flag; never steered outputs or the answer key, so the three-way test
of its label stays blind. Pilot on 25 significant + 25 random judge items. Labels read like "exponentiation and
squaring", "identity and credential documents", "Japanese language and locale".

**Pilot (n = 50, a direction rather than a result).** DeepSeek with token labels 28/50 (significant 20/25), with
Beam's labels 25/50 (significant 20/25, random 5 vs 8; McNemar p = 0.38). Beam as judge 23/50, but 16/17 right at
confidence 3. Beam's flag: of 37 "coherent", 22 matched and 21 significant; of 13 "incoherent", 3 matched and 4
significant. One call thought until the 4,096-token cap. (As with the judge, any − pole in the pilot was labelled
from its + side's tokens; found on 10-30.)

**Two bad numbers, corrected.** Hours: we had read "~100 calls per 3 min" off two billing snapshots that bracketed our
own scripting; file timestamps showed 50 label calls in ~15 s at 8 workers (≥3.3 calls/s). Cost: the bill read $0.20
against $0.165 at token price, which we first reported as a 22% markup. It was not: the billing API books each new
Baseten request at a provisional +$0.01 and revises later; settled, the pilot cost $0.160. Only settled billing gets
quoted now; mid-run, 22.5k calls could show up to ~$225 provisional.

**REAP cross-check, lexicon version.** Null: shuffle `kept` within each (layer, 32-expert group), preserving REAP's
16-of-32 rule. Code-token experts kept 57.2% (n = 582, p = 0.0006), steering-significant Latin-script experts 46.6%
(p = 0.0008), single-script CJK experts 55.7% (p = 0.02), so the card's predicted multilingual loss is not supported
at expert level. Health keywords 38.6% (n = 57, p = 0.10); law and history too sparse to test. Keyword categories were
too blunt, which is why the full run added a subject field.

## 2026-10-29 — the full self-label run, a spending limit, and coherence vs significance

**Run.** All 22,550 testable poles, judge-set poles first, exponential backoff with jitter honouring Retry-After, a
token-priced running cap ($34, ×1.1). It stopped at 4,562 labels with HTTP 403 "exceeded your monthly spending limit
for Inference Providers": account-wide, so the DeepSeek judge was blocked too (month: $24.80 = Baseten $20.36,
DeepInfra $4.14, nscale $0.30; limit not in the API, ~$25 inferred). Early-layer calls cost ~2× (mean 1,700–1,900
output tokens in L0–15 vs ~800 in L24–47: Beam thinks longer over noisy tokens), so pole order became judge set →
L24–51 → L23 down to L0, letting a cap cut the layers whose labels predict nothing. After the limit was raised:
22,335 / 22,550 labelled (214 thinking runaways at the 4,096 cap, 1 persistent 429), $38.07 token-priced. (On 10-30
this turned out to be 10,926 distinct + pole labels; the ~11k − pole labels describe the + tokens again, so about half
the spend bought duplicates.)

**Beam's labels vs tokens (as reported then).** DeepSeek re-judge with Beam's labels, $0.287: significant 63.7% vs
62.1% with tokens (paired 170 vs 137, p = 0.07), random 40.0 vs 39.9. Two-to-six-word names lose nothing against 12
raw tokens. On + poles only (10-30): 71.3% vs 49.2% with Beam's labels, 69.9% vs 48.8% with tokens.

**Beam knows which experts are readable.** The flag is judged from tokens only; significance from behaviour only (the
theme metric never sees the lens), so the comparison is not circular. As reported then, on the judge set (balanced by
construction): coherent 61% significant (1,820 / 2,992) vs incoherent 31% (270 / 880), Fisher p = 1e-56; over all
22,335 labelled poles (computed at export): 12.1% vs 3.7% (3.2×). Corrected to + poles, whose labels describe the
direction that was pushed: 19.5% (n = 7,305) vs 5.6% (n = 3,621), 3.5×. The judge-accuracy version of this is partly
built in (incoherent labels say "mixed tokens") and is not used. Coherence by depth (all labels, so − poles counted
twice): L0–7 28%, L8–15 41%, L16–23 67%, L24–31 74%, L32–39 90%, L40–47 88%, L48–51 84%, an independent, Beam-side
reading of the same depth curve the judge gave.

**REAP by Beam's subject (coherent poles; 18 subjects, Bonferroni 0.0028).** Pruned more: chemistry 21.6% kept
(n = 37), biology 34.3%, health 35.2%, math 43.0% (all p ≤ 0.0008). Kept more: psychology 67% (p = 0.001), general
53.5%. **Correction:** the lexicon's "code kept" was syntax: format/syntax experts 53.6%, code concepts 50.9% (ns).
(Found on 10-30: this test counted each − pole's duplicate label as a second observation, so these p-values are
optimistic; not yet rerun on + poles.)
Against peterhdd's per-subject MMLU-Pro drops: Spearman −0.20 over 13 subjects, the expected sign and underpowered
(30 questions per subject, a drop's SE ~0.11). Output-side labels are not routing behaviour, and n = 30 cannot settle
it; the proper test is causal: ban the experts and score the full benchmark.

## 2026-10-29 — exact bans and the MMLU-Pro job

**Why −30 is not a ban.** Beam selects by sigmoid(logit) + correction bias, and the bias is 18.4–28.4 per layer with a
spread across a layer's experts of 1.6e-4 to 0.021. A −30 logit leaves the banned expert at ~0 + its bias, still
selectable whenever a kept expert's sigmoid score falls below that spread. Rare, not excludable. The fix is the bias
itself: `beam_steer_set_bias` lowers the banned experts' correction bias by 1,000 for the whole batch, and since the
bias enters only the choice and never the routing weights, this is exactly what slicing the experts out of the router
(REAP) does. Every arm keeps ≥2 experts per routing group. Tiny rehearsal: bias-ban case held to a reference with the
same bias change, different from unbanned, and an exact restore after reset; $0.32.

**Scoring without thinking.** Analysis channel closed in the prompt, final answer opened at "The answer is (", one
token restricted to the option letters, processed logprobs renormalised over the letters (mass 1.0). 12,032 questions,
3.12M prompt tokens per arm at ~8.6k tok/s, ~6 min per arm. Arms: base; health, law and math (every expert with a
coherent pole Beam labelled with that subject: 165, 143, 588); count-matched controls (same layer, matched on REAP's
routing counts, outside every tested subject); REAP emulation (6,272 bans); a final unbanned restore check. Quoted
$11–13; run without a hard cap, with the launcher warning as spend crossed thresholds.

**Results (job 6ae32f72, 59.3 min, $10.87).** Base 62.3%. Paired against count-matched controls: law ban −13.4 points
on law (p = 1e-21), +0.2 elsewhere (ns); health −6.5 on health (p = 4e-8), −0.1 elsewhere; math −8.7 on math
(p = 3e-14) but also physics −9.2, chemistry −7.9, economics −6.2, business −4.8, verbal subjects ±1. Controls vs base
−0.2 to −0.4. The "math" experts carry quantitative work across number-heavy subjects, not math as a topic. REAP
emulation: 47.3% (−15.0): health −32.1, history −23.1, philosophy −20.6, law −20.1 … engineering −5.3, consistent with
a code-weighted calibration set. Spearman of per-subject drops vs peterhdd's +0.30 (n = 14), vs the share of
Beam-labelled experts REAP kept +0.30 (n = 13): right direction, not significant. The bans are the evidence; the REAP
link is supportive only. The 10-30 label bug does not touch the bans: they remove whole experts, and a − pole's
duplicate label still describes that expert's + direction.

**Restore "MISMATCH" was engine noise.** The final unbanned re-run differed from the first. Re-run on 500 questions in a
smaller batch: mean |Δ logprob| 0.117 and 93.2% same prediction, vs 0.931 and 61% for the REAP arm. TP-4 inference
at a different batch size is not bit-exact; the 0.01 threshold came from the tiny rehearsal. The check now compares
against the last banned arm (what a leaked bias would look like). The noise is random across questions, so paired
statistics stand.

## 2026-10-30 — export and packaging

`export_atlas.py` turns local run files into public tables: one row per expert pole (lens tokens, Beam's name,
description, subject and coherence, steering significance, judge verdict, REAP kept, ban arms), steered outputs, judge
items, MMLU-Pro ban arms, effort runs and calibration, ~106 MB. Benchmark rows point at source dataset and row with
no problem text; WikiText ships only as derived norms. The README doubles as the submission card (Apache-2.0); the
explorer is a static Space because a free Gradio Space needs PRO. Total spend about $120: ~$81 GPU (rehearsals, the
v1 pilot sweep and the cancelled run included), $38 of Beam labelling, under $1 of judge inference, plus the TB4
study's ~$9.50. Nothing published at the time of writing. The first README headlines (judge 62% vs 40%, coherent
3.2× as significant) were written before the pole-label bug below was found; `export_atlas.py` now fixes it at export
(commit f6e00bd): − poles get their own atlas tokens (the opposite sign of the same direction), and label-based
columns (Beam's label, judge verdict) are filled for + poles only.

## 2026-10-30 — the "−" pole labels were wrong

**The bug.** `rank_sweep.py` writes the unit's lens tokens (what its + push promotes) on both pole rows of a unit in
`ranked_matched.jsonl`. `judge_items.py` and `pole_rows.py` took each pole's label from there, so every − pole was
judged with, and self-labelled from, the + side's tokens. A − push promotes the opposite sign of the same weight
direction: L26 E174 direction 2 is COVID / coronavirus / pandemic on +, and resumed / halted / 中断 on −.

**Split by pole, the judge tells the story.** + poles: significant 69.9% (n = 1,627), random 48.8% (n = 982). − poles:
significant 34.7% (n = 464), random 31.2% (n = 1,018). Chance is 33%. The pooled 62.1% vs 39.9% mixed valid + items
with mislabelled − items. The − items are an accidental mismatched-label control: when the label describes another
direction, the judge is at chance, which is what a judge that reads labels rather than guessing from effect size
should do.

**Corrected numbers (+ poles only).** Judge 69.9% vs 48.8%; confidence-3 answers 98.7% correct (n = 670); with Beam's
own labels 71.3% vs 49.2%. Coherent + labels are steering-significant 19.5% of the time vs 5.6% for incoherent ones
(3.5×; previously reported as 3.2×, and "61% vs 31%" on the judge set). By band, judge on significant / random +
poles: L8–15 27% / 33%, L16–23 42% / 28%, L24–31 75% / 53%, L32–39 73% / 62%, L40–47 89% / 71%, L48–51 64% / 68%
(n = 25 / 31 there). Beam labelled 10,926 distinct + poles; the ~11k − pole labels are duplicates of the + tokens,
about half of the $38 labelling spend.

**What is unaffected.** Steering significance (behavioural; the theme metric never reads labels), the effort results,
and the MMLU-Pro bans (whole experts banned; a − pole's label still describes the expert's + direction). **What is
weakened.** The REAP-by-Beam-subject cross-check counted the duplicate labels twice, so its p-values are optimistic
and it has not been rerun. The random-pole rate rising to ~49% on + poles also means the gap over random poles
(~21 points) is smaller in relative terms than the pooled numbers suggested. Lesson: a label column that is the same
on both rows of a two-row unit should have been caught by a one-line uniqueness check before any judge spend.

## Open questions / what we'd do next

- **Early and late layers are unmeasured, not empty.** L0–23 needs per-unit doses large enough to move Beam (experts
  there absorb pushes 5–40× better than random directions); L48+ needs smaller ones (46% of their poles have no fair
  control). A per-unit re-sweep of ~700 units was estimated at ~10 GPU-minutes and never run.
- **One greedy sample everywhere.** Steered themes, thinking lengths and the 100-problem math arms all rest on one
  sample; sampled repeats would put intervals on the effort results.
- **Routing, not just writing.** Beam's labels describe what an expert writes. Capturing which experts each MMLU-Pro
  subject actually routes to would test whether the "law" experts are also the ones law questions select.
- **Thinking on.** MMLU-Pro was scored without thinking (62.3%). Whether thinking routes around a ban is open.
- **The real pruned model.** We emulated REAP on the NVFP4 model; running the REAP checkpoint would check the emulation.
- **Effort.** The swap is specific but weak (1.2× vs the dial's 2.55×). Untested: the direction at the think start
  with a dose calibrated to the cliff, and why low effort is fragile (what the trained low-effort state depends on).
- **The − poles were never labelled.** ~11k − poles need their own Beam labels and judge items (not yet quoted), and
  the REAP-by-subject test needs a rerun on + poles only.
- **The 3 lost waves** (478 experts) and the dense neurons, which got 48 units and a broken lens.
- **Never started:** the false-submit probe on the 65 TB4 trajectories; the held-back-expert reading of the router
  bias; co-firing structure among experts. None of this was pre-registered, unlike the TB4 study.
