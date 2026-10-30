---
license: apache-2.0
pretty_name: Beam Expert Atlas
language:
  - en
tags:
  - interpretability
  - mixture-of-experts
  - steering
  - activations
size_categories:
  - 100K<n<1M
configs:
  - config_name: poles
    default: true
    data_files: experts/poles.parquet
  - config_name: steering_outputs
    data_files: steering/outputs.parquet
  - config_name: steering_units
    data_files: steering/units.parquet
  - config_name: steering_baseline
    data_files: steering/baseline.parquet
  - config_name: judge_items
    data_files: judge/items.parquet
  - config_name: ban_arms
    data_files: bans/arms.parquet
  - config_name: mmlu_pro_bans
    data_files: bans/mmlu_pro.parquet
  - config_name: effort_gsm8k_dial
    data_files: effort/gsm8k_dial.parquet
  - config_name: effort_gsm8k_steer
    data_files: effort/gsm8k_steer.parquet
  - config_name: effort_competition_math
    data_files: effort/competition_math.parquet
  - config_name: calibration_resid_norms
    data_files: calibration/resid_norms.parquet
  - config_name: calibration_dose_response
    data_files: calibration/dose_response.parquet
  - config_name: captures_gsm8k_index
    data_files: captures/gsm8k_effort_index.parquet
  - config_name: captures_competition_math_index
    data_files: captures/competition_math_index.parquet
---

# Beam Expert Atlas

Every routed expert of [reflection/Beam-1.0-501B-A23B](https://huggingface.co/reflection/Beam-1.0-501B-A23B)
(49 MoE layers × 256 experts), pushed inside the running model and checked against random controls, labelled by
Beam itself, plus expert-ban experiments on MMLU-Pro and Beam's internal activations across reasoning-effort
settings. Browse it in the [Inside Beam explorer](https://huggingface.co/spaces/rycerzes/inside-beam); code and
method in [rycerzes/beam-inside](https://github.com/rycerzes/beam-inside).

All runs used the NVFP4 release in vLLM (tensor parallel 4, Marlin MoE kernels) with the `beam-steer` plugin,
greedy decoding, prefix caching off.

## Terms

- **Unit / direction.** For each expert, the top write directions of its `down_proj` (SVD of the FP8 weights) with
  a sign chosen so the + side promotes the tokens in `lens_tokens` (logit lens on the weights). The sweep also has
  48 dense-layer neurons and 832 random-direction controls (`steering_units`).
- **Pole.** A unit pushed + or −. A − push promotes the opposite side of the same direction.
- **Dose.** Per layer, calibrated on effect: 1.0 is the push at which random directions reach the median "largest
  coherent" ΔNLL (τ*), 0.4 its gentler point. Doses are +0.4, +1.0 (+ pole) and −1.0 (− pole).
- **Significant.** The pole's outputs drift toward one theme (`theme_words`) more than ≥ 95% of random-direction
  controls within ±8 layers whose own ΔNLL is within 1.5× of the pole's (`null = matched`). `under`: compared with
  the 20 weakest nearby controls. `untestable`: the pole moves Beam more than nearly every control, so there is no
  fair null (common in layers 48+).

## Configs

| Config | Rows | What |
|---|---|---|
| `poles` | 24,132 | one row per expert pole: tokens, Beam's label, steering statistics, judge verdict, ban arms |
| `steering_outputs` | 142,120 | every steered generation: unit × prompt (who / snake / orange / effort) × dose, thinking, answer, raw text and a `status` |
| `steering_units` | 13,424 | unit definitions, including dense neurons and controls |
| `steering_baseline` | 4 | unsteered answers to the four prompts |
| `judge_items` | 4,091 | blind three-way test (DeepSeek-V4-Flash): which of three poles' outputs matches a label |
| `ban_arms` | 8 | expert lists for the MMLU-Pro arms (`L{layer}_E{expert}`) |
| `mmlu_pro_bans` | 96,256 | per question and arm: prediction and the ten letters' log-probabilities |
| `effort_gsm8k_dial` | 900 | GSM8K at efforts 2–1000 |
| `effort_gsm8k_steer` | 1,200 | GSM8K with the effort direction steered at layers 26 / 36 |
| `effort_competition_math` | 1,400 | AIME 2024, AIME 2025, AMC 2023 under 14 effort arms |
| `calibration_*` | | residual norms per layer (WikiText-103 test, statistics only) and random-direction dose response |
| `captures_*_index` | 9,000 / 1,400 | index into the activation captures (below) |

`summary.json` has the headline numbers; `explorer/` holds the per-layer JSON the explorer loads.
`steering/directions.safetensors` holds every unit's direction (`dirs`, row = `unit`) and the per-layer median
residual norm (`median_resid`): a push is `dirs[unit] * dose_eff * median_resid[layer]`, added to the residual leaving
`layer` from position 1 on, which is what the [beam-steer](https://github.com/rycerzes/beam-inside/tree/main/steer)
plugin's `beam_steer_set` takes.

### Prompts

`who` "Who are you?", `snake` "Tell me something about snakes.", `orange` "Describe an orange in two sentences."
(effort 8, 160 tokens) and `effort` "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How
much does the ball cost?" (effort 32, 512 tokens).

### Output status

Generations were capped at 160 tokens (512 for `effort`), thinking included, so long answers are cut by design: the
unsteered `snake` answer itself is. `status` says what happened: `finished`; `budget` (cut while answering);
`thinking` (cut before answering); `loop` (ran to the cap repeating itself, distinct-4-gram share < 0.5); `format`
(left Beam's reply format: a corrupted channel header such as `final?xml`, an answer written into the header, a tool
call); `no-answer`. `raw` keeps the full text with Beam's special tokens.

### Labels and the − poles

Beam's labels (`beam_*`) and the judge verdicts (`judge_*`) exist for + poles only. A first pass also labelled and
judged − poles, but from the + side's tokens; those items stay in `judge_items` with `label_matches_pole = false`
and score at chance (34.7% significant, 31.2% random), which makes them a mismatched-label control. With matching
labels: 69.9% for significant + poles, 48.8% for random ones, chance 33%.

### MMLU-Pro bans

Bans set the expert's routing correction bias to −1000 for the whole batch, which removes it from top-k selection
exactly without changing the remaining experts' weights. Scoring is without thinking: the next-token log-probability
of each answer letter after the prompt, normalised over the allowed letters. `question_id` is the `question_id` of
[TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) (test split). Arms: `base`; `law`, `health`,
`math` (experts with a coherent Beam label of that subject: 143, 165, 588) with `*_ctrl` arms that ban the same
number of experts per layer, none labelled with a tested subject, each matched to a banned expert on routing load.
Beam labelled each direction's + tokens twice (the second time as its − pole, see above), and an expert is in an arm
if either label named the subject: 25, 18 and 54 of them only through the second label. Routing load is the
per-expert routing count in the REAP calibration file (`reap_saliency.json`, field `count`, from
[mrfakename/Beam-1.0-REAP-255B-A23B](https://huggingface.co/mrfakename/Beam-1.0-REAP-255B-A23B), Apache-2.0).

### Effort

`q` in `effort_gsm8k_*` is the row of [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) `main/test`. In
`effort_competition_math`, `source_dataset` and `source_row` point at the problem; problem text is not included.
`think_tokens` counts the thinking tokens (all tokens when the 12k cap was hit). Arms: `dial8/32/128/512` (effort
setting only); `swap` (effort 8 plus the N = 512 minus N = 8 mean residual difference at the effort digit, layer 12)
and `swap_ctrl` (random vector, same norm); `route_up_prompt` (37 effort experts forced in or banned on prompt tokens)
and `route_ctrl_prompt` (37 random experts); `route_up`, `route_ctrl`, `route_down` (the same on every token);
`dir_band`, `dir_band_ctrl`, `dir_spread` (the effort direction over layers 15–30, and controls).

### Activation captures

`captures/*.safetensors` hold `resid` [52 layers, rows, 5120] (BF16, the residual leaving each layer) and `router`
[52, rows, 256] (FP32 router logits, zero for dense layers 0–2). The index tables give each row's file, question,
effort, position, token id and kind:

- `gsm8k_effort` (5.3 GB): 200 GSM8K test questions × efforts 2, 8, 16, 32, 64, 128, 256, 512, 1000 × 5 positions:
  `n_last` (the effort number's last digit), `sys_end`, `think_start`, `thought1`, `thought2`.
- `competition_math` (0.8 GB): the 100 competition problems × efforts 8–512 × `n_last`, `prompt_end`.

```python
from huggingface_hub import hf_hub_download
from safetensors import safe_open
import pandas as pd

repo = "rycerzes/beam-expert-atlas"
idx = pd.read_parquet(hf_hub_download(repo, "captures/gsm8k_effort_index.parquet", repo_type="dataset"))
rows = idx[(idx.kind == "n_last") & (idx.file == "gsm8k_effort_00.safetensors")]
with safe_open(hf_hub_download(repo, "captures/gsm8k_effort_00.safetensors", repo_type="dataset"), "np") as f:
    x = f.get_slice("resid")[12, :, :]  # layer 12, every row of this file (BF16 needs framework="pt" for numpy-free use)
```

## Limitations

- One greedy sample per prompt and problem.
- Layers 0–23 are underdosed in effect terms; their few significant poles are at chance with the judge. Layers 48+
  collapse at the doses needed and often have no fair control.
- 478 experts have no sweep data (3 of 79 waves hit the job's time limit).
- Labels come from an expert's top tokens (what it writes), not from when the router selects it.

## Licence and sources

Released under Apache-2.0, like Beam. Generations are Beam's (Apache-2.0). Question references point at MMLU-Pro
(MIT), GSM8K (MIT), [HuggingFaceH4/aime_2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024),
[MathArena/aime_2025](https://huggingface.co/datasets/MathArena/aime_2025) (CC BY-NC-SA 4.0) and
[math-ai/amc23](https://huggingface.co/datasets/math-ai/amc23); no question text is copied, though Beam's outputs may
restate a problem. WikiText-103 (CC BY-SA) was used only for the residual-norm statistics. Routing counts for the ban controls come
from the REAP calibration file in mrfakename/Beam-1.0-REAP-255B-A23B (Apache-2.0).
