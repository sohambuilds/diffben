# CompScale Sanity Suite — Execution Runbook

Run these commands in order from the repo root. Every step is idempotent —
the generation + evaluation scripts skip already-produced outputs, so you can
stop and resume at any point.

Environment:
* `GEMINI_API_KEY` set in the shell for VLM evaluation steps.
* Primary model: FLUX.2 klein 9B (fits in 48GB VRAM bf16).
* Evaluator: Gemini 3 Flash Preview.

All gate outputs land in `compscale/sanity/`. The final aggregator is
`compscale/analysis/sanity_report.py`.

---

## Step 0 — Generate the benchmark prompts

```
python compscale/benchmark/generate_prompts.py
```

Emits:
* `compscale/benchmark/prompts/attribute_pilot_v2.json` (80 prompts, 20 per k)
* `compscale/benchmark/prompts/negation_pilot.json` (40)
* `compscale/benchmark/prompts/spatial_pilot.json` (40)
* `compscale/benchmark/prompts/numeracy_pilot_v3.json` (40)

Then build the order-permutation pilot:

```
python compscale/generation/permute_prompts.py
```

Emits `compscale/benchmark/prompts/order_pilot.json` (60 prompts = 5 base × 4 types × 3 perms).

---

## Step 1 — Module 3: token-length validation (before spending compute)

```
python compscale/benchmark/validate_lengths.py
```

Writes `compscale/sanity/token_length_report.json` and a violin plot.
Gate: median token count at k=8 within 1.5x of k=1 per type.
**If this fails, iterate on `generate_prompts.py` before proceeding.**

---

## Step 2 — Generate images (klein 9B, ~1,200 images total)

Module 1 baseline (N=4 per prompt across all four types + 20 attr prompts/k for Module 6):

```
python compscale/generation/generate_images.py --klein_9b --n_images 4 \
  --prompts compscale/benchmark/prompts/attribute_pilot_v2.json \
            compscale/benchmark/prompts/negation_pilot.json \
            compscale/benchmark/prompts/spatial_pilot.json \
            compscale/benchmark/prompts/numeracy_pilot_v3.json
```

Module 4 permutation pilot (N=4):

```
python compscale/generation/generate_images.py --klein_9b --n_images 4 \
  --prompts compscale/benchmark/prompts/order_pilot.json
```

Module 5 — pick the seed-variance subset, then extend that subset from N=4 to N=16:

```
python compscale/analysis/seed_variance.py subset
# Copy the printed --prompt_ids block into:
python compscale/generation/generate_images.py --klein_9b \
  --prompts compscale/benchmark/prompts/attribute_pilot_v2.json \
            compscale/benchmark/prompts/negation_pilot.json \
            compscale/benchmark/prompts/spatial_pilot.json \
            compscale/benchmark/prompts/numeracy_pilot_v3.json \
  --n_images 16 --start_image_index 4 \
  --prompt_ids <20 prompt ids from subset step>
```

Outputs land in `compscale/generation/outputs/flux2-klein-9b/{prompt_id}/img_{j}.png`.

---

## Step 3 — VLM evaluation (Gemini 3 Flash Preview)

Module 1 + Module 6 (all four pilots, N=4 images each):

```
python compscale/evaluation/vlm_verify.py \
  --prompts compscale/benchmark/prompts/attribute_pilot_v2.json \
            compscale/benchmark/prompts/negation_pilot.json \
            compscale/benchmark/prompts/spatial_pilot.json \
            compscale/benchmark/prompts/numeracy_pilot_v3.json \
  --images_dir compscale/generation/outputs/flux2-klein-9b \
  --n_images 4 \
  --output compscale/evaluation/results/sanity_results.json
```

### Step 3 (budget variant) — low-credit Gemini runs

If you are credit-constrained on Gemini 3 Flash Preview and cannot afford
~3,000 per-constraint calls for the main sanity block, use the budget
flags. They compose; start with only `--batch_constraints` and add the
subsamplers only if needed.

`--batch_constraints` bundles every constraint for a single (prompt, image)
pair into ONE VLM call (~3.75x fewer calls on the core suite: ~3,000 -> ~800).
Per-constraint scoring is identical, so Module 1 baseline + Module 6 fits
remain valid. If a batched response ever fails to parse, the script
transparently falls back to per-constraint calls for just that image.

```
python compscale/evaluation/vlm_verify.py \
  --prompts compscale/benchmark/prompts/attribute_pilot_v2.json \
            compscale/benchmark/prompts/negation_pilot.json \
            compscale/benchmark/prompts/spatial_pilot.json \
            compscale/benchmark/prompts/numeracy_pilot_v3.json \
  --images_dir compscale/generation/outputs/flux2-klein-9b \
  --n_images 4 \
  --batch_constraints \
  --output compscale/evaluation/results/sanity_results.json
```

Further levers (stack only if batching alone is not enough):

* `--max_prompts_per_k N` — keep only the first N prompts per (type, k).
  Deterministic by file order. Halves calls at N=10 for attribute, etc.
* `--max_k_levels 1 8` — drop the middle of the ladder. Only keeps
  endpoint deltas; Module 6 curve-fit is no longer meaningful (need >=3
  k-points), but the independence-ratio check at k=8 still works.
* `--n_images 2` — halves calls at the cost of wider seed CIs.

Example — maximally cheap smoke eval (batched + endpoints only + N=2):

```
python compscale/evaluation/vlm_verify.py \
  --prompts compscale/benchmark/prompts/attribute_pilot_v2.json \
            compscale/benchmark/prompts/negation_pilot.json \
            compscale/benchmark/prompts/spatial_pilot.json \
            compscale/benchmark/prompts/numeracy_pilot_v3.json \
  --images_dir compscale/generation/outputs/flux2-klein-9b \
  --n_images 2 --batch_constraints --max_k_levels 1 8 \
  --output compscale/evaluation/results/sanity_results_cheap.json
```

If you run the cheap smoke eval first and still have credits, rerun the
full command above — output files are written fresh each time, so pick
whichever you want `sanity_report.py` to consume.

Module 4 (permutation pilot):

```
python compscale/evaluation/vlm_verify.py \
  --prompts compscale/benchmark/prompts/order_pilot.json \
  --images_dir compscale/generation/outputs/flux2-klein-9b \
  --n_images 4 \
  --output compscale/evaluation/results/order_results.json
```

Module 5 (seed-variance subset, N=16):

```
python compscale/evaluation/vlm_verify.py \
  --prompts compscale/benchmark/prompts/attribute_pilot_v2.json \
            compscale/benchmark/prompts/negation_pilot.json \
            compscale/benchmark/prompts/spatial_pilot.json \
            compscale/benchmark/prompts/numeracy_pilot_v3.json \
  --images_dir compscale/generation/outputs/flux2-klein-9b \
  --n_images 16 \
  --prompt_ids <20 prompt ids from subset step> \
  --output compscale/evaluation/results/variance_results.json
```

---

## Step 4 — Module 2: VLM question-template reliability (hand-labeled)

Emit the labeling template:

```
python compscale/evaluation/sample_for_labeling.py
```

Writes `compscale/evaluation/human_eval/hand_labels_template.csv`.

Hand-label: open the CSV, fill `human_label` with `sat` or `unsat` for each
row (160 rows, ~10 min per type). Save a copy as
`compscale/evaluation/human_eval/hand_labels.csv`.

Run the reliability check:

```
python compscale/evaluation/vlm_reliability.py
```

Writes `compscale/sanity/vlm_reliability_report.json`.
Gate: per-type VLM agreement with human labels > 0.90.

---

## Step 5 — Module 4 analysis

```
python compscale/analysis/order_permutation.py \
  --base_results compscale/evaluation/results/sanity_results.json \
  --perm_results compscale/evaluation/results/order_results.json
```

Writes `compscale/sanity/order_permutation_report.json`.
Gate: mean per-base-prompt |Δ satisfaction| < 0.10 per type.

---

## Step 6 — Module 5 analysis

```
python compscale/analysis/seed_variance.py analyze \
  --results compscale/evaluation/results/variance_results.json
```

Writes `compscale/sanity/seed_variance_report.json`.
Gate: median per-prompt N=4 bootstrap CI width < 0.15.

---

## Step 7 — Module 6 attribute-binding decay fit

```
python compscale/analysis/fit_scaling_law.py \
  --results compscale/evaluation/results/sanity_results.json \
  --type attribute \
  --summary_name sanity_attr_summary.json \
  --figure_name sanity_attr_decay_curve.png
```

Writes `compscale/analysis/figures/sanity_attr_summary.json`.
Gate (checked by aggregator): independence ratio at k=8 < 1.0 with upper
95% bootstrap CI bound < 1.0, and sigmoid/exp/power fit reported.

Optional: repeat with `--type negation`, `--type spatial`, `--type numeracy`
to preview scaling curves for the other dimensions. Only types that passed
Module 1 (baseline > 0.7 at k=1) will produce meaningful fits.

---

## Step 8 — Aggregate and decide

```
python compscale/analysis/sanity_report.py
```

Writes `compscale/sanity/sanity_report.json` and prints a pass/fail table.

If all six modules pass, the pipeline is cleared for the full CompScale-Bench
run (model scaling trio, paraphrase loop, factorial interference matrix,
evaluator GT calibration — per the main plan in `CLAUDE.md`).

If any module fails, fix that dimension (iterate prompts, upgrade N, swap
evaluator, etc.) and re-run only the affected steps — the aggregator picks
up the latest report files automatically.
