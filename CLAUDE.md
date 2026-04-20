# CLAUDE.md

Agent-facing context for the **CompScale** repo. For the full research motivation, literature positioning, methodology, and historical pilot findings, read `README.md` — this file is just the operational layer on top of it.

## What this project is (one paragraph)

CompScale measures **scaling laws for compositional fidelity** in T2I diffusion models: how per-constraint satisfaction decays as constraint count `k` grows from 1 to 16, what mathematical form the decay takes, and whether cross-constraint interference between types (attribute / negation / spatial / numeracy) is asymmetric. Target venue: EMNLP 2026 main/findings. Framed as an **NLP paper** that uses image generation as a diagnostic probe for text-encoder limitations — not a vision paper, not a benchmark paper. Keep that framing in mind when writing prose.

## Repo layout

```
compscale/
├── benchmark/
│   ├── ontology/                  # countable_objects, negation_targets, spatial_objects
│   ├── prompts/                   # generated prompt JSONs (attribute, negation, spatial, numeracy, order)
│   ├── generate_prompts.py        # template generator — emits typed constraints + scene_prefix/suffix
│   └── validate_lengths.py        # Module 3 token-length gate
├── generation/
│   ├── generate_images.py         # --klein_9b etc.; stable seeds via sha256(prompt_id)+base_seed
│   ├── permute_prompts.py         # Module 4 order permutations
│   └── outputs/<model>/<prompt_id>/img_{j}.png
├── evaluation/
│   ├── vlm_verify.py              # Gemini 3 Flash Preview, type-routed VQA, --batch_constraints for budget
│   ├── sample_for_labeling.py
│   ├── vlm_reliability.py         # Module 2
│   ├── human_eval/
│   └── results/sanity_results*.json
├── analysis/
│   ├── fit_scaling_law.py         # exp/sigmoid/power + bootstrap independence ratio CI
│   ├── order_permutation.py       # Module 4
│   ├── seed_variance.py           # Module 5 (subset | analyze)
│   ├── diagnose_pilot.py          # per-type breakdown
│   ├── sanity_report.py           # aggregates all six modules
│   └── figures/
└── sanity/
    ├── RUN.md                     # sequential execution playbook
    └── *_report.json              # per-module gate outputs
```

Top-level: `main.py` (entry stub), `pyproject.toml` / `uv.lock` (uv-managed env), `testbook.ipynb` (latest-results scratchpad, git-untracked).

## Running things

Full sequential playbook: `compscale/sanity/RUN.md`. Short version:

```bash
# Step 0 — prompts (deterministic)
python compscale/benchmark/generate_prompts.py
python compscale/generation/permute_prompts.py

# Step 1 — Module 3 gate before burning GPU time
python compscale/benchmark/validate_lengths.py

# Step 2 — generate (klein 9B, idempotent, stable seeds)
python compscale/generation/generate_images.py --klein_9b --n_images 4 --prompts ...

# Step 3 — VQA eval (use --batch_constraints when credit-constrained)
python compscale/evaluation/vlm_verify.py --prompts ... --batch_constraints --output ...

# Step 8 — aggregate
python compscale/analysis/sanity_report.py
```

Generation runs on a Linux server with 3× 48GB GPUs — **local machine does not have generated images or results** beyond what gets rsync'd into `compscale/evaluation/results/`. Don't try to regenerate locally.

Environment: `GEMINI_API_KEY` must be set for VQA steps. FLUX.2 klein 9B fits native bf16 in 48GB.

## Non-negotiable conventions

1. **Every constraint carries an explicit `type` field** (`"attribute" | "negation" | "spatial" | "numeracy"`). The VLM verifier routes on it, diagnostic reporting signatures on it, analysis scripts filter on it. Never rely on heuristics to infer type.
2. **Every prompt stores `scene_prefix` AND `scene_suffix` alongside `constraints`.** Permutation and any rewriter rebuild strings from metadata — no string parsing of the rendered prompt.
3. **Stable per-prompt seeds:** `seed = sha256(prompt_id) + base_seed`. Extending N=4 → N=16 must re-use the first four images exactly.
4. **Token-length gate is bidirectional:** `median(k=8) / median(k=1) ∈ [1/1.5, 1.5]` per type. Either direction of imbalance confounds the k signal.
5. **Independence test is the real Module 6 signal,** not raw all-satisfied decay. Always compare observed against `p^k` where `p` = per-constraint sat at k=1. Bootstrap CI on the ratio.
6. **Countability tiers are load-bearing.** Only Tier 1–2 objects in numeracy prompts, counts restricted to 1–3. Tier 3 objects (beads, confetti, flowers in clusters, etc.) are explicitly blacklisted in `ontology/countable_objects.json`.
7. **Negation targets are additive/modular only,** never structural (no "no floor", "no ceiling"). That tests physical-prior override, not logical negation.

## Known gotchas

- **Module 6 gate is mis-specified, not just buggy** (April 20, 2026 run): the gate assumes every type can be measured at k=8, but detectability of interference against N=40 binomial noise requires `p^k ≳ 0.1`. Types where that fails collapse to observed=0 and trivially pass. Fix is not just a floor — the gate should emit `N/A (p^k too low)` when predicted independence is below ~0.1, and the k where interference is detectable must be chosen per-type from p_k1 (see per-type k-ladder below).
- **Interference detection is per-type-p_k1-dependent.** Attribute (p_k1=1.00) → interference visible at k=8–16. Numeracy (0.775) → k=3–6. Negation (0.675) → k=2–4. Spatial (0.425) → not detectable on klein. Budget allocation and k-ladder design for the full trio must follow this, not a uniform ladder.
- **Attribute scaling curves must be fit on all-satisfied, not per-constraint.** On klein 9B, per-constraint decay is 1.000 → 0.966 (3.4pp range, near-flat). The power-law fit with R²=0.997 is spurious. All interesting attribute signal lives in all-satisfied (1.000 → 0.738, 26pp range).
- **Numeracy is the cleanest scaling-law anchor we have.** On klein 9B, exponential fit R²=0.994, α=0.075 on per-constraint. Treat numeracy as the primary Contribution 2 datapoint when running the full trio; attribute is Contribution 3 (interference) fuel only.
- **Numeracy was at floor on klein 4B (~0.35)** but jumps to 0.775 on klein 9B. Do not generalize 4B numeracy findings up the scaling ladder.
- **Spatial fails Module 1 baseline on klein 9B** (p=0.425 < 0.5). Plan: keep spatial in the benchmark but measure only on frontier APIs (DALL-E 3 / GPT Image-1 / Gemini Imagen). Drop from klein-tier scaling figure. Do not soften the spatial VQA question — eval rigor over rescue.
- **Negation non-monotonicity (k=2 > k=1) is probably evaluator prior, not generation.** Candidate mechanism: VLM "Is there a {object}?" has a scene-incoherence prior that inflates per-constraint satisfaction when more out-of-scene negations stack. Test by partitioning current negation results on scene-coherence before any new generation. If confirmed, restrict `negation_targets.json` to scene-coherent objects only; this is load-bearing for Contribution 3.
- **N=4 images per prompt is insufficient for the full trio.** CI widths on klein 9B Module 1 are already ~0.28 (negation k=1). Bump to N=8–12 based on Module 5 seed-variance output before the trio run.
- **Per-object count difficulty dominates over k** if count distribution is not controlled — that's the whole reason numeracy prompts are count-1-to-3 only.
- **FLUX.2's text encoder is Mistral-3-family**; use `Mistral-Nemo-Base-2407` as the tokenizer proxy in `validate_lengths.py`.
- **`testbook.ipynb` is git-untracked** and is the local scratchpad for whatever results have rsync'd down from the server. Do not assume its contents match the current repo state without checking `compscale/evaluation/results/`.

## Do-not-do list

- Do **not** propose sequential prompt decomposition / recaption-then-compose as a *fix* — PRISM (ICLR 2026) and PASTA (ICML 2025) own that space. We do analysis, not engineering.
- Do **not** frame this as a vision or benchmark paper. It's computational linguistics using T2I as a diagnostic probe.
- Do **not** cite T2I-CompBench++ as a competitor — it tests at k=2–3, has no k-ladder, no decay fits, no interference analysis. Cite as complementary low-complexity baseline.
- Do **not** reintroduce Tier 3 objects or counts > 3 in numeracy prompts.
- Do **not** treat Module 6 pass/fail as authoritative for types where observed all-satisfied at k=8 is 0, or more generally where `p_k1^k < 0.1` (see mis-specification above).
- Do **not** report the klein 9B attribute power-law fit (R²=0.997 on per-constraint) — it's fitting noise on a near-flat line. Use all-satisfied.
- Do **not** run a uniform k-ladder across types in the full trio — attribute gets {1,2,3,4,6,8,12,16}, numeracy/negation get {1,2,3,4,6,8}, spatial is API-only.
- Do **not** soften the spatial VQA template to lift p_k1 — eval rigor over rescue.
- Do **not** regenerate images on the local Windows machine — generation lives on the Linux server; only results come down.
- Do **not** remove `scene_prefix` / `scene_suffix` from prompt JSONs to "clean them up" — permutation and diagnostics depend on them.

## Current status (April 20, 2026)

- Sanity suite Modules 1 + 6 have landed for all four types on klein 9B. See `testbook.ipynb` and the "Klein 9B Sanity Results" + "Forward Plan" sections of `README.md`.
- Modules 2, 3, 4, 5 reports not yet produced from the current prompt files — outstanding before the full trio run.
- **Next blocking action** (before any more compute): test the negation scene-coherence hypothesis by re-scoring current negation results. If evaluator prior explains the k=2 > k=1 jump, the negation ontology needs restricting and Contribution 3 needs re-analysis before the factorial runs.
- ARR deadline: ~May 25, 2026. ~5 weeks. Sanity cleanup + prompt/ladder re-spec: ~2 weeks. Full trio + frontier APIs: ~2–3 weeks. Writing: ~1–2 weeks in parallel from trio week 1.
- Update this file when significant and necessary information has been collected.