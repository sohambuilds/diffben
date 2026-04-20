# CLAUDE.md — CompScale Project Context (v4)

## What This Document Is

This is the canonical context file for the **CompScale** research project. Feed this to any agent (coding, writing, brainstorming, evaluation) working on any part of this project. It contains the full research motivation, literature positioning, methodology, technical specs, and open questions.

**v2 changelog (March 2026):** Killed ConDecomp (scooped by PRISM @ ICLR 2026). Repositioned benchmark against DetailMaster and T2I-CoReBench. Updated model roster to FLUX.2 family. Promoted asymmetric interference analysis to full Contribution 3.

**v5 changelog (April 20, 2026):** First klein 9B sanity results landed. Attribute gives a clean interference signal at k=8 (ratio 0.74, CI [0.64, 0.83]). Numeracy baseline jumped 4B→9B from ~0.35 to 0.775 — first real scaling evidence, and the cleanest scaling-law fit in the set (exponential R²=0.994, α=0.075). Negation is non-monotonic (k=2 > k=1) and needs diagnostic work. Spatial fails the Module 1 baseline gate (p=0.425). Identified Module 6 gate as **mis-specified**, not just floor-buggy: interference detectability depends on p_k1 of each type, and a uniform k=8 gate trivially passes three of four types via floor-collapse. Forward plan re-scopes the full-trio k-ladder per type, moves spatial to API-only, and prioritizes a scene-coherence diagnostic for negation before the factorial interference runs.

**v4 changelog (April 2026):** Built the full six-module sanity-check suite (`core_plus_confound` scope) on FLUX.2 klein 9B. Added type-aware prompt generator covering all four constraint dimensions (attribute / negation / spatial / numeracy) with an explicit `type` field per constraint. Added Module 3 token-length gate with a **bidirectional** `[1/1.5, 1.5]` ratio requirement. Hit and fixed a token-length confound: the first draft had k=1 prompts at 10-20 tokens vs k=8 at 36-66, blowing the gate for all four types. Introduced a four-tier scene-framing system (HEAVY/MEDIUM/SHORT/MINIMAL prefix + suffix banks) that pads low-k prompts heavily and strips padding at high k. Literature re-check: T2I-CompBench++ does *not* duplicate CompScale — it tests low-complexity (2-3 objects) with no parametric k-ladder, no decay-function fitting, and no directional interference analysis.

---

## Project Summary (TL;DR)

**CompScale** investigates **scaling laws for compositional fidelity in text-to-image (T2I) diffusion models**. The core finding we are pursuing: T2I models that succeed on simple compositional prompts (2-3 constraints) fail catastrophically as constraint count increases (4→8→16), and this degradation follows a predictable mathematical function that model scaling alone does not fundamentally resolve.

**Three contributions:**
1. **CompScale-Bench** — a parametric benchmark that isolates *constraint count* from prompt length, enabling formal scaling law measurement (distinct from DetailMaster which varies token length)
2. **Empirical scaling laws** — fitting mathematical decay functions to compositional fidelity across model scales
3. **Asymmetric semantic interference analysis** — proving that cross-constraint interference is directional (negation degrades numeracy more than numeracy degrades negation)

**Target venue:** EMNLP 2026 (main conference or findings)
**Submission deadline:** ARR cycle targeting ~May 25, 2026 for EMNLP commitment
**Hardware:** 3× 48GB VRAM GPUs (A6000 or equivalent)
**Team:** Solo researcher (Soham), pre-final year CS undergrad at KIIT Bhubaneswar, prior ICCV 2025 acceptance

---

## The Core Research Question

> How does T2I compositional fidelity degrade as the number of simultaneous constraints in a prompt increases, what mathematical function describes this decay, and is the cross-constraint interference symmetric or asymmetric?

### Concrete Examples of the Phenomenon

**Numeracy scaling:**
- ✅ "exactly 3 red apples and 2 green ones" → works (2 object-count-color groups)
- ❌ "exactly 3 red apples, 2 green apples, 7 blue oranges, 8 black bananas" → fails (4 groups with distinct counts + colors)

**Negation scaling:**
- ✅ "a room with no windows" → works (1 negation)
- ❌ "a room with no windows, no fan, no bed, no table, no floor" → fails (5 stacked negations)

**The key insight:** Existing benchmarks either test compositionality at low constraint counts (T2I-CompBench, GenEval) or measure degradation as *prompt length* increases (DetailMaster). Nobody has **parametrically isolated constraint count while controlling prompt length** to fit a mathematical decay function and measure directional interference between constraint types.

---

## Literature Positioning

### Critical Distinction: Why CompScale ≠ DetailMaster

| Dimension | DetailMaster | CompScale-Bench |
|---|---|---|
| **What varies** | Prompt token length (avg 284 tokens) | Constraint count (k=1→16), prompt length held ~constant |
| **Prompt source** | Naturalistic captions from DOCCI/Localized Narratives | Parametrically generated templates + LLM paraphrase |
| **What it proves** | Long prompts are hard; models degrade with token count | Constraint count independently drives failure; decay follows a mathematical function |
| **Evaluation** | Per-category accuracy across 4 dimensions | Per-constraint satisfaction enabling curve fitting |
| **Mathematical model** | None — reports accuracy percentages | Fits S(k) = S(1)·exp(-αk) or alternative functional forms |

**A 30-token prompt can have 8 constraints. A 284-token prompt might have only 4-5 independently verifiable ones.** These are orthogonal axes of difficulty. CompScale isolates the one DetailMaster doesn't.

### What Exists (Do Not Duplicate)

| Paper/Benchmark | What It Does | Relationship to CompScale |
|---|---|---|
| **DetailMaster (May 2025)** | 4,116 long prompts (avg 284 tokens), 4 eval dimensions. Shows ~50% accuracy on complex prompts. | Tests prompt *length*. We test constraint *count* at controlled length. Cite prominently as complementary. |
| **T2I-CoReBench (ICLR 2026)** | Decomposes complex outcomes into atomic verification questions. | Similar per-constraint eval method. We add parametric scaling + mathematical fitting. |
| **PRISM (ICLR 2026)** | Compositional prompt decomposition for long-text-to-image. Energy-based conjunction of sub-prompt outputs. | **This killed our ConDecomp contribution.** Do NOT propose sequential decomposition as a fix. |
| **PASTA (ICML 2025)** | Sequential T2I generation via multimodal RL agent. | Further saturates the decomposition space. |
| T2I-CompBench++ (TPAMI 2025) | 8K prompts, attribute binding, numeracy, spatial. | Tests at low complexity (2-3 objects). No scaling curves. |
| GenEval (NeurIPS 2023) | Object-focused eval: co-occurrence, position, count, color. | Max ~2-3 objects. No degradation curves. |
| "Right Looks, Wrong Reasons" (survey, 2025) | Survey identifying compositional collapse when primitives combine. | Position paper — no benchmark, no math, no scaling laws. Motivates our work. |
| "Relations, Negations, and Numbers" (2024) | Analyzes negation/numeracy/relations in DALL-E 3 vs grounded diffusion. | No systematic scaling. No mathematical formalization. |
| "Vision-Language Models Do Not Understand Negation" (CVPR 2025) | Shows CLIP/VLMs can't handle negation. | Understanding, not generation. Supports our NLP framing. |
| Attend-and-Excite (SIGGRAPH 2023) | Fixes catastrophic neglect via attention excitation. | Works for 2-3 objects. Untested at k>4. Include as baseline reference. |
| RPG (ICML 2024) | LLM recaption + regional diffusion. | Decomposition approach — related to PRISM lineage. |
| NAG (NeurIPS 2025) | Normalized attention guidance for negative prompting. | Relevant baseline for negation handling. |
| Dynamic Negative Guidance (ICLR 2025) | Posterior-aware negative prompting. | Relevant baseline for negation handling. |

### The Gap We Fill

No existing work provides:
1. A benchmark that **parametrically isolates constraint count from prompt length** with a controlled scaling ladder
2. **Mathematical decay functions** (exponential, sigmoid, power law) fitted to constraint-count vs. satisfaction curves across model architectures and scales
3. **Quantified asymmetric interference** between constraint types using factorial ANOVA, proving directional attention collapse in text encoders

---

## Three Contributions

### Contribution 1: CompScale-Bench

A parametric benchmark for measuring compositional scaling. Prompts are generated by controlled templates + LLM paraphrase for naturalness. **Key distinction from DetailMaster:** we hold prompt token length approximately constant (~30-60 tokens) while varying constraint count k from 1 to 16.

**Constraint Dimensions:**

| Dimension | Example at k=1 | Example at k=4 | Example at k=8 |
|---|---|---|---|
| **Numeracy** | "3 red apples" | "3 red apples, 2 green pears, 5 blue oranges, 1 yellow banana" | 8 distinct object-count-color groups |
| **Negation** | "a kitchen with no refrigerator" | "a kitchen with no refrigerator, no microwave, no toaster, no blender" | 8 negated elements |
| **Attribute Binding** | "a red car" | "a red car, a blue truck, a green bicycle, a yellow bus" | 8 object-color bindings |
| **Spatial Relations** | "a cat on a mat" | "a cat on a mat, a dog under a table, a bird above a lamp, a fish inside a bowl" | 8 spatial relations |
| **Cross-product** | Single dimension, k=1 | 2 numeracy + 2 negation | 3 numeracy + 3 negation + 2 spatial |

**Scaling ladder:** k ∈ {1, 2, 3, 4, 6, 8, 12, 16} constraints per prompt.

**Per-configuration:** ~50 prompts (template-generated + LLM-paraphrased for naturalness).

**Total:** ~3,000-4,000 prompts.

**Critical design constraint:** Prompt token length must be controlled. A k=1 prompt and a k=8 prompt should both be roughly 25-60 tokens. This is achieved by having k=1 prompts include neutral scene-setting filler ("In a brightly lit kitchen, there are 3 red apples on the counter") while k=8 prompts pack constraints tightly ("3 red apples, 2 green pears, 5 blue oranges, 1 yellow banana, no grapes, no watermelon, a cat left of the fruit bowl, a vase behind the bananas"). Run token-length distribution analysis to confirm overlap.

**Negation object restrictions:** All negation targets must be **additive/modular/accessory objects**, not structural elements. "No refrigerator" is valid. "No floor" is not — it tests physical prior override, not logical negation processing. Curate an ontology of ~100 valid negation targets per scene type.

**Evaluation per prompt:**
- Generate N=4 images per prompt per model
- Automated eval: VQA-based per-constraint verification using a calibrated VLM
- Per-constraint satisfaction rate → aggregate for overall prompt satisfaction
- Human eval on a stratified subset (~500 prompts × 3 models)

**Paraphrase Constraint Preservation Protocol:**
After LLM paraphrasing, feed the natural-language prompt into a second model in zero-shot extraction mode. It must reconstruct the original constraint list from scratch. If the reconstructed list doesn't achieve perfect one-to-one mapping with the ground-truth constraints, **discard the paraphrase and regenerate.** This bidirectional translation test guarantees that a prompt labeled k=8 genuinely contains 8 unambiguous, independently verifiable constraints.

### Contribution 2: Empirical Scaling Laws

**The core scientific claim:** Compositional fidelity decays as a predictable mathematical function of constraint count k, and the functional form is consistent across model scales (only the parameters change).

**Mathematical formalization:** Let S_T(k) be the satisfaction rate for constraint type T at complexity level k.

Test candidate functional forms:
- **Exponential decay:** S_T(k) = S_T(1) · exp(-α_T · k) — implies each constraint independently adds a constant failure probability
- **Sigmoid/logistic:** S_T(k) = 1 / (1 + exp(β_T · (k - k_0))) — implies a hard capacity threshold at k_0 where performance collapses
- **Power law:** S_T(k) = S_T(1) · k^(-γ_T) — implies diminishing marginal difficulty per additional constraint

The choice of best-fit form has **mechanistic implications:**
- Exponential → independent token interference in cross-attention
- Sigmoid → hard bottleneck in text encoder latent capacity
- Power law → soft saturation, model partially adapts to complexity

**Define "constraint capacity" C_T** as the maximum k where S_T(k) > 0.5 (or another threshold). Plot C_T against model parameter count. If C_T ∝ log(params), scaling alone will never solve compositionality. If C_T ∝ params^δ, there's hope.

**Models to evaluate (UPDATED — March 2026):**

| Model | Params | Architecture | Access | VRAM (est.) | Role in Analysis |
|---|---|---|---|---|---|
| **FLUX.2 [klein] 4B** | 4B | Rectified flow transformer | Open (Apache 2.0), local | ~13GB native BF16 | Lower bound, flow scaling |
| **FLUX.2 [klein] 9B** | 9B | Rectified flow transformer | Open (non-commercial), local | ~20GB native BF16 | Mid-tier, flow scaling |
| **FLUX.2 [dev] 32B** | 32B | Rectified flow transformer | Open (non-commercial), local | ~40GB with FP8 quant + offload | Upper bound, flow scaling |
| **SDXL** | 3.5B | UNet + CLIP+OpenCLIP | Open, local | ~12GB | Legacy baseline for context |
| **Janus Pro 7B** | 7B | Unified multimodal (joint embedding) | Open, local | ~16GB | Tests unified vs. bottlenecked cross-attention |
| **DALL-E 3** | Unknown | Unknown (frontier) | API (~$0.04/image) | N/A | Frontier closed-source reference |
| **Gemini Imagen 3** | Unknown | Unknown (frontier) | API | N/A | Frontier closed-source reference |
| **GPT Image-1** | Unknown | Unknown (frontier) | API | N/A | Latest frontier reference |

**Why FLUX.2 is the core scaling trio:** Same architecture family (rectified flow transformer), three parameter tiers (4B/9B/32B). This isolates the effect of parameter scaling on the decay constant α while controlling for architecture. This is the cleanest scaling law experiment possible with open-weight models.

**Legacy models (SDXL):** Include in one table for historical context. Do NOT make them central.

**Key plots to produce:**
1. **THE figure of the paper:** Constraint count k (x-axis) vs. overall satisfaction rate (y-axis), one curve per model, with fitted decay functions overlaid.
2. Per-constraint-type satisfaction faceted by type (numeracy, negation, spatial, attribute).
3. Model scale (params, log scale) vs. constraint capacity C_T.
4. Residual analysis showing goodness-of-fit for exponential vs. sigmoid vs. power law.

### Contribution 3: Asymmetric Semantic Interference Analysis

**(Promoted from sub-component to full contribution, replacing killed ConDecomp)**

**The hypothesis:** Cross-constraint interference is **directional**. Negation constraints act as destructive vectors in the attention mechanism, degrading adjacent nominal phrase processing more severely than numeracy or spatial constraints do.

**Factorial experimental design:**

Generate prompts with controlled mixtures:
- k_num numeracy constraints only (k_num ∈ {2, 4, 6})
- k_neg negation constraints only (k_neg ∈ {2, 4, 6})
- k_spat spatial constraints only (k_spat ∈ {2, 4, 6})
- k_num numeracy + k_neg negation (all 9 combinations)
- k_num numeracy + k_spat spatial (all 9 combinations)
- k_neg negation + k_spat spatial (all 9 combinations)

Total: ~50 prompts per cell × 27 cells = ~1,350 prompts for the interference analysis alone.

**Statistical analysis:**

Use **two-way ANOVA** to decompose variance:
- Main effect of numeracy difficulty
- Main effect of negation difficulty
- **Interaction effect** (this IS the interference signal)

The interference penalty I_{A→B} is defined as:
```
I_{A→B} = P(B_alone) - P(B | A_present)
```
where P(B_alone) is satisfaction rate of B-type constraints when presented alone, and P(B | A_present) is satisfaction rate of B-type constraints when A-type constraints are also in the prompt.

**The key finding we're hunting for:** I_{neg→num} >> I_{num→neg}. If true, this proves that negation tokens systematically corrupt the cross-attention maps that serve numeracy constraints, but not vice versa. This connects directly to:
- Rarity of negation in training data (text encoder hasn't learned robust negation representations)
- Cross-attention diffusion: negation requires *suppressing* features, which corrupts the surrounding semantic space in continuous latent models

**Why this is the strongest NLP contribution:** This transforms the paper from "models fail at hard prompts" (known) to "we can predict *which* constraint combinations will fail and *by how much*, and the directionality reveals fundamental asymmetries in how text encoders process discrete logic" (novel).

**Optional extension (if time permits):** Probe the text encoder internals directly. Extract attention maps from FLUX.2's Mistral-3 text encoder at each constraint count level. Visualize how attention mass distributes across constraint tokens as k increases. Show that negation tokens ("no") receive disproportionately low attention weight at high k, confirming the mechanism behind the interference asymmetry.

---

## Technical Implementation Details

### Prompt Generation Pipeline

```python
# Template-based generation with controlled complexity
# Then LLM paraphrase with bidirectional verification

NUMERACY_TEMPLATE = "{count} {color} {object}"
NEGATION_TEMPLATE = "no {object}"
SPATIAL_TEMPLATE = "{object_a} {relation} {object_b}"
ATTRIBUTE_TEMPLATE = "a {color} {size} {object}"

# Negation targets: ONLY additive/modular objects, NOT structural elements
VALID_NEGATION_TARGETS = {
    "kitchen": ["refrigerator", "microwave", "toaster", "blender", "kettle",
                 "dishwasher", "cutting board", "fruit bowl", "spice rack"],
    "bedroom": ["lamp", "alarm clock", "bookshelf", "rug", "curtains",
                 "mirror", "nightstand", "plant", "painting"],
    "street":  ["car", "bicycle", "mailbox", "bench", "trash can",
                "streetlight", "bus stop", "fire hydrant", "parking meter"],
}

def generate_prompt(constraint_type: str, k: int, scene: str) -> dict:
    """Generate a prompt with exactly k constraints of given type.
    Returns prompt string + ground truth constraint list for evaluation.
    Controls token length to ~30-60 tokens regardless of k."""
    ...

def paraphrase_and_verify(template_prompt: str, constraints: list, llm, verifier_llm) -> str:
    """
    1. LLM paraphrases template into natural language
    2. Verifier LLM extracts constraints from paraphrase in zero-shot
    3. Compare extracted vs. original constraints
    4. If mismatch: discard and retry (max 3 attempts)
    5. If 3 failures: keep template version, flag for review
    """
    paraphrased = llm.paraphrase(template_prompt)
    extracted = verifier_llm.extract_constraints(paraphrased)
    if constraints_match(extracted, constraints):
        return paraphrased
    else:
        return retry_or_fallback(...)

def verify_token_length_distribution():
    """After generating all prompts, verify that token length distributions
    overlap across k levels. If k=16 prompts are systematically longer,
    the scaling analysis is confounded."""
    ...
```

### Evaluation Pipeline

```python
def verify_constraint(image, constraint: dict, vlm) -> bool:
    """Per-constraint verification via VQA."""
    if constraint["type"] == "numeracy":
        q = f"How many {constraint['color']} {constraint['object']}s are in this image? Answer with just a number."
        answer = vlm.ask(image, q)
        return parse_int(answer) == constraint["expected_count"]
    elif constraint["type"] == "negation":
        q = f"Is there a {constraint['object']} in this image? Answer yes or no."
        answer = vlm.ask(image, q)
        return answer.lower().strip() == "no"
    elif constraint["type"] == "spatial":
        q = f"Is the {constraint['object_a']} {constraint['relation']} the {constraint['object_b']}? Answer yes or no."
        answer = vlm.ask(image, q)
        return answer.lower().strip() == "yes"
    ...

def compute_satisfaction(image, constraints: list, vlm) -> dict:
    results = [verify_constraint(image, c, vlm) for c in constraints]
    return {
        "per_constraint": results,
        "satisfaction_rate": sum(results) / len(results),
        "all_satisfied": all(results),
        "by_type": group_results_by_type(results, constraints)
    }
```

### Evaluator Calibration Protocol (MANDATORY — Run Before Main Eval)

The VLM evaluator must be validated against synthetic ground truth before trusting it at high k.

```python
def calibrate_evaluator(vlm, ground_truth_dataset):
    """
    ground_truth_dataset: 500-1000 images with mathematically guaranteed
    object counts, colors, positions. Source: Blender renders OR
    manually verified DALL-E 3 generations at low-k.

    For each k level (1, 2, 4, 8, 12, 16):
        - Run VLM on images with known ground truth
        - Compute VLM accuracy at each k
        - If accuracy < 90% at k=N, VLM is DISQUALIFIED for k >= N
        - For disqualified k levels: use human annotation or
          weight scores by inverse of VLM error rate
    """
    calibration_results = {}
    for k in [1, 2, 4, 8, 12, 16]:
        subset = ground_truth_dataset.filter(constraint_count=k)
        vlm_accuracy = evaluate_vlm_on_ground_truth(vlm, subset)
        calibration_results[k] = vlm_accuracy
        if vlm_accuracy < 0.90:
            print(f"WARNING: VLM accuracy {vlm_accuracy:.2f} at k={k}. "
                  f"Human annotation required for k>={k}.")
    return calibration_results
```

### Model Inference Setup

```python
# FLUX.2 family — the core scaling trio

# GPU 0: High-throughput node (small/mid models, run sequentially)
# FLUX.2 klein 4B (~13GB), FLUX.2 klein 9B (~20GB), SDXL (~12GB), Janus Pro 7B (~16GB)

# GPU 1: FLUX.2 dev 32B (dedicated — needs FP8 + offload)
from diffusers import Flux2Pipeline
pipe = Flux2Pipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev",
    torch_dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True)
)
pipe.enable_model_cpu_offload()

# GPU 2: Free for additional local model OR VLM evaluator
# Option A: Run Infinity-8B (VAR architecture) if time permits
# Option B: Run InternVL2-26B for local evaluation
# Option C: Keep free for API generation management

# API models (DALL-E 3, Gemini Imagen 3, GPT Image-1):
# No GPU needed — rate-limited API calls
# Budget estimate: ~$100-200 for 3 APIs × ~4000 prompts × 1 image each
```

### Statistical Analysis

```python
import scipy.stats as stats
from scipy.optimize import curve_fit
import numpy as np

# Contribution 2: Fit decay functions
def exponential_decay(k, s0, alpha):
    return s0 * np.exp(-alpha * k)

def sigmoid_decay(k, beta, k0):
    return 1.0 / (1.0 + np.exp(beta * (k - k0)))

def power_law_decay(k, s0, gamma):
    return s0 * k ** (-gamma)

def fit_and_compare(k_values, satisfaction_values):
    """Fit all three forms, compare AIC/BIC to select best model."""
    fits = {}
    for name, func in [("exp", exponential_decay),
                         ("sigmoid", sigmoid_decay),
                         ("power", power_law_decay)]:
        try:
            popt, pcov = curve_fit(func, k_values, satisfaction_values, maxfev=10000)
            residuals = satisfaction_values - func(k_values, *popt)
            aic = compute_aic(len(k_values), residuals, len(popt))
            fits[name] = {"params": popt, "aic": aic, "residuals": residuals}
        except RuntimeError:
            fits[name] = None
    return fits

# Contribution 3: ANOVA for interference
def compute_interference_anova(results_df):
    """
    results_df columns: k_numeracy, k_negation, satisfaction_numeracy, satisfaction_negation

    Two-way ANOVA:
    - Factor A: k_numeracy level (2, 4, 6)
    - Factor B: k_negation level (0, 2, 4, 6)  # 0 = absent
    - Response: satisfaction_numeracy

    The interaction term (A×B) IS the interference signal.
    """
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    model = ols('satisfaction_numeracy ~ C(k_numeracy) * C(k_negation)', data=results_df).fit()
    anova_table = anova_lm(model, typ=2)
    return anova_table

def compute_interference_penalty(results_df):
    """
    I_{neg→num} = P(num_alone) - P(num | neg_present)
    I_{num→neg} = P(neg_alone) - P(neg | num_present)

    If I_{neg→num} >> I_{num→neg}, interference is asymmetric.
    Use bootstrap CIs to test significance.
    """
    ...
```

### Key Libraries

```
diffusers >= 0.32.0       # Must support FLUX.2
transformers >= 4.45.0
torch >= 2.3.0
accelerate
bitsandbytes              # FP8/NF4 quantization
scipy                     # Curve fitting
statsmodels               # ANOVA
Pillow
openai                    # DALL-E 3 / GPT Image-1 API
google-generativeai       # Gemini Imagen API
vllm or ollama            # Local LLM for prompt paraphrasing
matplotlib, seaborn       # Publication figures
```

### Directory Structure

```
compscale/
├── CLAUDE.md                    # This file
├── benchmark/
│   ├── generate_prompts.py      # Template-based prompt generation
│   ├── paraphrase.py            # LLM paraphrase + bidirectional verification
│   ├── negation_ontology.json   # Valid negation targets per scene type
│   ├── prompts/                 # Generated prompt JSONs by type and k
│   ├── validate_prompts.py      # Verify constraint counts, token lengths
│   └── token_length_analysis.py # Confirm length doesn't confound k
├── calibration/
│   ├── generate_ground_truth.py # Blender renders or manual GT collection
│   ├── calibrate_vlm.py         # Evaluator confidence bounds per k
│   └── calibration_report.json  # Per-k VLM accuracy
├── generation/
│   ├── generate_images.py       # Main generation script (multi-model)
│   ├── model_configs.yaml       # Model names, dtypes, GPU assignments
│   ├── api_generate.py          # DALL-E 3 / Gemini / GPT Image-1 API
│   └── outputs/                 # model_name/prompt_id/img_{0..3}.png
├── evaluation/
│   ├── vlm_verify.py            # VQA-based constraint verification
│   ├── metrics.py               # Aggregate metrics computation
│   ├── human_eval/              # Interface + results for high-k subset
│   └── results/                 # Per-model, per-k CSVs
├── analysis/
│   ├── fit_scaling_law.py       # Curve fitting (exp, sigmoid, power law)
│   ├── interference_anova.py    # Factorial ANOVA for asymmetric interference
│   ├── scaling_curves.py        # Plot THE figure: k vs satisfaction per model
│   ├── constraint_capacity.py   # C_T vs model params plot
│   └── figures/                 # Publication-ready figures
└── paper/
    ├── main.tex
    ├── figures/
    └── tables/
```

---

## Writing/Framing Notes for the Paper

### Title Options
- "CompScale: Scaling Laws for Compositional Fidelity in Text-to-Image Generation"
- "How Many Constraints Can You Handle? Compositional Scaling Laws for Text-to-Image Models"
- "The Law of Compositional Decay: Measuring Constraint Interference in Vision-Language Architectures"

### EMNLP Framing (Critical — Read This Before Writing Anything)

This is NOT a vision paper. This is NOT a benchmark paper. This is a **computational linguistics paper that uses image generation as a downstream diagnostic tool to probe text encoder limitations.**

Frame every section through the NLP lens:
- **Introduction:** Root in compositional generalization literature (Hupkes et al., Lake & Baroni). The question isn't "can models make pretty pictures" but "can text encoders represent arbitrary combinations of logical constraints."
- **Related work:** Heavy emphasis on text encoder limitations (CLIP bottlenecks, T5 capacity bounds), multimodal representation learning, cross-attention mechanics. DetailMaster and T2I-CoReBench are complementary benchmarks (different axes). PRISM is a solution approach — we're doing analysis, not engineering.
- **Methodology:** The benchmark is an "adversarial diagnostic probe" for text encoders. The scaling law formalization parallels Kaplan/Chinchilla scaling laws for LLMs.
- **Results — Scaling laws:** "The decay constant α reveals the *linguistic fragility index* of each constraint type within the text encoder's representation space."
- **Results — Interference:** "The asymmetric interference proves that continuous cross-attention cannot implement discrete logical operations (negation) without corrupting adjacent semantic representations."
- **Discussion:** Connect to broader NLP questions about compositionality in neural networks, discrete vs. continuous representations, and whether scaling is sufficient for compositional generalization.

### Anticipated Reviewer Concerns

| Concern | Pre-emptive Response |
|---|---|
| "How is this different from DetailMaster?" | DetailMaster varies prompt *length*. We vary constraint *count* at controlled length. A 30-token prompt with 8 constraints and a 284-token prompt with 4 constraints test different things. We cite DetailMaster as complementary and show our prompts are ~30-60 tokens regardless of k. |
| "Bigger models will solve this" | We test FLUX.2 at 4B/9B/32B + frontier APIs. If α merely shifts but the functional form persists, scaling alone is insufficient. This is our central empirical finding. |
| "Why is this NLP and not CV?" | The failure is in the text encoder. We analyze decay constants per linguistic category and prove asymmetric interference between semantic types. The image is just a readout of the encoder's representational capacity. |
| "PRISM already solves this" | PRISM is an engineering solution (decompose and compose). We are doing empirical science (measure, formalize, explain). Our work motivates *why* approaches like PRISM are necessary by quantifying the fundamental limits they work around. |
| "Template prompts aren't natural" | We LLM-paraphrase all templates with bidirectional constraint verification. We include both versions and show results are consistent. |
| "VLM evaluator might be unreliable at high k" | We run a mandatory calibration protocol against synthetic ground truth and report evaluator confidence bounds per k. High-k evaluations below the confidence threshold use human annotation. |
| "This is just a benchmark paper" | No — we provide mathematical scaling laws (Contribution 2, empirical science) and asymmetric interference analysis (Contribution 3, mechanistic interpretability). The benchmark is a tool, not the contribution. |

---

## Timeline (Targeting ARR ~May 25, 2026)

| Phase | Weeks | Tasks |
|---|---|---|
| **Phase 1: Setup** | 1-2 | Finalize prompt templates + negation ontology. Run paraphrase + verification loop. Build calibration dataset. Calibrate VLM evaluator. |
| **Phase 2: Generation** | 3-5 | Parallel generation across 3 GPUs. GPU 0: FLUX.2 4B/9B, SDXL, Janus Pro. GPU 1: FLUX.2 32B (FP8). GPU 2/API: DALL-E 3, Gemini, GPT Image-1. |
| **Phase 3: Evaluation** | 6-8 | VLM verification pipeline. Human annotation for high-k where VLM falls below confidence threshold. Inter-annotator agreement. |
| **Phase 4: Analysis** | 9-10 | Fit decay functions + model selection (AIC/BIC). Compute interference ANOVA. Generate all plots. |
| **Phase 5: Writing** | 11-12 | Draft paper with strict NLP framing. |
| **Phase 6: Polish** | 13 | Camera-ready figures. Submit to ARR. |

---

## Open Questions (For Brainstorming Sessions)

1. **What functional form will win?** Exponential, sigmoid, or power law? If sigmoid — what determines k_0 (the collapse threshold)? Is it related to the number of attention heads in the text encoder?
2. **Token-length confound:** Can we truly hold token length constant from k=1 to k=16? At k=16, even terse prompts might be 80+ tokens. Consider normalizing by tokens-per-constraint rather than absolute length.
3. **Should we probe text encoder internals?** Extracting attention maps from FLUX.2's Mistral-3 encoder would massively strengthen the NLP story but adds engineering complexity. Consider doing this for FLUX.2 klein 4B where memory is less constrained.
4. **Which VLM for evaluation?** GPT-4o is most reliable but expensive at scale. InternVL2-26B is strong and local (~15GB). Run calibration on both and pick whichever maintains >90% accuracy to the highest k.
5. **Prompt-order effects:** Does "3 red apples, no bananas" degrade differently from "no bananas, 3 red apples"? If yes, this is evidence of positional encoding effects in the text encoder. Worth testing on a subset.
6. **Autoregressive baselines:** Should we include a VAR model (e.g., Infinity-8B) to test whether discrete token prediction handles constraint scaling differently? Strong scientific motivation but adds engineering burden. Include if time permits after core analysis.
7. **Cross-product factorial size:** 3 types × 3 levels = 27 pairwise cells. Triple interactions add another 27. Decide if triple interactions are worth the prompt budget.

---

## Related Repos and Resources

- [DetailMaster](https://huggingface.co/datasets/datajuicer/DetailMaster) — Complementary benchmark, cite prominently
- [PRISM](https://iclr.cc/virtual/2026/poster/10007837) — ICLR 2026, compositional decomposition (related work)
- [T2I-CompBench++](https://github.com/Karine-Huang/T2I-CompBench) — Low-complexity baseline
- [GenEval](https://github.com/djghosh13/geneval) — Object detection-based eval
- [FLUX.2 official repo](https://github.com/black-forest-labs/flux2) — Inference code
- [FLUX.2 dev weights](https://huggingface.co/black-forest-labs/FLUX.2-dev) — 32B model
- [FLUX.2 klein weights](https://huggingface.co/black-forest-labs) — 4B (Apache 2.0) and 9B
- [diffusers](https://huggingface.co/docs/diffusers) — Quantization + FLUX.2 guides

---

## Contact / Context

- **Researcher:** Soham, KIIT Bhubaneswar, pre-final year CS
- **Prior work:** ICCV 2025 paper (accepted), GeoAgent (ACL submission), SEA-Web (adversarial web agent benchmark)
- **Compute:** 3× 48GB VRAM GPUs, personal compute budget
- **This document last updated:** April 2026, v4

---

## Pilot Sanity Check Findings (March 25, 2026)

### Setup
- Model: FLUX.2 klein 4B, 4 steps, guidance_scale=1.0
- Evaluator: Gemini 3 Flash Preview (VQA-based counting)
- 40 numeracy prompts: 10 per k-level (k=1,2,4,8), N=4 images per prompt

### Key Finding: Per-Object Count Difficulty Dominates Over Constraint Count k

The pilot revealed that **per-object count difficulty** (how many of a single object are requested) is a far stronger predictor of failure than constraint count k. The k variable showed no clean decay because count distributions were uncontrolled across k levels.

**Satisfaction by expected per-object count (the real signal):**
| Expected count | Accuracy |
|---|---|
| 1 | 66.7% |
| 2 | 36.2% |
| 3 | 26.4% |
| 4 | 22.5% |
| 5 | 13.9% |
| 6 | 5.0% |
| 7 | 0.0% |

**Why the k-level analysis was flat/non-monotonic:**
- k=1 prompts included hard counts (5 tulips, 6 cookies) → low baseline
- k=4 prompts had many count=1 constraints (gift box, towel roll) → inflated scores
- Result: k=4 (0.431) outscored k=1 (0.375), masking any k-driven decay

### Object Countability Tiers (Empirical)

Objects fall into clear tiers based on generation+evaluation reliability:

**Tier 1 — Highly countable (75-100% accuracy at count 1-3):**
pumpkin, backpack, gift box, stapler, calculator, toothbrush, yarn ball, spool of thread, towel roll, bell pepper, bus, teddy bear, sponge, velvet cake, toast slice, cheese cube, mug, cupcake, bottle, cucumber

**Tier 2 — Moderately countable (40-70%):**
apple, tomato, pen, button, car, mushroom, cake, macaron, banana, pear, tangerine, folder, candle, sticky note

**Tier 3 — Uncountable (0-25%, DO NOT USE):**
bead, sequin, cotton ball, feather, forget-me-not, daisy, lavender, fern, marigold, sunflower, confetti, grape cluster, sheet of paper, paperclip, felt square, hair tie, croissant, meringue, lemon tart, mint cookie, cookie, soap bar, crayon, party hat, napkin, block, dinosaur, streamer

**Root causes for Tier 3 failure:**
- Objects that naturally cluster/scatter (beads, confetti, cotton balls)
- Objects that are small and hard to individuate (sequins, paperclips, feathers)
- Flowers that grow in clusters, not discrete units (daisies, forget-me-nots, lavender)
- Food items with irregular shapes that blend together (croissants piled up, cookies on a tray)

### Pilot v2: Controlled Count Distribution (March 25, 2026)

**Changes from v1:** Restricted all counts to 1-3, used only Tier 1-2 objects, balanced count distribution across k levels.

**Results:**
| k | Per-constraint satisfaction | All-satisfied rate |
|---|---|---|
| 1 | 35.0% | 35.0% |
| 2 | 36.2% | 10.0% |
| 4 | 43.8% | 2.5% |
| 8 | 31.2% | 0.0% |

**Curve fitting:** R²=0.10 for best fit (exponential). No meaningful per-constraint decay.

**Independence analysis:** With p=0.366 mean per-constraint satisfaction, the all-satisfied decay matches the independence model (p^k) closely:
| k | Predicted (p^k) | Observed | Ratio |
|---|---|---|---|
| 1 | 0.366 | 0.350 | 0.96 |
| 2 | 0.134 | 0.100 | 0.75 |
| 4 | 0.018 | 0.025 | 1.40 |
| 8 | 0.0003 | 0.000 | — |

**Key conclusion:** FLUX.2 klein 4B is at floor performance for numeracy (~35% per-constraint). The all-satisfied decay is just binomial math (p^k), not interference. **Cannot measure cross-constraint interference when baseline is this low.** Need either:
1. An easier constraint type with p > 0.7 at k=1 (→ attribute binding pilot)
2. A stronger model with better numeracy baseline (→ frontier API models later)

### Pilot v3: Attribute Binding (March 25, 2026) — SUCCESS

**Setup:** Same model (FLUX.2 klein 4B), attribute binding constraints ("a red car and a blue truck"), k={1,2,4,8}, 10 prompts per k, N=4 images. Evaluator: Gemini 3 Flash Preview asking "What color is the {object}?"

**Results:**
| k | Per-constraint satisfaction | All-satisfied rate |
|---|---|---|
| 1 | 97.5% | 97.5% |
| 2 | 98.8% | 97.5% |
| 4 | 97.5% | 90.0% |
| 8 | 93.8% | **65.0%** |

**Curve fitting:** Best fit is **sigmoid** (R²=0.90, k₀=21.2). Exponential R²=0.81, power law R²=0.55.

**Independence analysis — THE KEY FINDING:**
| k | Predicted (p^k) | Observed | Ratio |
|---|---|---|---|
| 1 | 0.969 | 0.975 | 1.01 |
| 2 | 0.939 | 0.975 | 1.04 |
| 4 | 0.881 | 0.900 | 1.02 |
| 8 | 0.776 | **0.650** | **0.84** |

**Interpretation:**
- k=1 through k=4: ratios ≈ 1.0 → constraints fail independently, no interference
- **k=8: ratio = 0.84 → active cross-constraint interference**, 12.6pp worse than independence predicts
- Sigmoid best fit implies **hard capacity bottleneck** in text encoder, not gradual degradation
- This is the first empirical evidence of the phenomenon CompScale is designed to measure

**What this validates:**
1. The pipeline works end-to-end (generation → VLM eval → curve fitting → independence test)
2. Attribute binding has high enough baseline (p=0.97) for interference to be detectable
3. Interference IS real and detectable at k=8 on FLUX.2 klein 4B
4. The sigmoid functional form is a meaningful finding (implies hard capacity, not gradual decay)

**What's needed next:**
1. Scale up attribute binding: more prompts per k, add k={3, 6, 12, 16} to the scaling ladder
2. Add more constraint types: negation, spatial relations (on models with strong enough baselines)
3. Test on more models (FLUX.2 klein 9B, FLUX.2 dev 32B) to see how α/k₀ shift with scale
4. Test on frontier APIs (DALL-E 3, GPT Image-1) for numeracy where baseline should be higher
5. Run the factorial interference analysis (Contribution 3) with attribute × numeracy cross-products

### Design Implications for CompScale-Bench

1. **CRITICAL: Hold per-object counts constant across all k levels.** Use only counts 1-3 for all constraints at all k. This isolates k as the sole independent variable.
2. **Use only Tier 1-2 objects.** Curate a verified countable object list (~30-40 objects).
3. **The "count difficulty" decay curve itself is a finding.** Consider reporting it as evidence of fundamental numeracy limitations — even a single constraint with count>4 fails reliably.
4. **VLM evaluator reliability:** Gemini 3 Flash Preview is reliable for counts 1-3 of Tier 1 objects. Unreliable for counts >5 or Tier 3 objects (impossible to distinguish evaluator failure from generation failure).
5. **Numeracy on weak models is at floor.** For the scaling law analysis to work, need per-constraint baseline > 70%. Attribute binding is the better first dimension to test on small models. Save numeracy for frontier APIs.
6. **Independence test is essential.** Always compare observed all-satisfied rates against the p^k independence model. The interesting finding is when observed < predicted (active interference), not when it merely follows binomial decay.

---

## Sanity Check Suite Build-Out (April 16, 2026)

### Scope and Rationale

Before committing GPU time to the full benchmark, build a single unified sanity suite that (a) demonstrates all four constraint dimensions (attribute / negation / spatial / numeracy) work end-to-end on a model we trust, and (b) closes out the three biggest known confounds that reviewers will raise. Selected scope: **`core_plus_confound`** — six modules gated with pass/fail thresholds:

| Module | Purpose | Gate |
|---|---|---|
| **1. Baseline viability** | Confirm per-constraint satisfaction is detectable for all four types (not at floor, not at ceiling) | p ≥ 0.5 at k=1 per type |
| **2. VLM reliability** | Verify Gemini 3 Flash Preview agrees with humans on all four VQA question templates, especially negation yes/no | Agreement > 0.90 per type |
| **3. Token-length distribution** | Close the DetailMaster-style confound that prompt length covaries with k | median(k=8) / median(k=1) ∈ [1/1.5, 1.5] per type |
| **4. Prompt-order permutation** | Confirm constraint ordering within a prompt does not drive the signal | Mean \|Δ satisfaction\| per base prompt < 0.10 |
| **5. Seed variance** | Confirm N=4 images per prompt is enough — CI width acceptable | Median bootstrap 95% CI width at N=4 < 0.15 |
| **6. Independence reproducibility** | Replicate the v3 attribute-binding finding (ratio < 1 at k=8) on klein 9B | Attribute k=8 observed/predicted ratio < 1, CI upper bound < 1 |

### Model Choice: FLUX.2 klein 9B

The pilot v3 result (attribute binding on klein 4B showing sigmoid decay + k=8 interference) is already re-provable at 4B. Running the suite on **klein 9B** instead serves a second purpose: an early scaling preview. If α or k₀ shifts between 4B and 9B in the sanity suite, that's a free direction signal on what the full scaling trio (4B/9B/32B) will show. Single GPU at 48GB VRAM fits klein 9B comfortably in native bfloat16.

### Literature Re-check: T2I-CompBench / CompBench++ Does Not Duplicate Us

Confirmed via paper + repo review that T2I-CompBench++ (TPAMI 2025) tests attribute binding, object relationships, and numeracy at **low fixed complexity** (2-3 objects), reports accuracy percentages, and does NOT fit decay functions, does NOT have a parametric k-ladder, and does NOT analyze cross-constraint interference. CompScale's three contributions are therefore uniquely ours: parametric k-isolation, mathematical scaling-law fitting, and asymmetric interference analysis. Cite T2I-CompBench++ as the low-complexity baseline, not as a competitor.

### Benchmark Infrastructure Built This Session

- **`compscale/benchmark/ontology/`** — three structured JSONs codifying pilot findings:
  - `countable_objects.json` — Tier 1 and Tier 2 countable objects only (Tier 3 explicitly excluded), max count = 3.
  - `negation_targets.json` — scene-scoped additive/modular objects; structural elements ("floor", "ceiling") blacklisted to avoid testing physical prior override.
  - `spatial_objects.json` — objects + surfaces + relations with inverse mappings and per-relation VQA templates.
- **`compscale/benchmark/generate_prompts.py`** — deterministic template generator emitting four pilot JSONs (attribute, negation, spatial, numeracy). Every constraint carries an explicit `type` field so the VLM verifier routes without heuristics. Every prompt stores `scene_prefix` and `scene_suffix` so the permutation module can reshuffle clauses without reparsing strings.
- **`compscale/generation/permute_prompts.py`** — reads base prompts, samples 5 k=4 prompts per type, and emits 3 distinct clause-order permutations per base.
- **`compscale/benchmark/validate_lengths.py`** — Module 3 gate. Tokenizes all prompts with Mistral-Nemo-Base-2407 (Mistral-3-family proxy for FLUX.2's text encoder), reports per-(type, k) min/median/mean/p95/max, emits a violin plot and JSON report, and applies the symmetric 1.5x ratio gate.
- **`compscale/generation/generate_images.py`** (extended) — added `--klein_9b` flag, multi-prompt-file batching, per-model output subdirectories, and **stable per-prompt seeds** derived from `sha256(prompt_id) + base_seed` so extending from N=4 to N=16 for Module 5 does not re-roll existing images.
- **`compscale/evaluation/vlm_verify.py`** (extended) — routes VQA questions on `constraint["type"]`: "What color is the {object}?" for attribute, "How many {color} {object}s..." for numeracy, "Is there a {object}..." (yes/no) for negation, "Is the {a} {relation} the {b}?" (yes/no) for spatial. Backward-compatible with un-typed prompts via `infer_type`.
- **`compscale/evaluation/sample_for_labeling.py`** + **`vlm_reliability.py`** — Module 2 pipeline: stratified sample of 160 (image, constraint) pairs for hand-labeling, then VLM-vs-human agreement with a confusion matrix (negation FP/FN called out separately).
- **`compscale/analysis/seed_variance.py`** — Module 5: subcommand `subset` picks 20 prompts for N=16 extension; subcommand `analyze` bootstraps per-prompt 95% CIs at N=4 and N=8 and applies the width gate.
- **`compscale/analysis/order_permutation.py`** — Module 4 analysis: per-base-prompt |Δ satisfaction| between base and its permuted variants.
- **`compscale/analysis/fit_scaling_law.py`** (extended) — `--type` filter, configurable output filenames, and bootstrap 95% CIs on the independence ratio at each k (for Module 6's CI-upper-bound < 1 check).
- **`compscale/analysis/diagnose_pilot.py`** (extended) — generalized from numeracy-only to all four types via a `_signature` helper that keys aggregation per constraint type.
- **`compscale/analysis/sanity_report.py`** — aggregates all six module gates into `compscale/sanity/sanity_report.json` with a pass/fail row per module.
- **`compscale/sanity/RUN.md`** — sequential execution playbook for the whole suite.

### Module 3 Gate: First Run Failed, Fixed in Session

First validation run on klein 9B (Mistral-Nemo-Base-2407 tokenizer, 260 prompts across the five pilot files):

**Failed results (one-sided gate, ratio ≤ 1.5):**

| type | median k=1 | median k=8 | ratio | status |
|---|---|---|---|---|
| attribute | 17.5 | 36.5 | 2.09 | FAIL |
| negation | 13.0 | 41.5 | 3.19 | FAIL |
| numeracy | 19.5 | 41.0 | 2.10 | FAIL |
| spatial | 10.5 | 66.5 | **6.33** | FAIL |

**Root cause:** the original `scene_prefix` logic gave k=1 only ~10 tokens of filler ("A scene with ... featuring") while k=8 used the same minimal prefix plus eight full constraint clauses. Spatial was the worst because each spatial clause ("a cat on a mat") is ~6 tokens, so k=8 spatial stacked to 52+ clause-tokens alone.

**Fix: four-tier scene framing (prefix + suffix).** Authored four length tiers of hand-written prefix templates (HEAVY ~25 tok, MEDIUM ~16 tok, SHORT ~8 tok, MINIMAL ~4 tok) and two tiers of trailing suffixes (HEAVY ~12 tok, MEDIUM ~6 tok). Tier selection inverts with k:

| k | prefix tier | suffix tier | total filler (tok) |
|---|---|---|---|
| 1 | HEAVY | HEAVY | ~37 |
| 2 | MEDIUM | MEDIUM | ~22 |
| 4 | SHORT | (none) | ~8 |
| 8 | MINIMAL | (none) | ~4 |

All four types now share a single framing bank via a `{scene}` placeholder. Generic types (attribute / numeracy / spatial) fill it from `GENERIC_SCENE_QUALIFIERS` ("living room", "kitchen", etc.); negation fills it with the ontology scene name so scene coherence is preserved. Expected post-fix ratios (ballpark, to be verified): attribute ~1.05, negation ~0.80, numeracy ~1.07, spatial ~1.15 — all within the new symmetric band.

**Gate is now symmetric: `ratio ∈ [1/1.5, 1.5]`.** The old one-sided check would silently pass the inverse confound (k=1 padded longer than k=8), which would also compromise the scaling-law causal story. We want k-dependent *constraint count* to be the only thing varying — length should be held constant in *either* direction.

### Design Implications (additive to earlier pilot findings)

1. **Always store `scene_prefix` AND `scene_suffix` alongside `constraints` in every prompt JSON.** Permutation, diagnostic reporting, and any later rewriter need to rebuild the string without parsing. The session's first permutation implementation tried brittle suffix-of-scene slicing and was replaced with metadata-driven rebuild.
2. **`constraint["type"]` is canonical.** The VLM verifier routes on it; the diagnostic reporter signatures on it; the analysis scripts filter on it. All generators must set it explicitly.
3. **Stable per-prompt seeds enable N extension.** Deriving seeds from `sha256(prompt_id) + base_seed` means Module 5's N=4→N=16 extension re-uses the first four images exactly and only generates the additional twelve. Any future "add more images to this prompt" workflow inherits this for free.
4. **Token-length gate must be bidirectional.** Either direction of length imbalance confounds the k signal.
5. **Tier-based scene framing generalizes beyond the pilot.** The same HEAVY/MEDIUM/SHORT/MINIMAL pattern will be needed for the full k={1,2,3,4,6,8,12,16} ladder; future extension should interpolate tiers at intermediate k (e.g., k=3 gets MEDIUM+SHORT, k=12 gets MINIMAL+none).

---

## Klein 9B Sanity Results (April 20, 2026)

First full results of the six-module sanity suite on FLUX.2 klein 9B. Two result files in `compscale/evaluation/results/`: `sanity_results.json` (attribute, 20 prompts/k) and `sanity_results_neg_spat_num.json` (negation / spatial / numeracy, 10 prompts/k each). 800 images, 3,000 per-constraint VQA judgements, N=4 per prompt across k∈{1,2,4,8}. Evaluator: Gemini 3 Flash Preview (≈60% of calls batched per-(prompt,image)).

### Per-(type, k) table

| type | k | per-constraint sat | all-satisfied |
|---|---|---|---|
| attribute | 1 | 1.000 | 1.000 |
| attribute | 2 | 0.988 | 0.975 |
| attribute | 4 | 0.978 | 0.913 |
| attribute | 8 | 0.966 | **0.738** |
| negation | 1 | 0.675 | 0.675 |
| negation | 2 | 0.913 | 0.825 |
| negation | 4 | 0.463 | 0.050 |
| negation | 8 | 0.594 | 0.000 |
| numeracy | 1 | 0.775 | 0.775 |
| numeracy | 2 | 0.700 | 0.525 |
| numeracy | 4 | 0.600 | 0.125 |
| numeracy | 8 | 0.459 | 0.000 |
| spatial | 1 | 0.425 | 0.425 |
| spatial | 2 | 0.325 | 0.175 |
| spatial | 4 | 0.494 | 0.050 |
| spatial | 8 | 0.428 | 0.000 |

### Module 1 (baseline p ≥ 0.5 at k=1)

| type | p(k=1) | 95% CI | status |
|---|---|---|---|
| attribute | 1.000 | [0.954, 1.000] | PASS (ceiling) |
| numeracy | 0.775 | [0.625, 0.877] | **PASS — vs ~0.35 on 4B pilot v2** |
| negation | 0.675 | [0.520, 0.799] | PASS |
| spatial | 0.425 | [0.285, 0.578] | **FAIL** |

The numeracy leap 4B → 9B (0.35 → 0.775) is the standout single-number finding of this run. On klein 4B numeracy sat at floor and could not be measured for interference; on 9B it is well-posed and shows clean monotone per-constraint decay (0.775 → 0.700 → 0.600 → 0.459). This is the first real direction-of-scaling datapoint we have for the full trio narrative.

### Module 6 (independence gate) — gate is giving false positives

| type | p_k1 | pred k=8 | obs k=8 | ratio | CI upper | pass? |
|---|---|---|---|---|---|---|
| attribute | 1.000 | 1.000 | 0.738 | 0.738 | 0.825 | TRUE |
| negation | 0.675 | 0.043 | 0.000 | 0.000 | 0.000 | TRUE* |
| numeracy | 0.775 | 0.130 | 0.000 | 0.000 | 0.000 | TRUE* |
| spatial | 0.425 | 0.001 | 0.000 | 0.000 | 0.000 | TRUE* |

Only **attribute** is a genuine interference signal: ratio 0.738 with 95% bootstrap CI [0.638, 0.825] entirely below 1.0 and a p^k baseline of exactly 1.0 (no confound from k=1 noise). The other three "pass" only because observed all-satisfied collapses to exactly 0/40 at k=8 → the bootstrap distribution degenerates to {0} → CI upper trivially < 1. This is a **floor-effect false positive**, not evidence of interference.

**Required fix:** add a floor to the Module 6 gate — e.g., require observed all-satisfied ≥ 0.05 *and* CI upper < 1. Equivalently, retire the gate whenever predicted p^k itself is below ≈0.05, since at that point binomial noise dominates any interference signal we can measure with N=40 images. Without this, the gate will auto-pass any constraint type the model simply cannot do at k=8.

### 4B → 9B interference comparison (attribute)

| scale | p_k1 | obs all-sat k=8 | predicted p^8 | ratio |
|---|---|---|---|---|
| klein 4B (pilot v3) | 0.975 | 0.650 | 0.776 | 0.84 |
| klein 9B (this run) | 1.000 | 0.738 | 1.000 | **0.74** |

Absolute all-satisfied improved (65% → 74%) but the ratio worsened (0.84 → 0.74). Reason: 9B's perfect k=1 baseline tightens the independence yardstick to 1.000. **Framing caveat for the paper:** do not report "interference grew with scale." The correct reading is "9B removes the k=1 floor noise, so the interference signal becomes measurable with no confound." Useful methodological point in its own right.

### Decay-function fits

Fit on per-constraint satisfaction vs k (4 datapoints, so treat AIC rankings cautiously):

| type | best form | exp R² | sigmoid R² | power R² | note |
|---|---|---|---|---|---|
| attribute | power | 0.924 | 0.819 | 0.997 | Near-flat line (0.97–1.00). Real decay lives in all-satisfied, not per-constraint. Re-fit on all-satisfied for this type. |
| numeracy | exp | **0.994** | 0.966 | 0.956 | Cleanest scaling-law candidate we have. α = 0.075. |
| negation | exp (by AIC) | 0.227 | 0.206 | 0.216 | All fits garbage. Non-monotonic raw data (see below). |
| spatial | power | 0.098 | −48.1 | 0.111 | At floor; noise. |

### Open issues from this run

1. **Negation non-monotonicity (k=2 beats k=1):** 0.675 → 0.913 → 0.463 → 0.594. With only 10 prompts/k this might be sample noise, but the k=2 bump is large enough to warrant inspection. Candidate explanations: (a) VLM "yes/no, is there a {object}?" question may behave differently when the prompt lists multiple negated objects vs one (evaluator-side bias); (b) single-negation prompts may have weaker scene-framing signal for the VLM to anchor on. Pull hand-labels from Module 2 for the k=1 negation subset to separate generation failure from evaluator failure.
2. **Module 6 gate is broken for floor-collapse types** (see above). Fix required in `sanity_report.py` before the scaling trio run.
3. **Spatial drops out on this model.** p=0.425 at k=1 fails Module 1; cannot be included in the klein-tier scaling-law figure. Either (a) drop spatial from the core contribution for open-weight models and report it only on frontier APIs, or (b) soften the spatial VQA question (current "Is the A {relation} the B?" may be too strict for near-misses).
4. **k=8 saturates all-satisfied to 0 for three of four types.** For the full ladder, k=6 and k=12 will matter more than k=16 for actually *measuring* the curve between floor and ceiling. Consider reallocating prompt budget away from k=16.
5. **Modules 2, 3, 4, 5 not yet reported.** Only Modules 1 + 6 have landed in `evaluation/results/`. Need token-length report, VLM reliability vs hand-labels, order permutation deltas, and seed-variance CI widths before the full benchmark is cleared.

### What's cleared for the scaling trio

- Attribute binding on klein 9B produces a real, bootstrap-significant interference signal — confirms the central phenomenon reproduces across model scales.
- Numeracy on klein 9B is measurable (not at floor) and exponentially decaying — good candidate for the main scaling figure once 4B + 32B data lands.
- The pipeline end-to-end works (prompt gen → generation → batched VQA → bootstrap CI → sanity aggregation).

### What's not cleared

- Module 6 gate logic (floor-effect fix needed)
- Spatial as a klein-tier constraint type
- Negation evaluator reliability (Module 2 hand-labels outstanding)
- Token-length parity post-fix verification on 9B prompts (Module 3 report not yet produced from these prompt files)

---

## Forward Plan — Re-scoping Before the Full Trio (April 20, 2026)

Second-pass analysis on top of the klein 9B sanity results. These are corrections to earlier plan assumptions based on what the 9B data actually shows.

### Module 6 is mis-specified, not floor-buggy

The earlier "floor-effect false positive" framing understates the issue. With N=40 images per (type, k) cell, binomial noise on `observed_all_sat` is ≈ ±0.05. The independence gate can only discriminate interference from p^k when the predicted baseline is well above that noise floor — roughly `p^k ≳ 0.1`. Below that, any observation is consistent with either perfect independence or catastrophic interference.

**Where each type's p^k crosses detectability thresholds (N=40):**

| type | p_k1 | k where p^k ≈ 0.5 (best detection) | k where p^k ≈ 0.1 (below → noise) |
|---|---|---|---|
| attribute | 1.000 | never reaches 0.5 | never reaches 0.1 (can push to k=16) |
| numeracy | 0.775 | k ≈ 2.7 | k ≈ 9.0 |
| negation | 0.675 | k ≈ 1.8 | k ≈ 5.9 |
| spatial | 0.425 | k ≈ 0.8 | k ≈ 2.7 |

**Consequence:** the full trio must use a **per-type k-ladder**, not a uniform ladder. Attribute can exercise the full k ∈ {1,2,3,4,6,8,12,16}. Numeracy/negation budget collapses into k ∈ {1,2,3,4,6,8} with most prompts at k ∈ {2,3,4}. k=12,16 for low-baseline types is wasted GPU.

**Required fix in `sanity_report.py` and `fit_scaling_law.py`:** gate emits `N/A (p^k < 0.1)` rather than `PASS` when the independence baseline is in the noise regime. The CI-upper-bound check is only authoritative above that threshold.

### Contribution 2 anchor is numeracy, not attribute

On klein 9B:

| type | per-constraint range (k=1→8) | best fit | R² | scaling-figure suitability |
|---|---|---|---|---|
| attribute | 1.000 → 0.966 (3.4pp) | power (R²=0.997) | spurious — fitting noise on flat line | NOT a Contribution 2 curve |
| numeracy | 0.775 → 0.459 (32pp) | exp, α=0.075 | 0.994 | cleanest scaling-law candidate |
| negation | non-monotonic | garbage | 0.23 | blocked on scene-coherence diagnostic |
| spatial | 0.425 → 0.428 (at floor) | garbage | 0.10 | excluded |

**Paper restructure:** numeracy carries Contribution 2 (decay form across scale). The central scaling figure is numeracy exponential α plotted across 4B / 9B / 32B + frontier APIs, not the four-type average. Attribute moves fully to Contribution 3 (interference at ceiling) where its 26pp all-satisfied range carries the signal. Per-constraint fits for attribute should be dropped from the paper; only all-satisfied fits are meaningful.

**Practical consequence for the trio:** prioritize numeracy data quality. Bump prompts/k for numeracy to 20 (matching attribute), keep attribute at 20. Paraphrase loop applied to numeracy first.

### Negation non-monotonicity: scene-coherence hypothesis

k=1 → 0.675, k=2 → 0.913 on n=40 and n=80 respectively. Proportion-difference z ≈ 3.6 — not sample noise at 95% confidence. Working hypothesis: **the VLM's "Is there a {object}?" yes/no question carries a strong prior for "no" when the negated object is scene-incoherent** (e.g. "microwave" in a bedroom), inflating per-constraint satisfaction independent of whether the generator actually suppressed the object.

**Diagnostic:** partition current negation k=1 results by scene-coherence of the negated object. If scene-incoherent negations score much higher than scene-coherent ones at k=1, the VLM prior is carrying the signal, not the generator. Test is cheap — re-score from `negation_pilot.json` metadata + existing results, no new generation.

**If confirmed:**
1. Restrict `negation_targets.json` to **scene-coherent objects only** (kitchen + microwave, yes; bedroom + microwave, no).
2. Regenerate the negation pilot with the restricted ontology.
3. Contribution 3's asymmetric interference claim (`I_{neg→num}`) needs re-analysis with the cleaned negation data before any factorial runs.

This is the single highest-leverage action before the full trio. A silent evaluator-prior confound would nuke Contribution 3 in review.

### Spatial: API-only, not dropped

Spatial fails Module 1 on klein 9B (p=0.425). Two options with different implications for paper scope:

| option | effect on Contribution 1 | effect on Contribution 3 | risk |
|---|---|---|---|
| drop entirely | reframe as three types | loses num×spat + neg×spat (2 of 3 cells) | clean but narrative weakens |
| API-only (plan of record) | four types, noting spatial is frontier-only | all three interference pairs preserved at the top of the scaling trio | reviewers may ask why spatial doesn't scale — that's itself a finding |

Plan of record: **spatial stays in the benchmark and is generated on DALL-E 3 / GPT Image-1 / Gemini Imagen only.** Cross-scale analysis drops spatial; the flagship interference analysis at the top tier keeps all three pairs. Softening the spatial VQA template is rejected — eval rigor over rescue.

### N=4 is insufficient for the full trio

Klein 9B Module 1 CI widths at N=40 constraints (10 prompts × 4 images):

- negation k=1: p=0.675, CI=[0.520, 0.799], **width 0.28**
- spatial k=1: p=0.425, CI=[0.285, 0.578], width 0.29
- numeracy k=1: p=0.775, CI=[0.625, 0.877], width 0.25

For the trio's cross-model decay curves, per-cell CI widths below 0.08 are needed to visually separate 4B, 9B, and 32B at typical decay magnitudes. That's ~N=100 constraints per cell. Realized via N=10 prompts × N=10 images at low k, or N=20 prompts × N=5 images, etc. **Bump per-prompt image count to N=8 minimum, N=12 preferred** — let Module 5's seed-variance report dictate the exact number once it runs.

### Revised next-step priority (replaces pre-sanity plan)

Ordered by blocking dependency:

1. **Scene-coherence diagnostic on negation** (notebook, 1–2 hours, zero new compute). Highest information-per-hour action available.
2. **Fix Module 6 gate spec** in `sanity_report.py` and `fit_scaling_law.py` — `N/A` below `p^k = 0.1`, not a floor patch.
3. **Re-fit attribute on all-satisfied** in `fit_scaling_law.py`; keep per-constraint fits for numeracy/negation/spatial.
4. **Run Modules 2, 3, 4, 5** on current data. Module 2 is the evidence base for the negation question — prioritize it.
5. **Regenerate negation prompts** with scene-coherent ontology (if #1 confirms the hypothesis) and re-run the negation pilot on klein 9B.
6. **Re-spec the k-ladder per type** in `generate_prompts.py`. Attribute: {1,2,3,4,6,8,12,16}. Numeracy/negation: {1,2,3,4,6,8}. Spatial: API-only prompt set.
7. **Bump N per prompt** based on Module 5 output (target N=8–12).
8. **Full trio generation** (4B / 9B / 32B on the revised ladder, ~2 weeks).
9. **Frontier API generation** for spatial + top-of-ladder attribute/numeracy/negation, + DALL-E 3 / GPT Image-1 / Gemini Imagen for the Contribution 3 factorial.
10. **Factorial interference matrix** (num×neg is the centerpiece pair; num×spat + neg×spat frontier-only) with two-way ANOVA + bootstrap CIs on directional penalties.
11. **Writing in parallel** from trio week 1 using the 9B data in hand; only the final figure set needs trio-complete data.

### What's at risk

The single biggest risk to the paper is the negation scene-coherence confound. If step 1 above confirms that VLM prior drives the k=2 > k=1 jump, Contribution 3's asymmetric interference claim (`I_{neg→num} >> I_{num→neg}`) was measured against a confounded evaluator and needs rebuilding on scene-coherent-only negation data. Every subsequent compute decision is conditional on resolving that diagnostic. Do not run additional negation generation or the factorial interference matrix until it's settled.