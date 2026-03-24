# CLAUDE.md — CompScale Project Context (v2)

## What This Document Is

This is the canonical context file for the **CompScale** research project. Feed this to any agent (coding, writing, brainstorming, evaluation) working on any part of this project. It contains the full research motivation, literature positioning, methodology, technical specs, and open questions.

**v2 changelog (March 2026):** Killed ConDecomp (scooped by PRISM @ ICLR 2026). Repositioned benchmark against DetailMaster and T2I-CoReBench. Updated model roster to FLUX.2 family. Promoted asymmetric interference analysis to full Contribution 3.

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
- **This document last updated:** March 2026, v3

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

### Design Implications for CompScale-Bench

1. **CRITICAL: Hold per-object counts constant across all k levels.** Use only counts 1-3 for all constraints at all k. This isolates k as the sole independent variable.
2. **Use only Tier 1-2 objects.** Curate a verified countable object list (~30-40 objects).
3. **The "count difficulty" decay curve itself is a finding.** Consider reporting it as evidence of fundamental numeracy limitations — even a single constraint with count>4 fails reliably.
4. **VLM evaluator reliability:** Gemini 3 Flash Preview is reliable for counts 1-3 of Tier 1 objects. Unreliable for counts >5 or Tier 3 objects (impossible to distinguish evaluator failure from generation failure).
5. **Numeracy on weak models is at floor.** For the scaling law analysis to work, need per-constraint baseline > 70%. Attribute binding is the better first dimension to test on small models. Save numeracy for frontier APIs.
6. **Independence test is essential.** Always compare observed all-satisfied rates against the p^k independence model. The interesting finding is when observed < predicted (active interference), not when it merely follows binomial decay.