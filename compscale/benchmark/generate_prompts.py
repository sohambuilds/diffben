"""Deterministic template-based prompt generator for the CompScale sanity suite.

Emits four pilot JSONs (attribute binding, negation, spatial, numeracy) plus an
order-permutation pilot. All prompts carry an explicit ``type`` field on each
constraint so the VLM verifier can route without heuristics. Generation is
seeded so re-running this script produces bit-identical files.

Design choices (see CLAUDE.md + the sanity-suite plan):

* k ladder: ``{1, 2, 4, 8}``. 10 prompts per k for negation/spatial/numeracy
  (Module 1 baseline viability). 20 prompts per k for attribute binding
  (Module 1 + Module 6 independence reproducibility).
* Token length target: ~30-60 tokens regardless of k. Low-k prompts include
  scene-setting filler; high-k prompts pack constraints tightly.
* Numeracy counts restricted to {1, 2, 3} and Tier 1/2 objects only
  (per pilot v1/v2 findings).
* Negation targets are strictly additive/modular (from the negation ontology);
  structural elements like "floor" are forbidden.
* Spatial relations are restricted to on / under / left of / right of /
  in front of / behind / inside / next to.
* Each emitted prompt stores ``scene_prefix`` so the order-permutation step
  can rebuild the prompt by re-joining reordered clauses.
"""

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ONTOLOGY_DIR = ROOT / "ontology"
PROMPTS_DIR = ROOT / "prompts"

K_LADDER = [1, 2, 4, 8]

ATTR_COLORS = [
    "red", "blue", "green", "yellow", "black", "white",
    "pink", "orange", "purple", "brown",
]

ATTR_OBJECTS = [
    "car", "truck", "bicycle", "bus", "motorcycle", "sofa", "armchair",
    "lamp", "cushion", "rug", "mug", "plate", "bowl", "kettle", "bottle",
    "backpack", "umbrella", "suitcase", "book", "notebook", "chair", "desk",
    "shelf", "vase", "teddy bear", "hat", "scarf", "towel", "basket",
    "watering can", "helmet", "bench", "stool",
]


# --- Scene framing banks ------------------------------------------------
#
# The gate in Module 3 requires median token count at k=8 to be within 1.5x of
# median at k=1 per type. Because each additional constraint clause is ~3-7
# tokens, raw prompts trend 10 tokens at k=1 to 50+ tokens at k=8 without
# framing, which blows the gate. To hold token length roughly constant across
# k, we pad low-k prompts heavily (a long preamble AND a trailing descriptor)
# and strip padding as k increases. Prefixes / suffixes are hand-authored at
# four length tiers and picked deterministically per-prompt.
#
# Token counts below are Mistral-Nemo-Base-2407 approximations.

# Long preamble (~24-28 tokens). Uses a scene-qualifier placeholder ``{scene}``
# so both generic prompts ("a spacious room") and negation prompts (using the
# ontology scene name like "kitchen" or "bedroom") can share the same bank.
HEAVY_PREFIX_TEMPLATES = [
    "In a warmly lit {scene} with hardwood floors, tall windows dressed in sheer curtains, and potted plants arranged near the far wall, the scene captures",
    "Inside a bright modern {scene} with soft ambient lighting, minimalist furniture clustered near the center, and wide pale walls throughout the frame, the view features",
    "In a quiet afternoon {scene} with warm natural light streaming through a window, subtle shadows across the floor, and a few small ornaments on nearby surfaces, the image presents",
    "Within a spacious well-kept {scene} with pale neutral walls, wide plank flooring, and gentle diffuse lighting that softens every edge in the composition, the frame depicts",
    "Inside a tidy comfortable {scene} with wooden furnishings arranged along the walls, a patterned rug underfoot, and daylight pouring in from a tall window, the arrangement shows",
]

# Medium preamble (~15-17 tokens).
MEDIUM_PREFIX_TEMPLATES = [
    "In a bright airy {scene} with light flooring and gentle overhead sunlight, the scene shows",
    "Inside a tidy modern {scene} with pale walls and clean architectural lines, the frame features",
    "In a cozy afternoon {scene} with soft ambient lighting and a warm palette, the view presents",
    "Within a minimalist {scene} with neutral finishes and abundant natural light, the image includes",
    "Inside a welcoming {scene} with wooden accents and a quiet domestic mood, the composition depicts",
]

# Short preamble (~7-9 tokens).
SHORT_PREFIX_TEMPLATES = [
    "In a bright {scene} the scene shows",
    "Inside a cozy {scene} the view features",
    "In a tidy {scene} the frame contains",
    "Within a modern {scene} the composition has",
    "Inside a warm {scene} the image presents",
]

# Minimal preamble (~3-5 tokens).
MINIMAL_PREFIX_TEMPLATES = [
    "A {scene} with",
    "In the {scene}:",
    "Inside the {scene},",
    "A {scene} containing",
]

# Long trailing descriptor (~11-14 tokens).
HEAVY_SUFFIXES = [
    ", captured in gentle afternoon light with soft dust motes drifting through the air",
    ", rendered in warm natural tones with a shallow depth of field throughout the composition",
    ", bathed in the golden glow of late afternoon sunlight streaming through tall windows",
    ", framed against a pale neutral background with diffused overhead lighting from above",
    ", illuminated by even diffused light from clerestory windows high on the opposite wall",
]

# Medium trailing descriptor (~5-7 tokens).
MEDIUM_SUFFIXES = [
    ", softly lit in warm tones",
    ", against a pale neutral backdrop",
    ", captured with shallow depth of field",
    ", in diffused afternoon light",
    ", with a gently blurred background",
]

# Scene qualifiers used to fill the ``{scene}`` placeholder for the three
# types that don't carry an ontology-level scene name (attribute / numeracy /
# spatial). Negation uses its ontology scene names directly instead.
GENERIC_SCENE_QUALIFIERS = [
    "living room", "kitchen", "studio space", "bedroom", "office",
    "sunroom", "loft interior", "dining room", "reading nook",
    "hallway", "workshop", "garden room", "parlor",
]


_FRAMING_BY_K = {
    1: ("heavy", "heavy"),
    2: ("medium", "medium"),
    4: ("short", ""),
    8: ("minimal", ""),
}


def _pick_framing(rng, k, scene_qualifier):
    """Return (scene_prefix, scene_suffix) tuned to k and filled with scene_qualifier."""
    prefix_level, suffix_level = _FRAMING_BY_K[k]
    prefix_bank = {
        "heavy": HEAVY_PREFIX_TEMPLATES,
        "medium": MEDIUM_PREFIX_TEMPLATES,
        "short": SHORT_PREFIX_TEMPLATES,
        "minimal": MINIMAL_PREFIX_TEMPLATES,
    }[prefix_level]
    prefix = rng.choice(prefix_bank).format(scene=scene_qualifier)

    if suffix_level == "heavy":
        suffix = rng.choice(HEAVY_SUFFIXES)
    elif suffix_level == "medium":
        suffix = rng.choice(MEDIUM_SUFFIXES)
    else:
        suffix = ""
    return prefix, suffix


def _load_ontology():
    count = json.loads((ONTOLOGY_DIR / "countable_objects.json").read_text())
    neg = json.loads((ONTOLOGY_DIR / "negation_targets.json").read_text())
    spat = json.loads((ONTOLOGY_DIR / "spatial_objects.json").read_text())
    return count, neg, spat


def _join_clauses(clauses):
    if len(clauses) == 1:
        return clauses[0]
    if len(clauses) == 2:
        return f"{clauses[0]} and {clauses[1]}"
    return ", ".join(clauses[:-1]) + f", and {clauses[-1]}"


# --- Clause renderers: one per constraint type --------------------------


def clause_for(constraint):
    t = constraint["type"]
    if t == "attribute":
        return f"a {constraint['color']} {constraint['object']}"
    if t == "negation":
        return f"no {constraint['object']}"
    if t == "spatial":
        return f"a {constraint['object_a']} {constraint['relation']} a {constraint['object_b']}"
    if t == "numeracy":
        word = constraint["plural"] if constraint["count"] != 1 else constraint["object"]
        number_word = {1: "one", 2: "two", 3: "three"}[constraint["count"]]
        return f"{number_word} {constraint['color']} {word}"
    raise ValueError(f"Unknown constraint type: {t}")


def render_prompt(scene_prefix, constraints, scene_suffix=""):
    clauses = [clause_for(c) for c in constraints]
    return f"{scene_prefix} {_join_clauses(clauses)}{scene_suffix}"


# --- Attribute binding --------------------------------------------------


def _attr_constraint(rng, used_objects):
    available = [o for o in ATTR_OBJECTS if o not in used_objects]
    obj = rng.choice(available)
    used_objects.add(obj)
    color = rng.choice(ATTR_COLORS)
    return {"type": "attribute", "color": color, "object": obj}


def build_attribute_prompt(rng, k, idx):
    used = set()
    cs = [_attr_constraint(rng, used) for _ in range(k)]
    scene_qualifier = rng.choice(GENERIC_SCENE_QUALIFIERS)
    scene_prefix, scene_suffix = _pick_framing(rng, k, scene_qualifier)
    return {
        "id": f"attr_k{k}_{idx:02d}",
        "k": k,
        "type": "attribute",
        "scene_prefix": scene_prefix,
        "scene_suffix": scene_suffix,
        "constraints": cs,
        "prompt": render_prompt(scene_prefix, cs, scene_suffix),
    }


# --- Negation -----------------------------------------------------------


def build_negation_prompt(rng, k, idx, neg_ontology):
    scene_name = rng.choice(list(neg_ontology["scenes"].keys()))
    scene_info = neg_ontology["scenes"][scene_name]
    targets = rng.sample(scene_info["valid_negation_targets"], k=k)
    cs = [{"type": "negation", "object": t, "scene": scene_name} for t in targets]
    scene_prefix, scene_suffix = _pick_framing(rng, k, scene_name)
    return {
        "id": f"neg_k{k}_{idx:02d}",
        "k": k,
        "type": "negation",
        "scene_prefix": scene_prefix,
        "scene_suffix": scene_suffix,
        "constraints": cs,
        "prompt": render_prompt(scene_prefix, cs, scene_suffix),
    }


# --- Spatial ------------------------------------------------------------


def _sample_object(rng, pool, used):
    available = [o for o in pool if o["object"] not in used]
    obj = rng.choice(available)
    used.add(obj["object"])
    return obj


def build_spatial_prompt(rng, k, idx, spat_ontology):
    objects = spat_ontology["objects"]
    surfaces = spat_ontology["surfaces_and_containers"]
    relations = spat_ontology["relations"]

    priority_rels = [r for r in relations if r["relation"] in
                     ("on", "under", "to the left of", "to the right of", "next to")]
    depth_rels = [r for r in relations if r["relation"] in
                  ("in front of", "behind", "inside")]

    used = set()
    cs = []
    for i in range(k):
        a = _sample_object(rng, objects, used)
        b_pool = surfaces if i < 2 else (objects + surfaces)
        b = _sample_object(rng, b_pool, used)
        rel_pool = priority_rels if i < (k // 2 + 1) else (priority_rels + depth_rels)
        rel = rng.choice(rel_pool)
        cs.append({
            "type": "spatial",
            "object_a": a["object"],
            "relation": rel["relation"],
            "object_b": b["object"],
        })

    scene_qualifier = rng.choice(GENERIC_SCENE_QUALIFIERS)
    scene_prefix, scene_suffix = _pick_framing(rng, k, scene_qualifier)
    return {
        "id": f"spat_k{k}_{idx:02d}",
        "k": k,
        "type": "spatial",
        "scene_prefix": scene_prefix,
        "scene_suffix": scene_suffix,
        "constraints": cs,
        "prompt": render_prompt(scene_prefix, cs, scene_suffix),
    }


# --- Numeracy -----------------------------------------------------------


def _num_constraint(rng, used_objects, count_ontology):
    pool = count_ontology["tier1"] + count_ontology["tier2"]
    available = [o for o in pool if o["object"] not in used_objects]
    obj_entry = rng.choice(available)
    used_objects.add(obj_entry["object"])
    color = rng.choice(obj_entry["plausible_colors"])
    count = rng.choice([1, 2, 3])
    return {
        "type": "numeracy",
        "count": count,
        "color": color,
        "object": obj_entry["object"],
        "plural": obj_entry["plural"],
    }


def build_numeracy_prompt(rng, k, idx, count_ontology):
    used = set()
    cs = [_num_constraint(rng, used, count_ontology) for _ in range(k)]
    scene_qualifier = rng.choice(GENERIC_SCENE_QUALIFIERS)
    scene_prefix, scene_suffix = _pick_framing(rng, k, scene_qualifier)
    return {
        "id": f"num_k{k}_{idx:02d}",
        "k": k,
        "type": "numeracy",
        "scene_prefix": scene_prefix,
        "scene_suffix": scene_suffix,
        "constraints": cs,
        "prompt": render_prompt(scene_prefix, cs, scene_suffix),
    }


# --- Driver -------------------------------------------------------------


def generate_all(seed=42, n_per_k_default=10, n_per_k_attr=20):
    rng_attr = random.Random(seed + 1)
    rng_neg = random.Random(seed + 2)
    rng_spat = random.Random(seed + 3)
    rng_num = random.Random(seed + 4)

    count_onto, neg_onto, spat_onto = _load_ontology()

    attribute_prompts = []
    for k in K_LADDER:
        for i in range(1, n_per_k_attr + 1):
            attribute_prompts.append(build_attribute_prompt(rng_attr, k, i))

    negation_prompts = []
    for k in K_LADDER:
        for i in range(1, n_per_k_default + 1):
            negation_prompts.append(build_negation_prompt(rng_neg, k, i, neg_onto))

    spatial_prompts = []
    for k in K_LADDER:
        for i in range(1, n_per_k_default + 1):
            spatial_prompts.append(build_spatial_prompt(rng_spat, k, i, spat_onto))

    numeracy_prompts = []
    for k in K_LADDER:
        for i in range(1, n_per_k_default + 1):
            numeracy_prompts.append(build_numeracy_prompt(rng_num, k, i, count_onto))

    return {
        "attribute_pilot_v2.json": attribute_prompts,
        "negation_pilot.json": negation_prompts,
        "spatial_pilot.json": spatial_prompts,
        "numeracy_pilot_v3.json": numeracy_prompts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_per_k", type=int, default=10,
                        help="Prompts per k for negation / spatial / numeracy")
    parser.add_argument("--n_per_k_attr", type=int, default=20,
                        help="Prompts per k for attribute binding (Module 6 needs 20)")
    parser.add_argument("--output_dir", default=str(PROMPTS_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = generate_all(
        seed=args.seed,
        n_per_k_default=args.n_per_k,
        n_per_k_attr=args.n_per_k_attr,
    )
    for name, prompts in files.items():
        path = out_dir / name
        path.write_text(json.dumps(prompts, indent=2))
        by_k = {}
        for p in prompts:
            by_k.setdefault(p["k"], 0)
            by_k[p["k"]] += 1
        print(f"  wrote {len(prompts):4d} prompts to {path}  (by k: {dict(sorted(by_k.items()))})")


if __name__ == "__main__":
    main()
