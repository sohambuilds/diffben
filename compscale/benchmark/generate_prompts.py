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

ATTR_SCENES = [
    "A bright parking lot with trees along the edges and a clear sky overhead",
    "A cozy living room with bookshelves along the walls and warm lighting",
    "A quiet residential street with brick houses and trimmed hedges",
    "A tidy kitchen with marble surfaces and wooden cabinets in afternoon light",
    "A modern office with a large window and potted plants on the sill",
    "A charming garden with stepping stones and blooming flower beds",
    "A children's bedroom with patterned wallpaper and wooden shelves of toys",
    "A city sidewalk with tall buildings in the background and shop windows",
    "A sunny beach with golden sand and gentle waves under a bright sky",
    "A farmhouse porch with wooden railings and hanging planters",
    "A home study lined with leather-bound books and a heavy wooden desk",
    "A craft room with shelves of supplies and a wide worktable under lamplight",
    "A loft apartment with exposed brick walls and tall factory windows",
    "A school hallway with lockers on both sides and polished floors",
    "A cafe interior with small round tables and chalkboard menus",
]


def _load_ontology():
    count = json.loads((ONTOLOGY_DIR / "countable_objects.json").read_text())
    neg = json.loads((ONTOLOGY_DIR / "negation_targets.json").read_text())
    spat = json.loads((ONTOLOGY_DIR / "spatial_objects.json").read_text())
    return count, neg, spat


def _attr_scene_prefix(rng, k):
    scene = rng.choice(ATTR_SCENES)
    if k == 1:
        return f"{scene}, featuring"
    if k == 2:
        return f"{scene}, showing"
    if k == 4:
        short = scene.split(",")[0]
        return f"{short} containing"
    return "A scene with"


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


def render_prompt(scene_prefix, constraints):
    clauses = [clause_for(c) for c in constraints]
    return f"{scene_prefix} {_join_clauses(clauses)}"


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
    scene_prefix = _attr_scene_prefix(rng, k)
    return {
        "id": f"attr_k{k}_{idx:02d}",
        "k": k,
        "type": "attribute",
        "scene_prefix": scene_prefix,
        "constraints": cs,
        "prompt": render_prompt(scene_prefix, cs),
    }


# --- Negation -----------------------------------------------------------


def build_negation_prompt(rng, k, idx, neg_ontology):
    scene_name = rng.choice(list(neg_ontology["scenes"].keys()))
    scene_info = neg_ontology["scenes"][scene_name]
    filler_phrase = rng.choice(scene_info["scene_filler_phrases"])
    targets = rng.sample(scene_info["valid_negation_targets"], k=k)
    cs = [{"type": "negation", "object": t, "scene": scene_name} for t in targets]
    joiner = "with" if k == 1 else "containing"
    scene_prefix = f"{filler_phrase}, {joiner}"
    return {
        "id": f"neg_k{k}_{idx:02d}",
        "k": k,
        "type": "negation",
        "scene_prefix": scene_prefix,
        "constraints": cs,
        "prompt": render_prompt(scene_prefix, cs),
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

    if k == 1:
        scene_prefix = "A simple indoor scene showing"
    elif k == 2:
        scene_prefix = "An indoor scene showing"
    else:
        scene_prefix = "A scene with"
    return {
        "id": f"spat_k{k}_{idx:02d}",
        "k": k,
        "type": "spatial",
        "scene_prefix": scene_prefix,
        "constraints": cs,
        "prompt": render_prompt(scene_prefix, cs),
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
    scene_prefix = _attr_scene_prefix(rng, k)
    return {
        "id": f"num_k{k}_{idx:02d}",
        "k": k,
        "type": "numeracy",
        "scene_prefix": scene_prefix,
        "constraints": cs,
        "prompt": render_prompt(scene_prefix, cs),
    }


# --- Order permutation pilot -------------------------------------------


def build_permutations(base_prompts, rng, n_perms=3):
    """For each base prompt, produce n_perms reshufflings of the constraint
    clauses. Scene-setting text is preserved; only constraint order changes."""
    out = []
    for base in base_prompts:
        constraints = base["constraints"]
        scene_prefix = base["scene_prefix"]
        seen_orders = {tuple(range(len(constraints)))}
        for p_idx in range(n_perms):
            order = list(range(len(constraints)))
            for _ in range(20):
                rng.shuffle(order)
                if tuple(order) not in seen_orders:
                    seen_orders.add(tuple(order))
                    break
            reordered = [constraints[i] for i in order]
            out.append({
                "id": f"{base['id']}_perm{p_idx}",
                "base_id": base["id"],
                "permutation_idx": p_idx,
                "order": order,
                "k": base["k"],
                "type": base["type"],
                "scene_prefix": scene_prefix,
                "constraints": reordered,
                "prompt": render_prompt(scene_prefix, reordered),
            })
    return out


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
