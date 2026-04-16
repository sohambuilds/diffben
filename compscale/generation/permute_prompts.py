"""Module 4 helper: produce order-permuted variants of existing pilot prompts.

Reads the four Module-1 pilot JSONs, selects the first 5 k=4 prompts per type,
and emits 3 distinct constraint-order permutations per base prompt into
``compscale/benchmark/prompts/order_pilot.json``. Scene-setting text
(``scene_prefix``) is preserved verbatim; only constraint order changes.

This is separate from the base generator so the permutation pilot can be
refreshed independently without regenerating the base prompts.
"""

import argparse
import json
import random
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
sys.path.insert(0, str(ROOT / "benchmark"))

from generate_prompts import render_prompt  # noqa: E402

PROMPTS_DIR = ROOT / "benchmark" / "prompts"

DEFAULT_SOURCES = [
    "attribute_pilot_v2.json",
    "negation_pilot.json",
    "spatial_pilot.json",
    "numeracy_pilot_v3.json",
]


def build_permutations(base_prompts, rng, n_perms=3):
    out = []
    for base in base_prompts:
        constraints = base["constraints"]
        scene_prefix = base["scene_prefix"]
        if len(constraints) < 2:
            continue
        seen_orders = {tuple(range(len(constraints)))}
        produced = 0
        attempts = 0
        while produced < n_perms and attempts < 100:
            attempts += 1
            order = list(range(len(constraints)))
            rng.shuffle(order)
            if tuple(order) in seen_orders:
                continue
            seen_orders.add(tuple(order))
            reordered = [constraints[i] for i in order]
            out.append({
                "id": f"{base['id']}_perm{produced}",
                "base_id": base["id"],
                "permutation_idx": produced,
                "order": order,
                "k": base["k"],
                "type": base["type"],
                "scene_prefix": scene_prefix,
                "constraints": reordered,
                "prompt": render_prompt(scene_prefix, reordered),
            })
            produced += 1
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES,
                        help="Filenames (relative to compscale/benchmark/prompts/) to pull bases from")
    parser.add_argument("--output", default=str(PROMPTS_DIR / "order_pilot.json"))
    parser.add_argument("--k", type=int, default=4, help="Which k level to permute")
    parser.add_argument("--n_bases_per_type", type=int, default=5)
    parser.add_argument("--n_perms", type=int, default=3)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    bases = []
    for name in args.sources:
        path = PROMPTS_DIR / name
        if not path.exists():
            print(f"  (skipping missing source {path})")
            continue
        prompts = json.loads(path.read_text())
        type_bases = [p for p in prompts if p["k"] == args.k][:args.n_bases_per_type]
        bases.extend(type_bases)

    permuted = build_permutations(bases, rng, n_perms=args.n_perms)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(permuted, indent=2))
    print(f"  wrote {len(permuted)} permuted prompts to {out}")
    by_type = {}
    for p in permuted:
        by_type.setdefault(p["type"], 0)
        by_type[p["type"]] += 1
    print(f"  by type: {by_type}")


if __name__ == "__main__":
    main()
