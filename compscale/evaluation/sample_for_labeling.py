"""Module 2 prep: sample (image, constraint) pairs for Soham to hand-label.

Produces ``compscale/evaluation/human_eval/hand_labels_template.csv`` with 40
rows per constraint type (160 total). Each row is one (image, constraint)
judgment, sampled from the Module 1 generation outputs. Soham then fills the
``human_label`` column ("sat" or "unsat") and saves as ``hand_labels.csv``.

Why sample pairs rather than whole images?
 * Satisfaction is per-constraint, so the VLM reliability number we care about
   is per (question-template, image) accuracy against a human label.
 * Sampling across the k ladder stratified by k catches evaluator drift at
   high constraint-count where images are densest.

Sampling strategy: stratified by k. We want ~10 per k level per type (so 40
per type across k in {1, 2, 4, 8}). Randomness is seeded.
"""

import argparse
import csv
import json
import random
from pathlib import Path

DEFAULT_SOURCES = {
    "attribute": "attribute_pilot_v2.json",
    "negation": "negation_pilot.json",
    "spatial": "spatial_pilot.json",
    "numeracy": "numeracy_pilot_v3.json",
}

K_LADDER = [1, 2, 4, 8]


def _load(prompts_dir: Path, name: str):
    path = prompts_dir / name
    if not path.exists():
        print(f"  (missing {path})")
        return []
    return json.loads(path.read_text())


def _constraint_brief(c):
    t = c.get("type") or ("numeracy" if "count" in c else "attribute")
    if t == "attribute":
        return f"attr: {c['color']} {c['object']}"
    if t == "numeracy":
        return f"num: {c['count']} {c['color']} {c['object']}"
    if t == "negation":
        return f"neg: no {c['object']}"
    if t == "spatial":
        return f"spat: {c['object_a']} {c['relation']} {c['object_b']}"
    return str(c)


def sample_pairs(prompts, images_dir: Path, n_per_k: int, rng, n_images: int):
    """Yield (prompt, image_index, constraint_index) tuples sampled evenly across k."""
    pool = {k: [p for p in prompts if p["k"] == k] for k in K_LADDER}
    picks = []
    for k in K_LADDER:
        candidates = []
        for p in pool.get(k, []):
            prompt_dir = images_dir / p["id"]
            for j in range(n_images):
                img_path = prompt_dir / f"img_{j}.png"
                if not img_path.exists():
                    continue
                for c_idx in range(len(p["constraints"])):
                    candidates.append((p, j, c_idx, img_path))
        if not candidates:
            continue
        n_take = min(n_per_k, len(candidates))
        chosen = rng.sample(candidates, n_take)
        picks.extend(chosen)
    return picks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts_dir", default="compscale/benchmark/prompts")
    parser.add_argument("--images_dir", default="compscale/generation/outputs/flux2-klein-9b")
    parser.add_argument("--output",
                        default="compscale/evaluation/human_eval/hand_labels_template.csv")
    parser.add_argument("--n_per_k", type=int, default=10)
    parser.add_argument("--n_images", type=int, default=4)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--types", nargs="*", default=list(DEFAULT_SOURCES.keys()))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    prompts_dir = Path(args.prompts_dir)
    images_dir = Path(args.images_dir)

    rows = []
    for ctype in args.types:
        source = DEFAULT_SOURCES.get(ctype)
        if not source:
            print(f"  unknown type {ctype}, skipping")
            continue
        prompts = _load(prompts_dir, source)
        if not prompts:
            continue
        picks = sample_pairs(prompts, images_dir, args.n_per_k, rng, args.n_images)
        print(f"  {ctype}: {len(picks)} pairs sampled")
        for p, j, c_idx, img_path in picks:
            c = p["constraints"][c_idx]
            rows.append({
                "image_path": str(img_path).replace("\\", "/"),
                "prompt_id": p["id"],
                "image_index": j,
                "constraint_index": c_idx,
                "constraint_type": ctype,
                "k": p["k"],
                "constraint_brief": _constraint_brief(c),
                "constraint_json": json.dumps(c),
                "human_label": "",
                "notes": "",
            })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "image_path", "prompt_id", "image_index", "constraint_index",
            "constraint_type", "k", "constraint_brief", "constraint_json",
            "human_label", "notes",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out}")
    print("Fill in 'human_label' with 'sat' or 'unsat' and save as 'hand_labels.csv'.")


if __name__ == "__main__":
    main()
