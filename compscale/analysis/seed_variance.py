"""Module 5 — seed-variance check.

Two subcommands:

* ``subset``: pick 20 prompts (5 per type, stratified across k in {1, 4, 8})
  and write their IDs to ``compscale/sanity/variance_subset.json``. The user
  then runs ``generate_images.py --prompt_ids ... --n_images 16
  --start_image_index 4`` to extend those prompts from N=4 to N=16.

* ``analyze``: read a VLM results JSON produced by ``vlm_verify.py`` over the
  full N=16 subset and decide whether N=4 bootstrap CIs are tight enough for
  decay fitting. Gate: median per-prompt CI width at N=4 is < 0.15.
  Writes ``compscale/sanity/seed_variance_report.json``.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

CI_GATE = 0.15

SOURCES = {
    "attribute": "attribute_pilot_v2.json",
    "negation": "negation_pilot.json",
    "spatial": "spatial_pilot.json",
    "numeracy": "numeracy_pilot_v3.json",
}

K_SUBSET = [1, 4, 8]


def cmd_subset(args):
    rng = random.Random(args.seed)
    prompts_dir = Path(args.prompts_dir)
    picked = []
    for ctype, fname in SOURCES.items():
        path = prompts_dir / fname
        if not path.exists():
            print(f"  (missing {path})")
            continue
        prompts = json.loads(path.read_text())
        by_k = defaultdict(list)
        for p in prompts:
            if p["k"] in K_SUBSET:
                by_k[p["k"]].append(p["id"])
        # take 5 per type: balance across k levels
        per_k_target = max(1, args.n_per_type // len(K_SUBSET))
        remaining = args.n_per_type
        for k in K_SUBSET:
            pool = by_k.get(k, [])
            if not pool:
                continue
            n_take = min(per_k_target, len(pool), remaining)
            chosen = rng.sample(pool, n_take)
            for pid in chosen:
                picked.append({"id": pid, "type": ctype, "k": k})
            remaining -= n_take
        # top up if needed from any k in the ladder
        leftover_pool = [
            p["id"]
            for p in prompts
            if p["k"] in K_SUBSET and p["id"] not in {x["id"] for x in picked}
        ]
        while remaining > 0 and leftover_pool:
            pid = rng.choice(leftover_pool)
            leftover_pool.remove(pid)
            k = next(p["k"] for p in prompts if p["id"] == pid)
            picked.append({"id": pid, "type": ctype, "k": k})
            remaining -= 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"subset": picked}, indent=2))
    print(f"Wrote {len(picked)} prompt IDs to {out}")
    for p in picked:
        print(f"  {p['type']:10s} k={p['k']}  {p['id']}")
    ids = " ".join(p["id"] for p in picked)
    print(f"\nTo extend these from N=4 to N=16, run:")
    print(
        "  python compscale/generation/generate_images.py --klein_9b \\\n"
        "    --prompts compscale/benchmark/prompts/*.json \\\n"
        f"    --n_images 16 --start_image_index 4 \\\n"
        f"    --prompt_ids {ids}"
    )


def _bootstrap_ci_width(values, ci=0.95, n_boot=500, seed=0):
    if len(values) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(n_boot)]
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return float(hi - lo)


def cmd_analyze(args):
    results = json.loads(Path(args.results).read_text())
    subset = json.loads(Path(args.subset).read_text())["subset"]
    subset_ids = {p["id"] for p in subset}

    by_prompt = defaultdict(list)
    for r in results:
        if r["prompt_id"] not in subset_ids:
            continue
        by_prompt[r["prompt_id"]].append(r["satisfaction_rate"])

    subset_type = {p["id"]: p["type"] for p in subset}
    subset_k = {p["id"]: p["k"] for p in subset}

    per_prompt = []
    for pid, sats in by_prompt.items():
        if len(sats) < 4:
            continue
        arr = np.array(sats, dtype=float)
        width_full = _bootstrap_ci_width(arr, seed=hash(pid) & 0xFFFFFFFF)
        widths_n4 = []
        widths_n8 = []
        rng = np.random.default_rng(hash(pid) & 0xFFFFFFFF)
        for _ in range(50):
            if len(arr) >= 4:
                sub4 = rng.choice(arr, size=4, replace=False)
                widths_n4.append(_bootstrap_ci_width(sub4, seed=0))
            if len(arr) >= 8:
                sub8 = rng.choice(arr, size=8, replace=False)
                widths_n8.append(_bootstrap_ci_width(sub8, seed=0))
        per_prompt.append({
            "prompt_id": pid,
            "type": subset_type.get(pid),
            "k": subset_k.get(pid),
            "n_draws": len(sats),
            "mean": float(arr.mean()),
            "ci_width_full": round(width_full, 4),
            "ci_width_n4_mean": float(np.mean(widths_n4)) if widths_n4 else None,
            "ci_width_n8_mean": float(np.mean(widths_n8)) if widths_n8 else None,
        })

    if not per_prompt:
        print("No overlap between subset IDs and results. Aborting.")
        return

    median_n4 = float(np.median([p["ci_width_n4_mean"] for p in per_prompt
                                 if p["ci_width_n4_mean"] is not None]))
    median_n8 = float(np.median([p["ci_width_n8_mean"] for p in per_prompt
                                 if p["ci_width_n8_mean"] is not None]))
    gate_status = "pass" if median_n4 < args.gate else "fail"

    report = {
        "gate": args.gate,
        "overall_status": gate_status,
        "median_ci_width_n4": round(median_n4, 4),
        "median_ci_width_n8": round(median_n8, 4),
        "n_prompts_analyzed": len(per_prompt),
        "per_prompt": per_prompt,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"Median per-prompt bootstrap CI width: N=4={median_n4:.3f}, N=8={median_n8:.3f}")
    print(f"Gate (N=4 median CI width < {args.gate}): {gate_status}")
    print(f"Report written to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("subset", help="Pick 20 prompts for the N=16 sub-experiment")
    s.add_argument("--prompts_dir", default="compscale/benchmark/prompts")
    s.add_argument("--output", default="compscale/sanity/variance_subset.json")
    s.add_argument("--n_per_type", type=int, default=5)
    s.add_argument("--seed", type=int, default=53)
    s.set_defaults(func=cmd_subset)

    a = sub.add_parser("analyze", help="Analyze N=16 VLM results")
    a.add_argument("--results", required=True,
                   help="Path to vlm_verify results JSON covering the N=16 subset")
    a.add_argument("--subset", default="compscale/sanity/variance_subset.json")
    a.add_argument("--output", default="compscale/sanity/seed_variance_report.json")
    a.add_argument("--gate", type=float, default=CI_GATE)
    a.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
