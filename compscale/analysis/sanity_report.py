"""Aggregate all six CompScale sanity-suite gates into a single report.

Reads the per-module JSON reports produced by the earlier stages and emits
``compscale/sanity/sanity_report.json`` plus a printed pass/fail table.

Module 1 (baseline viability) is computed inline from the main VLM results
(no dedicated report file — it's just filter-by-type + mean per-constraint
satisfaction at k=1, gated > 0.7).

Exits 0 if all modules pass, 2 otherwise.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

BASELINE_GATE = 0.70


def compute_module1(sanity_results_path: Path, gate: float):
    if not sanity_results_path.exists():
        return {"status": "missing", "reason": f"no results at {sanity_results_path}"}
    results = json.loads(sanity_results_path.read_text())

    by_type_k = defaultdict(lambda: defaultdict(list))
    for r in results:
        t = r.get("type")
        k = r.get("k")
        if t is None or k is None:
            continue
        by_type_k[t][k].append(r["satisfaction_rate"])

    per_type = {}
    for t, by_k in by_type_k.items():
        k1 = by_k.get(1) or by_k.get("1") or []
        mean_k1 = (sum(k1) / len(k1)) if k1 else 0.0
        status = "pass" if mean_k1 >= gate else "fail"
        per_type[t] = {
            "n_at_k1": len(k1),
            "mean_satisfaction_at_k1": round(mean_k1, 4),
            "by_k": {
                str(k): {
                    "n": len(sats),
                    "mean": round(sum(sats) / len(sats), 4) if sats else 0.0,
                }
                for k, sats in sorted(by_k.items())
            },
            "gate": gate,
            "status": status,
        }
    overall = (
        "pass" if per_type and all(r["status"] == "pass" for r in per_type.values())
        else "fail"
    )
    return {
        "overall_status": overall,
        "gate": gate,
        "per_type": per_type,
    }


def _load_maybe(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        return {"_error": str(e)}


def _status(block):
    if block is None:
        return "missing"
    if isinstance(block, dict) and "_error" in block:
        return "error"
    return block.get("overall_status", "unknown")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanity_results",
                        default="compscale/evaluation/results/sanity_results.json",
                        help="Main vlm_verify results covering Modules 1/4/5/6")
    parser.add_argument("--sanity_dir", default="compscale/sanity")
    parser.add_argument("--attr_summary",
                        default="compscale/analysis/figures/sanity_attr_summary.json",
                        help="fit_scaling_law.py --type attribute summary for Module 6")
    parser.add_argument("--baseline_gate", type=float, default=BASELINE_GATE)
    parser.add_argument("--output",
                        default="compscale/sanity/sanity_report.json")
    args = parser.parse_args()

    sanity_dir = Path(args.sanity_dir)

    module1 = compute_module1(Path(args.sanity_results), args.baseline_gate)
    module2 = _load_maybe(sanity_dir / "vlm_reliability_report.json")
    module3 = _load_maybe(sanity_dir / "token_length_report.json")
    module4 = _load_maybe(sanity_dir / "order_permutation_report.json")
    module5 = _load_maybe(sanity_dir / "seed_variance_report.json")

    attr_summary = _load_maybe(Path(args.attr_summary))
    module6 = None
    if attr_summary and "independence" in attr_summary:
        per_k = attr_summary["independence"]["per_k"]
        k8 = per_k.get("8")
        if k8 and k8.get("ratio_ci_95"):
            ci = k8["ratio_ci_95"]
            ratio = k8["ratio"]
            ci_excludes_1 = ci["upper"] < 1.0
            status = "pass" if (ratio < 1.0 and ci_excludes_1) else "fail"
            module6 = {
                "overall_status": status,
                "ratio_k8": ratio,
                "ratio_k8_ci_95": ci,
                "best_fit": attr_summary.get("best_fit"),
                "note": "independence ratio at k=8 must be <1.0 with upper CI bound <1.0",
            }
        else:
            module6 = {
                "overall_status": "missing",
                "reason": "no ratio_ci_95 at k=8 in attr_summary",
            }
    else:
        module6 = {"overall_status": "missing",
                   "reason": f"{args.attr_summary} not found or lacks independence block"}

    modules = {
        "module1_baseline_viability": module1,
        "module2_vlm_reliability": module2,
        "module3_token_length": module3,
        "module4_order_permutation": module4,
        "module5_seed_variance": module5,
        "module6_independence_reproducibility": module6,
    }

    status_table = {name: _status(block) for name, block in modules.items()}
    all_pass = all(s == "pass" for s in status_table.values())
    overall_status = "pass" if all_pass else "fail"

    print("=" * 72)
    print(" CompScale Sanity Report")
    print("=" * 72)
    for name, s in status_table.items():
        marker = "PASS" if s == "pass" else ("FAIL" if s == "fail" else s.upper())
        print(f"  [{marker:7s}] {name}")
    print("-" * 72)
    print(f"  Overall: {overall_status.upper()}")
    print("=" * 72)

    if module1 and module1.get("per_type"):
        print("\nModule 1 per-type k=1 baselines:")
        for t, info in module1["per_type"].items():
            print(
                f"  {t:10s} k=1 mean={info['mean_satisfaction_at_k1']:.3f} "
                f"[{info['status']}]"
            )

    report = {
        "overall_status": overall_status,
        "modules": modules,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {out}")

    sys.exit(0 if all_pass else 2)


if __name__ == "__main__":
    main()
