"""Fit scaling law decay curves to pilot evaluation results."""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit


def exponential_decay(k, s0, alpha):
    return s0 * np.exp(-alpha * k)


def sigmoid_decay(k, beta, k0):
    return 1.0 / (1.0 + np.exp(beta * (k - k0)))


def power_law_decay(k, s0, gamma):
    return s0 * np.power(k, -gamma)


def bootstrap_ci(values, n_boot=1000, ci=0.95):
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(42)
    means = [
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ]
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return lower, upper


def main():
    parser = argparse.ArgumentParser(description="Fit decay curves to pilot results")
    parser.add_argument(
        "--results",
        default="compscale/evaluation/results/pilot_results.json",
        help="Path to evaluation results JSON",
    )
    parser.add_argument(
        "--output_dir",
        default="compscale/analysis/figures",
        help="Output directory for figures",
    )
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text())

    # Aggregate by k
    k_levels = sorted(set(r["k"] for r in results))
    per_k = {k: [] for k in k_levels}
    per_k_all_sat = {k: [] for k in k_levels}

    for r in results:
        per_k[r["k"]].append(r["satisfaction_rate"])
        per_k_all_sat[r["k"]].append(1.0 if r["all_satisfied"] else 0.0)

    means = [np.mean(per_k[k]) for k in k_levels]
    sems = [np.std(per_k[k]) / np.sqrt(len(per_k[k])) for k in k_levels]
    all_sat_means = [np.mean(per_k_all_sat[k]) for k in k_levels]

    k_arr = np.array(k_levels, dtype=float)
    mean_arr = np.array(means)

    print("=" * 60)
    print("CompScale Pilot Results")
    print("=" * 60)
    for k in k_levels:
        ci_lo, ci_hi = bootstrap_ci(per_k[k])
        print(
            f"  k={k:2d}: satisfaction={np.mean(per_k[k]):.3f} "
            f"[{ci_lo:.3f}, {ci_hi:.3f}] "
            f"all_satisfied={np.mean(per_k_all_sat[k]):.3f} "
            f"(n={len(per_k[k])})"
        )

    # Fit curves
    candidates = [
        ("Exponential", exponential_decay, [1.0, 0.1]),
        ("Sigmoid", sigmoid_decay, [1.0, 4.0]),
        ("Power law", power_law_decay, [1.0, 0.5]),
    ]

    fits = {}
    print("\nCurve Fitting:")
    for name, func, p0 in candidates:
        try:
            popt, pcov = curve_fit(
                func, k_arr, mean_arr, p0=p0, maxfev=10000,
                bounds=([0, 0], [np.inf, np.inf]),
            )
            predicted = func(k_arr, *popt)
            residuals = mean_arr - predicted
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((mean_arr - np.mean(mean_arr)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            n = len(k_arr)
            p = len(popt)
            aic = n * np.log(ss_res / n + 1e-10) + 2 * p

            fits[name] = {
                "params": popt,
                "r_squared": r_squared,
                "aic": aic,
                "func": func,
            }
            param_str = ", ".join(f"{v:.4f}" for v in popt)
            print(f"  {name}: R²={r_squared:.4f}, AIC={aic:.2f}, params=({param_str})")
        except RuntimeError as e:
            print(f"  {name}: fitting failed ({e})")

    if not fits:
        print("\nNo curves could be fitted. Check your data.")
        return

    best_name = min(fits, key=lambda x: fits[x]["aic"])
    print(f"\nBest fit (by AIC): {best_name}")

    # Plot
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Per-constraint satisfaction with fits
    ax1.errorbar(
        k_levels, means, yerr=sems, fmt="ko", capsize=5, capthick=1.5,
        markersize=8, label="Data (mean +/- SEM)", zorder=5,
    )
    k_smooth = np.linspace(max(0.8, min(k_levels) - 0.5), max(k_levels) + 1, 200)
    colors = {"Exponential": "#e74c3c", "Sigmoid": "#3498db", "Power law": "#2ecc71"}
    for name, fit in fits.items():
        style = "-" if name == best_name else "--"
        alpha = 1.0 if name == best_name else 0.6
        ax1.plot(
            k_smooth,
            fit["func"](k_smooth, *fit["params"]),
            style,
            color=colors.get(name, "gray"),
            alpha=alpha,
            linewidth=2,
            label=f'{name} (R²={fit["r_squared"]:.3f})',
        )
    ax1.set_xlabel("Constraint count (k)", fontsize=12)
    ax1.set_ylabel("Per-constraint satisfaction rate", fontsize=12)
    ax1.set_title("Numeracy: Satisfaction vs. Constraint Count", fontsize=13)
    ax1.set_xticks(k_levels)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: All-constraints-satisfied rate
    ax2.bar(
        k_levels, all_sat_means, color="#3498db", alpha=0.7, width=0.6,
        edgecolor="black", linewidth=0.5,
    )
    ax2.set_xlabel("Constraint count (k)", fontsize=12)
    ax2.set_ylabel("Full-prompt satisfaction rate", fontsize=12)
    ax2.set_title("Fraction of Images Satisfying ALL Constraints", fontsize=13)
    ax2.set_xticks(k_levels)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig_path = output_dir / "pilot_decay_curve.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved to {fig_path}")

    # Save numerical summary
    summary = {
        "k_levels": k_levels,
        "per_constraint_satisfaction": {
            str(k): {
                "mean": float(np.mean(per_k[k])),
                "sem": float(np.std(per_k[k]) / np.sqrt(len(per_k[k]))),
                "ci_95": [float(x) for x in bootstrap_ci(per_k[k])],
                "n": len(per_k[k]),
            }
            for k in k_levels
        },
        "all_satisfied_rate": {str(k): float(np.mean(per_k_all_sat[k])) for k in k_levels},
        "fits": {
            name: {
                "params": [float(x) for x in fit["params"]],
                "r_squared": float(fit["r_squared"]),
                "aic": float(fit["aic"]),
            }
            for name, fit in fits.items()
        },
        "best_fit": best_name,
    }
    summary_path = output_dir / "pilot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
