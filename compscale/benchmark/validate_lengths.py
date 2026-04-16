"""Module 3 — token-length distribution sanity check.

Tokenizes every prompt in the sanity-suite pilot JSONs with the FLUX.2 text
encoder tokenizer (Mistral-3-family). Reports per (type, k) descriptive
statistics and applies the gate:

* median token count at k=8 must be within 1.5x of median at k=1, per type.

If any type fails, iterate on scene_prefix filler before running Module 1
generation — otherwise prompt length will confound the k signal and the
scaling analysis is compromised (this is our defense against the
DetailMaster critique).

A JSON report and a violin plot are emitted to ``compscale/sanity/``.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SANITY_DIR = ROOT / "sanity"
PROMPTS_DIR = ROOT / "benchmark" / "prompts"

DEFAULT_SOURCES = [
    "attribute_pilot_v2.json",
    "negation_pilot.json",
    "spatial_pilot.json",
    "numeracy_pilot_v3.json",
    "order_pilot.json",
]

MEDIAN_RATIO_GATE = 1.5


def _load_tokenizer(name):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(name)
    except Exception as e:
        print(f"  could not load tokenizer '{name}': {e}")
        return None


def _median(values):
    if not values:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_v[mid])
    return (sorted_v[mid - 1] + sorted_v[mid]) / 2


def _p95(values):
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = max(0, min(len(sorted_v) - 1, int(round(0.95 * (len(sorted_v) - 1)))))
    return float(sorted_v[idx])


def tokenize_all(prompts, tokenizer):
    """Returns list of (prompt_id, type, k, token_count) tuples."""
    out = []
    for p in prompts:
        tokens = tokenizer(p["prompt"], add_special_tokens=False)["input_ids"]
        out.append((p["id"], p["type"], p["k"], len(tokens)))
    return out


def summarize(records):
    """records: list of (id, type, k, count). Returns nested dict by (type, k)."""
    by_tk = defaultdict(list)
    for _, t, k, n in records:
        by_tk[(t, k)].append(n)
    summary = {}
    for (t, k), counts in sorted(by_tk.items()):
        summary.setdefault(t, {})
        summary[t][str(k)] = {
            "n_prompts": len(counts),
            "min": min(counts),
            "median": _median(counts),
            "mean": round(sum(counts) / len(counts), 2),
            "p95": _p95(counts),
            "max": max(counts),
        }
    return summary


def apply_gate(summary, gate_ratio=MEDIAN_RATIO_GATE):
    """For each type, check that median token count at k=8 and k=1 agree
    within ``gate_ratio`` in either direction.

    Bidirectional gate: ratio in [1/gate_ratio, gate_ratio]. This catches both
    (a) k=8 ballooning longer than k=1 (the DetailMaster confound) AND
    (b) k=1 being padded so heavily it exceeds k=8 (the inverse confound —
    also compromises the scaling analysis)."""
    lo = 1.0 / gate_ratio
    hi = gate_ratio
    gate_results = {}
    for t, by_k in summary.items():
        if "1" not in by_k or "8" not in by_k:
            gate_results[t] = {"status": "skipped", "reason": "k=1 or k=8 missing"}
            continue
        med1 = by_k["1"]["median"]
        med8 = by_k["8"]["median"]
        ratio = med8 / med1 if med1 > 0 else float("inf")
        status = "pass" if (lo <= ratio <= hi) else "fail"
        gate_results[t] = {
            "median_k1": med1,
            "median_k8": med8,
            "ratio_k8_over_k1": round(ratio, 3),
            "gate_ratio": gate_ratio,
            "allowed_range": [round(lo, 3), round(hi, 3)],
            "status": status,
        }
    return gate_results


def _maybe_plot(records, out_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plot")
        return

    by_tk = defaultdict(list)
    for _, t, k, n in records:
        by_tk[(t, k)].append(n)

    types = sorted({t for (t, _) in by_tk.keys()})
    ks = sorted({k for (_, k) in by_tk.keys()})

    fig, axes = plt.subplots(1, len(types), figsize=(4 * len(types), 4), sharey=True)
    if len(types) == 1:
        axes = [axes]
    for ax, t in zip(axes, types):
        data = [by_tk.get((t, k), []) for k in ks]
        nonempty = [(k, d) for k, d in zip(ks, data) if d]
        if not nonempty:
            continue
        ks_plot = [k for k, _ in nonempty]
        data_plot = [d for _, d in nonempty]
        parts = ax.violinplot(data_plot, positions=ks_plot, widths=0.8, showmedians=True)
        for body in parts["bodies"]:
            body.set_alpha(0.6)
        ax.set_title(t)
        ax.set_xlabel("k")
        ax.set_xticks(ks_plot)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Token count")
    fig.suptitle("Token-length distribution per (constraint type, k)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    parser.add_argument("--prompts_dir", default=str(PROMPTS_DIR))
    parser.add_argument(
        "--tokenizer",
        default="mistralai/Mistral-Nemo-Base-2407",
        help="HF repo of the tokenizer used by the T2I text encoder. "
             "Default is a Mistral-family proxy for FLUX.2's text encoder.",
    )
    parser.add_argument(
        "--fallback_tokenizer",
        default="openai-community/gpt2",
        help="Fallback tokenizer if the primary cannot be loaded.",
    )
    parser.add_argument("--output_dir", default=str(SANITY_DIR))
    parser.add_argument("--gate_ratio", type=float, default=MEDIAN_RATIO_GATE)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer(args.tokenizer)
    tokenizer_name = args.tokenizer
    if tokenizer is None:
        tokenizer = _load_tokenizer(args.fallback_tokenizer)
        tokenizer_name = args.fallback_tokenizer
    if tokenizer is None:
        print("No tokenizer could be loaded. Aborting.")
        sys.exit(1)
    print(f"Using tokenizer: {tokenizer_name}")

    prompts_dir = Path(args.prompts_dir)
    all_prompts = []
    for name in args.sources:
        path = prompts_dir / name
        if not path.exists():
            print(f"  (skipping missing {path})")
            continue
        prompts = json.loads(path.read_text())
        all_prompts.extend(prompts)
    print(f"Loaded {len(all_prompts)} prompts across {len(args.sources)} files")

    records = tokenize_all(all_prompts, tokenizer)
    summary = summarize(records)
    gates = apply_gate(summary, args.gate_ratio)

    report = {
        "tokenizer": tokenizer_name,
        "n_prompts": len(all_prompts),
        "gate_ratio": args.gate_ratio,
        "per_type_by_k": summary,
        "gate_results": gates,
        "overall_status": (
            "pass" if all(g.get("status") == "pass" for g in gates.values()) else "fail"
        ),
    }

    print("\nPer-type token-length summary:")
    for t, by_k in summary.items():
        print(f"  {t}:")
        for k, stats in by_k.items():
            print(
                f"    k={k}: n={stats['n_prompts']} "
                f"median={stats['median']:.1f} "
                f"mean={stats['mean']:.1f} "
                f"min={stats['min']} max={stats['max']}"
            )

    lo = 1.0 / args.gate_ratio
    hi = args.gate_ratio
    print(
        "\nGate results (median k=8 / median k=1 in [{:.2f}, {:.2f}]):".format(lo, hi)
    )
    for t, g in gates.items():
        status = g.get("status", "unknown")
        if status == "pass" or status == "fail":
            print(
                f"  [{status.upper()}] {t}: "
                f"median k=1={g['median_k1']:.1f}, k=8={g['median_k8']:.1f}, "
                f"ratio={g['ratio_k8_over_k1']:.2f}"
            )
        else:
            print(f"  [{status.upper()}] {t}: {g.get('reason')}")

    report_path = out_dir / "token_length_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {report_path}")

    plot_path = out_dir / "token_length_violin.png"
    _maybe_plot(records, plot_path)
    if plot_path.exists():
        print(f"Plot written to {plot_path}")

    sys.exit(0 if report["overall_status"] == "pass" else 2)


if __name__ == "__main__":
    main()
