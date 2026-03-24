"""Diagnose pilot results: show per-prompt breakdown to identify failure patterns."""

import json
from collections import defaultdict
from pathlib import Path


def main():
    results = json.loads(
        Path("compscale/evaluation/results/pilot_results.json").read_text()
    )

    # Group by prompt
    by_prompt = defaultdict(list)
    for r in results:
        by_prompt[r["prompt_id"]].append(r)

    # Load prompts for context
    prompts = json.loads(
        Path("compscale/benchmark/prompts/numeracy_pilot_v2.json").read_text()
    )
    prompt_lookup = {p["id"]: p for p in prompts}

    print("=" * 80)
    print("PER-PROMPT DIAGNOSTIC")
    print("=" * 80)

    for k in [1, 2, 4, 8]:
        print(f"\n{'─' * 80}")
        print(f"  k = {k}")
        print(f"{'─' * 80}")

        k_prompts = [pid for pid, rs in by_prompt.items() if rs[0]["k"] == k]
        k_prompts.sort()

        for pid in k_prompts:
            rs = by_prompt[pid]
            prompt_info = prompt_lookup[pid]
            avg_sat = sum(r["satisfaction_rate"] for r in rs) / len(rs)
            n_all_sat = sum(1 for r in rs if r["all_satisfied"])

            print(f"\n  {pid}: avg_satisfaction={avg_sat:.2f}, all_satisfied={n_all_sat}/{len(rs)}")

            # Show per-constraint breakdown across all images
            for c_idx, constraint in enumerate(prompt_info["constraints"]):
                expected = constraint["count"]
                color = constraint["color"]
                obj = constraint["object"]

                vlm_answers = []
                for r in rs:
                    if c_idx < len(r["constraints"]):
                        cr = r["constraints"][c_idx]
                        vlm_answers.append(cr.get("vlm_answer"))

                n_correct = sum(1 for a in vlm_answers if a == expected)
                print(
                    f"    {expected} {color} {obj}: "
                    f"VLM answers={vlm_answers}, "
                    f"correct={n_correct}/{len(vlm_answers)}"
                )

    # Summary: which objects/counts are hardest?
    print(f"\n{'=' * 80}")
    print("FAILURE ANALYSIS BY OBJECT AND COUNT")
    print(f"{'=' * 80}")

    by_object = defaultdict(lambda: {"total": 0, "satisfied": 0})
    by_count = defaultdict(lambda: {"total": 0, "satisfied": 0})
    by_color = defaultdict(lambda: {"total": 0, "satisfied": 0})

    for r in results:
        for c in r["constraints"]:
            by_object[c["object"]]["total"] += 1
            by_count[c["count"]]["total"] += 1
            by_color[c["color"]]["total"] += 1
            if c["satisfied"]:
                by_object[c["object"]]["satisfied"] += 1
                by_count[c["count"]]["satisfied"] += 1
                by_color[c["color"]]["satisfied"] += 1

    print("\nBy expected count:")
    for count in sorted(by_count.keys()):
        d = by_count[count]
        rate = d["satisfied"] / d["total"] if d["total"] > 0 else 0
        print(f"  count={count}: {d['satisfied']}/{d['total']} ({rate:.1%})")

    print("\nBy object (sorted by accuracy):")
    for obj, d in sorted(by_object.items(), key=lambda x: x[1]["satisfied"] / max(x[1]["total"], 1)):
        rate = d["satisfied"] / d["total"] if d["total"] > 0 else 0
        print(f"  {obj}: {d['satisfied']}/{d['total']} ({rate:.1%})")

    print("\nBy color (sorted by accuracy):")
    for color, d in sorted(by_color.items(), key=lambda x: x[1]["satisfied"] / max(x[1]["total"], 1)):
        rate = d["satisfied"] / d["total"] if d["total"] > 0 else 0
        print(f"  {color}: {d['satisfied']}/{d['total']} ({rate:.1%})")


if __name__ == "__main__":
    main()
