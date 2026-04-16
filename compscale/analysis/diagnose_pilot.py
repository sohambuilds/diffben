"""Per-prompt / per-constraint breakdown across all four constraint types.

Generalized from the numeracy-only pilot tool: groups per-constraint results by
the natural key(s) of each type (count/color/object for numeracy, color/object
for attribute, object/scene for negation, relation for spatial).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _signature(c):
    t = c.get("type") or ("numeracy" if "count" in c else "attribute")
    if t == "numeracy":
        return f"num:{c['count']} {c['color']} {c['object']}"
    if t == "attribute":
        return f"attr:{c['color']} {c['object']}"
    if t == "negation":
        return f"neg:no {c['object']}"
    if t == "spatial":
        return f"spat:{c['object_a']} {c['relation']} {c['object_b']}"
    return json.dumps(c)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True,
                        help="vlm_verify results JSON")
    parser.add_argument("--type", default=None,
                        help="Optional filter: attribute/negation/spatial/numeracy")
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text())
    if args.type:
        results = [r for r in results if r.get("type") == args.type]

    by_prompt = defaultdict(list)
    for r in results:
        by_prompt[r["prompt_id"]].append(r)

    print("=" * 80)
    print("PER-PROMPT DIAGNOSTIC")
    print("=" * 80)

    k_levels = sorted({r["k"] for r in results if "k" in r})
    for k in k_levels:
        print(f"\n{'-' * 80}\n  k = {k}\n{'-' * 80}")
        k_prompts = [pid for pid, rs in by_prompt.items() if rs[0]["k"] == k]
        k_prompts.sort()
        for pid in k_prompts:
            rs = by_prompt[pid]
            avg = sum(r["satisfaction_rate"] for r in rs) / len(rs)
            all_sat = sum(1 for r in rs if r["all_satisfied"])
            print(f"\n  {pid}: avg={avg:.2f}, all_satisfied={all_sat}/{len(rs)}")
            if not rs[0].get("constraints"):
                continue
            n_constraints = len(rs[0]["constraints"])
            for c_idx in range(n_constraints):
                sig = _signature(rs[0]["constraints"][c_idx])
                sats = [int(r["constraints"][c_idx]["satisfied"]) for r in rs
                        if c_idx < len(r["constraints"])]
                answers = [r["constraints"][c_idx].get("vlm_answer") for r in rs
                           if c_idx < len(r["constraints"])]
                correct = sum(sats)
                print(f"    {sig}: answers={answers}, satisfied={correct}/{len(sats)}")

    print(f"\n{'=' * 80}\nAGGREGATE FAILURE ANALYSIS\n{'=' * 80}")

    by_sig = defaultdict(lambda: {"total": 0, "satisfied": 0})
    by_type_k = defaultdict(lambda: defaultdict(lambda: {"total": 0, "satisfied": 0}))
    for r in results:
        t = r.get("type")
        k = r.get("k")
        for c in r["constraints"]:
            sig = _signature(c)
            by_sig[sig]["total"] += 1
            by_type_k[t][k]["total"] += 1
            if c["satisfied"]:
                by_sig[sig]["satisfied"] += 1
                by_type_k[t][k]["satisfied"] += 1

    print("\nBy (type, k):")
    for t, by_k in by_type_k.items():
        for k, d in sorted(by_k.items()):
            rate = d["satisfied"] / d["total"] if d["total"] else 0
            print(f"  {t}/k={k}: {d['satisfied']}/{d['total']} ({rate:.1%})")

    print("\nWorst constraint signatures:")
    sorted_sigs = sorted(
        by_sig.items(),
        key=lambda kv: kv[1]["satisfied"] / max(kv[1]["total"], 1),
    )
    for sig, d in sorted_sigs[:20]:
        rate = d["satisfied"] / d["total"] if d["total"] else 0
        print(f"  {sig}: {d['satisfied']}/{d['total']} ({rate:.1%})")


if __name__ == "__main__":
    main()
