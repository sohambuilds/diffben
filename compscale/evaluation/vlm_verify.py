"""VLM-based per-constraint verification for all four sanity-suite types.

Routes on ``constraint["type"]`` explicitly:

* ``attribute`` — "What color is the {object}?"; satisfied if expected color
  substring appears in the answer.
* ``numeracy``  — "How many {color} {plural}?"; parse integer; satisfied if
  parsed == expected count.
* ``negation``  — "Is there a {object} visible in this image?"; satisfied if
  answer parses as "no".
* ``spatial``   — "Is the {a} {relation} the {b}?"; satisfied if answer
  parses as "yes".

Backward-compat: constraints without an explicit ``type`` field are routed by
heuristic (``count`` key -> numeracy, else attribute) so legacy
``attribute_pilot.json`` and ``numeracy_pilot_v2.json`` still work.

Outputs a single result JSON with one record per (prompt, image) carrying
``type``, ``k``, per-constraint answers, and aggregate satisfaction.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from google import genai
from PIL import Image


# --- Answer parsing -----------------------------------------------------


def parse_count(response: str) -> int | None:
    response = response.strip()
    try:
        return int(response)
    except ValueError:
        match = re.search(r"\d+", response)
        return int(match.group()) if match else None


_AFFIRMATIVE = {"yes", "yeah", "yep", "correct", "true"}
_NEGATIVE = {"no", "nope", "not", "false"}


def parse_yes_no(response: str) -> str | None:
    """Returns 'yes', 'no', or None if ambiguous."""
    r = response.strip().lower()
    r = re.sub(r"[^a-z ,.]", "", r)
    first = r.split()[0] if r.split() else ""
    if first in _AFFIRMATIVE:
        return "yes"
    if first in _NEGATIVE:
        return "no"
    if any(w in _NEGATIVE for w in r.split()[:3]):
        return "no"
    if any(w in _AFFIRMATIVE for w in r.split()[:3]):
        return "yes"
    return None


# --- Type inference (backward compat) ----------------------------------


def infer_type(constraint):
    if "type" in constraint:
        return constraint["type"]
    if "count" in constraint:
        return "numeracy"
    if "relation" in constraint or "object_a" in constraint:
        return "spatial"
    if "color" in constraint and "object" in constraint:
        return "attribute"
    raise ValueError(f"Cannot infer type for constraint: {constraint}")


# --- Per-type question templates ---------------------------------------


def question_for(constraint, ctype):
    if ctype == "attribute":
        return (
            f"What color is the {constraint['object']} in this image? "
            f"Answer with just the color name."
        )
    if ctype == "numeracy":
        plural = constraint.get("plural")
        if not plural:
            obj = constraint["object"]
            plural = obj + ("es" if obj.endswith(("s", "x", "z", "ch", "sh")) else "s")
        return (
            f"How many {constraint['color']} {plural} are in this image? "
            f"Answer with just a number."
        )
    if ctype == "negation":
        return (
            f"Is there a {constraint['object']} visible in this image? "
            f"Answer yes or no."
        )
    if ctype == "spatial":
        return (
            f"Is the {constraint['object_a']} {constraint['relation']} the "
            f"{constraint['object_b']}? Answer yes or no."
        )
    raise ValueError(f"Unknown type: {ctype}")


def score_constraint(constraint, ctype, raw_answer):
    if ctype == "attribute":
        parsed = raw_answer.strip().lower()
        return parsed, constraint["color"].lower() in parsed
    if ctype == "numeracy":
        parsed = parse_count(raw_answer)
        return parsed, parsed == constraint["count"]
    if ctype == "negation":
        parsed = parse_yes_no(raw_answer)
        return parsed, parsed == "no"
    if ctype == "spatial":
        parsed = parse_yes_no(raw_answer)
        return parsed, parsed == "yes"
    raise ValueError(f"Unknown type: {ctype}")


# --- VLM call -----------------------------------------------------------


def ask_vlm(client, model_name: str, image_path: str, question: str) -> str:
    img = Image.open(image_path)
    response = client.models.generate_content(
        model=model_name,
        contents=[question, img],
    )
    return response.text or ""


# --- Driver -------------------------------------------------------------


def _iter_image_paths(prompt_dir: Path, n_images: int):
    for j in range(n_images):
        path = prompt_dir / f"img_{j}.png"
        if path.exists():
            yield j, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", nargs="+",
                        default=["compscale/benchmark/prompts/attribute_pilot_v2.json"],
                        help="One or more prompt JSON files")
    parser.add_argument("--images_dir",
                        default="compscale/generation/outputs/flux2-klein-9b",
                        help="Directory containing {prompt_id}/img_{j}.png. "
                             "Point at the model subfolder directly.")
    parser.add_argument("--output",
                        default="compscale/evaluation/results/sanity_results.json")
    parser.add_argument("--n_images", type=int, default=4)
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print VLM responses without writing results")
    parser.add_argument("--prompt_ids", nargs="*", default=None)
    parser.add_argument("--types", nargs="*", default=None,
                        help="Only evaluate these constraint types")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return
    client = genai.Client(api_key=api_key)

    prompts = []
    for path in args.prompts:
        p = Path(path)
        if not p.exists():
            print(f"  (skipping missing prompts file {p})")
            continue
        items = json.loads(p.read_text())
        print(f"  loaded {len(items)} prompts from {p}")
        prompts.extend(items)

    if args.prompt_ids:
        prompts = [p for p in prompts if p["id"] in args.prompt_ids]
    if args.types:
        prompts = [p for p in prompts if p.get("type") in set(args.types)]

    images_dir = Path(args.images_dir)
    all_results = []
    total_constraints = 0
    total_satisfied = 0

    for prompt_data in prompts:
        prompt_dir = images_dir / prompt_data["id"]
        if not prompt_dir.exists():
            print(f"  skip {prompt_data['id']}: no images dir")
            continue

        prompt_type = prompt_data.get("type")

        for img_idx, img_path in _iter_image_paths(prompt_dir, args.n_images):
            constraint_results = []
            for c in prompt_data["constraints"]:
                ctype = infer_type(c)
                q = question_for(c, ctype)
                try:
                    raw = ask_vlm(client, args.model, str(img_path), q)
                except Exception as e:
                    print(f"    VLM error on {img_path} / {ctype}: {e}")
                    raw = ""
                parsed, satisfied = score_constraint(c, ctype, raw)

                if args.dry_run:
                    print(
                        f"    {prompt_data['id']}/img_{img_idx} [{ctype}]: "
                        f"Q={q!r} A={raw.strip()!r} parsed={parsed} sat={satisfied}"
                    )

                constraint_results.append({
                    **c,
                    "type": ctype,
                    "vlm_question": q,
                    "vlm_raw": raw.strip(),
                    "vlm_answer": parsed,
                    "satisfied": bool(satisfied),
                })
                total_constraints += 1
                if satisfied:
                    total_satisfied += 1
                time.sleep(args.delay)

            sat_rate = (
                sum(1 for r in constraint_results if r["satisfied"])
                / len(constraint_results)
                if constraint_results else 0.0
            )
            result = {
                "prompt_id": prompt_data["id"],
                "type": prompt_type,
                "k": prompt_data["k"],
                "image_index": img_idx,
                "constraints": constraint_results,
                "satisfaction_rate": sat_rate,
                "all_satisfied": all(r["satisfied"] for r in constraint_results)
                                 if constraint_results else False,
            }
            all_results.append(result)
            print(
                f"  {prompt_data['id']}/img_{img_idx} "
                f"({prompt_type}, k={prompt_data['k']}): "
                f"satisfaction={sat_rate:.2f} "
                f"{'ALL' if result['all_satisfied'] else 'PARTIAL'}"
            )

    if args.dry_run:
        print("\n(dry run — no output file written)")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, indent=2))

    print(f"\nResults saved to {output_path}")
    if total_constraints > 0:
        print(
            f"Overall: {total_satisfied}/{total_constraints} constraints satisfied "
            f"({total_satisfied / total_constraints:.1%})"
        )


if __name__ == "__main__":
    main()
