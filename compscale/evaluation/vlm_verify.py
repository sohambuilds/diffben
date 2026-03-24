"""Evaluate generated images using Gemini 3 Flash Preview for counting verification."""

import argparse
import json
import os
import re
import time
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image


def parse_count(response: str) -> int | None:
    """Extract an integer count from VLM response."""
    response = response.strip()
    try:
        return int(response)
    except ValueError:
        match = re.search(r"\d+", response)
        return int(match.group()) if match else None


def ask_vlm(client, model_name: str, image_path: str, question: str) -> str:
    """Ask the VLM a question about an image."""
    img = Image.open(image_path)
    response = client.models.generate_content(
        model=model_name,
        contents=[question, img],
    )
    return response.text


def main():
    parser = argparse.ArgumentParser(description="VLM-based constraint verification")
    parser.add_argument(
        "--prompts",
        default="compscale/benchmark/prompts/numeracy_pilot_v2.json",
        help="Path to prompts JSON",
    )
    parser.add_argument(
        "--images_dir",
        default="compscale/generation/outputs",
        help="Directory containing generated images",
    )
    parser.add_argument(
        "--output",
        default="compscale/evaluation/results/pilot_results.json",
        help="Output results file",
    )
    parser.add_argument(
        "--n_images", type=int, default=4, help="Number of images per prompt"
    )
    parser.add_argument(
        "--model", default="gemini-3-flash-preview", help="Gemini model name"
    )
    parser.add_argument(
        "--delay", type=float, default=0.5, help="Delay between API calls (seconds)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print VLM responses without scoring",
    )
    parser.add_argument(
        "--prompt_ids",
        nargs="*",
        default=None,
        help="Only evaluate these prompt IDs",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return

    client = genai.Client(api_key=api_key)

    prompts = json.loads(Path(args.prompts).read_text())
    if args.prompt_ids:
        prompts = [p for p in prompts if p["id"] in args.prompt_ids]

    images_dir = Path(args.images_dir)
    all_results = []
    total_constraints = 0
    total_satisfied = 0

    for prompt_data in prompts:
        prompt_dir = images_dir / prompt_data["id"]
        if not prompt_dir.exists():
            print(f"Skipping {prompt_data['id']}: no images found")
            continue

        for img_idx in range(args.n_images):
            img_path = prompt_dir / f"img_{img_idx}.png"
            if not img_path.exists():
                continue

            constraint_results = []
            for c in prompt_data["constraints"]:
                constraint_type = c.get("type", "numeracy" if "count" in c else "attribute")

                if constraint_type == "attribute" or "count" not in c:
                    # Attribute binding: check if object has correct color
                    question = (
                        f"What color is the {c['object']} in this image? "
                        f"Answer with just the color name."
                    )
                    answer_text = ask_vlm(client, args.model, str(img_path), question)
                    parsed = answer_text.strip().lower()
                    satisfied = c["color"].lower() in parsed

                    if args.dry_run:
                        print(
                            f"  {prompt_data['id']}/img_{img_idx}: "
                            f"'{question}' -> '{answer_text.strip()}' "
                            f"(expected={c['color']}, satisfied={satisfied})"
                        )
                else:
                    # Numeracy: check count
                    obj = c["object"]
                    obj_plural = obj + "s" if not obj.endswith("s") else obj
                    question = (
                        f"How many {c['color']} {obj_plural} are in this image? "
                        f"Answer with just a number."
                    )
                    answer_text = ask_vlm(client, args.model, str(img_path), question)
                    parsed = parse_count(answer_text)
                    satisfied = parsed == c["count"]

                    if args.dry_run:
                        print(
                            f"  {prompt_data['id']}/img_{img_idx}: "
                            f"'{question}' -> '{answer_text.strip()}' "
                            f"(parsed={parsed}, expected={c['count']})"
                        )

                constraint_results.append(
                    {
                        **c,
                        "vlm_raw": answer_text.strip(),
                        "vlm_answer": parsed,
                        "satisfied": satisfied,
                    }
                )
                total_constraints += 1
                if satisfied:
                    total_satisfied += 1

                time.sleep(args.delay)

            sat_rate = (
                sum(1 for r in constraint_results if r["satisfied"])
                / len(constraint_results)
                if constraint_results
                else 0.0
            )
            result = {
                "prompt_id": prompt_data["id"],
                "k": prompt_data["k"],
                "image_index": img_idx,
                "constraints": constraint_results,
                "satisfaction_rate": sat_rate,
                "all_satisfied": all(r["satisfied"] for r in constraint_results),
            }
            all_results.append(result)
            print(
                f"  {prompt_data['id']}/img_{img_idx}: "
                f"satisfaction={sat_rate:.2f} "
                f"({'ALL' if result['all_satisfied'] else 'PARTIAL'})"
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, indent=2))

    print(f"\nResults saved to {output_path}")
    print(
        f"Overall: {total_satisfied}/{total_constraints} constraints satisfied "
        f"({total_satisfied / total_constraints:.1%})"
        if total_constraints > 0
        else "No constraints evaluated."
    )


if __name__ == "__main__":
    main()
