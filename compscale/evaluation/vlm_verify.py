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

Budget modes (for API-credit-limited runs, e.g. Gemini 3 Flash Preview
billing constraints):

* ``--batch_constraints`` — bundle every constraint question for a single
  (prompt, image) pair into ONE VLM call. The model replies with one
  numbered line per question. Per-constraint scoring is identical to the
  unbatched path, so results are comparable. Call count drops from
  ``sum(k)`` to ``num_prompts * n_images``; on the core sanity suite that's
  ~3000 -> ~800 calls (a ~3.75x reduction). On any API failure or parse
  mismatch the row falls back to per-constraint calls.

* ``--max_prompts_per_k N`` — keep only the first N prompts at each
  (type, k) level. Uses the deterministic ordering from
  ``generate_prompts.py`` so re-runs hit the same subset.

* ``--max_k_levels`` — keep only the listed k levels (e.g. ``1 8``) to
  drop the middle of the ladder when you only need endpoint deltas.

The three flags compose. Module 1 baseline + Module 6 scaling fit remain
valid under ``--batch_constraints`` alone; stacking subsampling shortens
the scaling ladder and should only be used when credits are tight.
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


# --- Batched VQA (budget mode) -----------------------------------------
#
# Bundles every constraint question for a single (prompt, image) pair into
# one VLM call. The model replies with one numbered line per question; we
# split the response back into per-constraint raw answers that feed the
# existing ``score_constraint`` logic unchanged.


_BATCH_INSTRUCTION = (
    "Answer each of the following numbered questions about this image. "
    "Respond with ONE answer per line, in order, prefixed with the "
    "question number and a closing parenthesis (e.g. '1) red', '2) 3', "
    "'3) yes'). Do not add any other text, explanation, or blank lines. "
    "If a question has no clear answer, still reply on its own line "
    "(e.g. '4) unknown')."
)


_NUMBERED_LINE = re.compile(r"^\s*(\d+)\s*[\)\.\:\-]\s*(.+?)\s*$")


def _build_batched_prompt(questions: list[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    return f"{_BATCH_INSTRUCTION}\n\n{numbered}"


def _parse_batched_response(raw: str, n: int) -> list[str] | None:
    """Returns a list of ``n`` answer strings, or ``None`` on mismatch."""
    if not raw:
        return None
    answers: dict[int, str] = {}
    for line in raw.splitlines():
        m = _NUMBERED_LINE.match(line)
        if not m:
            continue
        idx = int(m.group(1))
        if 1 <= idx <= n and idx not in answers:
            answers[idx] = m.group(2)
    if len(answers) != n:
        return None
    return [answers[i + 1] for i in range(n)]


def ask_vlm_batched(
    client, model_name: str, image_path: str, questions: list[str]
) -> list[str] | None:
    if not questions:
        return []
    img = Image.open(image_path)
    prompt = _build_batched_prompt(questions)
    response = client.models.generate_content(
        model=model_name,
        contents=[prompt, img],
    )
    return _parse_batched_response(response.text or "", len(questions))


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
    parser.add_argument("--batch_constraints", action="store_true",
                        help="Budget mode: one VLM call per (prompt, image) "
                             "instead of one per constraint (~k-fold saving).")
    parser.add_argument("--max_prompts_per_k", type=int, default=None,
                        help="Budget mode: keep only the first N prompts per "
                             "(type, k) level. Deterministic by prompt order.")
    parser.add_argument("--max_k_levels", nargs="*", type=int, default=None,
                        help="Budget mode: keep only prompts at these k "
                             "levels (e.g. --max_k_levels 1 8).")
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
    if args.max_k_levels:
        keep_k = set(args.max_k_levels)
        prompts = [p for p in prompts if p.get("k") in keep_k]
    if args.max_prompts_per_k:
        # Deterministic: rely on the file order from generate_prompts.py.
        # Keep only the first N within each (type, k) bucket.
        kept, counts = [], {}
        for p in prompts:
            key = (p.get("type"), p.get("k"))
            counts[key] = counts.get(key, 0) + 1
            if counts[key] <= args.max_prompts_per_k:
                kept.append(p)
        dropped = len(prompts) - len(kept)
        if dropped:
            print(f"  max_prompts_per_k={args.max_prompts_per_k}: kept "
                  f"{len(kept)} prompts, dropped {dropped}")
        prompts = kept

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
            constraints = prompt_data["constraints"]
            ctypes = [infer_type(c) for c in constraints]
            questions = [question_for(c, t) for c, t in zip(constraints, ctypes)]
            raw_answers: list[str] = []

            used_batch = False
            if args.batch_constraints and constraints:
                try:
                    batched = ask_vlm_batched(
                        client, args.model, str(img_path), questions
                    )
                except Exception as e:
                    print(f"    batched VLM error on {img_path}: {e} "
                          f"(falling back to per-constraint)")
                    batched = None
                if batched is not None:
                    raw_answers = batched
                    used_batch = True
                    time.sleep(args.delay)
                else:
                    print(f"    batch parse failed on {prompt_data['id']}/"
                          f"img_{img_idx}; falling back to per-constraint")

            if not used_batch:
                for q in questions:
                    try:
                        raw = ask_vlm(client, args.model, str(img_path), q)
                    except Exception as e:
                        print(f"    VLM error on {img_path}: {e}")
                        raw = ""
                    raw_answers.append(raw)
                    time.sleep(args.delay)

            constraint_results = []
            for c, ctype, q, raw in zip(constraints, ctypes, questions, raw_answers):
                parsed, satisfied = score_constraint(c, ctype, raw)
                if args.dry_run:
                    print(
                        f"    {prompt_data['id']}/img_{img_idx} [{ctype}]: "
                        f"Q={q!r} A={raw.strip()!r} parsed={parsed} "
                        f"sat={satisfied} batched={used_batch}"
                    )
                constraint_results.append({
                    **c,
                    "type": ctype,
                    "vlm_question": q,
                    "vlm_raw": raw.strip(),
                    "vlm_answer": parsed,
                    "satisfied": bool(satisfied),
                    "vlm_batched": used_batch,
                })
                total_constraints += 1
                if satisfied:
                    total_satisfied += 1

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
