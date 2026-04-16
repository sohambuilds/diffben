"""Module 2 — VLM question-template reliability check.

Reads ``compscale/evaluation/human_eval/hand_labels.csv`` (filled in by Soham
from the template emitted by ``sample_for_labeling.py``), runs the VLM on each
(image, constraint) pair using the same question template the sanity verifier
uses, and compares the VLM judgment to the human label.

Gates:
* Per constraint type, agreement with human label must be > 0.9.
* For negation specifically, the false-positive rate ("yes, the object is
  there" when human says no, i.e. the VLM missed the absence) is reported
  separately — a skewed evaluator would systematically misread negation.

Writes ``compscale/sanity/vlm_reliability_report.json``.
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from google import genai
from PIL import Image

from vlm_verify import infer_type, question_for, score_constraint

RELIABILITY_GATE = 0.90


def ask_vlm(client, model_name: str, image_path: str, question: str) -> str:
    img = Image.open(image_path)
    response = client.models.generate_content(
        model=model_name,
        contents=[question, img],
    )
    return response.text or ""


def _norm_human(label: str) -> bool | None:
    l = label.strip().lower()
    if l in {"sat", "yes", "y", "1", "true", "satisfied"}:
        return True
    if l in {"unsat", "no", "n", "0", "false", "unsatisfied"}:
        return False
    return None


def _confusion(records):
    tp = fp = tn = fn = 0
    for r in records:
        human = r["human_satisfied"]
        vlm = r["vlm_satisfied"]
        if human and vlm:
            tp += 1
        elif human and not vlm:
            fn += 1
        elif not human and vlm:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels",
                        default="compscale/evaluation/human_eval/hand_labels.csv")
    parser.add_argument("--output",
                        default="compscale/sanity/vlm_reliability_report.json")
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--gate", type=float, default=RELIABILITY_GATE)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    if not labels_path.exists():
        print(f"Labels file {labels_path} not found. Run sample_for_labeling.py "
              f"first, fill in the 'human_label' column, then save as hand_labels.csv.")
        return

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return
    client = genai.Client(api_key=api_key)

    rows = []
    with labels_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            human = _norm_human(row.get("human_label", ""))
            if human is None:
                continue
            rows.append((row, human))
    print(f"Loaded {len(rows)} labeled rows")

    records_by_type = defaultdict(list)
    for row, human_sat in rows:
        constraint = json.loads(row["constraint_json"])
        ctype = row.get("constraint_type") or infer_type(constraint)
        q = question_for(constraint, ctype)
        try:
            raw = ask_vlm(client, args.model, row["image_path"], q)
        except Exception as e:
            print(f"    VLM error on {row['image_path']}: {e}")
            raw = ""
        parsed, vlm_sat = score_constraint(constraint, ctype, raw)
        agreement = bool(vlm_sat) == bool(human_sat)

        rec = {
            "image_path": row["image_path"],
            "prompt_id": row.get("prompt_id"),
            "constraint_type": ctype,
            "k": int(row.get("k") or 0),
            "constraint": constraint,
            "question": q,
            "vlm_raw": raw.strip(),
            "vlm_parsed": parsed,
            "vlm_satisfied": bool(vlm_sat),
            "human_satisfied": bool(human_sat),
            "agreement": agreement,
        }
        records_by_type[ctype].append(rec)
        if args.dry_run:
            print(f"  [{ctype}] {row['image_path']}: "
                  f"vlm={vlm_sat} human={human_sat} match={agreement}")
        time.sleep(args.delay)

    type_reports = {}
    for ctype, recs in records_by_type.items():
        n = len(recs)
        n_agree = sum(1 for r in recs if r["agreement"])
        accuracy = n_agree / n if n > 0 else 0.0
        confusion = _confusion(recs)
        gate_status = "pass" if accuracy >= args.gate else "fail"
        type_reports[ctype] = {
            "n": n,
            "agreement": round(accuracy, 4),
            "gate": args.gate,
            "status": gate_status,
            "confusion": confusion,
            "fpr": (confusion["fp"] / (confusion["fp"] + confusion["tn"])
                    if (confusion["fp"] + confusion["tn"]) > 0 else 0.0),
            "fnr": (confusion["fn"] / (confusion["fn"] + confusion["tp"])
                    if (confusion["fn"] + confusion["tp"]) > 0 else 0.0),
        }
        print(
            f"  {ctype}: n={n} agreement={accuracy:.3f} "
            f"TP={confusion['tp']} FP={confusion['fp']} "
            f"TN={confusion['tn']} FN={confusion['fn']} [{gate_status}]"
        )

    overall_status = (
        "pass" if all(r["status"] == "pass" for r in type_reports.values())
        else "fail"
    )

    report = {
        "model": args.model,
        "gate": args.gate,
        "overall_status": overall_status,
        "per_type": type_reports,
        "records": [r for recs in records_by_type.values() for r in recs],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {out}")
    print(f"Overall status: {overall_status}")


if __name__ == "__main__":
    main()
