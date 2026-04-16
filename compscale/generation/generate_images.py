"""Generate images for the CompScale sanity suite using a FLUX.2 klein pipeline.

Supports:
* Single or multiple prompt JSON files in one run (``--prompts`` accepts a list).
* Model selector (``--model_id`` explicit or ``--klein_9b`` / ``--klein_4b`` convenience flags).
* Output segregation by model (``outputs/{model_short}/{prompt_id}/img_{j}.png``)
  so klein 4B and klein 9B runs don't collide.
* Stable per-prompt seeds derived from the prompt id, so re-running with a
  larger ``--n_images`` deterministically extends the existing set instead of
  shifting earlier images (critical for Module 5 seed-variance, where we
  take the same prompts from N=4 to N=16).
* Resumable: existing images are skipped.
"""

import argparse
import hashlib
import json
from pathlib import Path


MODEL_ALIASES = {
    "klein_4b": "black-forest-labs/FLUX.2-klein-4B",
    "klein_9b": "black-forest-labs/FLUX.2-klein-9B",
}


def _short_name(model_id: str) -> str:
    return model_id.split("/")[-1].lower().replace(".", "").replace("_", "-")


def _prompt_base_seed(prompt_id: str, base_seed: int) -> int:
    """Stable per-prompt base seed from prompt_id + a user-supplied base offset."""
    h = hashlib.sha256(prompt_id.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) + base_seed) % (2**31 - 1)


def _load_prompts(paths):
    all_prompts = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"  (skipping missing prompts file {p})")
            continue
        items = json.loads(p.read_text())
        print(f"  loaded {len(items)} prompts from {p}")
        all_prompts.extend(items)
    seen = set()
    deduped = []
    for p in all_prompts:
        if p["id"] in seen:
            continue
        seen.add(p["id"])
        deduped.append(p)
    if len(deduped) != len(all_prompts):
        print(f"  (deduped to {len(deduped)} unique prompts)")
    return deduped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model_id", default=None, help="HF model ID (explicit)")
    model_group.add_argument("--klein_4b", action="store_true")
    model_group.add_argument("--klein_9b", action="store_true")
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["compscale/benchmark/prompts/attribute_pilot_v2.json"],
        help="One or more prompt JSON files",
    )
    parser.add_argument(
        "--output_dir",
        default="compscale/generation/outputs",
        help="Base output directory. Images land in {output_dir}/{model_short}/{prompt_id}/",
    )
    parser.add_argument("--n_images", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42,
                        help="Additive base seed combined with sha256(prompt_id)")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument(
        "--prompt_ids", nargs="*", default=None,
        help="Only generate for these prompt IDs (for testing)",
    )
    parser.add_argument(
        "--start_image_index", type=int, default=0,
        help="Skip images 0..start_image_index-1 and start generating from here. "
             "Useful for extending an existing set from N=4 to N=16 without "
             "re-generating the first 4.",
    )
    parser.add_argument("--dry_run", action="store_true",
                        help="List what would be generated without loading the model")
    args = parser.parse_args()

    if args.klein_4b:
        model_id = MODEL_ALIASES["klein_4b"]
    elif args.klein_9b:
        model_id = MODEL_ALIASES["klein_9b"]
    elif args.model_id:
        model_id = args.model_id
    else:
        model_id = MODEL_ALIASES["klein_4b"]
    model_short = _short_name(model_id)

    prompts = _load_prompts(args.prompts)
    if args.prompt_ids:
        prompts = [p for p in prompts if p["id"] in args.prompt_ids]
        print(f"Filtered to {len(prompts)} prompts: {[p['id'] for p in prompts]}")

    base_out = Path(args.output_dir) / model_short
    base_out.mkdir(parents=True, exist_ok=True)

    to_generate = []
    for prompt_data in prompts:
        prompt_out = base_out / prompt_data["id"]
        for j in range(args.start_image_index, args.n_images):
            img_path = prompt_out / f"img_{j}.png"
            if img_path.exists():
                continue
            to_generate.append((prompt_data, j, img_path))
    total_existing = sum(
        1
        for p in prompts
        for j in range(args.start_image_index, args.n_images)
        if (base_out / p["id"] / f"img_{j}.png").exists()
    )
    print(
        f"\nPlan: {len(prompts)} prompts x images [{args.start_image_index}, {args.n_images}) "
        f"on model {model_short}. {total_existing} already exist, "
        f"{len(to_generate)} to generate."
    )

    if args.dry_run:
        for p, j, path in to_generate[:10]:
            seed = _prompt_base_seed(p["id"], args.seed) + j
            print(f"  would gen: {path}  (seed={seed})")
        if len(to_generate) > 10:
            print(f"  ... and {len(to_generate) - 10} more")
        return

    if not to_generate:
        print("Nothing to do. All images already exist.")
        return

    import torch  # noqa: PLC0415
    from diffusers import Flux2KleinPipeline  # noqa: PLC0415

    print(f"\nLoading {model_id}...")
    pipe = Flux2KleinPipeline.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    ).to("cuda")
    print("Model loaded.")

    generated = 0
    for prompt_data in prompts:
        prompt_out = base_out / prompt_data["id"]
        prompt_out.mkdir(parents=True, exist_ok=True)

        metadata_path = prompt_out / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            metadata.setdefault("images", [])
        else:
            metadata = {
                "prompt_id": prompt_data["id"],
                "k": prompt_data["k"],
                "type": prompt_data.get("type"),
                "prompt": prompt_data["prompt"],
                "constraints": prompt_data["constraints"],
                "model": model_id,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "images": [],
            }
        existing_indices = {im["index"] for im in metadata.get("images", [])}

        for j in range(args.start_image_index, args.n_images):
            img_path = prompt_out / f"img_{j}.png"
            if img_path.exists():
                if j not in existing_indices:
                    metadata["images"].append(
                        {
                            "index": j,
                            "seed": _prompt_base_seed(prompt_data["id"], args.seed) + j,
                            "path": str(img_path),
                        }
                    )
                continue

            seed = _prompt_base_seed(prompt_data["id"], args.seed) + j
            gen = torch.Generator(device="cuda").manual_seed(seed)
            image = pipe(
                prompt=prompt_data["prompt"],
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                generator=gen,
            ).images[0]
            image.save(img_path)
            metadata["images"].append(
                {"index": j, "seed": seed, "path": str(img_path)}
            )
            generated += 1
            print(
                f"  [{generated}/{len(to_generate)}] {prompt_data['id']}/img_{j}.png "
                f"seed={seed}"
            )

        metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"\nDone. Generated {generated} images.")


if __name__ == "__main__":
    main()
