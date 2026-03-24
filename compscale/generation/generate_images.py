"""Generate images for the CompScale pilot using FLUX.2 klein 4B."""

import argparse
import json
import torch
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate images for CompScale pilot")
    parser.add_argument(
        "--model_id",
        default="black-forest-labs/FLUX.2-klein-4B",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--prompts",
        default="compscale/benchmark/prompts/numeracy_pilot.json",
        help="Path to prompts JSON",
    )
    parser.add_argument(
        "--output_dir",
        default="compscale/generation/outputs",
        help="Output directory for generated images",
    )
    parser.add_argument("--n_images", type=int, default=4, help="Images per prompt")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--steps", type=int, default=4, help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=1.0, help="Guidance scale"
    )
    parser.add_argument(
        "--height", type=int, default=1024, help="Image height"
    )
    parser.add_argument(
        "--width", type=int, default=1024, help="Image width"
    )
    parser.add_argument(
        "--prompt_ids",
        nargs="*",
        default=None,
        help="Only generate for these prompt IDs (for testing)",
    )
    args = parser.parse_args()

    from diffusers import Flux2KleinPipeline

    print(f"Loading model: {args.model_id}")
    pipe = Flux2KleinPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()
    print("Model loaded.")

    prompts = json.loads(Path(args.prompts).read_text())

    if args.prompt_ids:
        prompts = [p for p in prompts if p["id"] in args.prompt_ids]
        print(f"Filtered to {len(prompts)} prompts: {[p['id'] for p in prompts]}")

    output_dir = Path(args.output_dir)
    total = len(prompts) * args.n_images
    generated = 0

    for i, prompt_data in enumerate(prompts):
        out_dir = output_dir / prompt_data["id"]
        out_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "prompt_id": prompt_data["id"],
            "k": prompt_data["k"],
            "prompt": prompt_data["prompt"],
            "constraints": prompt_data["constraints"],
            "model": args.model_id,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "images": [],
        }

        for j in range(args.n_images):
            img_path = out_dir / f"img_{j}.png"
            if img_path.exists():
                print(f"  Skipping {img_path} (already exists)")
                generated += 1
                metadata["images"].append(
                    {"index": j, "seed": args.seed + i * args.n_images + j, "path": str(img_path)}
                )
                continue

            seed = args.seed + i * args.n_images + j
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
            print(f"  [{generated}/{total}] Saved {img_path} (seed={seed})")

        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\nDone. Generated {generated} images across {len(prompts)} prompts.")


if __name__ == "__main__":
    main()
