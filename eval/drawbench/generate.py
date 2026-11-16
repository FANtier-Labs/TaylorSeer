import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import time

import torch
from diffusers import FluxPipeline

from taylorseer import CacheManager, FLUXAdapter, TaylorSeerConfig

from .common import load, save, fingerprint, tasks, ensure_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=str(Path(__file__).with_name("data.jsonl")),
        help="JSONL with prompt and original metadata",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", choices=["baseline", "taylorseer"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--interval", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--limit", type=int, help="First N prompts for smoke tests")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if args.samples is not None and args.samples < 1:
        parser.error("samples must be positive")
    return args


@torch.inference_mode()
def main():
    # Parse arguments and prepare this worker's samples.
    args = parse_args()
    rank, world = int(os.getenv("RANK", 0)), int(os.getenv("WORLD_SIZE", 1))
    device = "cuda:" + os.getenv("LOCAL_RANK", "0")
    torch.cuda.set_device(device)
    prompts = load(args.data)
    if not prompts or any(not row.get("prompt") for row in prompts):
        raise ValueError("empty dataset or missing prompt")
    if args.limit is not None:
        prompts = prompts[: args.limit]
    count = 1 if args.samples is None else args.samples
    root = Path(args.output)
    config = dict(
        vars(args),
        data_sha256=fingerprint(args.data),
        count=count,
        expected=len(prompts) * count,
        dtype="bfloat16",
    )
    config.pop("output")
    root.mkdir(parents=True, exist_ok=True)
    ensure_config(root / "config.json", config)
    local = list(tasks(prompts, count, args.seed, rank, world))
    if not local:
        return

    # Load the model and configure TaylorSeer.
    pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(
        device
    )
    pipe.transformer.eval()
    manager = CacheManager(TaylorSeerConfig(args.order, args.interval, args.warmup))

    def generate(task):
        # Each image gets its own patch scope and fresh cache.
        adapter = nullcontext()
        if args.method == "taylorseer":
            adapter = FLUXAdapter().patch(pipe, manager)
        with adapter:
            return pipe(
                prompt=task["metadata"]["prompt"],
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=torch.Generator(device=device).manual_seed(task["seed"]),
            ).images[0]

    # Warm up before timing the dataset generation.
    generate(local[0])
    for task in local:
        index = task["sample_id"]
        image = root / f"{task['prompt_id']:05d}/samples/{task['sample_index']:04d}.png"
        record = root / f"records/{index:08d}.json"
        if image.is_file() and record.is_file():
            continue
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = generate(task)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        # Save the image and its generation record.
        image.parent.mkdir(parents=True, exist_ok=True)
        result.save(image)
        save(image.parent.parent / "metadata.jsonl", task["metadata"])
        save(record, dict(task, image=str(image.resolve()), seconds=seconds, rank=rank))


if __name__ == "__main__":
    main()
