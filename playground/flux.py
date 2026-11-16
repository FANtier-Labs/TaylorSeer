import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from diffusers import FluxPipeline

from taylorseer import CacheManager, FLUXAdapter, TaylorSeerConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run FLUX image generation")
    parser.add_argument("--enable-taylorseer", action="store_true")
    parser.add_argument("--model-path", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--interval", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", default="flux.png")
    parser.add_argument("--prompt", default="A cat holding a sign that says hello world")
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    return parser.parse_args()


@torch.inference_mode()
def main():
    # parse arguments and load model
    args = parse_args()
    pipe = FluxPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to("cuda")

    # apply monkeypatch
    adapter = nullcontext()
    if args.enable_taylorseer:
        manager = CacheManager(TaylorSeerConfig(args.order, args.interval, args.warmup))
        adapter = FLUXAdapter().patch(pipe, manager)

    param_peak_memory = torch.cuda.max_memory_allocated(device="cuda") / 1e9
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with adapter:
        start.record()
        torch.cuda.synchronize()
        image = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=torch.Generator("cpu").manual_seed(args.seed),
        ).images[0]
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) * 1e-3
    peak_memory = torch.cuda.max_memory_allocated(device="cuda")

    # save image.
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)

    print(f"Elapsed time: {elapsed:.2f} sec")
    print(f"Parameter memory: {param_peak_memory:.2f} GB")
    print(f"Runtime memory: {peak_memory:.2f} GB")


if __name__ == "__main__":
    main()
