"""Generate and verify a minimal fixed-seed FLUX.1-schnell baseline.

The model is loaded only from the existing Hugging Face cache. Sequential CPU
offload is used because the complete BF16 FLUX pipeline does not fit safely in
24 GB of VRAM.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import diffusers
import torch
from diffusers import FluxPipeline


MODEL_ID = "black-forest-labs/FLUX.1-schnell"
PROMPT = "a red apple and a blue cup, realistic photo"
SEED = 42
WIDTH = 768
HEIGHT = 768
NUM_INFERENCE_STEPS = 4
GUIDANCE_SCALE = 0.0
DTYPE = torch.bfloat16
OFFLOAD_STRATEGY = "enable_sequential_cpu_offload"

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs" / "baseline"


def pixel_sha256(image) -> str:
    """Hash decoded RGB pixels, independent of PNG metadata/compression."""
    return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def generate_once(pipe: FluxPipeline, output_path: Path) -> dict[str, float | str]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device="cpu").manual_seed(SEED)

    start = time.perf_counter()
    image = pipe(
        prompt=PROMPT,
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        num_images_per_prompt=1,
        generator=generator,
    ).images[0]
    runtime_seconds = time.perf_counter() - start
    image.save(output_path)

    return {
        "runtime_seconds": runtime_seconds,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "pixel_sha256": pixel_sha256(image),
        "output_path": str(output_path),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this FLUX baseline.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        local_files_only=True,
    )
    pipe.enable_sequential_cpu_offload()

    first_path = OUTPUT_DIR / "flux-1-schnell_seed-42_steps-4_run-1.png"
    second_path = OUTPUT_DIR / "flux-1-schnell_seed-42_steps-4_run-2.png"
    first = generate_once(pipe, first_path)
    second = generate_once(pipe, second_path)
    reproducible = first["pixel_sha256"] == second["pixel_sha256"]

    record = {
        "model_id": MODEL_ID,
        "diffusers_version": diffusers.__version__,
        "torch_version": torch.__version__,
        "prompt": PROMPT,
        "seed": SEED,
        "width": WIDTH,
        "height": HEIGHT,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "dtype": str(DTYPE),
        "offload_strategy": OFFLOAD_STRATEGY,
        "first_run": first,
        "second_run": second,
        "fixed_seed_pixel_reproducible": reproducible,
    }
    record_path = OUTPUT_DIR / "flux-1-schnell_seed-42_steps-4_metrics.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))

    pipe.maybe_free_model_hooks()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
