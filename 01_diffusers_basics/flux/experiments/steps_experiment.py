"""Single-variable FLUX.1-schnell experiment: vary only num_inference_steps."""

import csv
import os
import time
from pathlib import Path

import torch
from diffusers import FluxPipeline


MODEL_ID = "black-forest-labs/FLUX.1-schnell"
PROMPT = "A small red cabin beside a calm mountain lake at sunrise, realistic photograph"
SEED = 42
GUIDANCE_SCALE = 0.0
HEIGHT = 1024
WIDTH = 1024
MAX_SEQUENCE_LENGTH = 256
STEP_COUNTS = [1, 2, 4, 8]

FLUX_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = FLUX_DIR / "outputs"
RESULTS_PATH = Path(__file__).resolve().with_name("steps_results.csv")


if not os.environ.get("HF_HOME"):
    raise RuntimeError("HF_HOME is not set. Configure it to the shared data-disk cache first.")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This experiment requires an NVIDIA GPU.")

OUTPUT_DIR.mkdir(exist_ok=True)

# Load once: model-loading time is excluded and is identical for every group.
pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

results = []
for num_inference_steps in STEP_COUNTS:
    # Recreate the same initial noise for every steps setting.
    generator = torch.Generator(device="cpu").manual_seed(SEED)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    image = pipe(
        prompt=PROMPT,
        generator=generator,
        num_inference_steps=num_inference_steps,
        guidance_scale=GUIDANCE_SCALE,
        height=HEIGHT,
        width=WIDTH,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    ).images[0]

    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start_time
    peak_allocated_mib = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved_mib = torch.cuda.max_memory_reserved() / 1024**2

    output_path = OUTPUT_DIR / f"flux_schnell_steps{num_inference_steps}_seed{SEED}.png"
    image.save(output_path)

    result = {
        "num_inference_steps": num_inference_steps,
        "inference_seconds": f"{inference_seconds:.3f}",
        "peak_allocated_mib": f"{peak_allocated_mib:.1f}",
        "peak_reserved_mib": f"{peak_reserved_mib:.1f}",
        "output_path": str(output_path),
    }
    results.append(result)
    print(result)

with RESULTS_PATH.open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"Saved results to: {RESULTS_PATH}")
