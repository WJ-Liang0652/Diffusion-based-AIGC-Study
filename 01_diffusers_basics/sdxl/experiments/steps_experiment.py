"""Single-variable SDXL experiment: vary only num_inference_steps."""

import csv
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline


MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
PROMPT = "A small red cabin beside a calm mountain lake at sunrise, realistic photograph"
SEED = 42
GUIDANCE_SCALE = 7.0
HEIGHT = 1024
WIDTH = 1024
STEP_COUNTS = [10, 20, 30, 50]

SDXL_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = SDXL_DIR / "outputs"
RESULTS_PATH = Path(__file__).resolve().with_name("steps_results.csv")


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This experiment requires an NVIDIA GPU.")

OUTPUT_DIR.mkdir(exist_ok=True)

# Load the model once so loading time is not included in any steps measurement.
pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
pipe.to("cuda")

results = []
for num_inference_steps in STEP_COUNTS:
    # Recreate the same initial random latent for every steps setting.
    generator = torch.Generator(device="cuda").manual_seed(SEED)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    image = pipe(
        prompt=PROMPT,
        generator=generator,
        num_inference_steps=num_inference_steps,
        guidance_scale=GUIDANCE_SCALE,
        height=HEIGHT,
        width=WIDTH,
    ).images[0]

    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start_time
    peak_allocated_mib = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved_mib = torch.cuda.max_memory_reserved() / 1024**2

    output_path = OUTPUT_DIR / f"steps_{num_inference_steps}_seed{SEED}.png"
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
