"""Minimal FLUX.1-schnell text-to-image baseline using Hugging Face Diffusers."""

import os
import shutil
import time
from pathlib import Path

import torch
from diffusers import FluxPipeline


MODEL_ID = "black-forest-labs/FLUX.1-schnell"
PROMPT = "A small red cabin beside a calm mountain lake at sunrise, realistic photograph"
SEED = 42
NUM_INFERENCE_STEPS = 4
GUIDANCE_SCALE = 0.0
HEIGHT = 1024
WIDTH = 1024
MAX_SEQUENCE_LENGTH = 256


# FLUX model files must be cached on the data disk, not on the system disk.
hf_home = os.environ.get("HF_HOME")
if not hf_home:
    raise RuntimeError("HF_HOME is not set. Configure it to a directory on the data disk first.")

hf_home_path = Path(hf_home)
if not hf_home_path.is_dir():
    raise RuntimeError(f"HF_HOME does not exist: {hf_home_path}")

model_cache_path = hf_home_path / "hub" / "models--black-forest-labs--FLUX.1-schnell"
free_gib = shutil.disk_usage(hf_home_path).free / 1024**3
print(f"HF_HOME: {hf_home_path}")
print(f"Model cache location: {model_cache_path}")
print(f"Free space for model cache: {free_gib:.1f} GiB")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This baseline requires an NVIDIA GPU.")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# bfloat16 is the recommended FLUX dtype; CPU offload keeps the 24 GB GPU usable.
pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

# The official FLUX example uses a CPU generator with model CPU offload.
generator = torch.Generator(device="cpu").manual_seed(SEED)
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
start_time = time.perf_counter()

image = pipe(
    prompt=PROMPT,
    generator=generator,
    num_inference_steps=NUM_INFERENCE_STEPS,
    guidance_scale=GUIDANCE_SCALE,
    height=HEIGHT,
    width=WIDTH,
    max_sequence_length=MAX_SEQUENCE_LENGTH,
).images[0]

torch.cuda.synchronize()
inference_seconds = time.perf_counter() - start_time
peak_allocated_mib = torch.cuda.max_memory_allocated() / 1024**2
peak_reserved_mib = torch.cuda.max_memory_reserved() / 1024**2

output_path = Path(__file__).resolve().parent / "outputs" / "flux_schnell_seed42.png"
image.save(output_path)
print(f"Saved image to: {output_path}")
print(f"Pure pipe(...) inference time: {inference_seconds:.3f} seconds")
print(f"Peak allocated GPU memory: {peak_allocated_mib:.1f} MiB")
print(f"Peak reserved GPU memory: {peak_reserved_mib:.1f} MiB")
