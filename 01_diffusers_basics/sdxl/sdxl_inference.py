"""Minimal SDXL text-to-image baseline using Hugging Face Diffusers."""

import os
import shutil
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline


MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
PROMPT = "A small red cabin beside a calm mountain lake at sunrise, realistic photograph"
SEED = 42
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.0


# Model files must be cached on the data disk, rather than the small system disk.
hf_home = os.environ.get("HF_HOME")
if not hf_home:
    raise RuntimeError("HF_HOME is not set. Configure it to a directory on the data disk first.")

hf_home_path = Path(hf_home)
if not hf_home_path.is_dir():
    raise RuntimeError(f"HF_HOME does not exist: {hf_home_path}")

free_gib = shutil.disk_usage(hf_home_path).free / 1024**3
print(f"HF_HOME: {hf_home_path}")
print(f"Free space for model cache: {free_gib:.1f} GiB")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This baseline requires an NVIDIA GPU.")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# float16 reduces VRAM usage and is well suited to RTX 4090 inference.
pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
pipe.to("cuda")

# A seeded CUDA generator makes the initial random latent reproducible.
generator = torch.Generator(device="cuda").manual_seed(SEED)
image = pipe(
    prompt=PROMPT,
    generator=generator,
    num_inference_steps=NUM_INFERENCE_STEPS,
    guidance_scale=GUIDANCE_SCALE,
).images[0]

output_path = Path(__file__).resolve().parent / "outputs" / "sdxl_seed42.png"
image.save(output_path)
print(f"Saved image to: {output_path}")
