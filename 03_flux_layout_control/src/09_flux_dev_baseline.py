"""Generate a fixed FLUX.1-dev baseline and save its denoising state.

This is the dev counterpart of ``00_flux_schnell_baseline.py``.  The initial
packed latents are created once and passed explicitly to the official
``FluxPipeline``.  A callback records the packed latent after every scheduler
step so the explicit implementation can be compared step by step later.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import diffusers
import numpy as np
import torch
from diffusers import FluxPipeline


MODEL_ID = "black-forest-labs/FLUX.1-dev"
PROMPT = "a red apple and a blue cup, realistic photo"
SEED = 42
WIDTH = 768
HEIGHT = 768
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 3.5
DTYPE = torch.bfloat16
OFFLOAD_STRATEGY = "enable_sequential_cpu_offload"

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs" / "baseline"
IMAGE_PATH = OUTPUT_DIR / "flux_dev_baseline_seed-42_steps-50.png"
METRICS_PATH = OUTPUT_DIR / "flux_dev_baseline_metrics.json"
STATE_PATH = OUTPUT_DIR / "flux_dev_baseline_state.pt"


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor values without relying on NumPy bfloat16 support."""
    cpu_tensor = tensor.detach().cpu().contiguous()
    raw = cpu_tensor.view(torch.uint16).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def pixel_sha256(image) -> str:
    return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def cuda_peaks() -> dict[str, float]:
    return {
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def make_initial_latents(pipe: FluxPipeline, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    num_channels_latents = pipe.transformer.config.in_channels // 4
    return pipe.prepare_latents(
        1,
        num_channels_latents,
        HEIGHT,
        WIDTH,
        DTYPE,
        device,
        generator,
        latents=None,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this FLUX experiment.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    load_start = time.perf_counter()
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE, local_files_only=True)
    pipe.enable_sequential_cpu_offload()
    load_runtime = time.perf_counter() - load_start

    device = pipe._execution_device
    initial_latents, initial_image_ids = make_initial_latents(pipe, device)
    initial_latents_cpu = initial_latents.detach().cpu().clone()

    pipeline_step_latents: list[torch.Tensor] = []

    def capture_step(_pipe, _step_index, _timestep, callback_kwargs):
        pipeline_step_latents.append(callback_kwargs["latents"].detach().cpu().clone())
        return callback_kwargs

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    generation_start = time.perf_counter()
    result = pipe(
        prompt=PROMPT,
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        num_images_per_prompt=1,
        generator=generator,
        latents=initial_latents.clone(),
        callback_on_step_end=capture_step,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    generation_runtime = time.perf_counter() - generation_start
    image = result.images[0]
    image.save(IMAGE_PATH)

    if len(pipeline_step_latents) != NUM_INFERENCE_STEPS:
        raise RuntimeError(
            f"Expected {NUM_INFERENCE_STEPS} callback states, got {len(pipeline_step_latents)}"
        )
    final_latents_cpu = pipeline_step_latents[-1]
    step_latents_cpu = torch.stack(pipeline_step_latents)

    state = {
        "model_id": MODEL_ID,
        "prompt": PROMPT,
        "seed": SEED,
        "width": WIDTH,
        "height": HEIGHT,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "dtype": str(DTYPE),
        "initial_latents": initial_latents_cpu,
        "pipeline_step_latents": step_latents_cpu,
        "pipeline_final_latents": final_latents_cpu,
        "latent_image_ids": initial_image_ids.detach().cpu().clone(),
    }
    torch.save(state, STATE_PATH)

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
        "initial_latents_shape": list(initial_latents_cpu.shape),
        "initial_latents_sha256": tensor_sha256(initial_latents_cpu),
        "pipeline_final_packed_latents_shape": list(final_latents_cpu.shape),
        "pipeline_final_packed_latents_sha256": tensor_sha256(final_latents_cpu),
        "pipeline_step_count": len(pipeline_step_latents),
        "scheduler_class": type(pipe.scheduler).__name__,
        "scheduler_config": {
            "shift": pipe.scheduler.config.shift,
            "use_dynamic_shifting": pipe.scheduler.config.use_dynamic_shifting,
            "base_shift": pipe.scheduler.config.base_shift,
            "max_shift": pipe.scheduler.config.max_shift,
        },
        "transformer_guidance_embeds": bool(pipe.transformer.config.guidance_embeds),
        "runtime_seconds": generation_runtime,
        "model_load_seconds": load_runtime,
        "image_path": str(IMAGE_PATH),
        "state_path": str(STATE_PATH),
        "pixel_sha256": pixel_sha256(image),
        **cuda_peaks(),
    }
    METRICS_PATH.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))

    pipe.maybe_free_model_hooks()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
