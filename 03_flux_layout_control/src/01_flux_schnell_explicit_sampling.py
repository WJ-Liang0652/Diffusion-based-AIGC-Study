"""Version-aligned explicit FLUX.1-schnell sampling for Diffusers 0.32.2.

The explicit path intentionally mirrors FluxPipeline.__call__'s prompt encoding,
latent preparation, timestep construction, denoising loop, unpacking, and VAE
decode. It is a baseline-alignment experiment only: no attention interception or
layout guidance is applied.
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
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps


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
OUTPUT_DIR = PROJECT_DIR / "outputs" / "explicit_sampling"


def pixel_sha256(image) -> str:
    return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def cuda_peaks() -> dict[str, float]:
    return {
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def run_pipeline(pipe: FluxPipeline, output_path: Path) -> tuple[object, dict[str, float | str]]:
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
    metrics = {"runtime_seconds": runtime_seconds, "pixel_sha256": pixel_sha256(image), **cuda_peaks()}
    return image, metrics


@torch.no_grad()
def run_explicit(pipe: FluxPipeline, output_path: Path) -> tuple[object, dict[str, object]]:
    """Expand the Diffusers 0.32.2 FluxPipeline.__call__ path without alteration."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    start = time.perf_counter()

    # __call__: input state and prompt encoding.
    pipe.check_inputs(PROMPT, None, HEIGHT, WIDTH, max_sequence_length=512)
    pipe._guidance_scale = GUIDANCE_SCALE
    pipe._joint_attention_kwargs = {}
    pipe._interrupt = False
    batch_size = 1
    device = pipe._execution_device
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=PROMPT,
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=512,
        lora_scale=None,
    )

    # __call__: FLUX packed noise latents and 3-coordinate image IDs.
    num_channels_latents = pipe.transformer.config.in_channels // 4
    latents, latent_image_ids = pipe.prepare_latents(
        batch_size,
        num_channels_latents,
        HEIGHT,
        WIDTH,
        prompt_embeds.dtype,
        device,
        generator,
        latents=None,
    )

    # __call__: FlowMatch Euler schedule, including FLUX's sequence-length shift.
    sigmas = np.linspace(1.0, 1 / NUM_INFERENCE_STEPS, NUM_INFERENCE_STEPS)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.base_image_seq_len,
        pipe.scheduler.config.max_image_seq_len,
        pipe.scheduler.config.base_shift,
        pipe.scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        NUM_INFERENCE_STEPS,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    pipe._num_timesteps = len(timesteps)
    guidance = (
        torch.full([1], GUIDANCE_SCALE, device=device, dtype=torch.float32).expand(latents.shape[0])
        if pipe.transformer.config.guidance_embeds
        else None
    )

    tensor_shapes: dict[str, object] = {
        "prompt_embeds_t5": list(prompt_embeds.shape),
        "pooled_prompt_embeds_clip": list(pooled_prompt_embeds.shape),
        "text_ids": list(text_ids.shape),
        "packed_latents_initial": list(latents.shape),
        "latent_image_ids": list(latent_image_ids.shape),
        "timesteps": list(timesteps.shape),
        "guidance": None if guidance is None else list(guidance.shape),
    }
    step_records = []

    # __call__: transformer prediction then scheduler x_t -> x_t-1 update.
    for index, timestep_value in enumerate(timesteps):
        timestep = timestep_value.expand(latents.shape[0]).to(latents.dtype)
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=pipe.joint_attention_kwargs,
            return_dict=False,
        )[0]
        latents = pipe.scheduler.step(noise_pred, timestep_value, latents, return_dict=False)[0]
        step_records.append(
            {
                "index": index,
                "timestep": float(timestep_value.item()),
                "transformer_noise_pred": list(noise_pred.shape),
                "updated_packed_latents": list(latents.shape),
            }
        )

    # __call__: unpack FLUX latents, reverse VAE normalization, decode and postprocess.
    unpacked_latents = pipe._unpack_latents(latents, HEIGHT, WIDTH, pipe.vae_scale_factor)
    vae_latents = (unpacked_latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    decoded = pipe.vae.decode(vae_latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
    runtime_seconds = time.perf_counter() - start
    image.save(output_path)
    pipe.maybe_free_model_hooks()

    metrics: dict[str, object] = {
        "runtime_seconds": runtime_seconds,
        "pixel_sha256": pixel_sha256(image),
        **cuda_peaks(),
        "tensor_shapes": tensor_shapes,
        "steps": step_records,
        "unpacked_latents": list(unpacked_latents.shape),
        "vae_latents": list(vae_latents.shape),
        "decoded_image_tensor": list(decoded.shape),
    }
    return image, metrics


def compare_images(pipeline_image, explicit_image) -> dict[str, object]:
    pipeline_pixels = np.asarray(pipeline_image.convert("RGB"), dtype=np.int16)
    explicit_pixels = np.asarray(explicit_image.convert("RGB"), dtype=np.int16)
    absolute_difference = np.abs(pipeline_pixels - explicit_pixels)
    return {
        "pixel_exact_match": bool(np.array_equal(pipeline_pixels, explicit_pixels)),
        "max_absolute_pixel_difference": int(absolute_difference.max()),
        "mean_absolute_pixel_difference": float(absolute_difference.mean()),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this FLUX experiment.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE, local_files_only=True)
    pipe.enable_sequential_cpu_offload()

    pipeline_path = OUTPUT_DIR / "flux-1-schnell_pipeline_seed-42_steps-4.png"
    explicit_path = OUTPUT_DIR / "flux-1-schnell_explicit_seed-42_steps-4.png"
    pipeline_image, pipeline_metrics = run_pipeline(pipe, pipeline_path)
    explicit_image, explicit_metrics = run_explicit(pipe, explicit_path)

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
        "pipeline": pipeline_metrics,
        "explicit": explicit_metrics,
        "comparison": compare_images(pipeline_image, explicit_image),
    }
    record_path = OUTPUT_DIR / "flux-1-schnell_explicit_sampling_metrics.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
