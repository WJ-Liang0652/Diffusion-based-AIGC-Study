"""Align explicit FLUX.1-dev sampling with the official Diffusers pipeline.

The baseline state produced by ``09_flux_dev_baseline.py`` supplies the exact
same initial packed latents.  The explicit path follows Diffusers 0.32.2 for
prompt encoding, latent/image-ID preparation, dynamic-shifted FlowMatch
timesteps, guidance embeddings, transformer calls, scheduler steps, and VAE
decode.  Every scheduler state is compared with the saved Pipeline state.
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
OUTPUT_DIR = PROJECT_DIR / "outputs" / "explicit_sampling"
BASELINE_STATE_PATH = PROJECT_DIR / "outputs" / "baseline" / "flux_dev_baseline_state.pt"
IMAGE_PATH = OUTPUT_DIR / "flux_dev_explicit_seed-42_steps-50.png"
METRICS_PATH = OUTPUT_DIR / "flux_dev_explicit_sampling_metrics.json"
STATE_PATH = OUTPUT_DIR / "flux_dev_explicit_state.pt"


def tensor_sha256(tensor: torch.Tensor) -> str:
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


def image_difference(first, second) -> dict[str, object]:
    first_pixels = np.asarray(first.convert("RGB"), dtype=np.int16)
    second_pixels = np.asarray(second.convert("RGB"), dtype=np.int16)
    difference = np.abs(first_pixels - second_pixels)
    return {
        "max_absolute_pixel_difference": int(difference.max()),
        "mean_absolute_pixel_difference": float(difference.mean()),
        "pixel_exact_match": bool(np.array_equal(first_pixels, second_pixels)),
    }


def decode(pipe: FluxPipeline, packed_latents: torch.Tensor):
    latents = pipe._unpack_latents(packed_latents, HEIGHT, WIDTH, pipe.vae_scale_factor)
    latents = latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    decoded = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(decoded, output_type="pil")[0]


@torch.no_grad()
def run_explicit(
    pipe: FluxPipeline,
    initial_latents_cpu: torch.Tensor,
    reference_step_latents_cpu: torch.Tensor,
):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    # Match FluxPipeline.__call__ input state and prompt encoding.
    pipe.check_inputs(PROMPT, None, HEIGHT, WIDTH, max_sequence_length=512)
    pipe._guidance_scale = GUIDANCE_SCALE
    pipe._joint_attention_kwargs = {}
    pipe._interrupt = False
    device = pipe._execution_device
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=PROMPT,
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=512,
        lora_scale=None,
    )

    # Reuse the exact baseline initial state; prepare_latents only creates IDs
    # when a latent tensor is supplied.
    num_channels_latents = pipe.transformer.config.in_channels // 4
    initial_latents = initial_latents_cpu.to(device=device, dtype=prompt_embeds.dtype)
    latents, latent_image_ids = pipe.prepare_latents(
        1,
        num_channels_latents,
        HEIGHT,
        WIDTH,
        prompt_embeds.dtype,
        device,
        generator=None,
        latents=initial_latents,
    )
    if reference_step_latents_cpu.ndim != 4 or tuple(reference_step_latents_cpu.shape[1:]) != tuple(latents.shape):
        raise RuntimeError(
            "Baseline step-state shape mismatch: "
            f"reference={tuple(reference_step_latents_cpu.shape)}, current={(NUM_INFERENCE_STEPS, *latents.shape)}"
        )

    # Match FluxPipeline.__call__: sigma schedule plus sequence-length mu.
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

    # FLUX.1-dev has guidance_embeds=True; this is a model conditioning tensor,
    # not a second classifier-free transformer pass.
    guidance = (
        torch.full([1], GUIDANCE_SCALE, device=device, dtype=torch.float32).expand(latents.shape[0])
        if pipe.transformer.config.guidance_embeds
        else None
    )
    if guidance is None:
        raise RuntimeError("FLUX.1-dev unexpectedly has guidance_embeds=False")

    step_records = []
    first_divergence = None
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

        actual = latents.detach().cpu()
        reference = reference_step_latents_cpu[index]
        absolute_difference = (actual.float() - reference.float()).abs()
        step_record = {
            "index": index,
            "timestep": float(timestep_value.item()),
            "transformer_noise_pred_shape": list(noise_pred.shape),
            "updated_packed_latents_shape": list(latents.shape),
            "packed_latent_exact_match": bool(torch.equal(actual, reference)),
            "packed_latent_max_abs_diff": float(absolute_difference.max().item()),
            "packed_latent_mean_abs_diff": float(absolute_difference.mean().item()),
        }
        step_records.append(step_record)
        if first_divergence is None and not step_record["packed_latent_exact_match"]:
            first_divergence = step_record

    explicit_final_latents_cpu = latents.detach().cpu().clone()
    image = decode(pipe, latents)
    runtime = time.perf_counter() - start
    return image, explicit_final_latents_cpu, timesteps, mu, guidance, step_records, runtime


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this FLUX experiment.")
    if not BASELINE_STATE_PATH.exists():
        raise FileNotFoundError(f"Run the dev baseline first: {BASELINE_STATE_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_state = torch.load(BASELINE_STATE_PATH, map_location="cpu", weights_only=False)
    if baseline_state["model_id"] != MODEL_ID:
        raise RuntimeError(f"Baseline state model mismatch: {baseline_state['model_id']}")
    for key, expected in {
        "prompt": PROMPT,
        "seed": SEED,
        "width": WIDTH,
        "height": HEIGHT,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
    }.items():
        if baseline_state[key] != expected:
            raise RuntimeError(f"Baseline state {key} mismatch: {baseline_state[key]} != {expected}")

    load_start = time.perf_counter()
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE, local_files_only=True)
    pipe.enable_sequential_cpu_offload()
    load_runtime = time.perf_counter() - load_start

    baseline_initial = baseline_state["initial_latents"]
    reference_steps = baseline_state["pipeline_step_latents"]
    if tensor_sha256(baseline_initial) != baseline_state.get("initial_latents_sha256", tensor_sha256(baseline_initial)):
        raise RuntimeError("Baseline initial latent artifact failed its recorded hash check")

    image, final_latents, timesteps, mu, guidance, step_records, runtime = run_explicit(
        pipe, baseline_initial, reference_steps
    )
    image.save(IMAGE_PATH)

    baseline_image_path = PROJECT_DIR / "outputs" / "baseline" / "flux_dev_baseline_seed-42_steps-50.png"
    from PIL import Image

    baseline_image = Image.open(baseline_image_path)
    packed_difference = (final_latents.float() - baseline_state["pipeline_final_latents"].float()).abs()
    decoded_difference = image_difference(baseline_image, image)
    first_divergence = next((record for record in step_records if not record["packed_latent_exact_match"]), None)

    torch.save(
        {
            "model_id": MODEL_ID,
            "initial_latents": baseline_initial,
            "explicit_final_packed_latents": final_latents,
            "timesteps": timesteps.detach().cpu(),
            "mu": mu,
        },
        STATE_PATH,
    )
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
        "transformer_guidance_embeds": bool(pipe.transformer.config.guidance_embeds),
        "guidance_tensor_shape": list(guidance.shape),
        "guidance_tensor_values": guidance.detach().cpu().tolist(),
        "image_seq_len": int(reference_steps.shape[2]),
        "mu": float(mu),
        "timesteps": [float(value) for value in timesteps.detach().cpu().tolist()],
        "initial_latents_sha256": tensor_sha256(baseline_initial),
        "explicit_final_packed_latents_sha256": tensor_sha256(final_latents),
        "baseline_final_packed_latents_sha256": tensor_sha256(baseline_state["pipeline_final_latents"]),
        "final_packed_latent_max_abs_diff": float(packed_difference.max().item()),
        "final_packed_latent_mean_abs_diff": float(packed_difference.mean().item()),
        "decoded_image_difference": decoded_difference,
        "first_packed_latent_divergence": first_divergence,
        "step_records": step_records,
        "runtime_seconds": runtime,
        "model_load_seconds": load_runtime,
        "image_path": str(IMAGE_PATH),
        "state_path": str(STATE_PATH),
        **cuda_peaks(),
    }
    METRICS_PATH.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))

    pipe.maybe_free_model_hooks()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
