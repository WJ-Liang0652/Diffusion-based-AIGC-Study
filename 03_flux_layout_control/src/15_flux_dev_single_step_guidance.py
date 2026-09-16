"""Run one FLUX.1-dev Backward Guidance intervention at index 35.

Baseline and guided branches share the exact saved Stage-1 state entering
denoising index 35.  The guided branch takes one normalized negative-gradient
step on the Stage-3 Layout-B objective from Double block 18, then both branches
complete ordinary denoising.  No inner loop or multi-timestep guidance is used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import diffusers
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from PIL import Image, ImageDraw


PROJECT_DIR = Path(__file__).resolve().parents[1]
STAGE4_PATH = Path(__file__).with_name("14_flux_dev_gradient_probe.py")
OUTPUT_DIR = PROJECT_DIR / "outputs" / "single_step_guidance"
BASELINE_IMAGE_PATH = PROJECT_DIR / "outputs" / "baseline" / "flux_dev_baseline_seed-42_steps-50.png"

PROBE_DENOISING_INDEX = 35
DOUBLE_BLOCK_INDEX = 18
DEFAULT_RELATIVE_STEP_SIZE = 5e-4


def load_stage4():
    spec = importlib.util.spec_from_file_location("flux_dev_gradient_probe", STAGE4_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Stage-4 helpers from {STAGE4_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pixel_sha256(image: Image.Image) -> str:
    return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def strength_label(value: float) -> str:
    return format(value, ".6g").replace(".", "p").replace("-", "m")


def set_schedule(pipe: FluxPipeline, helpers, image_seq_len: int, device: torch.device) -> torch.Tensor:
    sigmas = np.linspace(1.0, 1 / helpers.NUM_INFERENCE_STEPS, helpers.NUM_INFERENCE_STEPS)
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.base_image_seq_len,
        pipe.scheduler.config.max_image_seq_len,
        pipe.scheduler.config.base_shift,
        pipe.scheduler.config.max_shift,
    )
    timesteps, _ = retrieve_timesteps(
        pipe.scheduler,
        helpers.NUM_INFERENCE_STEPS,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    return timesteps


def transformer_forward(pipe, state, latents: torch.Tensor, timestep_value: torch.Tensor) -> torch.Tensor:
    timestep = timestep_value.expand(latents.shape[0]).to(latents.dtype)
    return pipe.transformer(
        hidden_states=latents,
        timestep=timestep / 1000,
        guidance=state["guidance"],
        pooled_projections=state["pooled_prompt_embeds"],
        encoder_hidden_states=state["prompt_embeds"],
        txt_ids=state["text_ids"],
        img_ids=state["latent_image_ids"],
        joint_attention_kwargs=pipe.joint_attention_kwargs,
        return_dict=False,
    )[0]


@torch.no_grad()
def decode(pipe: FluxPipeline, helpers, packed_latents: torch.Tensor) -> Image.Image:
    latents = pipe._unpack_latents(
        packed_latents, helpers.HEIGHT, helpers.WIDTH, pipe.vae_scale_factor
    )
    latents = latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    decoded = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(decoded, output_type="pil")[0]


@torch.no_grad()
def continue_without_guidance(pipe, helpers, state) -> tuple[torch.Tensor, Image.Image, float]:
    start = time.perf_counter()
    latents = state["latents"].clone()
    timesteps = set_schedule(pipe, helpers, latents.shape[1], state["latents"].device)
    for timestep_value in timesteps[PROBE_DENOISING_INDEX:]:
        noise_pred = transformer_forward(pipe, state, latents, timestep_value)
        latents = pipe.scheduler.step(noise_pred, timestep_value, latents, return_dict=False)[0]
    image = decode(pipe, helpers, latents)
    return latents.detach(), image, time.perf_counter() - start


def compute_gradient(pipe, stage3, stage4, state, attention, original_processor, target_spans):
    probe = stage4.GraphLayoutAttentionProbe(original_processor, target_spans)
    attention.set_processor(probe)
    variable = state["latents"].detach().clone().requires_grad_(True)
    noise_pred = transformer_forward(pipe, state, variable, state["timestep"])
    initial_layout = stage4.serializable_layout(stage3, probe.maps, "layout_b")
    initial_maps = {word: value.detach().cpu().clone() for word, value in probe.maps.items()}
    objective, _ = stage4.layout_tensors(stage3, probe.maps, "layout_b")
    del noise_pred
    gradient = torch.autograd.grad(objective, variable, create_graph=False, retain_graph=False)[0]
    parameter_grads = [
        name for name, parameter in pipe.transformer.named_parameters() if parameter.grad is not None
    ]
    probe.release()
    attention.set_processor(original_processor)
    return variable.detach(), gradient.detach(), initial_layout, initial_maps, {
        "gradient_norm_l2": float(gradient.float().norm().item()),
        "gradient_abs_max": float(gradient.float().abs().max().item()),
        "gradient_abs_mean": float(gradient.float().abs().mean().item()),
        "gradient_is_finite": bool(torch.isfinite(gradient).all().item()),
        "gradient_is_nonzero": bool(torch.count_nonzero(gradient).item() > 0),
        "model_parameter_grad_count": len(parameter_grads),
        "model_parameters_with_grad": parameter_grads,
    }


@torch.no_grad()
def continue_with_single_update(
    pipe,
    helpers,
    stage3,
    stage4,
    state,
    updated_latents,
    attention,
    original_processor,
    target_spans,
):
    start = time.perf_counter()
    probe = stage4.GraphLayoutAttentionProbe(original_processor, target_spans)
    attention.set_processor(probe)
    latents = updated_latents.clone()
    timesteps = set_schedule(pipe, helpers, latents.shape[1], state["latents"].device)
    timestep_value = timesteps[PROBE_DENOISING_INDEX]
    noise_pred = transformer_forward(pipe, state, latents, timestep_value)
    updated_layout = stage4.serializable_layout(stage3, probe.maps, "layout_b")
    updated_maps = {word: value.detach().cpu().clone() for word, value in probe.maps.items()}
    probe.release()
    attention.set_processor(original_processor)
    latents = pipe.scheduler.step(noise_pred, timestep_value, latents, return_dict=False)[0]
    for timestep_value in timesteps[PROBE_DENOISING_INDEX + 1 :]:
        noise_pred = transformer_forward(pipe, state, latents, timestep_value)
        latents = pipe.scheduler.step(noise_pred, timestep_value, latents, return_dict=False)[0]
    image = decode(pipe, helpers, latents)
    return latents.detach(), image, updated_layout, updated_maps, time.perf_counter() - start


def map_statistics(attention_map: torch.Tensor) -> dict[str, object]:
    array = attention_map.double().numpy()
    distribution = array / array.sum()
    y_grid, x_grid = np.indices(array.shape)
    return {
        "center_of_mass_xy": [
            float((distribution * x_grid).sum()),
            float((distribution * y_grid).sum()),
        ],
        "left_half_mass_ratio": float(distribution[:, : array.shape[1] // 2].sum()),
        "right_half_mass_ratio": float(distribution[:, array.shape[1] // 2 :].sum()),
    }


def attention_change(initial_maps, updated_maps, initial_layout, updated_layout) -> dict[str, object]:
    records = {}
    target_directions_pass = []
    for word in initial_maps:
        before = map_statistics(initial_maps[word])
        after = map_statistics(updated_maps[word])
        target_region = initial_layout["objects"][word]["region"]
        target_mass_before = initial_layout["objects"][word]["mass_ratio"]
        target_mass_after = updated_layout["objects"][word]["mass_ratio"]
        center_delta_x = after["center_of_mass_xy"][0] - before["center_of_mass_xy"][0]
        center_moves_toward_target = center_delta_x > 0 if target_region == "right" else center_delta_x < 0
        target_mass_increased = target_mass_after > target_mass_before
        target_directions_pass.append(target_mass_increased and center_moves_toward_target)
        records[word] = {
            "target_region": target_region,
            "target_mass_before": target_mass_before,
            "target_mass_after": target_mass_after,
            "target_mass_delta": target_mass_after - target_mass_before,
            "center_of_mass_before_xy": before["center_of_mass_xy"],
            "center_of_mass_after_xy": after["center_of_mass_xy"],
            "center_of_mass_delta_xy": [
                center_delta_x,
                after["center_of_mass_xy"][1] - before["center_of_mass_xy"][1],
            ],
            "target_mass_increased": target_mass_increased,
            "center_moves_toward_target": center_moves_toward_target,
        }
    return {
        "objects": records,
        "both_attention_maps_move_toward_target": all(target_directions_pass),
    }


def image_difference(reference: Image.Image, candidate: Image.Image) -> dict[str, object]:
    first = np.asarray(reference.convert("RGB"), dtype=np.float32)
    second = np.asarray(candidate.convert("RGB"), dtype=np.float32)
    difference = np.abs(second - first)
    return {
        "pixel_exact_match": bool(np.array_equal(first, second)),
        "max_absolute_pixel_difference": float(difference.max()),
        "mean_absolute_pixel_difference": float(difference.mean()),
        "rmse": float(np.sqrt(np.mean((second - first) ** 2))),
        "changed_pixel_fraction": float(np.any(difference > 0, axis=2).mean()),
        "mean_signed_rgb_difference": [float(value) for value in (second - first).mean(axis=(0, 1))],
    }


def color_centroids(image: Image.Image, width: int, height: int) -> dict[str, object]:
    pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    masks = {
        "apple_red_proxy": (red > 90) & (red - green > 35) & (red - blue > 25),
        "cup_blue_proxy": (blue > 80) & (blue - red > 30) & (green - red > 15),
    }
    records = {}
    for name, mask in masks.items():
        ys, xs = np.nonzero(mask)
        records[name] = {
            "pixel_count": int(mask.sum()),
            "centroid_xy_pixels": [float(xs.mean()), float(ys.mean())] if len(xs) else None,
            "centroid_xy_normalized": [float(xs.mean() / width), float(ys.mean() / height)] if len(xs) else None,
        }
    return records


def centroid_changes(baseline, guided) -> dict[str, object]:
    records = {}
    for name in baseline:
        before = baseline[name]["centroid_xy_pixels"]
        after = guided[name]["centroid_xy_pixels"]
        records[name] = {
            "baseline_centroid_xy_pixels": before,
            "guided_centroid_xy_pixels": after,
            "delta_xy_pixels": [after[0] - before[0], after[1] - before[1]] if before and after else None,
            "baseline_pixel_count": baseline[name]["pixel_count"],
            "guided_pixel_count": guided[name]["pixel_count"],
        }
    return records


def save_image_comparison(baseline: Image.Image, guided: Image.Image, relative_step: float, path: Path) -> None:
    width, height = baseline.size
    header = 38
    baseline_array = np.asarray(baseline.convert("RGB"), dtype=np.int16)
    guided_array = np.asarray(guided.convert("RGB"), dtype=np.int16)
    amplified = np.clip(128 + 8 * (guided_array - baseline_array), 0, 255).astype(np.uint8)
    panels = [
        ("baseline", baseline.convert("RGB")),
        (f"guided relative={relative_step:g}", guided.convert("RGB")),
        ("signed pixel difference x8 + 128", Image.fromarray(amplified, mode="RGB")),
    ]
    canvas = Image.new("RGB", (width * len(panels), height + header), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        canvas.paste(image, (index * width, header))
        draw.text((index * width + 12, 11), label, fill="black")
    canvas.save(path)


def save_attention_comparison(initial_maps, updated_maps, timestep: float, path: Path) -> None:
    words = tuple(initial_maps)
    figure, axes = plt.subplots(len(words), 2, figsize=(10, 9), squeeze=False)
    for row, word in enumerate(words):
        vmin = min(float(initial_maps[word].min()), float(updated_maps[word].min()))
        vmax = max(float(initial_maps[word].max()), float(updated_maps[word].max()))
        for column, (label, attention_map) in enumerate(
            (("before", initial_maps[word]), ("after one update", updated_maps[word]))
        ):
            axis = axes[row, column]
            rendered = axis.imshow(attention_map.float().numpy(), cmap="magma", vmin=vmin, vmax=vmax)
            axis.axvline(23.5, color="cyan", linewidth=1.2)
            axis.set_title(f"{word}: {label}")
            axis.set_xlabel("image-token x")
            axis.set_ylabel("image-token y")
            figure.colorbar(rendered, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(f"Double block 18, denoise=35, t={timestep:.3f}, Layout B")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relative-step-size", type=float, default=DEFAULT_RELATIVE_STEP_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.relative_step_size <= 0:
        raise ValueError("--relative-step-size must be positive")
    stage4 = load_stage4()
    stage3 = stage4.load_stage3()
    helpers = stage3.load_helpers()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for FLUX.1-dev single-step guidance")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    label = strength_label(args.relative_step_size)
    baseline_path = OUTPUT_DIR / "flux_dev_single_step_baseline_seed-42_steps-50.png"
    guided_path = OUTPUT_DIR / f"flux_dev_single_step_guided_rel-{label}_seed-42_steps-50.png"
    comparison_path = OUTPUT_DIR / f"flux_dev_single_step_comparison_rel-{label}.png"
    attention_path = OUTPUT_DIR / f"flux_dev_single_step_attention_rel-{label}.png"
    metrics_path = OUTPUT_DIR / f"flux_dev_single_step_guidance_rel-{label}_metrics.json"

    total_start = time.perf_counter()
    baseline_state, baseline_metrics = helpers.load_baseline_state()
    pipe = FluxPipeline.from_pretrained(helpers.MODEL_ID, torch_dtype=helpers.DTYPE, local_files_only=True)
    target_spans, _ = helpers.locate_target_spans(pipe.tokenizer_2, helpers.PROMPT)
    for component_name in ("text_encoder", "text_encoder_2", "transformer", "vae"):
        component = getattr(pipe, component_name, None)
        if component is not None:
            component.requires_grad_(False)
    pipe.transformer.enable_gradient_checkpointing()
    pipe.enable_sequential_cpu_offload()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    state = stage4.prepare_probe_state(pipe, helpers, baseline_state)

    baseline_final_latents, baseline_image, baseline_runtime = continue_without_guidance(
        pipe, helpers, state
    )
    baseline_latent_difference = (
        baseline_final_latents.float().cpu() - baseline_state["pipeline_final_latents"].float()
    ).abs()
    baseline_image.save(baseline_path)
    baseline_hash = pixel_sha256(baseline_image)
    baseline_exact = bool(
        torch.equal(baseline_final_latents.cpu(), baseline_state["pipeline_final_latents"])
        and baseline_hash == baseline_metrics["pixel_sha256"]
    )
    if not baseline_exact:
        raise RuntimeError("Index-35 baseline continuation does not match the Stage-1 baseline")

    attention = pipe.transformer.transformer_blocks[DOUBLE_BLOCK_INDEX].attn
    original_processor = attention.get_processor()
    gradient_start = time.perf_counter()
    base_latents, gradient, initial_layout, initial_maps, gradient_metrics = compute_gradient(
        pipe, stage3, stage4, state, attention, original_processor, target_spans
    )
    gradient_runtime = time.perf_counter() - gradient_start
    latent_norm = base_latents.float().norm()
    gradient_norm = gradient.float().norm()
    desired_update_norm = args.relative_step_size * latent_norm
    effective_eta = desired_update_norm / gradient_norm
    update = (effective_eta * gradient.float()).to(base_latents.dtype)
    updated_latents = base_latents - update
    actual_update_norm = (updated_latents.float() - base_latents.float()).norm()

    guided_final_latents, guided_image, updated_layout, updated_maps, guided_runtime = (
        continue_with_single_update(
            pipe,
            helpers,
            stage3,
            stage4,
            state,
            updated_latents,
            attention,
            original_processor,
            target_spans,
        )
    )
    guided_image.save(guided_path)
    save_image_comparison(baseline_image, guided_image, args.relative_step_size, comparison_path)
    save_attention_comparison(
        initial_maps, updated_maps, float(state["timestep"].item()), attention_path
    )

    initial_energy = initial_layout["total_energy"]
    updated_energy = updated_layout["total_energy"]
    attention_movement = attention_change(initial_maps, updated_maps, initial_layout, updated_layout)
    baseline_centroids = color_centroids(baseline_image, helpers.WIDTH, helpers.HEIGHT)
    guided_centroids = color_centroids(guided_image, helpers.WIDTH, helpers.HEIGHT)
    record = {
        "stage": "Stage 5 - dev single-step Backward Guidance",
        "model_id": helpers.MODEL_ID,
        "diffusers_version": diffusers.__version__,
        "torch_version": torch.__version__,
        "prompt": helpers.PROMPT,
        "seed": helpers.SEED,
        "width": helpers.WIDTH,
        "height": helpers.HEIGHT,
        "num_inference_steps": helpers.NUM_INFERENCE_STEPS,
        "guidance_scale": helpers.GUIDANCE_SCALE,
        "dtype": str(helpers.DTYPE),
        "offload_strategy": helpers.OFFLOAD_STRATEGY,
        "block_index": DOUBLE_BLOCK_INDEX,
        "denoising_index": PROBE_DENOISING_INDEX,
        "timestep": float(state["timestep"].item()),
        "layout_optimized": stage3.LAYOUTS["layout_b"],
        "single_intervention_only": True,
        "inner_loop_count": 1,
        "guided_timestep_count": 1,
        "relative_step_size": args.relative_step_size,
        "update_definition": {
            "formula": "x_new = x - eta*g",
            "gradient": "g = d(E_layout_b)/dx",
            "eta": "relative_step * ||x||_2 / ||g||_2",
            "desired_update_norm": float(desired_update_norm.item()),
            "actual_bf16_update_norm": float(actual_update_norm.item()),
            "actual_relative_update_norm": float((actual_update_norm / latent_norm).item()),
            "effective_eta": float(effective_eta.item()),
        },
        "gradient": gradient_metrics,
        "layout_objective": {
            "before": initial_layout,
            "after": updated_layout,
            "delta_after_minus_before": updated_energy - initial_energy,
            "decreased": updated_energy < initial_energy,
        },
        "attention_movement": attention_movement,
        "baseline": {
            "image_path": str(baseline_path),
            "pixel_sha256": baseline_hash,
            "matches_stage1_final_latent_exactly": bool(
                torch.equal(baseline_final_latents.cpu(), baseline_state["pipeline_final_latents"])
            ),
            "final_latent_max_abs_difference": float(baseline_latent_difference.max().item()),
            "final_latent_mean_abs_difference": float(baseline_latent_difference.mean().item()),
            "matches_stage1_image_exactly": baseline_hash == baseline_metrics["pixel_sha256"],
            "color_centroids": baseline_centroids,
        },
        "guided": {
            "image_path": str(guided_path),
            "pixel_sha256": pixel_sha256(guided_image),
            "pixel_difference_vs_baseline": image_difference(baseline_image, guided_image),
            "color_centroids": guided_centroids,
            "color_centroid_changes_vs_baseline": centroid_changes(baseline_centroids, guided_centroids),
            "final_packed_latent_l2_difference_vs_baseline": float(
                (guided_final_latents.float() - baseline_final_latents.float()).norm().item()
            ),
        },
        "manual_visual_assessment": "pending",
        "baseline_path": str(baseline_path),
        "guided_path": str(guided_path),
        "comparison_path": str(comparison_path),
        "attention_comparison_path": str(attention_path),
        "baseline_runtime_seconds_from_index_35": baseline_runtime,
        "gradient_runtime_seconds": gradient_runtime,
        "guided_runtime_seconds_from_index_35": guided_runtime,
        "total_runtime_seconds_including_load": time.perf_counter() - total_start,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }
    metrics_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    if not (
        baseline_exact
        and gradient_metrics["gradient_is_finite"]
        and gradient_metrics["gradient_is_nonzero"]
        and gradient_metrics["model_parameter_grad_count"] == 0
        and updated_energy < initial_energy
    ):
        raise RuntimeError("Single-step guidance mechanical validation failed")

    pipe.maybe_free_model_hooks()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
