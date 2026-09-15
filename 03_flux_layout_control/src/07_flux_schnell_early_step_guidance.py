"""Test one layout-gradient intervention at t=750 and compare with t=500.

The Q/K probe, partial sampling, update normalization, and decode helpers are
loaded from the already validated t=500 experiment so timestep is the only
algorithmic variable changed here.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import FluxPipeline


PROJECT_DIR = Path(__file__).resolve().parents[1]
HELPER_PATH = Path(__file__).with_name("06_flux_schnell_single_step_guidance.py")
OUTPUT_DIR = PROJECT_DIR / "outputs" / "early_step_guidance"
T500_METRICS_PATH = (
    PROJECT_DIR / "outputs" / "single_step_guidance" / "flux-1-schnell_single-step-guidance_metrics.json"
)

PROBE_DENOISING_INDEX = 1
EXPECTED_TIMESTEP = 750.0
RELATIVE_STEP_SIZES = (0.002, 0.005)
LAYOUT_A = {"apple": "left", "cup": "right"}
LAYOUT_B = {"apple": "right", "cup": "left"}


def load_validated_helpers():
    spec = importlib.util.spec_from_file_location("flux_single_step_guidance_helpers", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper implementation from {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PROBE_DENOISING_INDEX = PROBE_DENOISING_INDEX
    module.RELATIVE_STEP_SIZES = RELATIVE_STEP_SIZES
    return module


def evaluate_layout(helpers, maps: dict[str, torch.Tensor], assignments: dict[str, str]) -> dict[str, object]:
    objects = {}
    for word, region in assignments.items():
        attention_map = maps[word]
        x0, y0, x1, y1 = helpers.bbox_to_grid(
            helpers.NORMALIZED_REGIONS[region], attention_map.shape[0], attention_map.shape[1]
        )
        ratio = attention_map[y0:y1, x0:x1].sum() / attention_map.sum()
        energy = (1.0 - ratio).square()
        objects[word] = {
            "region": region,
            "mass_ratio": float(ratio.detach().float().item()),
            "energy": float(energy.detach().float().item()),
        }
    return {"objects": objects, "total_energy": sum(item["energy"] for item in objects.values())}


@torch.no_grad()
def check_spatial_signal(helpers, pipe, state, attention, original_processor, target_spans):
    probe = helpers.GraphLayoutAttentionProbe(original_processor, target_spans)
    attention.set_processor(probe)
    noise_pred = helpers.transformer_forward(pipe, state, state["latents"], state["timestep"])
    del noise_pred
    layout_a = evaluate_layout(helpers, probe.maps, LAYOUT_A)
    layout_b = evaluate_layout(helpers, probe.maps, LAYOUT_B)
    probe.release()
    attention.set_processor(original_processor)
    reliable = (
        layout_a["total_energy"] < layout_b["total_energy"]
        and layout_a["objects"]["apple"]["mass_ratio"] > 0.5
        and layout_a["objects"]["cup"]["mass_ratio"] > 0.5
    )
    return {
        "layout_a_baseline_consistent": layout_a,
        "layout_b_swapped": layout_b,
        "energy_gap_b_minus_a": layout_b["total_energy"] - layout_a["total_energy"],
        "reliability_criterion": "A<B and baseline-consistent apple-left/cup-right mass ratios both exceed 0.5",
        "reliable_for_update": reliable,
    }


def centroid_shift_x(baseline_centroids: dict[str, object], guided_centroids: dict[str, object]) -> dict[str, float]:
    return {
        name: guided_centroids[name]["centroid_xy_pixels"][0] - baseline_centroids[name]["centroid_xy_pixels"][0]
        for name in ("apple_red_proxy", "cup_blue_proxy")
    }


def main() -> None:
    helpers = load_validated_helpers()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this FLUX experiment")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()

    pipe = FluxPipeline.from_pretrained(
        helpers.MODEL_ID,
        torch_dtype=helpers.DTYPE,
        local_files_only=True,
    )
    target_spans = helpers.locate_target_spans(pipe.tokenizer_2, helpers.PROMPT)
    for component_name in ("text_encoder", "text_encoder_2", "transformer", "vae"):
        component = getattr(pipe, component_name, None)
        if component is not None:
            component.requires_grad_(False)
    pipe.transformer.enable_gradient_checkpointing()
    pipe.enable_sequential_cpu_offload()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    state = helpers.prepare_t500_state(pipe)
    actual_timestep = float(state["timestep"].item())
    if actual_timestep != EXPECTED_TIMESTEP:
        raise RuntimeError(f"Expected t={EXPECTED_TIMESTEP}, got t={actual_timestep}")
    attention = pipe.transformer.transformer_blocks[helpers.DOUBLE_BLOCK_INDEX].attn
    original_processor = attention.get_processor()
    spatial_signal = check_spatial_signal(
        helpers, pipe, state, attention, original_processor, target_spans
    )

    common_record = {
        "model_id": helpers.MODEL_ID,
        "diffusers_version": helpers.diffusers.__version__,
        "torch_version": torch.__version__,
        "prompt": helpers.PROMPT,
        "seed": helpers.SEED,
        "width": helpers.WIDTH,
        "height": helpers.HEIGHT,
        "num_inference_steps": helpers.NUM_INFERENCE_STEPS,
        "guidance_scale": helpers.GUIDANCE_SCALE,
        "dtype": str(helpers.DTYPE),
        "offload_strategy": helpers.OFFLOAD_STRATEGY,
        "block_index": helpers.DOUBLE_BLOCK_INDEX,
        "denoising_index": PROBE_DENOISING_INDEX,
        "timestep": actual_timestep,
        "layout_b": LAYOUT_B,
        "normalized_regions": helpers.NORMALIZED_REGIONS,
        "target_spans": target_spans,
        "optimization_variable": {
            "name": "packed_latents_at_transformer_input_before_t750",
            "shape": list(state["latents"].shape),
            "dtype": str(state["latents"].dtype),
        },
        "update_definition": {
            "formula": "x_new = x - eta*g",
            "gradient": "g = d(E_layout_b)/dx",
            "eta": "relative_step * ||x||_2 / ||g||_2",
            "interpretation": "identical to t=500; BF16-rounded update norm is recorded",
        },
        "spatial_signal_check": spatial_signal,
    }
    metrics_path = OUTPUT_DIR / "flux-1-schnell_t750-single-step-guidance_metrics.json"
    if not spatial_signal["reliable_for_update"]:
        common_record.update(
            {
                "stopped_before_update": True,
                "stop_reason": "t=750 spatial signal failed the predeclared reliability criterion",
                "total_runtime_seconds_including_load": time.perf_counter() - total_start,
                "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            }
        )
        metrics_path.write_text(json.dumps(common_record, indent=2) + "\n")
        print(json.dumps(common_record, indent=2))
        pipe.maybe_free_model_hooks()
        return

    baseline_start = time.perf_counter()
    baseline_image = helpers.continue_baseline(pipe, state)
    baseline_runtime = time.perf_counter() - baseline_start
    baseline_path = OUTPUT_DIR / "flux-1-schnell_t750-single-step_baseline_seed-42_steps-4.png"
    baseline_image.save(baseline_path)

    gradient_start = time.perf_counter()
    base_latents, gradient, initial_layout, gradient_metrics = helpers.compute_gradient(
        pipe, state, attention, original_processor, target_spans
    )
    gradient_runtime = time.perf_counter() - gradient_start
    latent_norm = base_latents.float().norm()
    gradient_norm = gradient.float().norm()
    baseline_centroids = helpers.color_centroids(baseline_image)

    guided_results = []
    comparison_images = [("baseline", baseline_image)]
    for relative_step_size in RELATIVE_STEP_SIZES:
        branch_start = time.perf_counter()
        desired_update_norm = relative_step_size * latent_norm
        effective_eta = desired_update_norm / gradient_norm
        update = (effective_eta * gradient.float()).to(base_latents.dtype)
        updated_latents = base_latents - update
        actual_update_norm = (updated_latents.float() - base_latents.float()).norm()
        guided_image, updated_layout = helpers.continue_guided(
            pipe, state, updated_latents, attention, original_processor, target_spans
        )
        label = str(relative_step_size).replace(".", "p")
        guided_path = OUTPUT_DIR / f"flux-1-schnell_t750-single-step-guided_rel-{label}_seed-42_steps-4.png"
        guided_image.save(guided_path)
        guided_centroids = helpers.color_centroids(guided_image)
        comparison_images.append((f"t=750 guided relative={relative_step_size}", guided_image))
        guided_results.append(
            {
                "relative_step_size": relative_step_size,
                "effective_eta": float(effective_eta.item()),
                "desired_update_norm": float(desired_update_norm.item()),
                "actual_bf16_update_norm": float(actual_update_norm.item()),
                "actual_relative_update_norm": float((actual_update_norm / latent_norm).item()),
                "initial_layout_b": initial_layout,
                "updated_layout_b": updated_layout,
                "objective_delta_updated_minus_initial": updated_layout["total_energy"] - initial_layout["total_energy"],
                "mass_ratio_delta": {
                    word: updated_layout["objects"][word]["mass_ratio"] - initial_layout["objects"][word]["mass_ratio"]
                    for word in helpers.TARGET_WORDS
                },
                "image_path": str(guided_path),
                "pixel_sha256": helpers.pixel_sha256(guided_image),
                "pixel_difference_vs_branch_baseline": helpers.image_difference(baseline_image, guided_image),
                "color_centroids": guided_centroids,
                "color_proxy_centroid_shift_x_pixels": centroid_shift_x(baseline_centroids, guided_centroids),
                "target_direction_by_color_proxy": {
                    "apple_right": centroid_shift_x(baseline_centroids, guided_centroids)["apple_red_proxy"] > 0,
                    "cup_left": centroid_shift_x(baseline_centroids, guided_centroids)["cup_blue_proxy"] < 0,
                },
                "branch_runtime_seconds": time.perf_counter() - branch_start,
            }
        )

    comparison_path = OUTPUT_DIR / "flux-1-schnell_t750-single-step-guidance_comparison.png"
    helpers.make_comparison(comparison_images, comparison_path)
    t500_metrics = json.loads(T500_METRICS_PATH.read_text())
    t500_by_step = {result["relative_step_size"]: result for result in t500_metrics["guided_results"]}
    direct_comparison = {}
    for current in guided_results:
        relative_step_size = current["relative_step_size"]
        previous = t500_by_step[relative_step_size]
        direct_comparison[str(relative_step_size)] = {
            "objective_delta": {
                "t750": current["objective_delta_updated_minus_initial"],
                "t500": previous["objective_delta_updated_minus_initial"],
            },
            "mean_absolute_pixel_difference": {
                "t750": current["pixel_difference_vs_branch_baseline"]["mean_absolute_pixel_difference"],
                "t500": previous["pixel_difference_vs_branch_baseline"]["mean_absolute_pixel_difference"],
                "t750_over_t500": current["pixel_difference_vs_branch_baseline"]["mean_absolute_pixel_difference"]
                / previous["pixel_difference_vs_branch_baseline"]["mean_absolute_pixel_difference"],
            },
            "rmse": {
                "t750": current["pixel_difference_vs_branch_baseline"]["rmse"],
                "t500": previous["pixel_difference_vs_branch_baseline"]["rmse"],
            },
        }

    reference_hash = helpers.pixel_sha256(helpers.Image.open(helpers.REFERENCE_BASELINE_PATH))
    record = {
        **common_record,
        "stopped_before_update": False,
        "initial_layout_b": initial_layout,
        "gradient": gradient_metrics,
        "baseline": {
            "image_path": str(baseline_path),
            "pixel_sha256": helpers.pixel_sha256(baseline_image),
            "matches_original_pipeline_baseline": helpers.pixel_sha256(baseline_image) == reference_hash,
            "runtime_seconds_from_t750": baseline_runtime,
            "color_centroids": baseline_centroids,
        },
        "guided_results": guided_results,
        "t750_vs_t500": direct_comparison,
        "comparison_path": str(comparison_path),
        "gradient_runtime_seconds": gradient_runtime,
        "total_runtime_seconds_including_load": time.perf_counter() - total_start,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }
    metrics_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    pipe.maybe_free_model_hooks()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
