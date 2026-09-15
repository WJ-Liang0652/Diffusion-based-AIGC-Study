"""Test iterative backward guidance at the single t=750 denoising step.

The experiment follows one five-update optimization trajectory and evaluates
checkpoints after 1, 3, and 5 fresh forward/backward/update iterations. All
other settings are inherited from the validated t=750 single-step experiment.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import torch
from diffusers import FluxPipeline


PROJECT_DIR = Path(__file__).resolve().parents[1]
HELPER_PATH = Path(__file__).with_name("07_flux_schnell_early_step_guidance.py")
OUTPUT_DIR = PROJECT_DIR / "outputs" / "inner_loop_guidance"

PROBE_DENOISING_INDEX = 1
EXPECTED_TIMESTEP = 750.0
RELATIVE_STEP = 0.005
INNER_LOOP_CHECKPOINTS = (1, 3, 5)
LAYOUT_B = {"apple": "right", "cup": "left"}


def load_validated_modules():
    spec = importlib.util.spec_from_file_location("flux_t750_guidance_helpers", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper implementation from {HELPER_PATH}")
    t750_helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t750_helpers)
    helpers = t750_helpers.load_validated_helpers()
    return t750_helpers, helpers


def centroid_shift_x(baseline_centroids: dict[str, object], guided_centroids: dict[str, object]) -> dict[str, float]:
    shifts = {}
    for name in ("apple_red_proxy", "cup_blue_proxy"):
        baseline = baseline_centroids[name]["centroid_xy_pixels"]
        guided = guided_centroids[name]["centroid_xy_pixels"]
        shifts[name] = float(guided[0] - baseline[0]) if baseline is not None and guided is not None else None
    return shifts


def optimize_one_iteration(
    helpers,
    pipe: FluxPipeline,
    state: dict[str, object],
    current_latents: torch.Tensor,
    initial_latents: torch.Tensor,
    attention,
    original_processor,
    target_spans: dict[str, dict[str, object]],
    iteration: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Run a fresh graph, take one normalized step, then release the graph."""
    probe = helpers.GraphLayoutAttentionProbe(original_processor, target_spans)
    attention.set_processor(probe)
    variable = current_latents.detach().clone().requires_grad_(True)
    try:
        noise_pred = helpers.transformer_forward(pipe, state, variable, state["timestep"])
        layout_before_update = helpers.serialize_layout(probe.maps)
        objective, _ = helpers.layout_tensors(probe.maps)
        del noise_pred
        gradient = torch.autograd.grad(objective, variable, create_graph=False, retain_graph=False)[0]
        gradient_norm = gradient.float().norm()
        if not torch.isfinite(gradient).all() or gradient_norm.item() == 0:
            raise RuntimeError(f"Iteration {iteration} produced an invalid gradient")

        current_norm = variable.detach().float().norm()
        eta = RELATIVE_STEP * current_norm / gradient_norm
        update = (eta * gradient.detach().float()).to(variable.dtype)
        updated_latents = (variable.detach() - update).detach()
        actual_update_norm = (updated_latents.float() - variable.detach().float()).norm()
        cumulative_delta_norm = (updated_latents.float() - initial_latents.float()).norm()
        initial_norm = initial_latents.float().norm()
        parameter_grads = [
            name for name, parameter in pipe.transformer.named_parameters() if parameter.grad is not None
        ]
        iteration_record = {
            "iteration": iteration,
            "total_objective_before_update": layout_before_update["total_energy"],
            "apple_right_mass_before_update": layout_before_update["objects"]["apple"]["mass_ratio"],
            "cup_left_mass_before_update": layout_before_update["objects"]["cup"]["mass_ratio"],
            "gradient_norm_l2": float(gradient_norm.item()),
            "gradient_is_finite": True,
            "model_parameter_grad_count": len(parameter_grads),
            "model_parameters_with_grad": parameter_grads,
            "effective_eta": float(eta.item()),
            "actual_bf16_update_norm": float(actual_update_norm.item()),
            "actual_step_relative_to_current_latent": float((actual_update_norm / current_norm).item()),
            "cumulative_delta_norm_from_initial_t750": float(cumulative_delta_norm.item()),
            "cumulative_relative_delta_from_initial_t750": float((cumulative_delta_norm / initial_norm).item()),
        }
    finally:
        probe.release()
        attention.set_processor(original_processor)

    del variable, gradient, objective, update, probe
    return updated_latents, iteration_record


def main() -> None:
    t750_helpers, helpers = load_validated_modules()
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
    initial_latents = state["latents"].detach().clone()
    attention = pipe.transformer.transformer_blocks[helpers.DOUBLE_BLOCK_INDEX].attn
    original_processor = attention.get_processor()

    spatial_signal = t750_helpers.check_spatial_signal(
        helpers, pipe, state, attention, original_processor, target_spans
    )
    if not spatial_signal["reliable_for_update"]:
        raise RuntimeError("t=750 spatial signal failed the predeclared reliability criterion")

    baseline_start = time.perf_counter()
    baseline_image = helpers.continue_baseline(pipe, state)
    baseline_runtime = time.perf_counter() - baseline_start
    baseline_path = OUTPUT_DIR / "flux-1-schnell_t750-inner-loop_baseline_seed-42_steps-4.png"
    baseline_image.save(baseline_path)
    baseline_centroids = helpers.color_centroids(baseline_image)

    optimization_start = time.perf_counter()
    current_latents = initial_latents
    iteration_records = []
    guided_results = []
    comparison_images = [("baseline", baseline_image)]

    for iteration in range(1, max(INNER_LOOP_CHECKPOINTS) + 1):
        current_latents, iteration_record = optimize_one_iteration(
            helpers,
            pipe,
            state,
            current_latents,
            initial_latents,
            attention,
            original_processor,
            target_spans,
            iteration,
        )
        iteration_records.append(iteration_record)
        if iteration not in INNER_LOOP_CHECKPOINTS:
            continue

        checkpoint_start = time.perf_counter()
        guided_image, final_layout = helpers.continue_guided(
            pipe,
            state,
            current_latents,
            attention,
            original_processor,
            target_spans,
        )
        guided_path = OUTPUT_DIR / (
            f"flux-1-schnell_t750-inner-loop-{iteration}_rel-0p005_seed-42_steps-4.png"
        )
        guided_image.save(guided_path)
        guided_centroids = helpers.color_centroids(guided_image)
        shifts = centroid_shift_x(baseline_centroids, guided_centroids)
        comparison_images.append((f"t=750 inner={iteration}", guided_image))
        guided_results.append(
            {
                "inner_loop_count": iteration,
                "iterations": [dict(item) for item in iteration_records],
                "layout_after_final_update": final_layout,
                "objective_delta_final_minus_initial": (
                    final_layout["total_energy"] - iteration_records[0]["total_objective_before_update"]
                ),
                "mass_ratio_delta_final_minus_initial": {
                    "apple": final_layout["objects"]["apple"]["mass_ratio"]
                    - iteration_records[0]["apple_right_mass_before_update"],
                    "cup": final_layout["objects"]["cup"]["mass_ratio"]
                    - iteration_records[0]["cup_left_mass_before_update"],
                },
                "cumulative_relative_delta_from_initial_t750": iteration_record[
                    "cumulative_relative_delta_from_initial_t750"
                ],
                "image_path": str(guided_path),
                "pixel_sha256": helpers.pixel_sha256(guided_image),
                "pixel_difference_vs_baseline": helpers.image_difference(baseline_image, guided_image),
                "color_centroids": guided_centroids,
                "color_proxy_centroid_shift_x_pixels": shifts,
                "target_direction_by_color_proxy": {
                    "apple_right": shifts["apple_red_proxy"] is not None and shifts["apple_red_proxy"] > 0,
                    "cup_left": shifts["cup_blue_proxy"] is not None and shifts["cup_blue_proxy"] < 0,
                },
                "checkpoint_denoise_and_decode_runtime_seconds": time.perf_counter() - checkpoint_start,
            }
        )

    comparison_path = OUTPUT_DIR / "flux-1-schnell_t750-inner-loop-guidance_comparison.png"
    helpers.make_comparison(comparison_images, comparison_path)
    reference_hash = helpers.pixel_sha256(helpers.Image.open(helpers.REFERENCE_BASELINE_PATH))
    record = {
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
        "attention_objective": "joint-normalized image-query -> target text-key-span probability",
        "relative_step": RELATIVE_STEP,
        "inner_loop_checkpoints": list(INNER_LOOP_CHECKPOINTS),
        "trajectory_design": "one shared five-update trajectory, evaluated after updates 1, 3, and 5",
        "update_definition": {
            "formula": "x_new = x - eta*g",
            "gradient": "g = d(E_layout_b)/dx from a fresh forward at the current latent",
            "eta": "relative_step * ||current_x||_2 / ||g||_2",
            "graph_lifetime": "detach after every update; fresh requires_grad leaf and graph each iteration",
        },
        "spatial_signal_check": spatial_signal,
        "baseline": {
            "image_path": str(baseline_path),
            "pixel_sha256": helpers.pixel_sha256(baseline_image),
            "matches_original_pipeline_baseline": helpers.pixel_sha256(baseline_image) == reference_hash,
            "runtime_seconds_from_t750": baseline_runtime,
            "color_centroids": baseline_centroids,
        },
        "iteration_records": iteration_records,
        "guided_results": guided_results,
        "comparison_path": str(comparison_path),
        "optimization_and_checkpoint_generation_runtime_seconds": time.perf_counter() - optimization_start,
        "total_runtime_seconds_including_load": time.perf_counter() - total_start,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }
    metrics_path = OUTPUT_DIR / "flux-1-schnell_t750-inner-loop-guidance_metrics.json"
    metrics_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))

    pipe.maybe_free_model_hooks()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
