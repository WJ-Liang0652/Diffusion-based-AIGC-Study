"""Stage 6A: single-timestep inner-loop Backward Guidance on FLUX.1-dev.

At denoising index 35, a shared five-update trajectory repeatedly recomputes
Double-block-18 Layout-B attention, objective, and latent gradient. Checkpoints
K=1/3/5 then complete ordinary denoising. No other timestep is guided.
"""

from __future__ import annotations

import importlib.util
import json
import math
import time
from pathlib import Path

import diffusers
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import FluxPipeline
from PIL import Image, ImageDraw


PROJECT_DIR = Path(__file__).resolve().parents[1]
STAGE5_PATH = Path(__file__).with_name("15_flux_dev_single_step_guidance.py")
OUTPUT_DIR = PROJECT_DIR / "outputs" / "inner_loop_guidance"
STAGE5_BASELINE_PATH = (
    PROJECT_DIR / "outputs" / "single_step_guidance" / "flux_dev_single_step_baseline_seed-42_steps-50.png"
)
STAGE5_K1_IMAGE_PATH = (
    PROJECT_DIR
    / "outputs"
    / "single_step_guidance"
    / "flux_dev_single_step_guided_rel-0p005_seed-42_steps-50.png"
)
STAGE5_K1_METRICS_PATH = (
    PROJECT_DIR
    / "outputs"
    / "single_step_guidance"
    / "flux_dev_single_step_guidance_rel-0p005_metrics.json"
)

PROBE_DENOISING_INDEX = 35
DOUBLE_BLOCK_INDEX = 18
RELATIVE_STEP_SIZE = 0.005
INNER_LOOP_CHECKPOINTS = (1, 3, 5)


def load_stage5():
    spec = importlib.util.spec_from_file_location("flux_dev_single_step_guidance", STAGE5_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Stage-5 helpers from {STAGE5_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def map_snapshot(stage5, attention_map: torch.Tensor) -> dict[str, object]:
    return stage5.map_statistics(attention_map.detach().cpu())


def capture_layout_and_noise(
    pipe,
    stage3,
    stage4,
    stage5,
    state,
    latents,
    attention,
    original_processor,
    target_spans,
):
    probe = stage4.GraphLayoutAttentionProbe(original_processor, target_spans)
    attention.set_processor(probe)
    try:
        with torch.no_grad():
            noise_pred = stage5.transformer_forward(pipe, state, latents, state["timestep"])
        layout = stage4.serializable_layout(stage3, probe.maps, "layout_b")
        maps = {word: value.detach().cpu().clone() for word, value in probe.maps.items()}
    finally:
        probe.release()
        attention.set_processor(original_processor)
    return noise_pred, layout, maps


def optimize_one_iteration(
    pipe,
    stage3,
    stage4,
    stage5,
    state,
    current_latents,
    initial_latents,
    initial_latent_norm,
    attention,
    original_processor,
    target_spans,
    iteration: int,
):
    """Use one fresh graph and gradient, then independently evaluate the update."""
    probe = stage4.GraphLayoutAttentionProbe(original_processor, target_spans)
    attention.set_processor(probe)
    variable = current_latents.detach().clone().requires_grad_(True)
    try:
        noise_pred = stage5.transformer_forward(pipe, state, variable, state["timestep"])
        before_layout = stage4.serializable_layout(stage3, probe.maps, "layout_b")
        before_maps = {word: value.detach().cpu().clone() for word, value in probe.maps.items()}
        objective, _ = stage4.layout_tensors(stage3, probe.maps, "layout_b")
        del noise_pred
        gradient = torch.autograd.grad(objective, variable, create_graph=False, retain_graph=False)[0]
        gradient_float = gradient.float()
        gradient_norm = gradient_float.norm()
        if not torch.isfinite(gradient).all() or gradient_norm.item() == 0:
            raise RuntimeError(f"Iteration {iteration} produced an invalid gradient")
        parameter_grads = [
            name for name, parameter in pipe.transformer.named_parameters() if parameter.grad is not None
        ]
        if parameter_grads:
            raise RuntimeError(f"Iteration {iteration} accumulated model gradients: {parameter_grads[:3]}")

        current_norm = variable.detach().float().norm()
        effective_eta = RELATIVE_STEP_SIZE * current_norm / gradient_norm
        update = (effective_eta * gradient_float).to(variable.dtype)
        updated_latents = (variable.detach() - update).detach()
        actual_update_norm = (updated_latents.float() - variable.detach().float()).norm()
    finally:
        probe.release()
        attention.set_processor(original_processor)

    post_noise_pred, after_layout, after_maps = capture_layout_and_noise(
        pipe,
        stage3,
        stage4,
        stage5,
        state,
        updated_latents,
        attention,
        original_processor,
        target_spans,
    )
    object_records = {}
    for word in target_spans:
        before_stats = map_snapshot(stage5, before_maps[word])
        after_stats = map_snapshot(stage5, after_maps[word])
        target_region = before_layout["objects"][word]["region"]
        center_delta_x = after_stats["center_of_mass_xy"][0] - before_stats["center_of_mass_xy"][0]
        object_records[word] = {
            "target_region": target_region,
            "target_mass_before": before_layout["objects"][word]["mass_ratio"],
            "target_mass_after": after_layout["objects"][word]["mass_ratio"],
            "target_mass_delta": (
                after_layout["objects"][word]["mass_ratio"]
                - before_layout["objects"][word]["mass_ratio"]
            ),
            "attention_centroid_before_xy": before_stats["center_of_mass_xy"],
            "attention_centroid_after_xy": after_stats["center_of_mass_xy"],
            "attention_centroid_delta_xy": [
                center_delta_x,
                after_stats["center_of_mass_xy"][1] - before_stats["center_of_mass_xy"][1],
            ],
            "moves_toward_target": center_delta_x > 0 if target_region == "right" else center_delta_x < 0,
        }
    cumulative_norm = (updated_latents.float() - initial_latents.float()).norm()
    record = {
        "iteration": iteration,
        "objective_before": before_layout["total_energy"],
        "objective_after": after_layout["total_energy"],
        "objective_delta": after_layout["total_energy"] - before_layout["total_energy"],
        "objective_decreased": after_layout["total_energy"] < before_layout["total_energy"],
        "gradient_norm_l2": float(gradient_norm.item()),
        "gradient_abs_max": float(gradient_float.abs().max().item()),
        "gradient_abs_mean": float(gradient_float.abs().mean().item()),
        "gradient_is_finite": True,
        "model_parameter_grad_count": 0,
        "effective_eta": float(effective_eta.item()),
        "actual_bf16_update_norm": float(actual_update_norm.item()),
        "actual_relative_update_norm": float((actual_update_norm / current_norm).item()),
        "cumulative_delta_norm_from_initial": float(cumulative_norm.item()),
        "cumulative_relative_delta_from_initial": float((cumulative_norm / initial_latent_norm).item()),
        "objects": object_records,
    }
    del variable, gradient, gradient_float, objective, update
    torch.cuda.empty_cache()
    return updated_latents, post_noise_pred, after_layout, after_maps, record


@torch.no_grad()
def continue_from_checkpoint(pipe, helpers, stage5, state, checkpoint_latents, first_noise_pred):
    start = time.perf_counter()
    latents = checkpoint_latents.clone()
    timesteps = stage5.set_schedule(pipe, helpers, latents.shape[1], latents.device)
    timestep_value = timesteps[PROBE_DENOISING_INDEX]
    latents = pipe.scheduler.step(first_noise_pred, timestep_value, latents, return_dict=False)[0]
    for timestep_value in timesteps[PROBE_DENOISING_INDEX + 1 :]:
        noise_pred = stage5.transformer_forward(pipe, state, latents, timestep_value)
        latents = pipe.scheduler.step(noise_pred, timestep_value, latents, return_dict=False)[0]
    image = stage5.decode(pipe, helpers, latents)
    return latents.detach(), image, time.perf_counter() - start


def save_image_comparison(baseline: Image.Image, guided_images: dict[int, Image.Image], path: Path) -> None:
    panels = [("baseline", baseline.convert("RGB"))] + [
        (f"inner K={count}, rel={RELATIVE_STEP_SIZE:g}", guided_images[count].convert("RGB"))
        for count in INNER_LOOP_CHECKPOINTS
    ]
    width, height = baseline.size
    header = 38
    canvas = Image.new("RGB", (width * len(panels), height + header), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        canvas.paste(image, (index * width, header))
        draw.text((index * width + 12, 11), label, fill="black")
    canvas.save(path)


def save_attention_trajectory(initial_maps, checkpoint_maps, timestep: float, path: Path) -> None:
    words = tuple(initial_maps)
    columns = ((0, initial_maps),) + tuple((count, checkpoint_maps[count]) for count in INNER_LOOP_CHECKPOINTS)
    figure, axes = plt.subplots(len(words), len(columns), figsize=(16, 8), squeeze=False)
    for row, word in enumerate(words):
        all_maps = [maps[word] for _, maps in columns]
        vmin = min(float(value.min()) for value in all_maps)
        vmax = max(float(value.max()) for value in all_maps)
        for column, (count, maps) in enumerate(columns):
            axis = axes[row, column]
            rendered = axis.imshow(maps[word].float().numpy(), cmap="magma", vmin=vmin, vmax=vmax)
            axis.axvline(23.5, color="cyan", linewidth=1.0)
            axis.set_title(f"{word}: initial" if count == 0 else f"{word}: K={count}")
            axis.set_xlabel("image-token x")
            axis.set_ylabel("image-token y")
            figure.colorbar(rendered, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(f"Block 18 Layout-B attention trajectory, denoise=35, t={timestep:.3f}")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> None:
    stage5 = load_stage5()
    stage4 = stage5.load_stage4()
    stage3 = stage4.load_stage3()
    helpers = stage3.load_helpers()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for Stage 6A")
    for required in (STAGE5_BASELINE_PATH, STAGE5_K1_IMAGE_PATH, STAGE5_K1_METRICS_PATH):
        if not required.exists():
            raise FileNotFoundError(f"Required Stage-5 artifact missing: {required}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()
    baseline_state, baseline_metrics = helpers.load_baseline_state()
    stage5_k1_metrics = json.loads(STAGE5_K1_METRICS_PATH.read_text())
    baseline_image = Image.open(STAGE5_BASELINE_PATH).convert("RGB")
    if stage5.pixel_sha256(baseline_image) != baseline_metrics["pixel_sha256"]:
        raise RuntimeError("Stage-5 baseline image does not match Stage 1")

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
    initial_latents = state["latents"].detach().clone()
    initial_latent_norm = initial_latents.float().norm()
    attention = pipe.transformer.transformer_blocks[DOUBLE_BLOCK_INDEX].attn
    original_processor = attention.get_processor()

    current_latents = initial_latents
    iteration_records = []
    guided_results = []
    guided_images = {}
    checkpoint_maps = {}
    initial_maps = None
    initial_layout = None
    optimization_start = time.perf_counter()
    for iteration in range(1, max(INNER_LOOP_CHECKPOINTS) + 1):
        current_latents, post_noise_pred, after_layout, after_maps, iteration_record = (
            optimize_one_iteration(
                pipe,
                stage3,
                stage4,
                stage5,
                state,
                current_latents,
                initial_latents,
                initial_latent_norm,
                attention,
                original_processor,
                target_spans,
                iteration,
            )
        )
        iteration_records.append(iteration_record)
        if iteration == 1:
            _, initial_layout, initial_maps = capture_layout_and_noise(
                pipe,
                stage3,
                stage4,
                stage5,
                state,
                initial_latents,
                attention,
                original_processor,
                target_spans,
            )
        if iteration not in INNER_LOOP_CHECKPOINTS:
            del post_noise_pred
            continue

        checkpoint_maps[iteration] = after_maps
        final_latents, guided_image, continuation_runtime = continue_from_checkpoint(
            pipe, helpers, stage5, state, current_latents, post_noise_pred
        )
        guided_images[iteration] = guided_image
        guided_path = OUTPUT_DIR / f"flux_dev_inner_loop_k-{iteration}_rel-0p005_seed-42_steps-50.png"
        guided_image.save(guided_path)
        baseline_centroids = stage5.color_centroids(baseline_image, helpers.WIDTH, helpers.HEIGHT)
        guided_centroids = stage5.color_centroids(guided_image, helpers.WIDTH, helpers.HEIGHT)
        cumulative_attention = stage5.attention_change(
            initial_maps, after_maps, initial_layout, after_layout
        )
        guided_results.append(
            {
                "inner_loop_count": iteration,
                "layout_after_final_update": after_layout,
                "objective_delta_final_minus_initial": (
                    after_layout["total_energy"] - initial_layout["total_energy"]
                ),
                "cumulative_attention_movement": cumulative_attention,
                "cumulative_relative_latent_delta": iteration_record[
                    "cumulative_relative_delta_from_initial"
                ],
                "image_path": str(guided_path),
                "pixel_sha256": stage5.pixel_sha256(guided_image),
                "pixel_difference_vs_baseline": stage5.image_difference(baseline_image, guided_image),
                "color_centroid_changes_vs_baseline": stage5.centroid_changes(
                    baseline_centroids, guided_centroids
                ),
                "final_packed_latent_l2_difference_vs_baseline": float(
                    (final_latents.float() - baseline_state["pipeline_final_latents"].to(final_latents.device).float())
                    .norm()
                    .item()
                ),
                "continuation_runtime_seconds": continuation_runtime,
            }
        )
        del post_noise_pred, final_latents
        torch.cuda.empty_cache()

    comparison_path = OUTPUT_DIR / "flux_dev_inner_loop_k-1-3-5_rel-0p005_comparison.png"
    attention_path = OUTPUT_DIR / "flux_dev_inner_loop_k-1-3-5_rel-0p005_attention.png"
    save_image_comparison(baseline_image, guided_images, comparison_path)
    save_attention_trajectory(
        initial_maps, checkpoint_maps, float(state["timestep"].item()), attention_path
    )

    post_objectives = [record["objective_after"] for record in iteration_records]
    objective_stable_decrease = bool(
        all(record["objective_decreased"] for record in iteration_records)
        and all(later < earlier for earlier, later in zip(post_objectives, post_objectives[1:]))
    )
    apple_centers = [
        record["objects"]["apple"]["attention_centroid_after_xy"][0] for record in iteration_records
    ]
    cup_centers = [
        record["objects"]["cup"]["attention_centroid_after_xy"][0] for record in iteration_records
    ]
    attention_direction_consistent = bool(
        all(later > earlier for earlier, later in zip(apple_centers, apple_centers[1:]))
        and all(later < earlier for earlier, later in zip(cup_centers, cup_centers[1:]))
    )
    k1_result = next(item for item in guided_results if item["inner_loop_count"] == 1)
    stage5_k1_image = Image.open(STAGE5_K1_IMAGE_PATH).convert("RGB")
    k1_stage5_comparison = {
        "objective_absolute_difference": abs(
            k1_result["layout_after_final_update"]["total_energy"]
            - stage5_k1_metrics["layout_objective"]["after"]["total_energy"]
        ),
        "pixel_hash_exact_match": k1_result["pixel_sha256"] == stage5.pixel_sha256(stage5_k1_image),
        "pixel_difference": stage5.image_difference(stage5_k1_image, guided_images[1]),
    }
    record = {
        "stage": "Stage 6A - dev single-timestep inner-loop Backward Guidance",
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
        "relative_step_size_per_update": RELATIVE_STEP_SIZE,
        "inner_loop_checkpoints": list(INNER_LOOP_CHECKPOINTS),
        "guided_timestep_count": 1,
        "trajectory_design": "one shared five-update trajectory evaluated at K=1/3/5",
        "update_definition": {
            "formula": "x_new = x - eta*g",
            "gradient": "fresh g = d(E_layout_b)/dx at the current latent for every iteration",
            "eta": "relative_step * ||current_x||_2 / ||g||_2",
            "graph_lifetime": "fresh leaf/graph per iteration; detach after every update",
        },
        "baseline": {
            "image_path": str(STAGE5_BASELINE_PATH),
            "pixel_sha256": stage5.pixel_sha256(baseline_image),
            "matches_stage1_image_exactly": True,
        },
        "initial_layout_b": initial_layout,
        "iteration_records": iteration_records,
        "guided_results": guided_results,
        "objective_stable_decrease": objective_stable_decrease,
        "attention_direction_consistent_across_iterations": attention_direction_consistent,
        "k1_comparison_with_stage5_rel_0p005": k1_stage5_comparison,
        "manual_visual_assessment": "pending",
        "comparison_path": str(comparison_path),
        "attention_trajectory_path": str(attention_path),
        "optimization_and_checkpoint_runtime_seconds": time.perf_counter() - optimization_start,
        "total_runtime_seconds_including_load": time.perf_counter() - total_start,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }
    metrics_path = OUTPUT_DIR / "flux_dev_inner_loop_k-1-3-5_rel-0p005_metrics.json"
    metrics_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    if not (objective_stable_decrease and attention_direction_consistent):
        raise RuntimeError("Stage 6A mechanical validation failed")

    pipe.maybe_free_model_hooks()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
