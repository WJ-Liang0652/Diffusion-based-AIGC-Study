"""Stage 6B: sparse multi-timestep Backward Guidance on FLUX.1-dev.

Replay the exact Stage-1 trajectory from its saved initial packed latent and
apply one fresh Double-block-18 Layout-B gradient update at each selected
denoising index. New earlier interventions must pass a spatial-signal gate
before their update is accepted. This is a sparse MVP, not a timestep sweep.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import diffusers
import numpy as np
import torch
from diffusers import FluxPipeline
from PIL import Image, ImageDraw


PROJECT_DIR = Path(__file__).resolve().parents[1]
STAGE5_PATH = Path(__file__).with_name("15_flux_dev_single_step_guidance.py")
OUTPUT_DIR = PROJECT_DIR / "outputs" / "multi_timestep_guidance"
BASELINE_IMAGE_PATH = PROJECT_DIR / "outputs" / "baseline" / "flux_dev_baseline_seed-42_steps-50.png"
STAGE6A_IMAGE_PATH = (
    PROJECT_DIR / "outputs" / "inner_loop_guidance" / "flux_dev_inner_loop_k-5_rel-0p005_seed-42_steps-50.png"
)
STAGE6A_METRICS_PATH = (
    PROJECT_DIR / "outputs" / "inner_loop_guidance" / "flux_dev_inner_loop_k-1-3-5_rel-0p005_metrics.json"
)

DOUBLE_BLOCK_INDEX = 18
RELATIVE_STEP_SIZE = 0.005
DEFAULT_SCHEDULE = (20, 28, 35)
VALIDATED_INDEX = 35
OBSERVED_GRID_BOXES = {
    "apple": (6, 23, 25, 44),
    "cup": (25, 20, 48, 44),
}
MIN_OBSERVED_BOX_ENRICHMENT = 1.10
MIN_CENTROID_SEPARATION = 1.0


def load_stage5():
    spec = importlib.util.spec_from_file_location("flux_dev_single_step_guidance", STAGE5_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Stage-5 helpers from {STAGE5_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_schedule(value: str) -> tuple[int, ...]:
    indices = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(indices) != 3 or tuple(sorted(set(indices))) != indices:
        raise argparse.ArgumentTypeError("schedule must contain three unique ascending indices")
    if VALIDATED_INDEX not in indices:
        raise argparse.ArgumentTypeError(f"schedule must include validated index {VALIDATED_INDEX}")
    if indices[0] < 0 or indices[-1] >= 50:
        raise argparse.ArgumentTypeError("schedule indices must be in [0, 49]")
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=parse_schedule, default=DEFAULT_SCHEDULE)
    parser.add_argument("--skip-baseline-replay", action="store_true")
    return parser.parse_args()


def schedule_label(indices: tuple[int, ...]) -> str:
    return "-".join(str(value) for value in indices)


def serializable_map_stats(stage5, attention_map: torch.Tensor) -> dict[str, object]:
    return stage5.map_statistics(attention_map.detach().cpu())


def spatial_signal_metrics(stage5, maps: dict[str, torch.Tensor]) -> dict[str, object]:
    objects = {}
    all_finite = True
    all_positive = True
    all_enriched = True
    for word, attention_map in maps.items():
        detached = attention_map.detach().float()
        finite = bool(torch.isfinite(detached).all().item())
        total = float(detached.sum().item())
        x0, y0, x1, y1 = OBSERVED_GRID_BOXES[word]
        area_fraction = ((x1 - x0) * (y1 - y0)) / detached.numel()
        mass_ratio = float((detached[y0:y1, x0:x1].sum() / detached.sum()).item()) if total > 0 else math.nan
        enrichment = mass_ratio / area_fraction if total > 0 else math.nan
        stats = serializable_map_stats(stage5, detached.cpu()) if finite and total > 0 else None
        objects[word] = {
            "all_finite": finite,
            "total_attention_mass": total,
            "observed_box_xyxy": [x0, y0, x1, y1],
            "observed_box_mass_ratio": mass_ratio,
            "observed_box_enrichment_over_uniform": enrichment,
            "attention_centroid_xy": stats["center_of_mass_xy"] if stats else None,
        }
        all_finite = all_finite and finite
        all_positive = all_positive and total > 0
        all_enriched = all_enriched and math.isfinite(enrichment) and enrichment >= MIN_OBSERVED_BOX_ENRICHMENT
    separation = (
        objects["cup"]["attention_centroid_xy"][0] - objects["apple"]["attention_centroid_xy"][0]
        if all_finite and all_positive
        else math.nan
    )
    meaningful = bool(
        all_finite
        and all_positive
        and all_enriched
        and math.isfinite(separation)
        and separation >= MIN_CENTROID_SEPARATION
    )
    return {
        "objects": objects,
        "apple_cup_centroid_x_separation": separation,
        "thresholds": {
            "minimum_observed_box_enrichment": MIN_OBSERVED_BOX_ENRICHMENT,
            "minimum_centroid_x_separation": MIN_CENTROID_SEPARATION,
        },
        "spatially_meaningful": meaningful,
    }


def capture_layout_and_noise(
    pipe, stage3, stage4, stage5, state, latents, timestep, attention, original_processor, target_spans
):
    probe = stage4.GraphLayoutAttentionProbe(original_processor, target_spans)
    attention.set_processor(probe)
    try:
        with torch.no_grad():
            noise_pred = stage5.transformer_forward(pipe, state, latents, timestep)
        layout = stage4.serializable_layout(stage3, probe.maps, "layout_b")
        maps = {word: value.detach().cpu().clone() for word, value in probe.maps.items()}
    finally:
        probe.release()
        attention.set_processor(original_processor)
    return noise_pred, layout, maps


def intervene_once(
    pipe,
    stage3,
    stage4,
    stage5,
    state,
    latents,
    timestep,
    denoising_index,
    attention,
    original_processor,
    target_spans,
):
    probe = stage4.GraphLayoutAttentionProbe(original_processor, target_spans)
    attention.set_processor(probe)
    variable = latents.detach().clone().requires_grad_(True)
    try:
        noise_pred = stage5.transformer_forward(pipe, state, variable, timestep)
        before_layout = stage4.serializable_layout(stage3, probe.maps, "layout_b")
        before_maps = {word: value.detach().cpu().clone() for word, value in probe.maps.items()}
        spatial_signal = spatial_signal_metrics(stage5, before_maps)
        if denoising_index < VALIDATED_INDEX and not spatial_signal["spatially_meaningful"]:
            raise RuntimeError(
                f"Index {denoising_index} failed the pre-update spatial-signal gate: {spatial_signal}"
            )
        objective, _ = stage4.layout_tensors(stage3, probe.maps, "layout_b")
        del noise_pred
        gradient = torch.autograd.grad(objective, variable, create_graph=False, retain_graph=False)[0]
        gradient_float = gradient.float()
        gradient_norm = gradient_float.norm()
        if not torch.isfinite(objective).item() or not torch.isfinite(gradient).all().item():
            raise RuntimeError(f"Index {denoising_index} produced NaN/Inf objective or gradient")
        if gradient_norm.item() == 0:
            raise RuntimeError(f"Index {denoising_index} produced a zero gradient")
        parameter_grads = [
            name for name, parameter in pipe.transformer.named_parameters() if parameter.grad is not None
        ]
        if parameter_grads:
            raise RuntimeError(
                f"Index {denoising_index} accumulated model gradients: {parameter_grads[:3]}"
            )
        latent_norm = variable.detach().float().norm()
        effective_eta = RELATIVE_STEP_SIZE * latent_norm / gradient_norm
        update = (effective_eta * gradient_float).to(variable.dtype)
        updated_latents = (variable.detach() - update).detach()
        actual_update_norm = (updated_latents.float() - variable.detach().float()).norm()
    finally:
        probe.release()
        attention.set_processor(original_processor)

    post_noise, after_layout, after_maps = capture_layout_and_noise(
        pipe,
        stage3,
        stage4,
        stage5,
        state,
        updated_latents,
        timestep,
        attention,
        original_processor,
        target_spans,
    )
    movement = stage5.attention_change(before_maps, after_maps, before_layout, after_layout)
    objective_decreased = after_layout["total_energy"] < before_layout["total_energy"]
    if not math.isfinite(after_layout["total_energy"]) or not objective_decreased:
        raise RuntimeError(
            f"Index {denoising_index} update was unstable/non-descending: "
            f"{before_layout['total_energy']} -> {after_layout['total_energy']}"
        )
    record = {
        "denoising_index": denoising_index,
        "timestep": float(timestep.item()),
        "k_at_timestep": 1,
        "pre_update_spatial_signal": spatial_signal,
        "objective_before": before_layout["total_energy"],
        "objective_after": after_layout["total_energy"],
        "objective_delta": after_layout["total_energy"] - before_layout["total_energy"],
        "objective_decreased": objective_decreased,
        "gradient_norm_l2": float(gradient_norm.item()),
        "gradient_abs_max": float(gradient_float.abs().max().item()),
        "gradient_abs_mean": float(gradient_float.abs().mean().item()),
        "gradient_is_finite": True,
        "model_parameter_grad_count": 0,
        "effective_eta": float(effective_eta.item()),
        "actual_bf16_update_norm": float(actual_update_norm.item()),
        "actual_relative_update_norm": float((actual_update_norm / latent_norm).item()),
        "attention_movement": movement,
    }
    del variable, gradient, gradient_float, objective, update
    torch.cuda.empty_cache()
    return updated_latents, post_noise, record


@torch.no_grad()
def run_baseline(pipe, helpers, stage5, state, baseline_state):
    start = time.perf_counter()
    latents = baseline_state["initial_latents"].to(
        device=state["prompt_embeds"].device, dtype=state["prompt_embeds"].dtype
    )
    timesteps = stage5.set_schedule(pipe, helpers, latents.shape[1], latents.device)
    for timestep in timesteps:
        noise_pred = stage5.transformer_forward(pipe, state, latents, timestep)
        latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
    image = stage5.decode(pipe, helpers, latents)
    return latents.detach(), image, time.perf_counter() - start


def run_guided(
    pipe, helpers, stage3, stage4, stage5, state, baseline_state, schedule, attention,
    original_processor, target_spans
):
    start = time.perf_counter()
    latents = baseline_state["initial_latents"].to(
        device=state["prompt_embeds"].device, dtype=state["prompt_embeds"].dtype
    )
    timesteps = stage5.set_schedule(pipe, helpers, latents.shape[1], latents.device)
    intervention_records = []
    for index, timestep in enumerate(timesteps):
        if index in schedule:
            latents, noise_pred, record = intervene_once(
                pipe,
                stage3,
                stage4,
                stage5,
                state,
                latents,
                timestep,
                index,
                attention,
                original_processor,
                target_spans,
            )
            intervention_records.append(record)
        else:
            with torch.no_grad():
                noise_pred = stage5.transformer_forward(pipe, state, latents, timestep)
        with torch.no_grad():
            latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
        del noise_pred
    image = stage5.decode(pipe, helpers, latents)
    return latents.detach(), image, intervention_records, time.perf_counter() - start


def save_comparison(baseline: Image.Image, stage6a: Image.Image, guided: Image.Image, schedule, path: Path):
    panels = [
        ("exact Stage-1 baseline", baseline.convert("RGB")),
        ("Stage 6A index 35, K=5", stage6a.convert("RGB")),
        (f"Stage 6B indices {schedule_label(schedule)}, K=1 each", guided.convert("RGB")),
    ]
    width, height = baseline.size
    header = 38
    canvas = Image.new("RGB", (width * len(panels), height + header), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (label, image) in enumerate(panels):
        canvas.paste(image, (column * width, header))
        draw.text((column * width + 12, 11), label, fill="black")
    canvas.save(path)


def main() -> None:
    args = parse_args()
    schedule = tuple(args.schedule)
    stage5 = load_stage5()
    stage4 = stage5.load_stage4()
    stage3 = stage4.load_stage3()
    helpers = stage3.load_helpers()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for Stage 6B")
    for required in (BASELINE_IMAGE_PATH, STAGE6A_IMAGE_PATH, STAGE6A_METRICS_PATH):
        if not required.exists():
            raise FileNotFoundError(f"Required comparison artifact missing: {required}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    label = schedule_label(schedule)
    guided_path = OUTPUT_DIR / f"flux_dev_multi_timestep_indices-{label}_rel-0p005.png"
    comparison_path = OUTPUT_DIR / f"flux_dev_multi_timestep_indices-{label}_rel-0p005_comparison.png"
    metrics_path = OUTPUT_DIR / f"flux_dev_multi_timestep_indices-{label}_rel-0p005_metrics.json"
    total_start = time.perf_counter()
    baseline_state, baseline_metrics = helpers.load_baseline_state()
    baseline_image = Image.open(BASELINE_IMAGE_PATH).convert("RGB")
    stage6a_image = Image.open(STAGE6A_IMAGE_PATH).convert("RGB")
    stage6a_metrics = json.loads(STAGE6A_METRICS_PATH.read_text())
    if stage5.pixel_sha256(baseline_image) != baseline_metrics["pixel_sha256"]:
        raise RuntimeError("Saved comparison baseline does not match Stage 1")

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
    attention = pipe.transformer.transformer_blocks[DOUBLE_BLOCK_INDEX].attn
    original_processor = attention.get_processor()

    baseline_replay = None
    if not args.skip_baseline_replay:
        baseline_latents, replay_image, replay_runtime = run_baseline(
            pipe, helpers, stage5, state, baseline_state
        )
        latent_exact = torch.equal(baseline_latents.cpu(), baseline_state["pipeline_final_latents"])
        image_exact = stage5.pixel_sha256(replay_image) == baseline_metrics["pixel_sha256"]
        baseline_replay = {
            "performed": True,
            "final_latent_exact_match": latent_exact,
            "image_exact_match": image_exact,
            "runtime_seconds": replay_runtime,
        }
        if not (latent_exact and image_exact):
            raise RuntimeError(f"Full explicit baseline replay mismatch: {baseline_replay}")
        del baseline_latents, replay_image
        torch.cuda.empty_cache()
    else:
        baseline_replay = {
            "performed": False,
            "comparison_uses_verified_stage1_artifacts": True,
        }

    guided_latents, guided_image, interventions, guided_runtime = run_guided(
        pipe,
        helpers,
        stage3,
        stage4,
        stage5,
        state,
        baseline_state,
        schedule,
        attention,
        original_processor,
        target_spans,
    )
    guided_image.save(guided_path)
    save_comparison(baseline_image, stage6a_image, guided_image, schedule, comparison_path)
    baseline_centroids = stage5.color_centroids(baseline_image, helpers.WIDTH, helpers.HEIGHT)
    guided_centroids = stage5.color_centroids(guided_image, helpers.WIDTH, helpers.HEIGHT)
    stage6a_centroids = stage5.color_centroids(stage6a_image, helpers.WIDTH, helpers.HEIGHT)
    stage6a_k5 = next(
        item for item in stage6a_metrics["guided_results"] if item["inner_loop_count"] == 5
    )
    record = {
        "stage": "Stage 6B - dev sparse multi-timestep Backward Guidance MVP",
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
        "layout_optimized": stage3.LAYOUTS["layout_b"],
        "schedule_indices": list(schedule),
        "k_per_intervention": 1,
        "relative_step_size_per_intervention": RELATIVE_STEP_SIZE,
        "update_definition": "x_new = x - relative_step * ||x||_2/||g||_2 * g",
        "baseline": {
            "image_path": str(BASELINE_IMAGE_PATH),
            "pixel_sha256": stage5.pixel_sha256(baseline_image),
            "matches_stage1_image_exactly": True,
            "replay": baseline_replay,
        },
        "interventions": interventions,
        "all_interventions_stable": all(
            item["objective_decreased"]
            and item["gradient_is_finite"]
            and item["model_parameter_grad_count"] == 0
            for item in interventions
        ),
        "guided": {
            "image_path": str(guided_path),
            "pixel_sha256": stage5.pixel_sha256(guided_image),
            "pixel_difference_vs_baseline": stage5.image_difference(baseline_image, guided_image),
            "color_centroid_changes_vs_baseline": stage5.centroid_changes(
                baseline_centroids, guided_centroids
            ),
            "final_packed_latent_l2_difference_vs_baseline": float(
                (
                    guided_latents.float()
                    - baseline_state["pipeline_final_latents"].to(guided_latents.device).float()
                )
                .norm()
                .item()
            ),
        },
        "comparison_with_stage6a_k5": {
            "stage6a_image_path": str(STAGE6A_IMAGE_PATH),
            "stage6a_objective_at_index35_after_k5": stage6a_k5[
                "layout_after_final_update"
            ]["total_energy"],
            "pixel_difference_multi_vs_stage6a_k5": stage5.image_difference(
                stage6a_image, guided_image
            ),
            "stage6a_color_centroid_changes_vs_baseline": stage5.centroid_changes(
                baseline_centroids, stage6a_centroids
            ),
            "multi_color_centroid_changes_vs_baseline": stage5.centroid_changes(
                baseline_centroids, guided_centroids
            ),
        },
        "manual_visual_assessment": "pending",
        "comparison_path": str(comparison_path),
        "guided_runtime_seconds": guided_runtime,
        "total_runtime_seconds_including_load": time.perf_counter() - total_start,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }
    metrics_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    if len(interventions) != len(schedule) or not record["all_interventions_stable"]:
        raise RuntimeError("Stage 6B mechanical validation failed")

    pipe.maybe_free_model_hooks()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
