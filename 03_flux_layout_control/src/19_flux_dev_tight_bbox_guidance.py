"""Stage 6E: tight bounding-box Backward Guidance on FLUX.1-dev.

Keep the Stage-6D 0.25x configuration fixed and replace only the broad
left/right half-plane target regions with finite object-sized boxes derived
from the fixed-seed baseline annotations.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import diffusers
import torch
from diffusers import FluxPipeline
from PIL import Image, ImageDraw


PROJECT_DIR = Path(__file__).resolve().parents[1]
STAGE6D_PATH = Path(__file__).with_name("18_flux_dev_paper_aligned_guidance.py")
OUTPUT_DIR = PROJECT_DIR / "outputs" / "tight_bbox_guidance"
BASELINE_IMAGE_PATH = PROJECT_DIR / "outputs" / "baseline" / "flux_dev_baseline_seed-42_steps-50.png"
STAGE6D_IMAGE_PATH = (
    PROJECT_DIR
    / "outputs"
    / "paper_aligned_guidance"
    / "flux_dev_paper_aligned_blocks-18_eta-scale-0p25_guided.png"
)
STAGE6D_METRICS_PATH = (
    PROJECT_DIR
    / "outputs"
    / "paper_aligned_guidance"
    / "flux_dev_paper_aligned_blocks-18_eta-scale-0p25_metrics.json"
)
GUIDED_PATH = OUTPUT_DIR / "flux_dev_tight_bbox_block-18_eta-scale-0p25_guided.png"
OVERLAY_PATH = OUTPUT_DIR / "flux_dev_tight_bbox_targets_overlay.png"
COMPARISON_PATH = OUTPUT_DIR / "flux_dev_tight_bbox_baseline_stage6d_stage6e_comparison.png"
METRICS_PATH = OUTPUT_DIR / "flux_dev_tight_bbox_block-18_eta-scale-0p25_metrics.json"

GRID_SIZE = 48
IMAGE_SIZE = 768
BLOCKS = (18,)
ETA_SCALE = 0.25
SOURCE_BOXES_GRID = {
    "apple": [6, 23, 25, 44],
    "cup": [25, 20, 48, 44],
}
# Preserve each object's own width, height, and y range; shift horizontally by
# 21 tokens so its center occupies the other object's baseline side.
TARGET_BOXES_GRID = {
    "apple": [27, 23, 46, 44],
    "cup": [4, 20, 27, 44],
}
BROAD_REGIONS_GRID = {
    "left": [0, 0, 24, 48],
    "right": [24, 0, 48, 48],
}


def load_stage6d():
    spec = importlib.util.spec_from_file_location("flux_dev_paper_aligned_guidance", STAGE6D_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Stage-6D helpers from {STAGE6D_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalized(box):
    return [value / GRID_SIZE for value in box]


def image_box(box):
    scale = IMAGE_SIZE // GRID_SIZE
    return [value * scale for value in box]


def configure_tight_layout(stage3) -> None:
    stage3.NORMALIZED_REGIONS = {
        "apple_target": normalized(TARGET_BOXES_GRID["apple"]),
        "cup_target": normalized(TARGET_BOXES_GRID["cup"]),
    }
    stage3.LAYOUTS = {
        "layout_b": {"apple": "apple_target", "cup": "cup_target"},
    }
    for word, region_name in stage3.LAYOUTS["layout_b"].items():
        actual = stage3.normalized_bbox_to_grid(
            stage3.NORMALIZED_REGIONS[region_name], GRID_SIZE, GRID_SIZE
        )
        if actual != TARGET_BOXES_GRID[word]:
            raise RuntimeError(f"Tight-box round-trip failed for {word}: {actual}")


def save_target_overlay(baseline: Image.Image) -> None:
    image = baseline.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    colors = {"apple": "#ff4040", "cup": "#00d8ff"}
    for word, box in SOURCE_BOXES_GRID.items():
        x0, y0, x1, y1 = image_box(box)
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline="#ffd43b", width=4)
        draw.text((x0 + 5, y0 + 5), f"source {word}", fill="#ffd43b")
    for word, box in TARGET_BOXES_GRID.items():
        x0, y0, x1, y1 = image_box(box)
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=colors[word], width=5)
        draw.text((x0 + 5, y0 + 24), f"target {word}", fill=colors[word])
    image.save(OVERLAY_PATH)


def save_comparison(baseline: Image.Image, stage6d_image: Image.Image, guided: Image.Image) -> None:
    panels = [
        ("baseline", baseline.convert("RGB")),
        ("Stage 6D half-region eta 0.25x", stage6d_image.convert("RGB")),
        ("Stage 6E tight boxes eta 0.25x", guided.convert("RGB")),
    ]
    width, height = baseline.size
    header = 38
    canvas = Image.new("RGB", (width * len(panels), height + header), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (label, image) in enumerate(panels):
        canvas.paste(image, (column * width, header))
        draw.text((column * width + 12, 11), label, fill="black")
    canvas.save(COMPARISON_PATH)


def main() -> None:
    stage6d = load_stage6d()
    stage5 = stage6d.load_stage5()
    stage4 = stage5.load_stage4()
    stage3 = stage4.load_stage3()
    helpers = stage3.load_helpers()
    configure_tight_layout(stage3)

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for Stage 6E")
    for required in (
        BASELINE_IMAGE_PATH,
        STAGE6D_IMAGE_PATH,
        STAGE6D_METRICS_PATH,
        stage6d.STAGE6A_METRICS_PATH,
    ):
        if not required.exists():
            raise FileNotFoundError(f"Required artifact missing: {required}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()
    baseline_state, baseline_metrics = helpers.load_baseline_state()
    baseline_image = Image.open(BASELINE_IMAGE_PATH).convert("RGB")
    stage6d_image = Image.open(STAGE6D_IMAGE_PATH).convert("RGB")
    stage6d_metrics = json.loads(STAGE6D_METRICS_PATH.read_text())
    stage6a = json.loads(stage6d.STAGE6A_METRICS_PATH.read_text())
    if stage5.pixel_sha256(baseline_image) != baseline_metrics["pixel_sha256"]:
        raise RuntimeError("Baseline image does not match Stage 1")

    pipe = FluxPipeline.from_pretrained(
        helpers.MODEL_ID, torch_dtype=helpers.DTYPE, local_files_only=True
    )
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

    stage5.set_schedule(pipe, helpers, state["latents"].shape[1], state["latents"].device)
    reference_sigma = pipe.scheduler.sigmas[stage6d.REFERENCE_INDEX].float()
    reference_record = stage6a["iteration_records"][0]
    reference_coefficient = float(reference_record["effective_eta"])
    base_eta = reference_coefficient / float(reference_sigma.square().item())
    eta = base_eta * ETA_SCALE

    guided_latents, guided_image, inner_records, step_summaries, guided_runtime = stage6d.run_guided(
        pipe,
        helpers,
        stage3,
        stage4,
        stage5,
        state,
        baseline_state,
        BLOCKS,
        target_spans,
        eta,
    )
    guided_image.save(GUIDED_PATH)
    save_target_overlay(baseline_image)
    save_comparison(baseline_image, stage6d_image, guided_image)

    baseline_centroids = stage5.color_centroids(baseline_image, helpers.WIDTH, helpers.HEIGHT)
    guided_centroids = stage5.color_centroids(guided_image, helpers.WIDTH, helpers.HEIGHT)
    per_step_trajectory = []
    for summary in step_summaries:
        index_updates = [
            item for item in inner_records if item["denoising_index"] == summary["denoising_index"]
        ]
        last = index_updates[-1]["blocks"]["18"]["objects"]
        per_step_trajectory.append(
            {
                **summary,
                "final_box_inside_ratio": {
                    word: last[word]["target_mass_after"] for word in ("apple", "cup")
                },
            }
        )

    record = {
        "stage": "Stage 6E - tight bounding-box layout control check",
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
        "blocks": list(BLOCKS),
        "guided_indices": list(stage6d.GUIDED_INDICES),
        "max_inner_updates_per_index": stage6d.MAX_INNER_UPDATES,
        "objective_threshold": stage6d.PER_BLOCK_OBJECTIVE_THRESHOLD,
        "objective_definition": "unchanged Eq. (2), finite object-sized target boxes",
        "update_equation": "x_t <- x_t - eta * sigma_t^2 * grad_x E",
        "eta_scale": ETA_SCALE,
        "base_eta": base_eta,
        "eta": eta,
        "original_broad_regions": {
            name: {
                "grid_bbox_xyxy_half_open": box,
                "image_bbox_xyxy_half_open": image_box(box),
                "normalized_bbox_xyxy": normalized(box),
                "token_count": (box[2] - box[0]) * (box[3] - box[1]),
            }
            for name, box in BROAD_REGIONS_GRID.items()
        },
        "baseline_source_boxes": {
            word: {
                "grid_bbox_xyxy_half_open": box,
                "image_bbox_xyxy_half_open": image_box(box),
                "width_tokens": box[2] - box[0],
                "height_tokens": box[3] - box[1],
            }
            for word, box in SOURCE_BOXES_GRID.items()
        },
        "tight_target_boxes": {
            word: {
                "grid_bbox_xyxy_half_open": box,
                "image_bbox_xyxy_half_open": image_box(box),
                "normalized_bbox_xyxy": normalized(box),
                "width_tokens": box[2] - box[0],
                "height_tokens": box[3] - box[1],
                "token_count": (box[2] - box[0]) * (box[3] - box[1]),
            }
            for word, box in TARGET_BOXES_GRID.items()
        },
        "box_construction": (
            "Preserve each baseline object's own width, height, and y extent; "
            "translate horizontally by 21 image tokens to the opposite side."
        ),
        "inner_updates": inner_records,
        "per_step_trajectory": per_step_trajectory,
        "baseline": {
            "image_path": str(BASELINE_IMAGE_PATH),
            "pixel_sha256": stage5.pixel_sha256(baseline_image),
            "trajectory_source": "same saved Stage-1 initial packed latent",
        },
        "stage6d_half_region_reference": {
            "image_path": str(STAGE6D_IMAGE_PATH),
            "metrics_path": str(STAGE6D_METRICS_PATH),
            "pixel_difference_vs_baseline": stage6d_metrics["guided"]["pixel_difference_vs_baseline"],
        },
        "guided": {
            "image_path": str(GUIDED_PATH),
            "pixel_sha256": stage5.pixel_sha256(guided_image),
            "pixel_difference_vs_baseline": stage5.image_difference(baseline_image, guided_image),
            "color_centroid_changes_vs_baseline": stage5.centroid_changes(
                baseline_centroids, guided_centroids
            ),
            "final_packed_latent_l2_difference_vs_baseline": float(
                (
                    guided_latents.float()
                    - baseline_state["pipeline_final_latents"].to(guided_latents.device).float()
                ).norm().item()
            ),
        },
        "all_updates_finite_and_descending": all(
            item["gradient_is_finite"]
            and item["model_parameter_grad_count"] == 0
            and item["objective_after"] <= item["objective_before"]
            for item in inner_records
        ),
        "manual_visual_assessment": "pending",
        "target_overlay_path": str(OVERLAY_PATH),
        "comparison_path": str(COMPARISON_PATH),
        "guided_runtime_seconds": guided_runtime,
        "total_runtime_seconds_including_load": time.perf_counter() - total_start,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }
    METRICS_PATH.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                "eta": eta,
                "tight_target_boxes": record["tight_target_boxes"],
                "per_step_trajectory": per_step_trajectory,
                "guided": record["guided"],
                "all_updates_finite_and_descending": record["all_updates_finite_and_descending"],
                "guided_runtime_seconds": guided_runtime,
                "peak_cuda_allocated_mib": record["peak_cuda_allocated_mib"],
                "metrics_path": str(METRICS_PATH),
                "target_overlay_path": str(OVERLAY_PATH),
                "comparison_path": str(COMPARISON_PATH),
            },
            indent=2,
        )
    )
    if not record["all_updates_finite_and_descending"]:
        raise RuntimeError("Stage 6E mechanical validation failed")
    pipe.maybe_free_model_hooks()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
