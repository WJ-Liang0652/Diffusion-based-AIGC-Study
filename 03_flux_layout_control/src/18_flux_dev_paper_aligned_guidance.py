"""Stage 6C: paper-aligned early Backward Guidance on FLUX.1-dev.

Use the practical update from the released layout-guidance implementation,
adapted to the FlowMatch sigma sequence:

    x_t <- x_t - eta * sigma_t**2 * grad_x E

Eta is calibrated once from the validated Stage-6A index-35 update and then
held fixed. Guidance is applied only at the first ten sampling steps, with up
to five fresh inner-loop gradients per step.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import diffusers
import torch
from diffusers import FluxPipeline
from PIL import Image, ImageDraw


PROJECT_DIR = Path(__file__).resolve().parents[1]
STAGE5_PATH = Path(__file__).with_name("15_flux_dev_single_step_guidance.py")
OUTPUT_DIR = PROJECT_DIR / "outputs" / "paper_aligned_guidance"
BASELINE_IMAGE_PATH = PROJECT_DIR / "outputs" / "baseline" / "flux_dev_baseline_seed-42_steps-50.png"
STAGE6A_METRICS_PATH = (
    PROJECT_DIR / "outputs" / "inner_loop_guidance" / "flux_dev_inner_loop_k-1-3-5_rel-0p005_metrics.json"
)
BLOCK18_IMAGE_PATH = OUTPUT_DIR / "flux_dev_paper_aligned_blocks-18_guided.png"

REFERENCE_INDEX = 35
REFERENCE_RELATIVE_STEP = 0.005
GUIDED_INDICES = tuple(range(10))
MAX_INNER_UPDATES = 5
PER_BLOCK_OBJECTIVE_THRESHOLD = 0.2
SAFETY_MAX_RELATIVE_UPDATE = 0.1


def load_stage5():
    spec = importlib.util.spec_from_file_location("flux_dev_single_step_guidance", STAGE5_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Stage-5 helpers from {STAGE5_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_blocks(value: str) -> tuple[int, ...]:
    blocks = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if blocks not in ((18,), (9, 18)):
        raise argparse.ArgumentTypeError("--blocks must be exactly 18 or 9,18")
    return blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=parse_blocks, default=(18,))
    parser.add_argument("--eta-scale", type=float, default=1.0)
    return parser.parse_args()


def block_label(blocks: tuple[int, ...]) -> str:
    return "-".join(str(value) for value in blocks)


def scale_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def install_probes(pipe, stage4, blocks, target_spans):
    installed = {}
    for block_index in blocks:
        attention = pipe.transformer.transformer_blocks[block_index].attn
        original = attention.get_processor()
        probe = stage4.GraphLayoutAttentionProbe(original, target_spans)
        attention.set_processor(probe)
        installed[block_index] = (attention, original, probe)
    return installed


def restore_probes(installed) -> None:
    for attention, original, probe in installed.values():
        probe.release()
        attention.set_processor(original)


def snapshot_blocks(stage3, stage4, stage5, installed):
    layouts = {}
    maps = {}
    centroids = {}
    for block_index, (_, _, probe) in installed.items():
        layouts[block_index] = stage4.serializable_layout(stage3, probe.maps, "layout_b")
        maps[block_index] = {
            word: value.detach().cpu().clone() for word, value in probe.maps.items()
        }
        centroids[block_index] = {
            word: stage5.map_statistics(value.detach().cpu())["center_of_mass_xy"]
            for word, value in probe.maps.items()
        }
    return layouts, maps, centroids


def summed_objective(stage3, stage4, installed):
    objectives = {}
    for block_index, (_, _, probe) in installed.items():
        objectives[block_index], _ = stage4.layout_tensors(stage3, probe.maps, "layout_b")
    return sum(objectives.values()), objectives


@torch.no_grad()
def evaluate_latents(pipe, stage3, stage4, stage5, state, latents, timestep, blocks, target_spans):
    installed = install_probes(pipe, stage4, blocks, target_spans)
    try:
        noise_pred = stage5.transformer_forward(pipe, state, latents, timestep)
        layouts, maps, centroids = snapshot_blocks(stage3, stage4, stage5, installed)
    finally:
        restore_probes(installed)
    total = sum(layout["total_energy"] for layout in layouts.values())
    return noise_pred, total, layouts, maps, centroids


def optimize_once(
    pipe,
    stage3,
    stage4,
    stage5,
    state,
    latents,
    timestep,
    sigma,
    eta,
    blocks,
    target_spans,
    denoising_index,
    inner_index,
):
    installed = install_probes(pipe, stage4, blocks, target_spans)
    variable = latents.detach().clone().requires_grad_(True)
    try:
        noise_pred = stage5.transformer_forward(pipe, state, variable, timestep)
        before_layouts, _, before_centroids = snapshot_blocks(
            stage3, stage4, stage5, installed
        )
        objective, block_objectives = summed_objective(stage3, stage4, installed)
        del noise_pred
        gradient = torch.autograd.grad(objective, variable, create_graph=False, retain_graph=False)[0]
        gradient_float = gradient.float()
        gradient_norm = gradient_float.norm()
        latent_norm = variable.detach().float().norm()
        coefficient = eta * sigma.float().square()
        update = (coefficient * gradient_float).to(variable.dtype)
        updated = (variable.detach() - update).detach()
        actual_update_norm = (updated.float() - variable.detach().float()).norm()
        relative_update = actual_update_norm / latent_norm
        parameter_grads = [
            name for name, parameter in pipe.transformer.named_parameters() if parameter.grad is not None
        ]
        finite = bool(
            torch.isfinite(objective).item()
            and torch.isfinite(gradient).all().item()
            and torch.isfinite(update).all().item()
            and torch.isfinite(updated).all().item()
        )
        if not finite or gradient_norm.item() == 0:
            raise RuntimeError(
                f"Non-finite/zero-gradient update at index={denoising_index}, inner={inner_index}"
            )
        if parameter_grads:
            raise RuntimeError(
                f"Model gradients at index={denoising_index}, inner={inner_index}: {parameter_grads[:3]}"
            )
        if relative_update.item() > SAFETY_MAX_RELATIVE_UPDATE:
            raise RuntimeError(
                f"Relative update {relative_update.item():.6f} exceeds safety limit "
                f"{SAFETY_MAX_RELATIVE_UPDATE} at index={denoising_index}, inner={inner_index}"
            )
    finally:
        restore_probes(installed)

    post_noise, after_total, after_layouts, _, after_centroids = evaluate_latents(
        pipe,
        stage3,
        stage4,
        stage5,
        state,
        updated,
        timestep,
        blocks,
        target_spans,
    )
    before_total = float(objective.detach().float().item())
    if not math.isfinite(after_total) or after_total > before_total:
        raise RuntimeError(
            f"Non-descending paper update at index={denoising_index}, inner={inner_index}: "
            f"{before_total} -> {after_total}"
        )
    per_block = {}
    for block_index in blocks:
        objects = {}
        for word in target_spans:
            before_xy = before_centroids[block_index][word]
            after_xy = after_centroids[block_index][word]
            objects[word] = {
                "target_region": before_layouts[block_index]["objects"][word]["region"],
                "target_mass_before": before_layouts[block_index]["objects"][word]["mass_ratio"],
                "target_mass_after": after_layouts[block_index]["objects"][word]["mass_ratio"],
                "attention_centroid_before_xy": before_xy,
                "attention_centroid_after_xy": after_xy,
                "attention_centroid_delta_xy": [
                    after_xy[0] - before_xy[0],
                    after_xy[1] - before_xy[1],
                ],
            }
        per_block[str(block_index)] = {
            "objective_before": float(block_objectives[block_index].detach().float().item()),
            "objective_after": after_layouts[block_index]["total_energy"],
            "objects": objects,
        }
    record = {
        "denoising_index": denoising_index,
        "timestep": float(timestep.item()),
        "sigma": float(sigma.item()),
        "sigma_squared": float(sigma.float().square().item()),
        "inner_index": inner_index,
        "objective_before": before_total,
        "objective_after": after_total,
        "objective_delta": after_total - before_total,
        "gradient_norm_l2": float(gradient_norm.item()),
        "latent_norm_l2": float(latent_norm.item()),
        "eta_sigma_squared": float(coefficient.item()),
        "actual_update_norm_l2": float(actual_update_norm.item()),
        "actual_relative_update_norm": float(relative_update.item()),
        "gradient_is_finite": True,
        "model_parameter_grad_count": 0,
        "blocks": per_block,
    }
    del variable, objective, gradient, gradient_float, update
    torch.cuda.empty_cache()
    return updated, post_noise, record


def run_guided(
    pipe, helpers, stage3, stage4, stage5, state, baseline_state, blocks, target_spans, eta
):
    start = time.perf_counter()
    latents = baseline_state["initial_latents"].to(
        device=state["prompt_embeds"].device, dtype=state["prompt_embeds"].dtype
    )
    timesteps = stage5.set_schedule(pipe, helpers, latents.shape[1], latents.device)
    sigmas = pipe.scheduler.sigmas
    inner_records = []
    step_summaries = []
    threshold = PER_BLOCK_OBJECTIVE_THRESHOLD * len(blocks)
    for index, timestep in enumerate(timesteps):
        if index in GUIDED_INDICES:
            first_record = None
            last_record = None
            noise_pred = None
            for inner_index in range(MAX_INNER_UPDATES):
                latents, noise_pred, record = optimize_once(
                    pipe,
                    stage3,
                    stage4,
                    stage5,
                    state,
                    latents,
                    timestep,
                    sigmas[index],
                    eta,
                    blocks,
                    target_spans,
                    index,
                    inner_index,
                )
                inner_records.append(record)
                first_record = first_record or record
                last_record = record
                if record["objective_after"] <= threshold:
                    break
            step_summaries.append(
                {
                    "denoising_index": index,
                    "timestep": float(timestep.item()),
                    "sigma": float(sigmas[index].item()),
                    "sigma_squared": float(sigmas[index].float().square().item()),
                    "executed_inner_updates": last_record["inner_index"] + 1,
                    "objective_initial": first_record["objective_before"],
                    "objective_final": last_record["objective_after"],
                    "relative_update_first": first_record["actual_relative_update_norm"],
                    "relative_update_last": last_record["actual_relative_update_norm"],
                    "stopped_on_objective_threshold": last_record["objective_after"] <= threshold,
                    "final_attention_centroids": {
                        block: {
                            word: values["attention_centroid_after_xy"]
                            for word, values in block_record["objects"].items()
                        }
                        for block, block_record in last_record["blocks"].items()
                    },
                }
            )
        else:
            with torch.no_grad():
                noise_pred = stage5.transformer_forward(pipe, state, latents, timestep)
        with torch.no_grad():
            latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
        del noise_pred
    image = stage5.decode(pipe, helpers, latents)
    return latents.detach(), image, inner_records, step_summaries, time.perf_counter() - start


def save_comparison(baseline, guided, blocks, path):
    panels = [("exact Stage-1 baseline", baseline.convert("RGB"))]
    if blocks == (9, 18) and BLOCK18_IMAGE_PATH.exists():
        panels.append(("paper aligned block 18", Image.open(BLOCK18_IMAGE_PATH).convert("RGB")))
    panels.append((f"paper aligned blocks {block_label(blocks)}", guided.convert("RGB")))
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
    blocks = tuple(args.blocks)
    eta_scale = float(args.eta_scale)
    if eta_scale not in (0.25, 0.5, 1.0):
        raise ValueError("--eta-scale must be one of 0.25, 0.5, or 1.0")
    stage5 = load_stage5()
    stage4 = stage5.load_stage4()
    stage3 = stage4.load_stage3()
    helpers = stage3.load_helpers()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for Stage 6C")
    for required in (BASELINE_IMAGE_PATH, STAGE6A_METRICS_PATH):
        if not required.exists():
            raise FileNotFoundError(f"Required artifact missing: {required}")
    if blocks == (9, 18) and not BLOCK18_IMAGE_PATH.exists():
        raise FileNotFoundError("Run the required block-18 experiment before the 9+18 control")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    label = block_label(blocks)
    suffix = "" if eta_scale == 1.0 else f"_eta-scale-{scale_label(eta_scale)}"
    guided_path = OUTPUT_DIR / f"flux_dev_paper_aligned_blocks-{label}{suffix}_guided.png"
    comparison_path = OUTPUT_DIR / f"flux_dev_paper_aligned_blocks-{label}{suffix}_comparison.png"
    metrics_path = OUTPUT_DIR / f"flux_dev_paper_aligned_blocks-{label}{suffix}_metrics.json"
    total_start = time.perf_counter()
    baseline_state, baseline_metrics = helpers.load_baseline_state()
    baseline_image = Image.open(BASELINE_IMAGE_PATH).convert("RGB")
    if stage5.pixel_sha256(baseline_image) != baseline_metrics["pixel_sha256"]:
        raise RuntimeError("Baseline image does not match Stage 1")
    stage6a = json.loads(STAGE6A_METRICS_PATH.read_text())

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

    calibration_timesteps = stage5.set_schedule(
        pipe, helpers, state["latents"].shape[1], state["latents"].device
    )
    reference_sigma = pipe.scheduler.sigmas[REFERENCE_INDEX].float()
    reference_record = stage6a["iteration_records"][0]
    reference_coefficient = float(reference_record["effective_eta"])
    calibrated_eta = reference_coefficient / float(reference_sigma.square().item())
    eta = calibrated_eta * eta_scale
    expected_reference_relative = (
        eta
        * float(reference_sigma.square().item())
        * float(reference_record["gradient_norm_l2"])
        / float(reference_record["actual_bf16_update_norm"] / reference_record["actual_relative_update_norm"])
    )
    del calibration_timesteps

    guided_latents, guided_image, inner_records, step_summaries, guided_runtime = run_guided(
        pipe,
        helpers,
        stage3,
        stage4,
        stage5,
        state,
        baseline_state,
        blocks,
        target_spans,
        eta,
    )
    guided_image.save(guided_path)
    save_comparison(baseline_image, guided_image, blocks, comparison_path)
    baseline_centroids = stage5.color_centroids(baseline_image, helpers.WIDTH, helpers.HEIGHT)
    guided_centroids = stage5.color_centroids(guided_image, helpers.WIDTH, helpers.HEIGHT)
    record = {
        "stage": (
            "Stage 6C - paper-aligned FLUX Backward Guidance MVP"
            if eta_scale == 1.0
            else "Stage 6D - Backward Guidance strength stabilization"
        ),
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
        "blocks": list(blocks),
        "layout_optimized": stage3.LAYOUTS["layout_b"],
        "update_equation": "x_t <- x_t - eta * sigma_t^2 * grad_x sum_block E_block",
        "eta_calibration": {
            "performed_once": True,
            "reference_denoising_index": REFERENCE_INDEX,
            "reference_timestep": float(state["timestep"].item()),
            "reference_sigma": float(reference_sigma.item()),
            "reference_sigma_squared": float(reference_sigma.square().item()),
            "reference_normalized_relative_step": REFERENCE_RELATIVE_STEP,
            "stage6a_effective_gradient_coefficient": reference_coefficient,
            "base_eta": calibrated_eta,
            "eta_scale": eta_scale,
            "fixed_eta": eta,
            "expected_reference_relative_update_from_saved_norms": expected_reference_relative,
            "recalibrated_during_run": False,
        },
        "guided_indices": list(GUIDED_INDICES),
        "max_inner_updates_per_index": MAX_INNER_UPDATES,
        "objective_threshold": PER_BLOCK_OBJECTIVE_THRESHOLD * len(blocks),
        "objective_definition": "unchanged Eq. (2), summed across selected blocks",
        "flowmatch_guided_schedule": [
            {
                "index": item["denoising_index"],
                "timestep": item["timestep"],
                "sigma": item["sigma"],
                "sigma_squared": item["sigma_squared"],
            }
            for item in step_summaries
        ],
        "inner_updates": inner_records,
        "step_summaries": step_summaries,
        "baseline": {
            "image_path": str(BASELINE_IMAGE_PATH),
            "pixel_sha256": stage5.pixel_sha256(baseline_image),
            "matches_stage1_image_exactly": True,
            "trajectory_source": "same saved Stage-1 initial packed latent",
        },
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
        "all_updates_finite_and_descending": all(
            item["gradient_is_finite"]
            and item["model_parameter_grad_count"] == 0
            and item["objective_after"] <= item["objective_before"]
            for item in inner_records
        ),
        "manual_visual_assessment": "pending",
        "comparison_path": str(comparison_path),
        "guided_runtime_seconds": guided_runtime,
        "total_runtime_seconds_including_load": time.perf_counter() - total_start,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }
    metrics_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    if not record["all_updates_finite_and_descending"]:
        raise RuntimeError("Stage 6C mechanical validation failed")

    pipe.maybe_free_model_hooks()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
