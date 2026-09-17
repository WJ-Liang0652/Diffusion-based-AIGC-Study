"""Stage 7: frozen Stage-6E Backward Guidance repeatability check.

The guidance method and all hyperparameters are inherited from Stage 6E.
Only the prompt/seed, target token spans, and explicitly recorded tight target
boxes vary between cases.  ``--phase baseline`` intentionally runs before
guidance so source objects can be annotated from each exact baseline.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import diffusers
import numpy as np
import torch
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from PIL import Image, ImageDraw


PROJECT_DIR = Path(__file__).resolve().parents[1]
STAGE6E_PATH = Path(__file__).with_name("19_flux_dev_tight_bbox_guidance.py")
OUTPUT_DIR = PROJECT_DIR / "outputs" / "repeatability"
GRID_SIZE = 48
IMAGE_SIZE = 768
BLOCKS = (18,)
ETA_SCALE = 0.25

# Source boxes were annotated from exact baselines; target boxes were then fixed
# before any guidance run. Guidance refuses to run while either box is absent.
CASES = {
    "apple_cup_seed123": {
        "prompt": "a red apple and a blue cup, realistic photo",
        "seed": 123,
        "objects": ("apple", "cup"),
        "source_boxes": {"apple": [22, 17, 42, 38], "cup": [0, 20, 25, 38]},
        "target_boxes": {"apple": [0, 17, 20, 38], "cup": [23, 20, 48, 38]},
    },
    "apple_cup_seed2024": {
        "prompt": "a red apple and a blue cup, realistic photo",
        "seed": 2024,
        "objects": ("apple", "cup"),
        "source_boxes": {"apple": [3, 19, 24, 41], "cup": [22, 9, 44, 40]},
        "target_boxes": {"apple": [26, 19, 47, 41], "cup": [0, 9, 22, 40]},
    },
    "banana_bottle_seed31415": {
        "prompt": "a yellow banana and a green bottle, realistic photo",
        "seed": 31415,
        "objects": ("banana", "bottle"),
        "source_boxes": {"banana": [9, 18, 32, 44], "bottle": [24, 6, 34, 42]},
        "target_boxes": {"banana": [25, 18, 48, 44], "bottle": [5, 6, 15, 42]},
    },
}


def load_stage6e():
    spec = importlib.util.spec_from_file_location("flux_dev_tight_bbox_guidance", STAGE6E_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Stage 6E helpers from {STAGE6E_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def paths(case_name: str) -> dict[str, Path]:
    case_dir = OUTPUT_DIR / case_name
    return {
        "dir": case_dir,
        "baseline": case_dir / "baseline.png",
        "state": case_dir / "baseline_state.pt",
        "baseline_metrics": case_dir / "baseline_metrics.json",
        "overlay": case_dir / "target_overlay.png",
        "guided": case_dir / "guided.png",
        "comparison": case_dir / "comparison.jpg",
        "metrics": case_dir / "metrics.json",
    }


def configure_helpers(helpers, case: dict[str, object]) -> None:
    helpers.PROMPT = case["prompt"]
    helpers.SEED = case["seed"]
    helpers.TARGET_WORDS = case["objects"]


def make_initial_latents(pipe, helpers):
    generator = torch.Generator(device="cpu").manual_seed(helpers.SEED)
    channels = pipe.transformer.config.in_channels // 4
    return pipe.prepare_latents(
        1,
        channels,
        helpers.HEIGHT,
        helpers.WIDTH,
        helpers.DTYPE,
        pipe._execution_device,
        generator,
        latents=None,
    )


def run_baseline(pipe, helpers, case_name: str) -> None:
    output = paths(case_name)
    output["dir"].mkdir(parents=True, exist_ok=True)
    initial_latents, image_ids = make_initial_latents(pipe, helpers)
    final_latents = []

    def capture_final(_pipe, step_index, _timestep, callback_kwargs):
        if step_index == helpers.NUM_INFERENCE_STEPS - 1:
            final_latents.append(callback_kwargs["latents"].detach().cpu().clone())
        return callback_kwargs

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    image = pipe(
        prompt=helpers.PROMPT,
        width=helpers.WIDTH,
        height=helpers.HEIGHT,
        num_inference_steps=helpers.NUM_INFERENCE_STEPS,
        guidance_scale=helpers.GUIDANCE_SCALE,
        num_images_per_prompt=1,
        generator=torch.Generator(device="cpu").manual_seed(helpers.SEED),
        latents=initial_latents.clone(),
        callback_on_step_end=capture_final,
        callback_on_step_end_tensor_inputs=["latents"],
    ).images[0].convert("RGB")
    runtime = time.perf_counter() - start
    if len(final_latents) != 1:
        raise RuntimeError(f"Expected one final latent, got {len(final_latents)}")
    state = {
        "model_id": helpers.MODEL_ID,
        "prompt": helpers.PROMPT,
        "seed": helpers.SEED,
        "width": helpers.WIDTH,
        "height": helpers.HEIGHT,
        "num_inference_steps": helpers.NUM_INFERENCE_STEPS,
        "guidance_scale": helpers.GUIDANCE_SCALE,
        "dtype": str(helpers.DTYPE),
        "initial_latents": initial_latents.detach().cpu().clone(),
        "pipeline_final_latents": final_latents[0],
        "latent_image_ids": image_ids.detach().cpu().clone(),
    }
    torch.save(state, output["state"])
    image.save(output["baseline"])
    record = {
        "stage": "Stage 7 exact baseline",
        "case": case_name,
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
        "pixel_sha256": helpers.pixel_sha256(image),
        "runtime_seconds": runtime,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "image_path": str(output["baseline"]),
        "state_path": str(output["state"]),
    }
    output["baseline_metrics"].write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


@torch.no_grad()
def prepare_sampling_state(pipe, helpers, baseline_state):
    pipe.check_inputs(helpers.PROMPT, None, helpers.HEIGHT, helpers.WIDTH, max_sequence_length=512)
    pipe._guidance_scale = helpers.GUIDANCE_SCALE
    pipe._joint_attention_kwargs = {}
    pipe._interrupt = False
    device = pipe._execution_device
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=helpers.PROMPT,
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=512,
        lora_scale=None,
    )
    initial_latents = baseline_state["initial_latents"].to(device=device, dtype=prompt_embeds.dtype)
    channels = pipe.transformer.config.in_channels // 4
    _, image_ids = pipe.prepare_latents(
        1,
        channels,
        helpers.HEIGHT,
        helpers.WIDTH,
        prompt_embeds.dtype,
        device,
        generator=None,
        latents=initial_latents,
    )
    sigmas = np.linspace(1.0, 1 / helpers.NUM_INFERENCE_STEPS, helpers.NUM_INFERENCE_STEPS)
    mu = calculate_shift(
        initial_latents.shape[1],
        pipe.scheduler.config.base_image_seq_len,
        pipe.scheduler.config.max_image_seq_len,
        pipe.scheduler.config.base_shift,
        pipe.scheduler.config.max_shift,
    )
    timesteps, _ = retrieve_timesteps(
        pipe.scheduler, helpers.NUM_INFERENCE_STEPS, device, sigmas=sigmas, mu=mu
    )
    guidance = torch.full(
        [initial_latents.shape[0]], helpers.GUIDANCE_SCALE, device=device, dtype=torch.float32
    )
    return {
        "latents": initial_latents,
        "timestep": timesteps[35],
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "text_ids": text_ids,
        "latent_image_ids": image_ids,
        "guidance": guidance,
        "mu": float(mu),
        "source": "Stage 7 exact-baseline initial packed latent",
    }


def configure_layout(stage3, case: dict[str, object]) -> None:
    boxes = case["target_boxes"]
    if boxes is None:
        raise RuntimeError("Target boxes have not been annotated; run/review baselines first")
    stage3.NORMALIZED_REGIONS = {
        f"{word}_target": [value / GRID_SIZE for value in boxes[word]]
        for word in case["objects"]
    }
    stage3.LAYOUTS = {
        "layout_b": {word: f"{word}_target" for word in case["objects"]}
    }
    for word in case["objects"]:
        actual = stage3.normalized_bbox_to_grid(
            stage3.NORMALIZED_REGIONS[f"{word}_target"], GRID_SIZE, GRID_SIZE
        )
        if actual != boxes[word]:
            raise RuntimeError(f"Target-box round-trip failed for {word}: {actual}")


def image_box(box):
    return [value * (IMAGE_SIZE // GRID_SIZE) for value in box]


def save_visuals(case, baseline, guided, output):
    overlay = baseline.copy()
    draw = ImageDraw.Draw(overlay)
    colors = ("#ff4040", "#00d8ff")
    if case["source_boxes"]:
        for word, box in case["source_boxes"].items():
            x0, y0, x1, y1 = image_box(box)
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline="#ffd43b", width=4)
            draw.text((x0 + 5, y0 + 5), f"source {word}", fill="#ffd43b")
    for color, word in zip(colors, case["objects"]):
        x0, y0, x1, y1 = image_box(case["target_boxes"][word])
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color, width=5)
        draw.text((x0 + 5, y0 + 24), f"target {word}", fill=color)
    overlay.save(output["overlay"])
    header = 36
    canvas = Image.new("RGB", (IMAGE_SIZE * 3, IMAGE_SIZE + header), "white")
    labels_images = (("baseline", baseline), ("target overlay", overlay), ("guided", guided))
    canvas_draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(labels_images):
        canvas.paste(image, (index * IMAGE_SIZE, header))
        canvas_draw.text((index * IMAGE_SIZE + 10, 10), label, fill="black")
    canvas.save(output["comparison"], quality=92)


def run_guidance(pipe, helpers, stage6e, case_name: str, case: dict[str, object]) -> None:
    output = paths(case_name)
    for required in (output["baseline"], output["state"], output["baseline_metrics"]):
        if not required.exists():
            raise FileNotFoundError(f"Required Stage 7 baseline artifact missing: {required}")
    stage6d = stage6e.load_stage6d()
    stage5 = stage6d.load_stage5()
    stage4 = stage5.load_stage4()
    stage3 = stage4.load_stage3()
    configure_layout(stage3, case)
    target_spans, tokens = helpers.locate_target_spans(pipe.tokenizer_2, helpers.PROMPT)
    expected_words = set(case["objects"])
    if set(target_spans) != expected_words:
        target_spans = {word: target_spans[word] for word in case["objects"]}
    for component_name in ("text_encoder", "text_encoder_2", "transformer", "vae"):
        component = getattr(pipe, component_name, None)
        if component is not None:
            component.requires_grad_(False)
    pipe.transformer.enable_gradient_checkpointing()
    baseline_state = torch.load(output["state"], map_location="cpu", weights_only=False)
    baseline_image = Image.open(output["baseline"]).convert("RGB")
    state = prepare_sampling_state(pipe, helpers, baseline_state)
    stage5.set_schedule(pipe, helpers, state["latents"].shape[1], state["latents"].device)
    reference_sigma = pipe.scheduler.sigmas[stage6d.REFERENCE_INDEX].float()
    stage6a = json.loads(stage6d.STAGE6A_METRICS_PATH.read_text())
    reference_coefficient = float(stage6a["iteration_records"][0]["effective_eta"])
    base_eta = reference_coefficient / float(reference_sigma.square().item())
    eta = base_eta * ETA_SCALE
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    total_start = time.perf_counter()
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
    guided_image = guided_image.convert("RGB")
    guided_image.save(output["guided"])
    save_visuals(case, baseline_image, guided_image, output)
    last_record = inner_records[-1]["blocks"]["18"]["objects"]
    final_ratios = {
        word: float(last_record[word]["target_mass_after"]) for word in case["objects"]
    }
    final_centroids = {
        word: last_record[word]["attention_centroid_after_xy"] for word in case["objects"]
    }
    record = {
        "stage": "Stage 7 - frozen Stage 6E repeatability",
        "case": case_name,
        "model_id": helpers.MODEL_ID,
        "diffusers_version": diffusers.__version__,
        "torch_version": torch.__version__,
        "prompt": helpers.PROMPT,
        "seed": helpers.SEED,
        "target_tokens": {word: target_spans[word] for word in case["objects"]},
        "non_padding_tokens": tokens,
        "width": helpers.WIDTH,
        "height": helpers.HEIGHT,
        "num_inference_steps": helpers.NUM_INFERENCE_STEPS,
        "guidance_scale": helpers.GUIDANCE_SCALE,
        "dtype": str(helpers.DTYPE),
        "offload_strategy": helpers.OFFLOAD_STRATEGY,
        "frozen_method": {
            "blocks": list(BLOCKS),
            "guided_indices": list(stage6d.GUIDED_INDICES),
            "max_inner_updates_per_index": stage6d.MAX_INNER_UPDATES,
            "objective_threshold": stage6d.PER_BLOCK_OBJECTIVE_THRESHOLD,
            "objective": "Eq. (2), sum over object target-box energies",
            "update_equation": "x_t <- x_t - eta * sigma_t^2 * grad_x E",
            "eta_scale": ETA_SCALE,
            "base_eta": base_eta,
            "eta": eta,
        },
        "source_boxes_grid_xyxy_half_open": case["source_boxes"],
        "target_boxes_grid_xyxy_half_open": case["target_boxes"],
        "target_boxes_image_xyxy_half_open": {
            word: image_box(box) for word, box in case["target_boxes"].items()
        },
        "inner_updates": inner_records,
        "step_summaries": step_summaries,
        "final_box_inside_ratios": final_ratios,
        "final_attention_centroids_xy": final_centroids,
        "all_updates_finite_and_descending": all(
            item["gradient_is_finite"]
            and item["model_parameter_grad_count"] == 0
            and item["objective_after"] <= item["objective_before"]
            for item in inner_records
        ),
        "baseline": {
            "image_path": str(output["baseline"]),
            "pixel_sha256": helpers.pixel_sha256(baseline_image),
        },
        "guided": {
            "image_path": str(output["guided"]),
            "pixel_sha256": helpers.pixel_sha256(guided_image),
            "pixel_difference_vs_baseline": stage5.image_difference(baseline_image, guided_image),
            "final_packed_latent_l2_difference_vs_baseline": float(
                (guided_latents.float() - baseline_state["pipeline_final_latents"].to(
                    guided_latents.device
                ).float()).norm().item()
            ),
        },
        "manual_visual_assessment": "pending",
        "baseline_metrics_path": str(output["baseline_metrics"]),
        "target_overlay_path": str(output["overlay"]),
        "comparison_path": str(output["comparison"]),
        "guided_runtime_seconds": guided_runtime,
        "total_runtime_seconds_including_prompt_encoding": time.perf_counter() - total_start,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }
    output["metrics"].write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: record[key] for key in (
        "case", "prompt", "seed", "target_tokens", "final_box_inside_ratios",
        "final_attention_centroids_xy", "all_updates_finite_and_descending",
        "guided_runtime_seconds", "peak_cuda_allocated_mib", "comparison_path"
    )}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("baseline", "guidance", "all"), default="all")
    parser.add_argument("--case", choices=tuple(CASES), action="append")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for Stage 7")
    stage6e = load_stage6e()
    stage6d = stage6e.load_stage6d()
    stage5 = stage6d.load_stage5()
    stage4 = stage5.load_stage4()
    stage3 = stage4.load_stage3()
    helpers = stage3.load_helpers()
    selected = args.case or list(CASES)
    load_start = time.perf_counter()
    pipe = FluxPipeline.from_pretrained(
        helpers.MODEL_ID, torch_dtype=helpers.DTYPE, local_files_only=True
    )
    pipe.enable_sequential_cpu_offload()
    print(f"model_load_seconds={time.perf_counter() - load_start:.2f}")
    for case_name in selected:
        case = CASES[case_name]
        configure_helpers(helpers, case)
        if args.phase in ("baseline", "all"):
            run_baseline(pipe, helpers, case_name)
        if args.phase in ("guidance", "all"):
            run_guidance(pipe, helpers, stage6e, case_name, case)
    pipe.maybe_free_model_hooks()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
