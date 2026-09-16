"""Validate the unchanged paper Eq. (2) layout energy on FLUX.1-dev.

This Stage-3 experiment reuses the verified dev Pipeline path and native
joint-attention probe.  It evaluates only Double Stream blocks 18 and 9 at
denoising index 35; it performs no autograd, latent update, or guidance.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import diffusers
import matplotlib.pyplot as plt
import torch
from diffusers import FluxPipeline


PROJECT_DIR = Path(__file__).resolve().parents[1]
HELPER_PATH = Path(__file__).with_name("11_flux_dev_attention_probe.py")
OUTPUT_DIR = PROJECT_DIR / "outputs" / "layout_objective"
METRICS_PATH = OUTPUT_DIR / "flux_dev_layout_objective_metrics.json"
PANEL_PATH = OUTPUT_DIR / "flux_dev_layout_objective_joint_maps.png"

PROBE_DENOISING_INDEX = 35
DOUBLE_BLOCK_INDICES = (18, 9)
EXPECTED_GRID_SHAPE = (48, 48)
EXPECTED_TOKEN_SPANS = {
    "apple": {"token_indices": [3], "token_ids": [8947], "token_pieces": ["▁apple"]},
    "cup": {"token_indices": [8], "token_ids": [4119], "token_pieces": ["▁cup"]},
}

# Fixed normalized target regions, unchanged from the schnell objective.
NORMALIZED_REGIONS = {
    "left": [0.0, 0.0, 0.5, 1.0],
    "right": [0.5, 0.0, 1.0, 1.0],
}
LAYOUTS = {
    "layout_a": {"apple": "left", "cup": "right"},
    "layout_b": {"apple": "right", "cup": "left"},
}


def load_helpers():
    spec = importlib.util.spec_from_file_location("flux_dev_attention_helpers", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load dev attention helpers from {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalized_bbox_to_grid(normalized_bbox: list[float], grid_height: int, grid_width: int) -> list[int]:
    x0, y0, x1, y1 = normalized_bbox
    grid_bbox = [
        math.floor(x0 * grid_width),
        math.floor(y0 * grid_height),
        math.ceil(x1 * grid_width),
        math.ceil(y1 * grid_height),
    ]
    grid_bbox[0] = max(0, min(grid_width, grid_bbox[0]))
    grid_bbox[2] = max(0, min(grid_width, grid_bbox[2]))
    grid_bbox[1] = max(0, min(grid_height, grid_bbox[1]))
    grid_bbox[3] = max(0, min(grid_height, grid_bbox[3]))
    if grid_bbox[0] >= grid_bbox[2] or grid_bbox[1] >= grid_bbox[3]:
        raise ValueError(f"Degenerate grid bbox {grid_bbox} from {normalized_bbox}")
    return grid_bbox


def equation_2(attention_map: torch.Tensor, normalized_bbox: list[float]) -> dict[str, object]:
    """E(A,B,i) = (1 - sum_{u in B} A[u,i] / sum_u A[u,i])^2."""
    height, width = attention_map.shape
    x0, y0, x1, y1 = normalized_bbox_to_grid(normalized_bbox, height, width)
    total_mass = attention_map.double().sum()
    in_box_mass = attention_map[y0:y1, x0:x1].double().sum()
    if not torch.isfinite(total_mass) or total_mass <= 0:
        raise RuntimeError(f"Invalid total attention mass: {total_mass.item()}")
    mass_ratio = in_box_mass / total_mass
    energy = (1.0 - mass_ratio).square()
    return {
        "normalized_bbox_xyxy": normalized_bbox,
        "grid_bbox_xyxy": [x0, y0, x1, y1],
        "total_attention_mass": float(total_mass.item()),
        "in_box_attention_mass": float(in_box_mass.item()),
        "in_box_attention_mass_ratio": float(mass_ratio.item()),
        "energy": float(energy.item()),
    }


def evaluate_layouts(probe, target_words: tuple[str, ...]) -> dict[str, object]:
    results = {}
    for layout_name, assignments in LAYOUTS.items():
        objects = {}
        for word, region_name in assignments.items():
            attention_map = probe.maps[(PROBE_DENOISING_INDEX, "joint", word)]
            objects[word] = {
                "region": region_name,
                **equation_2(attention_map, NORMALIZED_REGIONS[region_name]),
            }
        results[layout_name] = {
            "objects": objects,
            "total_energy": sum(item["energy"] for item in objects.values()),
        }

    energy_a = results["layout_a"]["total_energy"]
    energy_b = results["layout_b"]["total_energy"]
    all_values = [
        results[layout]["objects"][word][field]
        for layout in LAYOUTS
        for word in target_words
        for field in ("total_attention_mass", "in_box_attention_mass", "in_box_attention_mass_ratio", "energy")
    ]
    results["comparison"] = {
        "layout_a_lower_than_layout_b": energy_a < energy_b,
        "energy_margin_b_minus_a": energy_b - energy_a,
        "energy_ratio_b_over_a": energy_b / energy_a,
        "all_object_metrics_finite": all(math.isfinite(value) for value in all_values),
    }
    return results


def validate_regions() -> dict[str, object]:
    masks = {}
    coverage = torch.zeros(EXPECTED_GRID_SHAPE, dtype=torch.int32)
    for name, region in NORMALIZED_REGIONS.items():
        x0, y0, x1, y1 = normalized_bbox_to_grid(region, *EXPECTED_GRID_SHAPE)
        mask = torch.zeros(EXPECTED_GRID_SHAPE, dtype=torch.bool)
        mask[y0:y1, x0:x1] = True
        masks[name] = {
            "normalized_bbox_xyxy": region,
            "grid_bbox_xyxy": [x0, y0, x1, y1],
            "token_count": int(mask.sum().item()),
        }
        coverage += mask
    valid = bool(torch.all(coverage == 1).item()) and all(item["token_count"] == 1152 for item in masks.values())
    if not valid:
        raise RuntimeError(f"Left/right region masks do not partition the 48x48 grid: {masks}")
    return {"regions": masks, "partition_48x48_exactly_once": valid}


def save_panel(probes, target_words: tuple[str, ...], timestep: float) -> None:
    figure, axes = plt.subplots(len(probes), len(target_words), figsize=(10, 9), squeeze=False)
    for row, probe in enumerate(probes):
        for column, word in enumerate(target_words):
            signal_map = probe.maps[(PROBE_DENOISING_INDEX, "joint", word)].float().numpy()
            axis = axes[row, column]
            rendered = axis.imshow(signal_map, cmap="magma", interpolation="nearest")
            axis.axvline(23.5, color="cyan", linewidth=1.2)
            desired_side = LAYOUTS["layout_a"][word]
            axis.set_title(f"block {probe.block_index}: {word} -> {desired_side}")
            axis.set_xlabel("image-token x")
            axis.set_ylabel("image-token y")
            figure.colorbar(rendered, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(f"FLUX.1-dev Eq.(2) inputs, denoise=35, t={timestep:.3f}")
    figure.tight_layout()
    figure.savefig(PANEL_PATH, dpi=150)
    plt.close(figure)


def main() -> None:
    helpers = load_helpers()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this FLUX layout-objective validation")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_state, baseline_metrics = helpers.load_baseline_state()
    pipe = FluxPipeline.from_pretrained(helpers.MODEL_ID, torch_dtype=helpers.DTYPE, local_files_only=True)
    target_spans, non_padding_tokens = helpers.locate_target_spans(pipe.tokenizer_2, helpers.PROMPT)
    for word, expected in EXPECTED_TOKEN_SPANS.items():
        for field, value in expected.items():
            if target_spans[word][field] != value:
                raise RuntimeError(f"Unexpected {word} {field}: {target_spans[word][field]} != {value}")

    probes = []
    for block_index in DOUBLE_BLOCK_INDICES:
        attention = pipe.transformer.transformer_blocks[block_index].attn
        probe = helpers.DevSpatialAttentionProbe(
            attention.get_processor(), block_index, target_spans, (PROBE_DENOISING_INDEX,)
        )
        attention.set_processor(probe)
        probes.append(probe)

    image, timesteps, runtime_record = helpers.run_pipeline_with_probes(
        pipe, probes, baseline_state, baseline_metrics
    )
    del image
    timestep = timesteps[PROBE_DENOISING_INDEX]
    if not math.isclose(timestep, 499.842, rel_tol=0.0, abs_tol=1e-3):
        raise RuntimeError(f"Unexpected denoising timestep at index 35: {timestep}")

    expected_keys = {
        (PROBE_DENOISING_INDEX, normalization, word)
        for normalization in ("joint", "text_only")
        for word in helpers.TARGET_WORDS
    }
    region_validation = validate_regions()
    block_results = {}
    for probe in probes:
        if set(probe.maps) != expected_keys:
            raise RuntimeError(f"Incomplete capture for block {probe.block_index}: {sorted(probe.maps)}")
        for word in helpers.TARGET_WORDS:
            shape = tuple(probe.maps[(PROBE_DENOISING_INDEX, "joint", word)].shape)
            if shape != EXPECTED_GRID_SHAPE:
                raise RuntimeError(f"Unexpected block {probe.block_index} {word} map shape: {shape}")
        result = evaluate_layouts(probe, helpers.TARGET_WORDS)
        if not result["comparison"]["all_object_metrics_finite"]:
            raise RuntimeError(f"Non-finite Eq.(2) metric at block {probe.block_index}")
        if not result["comparison"]["layout_a_lower_than_layout_b"]:
            raise RuntimeError(f"Layout A is not lower-energy than Layout B at block {probe.block_index}")
        block_results[str(probe.block_index)] = {
            "block_index": probe.block_index,
            "qk_metadata": probe.qk_metadata,
            **result,
        }

    save_panel(probes, helpers.TARGET_WORDS, timestep)
    objective_passed = all(
        result["comparison"]["layout_a_lower_than_layout_b"]
        and result["comparison"]["all_object_metrics_finite"]
        for result in block_results.values()
    )
    record = {
        "stage": "Stage 3 - dev layout objective",
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
        "probe_denoising_index": PROBE_DENOISING_INDEX,
        "probe_timestep": timestep,
        "selected_double_blocks": list(DOUBLE_BLOCK_INDICES),
        "attention_definition": "native joint-normalized P(target text-key span | image query)",
        "equation": "E(A,B,i) = (1 - sum_{u in B} A[u,i] / sum_u A[u,i])^2",
        "target_spans": target_spans,
        "t5_non_padding_token_sequence": non_padding_tokens,
        "grid_shape": list(EXPECTED_GRID_SHAPE),
        "normalized_regions": NORMALIZED_REGIONS,
        "layouts": LAYOUTS,
        "region_validation": region_validation,
        "blocks": block_results,
        "objective_validation_passed": objective_passed,
        "visualization_path": str(PANEL_PATH),
        **runtime_record,
    }
    METRICS_PATH.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                "target_spans": target_spans,
                "grid_shape": list(EXPECTED_GRID_SHAPE),
                "region_validation": region_validation,
                "probe": {"denoising_index": PROBE_DENOISING_INDEX, "timestep": timestep},
                "blocks": {
                    block: {
                        "layout_a_energy": result["layout_a"]["total_energy"],
                        "layout_b_energy": result["layout_b"]["total_energy"],
                        **result["comparison"],
                    }
                    for block, result in block_results.items()
                },
                "objective_validation_passed": objective_passed,
                "probe_non_invasive": runtime_record["probe_non_invasive"],
                "runtime_seconds": runtime_record["runtime_seconds"],
                "peak_cuda_allocated_mib": runtime_record["peak_cuda_allocated_mib"],
                "metrics_path": str(METRICS_PATH),
                "visualization_path": str(PANEL_PATH),
            },
            indent=2,
        )
    )
    pipe.maybe_free_model_hooks()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
