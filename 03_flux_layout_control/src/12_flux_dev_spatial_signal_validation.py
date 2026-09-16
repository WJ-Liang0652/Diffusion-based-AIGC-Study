"""Validate representative FLUX.1-dev Double Stream spatial signals.

This keeps the schnell experiment's selected blocks, approximate canonical
timesteps, attention definition, target-span aggregation, and diagnostic boxes.
Only Double blocks 0/9/18 and three denoising states near t=1000/500/250 are
captured; no all-block or all-timestep scan is performed.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import diffusers
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import FluxPipeline


PROJECT_DIR = Path(__file__).resolve().parents[1]
HELPER_PATH = Path(__file__).with_name("11_flux_dev_attention_probe.py")
OUTPUT_DIR = PROJECT_DIR / "outputs" / "spatial_validation"
SCHNELL_METRICS_PATH = OUTPUT_DIR / "flux-1-schnell_spatial-signal-validation_metrics.json"

DOUBLE_BLOCK_INDICES = (0, 9, 18)
PROBE_DENOISING_INDICES = (0, 35, 44)
CANONICAL_TIMESTEPS = {0: 1000.0, 35: 500.0, 44: 250.0}


def load_helpers():
    spec = importlib.util.spec_from_file_location("flux_dev_attention_helpers", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load dev attention helpers from {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_panel(helpers, image, probe, denoising_index: int, timestep: float, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    for column, word in enumerate(helpers.TARGET_WORDS):
        signal_map = probe.maps[(denoising_index, "joint", word)].float().numpy()
        heatmap_axis = axes[0, column]
        rendered = heatmap_axis.imshow(signal_map, cmap="magma", interpolation="nearest")
        x0, y0, x1, y1 = helpers.REFERENCE_BOXES[word]
        heatmap_axis.add_patch(
            plt.Rectangle(
                (x0 - 0.5, y0 - 0.5),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor="cyan",
                linewidth=1.3,
            )
        )
        heatmap_axis.set_title(f"{word}: joint heatmap")
        heatmap_axis.set_xlabel("image-token x")
        heatmap_axis.set_ylabel("image-token y")
        figure.colorbar(rendered, ax=heatmap_axis, fraction=0.046, pad=0.04)

        overlay_axis = axes[1, column]
        overlay_axis.imshow(image.convert("RGB"))
        overlay_axis.imshow(
            signal_map,
            cmap="magma",
            interpolation="bilinear",
            alpha=0.5,
            extent=(0, image.width, image.height, 0),
        )
        overlay_axis.set_title(f"{word}: image overlay")
        overlay_axis.axis("off")

    figure.suptitle(
        f"FLUX.1-dev Double block {probe.block_index}, denoise={denoising_index}, t={timestep:.3f}"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def paired_signal_evaluation(helpers, records_by_word: dict[str, dict[str, object]]) -> dict[str, object]:
    return helpers.evaluate_object_pair(records_by_word)


def build_schnell_lookup() -> tuple[dict[tuple[int, float, str], dict[str, object]], dict[str, object]]:
    schnell = json.loads(SCHNELL_METRICS_PATH.read_text())
    lookup = {}
    for record in schnell["map_records"]:
        if record["stream"] == "double" and record["normalization"] == "joint":
            lookup[(record["block_index"], float(record["timestep"]), record["word"])] = record["statistics"]
    return lookup, schnell


def main() -> None:
    helpers = load_helpers()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this FLUX spatial validation")
    if not SCHNELL_METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing schnell reference metrics: {SCHNELL_METRICS_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_state, baseline_metrics = helpers.load_baseline_state()
    schnell_lookup, schnell_metrics = build_schnell_lookup()
    pipe = FluxPipeline.from_pretrained(helpers.MODEL_ID, torch_dtype=helpers.DTYPE, local_files_only=True)
    target_spans, non_padding_tokens = helpers.locate_target_spans(pipe.tokenizer_2, helpers.PROMPT)
    if len(pipe.transformer.transformer_blocks) != 19:
        raise RuntimeError("Unexpected FLUX Double Stream block count")

    probes = []
    for block_index in DOUBLE_BLOCK_INDICES:
        attention = pipe.transformer.transformer_blocks[block_index].attn
        probe = helpers.DevSpatialAttentionProbe(
            attention.get_processor(), block_index, target_spans, PROBE_DENOISING_INDICES
        )
        attention.set_processor(probe)
        probes.append(probe)

    image, timesteps, runtime_record = helpers.run_pipeline_with_probes(
        pipe, probes, baseline_state, baseline_metrics
    )
    generation_path = OUTPUT_DIR / "flux_dev_spatial_validation_seed-42_steps-50.png"
    image.save(generation_path)

    map_records = []
    config_summaries = []
    panel_paths = []
    direct_schnell_comparisons = []
    for probe in probes:
        expected_keys = {
            (index, normalization, word)
            for index in PROBE_DENOISING_INDICES
            for normalization in ("joint", "text_only")
            for word in helpers.TARGET_WORDS
        }
        if set(probe.maps) != expected_keys:
            raise RuntimeError(
                f"Incomplete capture for {probe.name}: got {len(probe.maps)}, expected {len(expected_keys)}"
            )

        for denoising_index in PROBE_DENOISING_INDICES:
            timestep = timesteps[denoising_index]
            canonical_timestep = CANONICAL_TIMESTEPS[denoising_index]
            timestep_label = str(round(timestep, 3)).replace(".", "p")
            panel_path = OUTPUT_DIR / (
                f"flux_dev_double_block_{probe.block_index}_denoise-{denoising_index}_"
                f"t-{timestep_label}_joint_signals.png"
            )
            save_panel(helpers, image, probe, denoising_index, timestep, panel_path)
            panel_paths.append(str(panel_path))

            joint_by_word = {}
            for normalization in ("joint", "text_only"):
                for word in helpers.TARGET_WORDS:
                    signal_map = probe.maps[(denoising_index, normalization, word)]
                    statistics = helpers.spatial_statistics(signal_map, word)
                    if normalization == "joint":
                        joint_by_word[word] = statistics
                        schnell_statistics = schnell_lookup[(probe.block_index, canonical_timestep, word)]
                        direct_schnell_comparisons.append(
                            {
                                "block_index": probe.block_index,
                                "dev_denoising_index": denoising_index,
                                "dev_timestep": timestep,
                                "schnell_timestep": canonical_timestep,
                                "word": word,
                                "dev_reference_box_enrichment": statistics[
                                    "reference_box_enrichment_over_uniform"
                                ],
                                "schnell_reference_box_enrichment": schnell_statistics[
                                    "reference_box_enrichment_over_uniform"
                                ],
                                "dev_peak_in_reference_box": statistics["peak_in_reference_box"],
                                "schnell_peak_in_reference_box": schnell_statistics["peak_in_reference_box"],
                                "dev_center_of_mass_xy": statistics["center_of_mass_xy"],
                                "schnell_center_of_mass_xy": schnell_statistics["center_of_mass_xy"],
                            }
                        )
                    map_records.append(
                        {
                            "stream": "double",
                            "block_index": probe.block_index,
                            "denoising_index": denoising_index,
                            "timestep": timestep,
                            "canonical_comparison_timestep": canonical_timestep,
                            "normalization": normalization,
                            "word": word,
                            "token_indices": target_spans[word]["token_indices"],
                            "shape": list(signal_map.shape),
                            "statistics": statistics,
                        }
                    )

            evaluation = paired_signal_evaluation(helpers, joint_by_word)
            config_summaries.append(
                {
                    "block_index": probe.block_index,
                    "denoising_index": denoising_index,
                    "timestep": timestep,
                    "canonical_comparison_timestep": canonical_timestep,
                    "joint_statistics": joint_by_word,
                    "evaluation": evaluation,
                    "panel_path": str(panel_path),
                }
            )

    mid_late = [
        item
        for item in config_summaries
        if item["block_index"] in (9, 18) and item["canonical_comparison_timestep"] in (500.0, 250.0)
    ]
    reliable_mid_late = [item for item in mid_late if item["evaluation"]["reliable_spatial_signal"]]
    block0_mean_enrichment = float(
        np.mean(
            [
                stats["reference_box_enrichment_over_uniform"]
                for item in config_summaries
                if item["block_index"] == 0
                for stats in item["joint_statistics"].values()
            ]
        )
    )
    mid_late_mean_enrichment = float(
        np.mean(
            [
                stats["reference_box_enrichment_over_uniform"]
                for item in mid_late
                for stats in item["joint_statistics"].values()
            ]
        )
    )
    token_spans_match_schnell = target_spans == schnell_metrics["target_spans"]
    spatial_signal_established = len(reliable_mid_late) >= 2 and any(
        item["block_index"] == 18 for item in reliable_mid_late
    )
    schnell_profile_consistent = bool(
        token_spans_match_schnell
        and spatial_signal_established
        and mid_late_mean_enrichment > block0_mean_enrichment
    )

    conclusion = {
        "token_spans_match_schnell": token_spans_match_schnell,
        "reliable_mid_late_configuration_count": len(reliable_mid_late),
        "tested_mid_late_configuration_count": len(mid_late),
        "block0_mean_joint_reference_enrichment": block0_mean_enrichment,
        "blocks9_18_mid_late_mean_joint_reference_enrichment": mid_late_mean_enrichment,
        "image_query_to_text_key_spatial_signal_established": spatial_signal_established,
        "qualitative_mechanism_profile_consistent_with_schnell": schnell_profile_consistent,
        "criterion": (
            "at least two block-9/18 mid/late configurations, including block 18, must show both object maps "
            "preferring their observed-object box, apple center left of cup, and >1.2x uniform enrichment"
        ),
    }

    record = {
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
        "target_spans": target_spans,
        "t5_non_padding_token_sequence": non_padding_tokens,
        "attention_definition": "native joint-normalized P(target text-key span | image query)",
        "target_span_aggregation": "sum token probabilities within span, then mean over heads",
        "selected_double_blocks": list(DOUBLE_BLOCK_INDICES),
        "selected_denoising_indices": list(PROBE_DENOISING_INDICES),
        "actual_selected_timesteps": {
            str(index): timesteps[index] for index in PROBE_DENOISING_INDICES
        },
        "canonical_schnell_comparison_timesteps": {
            str(index): value for index, value in CANONICAL_TIMESTEPS.items()
        },
        "reference_boxes_48x48": {word: list(box) for word, box in helpers.REFERENCE_BOXES.items()},
        "probe_qk_metadata": {probe.name: probe.qk_metadata for probe in probes},
        "map_records": map_records,
        "configuration_summaries": config_summaries,
        "direct_schnell_comparisons": direct_schnell_comparisons,
        "conclusion": conclusion,
        "generation_path": str(generation_path),
        "panel_paths": panel_paths,
        **runtime_record,
    }
    metrics_path = OUTPUT_DIR / "flux_dev_spatial_signal_validation_metrics.json"
    metrics_path.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                "target_spans": target_spans,
                "selected_timesteps": record["actual_selected_timesteps"],
                "configuration_summaries": config_summaries,
                "conclusion": conclusion,
                "probe_non_invasive": runtime_record["probe_non_invasive"],
                "runtime_seconds": runtime_record["runtime_seconds"],
                "peak_cuda_allocated_mib": runtime_record["peak_cuda_allocated_mib"],
            },
            indent=2,
        )
    )
    pipe.maybe_free_model_hooks()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
