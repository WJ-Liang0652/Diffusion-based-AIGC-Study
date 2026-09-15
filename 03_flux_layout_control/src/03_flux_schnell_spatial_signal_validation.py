"""Minimal spatial-signal validation for FLUX.1-schnell / Diffusers 0.32.2.

For selected Double and Single Stream blocks at three denoising timesteps, this
script extracts two image-query -> target-text-key signals from the exact
RMS-normalized, RoPE-applied Q/K used by FLUX:

1. native joint-normalized attention probability (all text + image keys), and
2. a diagnostic text-only renormalization (all text keys only).

Only selected probability columns are retained. Full joint attention matrices
are never materialized or saved.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from itertools import combinations
from pathlib import Path

import diffusers
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import FluxPipeline
from diffusers.models.embeddings import apply_rotary_emb


MODEL_ID = "black-forest-labs/FLUX.1-schnell"
PROMPT = "a red apple and a blue cup, realistic photo"
SEED = 42
WIDTH = 768
HEIGHT = 768
NUM_INFERENCE_STEPS = 4
GUIDANCE_SCALE = 0.0
DTYPE = torch.bfloat16
OFFLOAD_STRATEGY = "enable_sequential_cpu_offload"

TARGET_WORDS = ("apple", "cup")
PROBE_DENOISING_INDICES = (0, 2, 3)
DOUBLE_BLOCK_INDICES = (0, 9, 18)
SINGLE_BLOCK_INDICES = (0, 18, 37)
KEY_CHUNK_SIZE = 128
TEXT_TOKEN_COUNT = 512

# (x0, y0, x1, y1) on the 48x48 token grid. These are approximate diagnostic
# annotations from the fixed generated baseline, not desired layout boxes.
REFERENCE_BOXES = {
    "apple": (6, 23, 25, 44),
    "cup": (25, 20, 48, 44),
}

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs" / "spatial_validation"


def pixel_sha256(image) -> str:
    return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def locate_target_spans(tokenizer, prompt: str) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    encoded = tokenizer(
        prompt,
        padding="max_length",
        max_length=TEXT_TOKEN_COUNT,
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    token_ids = encoded.input_ids[0].tolist()
    offsets = encoded.offset_mapping[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    non_padding_tokens = []
    for index, (token_id, token, offset) in enumerate(zip(token_ids, tokens, offsets)):
        if token_id != tokenizer.pad_token_id:
            non_padding_tokens.append(
                {"index": index, "id": token_id, "token": token, "character_offset": offset}
            )

    lower_prompt = prompt.lower()
    spans: dict[str, dict[str, object]] = {}
    for word in TARGET_WORDS:
        character_start = lower_prompt.index(word)
        character_end = character_start + len(word)
        indices = [
            index
            for index, (start, end) in enumerate(offsets)
            if start < character_end and end > character_start and end > start
        ]
        if not indices:
            raise RuntimeError(f"No T5 tokens overlap target word {word!r}")
        ids = [token_ids[index] for index in indices]
        spans[word] = {
            "character_span": [character_start, character_end],
            "token_indices": indices,
            "token_ids": ids,
            "token_pieces": [tokens[index] for index in indices],
            "decoded_span": tokenizer.decode(ids, skip_special_tokens=True),
            "is_single_token": len(indices) == 1,
        }
    return spans, non_padding_tokens


class SpatialSignalProbe:
    """Non-invasive wrapper around the installed FluxAttnProcessor2_0."""

    def __init__(self, base_processor, stream: str, block_index: int, target_spans: dict[str, dict[str, object]]):
        self.base_processor = base_processor
        self.stream = stream
        self.block_index = block_index
        self.target_spans = target_spans
        self.call_index = 0
        self.maps: dict[tuple[int, str, str], torch.Tensor] = {}
        self.qk_metadata: dict[str, object] = {}

    @property
    def name(self) -> str:
        return f"{self.stream}_block_{self.block_index}"

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        if self.call_index in PROBE_DENOISING_INDICES:
            self._capture(attn, hidden_states, encoder_hidden_states, image_rotary_emb, self.call_index)
        self.call_index += 1
        return self.base_processor(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )

    @torch.no_grad()
    def _capture(self, attn, hidden_states, encoder_hidden_states, image_rotary_emb, denoising_index: int) -> None:
        batch_size = hidden_states.shape[0]
        heads = attn.heads
        image_query = attn.to_q(hidden_states)
        image_key = attn.to_k(hidden_states)
        head_dim = image_key.shape[-1] // heads

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, -1, heads, head_dim).transpose(1, 2)

        image_query = split_heads(image_query)
        image_key = split_heads(image_key)
        if attn.norm_q is not None:
            image_query = attn.norm_q(image_query)
        if attn.norm_k is not None:
            image_key = attn.norm_k(image_key)

        if encoder_hidden_states is not None:
            text_query = split_heads(attn.add_q_proj(encoder_hidden_states))
            text_key = split_heads(attn.add_k_proj(encoder_hidden_states))
            if attn.norm_added_q is not None:
                text_query = attn.norm_added_q(text_query)
            if attn.norm_added_k is not None:
                text_key = attn.norm_added_k(text_key)
            query = torch.cat([text_query, image_query], dim=2)
            key = torch.cat([text_key, image_key], dim=2)
            projection_organization = "separate text added-QK and image QK, concatenated [text,image]"
        else:
            query, key = image_query, image_key
            projection_organization = "shared QK over already concatenated [text,image] tokens"

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        total_token_count = key.shape[2]
        image_token_count = query.shape[2] - TEXT_TOKEN_COUNT
        grid_size = math.isqrt(image_token_count)
        if grid_size * grid_size != image_token_count:
            raise RuntimeError(f"Image token count {image_token_count} is not square")

        image_queries = query[:, :, TEXT_TOKEN_COUNT:, :].float()
        scale = head_dim**-0.5
        joint_log_denominator = None
        text_log_denominator = None
        for start in range(0, total_token_count, KEY_CHUNK_SIZE):
            end = min(start + KEY_CHUNK_SIZE, total_token_count)
            logits = torch.matmul(image_queries, key[:, :, start:end, :].float().transpose(-1, -2)) * scale
            chunk_logsumexp = torch.logsumexp(logits, dim=-1)
            joint_log_denominator = (
                chunk_logsumexp
                if joint_log_denominator is None
                else torch.logaddexp(joint_log_denominator, chunk_logsumexp)
            )
            if start < TEXT_TOKEN_COUNT:
                text_end = min(end, TEXT_TOKEN_COUNT)
                text_logits = logits[..., : text_end - start]
                text_chunk_logsumexp = torch.logsumexp(text_logits, dim=-1)
                text_log_denominator = (
                    text_chunk_logsumexp
                    if text_log_denominator is None
                    else torch.logaddexp(text_log_denominator, text_chunk_logsumexp)
                )

        for word, span in self.target_spans.items():
            token_indices = span["token_indices"]
            selected_keys = key[:, :, token_indices, :].float()
            selected_logits = torch.matmul(image_queries, selected_keys.transpose(-1, -2)) * scale
            joint_probability = torch.exp(selected_logits - joint_log_denominator.unsqueeze(-1)).sum(dim=-1)
            text_only_probability = torch.exp(selected_logits - text_log_denominator.unsqueeze(-1)).sum(dim=-1)
            self.maps[(denoising_index, "joint", word)] = (
                joint_probability[0].mean(dim=0).reshape(grid_size, grid_size).cpu()
            )
            self.maps[(denoising_index, "text_only", word)] = (
                text_only_probability[0].mean(dim=0).reshape(grid_size, grid_size).cpu()
            )

        if not self.qk_metadata:
            full_elements = batch_size * heads * total_token_count * total_token_count
            self.qk_metadata = {
                "base_processor": type(self.base_processor).__name__,
                "heads": heads,
                "head_dim": head_dim,
                "text_tokens": TEXT_TOKEN_COUNT,
                "image_tokens": image_token_count,
                "joint_tokens": total_token_count,
                "joint_qk_shape": list(query.shape),
                "projection_organization": projection_organization,
                "full_joint_probability_bf16_mib": full_elements * 2 / 2**20,
                "key_chunk_size": KEY_CHUNK_SIZE,
            }


def spatial_statistics(signal_map: torch.Tensor, reference_box: tuple[int, int, int, int]) -> dict[str, object]:
    array = signal_map.double().numpy()
    total = float(array.sum())
    distribution = array / total
    peak_y, peak_x = np.unravel_index(np.argmax(array), array.shape)
    y_grid, x_grid = np.indices(array.shape)
    center_x = float((distribution * x_grid).sum())
    center_y = float((distribution * y_grid).sum())
    entropy = float(-(distribution * np.log(np.maximum(distribution, 1e-300))).sum())
    normalized_entropy = entropy / math.log(array.size)

    x0, y0, x1, y1 = reference_box
    box_mass = float(distribution[y0:y1, x0:x1].sum())
    area_fraction = (x1 - x0) * (y1 - y0) / array.size
    box_center_x = (x0 + x1 - 1) / 2
    box_center_y = (y0 + y1 - 1) / 2
    return {
        "raw_min": float(array.min()),
        "raw_max": float(array.max()),
        "raw_mean": float(array.mean()),
        "peak_xy": [int(peak_x), int(peak_y)],
        "peak_in_reference_box": bool(x0 <= peak_x < x1 and y0 <= peak_y < y1),
        "center_of_mass_xy": [center_x, center_y],
        "center_distance_to_reference_box_center": math.hypot(center_x - box_center_x, center_y - box_center_y),
        "normalized_spatial_entropy": normalized_entropy,
        "spatial_concentration_1_minus_entropy": 1 - normalized_entropy,
        "reference_box_mass": box_mass,
        "reference_box_area_fraction": area_fraction,
        "reference_box_enrichment_over_uniform": box_mass / area_fraction,
    }


def save_panel(probe: SpatialSignalProbe, denoising_index: int, timestep: float, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10, 9))
    for row, normalization in enumerate(("joint", "text_only")):
        for column, word in enumerate(TARGET_WORDS):
            axis = axes[row, column]
            array = probe.maps[(denoising_index, normalization, word)].float().numpy()
            rendered = axis.imshow(array, cmap="magma", interpolation="nearest")
            x0, y0, x1, y1 = REFERENCE_BOXES[word]
            axis.add_patch(
                plt.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, edgecolor="cyan", linewidth=1)
            )
            axis.set_title(f"{normalization}: {word}")
            axis.set_xlabel("image-token x")
            axis.set_ylabel("image-token y")
            figure.colorbar(rendered, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(f"{probe.name}, denoise={denoising_index}, t={timestep:g}")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def pearson_correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    first_array = first.double().flatten().numpy()
    second_array = second.double().flatten().numpy()
    return float(np.corrcoef(first_array, second_array)[0, 1])


def summarize(records: list[dict[str, object]], probes: list[SpatialSignalProbe]) -> dict[str, object]:
    aggregate = {}
    for stream in ("double", "single"):
        for normalization in ("joint", "text_only"):
            selected = [
                record
                for record in records
                if record["stream"] == stream and record["normalization"] == normalization
            ]
            aggregate[f"{stream}:{normalization}"] = {
                "mean_reference_box_enrichment": float(
                    np.mean([record["statistics"]["reference_box_enrichment_over_uniform"] for record in selected])
                ),
                "peak_in_reference_box_rate": float(
                    np.mean([record["statistics"]["peak_in_reference_box"] for record in selected])
                ),
                "mean_center_distance": float(
                    np.mean([record["statistics"]["center_distance_to_reference_box_center"] for record in selected])
                ),
                "mean_spatial_concentration": float(
                    np.mean([record["statistics"]["spatial_concentration_1_minus_entropy"] for record in selected])
                ),
            }

    stability = {}
    for stream in ("double", "single"):
        stream_probes = [probe for probe in probes if probe.stream == stream]
        for normalization in ("joint", "text_only"):
            for word in TARGET_WORDS:
                maps = [
                    probe.maps[(denoising_index, normalization, word)]
                    for probe in stream_probes
                    for denoising_index in PROBE_DENOISING_INDICES
                ]
                correlations = [pearson_correlation(first, second) for first, second in combinations(maps, 2)]
                stability[f"{stream}:{normalization}:{word}"] = {
                    "mean_pairwise_pearson": float(np.mean(correlations)),
                    "min_pairwise_pearson": float(np.min(correlations)),
                    "map_count": len(maps),
                }
    return {"reference_alignment": aggregate, "cross_block_timestep_stability": stability}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this FLUX validation.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE, local_files_only=True)
    target_spans, non_padding_tokens = locate_target_spans(pipe.tokenizer_2, PROMPT)
    if len(pipe.transformer.transformer_blocks) != 19 or len(pipe.transformer.single_transformer_blocks) != 38:
        raise RuntimeError("Unexpected FLUX block counts; reselect representative indices for this checkpoint")

    probes: list[SpatialSignalProbe] = []
    for stream, block_indices, blocks in (
        ("double", DOUBLE_BLOCK_INDICES, pipe.transformer.transformer_blocks),
        ("single", SINGLE_BLOCK_INDICES, pipe.transformer.single_transformer_blocks),
    ):
        for block_index in block_indices:
            attention = blocks[block_index].attn
            probe = SpatialSignalProbe(attention.get_processor(), stream, block_index, target_spans)
            attention.set_processor(probe)
            probes.append(probe)

    pipe.enable_sequential_cpu_offload()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    start = time.perf_counter()
    image = pipe(
        prompt=PROMPT,
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        num_images_per_prompt=1,
        generator=generator,
    ).images[0]
    runtime_seconds = time.perf_counter() - start
    generation_path = OUTPUT_DIR / "flux-1-schnell_spatial-validation_seed-42_steps-4.png"
    image.save(generation_path)
    timesteps = [float(value) for value in pipe.scheduler.timesteps.cpu().tolist()]

    map_records = []
    panel_paths = []
    for probe in probes:
        expected_keys = {
            (index, normalization, word)
            for index in PROBE_DENOISING_INDICES
            for normalization in ("joint", "text_only")
            for word in TARGET_WORDS
        }
        if set(probe.maps) != expected_keys:
            raise RuntimeError(f"Incomplete capture for {probe.name}: got {len(probe.maps)}, expected {len(expected_keys)}")
        for denoising_index in PROBE_DENOISING_INDICES:
            timestep = timesteps[denoising_index]
            panel_path = OUTPUT_DIR / f"{probe.name}_denoise-{denoising_index}_t-{timestep:g}_signals.png"
            save_panel(probe, denoising_index, timestep, panel_path)
            panel_paths.append(str(panel_path))
            for normalization in ("joint", "text_only"):
                for word in TARGET_WORDS:
                    signal_map = probe.maps[(denoising_index, normalization, word)]
                    map_records.append(
                        {
                            "stream": probe.stream,
                            "block_index": probe.block_index,
                            "denoising_index": denoising_index,
                            "timestep": timestep,
                            "normalization": normalization,
                            "word": word,
                            "token_indices": target_spans[word]["token_indices"],
                            "shape": list(signal_map.shape),
                            "statistics": spatial_statistics(signal_map, REFERENCE_BOXES[word]),
                        }
                    )

    record = {
        "model_id": MODEL_ID,
        "diffusers_version": diffusers.__version__,
        "torch_version": torch.__version__,
        "prompt": PROMPT,
        "seed": SEED,
        "width": WIDTH,
        "height": HEIGHT,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "dtype": str(DTYPE),
        "offload_strategy": OFFLOAD_STRATEGY,
        "target_spans": target_spans,
        "t5_non_padding_token_sequence": non_padding_tokens,
        "text_key_denominator_note": "all 512 model text positions, including padded positions used by FLUX",
        "signals": {
            "joint": "native P(target text-key span | image query), denominator is all 512 text + 2304 image keys",
            "text_only": "diagnostic renormalization using the same FLUX Q/K logits but only 512 text keys; not native FLUX attention probability",
            "target_span_aggregation": "sum token probabilities within each target word span, then mean over heads",
        },
        "selected_denoising_indices": list(PROBE_DENOISING_INDICES),
        "actual_timesteps": timesteps,
        "selected_double_blocks": list(DOUBLE_BLOCK_INDICES),
        "selected_single_blocks": list(SINGLE_BLOCK_INDICES),
        "reference_boxes_48x48": {
            word: {"xyxy": list(box), "role": "manual diagnostic annotation, not a layout target"}
            for word, box in REFERENCE_BOXES.items()
        },
        "probe_qk_metadata": {probe.name: probe.qk_metadata for probe in probes},
        "map_records": map_records,
        "summary": summarize(map_records, probes),
        "panel_paths": panel_paths,
        "generation_path": str(generation_path),
        "generation_pixel_sha256": pixel_sha256(image),
        "runtime_seconds": runtime_seconds,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }
    metrics_path = OUTPUT_DIR / "flux-1-schnell_spatial-signal-validation_metrics.json"
    metrics_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: record[key] for key in ("target_spans", "summary", "generation_pixel_sha256", "runtime_seconds", "peak_cuda_allocated_mib", "peak_cuda_reserved_mib")}, indent=2))

    pipe.maybe_free_model_hooks()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
