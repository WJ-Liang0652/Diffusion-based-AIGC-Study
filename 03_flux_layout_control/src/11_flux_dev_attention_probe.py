"""Minimal FLUX.1-dev image-query -> text-key attention probe.

The attention definition and aggregation match the validated schnell probe:
RMS-normalized and RoPE-applied Q/K, native joint softmax over all text and
image keys, target-token-span probability sum, then head mean.  Only selected
columns are materialized.  The official Stage-1 Pipeline sampling path and its
saved initial packed latents are reused unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import diffusers
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import FluxPipeline
from diffusers.models.embeddings import apply_rotary_emb
from PIL import Image


MODEL_ID = "black-forest-labs/FLUX.1-dev"
PROMPT = "a red apple and a blue cup, realistic photo"
SEED = 42
WIDTH = 768
HEIGHT = 768
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 3.5
DTYPE = torch.bfloat16
OFFLOAD_STRATEGY = "enable_sequential_cpu_offload"

TARGET_WORDS = ("apple", "cup")
TEXT_TOKEN_COUNT = 512
KEY_CHUNK_SIZE = 128
PROBE_DENOISING_INDEX = 35
DOUBLE_BLOCK_INDEX = 18

# Same 48x48 diagnostic boxes used by the schnell spatial validation.  They
# describe the observed fixed-seed apple-left/cup-right baseline, not a target
# layout supplied to the model.
REFERENCE_BOXES = {
    "apple": (6, 23, 25, 44),
    "cup": (25, 20, 48, 44),
}

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs" / "attention"
BASELINE_STATE_PATH = PROJECT_DIR / "outputs" / "baseline" / "flux_dev_baseline_state.pt"
BASELINE_METRICS_PATH = PROJECT_DIR / "outputs" / "baseline" / "flux_dev_baseline_metrics.json"


def pixel_sha256(image: Image.Image) -> str:
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


class DevSpatialAttentionProbe:
    """Delegate normal attention while capturing selected Double Stream maps."""

    def __init__(
        self,
        base_processor,
        block_index: int,
        target_spans: dict[str, dict[str, object]],
        capture_indices: tuple[int, ...],
    ):
        self.base_processor = base_processor
        self.block_index = block_index
        self.target_spans = target_spans
        self.capture_indices = capture_indices
        self.call_index = 0
        self.maps: dict[tuple[int, str, str], torch.Tensor] = {}
        self.qk_metadata: dict[str, object] = {}

    @property
    def name(self) -> str:
        return f"double_block_{self.block_index}"

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        if self.call_index in self.capture_indices:
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
        if encoder_hidden_states is None:
            raise RuntimeError("The dev probe expects a Double Stream block")

        batch_size = hidden_states.shape[0]
        heads = attn.heads
        image_query = attn.to_q(hidden_states)
        image_key = attn.to_k(hidden_states)
        text_query = attn.add_q_proj(encoder_hidden_states)
        text_key = attn.add_k_proj(encoder_hidden_states)
        head_dim = image_key.shape[-1] // heads

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, -1, heads, head_dim).transpose(1, 2)

        image_query = split_heads(image_query)
        image_key = split_heads(image_key)
        text_query = split_heads(text_query)
        text_key = split_heads(text_key)
        if attn.norm_q is not None:
            image_query = attn.norm_q(image_query)
        if attn.norm_k is not None:
            image_key = attn.norm_k(image_key)
        if attn.norm_added_q is not None:
            text_query = attn.norm_added_q(text_query)
        if attn.norm_added_k is not None:
            text_key = attn.norm_added_k(text_key)

        query = torch.cat([text_query, image_query], dim=2)
        key = torch.cat([text_key, image_key], dim=2)
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
                text_chunk_logsumexp = torch.logsumexp(logits[..., : text_end - start], dim=-1)
                text_log_denominator = (
                    text_chunk_logsumexp
                    if text_log_denominator is None
                    else torch.logaddexp(text_log_denominator, text_chunk_logsumexp)
                )

        for word, span in self.target_spans.items():
            selected_keys = key[:, :, span["token_indices"], :].float()
            selected_logits = torch.matmul(image_queries, selected_keys.transpose(-1, -2)) * scale
            for normalization, denominator in (
                ("joint", joint_log_denominator),
                ("text_only", text_log_denominator),
            ):
                probability = torch.exp(selected_logits - denominator.unsqueeze(-1)).sum(dim=-1)
                self.maps[(denoising_index, normalization, word)] = (
                    probability[0].mean(dim=0).reshape(grid_size, grid_size).cpu()
                )

        if not self.qk_metadata:
            full_elements = batch_size * heads * total_token_count * total_token_count
            self.qk_metadata = {
                "base_processor": type(self.base_processor).__name__,
                "stream": "double",
                "block_index": self.block_index,
                "heads": heads,
                "head_dim": head_dim,
                "text_tokens": TEXT_TOKEN_COUNT,
                "image_tokens": image_token_count,
                "joint_tokens": total_token_count,
                "joint_qk_shape": list(query.shape),
                "projection_organization": "separate text added-QK and image QK, concatenated [text,image]",
                "full_joint_probability_bf16_mib": full_elements * 2 / 2**20,
                "key_chunk_size": KEY_CHUNK_SIZE,
                "attention_direction": "image query -> text key",
                "target_aggregation": "sum over target token span, then mean over heads",
            }


def spatial_statistics(signal_map: torch.Tensor, word: str) -> dict[str, object]:
    array = signal_map.double().numpy()
    total = float(array.sum())
    if not math.isfinite(total) or total <= 0:
        raise RuntimeError(f"Invalid {word} attention-map total: {total}")
    distribution = array / total
    peak_y, peak_x = np.unravel_index(np.argmax(array), array.shape)
    y_grid, x_grid = np.indices(array.shape)
    center_x = float((distribution * x_grid).sum())
    center_y = float((distribution * y_grid).sum())
    entropy = float(-(distribution * np.log(np.maximum(distribution, 1e-300))).sum())

    x0, y0, x1, y1 = REFERENCE_BOXES[word]
    other_word = "cup" if word == "apple" else "apple"
    ox0, oy0, ox1, oy1 = REFERENCE_BOXES[other_word]
    target_mass = float(distribution[y0:y1, x0:x1].sum())
    competing_mass = float(distribution[oy0:oy1, ox0:ox1].sum())
    area_fraction = (x1 - x0) * (y1 - y0) / array.size
    left_mass = float(distribution[:, : array.shape[1] // 2].sum())
    right_mass = float(distribution[:, array.shape[1] // 2 :].sum())
    return {
        "raw_min": float(array.min()),
        "raw_max": float(array.max()),
        "raw_mean": float(array.mean()),
        "peak_xy": [int(peak_x), int(peak_y)],
        "peak_in_reference_box": bool(x0 <= peak_x < x1 and y0 <= peak_y < y1),
        "center_of_mass_xy": [center_x, center_y],
        "normalized_spatial_entropy": entropy / math.log(array.size),
        "spatial_concentration_1_minus_entropy": 1 - entropy / math.log(array.size),
        "reference_box_mass": target_mass,
        "reference_box_area_fraction": area_fraction,
        "reference_box_enrichment_over_uniform": target_mass / area_fraction,
        "competing_object_box_mass": competing_mass,
        "target_to_competing_box_mass_ratio": target_mass / max(competing_mass, 1e-300),
        "left_half_mass": left_mass,
        "right_half_mass": right_mass,
    }


def save_heatmap(signal_map: torch.Tensor, word: str, normalization: str, title: str, path: Path) -> None:
    array = signal_map.float().numpy()
    figure, axis = plt.subplots(figsize=(5.5, 5))
    rendered = axis.imshow(array, cmap="magma", interpolation="nearest")
    x0, y0, x1, y1 = REFERENCE_BOXES[word]
    axis.add_patch(
        plt.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, edgecolor="cyan", linewidth=1.3)
    )
    axis.set_title(title)
    axis.set_xlabel("image-token x")
    axis.set_ylabel("image-token y")
    figure.colorbar(rendered, ax=axis, label=f"{normalization} head-mean probability")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_overlay(image: Image.Image, signal_map: torch.Tensor, title: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 7))
    axis.imshow(image.convert("RGB"))
    rendered = axis.imshow(
        signal_map.float().numpy(),
        cmap="magma",
        interpolation="bilinear",
        alpha=0.5,
        extent=(0, image.width, image.height, 0),
    )
    axis.set_title(title)
    axis.axis("off")
    figure.colorbar(rendered, ax=axis, fraction=0.046, pad=0.04, label="joint attention probability")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def load_baseline_state() -> tuple[dict[str, object], dict[str, object]]:
    if not BASELINE_STATE_PATH.exists() or not BASELINE_METRICS_PATH.exists():
        raise FileNotFoundError("Stage 1 dev baseline state/metrics are required")
    state = torch.load(BASELINE_STATE_PATH, map_location="cpu", weights_only=False)
    metrics = json.loads(BASELINE_METRICS_PATH.read_text())
    expected = {
        "model_id": MODEL_ID,
        "prompt": PROMPT,
        "seed": SEED,
        "width": WIDTH,
        "height": HEIGHT,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
    }
    for key, value in expected.items():
        if state[key] != value:
            raise RuntimeError(f"Stage 1 state mismatch for {key}: {state[key]} != {value}")
    return state, metrics


def run_pipeline_with_probes(
    pipe: FluxPipeline,
    probes: list[DevSpatialAttentionProbe],
    baseline_state: dict[str, object],
    baseline_metrics: dict[str, object],
) -> tuple[Image.Image, list[float], dict[str, float | bool | str]]:
    pipe.enable_sequential_cpu_offload()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    image = pipe(
        prompt=PROMPT,
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        num_images_per_prompt=1,
        generator=torch.Generator(device="cpu").manual_seed(SEED),
        latents=baseline_state["initial_latents"].clone(),
    ).images[0]
    runtime = time.perf_counter() - start
    timesteps = [float(value) for value in pipe.scheduler.timesteps.detach().cpu().tolist()]

    for probe in probes:
        pipe.transformer.transformer_blocks[probe.block_index].attn.set_processor(probe.base_processor)

    image_hash = pixel_sha256(image)
    baseline_hash = baseline_metrics["pixel_sha256"]
    non_invasive = image_hash == baseline_hash
    if not non_invasive:
        raise RuntimeError(f"Attention probe changed generation: {image_hash} != {baseline_hash}")
    runtime_record = {
        "runtime_seconds": runtime,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "generation_pixel_sha256": image_hash,
        "baseline_pixel_sha256": baseline_hash,
        "probe_non_invasive": non_invasive,
    }
    return image, timesteps, runtime_record


def evaluate_object_pair(records: dict[str, dict[str, object]]) -> dict[str, object]:
    apple = records["apple"]
    cup = records["cup"]
    apple_preferred = apple["reference_box_mass"] > apple["competing_object_box_mass"]
    cup_preferred = cup["reference_box_mass"] > cup["competing_object_box_mass"]
    ordered = apple["center_of_mass_xy"][0] < cup["center_of_mass_xy"][0]
    enriched = (
        apple["reference_box_enrichment_over_uniform"] > 1.2
        and cup["reference_box_enrichment_over_uniform"] > 1.2
    )
    return {
        "apple_target_box_preferred": apple_preferred,
        "cup_target_box_preferred": cup_preferred,
        "apple_center_left_of_cup": ordered,
        "both_reference_boxes_enriched_over_1p2": enriched,
        "reliable_spatial_signal": bool(apple_preferred and cup_preferred and ordered and enriched),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this FLUX attention probe")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_state, baseline_metrics = load_baseline_state()
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE, local_files_only=True)
    target_spans, non_padding_tokens = locate_target_spans(pipe.tokenizer_2, PROMPT)
    attention = pipe.transformer.transformer_blocks[DOUBLE_BLOCK_INDEX].attn
    probe = DevSpatialAttentionProbe(
        attention.get_processor(), DOUBLE_BLOCK_INDEX, target_spans, (PROBE_DENOISING_INDEX,)
    )
    attention.set_processor(probe)

    image, timesteps, runtime_record = run_pipeline_with_probes(
        pipe, [probe], baseline_state, baseline_metrics
    )
    timestep = timesteps[PROBE_DENOISING_INDEX]
    generation_path = OUTPUT_DIR / "flux_dev_attention_probe_seed-42_steps-50.png"
    image.save(generation_path)

    expected_keys = {
        (PROBE_DENOISING_INDEX, normalization, word)
        for normalization in ("joint", "text_only")
        for word in TARGET_WORDS
    }
    if set(probe.maps) != expected_keys:
        raise RuntimeError(f"Incomplete dev attention capture: {sorted(probe.maps)}")

    map_records = []
    output_paths = []
    joint_statistics = {}
    timestep_label = str(round(timestep, 3)).replace(".", "p")
    for normalization in ("joint", "text_only"):
        for word in TARGET_WORDS:
            signal_map = probe.maps[(PROBE_DENOISING_INDEX, normalization, word)]
            statistics = spatial_statistics(signal_map, word)
            heatmap_path = OUTPUT_DIR / (
                f"flux_dev_double_block_18_denoise-35_t-{timestep_label}_{word}_{normalization}_heatmap.png"
            )
            save_heatmap(
                signal_map,
                word,
                normalization,
                f"dev Double block 18, t={timestep:.3f}: {word} ({normalization})",
                heatmap_path,
            )
            output_paths.append(str(heatmap_path))
            if normalization == "joint":
                joint_statistics[word] = statistics
                overlay_path = OUTPUT_DIR / (
                    f"flux_dev_double_block_18_denoise-35_t-{timestep_label}_{word}_joint_overlay.png"
                )
                save_overlay(
                    image,
                    signal_map,
                    f"dev Double block 18, t={timestep:.3f}: {word}",
                    overlay_path,
                )
                output_paths.append(str(overlay_path))
            map_records.append(
                {
                    "block_index": DOUBLE_BLOCK_INDEX,
                    "denoising_index": PROBE_DENOISING_INDEX,
                    "timestep": timestep,
                    "normalization": normalization,
                    "word": word,
                    "token_indices": target_spans[word]["token_indices"],
                    "shape": list(signal_map.shape),
                    "statistics": statistics,
                    "heatmap_path": str(heatmap_path),
                }
            )

    signal_evaluation = evaluate_object_pair(joint_statistics)
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
        "probe_block": DOUBLE_BLOCK_INDEX,
        "probe_denoising_index": PROBE_DENOISING_INDEX,
        "probe_timestep": timestep,
        "attention_definition": "native joint-normalized P(target text-key span | image query)",
        "target_span_aggregation": "sum token probabilities within span, then mean over heads",
        "diagnostic_signal": "text-only renormalization over the same FLUX Q/K logits",
        "reference_boxes_48x48": {word: list(box) for word, box in REFERENCE_BOXES.items()},
        "qk_metadata": probe.qk_metadata,
        "map_records": map_records,
        "joint_signal_evaluation": signal_evaluation,
        "generation_path": str(generation_path),
        "output_paths": output_paths,
        **runtime_record,
    }
    metrics_path = OUTPUT_DIR / "flux_dev_attention_probe_metrics.json"
    metrics_path.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                "target_spans": target_spans,
                "probe": {"block": DOUBLE_BLOCK_INDEX, "index": PROBE_DENOISING_INDEX, "timestep": timestep},
                "joint_statistics": joint_statistics,
                "signal_evaluation": signal_evaluation,
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
