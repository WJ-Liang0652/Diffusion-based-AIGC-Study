"""Minimal FLUX.1-schnell attention/token-space probe for Diffusers 0.32.2.

This experiment keeps the pipeline output path unchanged and probes only Double
Stream block 0 and Single Stream block 0 at denoising index 1. It computes exact
image-query -> selected-text-key softmax probabilities with a chunked full-key
denominator, without materializing the complete joint attention matrix.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import diffusers
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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

PROBE_DENOISING_INDEX = 1
DOUBLE_BLOCK_INDEX = 0
SINGLE_BLOCK_INDEX = 0
KEY_CHUNK_SIZE = 128
TARGET_WORDS = ("apple", "cup")

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs" / "attention"


def locate_target_tokens(tokenizer, prompt: str) -> tuple[dict[str, int], list[dict[str, int | str]]]:
    encoded = tokenizer(
        prompt,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    token_ids = encoded.input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    target_indices: dict[str, int] = {}
    non_padding_tokens = []
    for index, (token_id, token) in enumerate(zip(token_ids, tokens)):
        if token_id != tokenizer.pad_token_id:
            non_padding_tokens.append({"index": index, "id": token_id, "token": token})
        normalized = token.replace("▁", "").lower()
        for word in TARGET_WORDS:
            if normalized == word and word not in target_indices:
                target_indices[word] = index
    missing = sorted(set(TARGET_WORDS) - set(target_indices))
    if missing:
        raise RuntimeError(f"Could not locate target T5 token(s): {missing}; tokens={non_padding_tokens}")
    return target_indices, non_padding_tokens


class SelectedTokenProbabilityProbe:
    """Delegate normal attention while capturing selected probability columns."""

    def __init__(self, base_processor, name: str, text_token_count: int, target_indices: dict[str, int]):
        self.base_processor = base_processor
        self.name = name
        self.text_token_count = text_token_count
        self.target_indices = target_indices
        self.call_index = 0
        self.maps: dict[str, torch.Tensor] = {}
        self.metadata: dict[str, object] = {}

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        if self.call_index == PROBE_DENOISING_INDEX:
            self._capture(attn, hidden_states, encoder_hidden_states, image_rotary_emb)
        self.call_index += 1
        return self.base_processor(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )

    @torch.no_grad()
    def _capture(self, attn, hidden_states, encoder_hidden_states, image_rotary_emb) -> None:
        batch_size = hidden_states.shape[0]
        heads = attn.heads

        image_query = attn.to_q(hidden_states)
        image_key = attn.to_k(hidden_states)
        image_value = attn.to_v(hidden_states)
        head_dim = image_key.shape[-1] // heads

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, -1, heads, head_dim).transpose(1, 2)

        image_query = split_heads(image_query)
        image_key = split_heads(image_key)
        image_value = split_heads(image_value)
        if attn.norm_q is not None:
            image_query = attn.norm_q(image_query)
        if attn.norm_k is not None:
            image_key = attn.norm_k(image_key)

        if encoder_hidden_states is not None:
            text_query = split_heads(attn.add_q_proj(encoder_hidden_states))
            text_key = split_heads(attn.add_k_proj(encoder_hidden_states))
            text_value = split_heads(attn.add_v_proj(encoder_hidden_states))
            if attn.norm_added_q is not None:
                text_query = attn.norm_added_q(text_query)
            if attn.norm_added_k is not None:
                text_key = attn.norm_added_k(text_key)
            query = torch.cat([text_query, image_query], dim=2)
            key = torch.cat([text_key, image_key], dim=2)
            value = torch.cat([text_value, image_value], dim=2)
            stream_type = "double"
            projection_shapes = {
                "text_qkv": list(text_query.shape),
                "image_qkv": list(image_query.shape),
            }
        else:
            query, key, value = image_query, image_key, image_value
            stream_type = "single"
            projection_shapes = {"joint_qkv": list(query.shape)}

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        total_token_count = key.shape[2]
        image_token_count = query.shape[2] - self.text_token_count
        image_queries = query[:, :, self.text_token_count :, :].float()
        scale = head_dim**-0.5

        # Compute log(sum(exp(qk))) across every text+image key in bounded chunks.
        log_denominator = None
        for start in range(0, total_token_count, KEY_CHUNK_SIZE):
            end = min(start + KEY_CHUNK_SIZE, total_token_count)
            chunk_logits = torch.matmul(image_queries, key[:, :, start:end, :].float().transpose(-1, -2)) * scale
            chunk_logsumexp = torch.logsumexp(chunk_logits, dim=-1)
            log_denominator = (
                chunk_logsumexp
                if log_denominator is None
                else torch.logaddexp(log_denominator, chunk_logsumexp)
            )

        selected_indices = list(self.target_indices.values())
        selected_keys = key[:, :, selected_indices, :].float()
        selected_logits = torch.matmul(image_queries, selected_keys.transpose(-1, -2)) * scale
        selected_probabilities = torch.exp(selected_logits - log_denominator.unsqueeze(-1))

        grid_size = math.isqrt(image_token_count)
        if grid_size * grid_size != image_token_count:
            raise RuntimeError(f"Image token count {image_token_count} is not a square grid")
        for target_offset, word in enumerate(self.target_indices):
            # Head-averaged exact probability P(text key | image query).
            self.maps[word] = selected_probabilities[0, :, :, target_offset].mean(dim=0).reshape(
                grid_size, grid_size
            ).cpu()

        full_elements = batch_size * heads * total_token_count * total_token_count
        self.metadata = {
            "stream_type": stream_type,
            "processor_class": type(self.base_processor).__name__,
            "heads": heads,
            "head_dim": head_dim,
            "text_token_count": self.text_token_count,
            "image_token_count": image_token_count,
            "joint_token_count": total_token_count,
            "joint_qkv": list(query.shape),
            "value": list(value.shape),
            "projection_shapes": projection_shapes,
            "full_attention_elements": full_elements,
            "full_attention_bf16_mib": full_elements * 2 / 2**20,
            "full_attention_fp32_mib": full_elements * 4 / 2**20,
            "extraction": "exact selected softmax columns with chunked full-key denominator",
            "direction": "image queries -> selected text keys",
            "probability_normalization_keys": "all text and image keys",
            "key_chunk_size": KEY_CHUNK_SIZE,
        }


def save_heatmap(probability_map: torch.Tensor, path: Path, title: str) -> dict[str, float]:
    array = probability_map.float().numpy()
    figure, axis = plt.subplots(figsize=(5.5, 5))
    image = axis.imshow(array, cmap="magma", interpolation="nearest")
    axis.set_title(title)
    axis.set_xlabel("image-token x")
    axis.set_ylabel("image-token y")
    figure.colorbar(image, ax=axis, label="head-mean attention probability")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "sum": float(array.sum()),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this FLUX attention probe.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE, local_files_only=True)
    target_indices, non_padding_tokens = locate_target_tokens(pipe.tokenizer_2, PROMPT)
    text_token_count = 512

    double_attention = pipe.transformer.transformer_blocks[DOUBLE_BLOCK_INDEX].attn
    single_attention = pipe.transformer.single_transformer_blocks[SINGLE_BLOCK_INDEX].attn
    double_probe = SelectedTokenProbabilityProbe(
        double_attention.get_processor(), "double_block_0", text_token_count, target_indices
    )
    single_probe = SelectedTokenProbabilityProbe(
        single_attention.get_processor(), "single_block_0", text_token_count, target_indices
    )
    double_attention.set_processor(double_probe)
    single_attention.set_processor(single_probe)
    pipe.enable_sequential_cpu_offload()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    start = time.perf_counter()
    output = pipe(
        prompt=PROMPT,
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        num_images_per_prompt=1,
        generator=generator,
    ).images[0]
    runtime_seconds = time.perf_counter() - start
    generation_path = OUTPUT_DIR / "flux-1-schnell_attention-probe_seed-42_steps-4.png"
    output.save(generation_path)

    timestep_values = [1000.0, 750.0, 500.0, 250.0]
    heatmaps = {}
    for probe in (double_probe, single_probe):
        if set(probe.maps) != set(TARGET_WORDS):
            raise RuntimeError(f"Probe {probe.name} did not capture all targets: {sorted(probe.maps)}")
        for word, probability_map in probe.maps.items():
            heatmap_path = OUTPUT_DIR / f"{probe.name}_t750_{word}_image-to-text_probability.png"
            statistics = save_heatmap(
                probability_map,
                heatmap_path,
                f"{probe.name}: P({word} key | image query), t=750",
            )
            heatmaps[f"{probe.name}:{word}"] = {
                "path": str(heatmap_path),
                "shape": list(probability_map.shape),
                "statistics": statistics,
            }

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
        "t5_non_padding_token_sequence": non_padding_tokens,
        "target_token_indices": target_indices,
        "probe_denoising_index": PROBE_DENOISING_INDEX,
        "probe_timestep": timestep_values[PROBE_DENOISING_INDEX],
        "double_block_index": DOUBLE_BLOCK_INDEX,
        "single_block_index": SINGLE_BLOCK_INDEX,
        "double_stream": double_probe.metadata,
        "single_stream": single_probe.metadata,
        "heatmaps": heatmaps,
        "runtime_seconds": runtime_seconds,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "generation_path": str(generation_path),
    }
    metrics_path = OUTPUT_DIR / "flux-1-schnell_attention-probe_metrics.json"
    metrics_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))

    pipe.maybe_free_model_hooks()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
