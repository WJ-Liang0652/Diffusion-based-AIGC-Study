"""Validate the paper Eq. (2) layout energy on FLUX.1-schnell.

This MVP reads a non-invasive, head-mean, token-span-summed, joint-normalized
image-query -> text-key attention signal from Double Stream blocks 9 and 18 at
denoising index 2 (t=500). It computes energies only; no autograd, latent update,
or guidance is performed.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import diffusers
import torch
from diffusers import FluxPipeline
from diffusers.models.embeddings import apply_rotary_emb
from PIL import Image


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
TEXT_TOKEN_COUNT = 512
PROBE_DENOISING_INDEX = 2
DOUBLE_BLOCK_INDICES = (9, 18)
KEY_CHUNK_SIZE = 128

# Fixed normalized target regions, independent of objects in the generated image.
NORMALIZED_REGIONS = {
    "left": [0.0, 0.0, 0.5, 1.0],
    "right": [0.5, 0.0, 1.0, 1.0],
}
LAYOUTS = {
    "layout_a": {"apple": "left", "cup": "right"},
    "layout_b": {"apple": "right", "cup": "left"},
}

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs" / "layout_objective"
BASELINE_PATH = PROJECT_DIR / "outputs" / "baseline" / "flux-1-schnell_seed-42_steps-4_run-1.png"


def pixel_sha256(image: Image.Image) -> str:
    return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def locate_target_spans(tokenizer, prompt: str) -> dict[str, dict[str, object]]:
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
    lower_prompt = prompt.lower()
    spans = {}
    for word in TARGET_WORDS:
        character_start = lower_prompt.index(word)
        character_end = character_start + len(word)
        indices = [
            index
            for index, (start, end) in enumerate(offsets)
            if start < character_end and end > character_start and end > start
        ]
        if not indices:
            raise RuntimeError(f"No T5 tokens overlap target {word!r}")
        ids = [token_ids[index] for index in indices]
        spans[word] = {
            "character_span": [character_start, character_end],
            "token_indices": indices,
            "token_ids": ids,
            "token_pieces": [tokens[index] for index in indices],
            "decoded_span": tokenizer.decode(ids, skip_special_tokens=True),
        }
    return spans


class LayoutAttentionProbe:
    """Delegate native attention and capture only selected joint-probability columns."""

    def __init__(self, base_processor, block_index: int, target_spans: dict[str, dict[str, object]]):
        self.base_processor = base_processor
        self.block_index = block_index
        self.target_spans = target_spans
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
        if encoder_hidden_states is None:
            raise RuntimeError("Layout objective MVP expects a Double Stream block")

        batch_size = hidden_states.shape[0]
        heads = attn.heads
        image_query = attn.to_q(hidden_states)
        image_key = attn.to_k(hidden_states)
        head_dim = image_key.shape[-1] // heads

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, -1, heads, head_dim).transpose(1, 2)

        image_query = split_heads(image_query)
        image_key = split_heads(image_key)
        text_query = split_heads(attn.add_q_proj(encoder_hidden_states))
        text_key = split_heads(attn.add_k_proj(encoder_hidden_states))
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

        image_token_count = query.shape[2] - TEXT_TOKEN_COUNT
        grid_size = math.isqrt(image_token_count)
        if grid_size * grid_size != image_token_count:
            raise RuntimeError(f"Image token count {image_token_count} is not square")

        image_queries = query[:, :, TEXT_TOKEN_COUNT:, :].float()
        scale = head_dim**-0.5
        joint_log_denominator = None
        for start in range(0, key.shape[2], KEY_CHUNK_SIZE):
            end = min(start + KEY_CHUNK_SIZE, key.shape[2])
            logits = torch.matmul(image_queries, key[:, :, start:end, :].float().transpose(-1, -2)) * scale
            chunk_logsumexp = torch.logsumexp(logits, dim=-1)
            joint_log_denominator = (
                chunk_logsumexp
                if joint_log_denominator is None
                else torch.logaddexp(joint_log_denominator, chunk_logsumexp)
            )

        for word, span in self.target_spans.items():
            selected_keys = key[:, :, span["token_indices"], :].float()
            selected_logits = torch.matmul(image_queries, selected_keys.transpose(-1, -2)) * scale
            span_probability = torch.exp(selected_logits - joint_log_denominator.unsqueeze(-1)).sum(dim=-1)
            self.maps[word] = span_probability[0].mean(dim=0).reshape(grid_size, grid_size).cpu()

        self.metadata = {
            "base_processor": type(self.base_processor).__name__,
            "attention_direction": "image queries -> target text keys",
            "normalization": "native joint softmax over all 512 text + 2304 image keys",
            "span_aggregation": "sum target span token probabilities, then mean over 24 heads",
            "map_shape": [grid_size, grid_size],
            "requires_grad": any(attention_map.requires_grad for attention_map in self.maps.values()),
        }


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


def evaluate_layouts(probe: LayoutAttentionProbe) -> dict[str, object]:
    results = {}
    for layout_name, assignments in LAYOUTS.items():
        objects = {}
        for word, region_name in assignments.items():
            objects[word] = {
                "region": region_name,
                **equation_2(probe.maps[word], NORMALIZED_REGIONS[region_name]),
            }
        results[layout_name] = {
            "objects": objects,
            "total_energy": sum(item["energy"] for item in objects.values()),
        }
    results["comparison"] = {
        "layout_a_lower_than_layout_b": results["layout_a"]["total_energy"]
        < results["layout_b"]["total_energy"],
        "energy_gap_b_minus_a": results["layout_b"]["total_energy"]
        - results["layout_a"]["total_energy"],
        "energy_ratio_b_over_a": results["layout_b"]["total_energy"]
        / results["layout_a"]["total_energy"],
    }
    return results


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this FLUX objective MVP")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE, local_files_only=True)
    target_spans = locate_target_spans(pipe.tokenizer_2, PROMPT)
    probes = []
    for block_index in DOUBLE_BLOCK_INDICES:
        attention = pipe.transformer.transformer_blocks[block_index].attn
        probe = LayoutAttentionProbe(attention.get_processor(), block_index, target_spans)
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
    generation_path = OUTPUT_DIR / "flux-1-schnell_layout-objective_seed-42_steps-4.png"
    image.save(generation_path)

    actual_timestep = float(pipe.scheduler.timesteps[PROBE_DENOISING_INDEX].item())
    block_results = {}
    for probe in probes:
        if set(probe.maps) != set(TARGET_WORDS):
            raise RuntimeError(f"Block {probe.block_index} did not capture both target maps")
        block_results[str(probe.block_index)] = {
            "block_index": probe.block_index,
            "denoising_index": PROBE_DENOISING_INDEX,
            "timestep": actual_timestep,
            "signal": probe.metadata,
            **evaluate_layouts(probe),
        }

    generated_hash = pixel_sha256(image)
    baseline_hash = pixel_sha256(Image.open(BASELINE_PATH))
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
        "equation": "E(A,B,i) = (1 - sum_{u in B} A[u,i] / sum_u A[u,i])^2",
        "target_spans": target_spans,
        "normalized_regions": NORMALIZED_REGIONS,
        "layouts": LAYOUTS,
        "blocks": block_results,
        "non_invasive_check": {
            "generated_pixel_sha256": generated_hash,
            "baseline_pixel_sha256": baseline_hash,
            "pixel_exact_match": generated_hash == baseline_hash,
        },
        "generation_path": str(generation_path),
        "runtime_seconds": runtime_seconds,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }
    metrics_path = OUTPUT_DIR / "flux-1-schnell_layout-objective_metrics.json"
    metrics_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))

    if not all(result["comparison"]["layout_a_lower_than_layout_b"] for result in block_results.values()):
        raise RuntimeError("Sanity check failed: Layout A is not lower-energy than Layout B for every probe block")

    pipe.maybe_free_model_hooks()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
