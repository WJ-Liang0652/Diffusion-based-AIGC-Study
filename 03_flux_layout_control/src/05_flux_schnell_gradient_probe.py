"""Single-step FLUX.1-schnell layout-objective gradient probe.

The sole optimization variable is the packed denoising state entering the
transformer at denoising index 2 (t=500), with shape [1, 2304, 64]. Model
parameters are frozen. This script tests one normalized negative-gradient update
for a few small relative step sizes; it does not continue denoising or generate
an updated final image.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import diffusers
import numpy as np
import torch
from diffusers import FluxPipeline
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps


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
DOUBLE_BLOCK_INDICES = (18, 9)
KEY_CHUNK_SIZE = 128
RELATIVE_STEP_SIZES = (5e-4, 1e-3, 2e-3)

NORMALIZED_REGIONS = {
    "left": [0.0, 0.0, 0.5, 1.0],
    "right": [0.5, 0.0, 1.0, 1.0],
}
LAYOUTS = {
    "layout_a": {"apple": "left", "cup": "right"},
    "layout_b": {"apple": "right", "cup": "left"},
}

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs" / "gradient_probe"


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
            "token_indices": indices,
            "token_ids": ids,
            "token_pieces": [tokens[index] for index in indices],
            "decoded_span": tokenizer.decode(ids, skip_special_tokens=True),
        }
    return spans


class GraphLayoutAttentionProbe:
    """Capture graph-preserving selected joint-attention columns once per reset."""

    def __init__(self, base_processor, target_spans: dict[str, dict[str, object]]):
        self.base_processor = base_processor
        self.target_spans = target_spans
        self.maps: dict[str, torch.Tensor] = {}
        self.capture_enabled = True

    def reset(self) -> None:
        self.maps = {}
        self.capture_enabled = True

    def release(self) -> None:
        self.maps = {}
        self.capture_enabled = False

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        # Gradient checkpointing requires recomputation to execute the same
        # tensor operations as the original forward. Re-capturing here is safe:
        # the caller already holds its original objective tensor, while the
        # recomputed maps only replace this diagnostic attribute temporarily.
        if self.capture_enabled:
            self._capture(attn, hidden_states, encoder_hidden_states, image_rotary_emb)
        return self.base_processor(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )

    def _capture(self, attn, hidden_states, encoder_hidden_states, image_rotary_emb) -> None:
        if encoder_hidden_states is None:
            raise RuntimeError("Gradient probe expects a Double Stream block")

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
            self.maps[word] = span_probability[0].mean(dim=0).reshape(grid_size, grid_size)


def normalized_bbox_to_grid(normalized_bbox: list[float], height: int, width: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = normalized_bbox
    return (
        math.floor(x0 * width),
        math.floor(y0 * height),
        math.ceil(x1 * width),
        math.ceil(y1 * height),
    )


def layout_tensors(maps: dict[str, torch.Tensor], layout_name: str) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
    objects = {}
    for word, region_name in LAYOUTS[layout_name].items():
        attention_map = maps[word]
        x0, y0, x1, y1 = normalized_bbox_to_grid(
            NORMALIZED_REGIONS[region_name], attention_map.shape[0], attention_map.shape[1]
        )
        total_mass = attention_map.sum()
        in_box_mass = attention_map[y0:y1, x0:x1].sum()
        mass_ratio = in_box_mass / total_mass
        energy = (1.0 - mass_ratio).square()
        objects[word] = {"mass_ratio": mass_ratio, "energy": energy}
    total_energy = sum(item["energy"] for item in objects.values())
    return total_energy, objects


def serializable_layout(maps: dict[str, torch.Tensor], layout_name: str) -> dict[str, object]:
    total_energy, objects = layout_tensors(maps, layout_name)
    return {
        "objects": {
            word: {
                "region": LAYOUTS[layout_name][word],
                "mass_ratio": float(values["mass_ratio"].detach().float().item()),
                "energy": float(values["energy"].detach().float().item()),
            }
            for word, values in objects.items()
        },
        "total_energy": float(total_energy.detach().float().item()),
    }


@torch.no_grad()
def prepare_t500_state(pipe: FluxPipeline) -> dict[str, object]:
    pipe._guidance_scale = GUIDANCE_SCALE
    pipe._joint_attention_kwargs = {}
    pipe._interrupt = False
    device = pipe._execution_device
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=PROMPT,
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=512,
        lora_scale=None,
    )
    num_channels_latents = pipe.transformer.config.in_channels // 4
    latents, latent_image_ids = pipe.prepare_latents(
        1,
        num_channels_latents,
        HEIGHT,
        WIDTH,
        prompt_embeds.dtype,
        device,
        torch.Generator(device="cpu").manual_seed(SEED),
        latents=None,
    )
    sigmas = np.linspace(1.0, 1 / NUM_INFERENCE_STEPS, NUM_INFERENCE_STEPS)
    mu = calculate_shift(
        latents.shape[1],
        pipe.scheduler.config.base_image_seq_len,
        pipe.scheduler.config.max_image_seq_len,
        pipe.scheduler.config.base_shift,
        pipe.scheduler.config.max_shift,
    )
    timesteps, _ = retrieve_timesteps(
        pipe.scheduler,
        NUM_INFERENCE_STEPS,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    guidance = None
    if pipe.transformer.config.guidance_embeds:
        guidance = torch.full([1], GUIDANCE_SCALE, device=device, dtype=torch.float32).expand(latents.shape[0])

    for timestep_value in timesteps[:PROBE_DENOISING_INDEX]:
        timestep = timestep_value.expand(latents.shape[0]).to(latents.dtype)
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=pipe.joint_attention_kwargs,
            return_dict=False,
        )[0]
        latents = pipe.scheduler.step(noise_pred, timestep_value, latents, return_dict=False)[0]

    return {
        "latents": latents,
        "timestep": timesteps[PROBE_DENOISING_INDEX],
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "text_ids": text_ids,
        "latent_image_ids": latent_image_ids,
        "guidance": guidance,
    }


def transformer_forward(pipe: FluxPipeline, state: dict[str, object], latents: torch.Tensor) -> torch.Tensor:
    timestep_value = state["timestep"]
    timestep = timestep_value.expand(latents.shape[0]).to(latents.dtype)
    return pipe.transformer(
        hidden_states=latents,
        timestep=timestep / 1000,
        guidance=state["guidance"],
        pooled_projections=state["pooled_prompt_embeds"],
        encoder_hidden_states=state["prompt_embeds"],
        txt_ids=state["text_ids"],
        img_ids=state["latent_image_ids"],
        joint_attention_kwargs=pipe.joint_attention_kwargs,
        return_dict=False,
    )[0]


def run_block_probe(
    pipe: FluxPipeline,
    state: dict[str, object],
    block_index: int,
    target_spans: dict[str, dict[str, object]],
) -> dict[str, object]:
    attention = pipe.transformer.transformer_blocks[block_index].attn
    original_processor = attention.get_processor()
    probe = GraphLayoutAttentionProbe(original_processor, target_spans)
    attention.set_processor(probe)
    torch.cuda.reset_peak_memory_stats()
    block_start = time.perf_counter()

    optimization_variable = state["latents"].detach().clone().requires_grad_(True)
    noise_pred = transformer_forward(pipe, state, optimization_variable)
    if set(probe.maps) != set(TARGET_WORDS):
        raise RuntimeError(f"Block {block_index} failed to capture target maps")
    initial_layout_a = serializable_layout(probe.maps, "layout_a")
    initial_layout_b = serializable_layout(probe.maps, "layout_b")
    objective, _ = layout_tensors(probe.maps, "layout_b")
    del noise_pred

    gradient = torch.autograd.grad(objective, optimization_variable, create_graph=False, retain_graph=False)[0]
    gradient_norm = gradient.float().norm()
    latent_norm = optimization_variable.detach().float().norm()
    gradient_is_finite = bool(torch.isfinite(gradient).all().item())
    gradient_is_nonzero = bool(gradient_norm.item() > 0)
    model_parameter_grads = [
        name for name, parameter in pipe.transformer.named_parameters() if parameter.grad is not None
    ]
    probe.release()

    update_trials = []
    for relative_step_size in RELATIVE_STEP_SIZES:
        desired_update_norm = relative_step_size * latent_norm
        effective_eta = desired_update_norm / gradient_norm
        update = (effective_eta * gradient.float()).to(optimization_variable.dtype)
        updated_latents = optimization_variable.detach() - update
        probe.reset()
        with torch.no_grad():
            updated_noise_pred = transformer_forward(pipe, state, updated_latents)
        del updated_noise_pred
        updated_layout_b = serializable_layout(probe.maps, "layout_b")
        initial_energy = initial_layout_b["total_energy"]
        updated_energy = updated_layout_b["total_energy"]
        actual_update_norm = (updated_latents.float() - optimization_variable.detach().float()).norm()
        update_trials.append(
            {
                "relative_step_size": relative_step_size,
                "effective_eta": float(effective_eta.item()),
                "desired_update_norm": float(desired_update_norm.item()),
                "actual_bf16_update_norm": float(actual_update_norm.item()),
                "actual_relative_update_norm": float((actual_update_norm / latent_norm).item()),
                "updated_layout_b": updated_layout_b,
                "objective_delta_updated_minus_initial": updated_energy - initial_energy,
                "objective_decreased": updated_energy < initial_energy,
                "mass_ratio_delta": {
                    word: updated_layout_b["objects"][word]["mass_ratio"]
                    - initial_layout_b["objects"][word]["mass_ratio"]
                    for word in TARGET_WORDS
                },
            }
        )
        probe.release()
        del updated_latents, update
        torch.cuda.empty_cache()

    attention.set_processor(original_processor)
    result = {
        "block_index": block_index,
        "layout_optimized": "layout_b",
        "initial_layout_a": initial_layout_a,
        "initial_layout_b": initial_layout_b,
        "objective_backward_succeeded": True,
        "gradient_is_none": gradient is None,
        "gradient_is_nonzero": gradient_is_nonzero,
        "gradient_is_finite": gradient_is_finite,
        "gradient_norm_l2": float(gradient_norm.item()),
        "gradient_abs_mean": float(gradient.float().abs().mean().item()),
        "latent_norm_l2": float(latent_norm.item()),
        "model_parameters_with_grad": model_parameter_grads,
        "model_parameter_grad_count": len(model_parameter_grads),
        "update_trials": update_trials,
        "best_updated_energy": min(trial["updated_layout_b"]["total_energy"] for trial in update_trials),
        "any_update_decreased_objective": any(trial["objective_decreased"] for trial in update_trials),
        "runtime_seconds": time.perf_counter() - block_start,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }
    del optimization_variable, gradient, objective
    torch.cuda.empty_cache()
    return result


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this FLUX gradient probe")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE, local_files_only=True)
    target_spans = locate_target_spans(pipe.tokenizer_2, PROMPT)
    for component_name in ("text_encoder", "text_encoder_2", "transformer", "vae"):
        component = getattr(pipe, component_name, None)
        if component is not None:
            component.requires_grad_(False)
    pipe.transformer.enable_gradient_checkpointing()
    pipe.enable_sequential_cpu_offload()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    state = prepare_t500_state(pipe)
    results = {}
    for block_index in DOUBLE_BLOCK_INDICES:
        results[str(block_index)] = run_block_probe(pipe, state, block_index, target_spans)

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
        "gradient_checkpointing": True,
        "target_spans": target_spans,
        "normalized_regions": NORMALIZED_REGIONS,
        "layouts": LAYOUTS,
        "optimization_variable": {
            "name": "packed_latents_at_transformer_input_before_denoising_index_2",
            "shape": list(state["latents"].shape),
            "dtype": str(state["latents"].dtype),
            "timestep": float(state["timestep"].item()),
        },
        "blocks": results,
        "total_runtime_seconds_including_load_and_state_preparation": time.perf_counter() - total_start,
        "peak_cuda_allocated_mib_overall": max(
            result["peak_cuda_allocated_mib"] for result in results.values()
        ),
        "peak_cuda_reserved_mib_overall": max(
            result["peak_cuda_reserved_mib"] for result in results.values()
        ),
    }
    metrics_path = OUTPUT_DIR / "flux-1-schnell_gradient-probe_metrics.json"
    metrics_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))

    failures = [
        block_index
        for block_index, result in results.items()
        if not result["gradient_is_nonzero"]
        or not result["gradient_is_finite"]
        or result["model_parameter_grad_count"] != 0
        or not result["any_update_decreased_objective"]
    ]
    if failures:
        raise RuntimeError(f"Gradient probe checks failed for block(s): {failures}")

    pipe.maybe_free_model_hooks()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
