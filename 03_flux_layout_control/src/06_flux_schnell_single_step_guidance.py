"""Propagate one FLUX layout-gradient update to the final schnell image.

Baseline and guided branches start from the exact same packed latent state at
t=500. Each guided branch applies exactly one normalized negative-gradient
update, then resumes the ordinary t=500 and t=250 denoising steps and VAE decode.
"""

from __future__ import annotations

import hashlib
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
from PIL import Image, ImageDraw


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
DOUBLE_BLOCK_INDEX = 18
KEY_CHUNK_SIZE = 128
RELATIVE_STEP_SIZES = (0.002, 0.005)

NORMALIZED_REGIONS = {
    "left": [0.0, 0.0, 0.5, 1.0],
    "right": [0.5, 0.0, 1.0, 1.0],
}
LAYOUT_B = {"apple": "right", "cup": "left"}

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs" / "single_step_guidance"
REFERENCE_BASELINE_PATH = PROJECT_DIR / "outputs" / "baseline" / "flux-1-schnell_seed-42_steps-4_run-1.png"


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
        start = lower_prompt.index(word)
        end = start + len(word)
        indices = [index for index, (left, right) in enumerate(offsets) if left < end and right > start and right > left]
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
    """Capture graph-preserving selected joint-attention columns."""

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
        # Recompute the same branch during gradient checkpoint backward.
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
            raise RuntimeError("Single-step guidance expects a Double Stream block")
        batch_size = hidden_states.shape[0]
        heads = attn.heads
        image_query = attn.to_q(hidden_states)
        image_key = attn.to_k(hidden_states)
        text_query = attn.add_q_proj(encoder_hidden_states)
        text_key = attn.add_k_proj(encoder_hidden_states)
        head_dim = image_key.shape[-1] // heads

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, -1, heads, head_dim).transpose(1, 2)

        image_query, image_key = split_heads(image_query), split_heads(image_key)
        text_query, text_key = split_heads(text_query), split_heads(text_key)
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
        log_denominator = None
        for start in range(0, key.shape[2], KEY_CHUNK_SIZE):
            end = min(start + KEY_CHUNK_SIZE, key.shape[2])
            logits = torch.matmul(image_queries, key[:, :, start:end, :].float().transpose(-1, -2)) * scale
            chunk_logsumexp = torch.logsumexp(logits, dim=-1)
            log_denominator = chunk_logsumexp if log_denominator is None else torch.logaddexp(log_denominator, chunk_logsumexp)

        for word, span in self.target_spans.items():
            selected_keys = key[:, :, span["token_indices"], :].float()
            selected_logits = torch.matmul(image_queries, selected_keys.transpose(-1, -2)) * scale
            span_probability = torch.exp(selected_logits - log_denominator.unsqueeze(-1)).sum(dim=-1)
            self.maps[word] = span_probability[0].mean(dim=0).reshape(grid_size, grid_size)


def bbox_to_grid(bbox: list[float], height: int, width: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return math.floor(x0 * width), math.floor(y0 * height), math.ceil(x1 * width), math.ceil(y1 * height)


def layout_tensors(maps: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
    objects = {}
    for word, region in LAYOUT_B.items():
        attention_map = maps[word]
        x0, y0, x1, y1 = bbox_to_grid(NORMALIZED_REGIONS[region], *attention_map.shape)
        ratio = attention_map[y0:y1, x0:x1].sum() / attention_map.sum()
        objects[word] = {"mass_ratio": ratio, "energy": (1.0 - ratio).square()}
    return sum(item["energy"] for item in objects.values()), objects


def serialize_layout(maps: dict[str, torch.Tensor]) -> dict[str, object]:
    objective, objects = layout_tensors(maps)
    return {
        "objects": {
            word: {
                "target_region": LAYOUT_B[word],
                "mass_ratio": float(values["mass_ratio"].detach().float().item()),
                "energy": float(values["energy"].detach().float().item()),
            }
            for word, values in objects.items()
        },
        "total_energy": float(objective.detach().float().item()),
    }


def set_schedule(pipe: FluxPipeline, image_seq_len: int, device: torch.device) -> torch.Tensor:
    sigmas = np.linspace(1.0, 1 / NUM_INFERENCE_STEPS, NUM_INFERENCE_STEPS)
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.base_image_seq_len,
        pipe.scheduler.config.max_image_seq_len,
        pipe.scheduler.config.base_shift,
        pipe.scheduler.config.max_shift,
    )
    timesteps, _ = retrieve_timesteps(pipe.scheduler, NUM_INFERENCE_STEPS, device, sigmas=sigmas, mu=mu)
    return timesteps


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
    latents, latent_image_ids = pipe.prepare_latents(
        1,
        pipe.transformer.config.in_channels // 4,
        HEIGHT,
        WIDTH,
        prompt_embeds.dtype,
        device,
        torch.Generator(device="cpu").manual_seed(SEED),
        latents=None,
    )
    timesteps = set_schedule(pipe, latents.shape[1], device)
    guidance = None
    if pipe.transformer.config.guidance_embeds:
        guidance = torch.full([1], GUIDANCE_SCALE, device=device, dtype=torch.float32).expand(latents.shape[0])
    state = {
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "text_ids": text_ids,
        "latent_image_ids": latent_image_ids,
        "guidance": guidance,
        "device": device,
    }
    for timestep_value in timesteps[:PROBE_DENOISING_INDEX]:
        noise_pred = transformer_forward(pipe, state, latents, timestep_value)
        latents = pipe.scheduler.step(noise_pred, timestep_value, latents, return_dict=False)[0]
    state["latents"] = latents
    state["timestep"] = timesteps[PROBE_DENOISING_INDEX]
    return state


def transformer_forward(
    pipe: FluxPipeline, state: dict[str, object], latents: torch.Tensor, timestep_value: torch.Tensor
) -> torch.Tensor:
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


@torch.no_grad()
def decode(pipe: FluxPipeline, packed_latents: torch.Tensor) -> Image.Image:
    latents = pipe._unpack_latents(packed_latents, HEIGHT, WIDTH, pipe.vae_scale_factor)
    latents = latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    decoded = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(decoded, output_type="pil")[0]


@torch.no_grad()
def continue_baseline(pipe: FluxPipeline, state: dict[str, object]) -> Image.Image:
    latents = state["latents"].clone()
    timesteps = set_schedule(pipe, latents.shape[1], state["device"])
    for timestep_value in timesteps[PROBE_DENOISING_INDEX:]:
        noise_pred = transformer_forward(pipe, state, latents, timestep_value)
        latents = pipe.scheduler.step(noise_pred, timestep_value, latents, return_dict=False)[0]
    return decode(pipe, latents)


def compute_gradient(
    pipe: FluxPipeline,
    state: dict[str, object],
    attention,
    original_processor,
    target_spans: dict[str, dict[str, object]],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object], dict[str, object]]:
    probe = GraphLayoutAttentionProbe(original_processor, target_spans)
    attention.set_processor(probe)
    variable = state["latents"].detach().clone().requires_grad_(True)
    noise_pred = transformer_forward(pipe, state, variable, state["timestep"])
    initial_metrics = serialize_layout(probe.maps)
    objective, _ = layout_tensors(probe.maps)
    del noise_pred
    gradient = torch.autograd.grad(objective, variable, create_graph=False, retain_graph=False)[0]
    parameter_grads = [name for name, parameter in pipe.transformer.named_parameters() if parameter.grad is not None]
    attention.set_processor(original_processor)
    return variable.detach(), gradient.detach(), initial_metrics, {
        "objective_backward_succeeded": True,
        "gradient_norm_l2": float(gradient.float().norm().item()),
        "gradient_abs_mean": float(gradient.float().abs().mean().item()),
        "gradient_is_finite": bool(torch.isfinite(gradient).all().item()),
        "gradient_is_nonzero": bool(gradient.float().norm().item() > 0),
        "model_parameter_grad_count": len(parameter_grads),
        "model_parameters_with_grad": parameter_grads,
    }


@torch.no_grad()
def continue_guided(
    pipe: FluxPipeline,
    state: dict[str, object],
    updated_latents: torch.Tensor,
    attention,
    original_processor,
    target_spans: dict[str, dict[str, object]],
) -> tuple[Image.Image, dict[str, object]]:
    probe = GraphLayoutAttentionProbe(original_processor, target_spans)
    attention.set_processor(probe)
    latents = updated_latents.clone()
    timesteps = set_schedule(pipe, latents.shape[1], state["device"])
    timestep_value = timesteps[PROBE_DENOISING_INDEX]
    noise_pred = transformer_forward(pipe, state, latents, timestep_value)
    updated_metrics = serialize_layout(probe.maps)
    probe.release()
    attention.set_processor(original_processor)
    latents = pipe.scheduler.step(noise_pred, timestep_value, latents, return_dict=False)[0]
    for timestep_value in timesteps[PROBE_DENOISING_INDEX + 1 :]:
        noise_pred = transformer_forward(pipe, state, latents, timestep_value)
        latents = pipe.scheduler.step(noise_pred, timestep_value, latents, return_dict=False)[0]
    return decode(pipe, latents), updated_metrics


def image_difference(reference: Image.Image, candidate: Image.Image) -> dict[str, object]:
    first = np.asarray(reference.convert("RGB"), dtype=np.float32)
    second = np.asarray(candidate.convert("RGB"), dtype=np.float32)
    difference = np.abs(second - first)
    return {
        "pixel_exact_match": bool(np.array_equal(first, second)),
        "max_absolute_pixel_difference": float(difference.max()),
        "mean_absolute_pixel_difference": float(difference.mean()),
        "rmse": float(np.sqrt(np.mean((second - first) ** 2))),
        "changed_pixel_fraction": float(np.any(difference > 0, axis=2).mean()),
        "mean_signed_rgb_difference": [float(value) for value in (second - first).mean(axis=(0, 1))],
    }


def color_centroids(image: Image.Image) -> dict[str, object]:
    pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    masks = {
        "apple_red_proxy": (red > 90) & (red - green > 35) & (red - blue > 25),
        "cup_blue_proxy": (blue > 80) & (blue - red > 30) & (green - red > 15),
    }
    result = {}
    for name, mask in masks.items():
        ys, xs = np.nonzero(mask)
        result[name] = {
            "pixel_count": int(mask.sum()),
            "centroid_xy_pixels": [float(xs.mean()), float(ys.mean())] if len(xs) else None,
            "centroid_xy_normalized": [float(xs.mean() / WIDTH), float(ys.mean() / HEIGHT)] if len(xs) else None,
        }
    return result


def make_comparison(images: list[tuple[str, Image.Image]], path: Path) -> None:
    header = 36
    canvas = Image.new("RGB", (WIDTH * len(images), HEIGHT + header), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(images):
        canvas.paste(image.convert("RGB"), (index * WIDTH, header))
        draw.text((index * WIDTH + 12, 10), label, fill="black")
    canvas.save(path)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this FLUX experiment")
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
    baseline_start = time.perf_counter()
    baseline_image = continue_baseline(pipe, state)
    baseline_runtime = time.perf_counter() - baseline_start
    baseline_path = OUTPUT_DIR / "flux-1-schnell_single-step_baseline_seed-42_steps-4.png"
    baseline_image.save(baseline_path)

    attention = pipe.transformer.transformer_blocks[DOUBLE_BLOCK_INDEX].attn
    original_processor = attention.get_processor()
    gradient_start = time.perf_counter()
    base_latents, gradient, initial_layout, gradient_metrics = compute_gradient(
        pipe, state, attention, original_processor, target_spans
    )
    gradient_runtime = time.perf_counter() - gradient_start
    latent_norm = base_latents.float().norm()
    gradient_norm = gradient.float().norm()

    guided_results = []
    comparison_images = [("baseline", baseline_image)]
    for relative_step_size in RELATIVE_STEP_SIZES:
        branch_start = time.perf_counter()
        desired_update_norm = relative_step_size * latent_norm
        effective_eta = desired_update_norm / gradient_norm
        update = (effective_eta * gradient.float()).to(base_latents.dtype)
        updated_latents = base_latents - update
        actual_update_norm = (updated_latents.float() - base_latents.float()).norm()
        guided_image, updated_layout = continue_guided(
            pipe, state, updated_latents, attention, original_processor, target_spans
        )
        label = str(relative_step_size).replace(".", "p")
        guided_path = OUTPUT_DIR / f"flux-1-schnell_single-step-guided_rel-{label}_seed-42_steps-4.png"
        guided_image.save(guided_path)
        comparison_images.append((f"guided relative={relative_step_size}", guided_image))
        guided_results.append(
            {
                "relative_step_size": relative_step_size,
                "effective_eta": float(effective_eta.item()),
                "update_formula": "x_new = x - eta*g; eta = relative_step*||x||_2/||g||_2",
                "desired_update_norm": float(desired_update_norm.item()),
                "actual_bf16_update_norm": float(actual_update_norm.item()),
                "actual_relative_update_norm": float((actual_update_norm / latent_norm).item()),
                "initial_layout_b": initial_layout,
                "updated_layout_b": updated_layout,
                "objective_delta_updated_minus_initial": updated_layout["total_energy"] - initial_layout["total_energy"],
                "mass_ratio_delta": {
                    word: updated_layout["objects"][word]["mass_ratio"] - initial_layout["objects"][word]["mass_ratio"]
                    for word in TARGET_WORDS
                },
                "image_path": str(guided_path),
                "pixel_sha256": pixel_sha256(guided_image),
                "pixel_difference_vs_branch_baseline": image_difference(baseline_image, guided_image),
                "color_centroids": color_centroids(guided_image),
                "branch_runtime_seconds": time.perf_counter() - branch_start,
            }
        )

    comparison_path = OUTPUT_DIR / "flux-1-schnell_single-step-guidance_comparison.png"
    make_comparison(comparison_images, comparison_path)
    reference_baseline_hash = pixel_sha256(Image.open(REFERENCE_BASELINE_PATH))
    peak_allocated = torch.cuda.max_memory_allocated() / 2**20
    peak_reserved = torch.cuda.max_memory_reserved() / 2**20
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
        "block_index": DOUBLE_BLOCK_INDEX,
        "denoising_index": PROBE_DENOISING_INDEX,
        "timestep": float(state["timestep"].item()),
        "layout": LAYOUT_B,
        "normalized_regions": NORMALIZED_REGIONS,
        "target_spans": target_spans,
        "optimization_variable": {
            "name": "packed_latents_at_transformer_input_before_t500",
            "shape": list(base_latents.shape),
            "dtype": str(base_latents.dtype),
        },
        "update_definition": {
            "formula": "x_new = x - eta*g",
            "gradient": "g = d(E_layout_b)/dx",
            "eta": "relative_step * ||x||_2 / ||g||_2",
            "interpretation": "desired update L2 norm is relative_step times current packed-latent L2 norm; update is rounded to latent BF16",
        },
        "initial_layout_b": initial_layout,
        "gradient": gradient_metrics,
        "baseline": {
            "image_path": str(baseline_path),
            "pixel_sha256": pixel_sha256(baseline_image),
            "matches_original_pipeline_baseline": pixel_sha256(baseline_image) == reference_baseline_hash,
            "runtime_seconds_from_t500": baseline_runtime,
            "color_centroids": color_centroids(baseline_image),
        },
        "guided_results": guided_results,
        "comparison_path": str(comparison_path),
        "gradient_runtime_seconds": gradient_runtime,
        "total_runtime_seconds_including_load": time.perf_counter() - total_start,
        "peak_cuda_allocated_mib": peak_allocated,
        "peak_cuda_reserved_mib": peak_reserved,
    }
    metrics_path = OUTPUT_DIR / "flux-1-schnell_single-step-guidance_metrics.json"
    metrics_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))

    pipe.maybe_free_model_hooks()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
