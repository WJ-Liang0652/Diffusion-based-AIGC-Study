"""Validate one-step FLUX.1-dev layout-objective gradients.

The only optimization variable is the saved packed latent entering denoising
index 35 (t≈499.842).  The Stage-3 joint-attention signal and unchanged paper
Eq. (2) optimize swapped Layout B.  This is a gradient probe only: one tiny
normalized update is evaluated per block, without continued denoising.
"""

from __future__ import annotations

import importlib.util
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


PROJECT_DIR = Path(__file__).resolve().parents[1]
STAGE3_PATH = Path(__file__).with_name("13_flux_dev_layout_objective.py")
BASELINE_STATE_PATH = PROJECT_DIR / "outputs" / "baseline" / "flux_dev_baseline_state.pt"
STAGE3_METRICS_PATH = PROJECT_DIR / "outputs" / "layout_objective" / "flux_dev_layout_objective_metrics.json"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "gradient_probe"
METRICS_PATH = OUTPUT_DIR / "flux_dev_gradient_probe_metrics.json"

PROBE_DENOISING_INDEX = 35
DOUBLE_BLOCK_INDICES = (18, 9)
TEXT_TOKEN_COUNT = 512
KEY_CHUNK_SIZE = 128
RELATIVE_STEP_SIZE = 5e-4


def load_stage3():
    spec = importlib.util.spec_from_file_location("flux_dev_layout_objective", STAGE3_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Stage-3 helpers from {STAGE3_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GraphLayoutAttentionProbe:
    """Capture graph-preserving native joint-attention target columns."""

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


def layout_tensors(stage3, maps: dict[str, torch.Tensor], layout_name: str):
    objects = {}
    for word, region_name in stage3.LAYOUTS[layout_name].items():
        attention_map = maps[word]
        x0, y0, x1, y1 = stage3.normalized_bbox_to_grid(
            stage3.NORMALIZED_REGIONS[region_name], *attention_map.shape
        )
        total_mass = attention_map.sum()
        in_box_mass = attention_map[y0:y1, x0:x1].sum()
        mass_ratio = in_box_mass / total_mass
        energy = (1.0 - mass_ratio).square()
        objects[word] = {"mass_ratio": mass_ratio, "energy": energy}
    return sum(item["energy"] for item in objects.values()), objects


def serializable_layout(stage3, maps: dict[str, torch.Tensor], layout_name: str) -> dict[str, object]:
    total_energy, objects = layout_tensors(stage3, maps, layout_name)
    return {
        "objects": {
            word: {
                "region": stage3.LAYOUTS[layout_name][word],
                "mass_ratio": float(values["mass_ratio"].detach().float().item()),
                "energy": float(values["energy"].detach().float().item()),
            }
            for word, values in objects.items()
        },
        "total_energy": float(total_energy.detach().float().item()),
    }


@torch.no_grad()
def prepare_probe_state(pipe: FluxPipeline, helpers, baseline_state: dict[str, object]) -> dict[str, object]:
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
    num_channels_latents = pipe.transformer.config.in_channels // 4
    _, latent_image_ids = pipe.prepare_latents(
        1,
        num_channels_latents,
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
        pipe.scheduler,
        helpers.NUM_INFERENCE_STEPS,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    timestep = timesteps[PROBE_DENOISING_INDEX]
    if not math.isclose(float(timestep.item()), 499.842, rel_tol=0.0, abs_tol=1e-3):
        raise RuntimeError(f"Unexpected denoising timestep at index 35: {timestep.item()}")

    step_latents = baseline_state["pipeline_step_latents"]
    if step_latents.ndim != 4 or step_latents.shape[0] != helpers.NUM_INFERENCE_STEPS:
        raise RuntimeError(f"Unexpected saved step-latent shape: {tuple(step_latents.shape)}")
    latents = step_latents[PROBE_DENOISING_INDEX - 1].to(device=device, dtype=prompt_embeds.dtype)
    guidance = torch.full(
        [latents.shape[0]], helpers.GUIDANCE_SCALE, device=device, dtype=torch.float32
    )
    return {
        "latents": latents,
        "timestep": timestep,
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "text_ids": text_ids,
        "latent_image_ids": latent_image_ids,
        "guidance": guidance,
        "mu": float(mu),
        "source": "Stage-1 saved post-scheduler state at index 34 (input to denoising index 35)",
    }


def transformer_forward(pipe: FluxPipeline, state: dict[str, object], latents: torch.Tensor) -> torch.Tensor:
    timestep = state["timestep"].expand(latents.shape[0]).to(latents.dtype)
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


def run_block_probe(pipe, stage3, state, block_index: int, target_spans, stage3_metrics) -> dict[str, object]:
    attention = pipe.transformer.transformer_blocks[block_index].attn
    original_processor = attention.get_processor()
    probe = GraphLayoutAttentionProbe(original_processor, target_spans)
    attention.set_processor(probe)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    optimization_variable = state["latents"].detach().clone().requires_grad_(True)
    noise_pred = transformer_forward(pipe, state, optimization_variable)
    if set(probe.maps) != set(target_spans):
        raise RuntimeError(f"Block {block_index} failed to capture target maps")
    initial_layout_a = serializable_layout(stage3, probe.maps, "layout_a")
    initial_layout_b = serializable_layout(stage3, probe.maps, "layout_b")
    objective, _ = layout_tensors(stage3, probe.maps, "layout_b")
    if not objective.requires_grad:
        raise RuntimeError(f"Block {block_index} objective is detached from the latent graph")
    del noise_pred

    gradient = torch.autograd.grad(objective, optimization_variable, create_graph=False, retain_graph=False)[0]
    gradient_float = gradient.float()
    gradient_norm = gradient_float.norm()
    latent_norm = optimization_variable.detach().float().norm()
    gradient_finite = bool(torch.isfinite(gradient).all().item())
    gradient_nonzero_count = int(torch.count_nonzero(gradient).item())
    parameter_grads = [
        name for name, parameter in pipe.transformer.named_parameters() if parameter.grad is not None
    ]
    probe.release()

    desired_update_norm = RELATIVE_STEP_SIZE * latent_norm
    effective_eta = desired_update_norm / gradient_norm
    update = (effective_eta * gradient_float).to(optimization_variable.dtype)
    updated_latents = optimization_variable.detach() - update
    probe.reset()
    with torch.no_grad():
        updated_noise_pred = transformer_forward(pipe, state, updated_latents)
    del updated_noise_pred
    updated_layout_b = serializable_layout(stage3, probe.maps, "layout_b")
    probe.release()
    attention.set_processor(original_processor)

    initial_energy = initial_layout_b["total_energy"]
    updated_energy = updated_layout_b["total_energy"]
    actual_update_norm = (updated_latents.float() - optimization_variable.detach().float()).norm()
    stage3_a = stage3_metrics["blocks"][str(block_index)]["layout_a"]["total_energy"]
    stage3_b = stage3_metrics["blocks"][str(block_index)]["layout_b"]["total_energy"]
    stage3_match = abs(initial_layout_a["total_energy"] - stage3_a) < 1e-5 and abs(initial_energy - stage3_b) < 1e-5
    passed = bool(
        optimization_variable.requires_grad
        and optimization_variable.is_leaf
        and gradient is not None
        and gradient_nonzero_count > 0
        and gradient_finite
        and math.isfinite(float(gradient_norm.item()))
        and gradient_norm.item() > 0
        and len(parameter_grads) == 0
        and stage3_match
        and updated_energy < initial_energy
    )
    result = {
        "block_index": block_index,
        "layout_optimized": "layout_b",
        "initial_layout_a": initial_layout_a,
        "initial_layout_b": initial_layout_b,
        "stage3_objective_match_within_1e-5": stage3_match,
        "latent_requires_grad": optimization_variable.requires_grad,
        "latent_is_leaf": optimization_variable.is_leaf,
        "gradient_is_none": gradient is None,
        "gradient_nonzero_count": gradient_nonzero_count,
        "gradient_element_count": gradient.numel(),
        "gradient_is_nonzero": gradient_nonzero_count > 0,
        "gradient_is_finite": gradient_finite,
        "gradient_norm_l2": float(gradient_norm.item()),
        "gradient_abs_max": float(gradient_float.abs().max().item()),
        "gradient_abs_mean": float(gradient_float.abs().mean().item()),
        "model_parameter_grad_count": len(parameter_grads),
        "model_parameters_with_grad": parameter_grads,
        "single_update": {
            "relative_step_size": RELATIVE_STEP_SIZE,
            "effective_eta": float(effective_eta.item()),
            "desired_update_norm": float(desired_update_norm.item()),
            "actual_bf16_update_norm": float(actual_update_norm.item()),
            "actual_relative_update_norm": float((actual_update_norm / latent_norm).item()),
            "objective_before": initial_energy,
            "objective_after": updated_energy,
            "objective_delta_after_minus_before": updated_energy - initial_energy,
            "objective_decreased": updated_energy < initial_energy,
            "updated_layout_b": updated_layout_b,
        },
        "validation_passed": passed,
        "runtime_seconds": time.perf_counter() - start,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }
    del optimization_variable, gradient, gradient_float, objective, update, updated_latents
    torch.cuda.empty_cache()
    return result


def main() -> None:
    stage3 = load_stage3()
    helpers = stage3.load_helpers()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this FLUX.1-dev gradient probe")
    if not BASELINE_STATE_PATH.exists() or not STAGE3_METRICS_PATH.exists():
        raise FileNotFoundError("Stage 1 state and Stage 3 metrics are required")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()
    baseline_state, _ = helpers.load_baseline_state()
    stage3_metrics = json.loads(STAGE3_METRICS_PATH.read_text())
    pipe = FluxPipeline.from_pretrained(helpers.MODEL_ID, torch_dtype=helpers.DTYPE, local_files_only=True)
    target_spans, _ = helpers.locate_target_spans(pipe.tokenizer_2, helpers.PROMPT)
    for component_name in ("text_encoder", "text_encoder_2", "transformer", "vae"):
        component = getattr(pipe, component_name, None)
        if component is not None:
            component.requires_grad_(False)
    pipe.transformer.enable_gradient_checkpointing()
    pipe.enable_sequential_cpu_offload()
    state = prepare_probe_state(pipe, helpers, baseline_state)

    results = {}
    for block_index in DOUBLE_BLOCK_INDICES:
        results[str(block_index)] = run_block_probe(
            pipe, stage3, state, block_index, target_spans, stage3_metrics
        )

    all_blocks_passed = all(result["validation_passed"] for result in results.values())
    record = {
        "stage": "Stage 4 - dev gradient probe",
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
        "gradient_checkpointing": True,
        "target_spans": target_spans,
        "equation": "E(A,B,i) = (1 - sum_{u in B} A[u,i] / sum_u A[u,i])^2",
        "optimization_variable": {
            "name": "packed_latents_at_transformer_input_before_denoising_index_35",
            "source": state["source"],
            "shape": list(state["latents"].shape),
            "dtype": str(state["latents"].dtype),
            "timestep": float(state["timestep"].item()),
        },
        "relative_step_size": RELATIVE_STEP_SIZE,
        "blocks": results,
        "all_blocks_passed": all_blocks_passed,
        "total_runtime_seconds_including_load_and_state_preparation": time.perf_counter() - total_start,
        "peak_cuda_allocated_mib_overall": max(result["peak_cuda_allocated_mib"] for result in results.values()),
        "peak_cuda_reserved_mib_overall": max(result["peak_cuda_reserved_mib"] for result in results.values()),
    }
    METRICS_PATH.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    if not all_blocks_passed:
        raise RuntimeError("Dev gradient-probe validation failed")

    pipe.maybe_free_model_hooks()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
