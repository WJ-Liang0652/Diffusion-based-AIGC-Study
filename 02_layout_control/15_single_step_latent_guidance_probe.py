"""Probe one normalized single-mid attention-gradient latent update at fixed SDXL t=501."""

import csv
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline


MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
PROMPT = "A small red cabin beside a calm mountain lake at sunrise, realistic photograph"
SEED = 42
HEIGHT = 1024
WIDTH = 1024
NUM_INFERENCE_STEPS = 10
GUIDANCE_SCALE = 7.0
TARGET_STEP_INDEX = 4
TARGET_LAYER_NAME = "mid_block.attentions.0.transformer_blocks.0.attn2"
TARGET_WORD = "cabin"
TARGET_BOX = (0.10, 0.50, 0.40, 0.82)
ETA = 0.5
MAX_OPT_ITERS = 20
EPS = 1e-8
SAVE_ITERATIONS = {0, 1, 5, 10, 20}
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "single_step_latent_guidance_probe"


class IterativeRecordingProcessor:
    """Diffusers 0.32.2 AttnProcessor2_0 path with one graph-preserving capture."""

    def __init__(self):
        self.capture_enabled = False
        self.attention_probs = None
        self.qkv_shapes = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        head_dim = key.shape[-1] // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if self.capture_enabled:
            # This is only used to define the loss.  SDPA remains the normal output path.
            scores = torch.matmul(query, key.transpose(-2, -1)) * (head_dim**-0.5)
            if attention_mask is not None:
                scores = scores + attention_mask
            self.attention_probs = torch.softmax(scores.float(), dim=-1)
            self.qkv_shapes = {
                "query": tuple(query.shape), "key": tuple(key.shape),
                "value": tuple(value.shape), "attention_probs": tuple(self.attention_probs.shape),
            }
            del scores

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def find_word_indices(tokenizer, word):
    prompt_ids = tokenizer(
        PROMPT, padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids[0].tolist()
    word_ids = tokenizer(word, add_special_tokens=False).input_ids
    return sorted({index for start in range(len(prompt_ids) - len(word_ids) + 1)
                   if prompt_ids[start:start + len(word_ids)] == word_ids
                   for index in range(start, start + len(word_ids))})


def freeze_parameters(*models):
    parameter_count = 0
    for model in models:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter_count += parameter.numel()
    return parameter_count


def objective_from_raw_map(raw_map):
    """Raw mass objective.  No display normalization enters this computation."""
    side = raw_map.shape[0]
    x0, y0, x1, y1 = TARGET_BOX
    x_start, x_end = math.floor(x0 * side), math.ceil(x1 * side)
    y_start, y_end = math.floor(y0 * side), math.ceil(y1 * side)
    total_mass = raw_map.sum()
    inside_mass = raw_map[y_start:y_end, x_start:x_end].sum()
    inside_ratio = inside_mass / (total_mass + EPS)
    return inside_ratio, 1.0 - inside_ratio, (x_start, x_end, y_start, y_end)


def raw_map_from_capture(processor, token_indices):
    if processor.attention_probs is None:
        raise RuntimeError("Target attention probability was not captured.")
    # CFG order is [negative/unconditional, positive/conditional].
    conditional = processor.attention_probs[1]
    vector = conditional[:, :, token_indices].mean(dim=0).mean(dim=-1)
    side = math.isqrt(vector.numel())
    if side * side != vector.numel():
        raise RuntimeError(f"Cannot reshape {vector.numel()} query tokens into a square map.")
    return vector.reshape(side, side)


def save_attention_snapshot(raw_map, iteration, box_grid):
    """Save raw 32x32 data and a separately min-max-normalized visualization."""
    raw_cpu = raw_map.detach().float().cpu()
    torch.save(raw_cpu, OUTPUT_DIR / f"iteration_{iteration:02d}_raw_attention.pt")
    display = raw_cpu.numpy()
    display = (display - display.min()) / max(float(display.max() - display.min()), EPS)
    x_start, x_end, y_start, y_end = box_grid
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    image = ax.imshow(display, cmap="magma", origin="upper", vmin=0.0, vmax=1.0)
    ax.add_patch(plt.Rectangle((x_start - 0.5, y_start - 0.5), x_end - x_start, y_end - y_start,
                               fill=False, edgecolor="cyan", linewidth=2))
    ax.set_title(f"cabin raw attention, iteration {iteration}")
    ax.set_xlabel("x query position")
    ax.set_ylabel("y query position")
    fig.colorbar(image, ax=ax, label="display-normalized attention")
    fig.savefig(OUTPUT_DIR / f"iteration_{iteration:02d}_heatmap.png", dpi=180)
    plt.close(fig)


def check_finite(name, tensor):
    if torch.isnan(tensor).any().item() or torch.isinf(tensor).any().item():
        raise FloatingPointError(f"NaN or Inf detected in {name}.")


hf_home = os.environ.get("HF_HOME")
if not hf_home:
    raise RuntimeError("HF_HOME is not set. Configure the shared data-disk cache first.")
if not (Path(hf_home) / "hub" / "models--stabilityai--stable-diffusion-xl-base-1.0").is_dir():
    raise RuntimeError("The verified SDXL cache is not available under HF_HOME.")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, local_files_only=True)
pipe.to("cuda")
device = torch.device("cuda")
pipe._guidance_scale = GUIDANCE_SCALE
pipe._guidance_rescale = 0.0
frozen_parameter_count = freeze_parameters(pipe.unet, pipe.text_encoder, pipe.text_encoder_2, pipe.vae)

indices_1 = find_word_indices(pipe.tokenizer, TARGET_WORD)
indices_2 = find_word_indices(pipe.tokenizer_2, TARGET_WORD)
token_indices = sorted(set(indices_1) | set(indices_2))
if not token_indices:
    raise RuntimeError(f"Could not locate {TARGET_WORD!r} in either tokenizer output.")

target_attention = pipe.unet.get_submodule(TARGET_LAYER_NAME)
if not target_attention.is_cross_attention:
    raise RuntimeError(f"Target module is not cross-attention: {TARGET_LAYER_NAME}")
original_processor = target_attention.processor
processor = IterativeRecordingProcessor()
target_attention.set_processor(processor)

try:
    with torch.no_grad():
        positive, negative, positive_pooled, negative_pooled = pipe.encode_prompt(
            prompt=PROMPT, device=device, num_images_per_prompt=1,
            do_classifier_free_guidance=True, negative_prompt=None,
        )
        prompt_embeds = torch.cat([negative, positive], dim=0).to(device)
        add_text_embeds = torch.cat([negative_pooled, positive_pooled], dim=0).to(device)
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=device)
        timesteps = pipe.scheduler.timesteps
        extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta=0.0)
        latents = pipe.prepare_latents(1, pipe.unet.config.in_channels, HEIGHT, WIDTH, positive.dtype, device, generator)
        time_ids = pipe._get_add_time_ids((HEIGHT, WIDTH), (0, 0), (HEIGHT, WIDTH), positive.dtype, pipe.text_encoder_2.config.projection_dim)
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": torch.cat([time_ids, time_ids], dim=0).to(device)}
        # The same ordinary steps 0--3 establish the reference x_t from prior probes.
        for timestep in timesteps[:TARGET_STEP_INDEX]:
            processor.capture_enabled = False
            model_input = pipe.scheduler.scale_model_input(torch.cat([latents] * 2), timestep)
            noise = pipe.unet(model_input, timestep, encoder_hidden_states=prompt_embeds, timestep_cond=None, cross_attention_kwargs=None, added_cond_kwargs=added_cond_kwargs, return_dict=False)[0]
            uncond, text = noise.chunk(2)
            latents = pipe.scheduler.step(uncond + GUIDANCE_SCALE * (text - uncond), timestep, latents, **extra_step_kwargs, return_dict=False)[0]

    target_timestep = timesteps[TARGET_STEP_INDEX]
    z_t = latents.detach().requires_grad_(True)
    torch.cuda.reset_peak_memory_stats()
    processor.capture_enabled = True
    processor.attention_probs = None
    model_input = pipe.scheduler.scale_model_input(torch.cat([z_t] * 2), target_timestep)
    noise = pipe.unet(model_input, target_timestep, encoder_hidden_states=prompt_embeds, timestep_cond=None, cross_attention_kwargs=None, added_cond_kwargs=added_cond_kwargs, return_dict=False)[0]
    raw_before = raw_map_from_capture(processor, token_indices)
    ratio_before, loss_before, box_grid = objective_from_raw_map(raw_before)
    ratio_before_value, loss_before_value = ratio_before.item(), loss_before.item()
    gradient = torch.autograd.grad(loss_before, z_t)[0]
    check_finite("before attention", raw_before)
    check_finite("gradient", gradient)
    gradient_norm = gradient.norm()
    latent_norm = z_t.detach().norm()
    grad_normalized = gradient / (gradient_norm + EPS)
    update = -ETA * grad_normalized.detach()
    z_guided = z_t.detach() + update
    check_finite("guided latent", z_guided)
    update_norm = update.norm()
    relative_update = update_norm / (latent_norm + EPS)
    # Save the raw before map plus a separately normalized display-only heatmap.
    save_attention_snapshot(raw_before, 0, box_grid)
    processor.attention_probs = None
    del model_input, noise, raw_before, ratio_before, loss_before, grad_normalized

    # Recompute attention from z_guided at exactly the same timestep; no scheduler step occurs.
    with torch.no_grad():
        processor.capture_enabled = True
        processor.attention_probs = None
        model_input = pipe.scheduler.scale_model_input(torch.cat([z_guided] * 2), target_timestep)
        noise = pipe.unet(model_input, target_timestep, encoder_hidden_states=prompt_embeds, timestep_cond=None, cross_attention_kwargs=None, added_cond_kwargs=added_cond_kwargs, return_dict=False)[0]
        raw_after = raw_map_from_capture(processor, token_indices)
        ratio_after, loss_after, after_box_grid = objective_from_raw_map(raw_after)
        check_finite("after attention", raw_after)
        if after_box_grid != box_grid:
            raise RuntimeError("Attention grid changed between before and after evaluation.")
        save_attention_snapshot(raw_after, 1, box_grid)
finally:
    target_attention.set_processor(original_processor)

print("[1] Normalized single-step latent guidance probe")
print(f"target step_index / timestep: {TARGET_STEP_INDEX} / {float(target_timestep)}")
print(f"target layer: {TARGET_LAYER_NAME}")
print(f"cabin indices: tokenizer={indices_1}, tokenizer_2={indices_2}, used={token_indices}")
print(f"target box grid: x=[{box_grid[0]}, {box_grid[1]}), y=[{box_grid[2]}, {box_grid[3]})")
print(f"normalized update eta: {ETA}")
print(f"ratio before / after / delta: {ratio_before_value:.8f} / {ratio_after.item():.8f} / {ratio_after.item() - ratio_before_value:+.8e}")
print(f"loss before / after / delta: {loss_before_value:.8f} / {loss_after.item():.8f} / {loss_after.item() - loss_before_value:+.8e}")
print(f"gradient norm: {gradient_norm.item():.8e}")
print(f"latent norm: {latent_norm.item():.8e}")
print(f"update norm / relative update: {update_norm.item():.8e} / {relative_update.item():.8e}")
print(f"NaN / Inf (gradient, guided latent): {torch.isnan(gradient).any().item() or torch.isnan(z_guided).any().item()} / {torch.isinf(gradient).any().item() or torch.isinf(z_guided).any().item()}")
print(f"peak allocated/reserved MiB: {torch.cuda.max_memory_allocated()/1024**2:.1f} / {torch.cuda.max_memory_reserved()/1024**2:.1f}")
print(f"before/after snapshots: {OUTPUT_DIR}")
