"""Compare baseline and normalized single-mid guided full SDXL denoising trajectories."""

import csv
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
GUIDED_STEP_INDICES = {0, 1, 2, 3, 4}
GUIDED_STEP_INDICES = {0, 1, 2, 3, 4}
EPS = 1e-8
SAVE_ITERATIONS = {0, 1, 5, 10, 20}
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "full_denoising_guidance_probe"


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




def unet_noise_prediction(latent, timestep, capture_attention):
    processor.capture_enabled = capture_attention
    processor.attention_probs = None
    model_input = pipe.scheduler.scale_model_input(torch.cat([latent] * 2), timestep)
    prediction = pipe.unet(
        model_input, timestep, encoder_hidden_states=prompt_embeds, timestep_cond=None,
        cross_attention_kwargs=None, added_cond_kwargs=added_cond_kwargs, return_dict=False,
    )[0]
    uncond, text = prediction.chunk(2)
    return uncond + GUIDANCE_SCALE * (text - uncond), model_input


def decode_latent(latent):
    with torch.no_grad():
        if pipe.vae.config.force_upcast:
            pipe.upcast_vae()
        dtype = next(iter(pipe.vae.post_quant_conv.parameters())).dtype
        image = pipe.vae.decode((latent / pipe.vae.config.scaling_factor).to(dtype=dtype), return_dict=False)[0]
        return pipe.image_processor.postprocess(image, output_type="pil")[0]


records = []
try:
    with torch.no_grad():
        positive, negative, positive_pooled, negative_pooled = pipe.encode_prompt(
            prompt=PROMPT, device=device, num_images_per_prompt=1, do_classifier_free_guidance=True,
            negative_prompt=None,
        )
        prompt_embeds = torch.cat([negative, positive], dim=0).to(device)
        add_text_embeds = torch.cat([negative_pooled, positive_pooled], dim=0).to(device)
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=device)
        timesteps = pipe.scheduler.timesteps
        extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta=0.0)
        initial_latents = pipe.prepare_latents(1, pipe.unet.config.in_channels, HEIGHT, WIDTH, positive.dtype, device, generator)
        time_ids = pipe._get_add_time_ids((HEIGHT, WIDTH), (0, 0), (HEIGHT, WIDTH), positive.dtype, pipe.text_encoder_2.config.projection_dim)
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": torch.cat([time_ids, time_ids], dim=0).to(device)}

        # Baseline uses an independent clone of the identical initial x_T.
        baseline_latents = initial_latents.clone()
        for timestep in timesteps:
            prediction, model_input = unet_noise_prediction(baseline_latents, timestep, False)
            baseline_latents = pipe.scheduler.step(prediction, timestep, baseline_latents, **extra_step_kwargs, return_dict=False)[0]
            del prediction, model_input

    # Euler scheduler has internal state; rebuild the same schedule for the independent guided branch.
    pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=device)
    timesteps = pipe.scheduler.timesteps
    guided_latents = initial_latents.clone()
    torch.cuda.reset_peak_memory_stats()
    for step_index, timestep in enumerate(timesteps):
        if step_index in GUIDED_STEP_INDICES:
            z_before = guided_latents.detach()
            z_current = z_before.detach().requires_grad_(True)
            prediction, model_input = unet_noise_prediction(z_current, timestep, True)
            raw_before = raw_map_from_capture(processor, token_indices)
            ratio_before, loss_before, box_grid = objective_from_raw_map(raw_before)
            gradient = torch.autograd.grad(loss_before, z_current)[0]
            check_finite("before attention", raw_before)
            check_finite("gradient", gradient)
            gradient_norm = gradient.norm()
            latent_norm = z_current.detach().norm()
            update = -ETA * gradient.detach() / (gradient_norm + EPS)
            z_updated = z_current.detach() + update
            check_finite("updated latent", z_updated)
            update_norm = update.norm()
            relative_update = update_norm / (latent_norm + EPS)
            processor.attention_probs = None
            del prediction, model_input, raw_before

            # Re-evaluate at the same t on the updated latent.  This prediction is the only one
            # used in scheduler.step; no pre-update noise prediction is mixed with z_updated.
            with torch.no_grad():
                prediction, model_input = unet_noise_prediction(z_updated, timestep, True)
                raw_after = raw_map_from_capture(processor, token_indices)
                ratio_after, loss_after, after_box = objective_from_raw_map(raw_after)
                check_finite("after attention", raw_after)
                if after_box != box_grid:
                    raise RuntimeError("Attention-grid shape changed at the same timestep.")
                records.append({
                    "step_index": step_index, "timestep": float(timestep),
                    "ratio_before": ratio_before.item(), "ratio_after": ratio_after.item(),
                    "ratio_delta": (ratio_after - ratio_before).item(),
                    "loss_before": loss_before.item(), "loss_after": loss_after.item(),
                    "loss_delta": (loss_after - loss_before).item(),
                    "gradient_norm": gradient_norm.item(), "latent_norm": latent_norm.item(),
                    "update_norm": update_norm.item(), "relative_update": relative_update.item(),
                    "has_nan": bool(torch.isnan(gradient).any() or torch.isnan(z_updated).any()),
                    "has_inf": bool(torch.isinf(gradient).any() or torch.isinf(z_updated).any()),
                })
                guided_latents = pipe.scheduler.step(prediction, timestep, z_updated, **extra_step_kwargs, return_dict=False)[0]
            processor.attention_probs = None
            del prediction, model_input, raw_after, ratio_before, loss_before, ratio_after, loss_after, gradient, update, z_current, z_updated
        else:
            with torch.no_grad():
                prediction, model_input = unet_noise_prediction(guided_latents, timestep, False)
                guided_latents = pipe.scheduler.step(prediction, timestep, guided_latents, **extra_step_kwargs, return_dict=False)[0]
                del prediction, model_input
finally:
    target_attention.set_processor(original_processor)

if len(records) != len(GUIDED_STEP_INDICES):
    raise RuntimeError(f"Expected {len(GUIDED_STEP_INDICES)} guided records, got {len(records)}")
with (OUTPUT_DIR / "guided_step_metrics.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)

# Decode after all target-step autograd work, freeing the largest unused components first.
pipe.unet.to("cpu")
pipe.text_encoder.to("cpu")
pipe.text_encoder_2.to("cpu")
torch.cuda.synchronize()
torch.cuda.empty_cache()
baseline_image = decode_latent(baseline_latents)
guided_image = decode_latent(guided_latents)
baseline_path, guided_path = OUTPUT_DIR / "baseline.png", OUTPUT_DIR / "guided.png"
baseline_image.save(baseline_path)
guided_image.save(guided_path)

baseline_pixels = torch.from_numpy(np.asarray(baseline_image.convert("RGB"), dtype=np.int16))
guided_pixels = torch.from_numpy(np.asarray(guided_image.convert("RGB"), dtype=np.int16))
absolute_difference = (guided_pixels - baseline_pixels).abs()
final_delta = guided_latents.float() - baseline_latents.float()
print("[1] Full denoising single-mid guidance probe")
print(f"guided indices / actual timesteps: {[(row['step_index'], row['timestep']) for row in records]}")
print(f"normalized eta: {ETA}")
print("step | timestep | ratio_before | ratio_after | delta | loss_before | loss_after | grad_norm | relative_update | NaN | Inf")
for row in records:
    print(f"{row['step_index']:>4d} | {row['timestep']:>8.1f} | {row['ratio_before']:.8f} | {row['ratio_after']:.8f} | {row['ratio_delta']:+.8e} | {row['loss_before']:.8f} | {row['loss_after']:.8f} | {row['gradient_norm']:.8e} | {row['relative_update']:.8e} | {row['has_nan']} | {row['has_inf']}")
print(f"final latent L2 difference / relative: {final_delta.norm().item():.8e} / {(final_delta.norm() / (baseline_latents.float().norm() + EPS)).item():.8e}")
print(f"final image mean absolute difference: {absolute_difference.float().mean().item():.8f}")
print(f"final image max absolute difference: {absolute_difference.max().item()}")
print(f"final image different pixel ratio: {torch.any(absolute_difference != 0, dim=-1).float().mean().item():.8f}")
print(f"peak allocated/reserved MiB: {torch.cuda.max_memory_allocated()/1024**2:.1f} / {torch.cuda.max_memory_reserved()/1024**2:.1f}")
print(f"baseline / guided: {baseline_path} / {guided_path}")
