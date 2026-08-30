"""Repeat the validated SDXL layout-guidance run with eta=1000 only."""

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
ETA = 1000.0
INNER_ITERS = 5
GUIDED_STEP_INDICES = {1, 2, 3, 4}
MAX_OPT_ITERS = 20
EPS = 1e-8
SAVE_ITERATIONS = {0, 1, 5, 10, 20}
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "layout_guidance"


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


def unet_noise_prediction(latents, timestep, capture_attention):
    """Run one CFG UNet forward; caller decides whether its graph is needed."""
    processor.capture_enabled = capture_attention
    processor.attention_probs = None
    model_input = pipe.scheduler.scale_model_input(torch.cat([latents] * 2), timestep)
    noise_pred = pipe.unet(
        model_input, timestep, encoder_hidden_states=prompt_embeds, timestep_cond=None,
        cross_attention_kwargs=None, added_cond_kwargs=added_cond_kwargs, return_dict=False,
    )[0]
    noise_uncond, noise_text = noise_pred.chunk(2)
    return noise_uncond + GUIDANCE_SCALE * (noise_text - noise_uncond), model_input


def decode_latent(latents):
    """Decode only after denoising; move non-VAE components away to preserve VRAM."""
    with torch.no_grad():
        if pipe.vae.config.force_upcast:
            pipe.upcast_vae()
        vae_dtype = next(iter(pipe.vae.post_quant_conv.parameters())).dtype
        image = pipe.vae.decode(
            (latents / pipe.vae.config.scaling_factor).to(dtype=vae_dtype), return_dict=False
        )[0]
        return pipe.image_processor.postprocess(image, output_type="pil")[0]


records = []
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
        initial_latents = pipe.prepare_latents(
            1, pipe.unet.config.in_channels, HEIGHT, WIDTH, positive.dtype, device, generator
        )
        time_ids = pipe._get_add_time_ids(
            (HEIGHT, WIDTH), (0, 0), (HEIGHT, WIDTH), positive.dtype,
            pipe.text_encoder_2.config.projection_dim,
        )
        added_cond_kwargs = {
            "text_embeds": add_text_embeds,
            "time_ids": torch.cat([time_ids, time_ids], dim=0).to(device),
        }

        # Baseline consumes one independent clone of the exact seeded initial latent.
        baseline_latents = initial_latents.clone()
        for timestep in timesteps:
            noise_pred, model_input = unet_noise_prediction(baseline_latents, timestep, False)
            baseline_latents = pipe.scheduler.step(
                noise_pred, timestep, baseline_latents, **extra_step_kwargs, return_dict=False
            )[0]
            del noise_pred, model_input

    # Controlled run starts from the same tensor values, not the baseline terminal latent.
    # EulerDiscreteScheduler keeps an internal step_index; reset the identical schedule for this independent run.
    pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=device)
    timesteps = pipe.scheduler.timesteps
    controlled_latents = initial_latents.clone()
    torch.cuda.reset_peak_memory_stats()
    for step_index, timestep in enumerate(timesteps):
        if step_index in GUIDED_STEP_INDICES:
            x_before = controlled_latents.detach()
            x_current = x_before.clone()
            inner_losses = []
            pre_ratio = pre_loss = post_ratio = post_loss = None
            finite = True
            for inner_index in range(INNER_ITERS):
                # A new leaf and a new target-step graph are built for every inner update.
                x_current = x_current.detach().requires_grad_(True)
                noise_pred, model_input = unet_noise_prediction(x_current, timestep, True)
                raw_map = raw_map_from_capture(processor, token_indices)
                inside_ratio, layout_loss, box_grid = objective_from_raw_map(raw_map)
                gradient = torch.autograd.grad(layout_loss, x_current)[0]
                check_finite("raw attention", raw_map)
                check_finite("layout loss", layout_loss)
                check_finite("latent gradient", gradient)
                if inner_index == 0:
                    pre_ratio, pre_loss = inside_ratio.item(), layout_loss.item()
                inner_losses.append(layout_loss.item())
                x_next = x_current.detach() - ETA * gradient.detach()
                check_finite("guided latent", x_next)
                processor.attention_probs = None
                del noise_pred, model_input, raw_map, inside_ratio, layout_loss, gradient
                x_current = x_next
                del x_next

            # Measure the post-guidance objective on the final updated x_t.
            x_measure = x_current.detach().requires_grad_(True)
            noise_pred, model_input = unet_noise_prediction(x_measure, timestep, True)
            raw_map = raw_map_from_capture(processor, token_indices)
            inside_ratio, layout_loss, box_grid = objective_from_raw_map(raw_map)
            check_finite("post-guidance raw attention", raw_map)
            check_finite("post-guidance layout loss", layout_loss)
            post_ratio, post_loss = inside_ratio.item(), layout_loss.item()
            cumulative_update = x_measure.detach() - x_before
            records.append({
                "step_index": step_index,
                "timestep": float(timestep),
                "pre_inside_ratio": pre_ratio,
                "pre_layout_loss": pre_loss,
                "post_inside_ratio": post_ratio,
                "post_layout_loss": post_loss,
                "delta_ratio": post_ratio - pre_ratio,
                "delta_loss": post_loss - pre_loss,
                "inner_losses": ";".join(f"{value:.8f}" for value in inner_losses),
                "cumulative_update_norm": cumulative_update.norm().item(),
                "cumulative_relative_update": (
                    cumulative_update.norm() / (x_before.norm() + EPS)
                ).item(),
                "has_nan_or_inf": not finite,
            })
            # Discard the measurement graph.  Its noise pred is deliberately not reused.
            processor.attention_probs = None
            del noise_pred, model_input, raw_map, inside_ratio, layout_loss, cumulative_update
            del x_measure
            controlled_latents = x_current.detach()
            del x_current, x_before

        # For every timestep (including guided ones), predict noise again from the current
        # latent and only then advance Euler's scheduler.  No old optimization prediction leaks in.
        with torch.no_grad():
            noise_pred, model_input = unet_noise_prediction(controlled_latents, timestep, False)
            controlled_latents = pipe.scheduler.step(
                noise_pred, timestep, controlled_latents, **extra_step_kwargs, return_dict=False
            )[0]
            del noise_pred, model_input
finally:
    target_attention.set_processor(original_processor)

if len(records) != len(GUIDED_STEP_INDICES):
    raise RuntimeError(f"Expected {len(GUIDED_STEP_INDICES)} guided records, got {len(records)}.")
with (OUTPUT_DIR / "guided_timestep_metrics_eta1000.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)

# The target-step autograd work is done.  Move unused components off GPU before force-upcast VAE decode.
pipe.unet.to("cpu")
pipe.text_encoder.to("cpu")
pipe.text_encoder_2.to("cpu")
torch.cuda.synchronize()
torch.cuda.empty_cache()
baseline_image = decode_latent(baseline_latents)
controlled_image = decode_latent(controlled_latents)
baseline_path = OUTPUT_DIR / "baseline_seed42.png"
controlled_path = OUTPUT_DIR / "controlled_eta1000_seed42.png"
baseline_image.save(baseline_path)
controlled_image.save(controlled_path)

# This overlay is only a layout-condition reference, not evidence that image pixels are segmented.
fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
ax.imshow(controlled_image)
x0, y0, x1, y1 = TARGET_BOX
ax.add_patch(plt.Rectangle((x0 * WIDTH, y0 * HEIGHT), (x1 - x0) * WIDTH, (y1 - y0) * HEIGHT,
                           fill=False, edgecolor="cyan", linewidth=3))
ax.set_title("Controlled image with cabin target box (layout reference)")
ax.axis("off")
overlay_path = OUTPUT_DIR / "controlled_eta1000_seed42_target_box_reference.png"
fig.savefig(overlay_path, dpi=180)
plt.close(fig)

print("[1] SDXL single-layer layout guidance")
print(f"guided indices / actual timesteps: {[(item['step_index'], item['timestep']) for item in records]}")
print(f"eta / inner iterations: {ETA} / {INNER_ITERS}")
print("step | timestep | pre_ratio | post_ratio | delta_ratio | pre_loss | post_loss | delta_loss | cumulative_relative_update | NaN/Inf")
for item in records:
    print(
        f"{item['step_index']:>4d} | {item['timestep']:>8.1f} | {item['pre_inside_ratio']:.8f} | "
        f"{item['post_inside_ratio']:.8f} | {item['delta_ratio']:+.8e} | "
        f"{item['pre_layout_loss']:.8f} | {item['post_layout_loss']:.8f} | "
        f"{item['delta_loss']:+.8e} | {item['cumulative_relative_update']:.8e} | "
        f"{item['has_nan_or_inf']}"
    )
print(f"peak allocated/reserved MiB: {torch.cuda.max_memory_allocated() / 1024**2:.1f} / {torch.cuda.max_memory_reserved() / 1024**2:.1f}")
print(f"baseline: {baseline_path}")
print(f"controlled: {controlled_path}")
print(f"target-box reference: {overlay_path}")

# Numerical comparison uses saved 8-bit images and does not affect sampling or guidance.
import numpy as np
from PIL import Image

baseline_pixels = np.asarray(Image.open(OUTPUT_DIR / "baseline_seed42.png").convert("RGB"), dtype=np.int16)
eta1000_pixels = np.asarray(Image.open(controlled_path).convert("RGB"), dtype=np.int16)
if baseline_pixels.shape != eta1000_pixels.shape:
    raise RuntimeError(f"Saved image shape mismatch: {baseline_pixels.shape} vs {eta1000_pixels.shape}")
absolute_difference = np.abs(eta1000_pixels - baseline_pixels)
print("baseline vs eta=1000 pixel comparison")
print(f"mean absolute pixel difference: {absolute_difference.mean():.8f}")
print(f"RMSE: {np.sqrt(np.mean((eta1000_pixels - baseline_pixels).astype(np.float64) ** 2)):.8f}")
print(f"max absolute pixel difference: {absolute_difference.max()}")
print(f"different pixel ratio: {np.any(absolute_difference != 0, axis=-1).mean():.8f}")
