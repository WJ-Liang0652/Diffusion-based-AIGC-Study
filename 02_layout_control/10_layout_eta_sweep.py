"""Calibrate independent iterative layout-optimization step sizes at fixed SDXL t=501."""

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
ETAS = (100.0, 300.0, 1000.0)
MAX_OPT_ITERS = 20
EPS = 1e-8
SAVE_ITERATIONS = {0, 5, 10, 20}
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "layout_eta_sweep"


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


def save_attention_snapshot(raw_map, iteration, box_grid, output_dir):
    """Save raw 32x32 data and a separately min-max-normalized visualization."""
    raw_cpu = raw_map.detach().float().cpu()
    torch.save(raw_cpu, output_dir / f"iteration_{iteration:02d}_raw_attention.pt")
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
    fig.savefig(output_dir / f"iteration_{iteration:02d}_heatmap.png", dpi=180)
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


# All three eta values share exactly this fixed target latent, then branch independently.
all_records = {}
summary = []
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
        latents = pipe.prepare_latents(
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
        # Normal SDXL denoising stops at x_t for step_index=4.
        for timestep in timesteps[:TARGET_STEP_INDEX]:
            processor.capture_enabled = False
            model_input = pipe.scheduler.scale_model_input(torch.cat([latents] * 2), timestep)
            noise_pred = pipe.unet(
                model_input, timestep, encoder_hidden_states=prompt_embeds, timestep_cond=None,
                cross_attention_kwargs=None, added_cond_kwargs=added_cond_kwargs, return_dict=False,
            )[0]
            noise_uncond, noise_text = noise_pred.chunk(2)
            latents = pipe.scheduler.step(
                noise_uncond + GUIDANCE_SCALE * (noise_text - noise_uncond), timestep, latents,
                **extra_step_kwargs, return_dict=False,
            )[0]

    target_timestep = timesteps[TARGET_STEP_INDEX]
    x_initial = latents.detach()
    torch.cuda.reset_peak_memory_stats()

    for eta in ETAS:
        eta_dir = OUTPUT_DIR / f"eta_{eta:g}"
        eta_dir.mkdir(parents=True, exist_ok=True)
        # Critical: clone the same x_initial rather than chaining between eta settings.
        x_current = x_initial.clone()
        records = []
        stop_reason = None
        try:
            for iteration in range(MAX_OPT_ITERS + 1):
                x_current = x_current.detach().requires_grad_(True)
                processor.capture_enabled = True
                processor.attention_probs = None
                model_input = pipe.scheduler.scale_model_input(torch.cat([x_current] * 2), target_timestep)
                noise_pred = pipe.unet(
                    model_input, target_timestep, encoder_hidden_states=prompt_embeds, timestep_cond=None,
                    cross_attention_kwargs=None, added_cond_kwargs=added_cond_kwargs, return_dict=False,
                )[0]
                raw_map = raw_map_from_capture(processor, token_indices)
                inside_ratio, layout_loss, box_grid = objective_from_raw_map(raw_map)
                gradient = torch.autograd.grad(layout_loss, x_current)[0]
                check_finite("raw attention", raw_map)
                check_finite("layout loss", layout_loss)
                check_finite("latent gradient", gradient)

                update = -eta * gradient.detach()
                x_next = x_current.detach() + update
                check_finite("updated latent", x_next)
                initial_ratio = records[0]["inside_ratio"] if records else inside_ratio.item()
                initial_loss = records[0]["layout_loss"] if records else layout_loss.item()
                cumulative_update = x_current.detach() - x_initial
                record = {
                    "eta": eta,
                    "iteration": iteration,
                    "inside_ratio": inside_ratio.item(),
                    "layout_loss": layout_loss.item(),
                    "grad_norm": gradient.norm().item(),
                    "relative_update": (update.norm() / (x_current.detach().norm() + EPS)).item(),
                    "latent_norm": x_current.detach().norm().item(),
                    "delta_ratio_vs_initial": inside_ratio.item() - initial_ratio,
                    "delta_loss_vs_initial": layout_loss.item() - initial_loss,
                    "cumulative_update_norm": cumulative_update.norm().item(),
                    "cumulative_relative_update": (
                        cumulative_update.norm() / (x_initial.norm() + EPS)
                    ).item(),
                }
                records.append(record)
                if iteration in SAVE_ITERATIONS:
                    save_attention_snapshot(raw_map, iteration, box_grid, eta_dir)

                processor.attention_probs = None
                del model_input, noise_pred, raw_map, inside_ratio, layout_loss, gradient, update, cumulative_update
                if iteration == MAX_OPT_ITERS:
                    break
                x_current = x_next
                del x_next
                # Stop only meaningful, sustained worsening, not float16-scale fluctuations.
                if len(records) >= 3 and all(
                    records[-offset]["layout_loss"] > records[-offset - 1]["layout_loss"] + 1e-5
                    for offset in (1, 2)
                ):
                    stop_reason = "loss increased by more than 1e-5 for two consecutive updates"
                    break
        except (FloatingPointError, torch.cuda.OutOfMemoryError) as error:
            stop_reason = f"stopped: {type(error).__name__}: {error}"
            if isinstance(error, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()

        if not records:
            raise RuntimeError(f"eta={eta:g} produced no valid records: {stop_reason}")
        with (eta_dir / "iteration_metrics.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        all_records[eta] = records
        initial, final = records[0], records[-1]
        summary.append({
            "eta": eta,
            "initial_ratio": initial["inside_ratio"], "final_ratio": final["inside_ratio"],
            "delta_ratio": final["delta_ratio_vs_initial"],
            "initial_loss": initial["layout_loss"], "final_loss": final["layout_loss"],
            "delta_loss": final["delta_loss_vs_initial"],
            "cumulative_update_norm": final["cumulative_update_norm"],
            "cumulative_relative_update": final["cumulative_relative_update"],
            "stable": stop_reason is None,
            "stop_reason": stop_reason or "completed MAX_OPT_ITERS",
        })
finally:
    target_attention.set_processor(original_processor)

with (OUTPUT_DIR / "eta_sweep_summary.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(summary[0]) if summary else ["eta"])
    writer.writeheader()
    writer.writerows(summary)

print("[1] Fixed-timestep iterative layout eta sweep")
print(f"target step_index / timestep: {TARGET_STEP_INDEX} / {float(target_timestep)}")
print(f"target layer: {TARGET_LAYER_NAME}")
print(f"cabin indices: tokenizer={indices_1}, tokenizer_2={indices_2}, used={token_indices}")
print(f"target box grid: x=[{box_grid[0]}, {box_grid[1]}), y=[{box_grid[2]}, {box_grid[3]})")
print(f"etas / maximum iterations: {ETAS} / {MAX_OPT_ITERS}")
print(f"frozen parameter count: {frozen_parameter_count}")
print("eta | initial_ratio | final_ratio | delta_ratio | initial_loss | final_loss | delta_loss | cumulative_relative_update | stable")
for row in summary:
    print(
        f"{row['eta']:>4g} | {row['initial_ratio']:.8f} | {row['final_ratio']:.8f} | "
        f"{row['delta_ratio']:+.8e} | {row['initial_loss']:.8f} | {row['final_loss']:.8f} | "
        f"{row['delta_loss']:+.8e} | {row['cumulative_relative_update']:.8e} | {row['stable']}"
    )
    print(f"  stop reason: {row['stop_reason']}")
print(f"peak allocated/reserved MiB: {torch.cuda.max_memory_allocated() / 1024**2:.1f} / {torch.cuda.max_memory_reserved() / 1024**2:.1f}")
print(f"summary: {OUTPUT_DIR / 'eta_sweep_summary.csv'}")
