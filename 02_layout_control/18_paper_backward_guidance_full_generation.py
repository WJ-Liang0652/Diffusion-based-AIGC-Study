"""Generate SDXL baseline and paper-style backward-guided 51-step trajectories with Euler."""

import math
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import ImageDraw
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline
from diffusers.models.attention import BasicTransformerBlock


MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
PROMPT = "A small red cabin beside a calm mountain lake at sunrise, realistic photograph"
SEED = 42
HEIGHT = WIDTH = 1024
NUM_INFERENCE_STEPS = 10
GUIDANCE_SCALE = 7.0
TARGET_STEP_INDEX = 4
TARGET_WORD = "cabin"
TARGET_BOX = (0.10, 0.50, 0.40, 0.82)
EPS = 1e-8
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "paper_backward_guidance_full_generation"


class ShapeProbeProcessor:
    """Normal AttnProcessor2_0 output path while recording only query-token counts."""

    def __init__(self, name, shapes):
        self.name, self.shapes = name, shapes

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
        key, value = attn.to_k(encoder_hidden_states), attn.to_v(encoder_hidden_states)
        head_dim = key.shape[-1] // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        self.shapes[self.name] = tuple(query.shape)
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)
        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


class DifferentiableMapProcessor(ShapeProbeProcessor):
    """AttnProcessor2_0 plus a graph-preserving raw attention probability capture."""

    def __init__(self, name, maps):
        super().__init__(name, {})
        self.maps = maps

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = (hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)
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
        key, value = attn.to_k(encoder_hidden_states), attn.to_v(encoder_hidden_states)
        head_dim = key.shape[-1] // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        scores = torch.matmul(query, key.transpose(-2, -1)) * (head_dim**-0.5)
        if attention_mask is not None:
            scores = scores + attention_mask
        self.maps[self.name] = torch.softmax(scores.float(), dim=-1)
        del scores
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)
        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def find_word_indices(tokenizer, word):
    prompt_ids = tokenizer(PROMPT, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids[0].tolist()
    word_ids = tokenizer(word, add_special_tokens=False).input_ids
    return sorted({index for start in range(len(prompt_ids) - len(word_ids) + 1) if prompt_ids[start:start + len(word_ids)] == word_ids for index in range(start, start + len(word_ids))})


def freeze_parameters(*models):
    for model in models:
        for parameter in model.parameters():
            parameter.requires_grad_(False)


def region(name):
    return "down" if name.startswith("down_blocks") else "mid" if name.startswith("mid_block") else "up"


def objective(raw_map):
    side = raw_map.shape[0]
    x0, y0, x1, y1 = TARGET_BOX
    xs, xe, ys, ye = math.floor(x0 * side), math.ceil(x1 * side), math.floor(y0 * side), math.ceil(y1 * side)
    ratio = raw_map[ys:ye, xs:xe].sum() / (raw_map.sum() + EPS)
    return ratio, 1.0 - ratio, (xs, xe, ys, ye)


def reduce_to_map(probs, token_indices):
    vector = probs[1, :, :, token_indices].mean(dim=0).mean(dim=-1)
    side = math.isqrt(vector.numel())
    if side != 32 or side * side != vector.numel():
        raise RuntimeError(f"Expected 32x32 map, got {vector.numel()} query tokens.")
    return vector.reshape(side, side)


def save_heatmap(raw_map, name, box):
    raw = raw_map.detach().float().cpu()
    display = (raw - raw.min()) / (raw.max() - raw.min() + EPS)
    xs, xe, ys, ye = box
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    image = ax.imshow(display.numpy(), cmap="magma", origin="upper", vmin=0, vmax=1)
    ax.add_patch(plt.Rectangle((xs - .5, ys - .5), xe - xs, ye - ys, fill=False, edgecolor="cyan", linewidth=2))
    ax.set_title(name)
    fig.colorbar(image, ax=ax, label="display-normalized attention")
    fig.savefig(OUTPUT_DIR / f"{name.replace('/', '_')}_heatmap.png", dpi=180)
    plt.close(fig)



LOSS_SCALE = 30.0  # Author's released base_config.yaml default.
MAX_INNER_ITERS = 5
SCHEDULER_INDEX = 0  # First (early) of 51 timesteps; no scheduler.step is needed for this probe.
SAFETY_MAX_RELATIVE_UPDATE = 0.1
MID_LAYER = "mid_block.attentions.0.transformer_blocks.0.attn2"
UP_LAYER = "up_blocks.0.attentions.0.transformer_blocks.0.attn2"


def squared_energy(raw_map):
    ratio, _, box = objective(raw_map)
    return ratio, (1.0 - ratio) ** 2, box


def save_named_heatmaps(maps, tag, box):
    for name, raw_map in maps.items():
        save_heatmap(raw_map, f"{tag}_{name}", box)


# Author backward-guidance forward is conditional-only (batch size one), unlike earlier CFG probes.
def reduce_to_map(probs, token_indices):
    vector = probs[0, :, :, token_indices].mean(dim=0).mean(dim=-1)
    side = math.isqrt(vector.numel())
    if side != 32 or side * side != vector.numel():
        raise RuntimeError(f"Expected 32x32 map, got {vector.numel()} query tokens.")
    return vector.reshape(side, side)


LOSS_SCALE = 30.0
LOSS_THRESHOLD = 0.2
MAX_INNER_ITERS = 5
NUM_INFERENCE_STEPS = 51
MAX_GUIDED_STEPS = 10
CFG_SCALE = 7.0  # Existing project baseline setting; author config uses 7.5.
SAFETY_MAX_RELATIVE_UPDATE = 0.1
MID_LAYER = "mid_block.attentions.0.transformer_blocks.0.attn2"
UP_LAYER = "up_blocks.0.attentions.0.transformer_blocks.0.attn2"


def squared_energy(raw_map):
    ratio, _, box = objective(raw_map)
    return ratio, (1.0 - ratio) ** 2, box


def conditional_maps(latent, timestep, conditional_embeds, conditional_added):
    captured = {}
    for name, module in modules.items():
        module.set_processor(DifferentiableMapProcessor(name, captured))
    model_input = pipe.scheduler.scale_model_input(latent, timestep)
    _ = pipe.unet(model_input, timestep, encoder_hidden_states=conditional_embeds, added_cond_kwargs=conditional_added, return_dict=False)[0]
    if set(captured) != set(modules):
        raise RuntimeError(f"Missing conditional attention maps: {captured.keys()}")
    maps = {name: reduce_to_map(captured[name], token_indices) for name in modules}
    mid_ratio, mid_energy, box = squared_energy(maps["mid"])
    up_ratio, up_energy, up_box = squared_energy(maps["up"])
    if up_box != box:
        raise RuntimeError("mid/up attention grids do not share the target box.")
    return maps, mid_ratio, up_ratio, (mid_energy + up_energy) / 2, box, model_input


def cfg_noise_prediction(latent, timestep):
    # This forward is deliberately new: it corresponds to the final updated latent only.
    model_input = pipe.scheduler.scale_model_input(torch.cat([latent] * 2), timestep)
    prediction = pipe.unet(model_input, timestep, encoder_hidden_states=cfg_prompt_embeds, added_cond_kwargs=cfg_added, return_dict=False)[0]
    uncond, text = prediction.chunk(2)
    return uncond + CFG_SCALE * (text - uncond), model_input


def decode_latent(latent):
    with torch.no_grad():
        if pipe.vae.config.force_upcast:
            pipe.upcast_vae()
        dtype = next(iter(pipe.vae.post_quant_conv.parameters())).dtype
        image = pipe.vae.decode((latent / pipe.vae.config.scaling_factor).to(dtype=dtype), return_dict=False)[0]
        return pipe.image_processor.postprocess(image, output_type="pil")[0]


def save_heatmaps(maps, tag, box):
    for name, raw_map in maps.items():
        save_heatmap(raw_map, f"{tag}_{name}", box)


def save_box_copy(image, path):
    copy = image.convert("RGB").copy()
    draw = ImageDraw.Draw(copy)
    x0, y0, x1, y1 = TARGET_BOX
    draw.rectangle((x0 * WIDTH, y0 * HEIGHT, x1 * WIDTH, y1 * HEIGHT), outline="cyan", width=4)
    copy.save(path)


hf_home = os.environ.get("HF_HOME")
if not hf_home or not (Path(hf_home) / "hub" / "models--stabilityai--stable-diffusion-xl-base-1.0").is_dir():
    raise RuntimeError("Verified local SDXL cache under HF_HOME is required.")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, local_files_only=True)
pipe.to("cuda")
device = torch.device("cuda")
pipe._guidance_scale = CFG_SCALE
pipe._guidance_rescale = 0.0
freeze_parameters(pipe.unet, pipe.text_encoder, pipe.text_encoder_2, pipe.vae)
indices_1, indices_2 = find_word_indices(pipe.tokenizer, TARGET_WORD), find_word_indices(pipe.tokenizer_2, TARGET_WORD)
token_indices = sorted(set(indices_1) | set(indices_2))
if not token_indices:
    raise RuntimeError("cabin token was not found.")
modules = {"mid": pipe.unet.get_submodule(MID_LAYER), "up": pipe.unet.get_submodule(UP_LAYER)}
for name, module in modules.items():
    if not module.is_cross_attention:
        raise RuntimeError(f"{name} module is not cross-attention.")
original_processors = {name: module.processor for name, module in modules.items()}

inner_metrics, step_summaries = [], []
stop_reason = None
try:
    with torch.no_grad():
        positive, negative, positive_pooled, negative_pooled = pipe.encode_prompt(
            prompt=PROMPT, device=device, num_images_per_prompt=1, do_classifier_free_guidance=True,
            negative_prompt=None,
        )
        conditional_embeds = positive.to(device)
        conditional_added = {"text_embeds": positive_pooled.to(device)}
        time_ids = pipe._get_add_time_ids((HEIGHT, WIDTH), (0, 0), (HEIGHT, WIDTH), positive.dtype, pipe.text_encoder_2.config.projection_dim)
        conditional_added["time_ids"] = time_ids.to(device)
        cfg_prompt_embeds = torch.cat([negative, positive], dim=0).to(device)
        cfg_added = {"text_embeds": torch.cat([negative_pooled, positive_pooled], dim=0).to(device), "time_ids": torch.cat([time_ids, time_ids], dim=0).to(device)}
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=device)
        timesteps = pipe.scheduler.timesteps
        extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta=0.0)
        initial_latents = pipe.prepare_latents(1, pipe.unet.config.in_channels, HEIGHT, WIDTH, positive.dtype, device, generator)

        # Baseline is an independent trajectory from the exact same x_T.
        baseline_latents = initial_latents.clone()
        for timestep in timesteps:
            noise, model_input = cfg_noise_prediction(baseline_latents, timestep)
            baseline_latents = pipe.scheduler.step(noise, timestep, baseline_latents, **extra_step_kwargs, return_dict=False)[0]
            del noise, model_input

    # Euler holds an internal index. Reset the identical schedule before guided sampling.
    pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=device)
    timesteps = pipe.scheduler.timesteps
    guided_latents = initial_latents.clone()
    torch.cuda.reset_peak_memory_stats()
    for index, timestep in enumerate(timesteps):
        if index < MAX_GUIDED_STEPS:
            sigma = pipe.scheduler.sigmas[index].float()
            energy_for_condition = float("inf")  # Matches author code's large initial loss sentinel.
            inner_index = 0
            first = None
            last = None
            while energy_for_condition > LOSS_THRESHOLD and inner_index < MAX_INNER_ITERS:
                z_leaf = guided_latents.detach().requires_grad_(True)
                maps, mid_ratio, up_ratio, energy, box, model_input = conditional_maps(z_leaf, timestep, conditional_embeds, conditional_added)
                loss = energy * LOSS_SCALE
                gradient = torch.autograd.grad(loss, z_leaf)[0]
                candidate_update = gradient.detach() * sigma.square()
                latent_norm, gradient_norm = z_leaf.detach().norm(), gradient.norm()
                update_norm = candidate_update.norm()
                relative_update = update_norm / (latent_norm + EPS)
                finite = bool(torch.isfinite(energy).item() and torch.isfinite(gradient).all().item() and torch.isfinite(candidate_update).all().item())
                row = {
                    "denoising_index": index, "timestep": float(timestep), "sigma": sigma.item(), "inner_index": inner_index,
                    "mid_ratio": mid_ratio.item(), "up_ratio": up_ratio.item(), "aggregate_energy": energy.item(),
                    "loss_scaled": loss.item(), "gradient_norm": gradient_norm.item(), "latent_norm": latent_norm.item(),
                    "update_norm": update_norm.item(), "relative_update": relative_update.item(),
                    "has_nan_or_inf": not finite, "update_executed": False,
                }
                if first is None:
                    first = row.copy()
                    if index == 0:
                        save_heatmaps(maps, "guided_index_00_initial", box)
                if not finite or relative_update.item() > SAFETY_MAX_RELATIVE_UPDATE:
                    inner_metrics.append(row)
                    stop_reason = ("non-finite paper update" if not finite else f"safety stop at denoising index {index}: relative update {relative_update.item():.6f} > {SAFETY_MAX_RELATIVE_UPDATE}")
                    print(f"SAFETY WARNING: {stop_reason}")
                    break
                guided_latents = z_leaf.detach() - candidate_update
                row["update_executed"] = True
                inner_metrics.append(row)
                last = row.copy()
                energy_for_condition = energy.item()  # Exact author semantic: next while test uses prior E = loss/loss_scale.
                for name, module in modules.items():
                    module.set_processor(original_processors[name])
                del model_input, maps, gradient, candidate_update, z_leaf
                inner_index += 1
            if stop_reason:
                break

            # Measure the final guided z_t once before its normal CFG scheduler update.
            with torch.no_grad():
                final_maps, mid_final, up_final, energy_final, box, final_input = conditional_maps(guided_latents, timestep, conditional_embeds, conditional_added)
                if index == MAX_GUIDED_STEPS - 1:
                    save_heatmaps(final_maps, "guided_index_09_final", box)
            step_summaries.append({
                "denoising_index": index, "timestep": float(timestep), "sigma": sigma.item(),
                "energy_first": first["aggregate_energy"], "energy_last_pre_update": last["aggregate_energy"], "energy_final": energy_final.item(),
                "mid_ratio_first": first["mid_ratio"], "mid_ratio_last": mid_final.item(),
                "up_ratio_first": first["up_ratio"], "up_ratio_last": up_final.item(),
                "executed_inner_iterations": inner_index,
            })
            for name, module in modules.items():
                module.set_processor(original_processors[name])
            del final_maps, final_input

        # The sole scheduler.step at this diffusion index, after all inner guidance is finished.
        with torch.no_grad():
            noise, model_input = cfg_noise_prediction(guided_latents, timestep)
            guided_latents = pipe.scheduler.step(noise, timestep, guided_latents, **extra_step_kwargs, return_dict=False)[0]
            del noise, model_input
finally:
    for name, module in modules.items():
        module.set_processor(original_processors[name])

if stop_reason:
    raise RuntimeError(stop_reason)
if len(step_summaries) != MAX_GUIDED_STEPS:
    raise RuntimeError(f"Expected {MAX_GUIDED_STEPS} guided timestep summaries, got {len(step_summaries)}.")
with (OUTPUT_DIR / "guided_inner_iteration_metrics.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(inner_metrics[0]))
    writer.writeheader()
    writer.writerows(inner_metrics)
with (OUTPUT_DIR / "guided_step_metrics.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(step_summaries[0]))
    writer.writeheader()
    writer.writerows(step_summaries)

pipe.unet.to("cpu")
pipe.text_encoder.to("cpu")
pipe.text_encoder_2.to("cpu")
torch.cuda.synchronize()
torch.cuda.empty_cache()
baseline_image, guided_image = decode_latent(baseline_latents), decode_latent(guided_latents)
baseline_path, guided_path = OUTPUT_DIR / "baseline.png", OUTPUT_DIR / "guided.png"
baseline_image.save(baseline_path)
guided_image.save(guided_path)
save_box_copy(baseline_image, OUTPUT_DIR / "baseline_with_target_box.png")
save_box_copy(guided_image, OUTPUT_DIR / "guided_with_target_box.png")

latent_delta = guided_latents.float() - baseline_latents.float()
baseline_pixels = np.asarray(baseline_image.convert("RGB"), dtype=np.int16)
guided_pixels = np.asarray(guided_image.convert("RGB"), dtype=np.int16)
pixel_delta = np.abs(guided_pixels - baseline_pixels)
print("[1] Full paper-style backward-guidance generation (SDXL/Euler adaptation)")
print("author scheduler: LMSDiscreteScheduler; current experiment scheduler: EulerDiscreteScheduler (scheduler is intentionally not aligned in this run).")
print(f"CFG scale: {CFG_SCALE} (existing project baseline; author config default is 7.5)")
print(f"inner early-stop: author condition loss/loss_scale > loss_threshold, here E > {LOSS_THRESHOLD}")
for row in step_summaries:
    print(f"index={row['denoising_index']:02d} t={row['timestep']:.1f} sigma={row['sigma']:.6f} | E {row['energy_first']:.8f}->{row['energy_final']:.8f} | mid {row['mid_ratio_first']:.8f}->{row['mid_ratio_last']:.8f} | up {row['up_ratio_first']:.8f}->{row['up_ratio_last']:.8f} | inner={row['executed_inner_iterations']}")
print(f"final latent L2 / relative difference: {latent_delta.norm().item():.8e} / {(latent_delta.norm()/(baseline_latents.float().norm()+EPS)).item():.8e}")
print(f"image mean absolute / max difference / different pixel ratio: {pixel_delta.mean():.8f} / {pixel_delta.max()} / {np.any(pixel_delta != 0, axis=-1).mean():.8f}")
print(f"peak allocated/reserved MiB: {torch.cuda.max_memory_allocated()/1024**2:.1f} / {torch.cuda.max_memory_reserved()/1024**2:.1f}")
print(f"outputs: {OUTPUT_DIR}")
