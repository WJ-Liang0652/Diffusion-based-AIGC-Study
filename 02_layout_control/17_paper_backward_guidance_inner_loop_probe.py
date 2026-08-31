"""Probe paper-style squared cross-attention backward guidance at one SDXL timestep."""

import math
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
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
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "paper_backward_guidance_inner_loop_probe"


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

hf_home = os.environ.get("HF_HOME")
if not hf_home or not (Path(hf_home) / "hub" / "models--stabilityai--stable-diffusion-xl-base-1.0").is_dir():
    raise RuntimeError("Verified local SDXL cache under HF_HOME is required.")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, local_files_only=True)
pipe.to("cuda")
device = torch.device("cuda")
pipe._guidance_scale = GUIDANCE_SCALE
pipe._guidance_rescale = 0.0
freeze_parameters(pipe.unet, pipe.text_encoder, pipe.text_encoder_2, pipe.vae)
indices_1, indices_2 = find_word_indices(pipe.tokenizer, TARGET_WORD), find_word_indices(pipe.tokenizer_2, TARGET_WORD)
token_indices = sorted(set(indices_1) | set(indices_2))
if not token_indices:
    raise RuntimeError("cabin token was not found.")

selected = {"mid": MID_LAYER, "up": UP_LAYER}
modules = {name: pipe.unet.get_submodule(layer_name) for name, layer_name in selected.items()}
for name, module in modules.items():
    if not module.is_cross_attention:
        raise RuntimeError(f"{selected[name]} is not cross-attention.")
original_processors = {name: module.processor for name, module in modules.items()}

metrics = []
stop_reason = None
try:
    with torch.no_grad():
        positive, negative, positive_pooled, negative_pooled = pipe.encode_prompt(
            prompt=PROMPT, device=device, num_images_per_prompt=1,
            do_classifier_free_guidance=True, negative_prompt=None,
        )
        # Author backward-guidance forward is conditional-only; CFG is used only after guidance.
        conditional_embeds = positive.to(device)
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        pipe.scheduler.set_timesteps(51, device=device)
        timesteps = pipe.scheduler.timesteps
        latents = pipe.prepare_latents(1, pipe.unet.config.in_channels, HEIGHT, WIDTH, positive.dtype, device, generator)
        time_ids = pipe._get_add_time_ids((HEIGHT, WIDTH), (0, 0), (HEIGHT, WIDTH), positive.dtype, pipe.text_encoder_2.config.projection_dim)
        conditional_added = {"text_embeds": positive_pooled.to(device), "time_ids": time_ids.to(device)}

    if SCHEDULER_INDEX >= len(timesteps):
        raise RuntimeError("SCHEDULER_INDEX exceeds the 51-step schedule.")
    timestep = timesteps[SCHEDULER_INDEX]
    sigma = pipe.scheduler.sigmas[SCHEDULER_INDEX].float()
    timestep_index = int(timestep.item())
    alpha_cumprod = None
    if hasattr(pipe.scheduler, "alphas_cumprod") and 0 <= timestep_index < len(pipe.scheduler.alphas_cumprod):
        alpha_cumprod = pipe.scheduler.alphas_cumprod[timestep_index].float()
    print("[0] Scheduler / scaling information")
    print(f"scheduler type: {type(pipe.scheduler).__name__}")
    print(f"scheduler config: {pipe.scheduler.config}")
    print(f"51-step index / timestep: {SCHEDULER_INDEX} / {float(timestep)}")
    print(f"Euler sigma / sigma^2: {sigma.item():.8e} / {(sigma ** 2).item():.8e}")
    print(f"alpha_cumprod at integer timestep: {alpha_cumprod.item():.8e}" if alpha_cumprod is not None else "alpha_cumprod: unavailable")
    print("reference distinction: Eq.(3) states z <- z - sigma_t * eta * grad(E); author code uses grad(E * loss_scale) then z <- z - grad * LMS sigma[index]^2.")

    z_current = latents.detach()
    torch.cuda.reset_peak_memory_stats()
    for inner_iter in range(MAX_INNER_ITERS):
        z_leaf = z_current.detach().requires_grad_(True)
        captured = {}
        for name, module in modules.items():
            module.set_processor(DifferentiableMapProcessor(name, captured))
        model_input = pipe.scheduler.scale_model_input(z_leaf, timestep)
        _ = pipe.unet(model_input, timestep, encoder_hidden_states=conditional_embeds, added_cond_kwargs=conditional_added, return_dict=False)[0]
        if set(captured) != set(selected):
            raise RuntimeError(f"Missing attention capture: {captured.keys()}")
        maps_before = {name: reduce_to_map(captured[name], token_indices) for name in selected}
        mid_ratio, mid_energy, box = squared_energy(maps_before["mid"])
        up_ratio, up_energy, up_box = squared_energy(maps_before["up"])
        if up_box != box:
            raise RuntimeError("Selected attention grids do not share the expected 32x32 target box.")
        energy = (mid_energy + up_energy) / 2
        loss_scaled = energy * LOSS_SCALE
        gradient = torch.autograd.grad(loss_scaled, z_leaf)[0]
        latent_norm = z_leaf.detach().norm()
        raw_gradient_norm = gradient.norm()
        candidate_update = -gradient.detach() * sigma.square()
        candidate_update_norm = candidate_update.norm()
        candidate_relative_update = candidate_update_norm / (latent_norm + EPS)
        finite_before = bool(torch.isfinite(energy).item() and torch.isfinite(gradient).all().item() and torch.isfinite(candidate_update).all().item())
        if inner_iter == 0:
            save_named_heatmaps(maps_before, "inner_00_before", box)
        record = {
            "inner_iteration": inner_iter,
            "timestep": float(timestep),
            "mid_ratio_before": mid_ratio.item(), "up_ratio_before": up_ratio.item(),
            "mid_energy_before": mid_energy.item(), "up_energy_before": up_energy.item(),
            "energy_before": energy.item(), "loss_scaled": loss_scaled.item(),
            "raw_gradient_norm": raw_gradient_norm.item(), "latent_norm": latent_norm.item(),
            "sigma": sigma.item(), "sigma_squared": sigma.square().item(),
            "candidate_update_norm": candidate_update_norm.item(),
            "candidate_relative_update": candidate_relative_update.item(),
            "mid_ratio_after": None, "up_ratio_after": None, "energy_after": None,
            "has_nan_or_inf": not finite_before, "update_executed": False,
        }
        if not finite_before or candidate_relative_update.item() > SAFETY_MAX_RELATIVE_UPDATE:
            stop_reason = ("safety stop: non-finite candidate" if not finite_before else
                           f"safety stop: candidate relative update {candidate_relative_update.item():.6f} > {SAFETY_MAX_RELATIVE_UPDATE}")
            metrics.append(record)
            print(f"SAFETY WARNING: {stop_reason}")
            break

        # Exactly the author code's practical update: grad(E * 30) * scheduler.sigmas[index]^2.
        z_updated = z_leaf.detach() + candidate_update
        record["update_executed"] = True
        if not torch.isfinite(z_updated).all():
            record["has_nan_or_inf"] = True
            metrics.append(record)
            stop_reason = "safety stop: update produced NaN/Inf"
            break
        for name, module in modules.items():
            module.set_processor(original_processors[name])
        with torch.no_grad():
            captured_after = {}
            for name, module in modules.items():
                module.set_processor(DifferentiableMapProcessor(name, captured_after))
            model_input_after = pipe.scheduler.scale_model_input(z_updated, timestep)
            _ = pipe.unet(model_input_after, timestep, encoder_hidden_states=conditional_embeds, added_cond_kwargs=conditional_added, return_dict=False)[0]
            maps_after = {name: reduce_to_map(captured_after[name], token_indices) for name in selected}
            mid_ratio_after, mid_energy_after, _ = squared_energy(maps_after["mid"])
            up_ratio_after, up_energy_after, _ = squared_energy(maps_after["up"])
            energy_after = (mid_energy_after + up_energy_after) / 2
            record.update({"mid_ratio_after": mid_ratio_after.item(), "up_ratio_after": up_ratio_after.item(), "energy_after": energy_after.item()})
            if inner_iter == MAX_INNER_ITERS - 1:
                save_named_heatmaps(maps_after, "inner_05_after", box)
        metrics.append(record)
        print(f"inner={inner_iter} t={float(timestep):.1f} | mid {mid_ratio.item():.8f}->{record['mid_ratio_after']:.8f} | up {up_ratio.item():.8f}->{record['up_ratio_after']:.8f} | E {energy.item():.8f}->{record['energy_after']:.8f} | grad={raw_gradient_norm.item():.8e} | candidate rel={candidate_relative_update.item():.8e}")
        for name, module in modules.items():
            module.set_processor(original_processors[name])
        del model_input, model_input_after, maps_before, maps_after, gradient, candidate_update, z_leaf
        z_current = z_updated.detach()
        del z_updated
finally:
    for name, module in modules.items():
        module.set_processor(original_processors[name])

if not metrics:
    raise RuntimeError("No inner-loop metrics recorded.")
with (OUTPUT_DIR / "inner_iteration_metrics.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(metrics[0]))
    writer.writeheader()
    writer.writerows(metrics)
print("[1] Paper backward-guidance inner-loop summary")
print(f"completed inner iterations: {len(metrics)} / {MAX_INNER_ITERS}")
print(f"stop reason: {stop_reason or 'completed MAX_INNER_ITERS'}")
print(f"peak allocated/reserved MiB: {torch.cuda.max_memory_allocated()/1024**2:.1f} / {torch.cuda.max_memory_reserved()/1024**2:.1f}")
print(f"metrics / heatmaps: {OUTPUT_DIR}")
