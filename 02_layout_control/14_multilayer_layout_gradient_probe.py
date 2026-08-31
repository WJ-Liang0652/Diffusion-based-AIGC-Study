"""Probe a differentiable 3-layer SDXL cross-attention layout objective at t=501."""

import math
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
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "multilayer_layout_gradient_probe"


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

cross_modules = {}
for block_name, block in pipe.unet.named_modules():
    if isinstance(block, BasicTransformerBlock):
        for attention_name in ("attn1", "attn2"):
            attention = getattr(block, attention_name, None)
            if attention is not None and attention.is_cross_attention:
                cross_modules[f"{block_name}.{attention_name}"] = attention
original_processors = {name: module.processor for name, module in cross_modules.items()}

try:
    with torch.no_grad():
        positive, negative, positive_pooled, negative_pooled = pipe.encode_prompt(prompt=PROMPT, device=device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=None)
        prompt_embeds = torch.cat([negative, positive]).to(device)
        add_text_embeds = torch.cat([negative_pooled, positive_pooled]).to(device)
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=device)
        timesteps = pipe.scheduler.timesteps
        extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta=0.0)
        latents = pipe.prepare_latents(1, pipe.unet.config.in_channels, HEIGHT, WIDTH, positive.dtype, device, generator)
        time_ids = pipe._get_add_time_ids((HEIGHT, WIDTH), (0, 0), (HEIGHT, WIDTH), positive.dtype, pipe.text_encoder_2.config.projection_dim)
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": torch.cat([time_ids, time_ids]).to(device)}
        for timestep in timesteps[:TARGET_STEP_INDEX]:
            model_input = pipe.scheduler.scale_model_input(torch.cat([latents] * 2), timestep)
            noise = pipe.unet(model_input, timestep, encoder_hidden_states=prompt_embeds, added_cond_kwargs=added_cond_kwargs, return_dict=False)[0]
            uncond, text = noise.chunk(2)
            latents = pipe.scheduler.step(uncond + GUIDANCE_SCALE * (text - uncond), timestep, latents, **extra_step_kwargs, return_dict=False)[0]

    target_timestep = timesteps[TARGET_STEP_INDEX]
    # A no-grad shape-only probe selects actual 1024-query candidates on this forward path.
    shapes = {}
    for name, module in cross_modules.items():
        module.set_processor(ShapeProbeProcessor(name, shapes))
    with torch.no_grad():
        model_input = pipe.scheduler.scale_model_input(torch.cat([latents] * 2), target_timestep)
        _ = pipe.unet(model_input, target_timestep, encoder_hidden_states=prompt_embeds, added_cond_kwargs=added_cond_kwargs, return_dict=False)[0]
    for name, module in cross_modules.items():
        module.set_processor(original_processors[name])
    candidates = {which: [name for name, shape in shapes.items() if region(name) == which and shape[-2] == 1024] for which in ("down", "mid", "up")}
    if any(not names for names in candidates.values()):
        raise RuntimeError(f"Could not find actual 1024-query layer in every region: {candidates}")
    selected = {which: names[0] for which, names in candidates.items()}
    print("actual 1024-query candidates selected:")
    for which, name in selected.items():
        print(f"  {which}: {name} | Q shape={shapes[name]}")

    captured = {}
    for name in selected.values():
        cross_modules[name].set_processor(DifferentiableMapProcessor(name, captured))
    x = latents.detach().requires_grad_(True)
    torch.cuda.reset_peak_memory_stats()
    model_input = pipe.scheduler.scale_model_input(torch.cat([x] * 2), target_timestep)
    _ = pipe.unet(model_input, target_timestep, encoder_hidden_states=prompt_embeds, added_cond_kwargs=added_cond_kwargs, return_dict=False)[0]
    if set(captured) != set(selected.values()):
        raise RuntimeError(f"Missing target map capture: expected {selected.values()}, got {captured.keys()}")
    maps = {which: reduce_to_map(captured[name], token_indices) for which, name in selected.items()}
    aggregate_map = torch.stack(list(maps.values())).mean(dim=0)
    aggregate_ratio, aggregate_loss, box = objective(aggregate_map)
    mid_ratio, mid_loss, _ = objective(maps["mid"])
    gradient = torch.autograd.grad(aggregate_loss, x)[0]
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("Aggregate gradient has NaN or Inf.")
    print("[1] 3-layer differentiable aggregate")
    print(f"step_index / timestep: {TARGET_STEP_INDEX} / {float(target_timestep)}")
    print(f"cabin indices: tokenizer={indices_1}, tokenizer_2={indices_2}, used={token_indices}")
    for which, raw_map in maps.items():
        print(f"{which} map {selected[which]} | min/max/mean: {raw_map.min().item():.8e} / {raw_map.max().item():.8e} / {raw_map.mean().item():.8e}")
    print(f"single mid inside_ratio / loss: {mid_ratio.item():.8f} / {mid_loss.item():.8f}")
    print(f"aggregate inside_ratio / loss: {aggregate_ratio.item():.8f} / {aggregate_loss.item():.8f}")
    print(f"target box grid / area ratio: x=[{box[0]}, {box[1]}), y=[{box[2]}, {box[3]}) / {((box[1]-box[0])*(box[3]-box[2])) / 32**2:.8f}")
    print(f"aggregate grad shape / norm / abs mean: {tuple(gradient.shape)} / {gradient.norm().item():.8e} / {gradient.abs().mean().item():.8e}")
    print(f"aggregate grad NaN / Inf: {torch.isnan(gradient).any().item()} / {torch.isinf(gradient).any().item()}")
    print(f"peak allocated/reserved MiB: {torch.cuda.max_memory_allocated()/1024**2:.1f} / {torch.cuda.max_memory_reserved()/1024**2:.1f}")
    for which, raw_map in maps.items():
        save_heatmap(raw_map, which, box)
    save_heatmap(aggregate_map, "aggregate", box)
finally:
    for name, module in cross_modules.items():
        module.set_processor(original_processors[name])
