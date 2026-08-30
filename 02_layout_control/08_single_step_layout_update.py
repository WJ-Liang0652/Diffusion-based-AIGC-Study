"""Verify one negative-gradient latent update at a single SDXL denoising timestep."""

import math
import os
from pathlib import Path

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
ETAS = (1.0, 10.0, 100.0)
EPS = 1e-8


class SingleLayerUpdateProcessor:
    """AttnProcessor2_0 output plus either differentiable or CPU-reduced map capture."""

    def __init__(self, token_indices):
        self.token_indices = token_indices
        self.capture_mode = None  # None, "grad", or "cpu"
        self.attention_probs = None
        self.raw_map_cpu = None
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

        if self.capture_mode is not None:
            # 与 AttnProcessor2_0 的缩放一致；仅这个 target layer 显式获取 probability。
            scores = torch.matmul(query, key.transpose(-2, -1)) * (head_dim**-0.5)
            if attention_mask is not None:
                scores = scores + attention_mask
            probs = torch.softmax(scores.float(), dim=-1)
            self.qkv_shapes = {
                "query": tuple(query.shape), "key": tuple(key.shape),
                "value": tuple(value.shape), "attention_probs": tuple(probs.shape),
            }
            if self.capture_mode == "grad":
                # 保留 graph，供 autograd.grad(layout_loss, x) 使用。
                self.attention_probs = probs
            else:
                # verification 不需要二阶梯度：立即压缩为 cabin map 并移至 CPU。
                conditional = probs[1]
                vector = conditional[:, :, self.token_indices].mean(dim=0).mean(dim=-1)
                side = math.isqrt(vector.numel())
                if side * side != vector.numel():
                    raise RuntimeError(f"Non-square query layout: {vector.numel()}")
                self.raw_map_cpu = vector.reshape(side, side).detach().float().cpu()
            del scores

        # 保持本机 AttnProcessor2_0 的 SDPA 输出路径。
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
    indices = []
    for start in range(len(prompt_ids) - len(word_ids) + 1):
        if prompt_ids[start : start + len(word_ids)] == word_ids:
            indices.extend(range(start, start + len(word_ids)))
    return sorted(set(indices))


def freeze_parameters(*models):
    for model in models:
        for parameter in model.parameters():
            parameter.requires_grad_(False)


def objective_from_raw_map(raw_map):
    """Compute raw attention mass objective; no display normalization is used."""
    side = raw_map.shape[0]
    x0, y0, x1, y1 = TARGET_BOX
    x_start, x_end = math.floor(x0 * side), math.ceil(x1 * side)
    y_start, y_end = math.floor(y0 * side), math.ceil(y1 * side)
    total_mass = raw_map.sum()
    inside_mass = raw_map[y_start:y_end, x_start:x_end].sum()
    inside_ratio = inside_mass / (total_mass + EPS)
    return inside_mass, total_mass, inside_ratio, 1.0 - inside_ratio, (x_start, x_end, y_start, y_end)


hf_home = os.environ.get("HF_HOME")
if not hf_home:
    raise RuntimeError("HF_HOME is not set. Configure the shared data-disk cache first.")
if not (Path(hf_home) / "hub" / "models--stabilityai--stable-diffusion-xl-base-1.0").is_dir():
    raise RuntimeError("The verified SDXL cache is not available under HF_HOME.")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

# 复用既有 SDXL 环境；参数冻结后只允许 x_t 保留梯度。
pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, local_files_only=True)
pipe.to("cuda")
device = torch.device("cuda")
pipe._guidance_scale = GUIDANCE_SCALE
pipe._guidance_rescale = 0.0
freeze_parameters(pipe.unet, pipe.text_encoder, pipe.text_encoder_2, pipe.vae)

indices_1 = find_word_indices(pipe.tokenizer, TARGET_WORD)
indices_2 = find_word_indices(pipe.tokenizer_2, TARGET_WORD)
token_indices = sorted(set(indices_1) | set(indices_2))
if not token_indices:
    raise RuntimeError(f"Could not locate {TARGET_WORD!r}.")
print(f"cabin indices: tokenizer={indices_1}, tokenizer_2={indices_2}, used={token_indices}")

target_attention = pipe.unet.get_submodule(TARGET_LAYER_NAME)
if not target_attention.is_cross_attention:
    raise RuntimeError(f"Target module is not cross-attention: {TARGET_LAYER_NAME}")
original_processor = target_attention.processor
processor = SingleLayerUpdateProcessor(token_indices)
target_attention.set_processor(processor)

try:
    with torch.no_grad():
        # 准备固定文本条件、scheduler 和 seeded latent。
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

        # step 0..3 正常 denoise，不保留跨 step 的图。
        for timestep in timesteps[:TARGET_STEP_INDEX]:
            processor.capture_mode = None
            model_input = pipe.scheduler.scale_model_input(torch.cat([latents] * 2), timestep)
            noise_pred = pipe.unet(
                model_input, timestep, encoder_hidden_states=prompt_embeds,
                timestep_cond=None, cross_attention_kwargs=None,
                added_cond_kwargs=added_cond_kwargs, return_dict=False,
            )[0]
            noise_uncond, noise_text = noise_pred.chunk(2)
            latents = pipe.scheduler.step(
                noise_uncond + GUIDANCE_SCALE * (noise_text - noise_uncond),
                timestep, latents, **extra_step_kwargs, return_dict=False,
            )[0]

    target_timestep = timesteps[TARGET_STEP_INDEX]
    x_original = latents.detach()
    x = x_original.clone().requires_grad_(True)
    torch.cuda.reset_peak_memory_stats()

    # Baseline：只在当前 x_t 到目标 attention loss 的路径保留 autograd graph。
    processor.capture_mode = "grad"
    model_input = pipe.scheduler.scale_model_input(torch.cat([x] * 2), target_timestep)
    noise_pred = pipe.unet(
        model_input, target_timestep, encoder_hidden_states=prompt_embeds,
        timestep_cond=None, cross_attention_kwargs=None,
        added_cond_kwargs=added_cond_kwargs, return_dict=False,
    )[0]
    if processor.attention_probs is None:
        raise RuntimeError("Target attention probability was not captured.")
    conditional = processor.attention_probs[1]
    raw_vector = conditional[:, :, token_indices].mean(dim=0).mean(dim=-1)
    side = math.isqrt(raw_vector.numel())
    if side * side != raw_vector.numel():
        raise RuntimeError(f"Cannot reshape {raw_vector.numel()} query tokens into a square map.")
    raw_map = raw_vector.reshape(side, side)
    original_inside, original_total, original_ratio, original_loss, box_grid = objective_from_raw_map(raw_map)
    gradient = torch.autograd.grad(original_loss, x)[0]

    # 释放 baseline target-step graph；三个 eta 验证都不要求二阶梯度。
    processor.attention_probs = None
    del noise_pred, conditional, raw_vector, raw_map, model_input
    torch.cuda.empty_cache()

    results = []
    for eta in ETAS:
        # 每一组都从同一个 x_original 出发，绝不累积前一组更新。
        x_updated = x_original - eta * gradient
        update = x_updated - x_original
        update_norm = update.norm()
        relative_update = update_norm / (x_original.norm() + EPS)

        with torch.no_grad():
            processor.capture_mode = "cpu"
            processor.raw_map_cpu = None
            verification_input = pipe.scheduler.scale_model_input(torch.cat([x_updated] * 2), target_timestep)
            _ = pipe.unet(
                verification_input, target_timestep, encoder_hidden_states=prompt_embeds,
                timestep_cond=None, cross_attention_kwargs=None,
                added_cond_kwargs=added_cond_kwargs, return_dict=False,
            )[0]
        if processor.raw_map_cpu is None:
            raise RuntimeError(f"Verification attention map was not captured for eta={eta}.")
        inside_mass, total_mass, ratio, loss, verification_box = objective_from_raw_map(processor.raw_map_cpu)
        if verification_box != box_grid:
            raise RuntimeError("Verification attention grid changed unexpectedly.")
        results.append({
            "eta": eta,
            "inside_mass": inside_mass.item(),
            "total_mass": total_mass.item(),
            "inside_ratio": ratio.item(),
            "layout_loss": loss.item(),
            "delta_ratio": (ratio - original_ratio).item(),
            "delta_loss": (loss - original_loss).item(),
            "update_norm": update_norm.item(),
            "relative_update": relative_update.item(),
            "has_nan": bool(torch.isnan(x_updated).any().item()),
            "has_inf": bool(torch.isinf(x_updated).any().item()),
        })
        del x_updated, update, verification_input
        processor.raw_map_cpu = None
finally:
    target_attention.set_processor(original_processor)

print("[1] Single-step layout update")
print(f"target step_index / timestep: {TARGET_STEP_INDEX} / {float(target_timestep)}")
print(f"target layer: {TARGET_LAYER_NAME}")
print(f"target box grid: x=[{box_grid[0]}, {box_grid[1]}), y=[{box_grid[2]}, {box_grid[3]})")
print(f"latent shape: {tuple(x_original.shape)}")
print(f"original inside_mass / total_mass: {original_inside.item():.8f} / {original_total.item():.8f}")
print(f"original inside_ratio: {original_ratio.item():.8f}")
print(f"original layout_loss: {original_loss.item():.8f}")
print(f"gradient norm: {gradient.norm().item():.8e}")
print("eta | inside_ratio | layout_loss | delta_ratio | delta_loss | update_norm | relative_update | NaN | Inf")
for result in results:
    print(
        f"{result['eta']:>3.0f} | {result['inside_ratio']:.8f} | {result['layout_loss']:.8f} | "
        f"{result['delta_ratio']:+.8e} | {result['delta_loss']:+.8e} | "
        f"{result['update_norm']:.8e} | {result['relative_update']:.8e} | "
        f"{result['has_nan']} | {result['has_inf']}"
    )
print(f"peak allocated/reserved MiB: {torch.cuda.max_memory_allocated() / 1024**2:.1f} / {torch.cuda.max_memory_reserved() / 1024**2:.1f}")

# 只验证一次更新闭环；没有 scheduler.step、没有 VAE decode、没有 x_t 写回或更新。
del gradient, original_loss
