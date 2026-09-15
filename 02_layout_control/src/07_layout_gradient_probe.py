"""Verify that a single SDXL cross-attention layout loss has a latent gradient."""

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
EPS = 1e-8


class DifferentiableRecordingProcessor:
    """AttnProcessor2_0 output with an optional graph-preserving attention capture."""

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
            # 此 probability 保留计算图：不能 detach 或转 CPU，loss 要对 latent 求导。
            scores = torch.matmul(query, key.transpose(-2, -1)) * (head_dim**-0.5)
            if attention_mask is not None:
                scores = scores + attention_mask
            self.attention_probs = torch.softmax(scores.float(), dim=-1)
            self.qkv_shapes = {
                "query": tuple(query.shape),
                "key": tuple(key.shape),
                "value": tuple(value.shape),
                "attention_probs": tuple(self.attention_probs.shape),
            }

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
        PROMPT,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids[0].tolist()
    word_ids = tokenizer(word, add_special_tokens=False).input_ids
    indices = []
    for start in range(len(prompt_ids) - len(word_ids) + 1):
        if prompt_ids[start : start + len(word_ids)] == word_ids:
            indices.extend(range(start, start + len(word_ids)))
    return sorted(set(indices))


def freeze_parameters(*models):
    parameter_count = 0
    for model in models:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter_count += parameter.numel()
    return parameter_count


hf_home = os.environ.get("HF_HOME")
if not hf_home:
    raise RuntimeError("HF_HOME is not set. Configure the shared data-disk cache first.")
if not (Path(hf_home) / "hub" / "models--stabilityai--stable-diffusion-xl-base-1.0").is_dir():
    raise RuntimeError("The verified SDXL cache is not available under HF_HOME.")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

# 复用既有模型、float16、CUDA 与 cache；不下载任何内容。
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, local_files_only=True
)
pipe.to("cuda")
device = torch.device("cuda")
pipe._guidance_scale = GUIDANCE_SCALE
pipe._guidance_rescale = 0.0
frozen_parameter_count = freeze_parameters(pipe.unet, pipe.text_encoder, pipe.text_encoder_2, pipe.vae)

# 通过两套 tokenizer 实际寻找 cabin 位置，不硬编码。
indices_1 = find_word_indices(pipe.tokenizer, TARGET_WORD)
indices_2 = find_word_indices(pipe.tokenizer_2, TARGET_WORD)
target_token_indices = sorted(set(indices_1) | set(indices_2))
if not target_token_indices:
    raise RuntimeError(f"Could not locate {TARGET_WORD!r} in either tokenizer output.")
print(f"cabin tokenizer indices: tokenizer={indices_1}, tokenizer_2={indices_2}")
print(f"attention positions used: {target_token_indices}")
print(f"frozen parameter count: {frozen_parameter_count}")

# 只临时替换一个已验证可 reshape 为 32x32 的代表性 cross-attention 层。
target_attention = pipe.unet.get_submodule(TARGET_LAYER_NAME)
if not target_attention.is_cross_attention:
    raise RuntimeError(f"Target module is not cross-attention: {TARGET_LAYER_NAME}")
original_processor = target_attention.processor
recording_processor = DifferentiableRecordingProcessor()
target_attention.set_processor(recording_processor)

try:
    # 文本和初始化只需常数，不保留它们的计算图。
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
        time_ids_single = pipe._get_add_time_ids(
            (HEIGHT, WIDTH), (0, 0), (HEIGHT, WIDTH), positive.dtype,
            pipe.text_encoder_2.config.projection_dim,
        )
        added_cond_kwargs = {
            "text_embeds": add_text_embeds,
            "time_ids": torch.cat([time_ids_single, time_ids_single], dim=0).to(device),
        }

        # 前面四个 step 正常运行，但不跨 timestep 保留计算图。
        for step_index, timestep in enumerate(timesteps[:TARGET_STEP_INDEX]):
            latent_model_input = pipe.scheduler.scale_model_input(torch.cat([latents] * 2), timestep)
            recording_processor.capture_enabled = False
            noise_pred = pipe.unet(
                latent_model_input, timestep, encoder_hidden_states=prompt_embeds,
                timestep_cond=None, cross_attention_kwargs=None,
                added_cond_kwargs=added_cond_kwargs, return_dict=False,
            )[0]
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + GUIDANCE_SCALE * (noise_text - noise_uncond)
            latents = pipe.scheduler.step(
                noise_pred, timestep, latents, **extra_step_kwargs, return_dict=False
            )[0]

    # 目标 x_t 成为叶子变量：计算图只从此 timestep 开始。
    target_timestep = timesteps[TARGET_STEP_INDEX]
    latents = latents.detach().requires_grad_(True)
    torch.cuda.reset_peak_memory_stats()
    latent_model_input = pipe.scheduler.scale_model_input(torch.cat([latents] * 2), target_timestep)
    recording_processor.capture_enabled = True
    noise_pred = pipe.unet(
        latent_model_input, target_timestep, encoder_hidden_states=prompt_embeds,
        timestep_cond=None, cross_attention_kwargs=None,
        added_cond_kwargs=added_cond_kwargs, return_dict=False,
    )[0]

    if recording_processor.attention_probs is None:
        raise RuntimeError("Target layer did not expose differentiable attention probabilities.")

    # 从保留计算图的 raw probability 中取 conditional branch、cabin token并平均 heads。
    conditional_probs = recording_processor.attention_probs[1]
    raw_attention_vector = conditional_probs[:, :, target_token_indices].mean(dim=0).mean(dim=-1)
    query_tokens = raw_attention_vector.numel()
    spatial_side = math.isqrt(query_tokens)
    if spatial_side * spatial_side != query_tokens:
        raise RuntimeError(f"Cannot reshape {query_tokens} query tokens into a square attention map.")
    raw_attention_map = raw_attention_vector.reshape(spatial_side, spatial_side)

    x0, y0, x1, y1 = TARGET_BOX
    x_start, x_end = math.floor(x0 * spatial_side), math.ceil(x1 * spatial_side)
    y_start, y_end = math.floor(y0 * spatial_side), math.ceil(y1 * spatial_side)
    total_mass = raw_attention_map.sum()
    inside_mass = raw_attention_map[y_start:y_end, x_start:x_end].sum()
    inside_ratio = inside_mass / (total_mass + EPS)
    layout_loss = 1.0 - inside_ratio

    # 只求 dL/dx_t；不调用 backward，不累积任何模型参数梯度。
    gradient = torch.autograd.grad(layout_loss, latents)[0]
finally:
    target_attention.set_processor(original_processor)

# 输出数值后立即释放 graph 所需的中间 tensor；本 probe 不继续 scheduler/VAE。
print("[1] Gradient probe result")
print(f"target step_index / timestep: {TARGET_STEP_INDEX} / {float(target_timestep)}")
print(f"target attention layer: {TARGET_LAYER_NAME}")
print(f"target processor before replacement: {type(original_processor).__name__}")
print(f"latent shape: {tuple(latents.shape)}")
print(f"latents.requires_grad: {latents.requires_grad}")
print(f"Q/K/V/probability shapes: {recording_processor.qkv_shapes}")
print(f"raw attention map shape: {tuple(raw_attention_map.shape)}")
print(f"target box grid: x=[{x_start}, {x_end}), y=[{y_start}, {y_end})")
print(f"inside_ratio: {inside_ratio.item():.8f}")
print(f"layout_loss: {layout_loss.item():.8f}")
print(f"gradient shape: {tuple(gradient.shape)}")
print(f"gradient dtype: {gradient.dtype}")
print(f"grad min/max/mean: {gradient.min().item():.8e} / {gradient.max().item():.8e} / {gradient.mean().item():.8e}")
print(f"grad abs mean: {gradient.abs().mean().item():.8e}")
print(f"grad norm: {gradient.norm().item():.8e}")
print(f"grad has NaN: {torch.isnan(gradient).any().item()}")
print(f"grad has Inf: {torch.isinf(gradient).any().item()}")
print(f"gradient shape matches latents: {gradient.shape == latents.shape}")
print(f"peak allocated/reserved MiB: {torch.cuda.max_memory_allocated() / 1024**2:.1f} / {torch.cuda.max_memory_reserved() / 1024**2:.1f}")

# 显式断开并释放本次 target-step graph；没有执行任何 latent 更新。
recording_processor.attention_probs = None
del noise_pred, conditional_probs, raw_attention_vector, raw_attention_map, layout_loss, gradient
