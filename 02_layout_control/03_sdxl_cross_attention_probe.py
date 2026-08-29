"""Capture one SDXL cross-attention map without changing the denoising trajectory."""

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
CAPTURE_STEP_INDEX = 4
TARGET_LAYER_NAME = "mid_block.attentions.0.transformer_blocks.0.attn2"
TARGET_WORD = "cabin"


class RecordingAttnProcessor:
    """AttnProcessor2_0-equivalent output path plus an optional one-step CPU capture."""

    def __init__(self):
        self.capture_enabled = False
        self.attention_probs_cpu = None
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
            # 只在目标 step 额外计算一次 softmax；随后立即脱离 GPU。
            scores = torch.matmul(query, key.transpose(-2, -1)) * (head_dim**-0.5)
            if attention_mask is not None:
                scores = scores + attention_mask
            attention_probs = torch.softmax(scores.float(), dim=-1)
            self.attention_probs_cpu = attention_probs.detach().cpu()
            self.qkv_shapes = {
                "query": tuple(query.shape),
                "key": tuple(key.shape),
                "value": tuple(value.shape),
                "attention_probs": tuple(attention_probs.shape),
            }
            del scores, attention_probs

        # 保持当前 AttnProcessor2_0 的真实输出计算：仍然使用 SDPA。
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


def find_word_token_indices(tokenizer, prompt, word):
    """Find all prompt positions matching the tokenizer's complete word token sequence."""
    prompt_ids = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids[0].tolist()
    word_ids = tokenizer(word, add_special_tokens=False).input_ids
    matches = []
    for start in range(len(prompt_ids) - len(word_ids) + 1):
        if prompt_ids[start : start + len(word_ids)] == word_ids:
            matches.extend(range(start, start + len(word_ids)))
    return prompt_ids, word_ids, sorted(set(matches))


def print_tokenization(name, tokenizer):
    prompt_ids, word_ids, word_indices = find_word_token_indices(tokenizer, PROMPT, TARGET_WORD)
    tokens = tokenizer.convert_ids_to_tokens(prompt_ids)
    print(f"{name} prompt token count: {len(prompt_ids)}")
    print(f"{name} '{TARGET_WORD}' standalone token ids: {word_ids}")
    print(f"{name} '{TARGET_WORD}' prompt indices: {word_indices}")
    for index in word_indices:
        print(f"  index={index}, id={prompt_ids[index]}, token={tokens[index]!r}, decoded={tokenizer.decode([prompt_ids[index]])!r}")
    return word_indices


@torch.no_grad()
def decode_and_save(pipe, latents, output_path):
    """Apply the same VAE scaling/upcast logic used by this pipeline version."""
    needs_upcasting = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
    if needs_upcasting:
        pipe.upcast_vae()
        latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)

    has_mean = hasattr(pipe.vae.config, "latents_mean") and pipe.vae.config.latents_mean is not None
    has_std = hasattr(pipe.vae.config, "latents_std") and pipe.vae.config.latents_std is not None
    if has_mean and has_std:
        mean = torch.tensor(pipe.vae.config.latents_mean).view(1, 4, 1, 1).to(latents)
        std = torch.tensor(pipe.vae.config.latents_std).view(1, 4, 1, 1).to(latents)
        latents_for_vae = latents * std / pipe.vae.config.scaling_factor + mean
    else:
        latents_for_vae = latents / pipe.vae.config.scaling_factor

    image_tensor = pipe.vae.decode(latents_for_vae, return_dict=False)[0]
    if needs_upcasting:
        pipe.vae.to(dtype=torch.float16)
    if pipe.watermark is not None:
        image_tensor = pipe.watermark.apply_watermark(image_tensor)
    image = pipe.image_processor.postprocess(image_tensor, output_type="pil")[0]
    image.save(output_path)


hf_home = os.environ.get("HF_HOME")
if not hf_home:
    raise RuntimeError("HF_HOME is not set. Configure the shared data-disk cache first.")
if not (Path(hf_home) / "hub" / "models--stabilityai--stable-diffusion-xl-base-1.0").is_dir():
    raise RuntimeError("The verified SDXL cache is not available under HF_HOME.")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

# 复用 baseline 的 float16 + CUDA，并强制不访问网络。
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    local_files_only=True,
)
pipe.to("cuda")
device = torch.device("cuda")
do_classifier_free_guidance = GUIDANCE_SCALE > 1.0
pipe._guidance_scale = GUIDANCE_SCALE
pipe._guidance_rescale = 0.0

print("[1] Tokenization")
tokenizer_1_indices = print_tokenization("tokenizer", pipe.tokenizer)
tokenizer_2_indices = print_tokenization("tokenizer_2", pipe.tokenizer_2)
target_indices = sorted(set(tokenizer_1_indices) | set(tokenizer_2_indices))
if not target_indices:
    raise RuntimeError(f"Could not locate {TARGET_WORD!r} in either tokenizer output.")
print(f"attention positions used for {TARGET_WORD!r}: {target_indices}")
if tokenizer_1_indices != tokenizer_2_indices:
    print("Tokenizer positions differ; using their union because SDXL concatenates features by sequence position.")
print()

# 仅替换一个指定 attn2；其余 139 个 processor 保持原样。
target_attention = pipe.unet.get_submodule(TARGET_LAYER_NAME)
original_processor = target_attention.processor
recording_processor = RecordingAttnProcessor()
target_attention.set_processor(recording_processor)

try:
    with torch.no_grad():
        # 文本编码与 CFG 条件准备。
        positive_prompt_embeds, negative_prompt_embeds, positive_pooled, negative_pooled = pipe.encode_prompt(
            prompt=PROMPT,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=None,
        )
        prompt_embeds = torch.cat([negative_prompt_embeds, positive_prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled, positive_pooled], dim=0)

        # Scheduler 和固定 seed latent 初始化，复用显式 denoising 脚本的路径。
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=device)
        timesteps = pipe.scheduler.timesteps
        extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta=0.0)
        latents = pipe.prepare_latents(
            1,
            pipe.unet.config.in_channels,
            HEIGHT,
            WIDTH,
            positive_prompt_embeds.dtype,
            device,
            generator,
            None,
        )

        # SDXL extra conditioning：pooled text embedding + original/crop/target size ids。
        add_time_ids_single = pipe._get_add_time_ids(
            (HEIGHT, WIDTH),
            (0, 0),
            (HEIGHT, WIDTH),
            dtype=positive_prompt_embeds.dtype,
            text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
        )
        add_time_ids = torch.cat([add_time_ids_single, add_time_ids_single], dim=0).to(device)
        added_cond_kwargs = {"text_embeds": add_text_embeds.to(device), "time_ids": add_time_ids}
        prompt_embeds = prompt_embeds.to(device)

        print("[2] Capture setup")
        print(f"target layer name: {TARGET_LAYER_NAME}")
        print(f"target layer processor before replacement: {type(original_processor).__name__}")
        print(f"capture step index: {CAPTURE_STEP_INDEX}")
        print(f"timesteps: {timesteps.detach().cpu().tolist()}")
        print(f"prompt_embeds.shape: {tuple(prompt_embeds.shape)}")
        print(f"add_text_embeds.shape: {tuple(add_text_embeds.shape)}")
        print(f"add_time_ids.shape: {tuple(add_time_ids.shape)}")

        # 显式 denoising：只在一个中间 step 打开记录开关，其余步维持正常 SDPA 输出。
        for step_index, timestep in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
            recording_processor.capture_enabled = step_index == CAPTURE_STEP_INDEX

            noise_pred = pipe.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=None,
                cross_attention_kwargs=None,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + GUIDANCE_SCALE * (noise_text - noise_uncond)
            latents = pipe.scheduler.step(
                noise_pred, timestep, latents, **extra_step_kwargs, return_dict=False
            )[0]

            if step_index == CAPTURE_STEP_INDEX:
                print(f"capture timestep: {float(timestep)}")
                print(f"capture latent_model_input.shape: {tuple(latent_model_input.shape)}")

    if recording_processor.attention_probs_cpu is None:
        raise RuntimeError("The target attention layer did not record probabilities at the requested step.")
finally:
    # 恢复本次进程中该层原 processor；不永久改变 UNet 的任何层。
    target_attention.set_processor(original_processor)

print("[3] Captured cross-attention")
print(f"Q shape [batch, heads, image queries, head dim]: {recording_processor.qkv_shapes['query']}")
print(f"K shape [batch, heads, text tokens, head dim]: {recording_processor.qkv_shapes['key']}")
print(f"V shape [batch, heads, text tokens, head dim]: {recording_processor.qkv_shapes['value']}")
print(
    "raw attention probability shape [CFG batch, heads, image query tokens, text tokens]: "
    f"{recording_processor.qkv_shapes['attention_probs']}"
)

# CFG concat order is [negative/unconditional, positive/conditional], so branch 1 is the prompt branch.
conditional_probs = recording_processor.attention_probs_cpu[1]
mean_over_heads = conditional_probs.mean(dim=0)
cabin_attention = mean_over_heads[:, target_indices].mean(dim=-1)
query_tokens = cabin_attention.numel()
spatial_side = math.isqrt(query_tokens)
if spatial_side * spatial_side != query_tokens:
    raise RuntimeError(f"Cannot reliably reshape {query_tokens} query tokens into a square spatial map.")
heatmap = cabin_attention.reshape(spatial_side, spatial_side)
heatmap_min, heatmap_max = heatmap.min(), heatmap.max()
heatmap_for_display = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)

output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
heatmap_path = output_dir / "cross_attention_cabin_step4.png"
image_path = output_dir / "sdxl_cross_attention_probe_seed42.png"
plt.imsave(heatmap_path, heatmap_for_display.numpy(), cmap="magma")
# denoising 已完成：将不再需要的 UNet 与两个 text encoder 暂移到 CPU，为 VAE float32 upcast/decode 释放显存。
pipe.unet.to("cpu")
pipe.text_encoder.to("cpu")
pipe.text_encoder_2.to("cpu")
torch.cuda.synchronize()
torch.cuda.empty_cache()
decode_and_save(pipe, latents, image_path)

print("[4] Saved outputs")
print(f"conditional branch index: 1")
print(f"heads averaged: {conditional_probs.shape[0]}")
print(f"query token count: {query_tokens}")
print(f"text token count: {conditional_probs.shape[-1]}")
print(f"heatmap shape: {tuple(heatmap.shape)}")
print(f"heatmap path: {heatmap_path}")
print(f"generated image path: {image_path}")
