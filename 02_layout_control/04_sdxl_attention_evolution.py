"""Visualize one SDXL cross-attention layer across three denoising steps."""

import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline
from PIL import Image


MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
PROMPT = "A small red cabin beside a calm mountain lake at sunrise, realistic photograph"
SEED = 42
HEIGHT = 1024
WIDTH = 1024
NUM_INFERENCE_STEPS = 10
GUIDANCE_SCALE = 7.0
TARGET_LAYER_NAME = "mid_block.attentions.0.transformer_blocks.0.attn2"
CAPTURE_STEP_INDICES = {1, 4, 7}
TARGET_WORDS = ("cabin", "lake", "sunrise")


class MultiStepRecordingAttnProcessor:
    """Keep AttnProcessor2_0 output behavior and capture selected probabilities on CPU."""

    def __init__(self):
        self.capture_step_index = None
        self.captures_cpu = {}
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

        if self.capture_step_index is not None:
            # 仅为可视化额外求一次 softmax，随后 float + detach + CPU，不滞留 GPU。
            scores = torch.matmul(query, key.transpose(-2, -1)) * (head_dim**-0.5)
            if attention_mask is not None:
                scores = scores + attention_mask
            attention_probs = torch.softmax(scores.float(), dim=-1)
            self.captures_cpu[self.capture_step_index] = attention_probs.detach().float().cpu()
            self.qkv_shapes = {
                "query": tuple(query.shape),
                "key": tuple(key.shape),
                "value": tuple(value.shape),
                "attention_probs": tuple(attention_probs.shape),
            }
            del scores, attention_probs

        # 与本机 AttnProcessor2_0 一样，使用 SDPA 计算本层的真实输出。
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


def find_word_token_indices(tokenizer, word):
    """Locate every occurrence of the tokenizer's word-piece sequence in the padded prompt."""
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
    return prompt_ids, word_ids, sorted(set(indices))


def inspect_target_tokens(pipe):
    """Print both tokenizer results and return the sequence positions used per target word."""
    target_indices = {}
    for word in TARGET_WORDS:
        print(f"token: {word}")
        per_tokenizer = []
        for name, tokenizer in (("tokenizer", pipe.tokenizer), ("tokenizer_2", pipe.tokenizer_2)):
            prompt_ids, word_ids, indices = find_word_token_indices(tokenizer, word)
            pieces = tokenizer.convert_ids_to_tokens(prompt_ids)
            print(f"  {name} standalone ids: {word_ids}; prompt indices: {indices}")
            for index in indices:
                print(f"    index={index}, token={pieces[index]!r}, decoded={tokenizer.decode([prompt_ids[index]])!r}")
            per_tokenizer.append(indices)
        chosen = sorted(set(per_tokenizer[0]) | set(per_tokenizer[1]))
        if not chosen:
            raise RuntimeError(f"Could not locate {word!r} in the prompt token sequence.")
        if per_tokenizer[0] != per_tokenizer[1]:
            print("  tokenizer indices differ; using their union over SDXL's shared sequence positions.")
        if len(chosen) > 1:
            print("  multiple word pieces found; their token maps will be averaged equally.")
        target_indices[word] = chosen
    return target_indices


@torch.no_grad()
def decode_image(pipe, latents, output_path):
    """Use the same VAE scaling/upcast behavior as StableDiffusionXLPipeline.__call__."""
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
    return image


def save_token_visualizations(raw_map, token, step_index, timestep, image, output_dir):
    """Save raw data, a standalone heatmap, and a separate upsampled overlay."""
    stem = f"{token}_step{step_index}_t{int(timestep)}"
    raw_path = output_dir / "raw_maps" / f"{stem}.pt"
    heatmap_path = output_dir / "heatmaps" / f"{stem}_heatmap.png"
    overlay_path = output_dir / "overlays" / f"{stem}_overlay.png"

    # raw_map remains an unnormalized attention probability map for future analysis.
    torch.save(raw_map, raw_path)
    # Matplotlib maps the raw values to colors for a standalone view without modifying raw_map.
    plt.imsave(heatmap_path, raw_map.numpy(), cmap="magma")

    # visualization_map is a separate min-max-normalized copy, only for display/overlay.
    visualization_map = (raw_map - raw_map.min()) / (raw_map.max() - raw_map.min() + 1e-8)
    upsampled_map = F.interpolate(
        visualization_map[None, None],
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )[0, 0]
    heatmap_rgb = plt.get_cmap("magma")(upsampled_map.numpy())[..., :3]
    base_rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    overlay = (0.58 * base_rgb + 0.42 * heatmap_rgb).clip(0, 1)
    Image.fromarray((overlay * 255).astype(np.uint8)).save(overlay_path)
    return raw_path, heatmap_path, overlay_path


hf_home = os.environ.get("HF_HOME")
if not hf_home:
    raise RuntimeError("HF_HOME is not set. Configure the shared data-disk cache first.")
if not (Path(hf_home) / "hub" / "models--stabilityai--stable-diffusion-xl-base-1.0").is_dir():
    raise RuntimeError("The verified SDXL cache is not available under HF_HOME.")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

# 复用 baseline 的模型、float16、CUDA、prompt、seed 与 CFG 设置，且禁止网络下载。
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    local_files_only=True,
)
pipe.to("cuda")
device = torch.device("cuda")
pipe._guidance_scale = GUIDANCE_SCALE
pipe._guidance_rescale = 0.0

print("[1] Tokenization")
token_to_indices = inspect_target_tokens(pipe)
print(f"token -> indices: {token_to_indices}")

# 只替换这个 attn2；所有其他 processor 维持原 AttnProcessor2_0。
target_attention = pipe.unet.get_submodule(TARGET_LAYER_NAME)
original_processor = target_attention.processor
recording_processor = MultiStepRecordingAttnProcessor()
target_attention.set_processor(recording_processor)

try:
    with torch.no_grad():
        # 文本编码和 SDXL CFG 条件。
        positive, negative, positive_pooled, negative_pooled = pipe.encode_prompt(
            prompt=PROMPT,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=None,
        )
        prompt_embeds = torch.cat([negative, positive], dim=0).to(device)
        add_text_embeds = torch.cat([negative_pooled, positive_pooled], dim=0).to(device)

        # Scheduler、固定随机初始 latent 和 SDXL size conditioning。
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=device)
        timesteps = pipe.scheduler.timesteps
        extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta=0.0)
        latents = pipe.prepare_latents(1, pipe.unet.config.in_channels, HEIGHT, WIDTH, positive.dtype, device, generator)
        time_ids_single = pipe._get_add_time_ids(
            (HEIGHT, WIDTH), (0, 0), (HEIGHT, WIDTH), positive.dtype, pipe.text_encoder_2.config.projection_dim
        )
        added_cond_kwargs = {
            "text_embeds": add_text_embeds,
            "time_ids": torch.cat([time_ids_single, time_ids_single], dim=0).to(device),
        }

        print("[2] Capture timesteps")
        print(f"target layer: {TARGET_LAYER_NAME}")
        print(f"timesteps: {timesteps.detach().cpu().tolist()}")
        for step_index, timestep in enumerate(timesteps):
            latent_model_input = pipe.scheduler.scale_model_input(torch.cat([latents] * 2), timestep)
            recording_processor.capture_step_index = step_index if step_index in CAPTURE_STEP_INDICES else None
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
            latents = pipe.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs, return_dict=False)[0]
            if step_index in CAPTURE_STEP_INDICES:
                print(f"step_index={step_index} -> timestep={float(timestep)}")
finally:
    # 不永久改变该层 processor。
    target_attention.set_processor(original_processor)

if set(recording_processor.captures_cpu) != CAPTURE_STEP_INDICES:
    raise RuntimeError(f"Expected captures at {CAPTURE_STEP_INDICES}, got {set(recording_processor.captures_cpu)}")

# 每个 capture 仅使用 CFG 条件 branch，并对 heads 和 word pieces 求平均。
first_probs = recording_processor.captures_cpu[min(CAPTURE_STEP_INDICES)]
query_tokens = first_probs.shape[-2]
spatial_side = math.isqrt(query_tokens)
if spatial_side * spatial_side != query_tokens:
    raise RuntimeError(f"Cannot reshape {query_tokens} image query tokens into a square heatmap.")

# denoising 已结束；沿用已验证的显存处理，在 VAE upcast/decode 前释放不再需要的模型。
pipe.unet.to("cpu")
pipe.text_encoder.to("cpu")
pipe.text_encoder_2.to("cpu")
torch.cuda.synchronize()
torch.cuda.empty_cache()

output_dir = Path(__file__).resolve().parent / "outputs" / "attention_evolution"
for directory in (output_dir, output_dir / "raw_maps", output_dir / "heatmaps", output_dir / "overlays"):
    directory.mkdir(parents=True, exist_ok=True)
generated_image_path = output_dir / "sdxl_attention_evolution_seed42.png"
generated_image = decode_image(pipe, latents, generated_image_path)

print("[3] Raw attention statistics and outputs")
print(f"Q/K/V/probability shapes: {recording_processor.qkv_shapes}")
print(f"conditional CFG branch: 1; heads averaged: {first_probs.shape[1]}")
for step_index in sorted(CAPTURE_STEP_INDICES):
    timestep = timesteps[step_index].item()
    conditional_probs = recording_processor.captures_cpu[step_index][1]
    mean_heads = conditional_probs.mean(dim=0)
    for token, indices in token_to_indices.items():
        raw_map = mean_heads[:, indices].mean(dim=-1).reshape(spatial_side, spatial_side)
        raw_path, heatmap_path, overlay_path = save_token_visualizations(
            raw_map, token, step_index, timestep, generated_image, output_dir
        )
        print(
            f"{token}: step={step_index}, timestep={timestep:.0f}, indices={indices}, "
            f"raw min={raw_map.min():.8f}, max={raw_map.max():.8f}, mean={raw_map.mean():.8f}, "
            f"shape={tuple(raw_map.shape)}"
        )
        print(f"  raw={raw_path.name}; heatmap={heatmap_path.name}; overlay={overlay_path.name}")

print(f"generated image: {generated_image_path}")
