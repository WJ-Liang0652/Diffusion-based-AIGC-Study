"""Aggregate SDXL cross-attention maps from all attn2 layers at one timestep."""

import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline
from diffusers.models.attention import BasicTransformerBlock
from PIL import Image


MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
PROMPT = "A small red cabin beside a calm mountain lake at sunrise, realistic photograph"
SEED = 42
HEIGHT = 1024
WIDTH = 1024
NUM_INFERENCE_STEPS = 10
GUIDANCE_SCALE = 7.0
CAPTURE_STEP_INDEX = 4
TARGET_WORDS = ("cabin", "lake", "sunrise")
ANALYSIS_SIZE = (64, 64)


def region_from_name(module_name):
    if module_name.startswith("down_blocks"):
        return "down"
    if module_name.startswith("mid_block"):
        return "mid"
    if module_name.startswith("up_blocks"):
        return "up"
    return "other"


class CrossAttentionMapRecorder:
    """AttnProcessor2_0 output plus a one-step, reduced CPU attention capture."""

    def __init__(self, module_name, region, token_to_indices, shared):
        self.module_name = module_name
        self.region = region
        self.token_to_indices = token_to_indices
        self.shared = shared
        self.capture_enabled = False

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
            # 仅在目标 step 显式求一次 probability；完整 tensor 不会被保留在 GPU。
            scores = torch.matmul(query, key.transpose(-2, -1)) * (head_dim**-0.5)
            if attention_mask is not None:
                scores = scores + attention_mask
            attention_probs = torch.softmax(scores.float(), dim=-1)
            conditional_probs = attention_probs[1]  # CFG concat order: [negative, positive]
            query_tokens = conditional_probs.shape[-2]
            side = math.isqrt(query_tokens)

            if side * side != query_tokens:
                self.shared["skipped"].append((self.module_name, query_tokens))
            else:
                reduced_maps = {}
                for token, indices in self.token_to_indices.items():
                    # [heads, query, selected tokens] -> mean heads and word pieces -> [query].
                    reduced = conditional_probs[:, :, indices].mean(dim=0).mean(dim=-1)
                    reduced_maps[token] = reduced.reshape(side, side).detach().float().cpu()
                self.shared["records"][self.module_name] = {
                    "region": self.region,
                    "heads": attn.heads,
                    "query_tokens": query_tokens,
                    "spatial_shape": (side, side),
                    "maps": reduced_maps,
                }
            del scores, attention_probs, conditional_probs

        # 保持本机 AttnProcessor2_0 的实际输出路径。
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
    return prompt_ids, word_ids, sorted(set(indices))


def inspect_tokens(pipe):
    token_to_indices = {}
    print("[1] Tokenization")
    for word in TARGET_WORDS:
        per_tokenizer = []
        print(f"token: {word}")
        for name, tokenizer in (("tokenizer", pipe.tokenizer), ("tokenizer_2", pipe.tokenizer_2)):
            prompt_ids, word_ids, indices = find_word_indices(tokenizer, word)
            pieces = tokenizer.convert_ids_to_tokens(prompt_ids)
            print(f"  {name}: standalone ids={word_ids}, prompt indices={indices}")
            for index in indices:
                print(f"    index={index}, token={pieces[index]!r}")
            per_tokenizer.append(indices)
        indices = sorted(set(per_tokenizer[0]) | set(per_tokenizer[1]))
        if not indices:
            raise RuntimeError(f"Could not locate {word!r} in the prompt.")
        if len(indices) > 1:
            print("  multiple token pieces: attention maps use their equal arithmetic mean.")
        token_to_indices[word] = indices
    print(f"token -> indices: {token_to_indices}")
    return token_to_indices


@torch.no_grad()
def decode_reference(pipe, latents, output_path):
    """Mirror the current pipeline's VAE scaling and upcast behavior."""
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


def resize_raw_map(raw_map):
    """Resize raw probabilities for analysis; no per-layer normalization occurs here."""
    return F.interpolate(raw_map[None, None], size=ANALYSIS_SIZE, mode="bilinear", align_corners=False)[0, 0]


def save_aggregate_visuals(raw_aggregate, token, group, reference_image, output_dir):
    """Save raw aggregate data, a heatmap, and a separately normalized overlay."""
    raw_path = output_dir / "raw_aggregates" / f"{token}_{group}_aggregate.pt"
    heatmap_path = output_dir / "heatmaps" / f"{token}_{group}_aggregate_heatmap.png"
    overlay_path = output_dir / "overlays" / f"{token}_{group}_aggregate_overlay.png"
    torch.save(raw_aggregate, raw_path)
    plt.imsave(heatmap_path, raw_aggregate.numpy(), cmap="magma")

    # 只在聚合完成后复制一份作 min-max normalization，用于显示而非计算。
    visualization_map = (raw_aggregate - raw_aggregate.min()) / (
        raw_aggregate.max() - raw_aggregate.min() + 1e-8
    )
    display_map = F.interpolate(
        visualization_map[None, None],
        size=reference_image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )[0, 0]
    heatmap_rgb = plt.get_cmap("magma")(display_map.numpy())[..., :3]
    image_rgb = np.asarray(reference_image.convert("RGB"), dtype=np.float32) / 255.0
    overlay = (0.58 * image_rgb + 0.42 * heatmap_rgb).clip(0, 1)
    Image.fromarray((overlay * 255).astype(np.uint8)).save(overlay_path)
    return raw_path, heatmap_path, overlay_path


hf_home = os.environ.get("HF_HOME")
if not hf_home:
    raise RuntimeError("HF_HOME is not set. Configure the shared data-disk cache first.")
if not (Path(hf_home) / "hub" / "models--stabilityai--stable-diffusion-xl-base-1.0").is_dir():
    raise RuntimeError("The verified SDXL cache is not available under HF_HOME.")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

# 复用此前验证的模型、float16、CUDA 和显式 denoising 设置，且禁止下载。
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, local_files_only=True
)
pipe.to("cuda")
device = torch.device("cuda")
pipe._guidance_scale = GUIDANCE_SCALE
pipe._guidance_rescale = 0.0
token_to_indices = inspect_tokens(pipe)

# 用实际 Attention.is_cross_attention 识别 cross-attention，而不是仅按 attn2 名称猜测。
cross_attention_modules = {}
for block_name, block in pipe.unet.named_modules():
    if not isinstance(block, BasicTransformerBlock):
        continue
    for attention_name in ("attn1", "attn2"):
        attention = getattr(block, attention_name, None)
        if attention is not None and attention.is_cross_attention:
            module_name = f"{block_name}.{attention_name}"
            cross_attention_modules[module_name] = attention

shared = {"records": {}, "skipped": []}
original_processors = {}
recorders = {}
for module_name, attention in cross_attention_modules.items():
    original_processors[module_name] = attention.processor
    recorder = CrossAttentionMapRecorder(
        module_name, region_from_name(module_name), token_to_indices, shared
    )
    attention.set_processor(recorder)
    recorders[module_name] = recorder

try:
    with torch.no_grad():
        # 文本、CFG、scheduler、fixed-seed latent 与 SDXL size conditioning。
        positive, negative, positive_pooled, negative_pooled = pipe.encode_prompt(
            prompt=PROMPT, device=device, num_images_per_prompt=1,
            do_classifier_free_guidance=True, negative_prompt=None
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

        print("[2] Capture setup")
        print(f"capture step_index={CAPTURE_STEP_INDEX}, timestep={float(timesteps[CAPTURE_STEP_INDEX])}")
        print(f"cross-attention targets: {len(cross_attention_modules)}")
        for step_index, timestep in enumerate(timesteps):
            for recorder in recorders.values():
                recorder.capture_enabled = step_index == CAPTURE_STEP_INDEX
            latent_model_input = pipe.scheduler.scale_model_input(torch.cat([latents] * 2), timestep)
            noise_pred = pipe.unet(
                latent_model_input, timestep, encoder_hidden_states=prompt_embeds,
                timestep_cond=None, cross_attention_kwargs=None,
                added_cond_kwargs=added_cond_kwargs, return_dict=False,
            )[0]
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + GUIDANCE_SCALE * (noise_text - noise_uncond)
            latents = pipe.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs, return_dict=False)[0]
finally:
    # 本次进程结束前逐层恢复原 AttnProcessor2_0，不永久修改 UNet。
    for module_name, attention in cross_attention_modules.items():
        attention.set_processor(original_processors[module_name])

if not shared["records"]:
    raise RuntimeError("No cross-attention maps were captured.")

print("[3] Recorded cross-attention layers")
for module_name, record in shared["records"].items():
    print(
        f"{module_name} | region={record['region']} | heads={record['heads']} "
        f"| query_tokens={record['query_tokens']} | spatial={record['spatial_shape']}"
    )
if shared["skipped"]:
    print(f"skipped non-square query layouts: {shared['skipped']}")

region_counts = Counter(record["region"] for record in shared["records"].values())
resolution_counts = Counter(record["spatial_shape"] for record in shared["records"].values())
print(f"recorded layer count: {len(shared['records'])}")
print(f"region counts: {dict(region_counts)}")
print(f"spatial resolutions: {dict(resolution_counts)}")

# denoising 已完成，释放不再需要的 UNet/text encoders，再进行 VAE decode，避免已知 OOM。
pipe.unet.to("cpu")
pipe.text_encoder.to("cpu")
pipe.text_encoder_2.to("cpu")
torch.cuda.synchronize()
torch.cuda.empty_cache()

output_dir = Path(__file__).resolve().parent / "outputs" / "multilayer_attention"
for directory in (output_dir, output_dir / "raw_aggregates", output_dir / "heatmaps", output_dir / "overlays"):
    directory.mkdir(parents=True, exist_ok=True)
reference_path = output_dir / "sdxl_multilayer_attention_seed42.png"
reference_image = decode_reference(pipe, latents, reference_path)

print("[4] Raw aggregates")
groups = ("down", "mid", "up", "all")
for token in TARGET_WORDS:
    for group in groups:
        selected = [
            record["maps"][token]
            for record in shared["records"].values()
            if group == "all" or record["region"] == group
        ]
        if not selected:
            print(f"{token} | {group}: no compatible layers")
            continue
        # Arithmetic mean of resized raw maps: no layer weights and no pre-normalization.
        raw_aggregate = torch.stack([resize_raw_map(raw_map) for raw_map in selected]).mean(dim=0)
        raw_path, heatmap_path, overlay_path = save_aggregate_visuals(
            raw_aggregate, token, group, reference_image, output_dir
        )
        print(
            f"{token} | {group} | layers={len(selected)} | raw min={raw_aggregate.min():.8f} "
            f"max={raw_aggregate.max():.8f} mean={raw_aggregate.mean():.8f} | "
            f"shape={tuple(raw_aggregate.shape)}"
        )
        print(f"  raw={raw_path.name}; heatmap={heatmap_path.name}; overlay={overlay_path.name}")

print(f"reference image: {reference_path}")
