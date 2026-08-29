"""Run SDXL's denoising loop explicitly, without calling the full pipeline."""

import os
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline


MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
PROMPT = "A small red cabin beside a calm mountain lake at sunrise, realistic photograph"
SEED = 42
HEIGHT = 1024
WIDTH = 1024
NUM_INFERENCE_STEPS = 10
GUIDANCE_SCALE = 7.0


hf_home = os.environ.get("HF_HOME")
if not hf_home:
    raise RuntimeError("HF_HOME is not set. Configure the shared data-disk cache first.")
if not (Path(hf_home) / "hub" / "models--stabilityai--stable-diffusion-xl-base-1.0").is_dir():
    raise RuntimeError("The verified SDXL cache is not available under HF_HOME.")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

# 复用 baseline 的模型、float16 和整条 pipeline 常驻 CUDA 的方式。
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    local_files_only=True,
)
pipe.to("cuda")
device = torch.device("cuda")
do_classifier_free_guidance = GUIDANCE_SCALE > 1.0

# 与 __call__ 相同：保存 guidance 值，供 pipeline 的 CFG 相关属性使用。
pipe._guidance_scale = GUIDANCE_SCALE
pipe._guidance_rescale = 0.0

with torch.no_grad():
    # 1. 文本编码：得到正/负文本序列 embedding 和 SDXL 的 pooled embedding。
    (
        positive_prompt_embeds,
        negative_prompt_embeds,
        positive_pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=PROMPT,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=None,
    )

    # 2. 同一个 generator 同时交给 scheduler 辅助逻辑与 latent 初始化。
    generator = torch.Generator(device="cuda").manual_seed(SEED)

    # 3. Scheduler：建立与本次 10-step sampling 对应的降噪时间表。
    pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=device)
    timesteps = pipe.scheduler.timesteps
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(
        generator, eta=0.0
    )

    # 4. Latent：prepare_latents 内部会采样固定 seed 的噪声并乘 init_noise_sigma。
    latents = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=pipe.unet.config.in_channels,
        height=HEIGHT,
        width=WIDTH,
        dtype=positive_prompt_embeds.dtype,
        device=device,
        generator=generator,
        latents=None,
    )

    # 5. SDXL micro-conditioning：把图像尺寸、crop 坐标和目标尺寸编码为 time ids。
    original_size = (HEIGHT, WIDTH)
    target_size = (HEIGHT, WIDTH)
    crops_coords_top_left = (0, 0)
    projection_dim = pipe.text_encoder_2.config.projection_dim
    positive_add_time_ids = pipe._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=positive_prompt_embeds.dtype,
        text_encoder_projection_dim=projection_dim,
    )
    negative_add_time_ids = positive_add_time_ids

    # 6. CFG：把无条件与有条件的文本条件沿 batch 维拼接。
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, positive_prompt_embeds], dim=0)
        add_text_embeds = torch.cat(
            [negative_pooled_prompt_embeds, positive_pooled_prompt_embeds], dim=0
        )
        add_time_ids = torch.cat([negative_add_time_ids, positive_add_time_ids], dim=0)
    else:
        prompt_embeds = positive_prompt_embeds
        add_text_embeds = positive_pooled_prompt_embeds
        add_time_ids = positive_add_time_ids

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device)

    print(f"timesteps: {timesteps.detach().cpu().tolist()}")
    print(f"initial latents.shape: {tuple(latents.shape)}")
    print(f"prompt_embeds.shape (after CFG concat): {tuple(prompt_embeds.shape)}")
    print(f"add_text_embeds.shape: {tuple(add_text_embeds.shape)}")
    print(f"add_time_ids.shape: {tuple(add_time_ids.shape)}")

    # 7. 显式 denoising loop：UNet 预测噪声，CFG 合并，再由 scheduler 更新 latent。
    for step_index, timestep in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)

        added_cond_kwargs = {
            "text_embeds": add_text_embeds,
            "time_ids": add_time_ids,
        }
        noise_pred = pipe.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=None,
            cross_attention_kwargs=None,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_cfg = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)
        else:
            noise_pred_cfg = noise_pred

        if step_index == 0:
            print(f"first-step latent_model_input.shape: {tuple(latent_model_input.shape)}")
            print(f"first-step UNet noise_pred.shape: {tuple(noise_pred.shape)}")
            print(f"first-step CFG noise_pred.shape: {tuple(noise_pred_cfg.shape)}")

        latents = pipe.scheduler.step(
            noise_pred_cfg,
            timestep,
            latents,
            **extra_step_kwargs,
            return_dict=False,
        )[0]

    print(f"final latents.shape: {tuple(latents.shape)}")

    # 8. VAE：严格沿用当前 pipeline 的 upcast 与 latent scaling 逻辑后解码。
    needs_upcasting = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
    if needs_upcasting:
        pipe.upcast_vae()
        latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)

    has_latents_mean = hasattr(pipe.vae.config, "latents_mean") and pipe.vae.config.latents_mean is not None
    has_latents_std = hasattr(pipe.vae.config, "latents_std") and pipe.vae.config.latents_std is not None
    if has_latents_mean and has_latents_std:
        latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, 4, 1, 1).to(latents)
        latents_std = torch.tensor(pipe.vae.config.latents_std).view(1, 4, 1, 1).to(latents)
        latents_for_vae = latents * latents_std / pipe.vae.config.scaling_factor + latents_mean
    else:
        latents_for_vae = latents / pipe.vae.config.scaling_factor

    image_tensor = pipe.vae.decode(latents_for_vae, return_dict=False)[0]
    print(f"VAE decoded image tensor.shape: {tuple(image_tensor.shape)}")

    if needs_upcasting:
        pipe.vae.to(dtype=torch.float16)

# 9. 后处理仅将已解码 tensor 转成 PIL 并保存，不会再调用生成 pipeline。
if pipe.watermark is not None:
    image_tensor = pipe.watermark.apply_watermark(image_tensor)
image = pipe.image_processor.postprocess(image_tensor, output_type="pil")[0]
output_path = Path(__file__).resolve().parent / "outputs" / "sdxl_explicit_denoising_seed42.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
image.save(output_path)
print(f"Saved image to: {output_path}")
