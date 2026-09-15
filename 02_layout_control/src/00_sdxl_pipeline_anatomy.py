"""Inspect the already cached SDXL pipeline without generating an image."""

import os
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline


MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
PROMPT = "A small red cabin beside a calm mountain lake at sunrise, realistic photograph"
HEIGHT = 1024
WIDTH = 1024
NUM_INFERENCE_STEPS = 5
TOKENS_TO_SHOW = 16


def print_tokenization(name, tokenizer, prompt):
    """Show the fixed-length IDs SDXL passes to one CLIP text encoder."""
    encoded = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded.input_ids
    token_ids = input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    print(f"{name}.input_ids.shape: {tuple(input_ids.shape)}")
    print(f"{name} token count (including special/padding tokens): {input_ids.shape[-1]}")
    print(f"{name} first {TOKENS_TO_SHOW} token IDs and decoded pieces:")
    for token_id, token in zip(token_ids[:TOKENS_TO_SHOW], tokens[:TOKENS_TO_SHOW]):
        decoded_piece = tokenizer.decode([token_id])
        print(f"  id={token_id:>5}  token={token!r:<18} decoded={decoded_piece!r}")


hf_home = os.environ.get("HF_HOME")
if not hf_home:
    raise RuntimeError("HF_HOME is not set. Configure the shared data-disk cache first.")

cache_path = Path(hf_home) / "hub" / "models--stabilityai--stable-diffusion-xl-base-1.0"
if not cache_path.is_dir():
    raise RuntimeError(f"Expected cached SDXL model was not found: {cache_path}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This inspection uses the verified SDXL GPU setup.")

# local_files_only prevents any network download. This matches the baseline dtype/device setup.
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    local_files_only=True,
)
pipe.to("cuda")
device = torch.device("cuda")

print("[1] Pipeline components")
for name in ("tokenizer", "tokenizer_2", "text_encoder", "text_encoder_2", "unet", "scheduler", "vae"):
    component = getattr(pipe, name)
    print(f"{name}: {type(component).__module__}.{type(component).__name__}")
print(f"pipeline dtype: {pipe.unet.dtype}")
print(f"pipeline execution device: {pipe._execution_device}")
print()

print("[2] Tokenization")
print(f"prompt: {PROMPT!r}")
print_tokenization("tokenizer", pipe.tokenizer, PROMPT)
print_tokenization("tokenizer_2", pipe.tokenizer_2, PROMPT)
print()

print("[3] Prompt embeddings")
do_classifier_free_guidance = True  # Same CFG path as the baseline's guidance_scale=7.0.
with torch.no_grad():
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=PROMPT,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=None,
    )

print(f"CFG enabled: {do_classifier_free_guidance}")
print(f"force_zeros_for_empty_prompt: {pipe.config.force_zeros_for_empty_prompt}")
print(f"prompt_embeds.shape: {tuple(prompt_embeds.shape)}")
print(f"negative_prompt_embeds.shape: {tuple(negative_prompt_embeds.shape)}")
print(f"pooled_prompt_embeds.shape: {tuple(pooled_prompt_embeds.shape)}")
print(f"negative_pooled_prompt_embeds.shape: {tuple(negative_pooled_prompt_embeds.shape)}")
print()

print("[4] Scheduler timesteps")
pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=device)
print(f"num_inference_steps: {NUM_INFERENCE_STEPS}")
print(f"scheduler timesteps: {pipe.scheduler.timesteps.detach().cpu().tolist()}")
print()

print("[5] Latent shape")
batch_size = 1
latent_channels = pipe.unet.config.in_channels
latent_height = HEIGHT // pipe.vae_scale_factor
latent_width = WIDTH // pipe.vae_scale_factor
latent_shape = (batch_size, latent_channels, latent_height, latent_width)
print(f"image size: {HEIGHT} x {WIDTH}")
print(f"vae_scale_factor: {pipe.vae_scale_factor}")
print(f"latent shape [B, C, H, W]: {latent_shape}")
print("B=batch size; C=latent channels; H/W=spatial dimensions after VAE downsampling.")
print("No denoising loop, scheduler step, VAE decode, or image generation was run.")
