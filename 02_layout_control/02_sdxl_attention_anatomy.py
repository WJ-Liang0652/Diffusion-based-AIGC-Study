"""Inspect SDXL UNet attention structure without running denoising."""

import os
from collections import Counter
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline
from diffusers.models.attention import BasicTransformerBlock


MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


def block_region(module_name):
    if module_name.startswith("down_blocks"):
        return "down block"
    if module_name.startswith("mid_block"):
        return "mid block"
    if module_name.startswith("up_blocks"):
        return "up block"
    return "other"


def attention_summary(module_name, attention):
    """Print a compact, useful summary instead of a full module repr."""
    print(f"module name: {module_name}")
    print(f"module class: {type(attention).__module__}.{type(attention).__name__}")
    print(f"heads: {attention.heads}")
    print(f"head dimension (derived): {attention.inner_dim // attention.heads}")
    print(f"inner dimension: {attention.inner_dim}")
    print(f"is_cross_attention: {attention.is_cross_attention}")
    print(f"cross_attention_dim: {attention.cross_attention_dim}")
    print(
        "projections: "
        f"to_q={attention.to_q.in_features}->{attention.to_q.out_features}, "
        f"to_k={attention.to_k.in_features}->{attention.to_k.out_features}, "
        f"to_v={attention.to_v.in_features}->{attention.to_v.out_features}"
    )
    print(f"processor: {type(attention.processor).__name__}")


hf_home = os.environ.get("HF_HOME")
if not hf_home:
    raise RuntimeError("HF_HOME is not set. Configure the shared data-disk cache first.")
if not (Path(hf_home) / "hub" / "models--stabilityai--stable-diffusion-xl-base-1.0").is_dir():
    raise RuntimeError("The verified SDXL cache is not available under HF_HOME.")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

# 只从已有 cache 加载，并沿用 SDXL baseline 的 float16 + CUDA 配置。
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    local_files_only=True,
)
pipe.to("cuda")
unet = pipe.unet

# 从实际 BasicTransformerBlock 收集 attn1 / attn2 及其构造时记录的语义。
attention_modules = {}
for block_name, block in unet.named_modules():
    if not isinstance(block, BasicTransformerBlock):
        continue
    for attention_name in ("attn1", "attn2"):
        attention = getattr(block, attention_name, None)
        if attention is not None:
            attention_modules[f"{block_name}.{attention_name}"] = attention

print("[1] Attention processor summary")
processors = unet.attn_processors
print(f"attention processor total: {len(processors)}")
for processor_name, processor in processors.items():
    module_name = processor_name.removesuffix(".processor")
    print(
        f"{processor_name} | region={block_region(module_name)} "
        f"| processor={type(processor).__name__}"
    )
print()

print("[2] Self vs cross attention")
self_attention = []
cross_attention = []
for module_name, attention in attention_modules.items():
    # In this Diffusers version, this boolean preserves whether cross context was supplied.
    # For self-attention, cross_attention_dim is later set equal to query_dim for projections.
    if attention.is_cross_attention:
        cross_attention.append((module_name, attention))
    else:
        self_attention.append((module_name, attention))

print(f"BasicTransformerBlock attention modules: {len(attention_modules)}")
print(f"self-attention count: {len(self_attention)}")
print(f"cross-attention count: {len(cross_attention)}")
print(
    "attn1 is_cross_attention values:",
    sorted({a.is_cross_attention for n, a in attention_modules.items() if n.endswith(".attn1")}),
)
print(
    "attn2 is_cross_attention values:",
    sorted({a.is_cross_attention for n, a in attention_modules.items() if n.endswith(".attn2")}),
)
print(
    "attn1 projection input dimensions:",
    sorted({a.cross_attention_dim for n, a in attention_modules.items() if n.endswith(".attn1")}),
)
print(
    "attn2 cross-context dimensions:",
    sorted({a.cross_attention_dim for n, a in attention_modules.items() if n.endswith(".attn2")}),
)
print(
    "Source-confirmed rule: BasicTransformerBlock constructs attn1 as self-attention "
    "when only_cross_attention=False, and constructs attn2 with cross_attention_dim. "
    "Attention.is_cross_attention records whether cross context was originally supplied."
)
print(
    "This SDXL UNet's actual modules have attn1.is_cross_attention=False and "
    "attn2.is_cross_attention=True: attn1 is self-attention; attn2 is cross-attention."
)
print()

print("[3] Representative cross-attention modules")
for region in ("down block", "mid block", "up block"):
    module_name, attention = next(
        (name, module) for name, module in cross_attention if block_region(name) == region
    )
    print(f"--- {region} ---")
    attention_summary(module_name, attention)
print()

print("[4] Current attention processor implementation")
for processor_type, count in sorted(Counter(type(p).__name__ for p in processors.values()).items()):
    print(f"{processor_type}: {count}")
processor_keys = set(processors)
module_processor_keys = {f"{name}.processor" for name in attention_modules}
print(f"processors matched to BasicTransformerBlock attention modules: {processor_keys == module_processor_keys}")
print("No hook was registered, no processor was replaced, and no denoising was run.")
