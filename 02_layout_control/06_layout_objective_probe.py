"""Compute a simple bounding-box layout objective from a saved raw attention map."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw


TARGET_BOX = (0.10, 0.50, 0.40, 0.82)  # x0, y0, x1, y1 in normalized image coordinates
EPS = 1e-8

ROOT = Path(__file__).resolve().parent
RAW_MAP_PATH = ROOT / "outputs" / "multilayer_attention" / "raw_aggregates" / "cabin_all_aggregate.pt"
REFERENCE_IMAGE_PATH = ROOT / "outputs" / "multilayer_attention" / "sdxl_multilayer_attention_seed42.png"
OUTPUT_DIR = ROOT / "outputs" / "layout_objective"
OUTPUT_PATH = OUTPUT_DIR / "cabin_all_target_box_overlay.png"


# 读取 05 脚本保存的 raw aggregate；这里绝不做 min-max normalization。
raw_map = torch.load(RAW_MAP_PATH, map_location="cpu")
if not isinstance(raw_map, torch.Tensor) or raw_map.ndim != 2:
    raise RuntimeError(f"Expected a 2D raw attention tensor, got {type(raw_map)} with shape {getattr(raw_map, 'shape', None)}")
raw_map = raw_map.float()
grid_height, grid_width = raw_map.shape

# 将连续坐标映射为左闭右开整数 grid 区间，避免边界像素重复计入。
x0, y0, x1, y1 = TARGET_BOX
x_start = max(0, min(grid_width, math.floor(x0 * grid_width)))
x_end = max(0, min(grid_width, math.ceil(x1 * grid_width)))
y_start = max(0, min(grid_height, math.floor(y0 * grid_height)))
y_end = max(0, min(grid_height, math.ceil(y1 * grid_height)))
if not (x_start < x_end and y_start < y_end):
    raise RuntimeError(f"Target box becomes empty on the {grid_width}x{grid_height} grid.")

# Attention mass 是各 grid cell 的 raw attention 值之和；目标是将更多 mass 放入 box。
total_mass = raw_map.sum()
inside_mass = raw_map[y_start:y_end, x_start:x_end].sum()
inside_ratio = inside_mass / (total_mass + EPS)
layout_loss = 1.0 - inside_ratio
outside_mass = total_mass - inside_mass
mass_error = (inside_mass + outside_mass - total_mass).abs()

print("[1] Raw attention map")
print(f"raw map path: {RAW_MAP_PATH}")
print(f"raw map shape: {tuple(raw_map.shape)}")
print(f"raw min/max/mean: {raw_map.min().item():.8f} / {raw_map.max().item():.8f} / {raw_map.mean().item():.8f}")
print()
print("[2] Target box on the 64x64 grid")
print(f"normalized box (x0, y0, x1, y1): {TARGET_BOX}")
print(f"grid x range: [{x_start}, {x_end}) -> columns {x_start}..{x_end - 1}")
print(f"grid y range: [{y_start}, {y_end}) -> rows {y_start}..{y_end - 1}")
print()
print("[3] Layout objective")
print(f"total_mass: {total_mass.item():.8f}")
print(f"inside_mass: {inside_mass.item():.8f}")
print(f"outside_mass: {outside_mass.item():.8f}")
print(f"inside_mass + outside_mass - total_mass abs error: {mass_error.item():.12f}")
print(f"inside_ratio: {inside_ratio.item():.8f}")
print(f"layout_loss = 1 - inside_ratio: {layout_loss.item():.8f}")

# 以下只用于解释图：复制 raw map 后才做显示归一化，不参与上面的 loss。
reference = Image.open(REFERENCE_IMAGE_PATH).convert("RGB")
visualization_map = (raw_map - raw_map.min()) / (raw_map.max() - raw_map.min() + EPS)
visualization_image = Image.fromarray((visualization_map.numpy() * 255).astype(np.uint8)).resize(
    reference.size, Image.Resampling.BICUBIC
)
heatmap_rgb = (plt.get_cmap("magma")(np.asarray(visualization_image) / 255.0)[..., :3] * 255).astype(np.uint8)
overlay = (0.58 * np.asarray(reference, dtype=np.float32) + 0.42 * heatmap_rgb).clip(0, 255).astype(np.uint8)
overlay_image = Image.fromarray(overlay)

# 绘制的是与实际 grid slicing 一致的离散 box 边界。
draw = ImageDraw.Draw(overlay_image)
box_pixels = (
    round(x_start / grid_width * reference.width),
    round(y_start / grid_height * reference.height),
    round(x_end / grid_width * reference.width),
    round(y_end / grid_height * reference.height),
)
draw.rectangle(box_pixels, outline="red", width=5)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
overlay_image.save(OUTPUT_PATH)
print()
print("[4] Explanation visualization")
print(f"overlay path: {OUTPUT_PATH}")
print(f"box pixels on reference image: {box_pixels}")
print("The heatmap is normalized only for display; the loss above uses raw_map directly.")
