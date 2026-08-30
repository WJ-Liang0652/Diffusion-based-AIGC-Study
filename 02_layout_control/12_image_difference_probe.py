"""Numerically compare the already-saved baseline and controlled layout-guidance images."""

from pathlib import Path

import numpy as np
from PIL import Image


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "layout_guidance"
BASELINE_PATH = OUTPUT_DIR / "baseline_seed42.png"
CONTROLLED_PATH = OUTPUT_DIR / "controlled_seed42.png"
DIFFERENCE_PATH = OUTPUT_DIR / "amplified_difference.png"

if not BASELINE_PATH.is_file() or not CONTROLLED_PATH.is_file():
    raise FileNotFoundError("Expected baseline_seed42.png and controlled_seed42.png in layout_guidance outputs.")

baseline_image = Image.open(BASELINE_PATH).convert("RGB")
controlled_image = Image.open(CONTROLLED_PATH).convert("RGB")
if baseline_image.size != controlled_image.size:
    raise RuntimeError(f"Image dimensions differ: {baseline_image.size} vs {controlled_image.size}")

baseline = np.asarray(baseline_image, dtype=np.int16)
controlled = np.asarray(controlled_image, dtype=np.int16)
absolute_difference = np.abs(controlled - baseline)
pixel_difference_mask = np.any(absolute_difference != 0, axis=-1)

# Display-only image: per-channel absolute differences rescaled to the full 8-bit range.
# The original, unscaled absolute_difference tensor is used for every metric above.
max_difference = int(absolute_difference.max())
if max_difference == 0:
    amplified = np.zeros_like(absolute_difference, dtype=np.uint8)
else:
    amplified = np.rint(absolute_difference.astype(np.float32) * (255.0 / max_difference)).astype(np.uint8)
Image.fromarray(amplified, mode="RGB").save(DIFFERENCE_PATH)

print("[1] Existing image difference probe")
print(f"baseline size: {baseline_image.size}")
print(f"controlled size: {controlled_image.size}")
print(f"pixelwise exactly identical: {bool(np.array_equal(baseline, controlled))}")
print(f"mean absolute pixel difference: {absolute_difference.mean():.8f}")
print(f"max absolute pixel difference: {max_difference}")
print(f"RMSE: {np.sqrt(np.mean((controlled - baseline).astype(np.float64) ** 2)):.8f}")
print(f"different pixel ratio: {pixel_difference_mask.mean():.8f}")
print(f"amplified display-only difference: {DIFFERENCE_PATH}")
