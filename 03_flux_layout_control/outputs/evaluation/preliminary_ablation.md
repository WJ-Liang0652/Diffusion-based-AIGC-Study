# Preliminary ablation from existing experiments

No new sweep was run. These comparisons reuse Stage 1–7 outputs and are preliminary, mostly single-prompt/single-seed. Timing/strategy changes are not a strictly isolated ablation because the update parameterization also changes at Stage 6C.

## 1. Guidance timing and strategy

| Variant | Existing configuration | Key measurement | Conclusion |
|---|---|---|---|
| Stage 5: late single-step | index 35, t=499.842; one normalized 0.005 update | E 1.3991->1.3197; COM-x apple +0.502, cup -0.373 tokens | No visible relocation; only local pixel/texture change. |
| Stage 6A: same-timestep inner loop | index 35; K=1/3/5; normalized 0.005 each update | Final E 1.3200/1.1562/0.9943; K=5 COM-x apple +2.783, cup -2.305 tokens | Monotonic objective/attention movement, but no visible relocation at K<=5. |
| Stage 6B: sparse multi-timestep | K=1 at indices 20/28/35; alternate 12/24/35 | Main E 1.442->1.378 / 1.395->1.330 / 1.346->1.265; alternate E 1.524->1.457 / 1.418->1.353 / 1.369->1.289 | All interventions stable and directional; neither schedule visibly relocated objects. |
| Stage 6C: early sigma^2 guidance | indices 0-9; x <- x - eta*sigma^2*grad(E); eta=96.6173 | Index 0 E 0.8525->0.1531; COM-x apple +9.701, cup -8.836 tokens | First clear visible apple/cup swap; strongest fidelity damage. |


The existing evidence identifies early denoising as decisive for visible geometry: late and sparse-middle interventions move the attention objective without propagating to final object position, whereas early sigma-squared guidance produces a visible swap.

## 2. Eta strength

All variants use the same seed-42 half-region task, block 18, indices 0–9, sigma-squared update, K<=5, and threshold 0.2.

| Scale | Eta | Updates | Pixel MAE / RMSE | Guided runtime (s) | Existing visual conclusion |
|---|---|---|---|---|---|
| 1.0x | 96.6173 | 10 | 94.89 / 110.97 | 262.83 | Clear swap; strongest distortion: oversized objects, simplified handleless cup, large background/composition/lighting change. |
| 0.5x | 48.3087 | 11 | 66.71 / 82.55 | 302.85 | Clear swap retained; apple shape/scale and overall realism improve over 1.0x, but cup remains handleless and background/composition change materially. |
| 0.25x | 24.1543 | 13 | 61.98 / 77.90 | 334.95 | Clear swap retained; best fidelity of the tested guided runs by visual inspection and pixel MAE/RMSE, but cup remains handleless and background/composition are not preserved. |


All three strengths retain relocation. Lower eta improves the measured and visible fidelity tradeoff, while the fixed threshold requires more updates and runtime. `0.25x` is the tested low-strength choice used by Stage 6E/7, not a globally optimized value.

## 3. Region definition

This is the cleanest existing comparison: seed, latent, eta 0.25x, block, timing, threshold, and objective are held fixed; only the target region changes.

| Region | Definition | Key measurement | Conclusion |
|---|---|---|---|
| Half regions, eta 0.25x | apple right half; cup left half | MAE/RMSE 61.98/77.90; final COM-x 31.32/15.14 | Clear swap, but scale inflation and major background/identity drift. |
| Tight boxes, eta 0.25x | apple [27,23,46,44); cup [4,20,27,44) | MAE/RMSE 53.46/74.52; box ratios 0.7389/0.6870 | Clear swap with improved extent/scale; cup identity and background still drift. |


Tight boxes reduce pixel MAE by **8.53** and RMSE by **3.38** versus half regions, while preserving relocation. They improve scale/extent but do not solve semantic or background preservation.

## 4. Block 9 versus Block 18 (mechanism-level only)

Values are from the existing index-35 (`t=499.842`) spatial-signal, layout-objective, and gradient probes.

| Double block | Observed-box enrichment apple/cup | Observed A / swapped B energy | B-A margin (ratio) | Gradient L2 | Conclusion |
|---|---|---|---|---|---|
| 9 | 4.188x / 2.631x | 0.0699 / 1.3413 | 1.2713 (19.18x) | 0.03748 | Useful mechanism-level control; retained as comparison block. |
| 18 | 4.068x / 3.008x | 0.0556 / 1.3991 | 1.3435 (25.16x) | 0.06371 | Useful mechanism-level spatial signal; block 18 gives the larger objective margin and gradient norm. |


This is **not a complete end-to-end block ablation**: no matched full Backward Guidance generation was run with block 9. Both blocks encode useful spatial separation, while block 18 provides the stronger objective margin and gradient in the existing probe and is therefore the current guidance block.
