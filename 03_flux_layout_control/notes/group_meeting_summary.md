# FLUX Layout Control — Group Meeting Summary

## Final method/configuration used in the current tight-box checkpoint

- Model: `black-forest-labs/FLUX.1-dev`; 768x768, 50 FlowMatch steps, guidance scale 3.5, BF16, batch size 1, sequential CPU offload.
- Signal/objective: Double Stream block 18 native joint attention, image-query to target-text-key probability; Eq. (2) target-box attention-mass energy.
- Backward Guidance: early denoising indices 0–9, `x_t <- x_t - eta * sigma_t^2 * grad_x E`, `eta_scale=0.25` (`eta=24.154335`), K<=5, objective threshold 0.2.
- Only packed latents receive gradients. Text encoders/VAE do not backpropagate and model parameters remain frozen.
- Target regions are finite object-sized boxes. This is the frozen Stage 6E/7 checkpoint configuration, not a globally finalized optimum.

## Basic quantitative evaluation

| Case | Prompt | Seed | Object 1 ratio | Object 2 ratio | Coarse relocation | Baseline (s) | Guided (s) | Multiplier |
|---|---|---:|---:|---:|---|---:|---:|---:|
| apple_cup_seed42 | a red apple and a blue cup, realistic photo | 42 | apple 0.7389 | cup 0.6870 | yes | 213.07 | 430.99 | 2.023x |
| apple_cup_seed123 | a red apple and a blue cup, realistic photo | 123 | apple 0.7420 | cup 0.7139 | yes | 224.16 | 421.41 | 1.880x |
| apple_cup_seed2024 | a red apple and a blue cup, realistic photo | 2024 | apple 0.7486 | cup 0.7219 | yes | 228.56 | 386.98 | 1.693x |
| banana_bottle_seed31415 | a yellow banana and a green bottle, realistic photo | 31415 | banana 0.7641 | bottle 0.6259 | yes | 234.72 | 733.49 | 3.125x |

Aggregate: eight-object mean/min box-inside attention ratio = **0.7178/0.6259**. Coarse relocation is observed in **4/4 preliminary tested cases**; this is not a formal benchmark success rate. Stage 7 mean runtime overhead is **+284.81 s**, or **2.233x (+123.27%)** versus matching baselines.

## Preliminary ablation summary

| Axis | Existing evidence | Preliminary conclusion |
|---|---|---|
| Timing/strategy | Stage 5 late single update and Stage 6A/6B late/sparse updates reduce energy and move attention but do not visibly relocate objects; Stage 6C early sigma-squared guidance visibly swaps them. | Early high-noise intervention is the decisive existing factor for geometry propagation. |
| Eta | 1.0x/0.5x/0.25x all swap; MAE falls 94.89 -> 66.71 -> 61.98, while runtime rises 262.83 -> 302.85 -> 334.95 s. | 0.25x gives the best tested half-region fidelity/control tradeoff, but is not globally tuned. |
| Region | Tight boxes versus 0.25x half regions reduce MAE 61.98 -> 53.46 and RMSE 77.90 -> 74.52. | Tight boxes reduce scale inflation and improve extent; identity/background drift remains. |
| Block | At index 35, block 9/18 observed-vs-swapped energy margin is 1.2713/1.3435; gradient L2 is 0.03748/0.06371. | Both are useful; block 18 is stronger in existing mechanism probes. This is not an end-to-end block ablation. |

Full extracted tables: [preliminary_ablation.md](../outputs/evaluation/preliminary_ablation.md).

## Main findings

1. Native FLUX joint attention supplies a differentiable text-to-image spatial signal; block 18 is the strongest current probe target, with block 9 also spatially meaningful.
2. Objective reduction or attention-centroid movement alone is insufficient: late single-step, same-timestep inner-loop, and sparse multi-timestep strategies did not yield visible relocation.
3. Early sigma-squared Backward Guidance is the first existing strategy that consistently changes object geometry and produces the requested swap.
4. The frozen tight-box method relocates both objects in all four preliminary tested cases; mean/min attention box-inside ratios are 0.7178/0.6259.
5. Tight boxes and lower eta improve fidelity, but background/composition and object identity/shape drift remain, and cross-prompt compute cost is variable.

## Limitations

- Four cases, two prompts, and four seeds are a sanity set, not a benchmark; no confidence interval or statistical success rate is justified.
- Box-inside values measure attention mass, not detected/segmented visible-object IoU. All four cases have some visible object extent outside the tight targets.
- Manual coarse-relocation and quality judgements are not yet backed by detector, segmentation, CLIP/DINO, or human-rater metrics.
- Eta and region comparisons are seed-42/apple-cup only. The timing comparison is historically sequential rather than fully factorial and controlled.
- Block 9 versus 18 is mechanism-level only; there is no matched block-9 full-generation guidance run.
- Fidelity remains unresolved: backgrounds change, cup identity can drift, and the banana/bottle case changes instance count and bottle material/shape.

## Next work

- Do not add a new algorithm branch before the group-meeting checkpoint. First define a representative evaluation protocol with object localization/segmentation, semantic preservation, image quality, runtime, and VRAM metrics.
- After agreeing on that protocol, run controlled single-variable ablations for timing, block, eta, K/threshold, and region definition across multiple prompts/seeds.
- Treat fidelity preservation as the next method problem only after the basic evaluation protocol is fixed. Forward Guidance remains deferred.
