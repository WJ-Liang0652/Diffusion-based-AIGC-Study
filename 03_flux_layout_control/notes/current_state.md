# FLUX Layout Control — Checkpoint

## Final goal

Complete a method-level reproduction of Training-Free Layout Control / Attention Guidance on `FLUX.1-dev`, with Backward Guidance as the main line. Final results must include qualitative examples, representative quantitative evaluation, timestep/block/attention/strength/iteration ablations, and comparison with the SDXL reference. Do not claim equivalence to SDXL cross-attention without experimental evidence.

## Completed stages and verified conclusions

### schnell mechanism reference

- `FLUX.1-schnell` baseline is reproducible, and explicit sampling matches `FluxPipeline.__call__` exactly.
- The useful FLUX signal is native joint attention, not SDXL-style cross-attention: image-query → target-text-key probability, joint-normalized over text and image keys.
- Double Stream blocks 9 and 18 contain useful middle/late spatial semantics; block 0 is nearly uniform. Block 18 is the primary source, block 9 is the control, and Single Stream is not the primary path.
- Paper Eq. (2) separates observed apple-left/cup-right from the swapped layout. Its gradient with respect to packed latents is finite/nonzero while model parameters remain gradient-free.
- schnell single-step and limited inner-loop guidance reduce attention energy but have not shown stable, clear object relocation. schnell remains a mechanism reference, not a complete reproduction.

### Stage 1 — dev baseline and explicit sampling

- Fixed `FLUX.1-dev` Pipeline baseline completed successfully.
- Explicit sampling matches all 50 Pipeline post-scheduler packed-latent states exactly.
- Final packed-latent max/mean absolute differences: `0.0 / 0.0`.
- Decoded RGB max/mean absolute differences: `0 / 0.0`; images are pixel-identical.
- Dev guidance embedding and dynamic-shifted FlowMatch schedule are correctly handled.

### Stage 2 — dev attention probe

- T5 spans match schnell: `apple` → index `[3]`, id `[8947]`, piece `▁apple`; `cup` → index `[8]`, id `[4119]`, piece `▁cup`.
- Tested Double blocks `0/9/18` at denoising indices `0/35/44`, actual timesteps `1000.0/499.842/241.264`.
- Block 0 remains nearly uniform (`1.017x` mean reference-box enrichment).
- All block-9/18 middle/late configurations pass the spatial-separation criterion (`3.434x` mean enrichment). At block 18 / `t=499.842`, apple/cup enrichment is `4.068x / 3.008x`.
- The dev profile matches schnell: weak early block and strong middle/late Double Stream localization. This is verified only for the fixed prompt/seed/layout.
- Attention probes are non-invasive: generated hashes match the Stage 1 baseline exactly.

### Stage 3 — dev layout objective

- Added `src/13_flux_dev_layout_objective.py`; the schnell Eq. (2) definition was migrated unchanged.
- At denoising index `35`, actual `t=499.842194`, token spans remain apple `[3]` / cup `[8]`, maps are `48x48`, and left/right masks are `[0,0,24,48]` / `[24,0,48,48]` with 1152 tokens each.
- Double block 18: Layout A/B energy `0.055602 / 1.399058`, margin `1.343456`, ratio `25.1621x`.
- Double block 9: Layout A/B energy `0.069931 / 1.341281`, margin `1.271349`, ratio `19.1799x`.
- All energies are finite; observed Layout A is lower-energy than swapped Layout B at both blocks. The probe is non-invasive: its generated pixel hash exactly matches Stage 1.
- Runtime was `235.85 s`; peak CUDA allocation was `1383.83 MiB`, with no OOM or abnormal VRAM growth.

### Stage 4 — dev gradient probe

- Added `src/14_flux_dev_gradient_probe.py`; the sole optimization variable is the saved packed latent entering denoising index `35` (`t=499.842194`, shape `[1,2304,64]`).
- The unchanged Stage 3 Layout B objective matches within `1e-5`, `requires_grad=True`, and no model parameter accumulates gradients.
- Block 18: loss `1.399058`; gradient L2/max/mean `0.063709 / 0.004059 / 9.24861e-05`. A normalized `5e-4` step lowers loss to `1.395208` (`-0.003850`).
- Block 9: loss `1.341281`; gradient L2/max/mean `0.037479 / 0.001549 / 6.69663e-05`. A normalized `5e-4` step lowers loss to `1.339235` (`-0.002046`).
- Both gradients are finite and all `147456/147456` elements are nonzero; both blocks pass. Overall peak CUDA allocation was `2004.13 MiB`; no OOM occurred.

### Stage 5 — dev single-step Backward Guidance

- Added `src/15_flux_dev_single_step_guidance.py`; baseline and guided branches share the exact saved index-35 input state. Baseline continuation matches Stage 1 final latent and pixels exactly.
- One Double-block-18 intervention was tested at `t=499.842194` for `relative_step=5e-4`, then once more at the schnell-tested `0.005` after the default was visually imperceptible. No inner loop or multi-timestep guidance was used.
- At `5e-4`, actual BF16 update L2/relative norm is `0.121205 / 0.000394`; Layout B energy falls `1.399058 → 1.395609`. Apple-right/cup-left attention mass increases by `0.000817 / 0.001262` and attention COM moves `+0.0111 / -0.0275` image tokens.
- At `0.005`, actual BF16 update L2/relative norm is `1.565214 / 0.005089`; Layout B energy falls `1.399058 → 1.319662`. Apple-right/cup-left attention mass increases by `0.027678 / 0.020259` and attention COM moves `+0.5024 / -0.3728` image tokens.
- Both strengths move attention in the requested swapped direction, but neither produces visible apple/cup relocation in the final image. At `0.005`, color-proxy x shifts are only apple `+0.0338 px`, cup `-0.0335 px`; differences are local pixel/texture perturbations. Single-step propagation is mechanically valid but does not satisfy visible layout control.
- Peak CUDA allocation was `2002.92 MiB`; no OOM or abnormal memory growth occurred.

### Stage 6A — dev single-timestep inner-loop Backward Guidance

- Added `src/16_flux_dev_inner_loop_guidance.py`; one shared index-35 trajectory uses `relative_step=0.005` and fresh forward/backward gradients on every update, evaluated at K=`1/3/5`. No other timestep is guided.
- Layout B energy decreases monotonically: `1.399058 → 1.319962` (K=1), `1.156224` (K=3), `0.994316` (K=5); every individual update lowers the objective with finite gradients and zero parameter gradients.
- Cumulative attention-centroid x movement at K=`1/3/5`: apple `+0.4971 / +1.6049 / +2.7830` tokens toward right; cup `-0.3713 / -1.3335 / -2.3050` tokens toward left. Direction is consistent across all five iterations.
- Final-image geometry remains visually unchanged at K=1/3/5. Color-proxy x shifts are tiny and not consistently directional; K=3 is apple `+0.0575 px`, cup `-0.0145 px`, while K=5 reverses to apple `-0.0621 px`, cup `+0.0233 px`.
- K=5 is the strongest stable objective/attention configuration, but it is not successful visible layout control. More same-timestep inner iterations are not currently justified. Peak CUDA allocation was `2004.48 MiB`; no OOM occurred.


### Stage 6B — dev sparse multi-timestep Backward Guidance MVP

- Added `src/17_flux_dev_multi_timestep_guidance.py`; it replays the Stage 1 initial latent, applies one fresh block-18 Layout-B update per selected timestep at `relative_step=0.005`, and stops on failed spatial gating, non-finite gradients, parameter gradients, or a non-descending update.
- Tested only two three-point schedules: indices `20/28/35` (`t=777.669/646.915/499.842`) and the one allowed earlier alternative `12/24/35` (`t=880.728/716.407/499.842`), always K=1 per intervention. The first run independently reproduced the Stage 1 final latent and image exactly.
- Every new earlier signal was finite and spatially meaningful: observed-box enrichment remained `2.922x–3.839x`, and apple/cup attention centroids stayed separated by `14.675–16.687` image tokens. All six gradients were finite, no model parameter accumulated gradients, and peak CUDA allocation remained `2004.42 MiB` or lower.
- All interventions lowered Layout-B energy and moved attention in the requested direction. For `20/28/35`, energy changes were `1.4422→1.3782`, `1.3954→1.3300`, and `1.3464→1.2654`; apple/cup centroid-x deltas were `+0.419/-0.210`, `+0.436/-0.338`, and `+0.551/-0.444` tokens. For `12/24/35`, the corresponding energy changes were `1.5237→1.4573`, `1.4176→1.3530`, and `1.3685→1.2893`; centroid-x deltas were `+0.505/-0.281`, `+0.427/-0.287`, and `+0.512/-0.425` tokens.
- Neither schedule produces visible apple/cup relocation or a stronger geometric change than Stage 6A K=5. Final color-proxy x shifts are `+0.047/-0.132 px` for `20/28/35` and `+0.237/+0.139 px` for `12/24/35`; the earlier schedule mainly increases local appearance differences and loses joint final-direction consistency.
- `20/28/35` is the best stable Stage 6B configuration, but it fails the visible-layout success criterion. A full strength/timestep/block hyperparameter sweep is not justified with the current objective-to-latent intervention; the next work should first test why strong attention movement does not propagate to geometry.


### Stage 6C — paper-aligned FLUX Backward Guidance MVP

- Added `src/18_flux_dev_paper_aligned_guidance.py`. Stage 1–4 remain unchanged, and Stage 5–6B remain diagnostic history. The implementation keeps Eq. (2) and Layout B but replaces per-update norm normalization with the author-code-style practical update `x_t ← x_t - eta * sigma_t^2 * grad(E)` using the native FlowMatch sigma sequence.
- Eta was calibrated once from the validated Stage 6A index-35 update: `sigma_35=0.499842`, Stage 6A gradient coefficient `24.139091`, hence fixed `eta=24.139091/sigma_35^2=96.617341`. This reconstructs relative update `0.005` from the saved index-35 norms; eta is never recalibrated during sampling.
- Guided only indices `0–9` (`sigma=1.000000→0.913963`) with Double block 18, at most K=5, and Eq. (2) threshold `0.2`. Every step stopped after K=1 because the first index-0 update reduced energy below threshold and later steps entered below threshold. All updates were finite and descending, no model parameter accumulated gradients, and peak CUDA allocation was `2004.41 MiB`.
- Index 0 is the decisive intervention: energy `0.852478→0.153076`, relative update `0.030921`, apple centroid-x `19.813→29.514`, and cup centroid-x `27.341→18.506`. Across indices 1–9, energy remains low (`0.092347→0.070315` at index 1 and `0.027374→0.026347` at index 9), relative update decays `0.004803→0.000901`, and final attention centroids stabilize near apple `x=31.948`, cup `x=12.035`.
- The final image shows a clear object-level Layout-B swap: the blue cup relocates from right to left and the red apple relocates from left to right. Color-proxy x shifts are apple `+321.49 px` and cup `-425.65 px`. This is genuine geometry/layout relocation, with a material fidelity cost: objects enlarge, the cup shape/handle is simplified, and composition/background/lighting change.
- Because block 18 already produced stable visible relocation, the conditional blocks 9+18 control was not run. The paper-aligned FLUX Backward Guidance MVP is complete for this fixed prompt/seed/layout, but multi-seed repeatability, fidelity/control tradeoff, and method-level ablations remain open.

### Stage 6D — Backward Guidance strength stabilization

- Kept the successful Stage 6C configuration fixed (FLUX.1-dev, block 18, Layout B, indices 0–9, sigma-squared update, K<=5/threshold 0.2, seed/prompt/initial latent) and tested only eta scales `0.5` and `0.25` against the existing `1.0`; eta values are `96.617341 / 48.308670 / 24.154335`.
- All updates were finite and descending with zero model-parameter gradients. Index 0 used `1 / 2 / 4` inner updates for `1.0 / 0.5 / 0.25`; its relative-update sequences were `0.030921`, `0.015519 -> 0.007964`, and `0.007853 -> 0.006635 -> 0.004438 -> 0.002877`. Later per-step updates remained smaller. Peak CUDA allocation was unchanged at `2004.41 MiB`; guided runtimes were `262.83 / 302.85 / 334.95 s`.
- All three strengths visibly swap the apple to the right and cup to the left. Final block-18 centroid x positions at index 9 are apple/cup `31.948/12.035`, `31.172/13.997`, and `31.320/15.142`; color-proxy x shifts remain strongly directional.
- Fidelity ranks `0.25 > 0.5 > 1.0`: pixel MAE/RMSE versus baseline are `61.98/77.90`, `66.71/82.55`, and `94.89/110.97`, and the lower strengths preserve apple shape/scale and realism better. However, even `0.25` changes the background/composition and turns the original handled ceramic cup into a handleless tall cup, so fidelity is improved but not fully preserved.
- `eta=24.154335` (`0.25x`) was the smallest Stage 6D strength retaining clear relocation, but it is now provisional rather than a final frozen configuration: Stage 6E shows target-region geometry materially affects fidelity. Summary: `outputs/paper_aligned_guidance/flux_dev_stage6d_strength_summary.json`; visual grid: `outputs/paper_aligned_guidance/flux_dev_stage6d_strength_comparison.jpg`.


### Stage 6E — tight bounding-box layout control check

- Read-only verification confirmed the original Layout A/B targets are exact half-planes on the 48x48 grid: left `[0,0,24,48)` and right `[24,0,48,48)`, mapping to image coordinates `[0,0,384,768)` and `[384,0,768,768)`, with 1152 tokens each. Layout B assigns apple-right and cup-left.
- Kept Eq. (2), block 18, indices 0–9, sigma-squared update, K≤5/threshold 0.2, sampling configuration, seed, prompt, initial latent, and provisional `eta_scale=0.25` unchanged. Replaced only the half-plane targets with object-sized boxes derived from baseline annotations.
- Baseline source boxes are apple `[6,23,25,44)` (19x21) and cup `[25,20,48,44)` (23x24). Tight swap targets preserve each object's own size and y extent while translating 21 tokens horizontally: apple `[27,23,46,44)` / image `[432,368,736,704)`; cup `[4,20,27,44)` / image `[64,320,432,704)`.
- Objective decreases normally on every update. Indices 0/1/2 use K=5 and reduce total energy `1.4324→0.8370`, `0.7436→0.4474`, and `0.3725→0.2317`; indices 3–9 stop after K=1, ending at `0.1661`. Gradients are finite, model parameters remain gradient-free, runtime is `430.99 s`, and peak CUDA allocation is `2004.41 MiB`.
- Final attention centroids are apple `(32.203,28.607)` and cup `(17.064,27.994)`, both inside their target boxes. Final box-inside ratios are apple `0.7389` and cup `0.6870`. Both objects visibly relocate.
- Tight boxes materially improve extent fidelity versus Stage 6D 0.25x: pixel MAE/RMSE fall from `61.98/77.90` to `53.46/74.52`; apple shape and both object scales are visibly more reasonable. However, the cup still loses its handle and the indoor tabletop background changes to outdoor foliage. Broad half-regions were a major cause of scale inflation, but not the sole cause of identity/shape and background drift.
- Do not freeze the Backward Guidance MVP yet, and do not treat `eta_scale=0.25` as final. Relocation and scale control pass, while cup identity and background preservation remain unresolved.

### Stage 7 — Backward Guidance repeatability sanity check

- Froze the complete Stage 6E method: FLUX.1-dev, 768x768/50 steps/guidance 3.5, BF16 + sequential CPU offload, Eq. (2), Double block 18, guided indices 0–9, `eta_scale=0.25` (`eta=24.154335`), sigma-squared updates, K<=5, and threshold 0.2. Only seed, target token spans, and predeclared tight target boxes changed.
- Exact official-Pipeline baselines and frozen guidance were run for apple/cup seeds 123 and 2024 plus `a yellow banana and a green bottle, realistic photo` at seed 31415. All gradients/updates were finite and descending, no parameter gradients accumulated, and peak CUDA allocation was stable at 2004.70 MiB.
- All 3 new cases pass coarse visible relocation: both object centers/main bodies enter the intended swapped boxes. Final attention box-inside ratios are seed 123 apple/cup `0.7420/0.7139`, seed 2024 `0.7486/0.7219`, and banana/bottle `0.7641/0.6259`; all final attention centroids are inside their half-open targets. No case achieves full visible-instance extent containment.
- Fidelity is not repeatable: every case changes background/composition. Apple/cup identities remain clear (seed 123 preserves the handle; seed 2024 is handleless already at baseline). The extra prompt has clear drift: two baseline bananas become one, and the thin glass bottle becomes a bulky plastic-like bottle.
- Compute cost varies under the frozen threshold: inner updates/runtime are `21 / 421.41 s`, `18 / 386.98 s`, and `48 / 733.49 s`; the banana/bottle case reaches K=5 at indices 0–8 and K=3 at index 9. Conclusion: Stage 6E is repeatable for coarse relocation across the tested seeds/prompt, but not for strict box extent, fidelity preservation, or stable runtime. This is a sanity check, not a tuned or complete method-level evaluation.

## Current key scripts

- `src/09_flux_dev_baseline.py`: official dev baseline and saved initial/intermediate packed latents.
- `src/10_flux_dev_explicit_sampling.py`: explicit dev denoising and stepwise Pipeline comparison.
- `src/11_flux_dev_attention_probe.py`: selected-column attention capture, heatmaps, overlays, and reusable probe helpers.
- `src/12_flux_dev_spatial_signal_validation.py`: representative block/timestep validation and schnell comparison.
- `src/13_flux_dev_layout_objective.py`: dev Eq. (2) layout-objective validation.
- `src/14_flux_dev_gradient_probe.py`: dev single-step latent-gradient validation.
- `src/15_flux_dev_single_step_guidance.py`: dev single-intervention guidance and final-image comparison.
- `src/16_flux_dev_inner_loop_guidance.py`: dev K=1/3/5 single-timestep inner-loop validation.
- `src/17_flux_dev_multi_timestep_guidance.py`: dev sparse three-timestep K=1 guidance with pre-update spatial-signal gating.
- `src/18_flux_dev_paper_aligned_guidance.py`: paper-aligned early FlowMatch sigma-squared Backward Guidance; Stage 6D adds isolated `--eta-scale` outputs while preserving the Stage 6C default.
- `src/19_flux_dev_tight_bbox_guidance.py`: Stage 6E finite object-sized target boxes with the Stage 6D 0.25x sampling/guidance path unchanged.
- `src/20_flux_dev_repeatability.py`: Stage 7 exact baselines and frozen Stage 6E repeatability runner.
- `src/04_flux_schnell_layout_objective.py`: schnell layout-energy reference.
- `src/05_flux_schnell_gradient_probe.py`: schnell gradient-stage reference.
- `src/06_flux_schnell_single_step_guidance.py` and `08_flux_schnell_inner_loop_guidance.py`: later Backward Guidance references.

Key metrics:

- `outputs/baseline/flux_dev_baseline_metrics.json`
- `outputs/explicit_sampling/flux_dev_explicit_sampling_metrics.json`
- `outputs/attention/flux_dev_attention_probe_metrics.json`
- `outputs/spatial_validation/flux_dev_spatial_signal_validation_metrics.json`
- `outputs/layout_objective/flux_dev_layout_objective_metrics.json`
- `outputs/gradient_probe/flux_dev_gradient_probe_metrics.json`
- `outputs/single_step_guidance/flux_dev_single_step_guidance_rel-{0p0005,0p005}_metrics.json`
- `outputs/inner_loop_guidance/flux_dev_inner_loop_k-1-3-5_rel-0p005_metrics.json`
- `outputs/multi_timestep_guidance/flux_dev_multi_timestep_indices-{20-28-35,12-24-35}_rel-0p005_metrics.json`
- `outputs/paper_aligned_guidance/flux_dev_paper_aligned_blocks-18{,_eta-scale-0p5,_eta-scale-0p25}_metrics.json`
- `outputs/paper_aligned_guidance/flux_dev_stage6d_strength_summary.json`
- `outputs/tight_bbox_guidance/flux_dev_tight_bbox_block-18_eta-scale-0p25_metrics.json`
- `outputs/repeatability/stage7_repeatability_summary.json`

## Verified dev configuration

```text
model: black-forest-labs/FLUX.1-dev
prompt: a red apple and a blue cup, realistic photo
seed: 42
resolution: 768x768
steps: 50
guidance_scale: 3.5
dtype: bfloat16
offload: enable_sequential_cpu_offload
diffusers: 0.32.2
torch: 2.3.0+cu121
GPU: RTX 4090 24GB
packed latent: [1, 2304, 64]
image-token grid: 48x48
text/image/joint tokens: 512 / 2304 / 2816
heads × head_dim: 24 × 128
guidance_embeds: true
scheduler: FlowMatchEulerDiscreteScheduler
dynamic shifting: true
mu: 0.8466666667
```

Use the Stage 1 saved initial latent for direct comparisons. Keep `local_files_only=True`, BF16, batch size 1, and sequential CPU offload unless a stage explicitly tests another setting.

## Important implementation findings

- Double Stream separately projects normalized text and image Q/K, concatenates `[text, image]`, applies RoPE, then performs joint attention.
- Selected spatial signal:

  `A[u,i] = mean_heads sum_{k in token_span(i)} exp(logit[u,k]) / sum_{j in all text+image keys} exp(logit[u,j])`

- Compute the denominator in key chunks; never materialize/save full joint attention matrices. Text-only renormalization is diagnostic only.
- Locate target spans from tokenizer character offsets and support multi-token spans.
- Observed-layout diagnostic boxes on the 48×48 grid are apple `(6,23,25,44)` and cup `(25,20,48,44)`; these are evaluation annotations, not desired-layout inputs.
- Layout objective reference:

  `E(A,B,i) = (1 - sum_{u in B} A[u,i] / sum_u A[u,i])^2`

- Historical Stage 3–6D regions are broad halves: left `[0,0,0.5,1]` and right `[0.5,0,1,1]`. Stage 6E keeps Layout B semantics but uses recorded finite apple/cup target boxes; do not silently substitute one target definition for the other.
- Later guidance must optimize only the packed latent. Freeze all model parameters; text encoders and VAE do not participate in backward.
- Do not modify `site-packages`; use project-local processors/wrappers. Do not save all-head/all-block/all-timestep raw attention.

## Next and only next task

Stage 7 is complete. Treat `eta_scale=0.25` and the Stage 6E path as frozen only for this repeatability checkpoint, not as a final method configuration. The evidence supports repeatable coarse relocation but exposes strict-extent, fidelity, and cross-prompt runtime failures. Do not rescue individual Stage 7 cases or silently tune the frozen configuration. The next stage should explicitly choose between an isolated fidelity-preservation redesign and systematic frozen-method ablation/evaluation; no further high-cost run is authorized by this checkpoint.
