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

## Current key scripts

- `src/09_flux_dev_baseline.py`: official dev baseline and saved initial/intermediate packed latents.
- `src/10_flux_dev_explicit_sampling.py`: explicit dev denoising and stepwise Pipeline comparison.
- `src/11_flux_dev_attention_probe.py`: selected-column attention capture, heatmaps, overlays, and reusable probe helpers.
- `src/12_flux_dev_spatial_signal_validation.py`: representative block/timestep validation and schnell comparison.
- `src/13_flux_dev_layout_objective.py`: dev Eq. (2) layout-objective validation.
- `src/14_flux_dev_gradient_probe.py`: dev single-step latent-gradient validation.
- `src/15_flux_dev_single_step_guidance.py`: dev single-intervention guidance and final-image comparison.
- `src/16_flux_dev_inner_loop_guidance.py`: dev K=1/3/5 single-timestep inner-loop validation.
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

- Desired regions remain left `[0,0,0.5,1]` and right `[0.5,0,1,1]`; Layout A is apple-left/cup-right and Layout B is apple-right/cup-left.
- Later guidance must optimize only the packed latent. Freeze all model parameters; text encoders and VAE do not participate in backward.
- Do not modify `site-packages`; use project-local processors/wrappers. Do not save all-head/all-block/all-timestep raw attention.

## Next and only next task

Stage 6B: test a minimal multi-timestep `FLUX.1-dev` Backward Guidance run.

- Keep Double block 18, Layout B, the unchanged objective, and normalized `relative_step=0.005` update; use one fresh update at each of a small fixed set of timesteps rather than more inner iterations at index 35.
- Verify the spatial signal at every selected timestep, then complete normal denoising and compare against the exact baseline plus Stage 6A K=5.
- Record per-timestep objective/attention changes, final geometry, runtime, and VRAM; visible position change remains the success criterion.
- Do not perform a systematic timestep/strength sweep in the same stage.
