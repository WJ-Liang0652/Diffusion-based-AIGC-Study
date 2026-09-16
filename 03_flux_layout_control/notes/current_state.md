# FLUX Layout Control — Current State

## Fixed baseline

- Model: `black-forest-labs/FLUX.1-schnell`
- Prompt: `a red apple and a blue cup, realistic photo`
- Seed: `42`
- Resolution: `768x768`
- Steps: `4`
- Guidance scale: `0.0`
- Dtype/offload: BF16 + `enable_sequential_cpu_offload()`
- Environment: Diffusers `0.32.2`, PyTorch `2.3.0+cu121`, RTX 4090 24 GB

## Completed scripts

- `00_flux_schnell_baseline.py`: generated the fixed baseline twice. Passed; decoded pixels and PNG hashes were identical. Runtime after load was about 17–32 s; peak allocated VRAM about 1.38 GiB.
- `01_flux_schnell_explicit_sampling.py`: explicitly expanded prompt encoding, FLUX latent packing/image IDs, FlowMatch schedule, transformer loop, scheduler steps, unpacking, and VAE decode. Passed; output matched `FluxPipeline.__call__` exactly (`max/mean pixel difference = 0`).
- `02_flux_schnell_attention_probe.py`: verified installed `FluxAttnProcessor2_0`, Q/K/V organization, token counts, and memory-bounded extraction of selected image-query → text-key probabilities. Passed and non-invasive; generated image hash matched baseline.
- `03_flux_schnell_spatial_signal_validation.py`: compared joint-normalized and text-only-renormalized signals at representative timesteps and early/middle/late Double/Single blocks. Passed; native joint-normalized signal from middle/late Double blocks localized the objects most reliably.
- `04_flux_schnell_layout_objective.py`: implemented paper Eq. (2) for layouts A/B at `t=500`, Double blocks 9/18. Passed; Layout A energy was much lower than swapped Layout B for both blocks.
- `05_flux_schnell_gradient_probe.py`: differentiated Layout B energy with respect to the packed `t=500` denoising state. Passed; gradients were finite/nonzero, model parameters had no gradients, and small one-step updates reduced the objective for blocks 18 and 9.
- `06_flux_schnell_single_step_guidance.py`: applied one block-18 update at `t=500`, then completed normal denoising. Mechanically passed, but spatial-control criterion did not: objective decreased and final pixels changed slightly, while apple/cup geometry did not visibly move.
- `07_flux_schnell_early_step_guidance.py`: repeated the same intervention at `t=750`. Signal check and gradient passed. Final pixel effects were stronger than at `t=500`, but still no clearly visible object relocation.

## Confirmed attention/spatial signal

- T5 context has 512 positions; packed image sequence has 2304 tokens, mapped row-major to a `48x48` grid. Joint sequence length is 2816.
- This checkpoint uses 24 heads with head dimension 128.
- Current prompt: `apple` is T5 token index 3 (`▁apple`, id 8947); `cup` is index 8 (`▁cup`, id 4119). Code supports multi-token spans by summing probabilities over all tokens overlapping the target character span.
- Double Stream separately projects normalized text and image Q/K/V, concatenates them as `[text, image]`, applies RoPE, and performs joint attention. Single Stream receives an already concatenated `[text, image]` sequence and uses shared Q/K/V projections.
- Selected signal: target-token-span summed, head-mean, native joint-normalized attention probability in the direction image query → text key:

  `A[u,i] = mean_heads sum_{k in token_span(i)} exp(logit[u,k]) / sum_{j in all text+image keys} exp(logit[u,j])`

- The denominator is computed in key chunks without materializing the full joint matrix. A full per-block matrix would be about 363 MiB in BF16.
- Text-only renormalization uses the same RoPE Q/K logits but normalizes over only the 512 text keys. It is a diagnostic proxy comparable to SDXL cross-attention, not native FLUX attention probability. It showed more background leakage and weaker localization than joint normalization.

## Block choice

- Double block 0 was almost spatially uniform.
- Double blocks 9 and 18 showed clear apple-left/cup-right maps at middle/late timesteps.
- Double block 18 had the strongest overall localization/objective separation and a slightly stronger gradient/update response than block 9, so it is the current primary block. Block 9 remains a minimal control, not a second default target.
- Single block 0 retained some spatial semantics; middle/late Single blocks were less stable or nearly uniform. Current evidence does not support using Single Stream as the primary objective source.

## Layout objective

For an aggregated spatial map `A`, region `B`, and object `i`:

`E(A,B,i) = (1 - sum_{u in B} A[u,i] / sum_u A[u,i])^2`

Total energy is the sum of apple and cup energies. Fixed regions are independent of the generated image:

- Left: normalized `[0.0, 0.0, 0.5, 1.0]` → grid `[0, 0, 24, 48]`
- Right: normalized `[0.5, 0.0, 1.0, 1.0]` → grid `[24, 0, 48, 48]`
- Layout A: apple → left, cup → right
- Layout B: apple → right, cup → left

At block 18 / `t=500`, Layout A total energy was `0.05385`; Layout B was `1.39974`. At block 9 they were `0.06201` and `1.37999`.

## Latent update definition

The only optimized variable is the packed latent/state entering the transformer at the selected timestep, shape `[1, 2304, 64]`, BF16. Model parameters are frozen.

`g = ∂E_layout / ∂x`

`eta = relative_step * ||x||_2 / ||g||_2`

`x_new = x - eta * g`

Thus the desired update L2 norm is `relative_step * ||x||_2`. The update is cast to BF16 before subtraction, so metrics also record the actual post-rounding relative update norm. Future guidance experiments must keep this definition unless an explicit ablation changes it.

## Single-step results

### t=500, Double block 18, Layout B

- Gradient norm: `0.04655`; model parameter grad count: `0`.
- Relative `0.002`: energy `1.39974 → 1.37364`; apple/right mass `0.14990 → 0.15632`; cup/left mass `0.17715 → 0.18647`.
- Relative `0.005`: energy `1.39974 → 1.32994`; apple/right mass `0.14990 → 0.17056`; cup/left mass `0.17715 → 0.19878`.
- Final mean absolute pixel differences were `0.2458/255` and `0.3242/255`. No visible object relocation; changes were mainly weak pixel/texture perturbations.

### t=750, Double block 18, Layout B

- Spatial signal remained reliable: baseline-consistent Layout A energy `0.04007`, swapped Layout B energy `1.47452`.
- Gradient norm: `0.04774`; model parameter grad count: `0`.
- Relative `0.002`: energy `1.47452 → 1.44933`; apple/right mass `0.14811 → 0.15406`; cup/left mass `0.13467 → 0.14343`.
- Relative `0.005`: energy `1.47452 → 1.41726`; apple/right mass `0.14811 → 0.16181`; cup/left mass `0.13467 → 0.15460`.
- Final mean absolute pixel differences were `0.3791/255` and `0.6052/255`, respectively 1.54x and 1.87x the matching `t=500` effects. Despite stronger propagation, there was still no clearly visible geometric movement; color-mask centroid shifts at `0.005` were only about apple `+0.21 px`, cup `-0.03 px`.

## Evidence boundary

Experimentally supported:

- schnell baseline is reproducible; explicit sampling matches the installed pipeline exactly.
- The selected quantity is a genuine joint-attention probability derived from the actual normalized/RoPE Q/K, not an SDXL cross-attention map.
- Double block 18 at `t=500` and `t=750` provides semantically meaningful spatial maps for this prompt/seed.
- Eq. (2) distinguishes the observed layout from its left/right swap.
- The objective has a finite, nonzero gradient with respect to packed latents; model parameters remain gradient-free.
- One normalized negative-gradient step lowers the objective and increases target-region attention mass.
- Moving the intervention from `t=500` to `t=750` increases final pixel propagation but does not by itself produce visible layout relocation.

Not yet established:

- That the selected signal/block generalizes across prompts, layouts, seeds, or FLUX.1-dev.
- That block 18 is globally optimal rather than the best current MVP choice.
- That attention-energy reduction reliably causes object geometry or position changes.
- That repeated optimization remains stable, preserves quality, and avoids graph/memory growth.
- Any claim of method-level complete reproduction or equivalence to SDXL cross-attention guidance.

## Latest valid progress: FLUX.1-dev Stage 1

- Added `src/09_flux_dev_baseline.py` and `src/10_flux_dev_explicit_sampling.py`; existing schnell scripts and outputs were not modified.
- The dev baseline uses the fixed prompt `a red apple and a blue cup, realistic photo`, seed `42`, resolution `768x768`, `50` steps, BF16, sequential CPU offload, and `guidance_scale=3.5`.
- The baseline creates one initial packed latent tensor with shape `[1, 2304, 64]`, passes it explicitly to the official `FluxPipeline`, and saves all 50 post-scheduler packed latent states for alignment.
- Dev-specific behavior was confirmed: `transformer.config.guidance_embeds=True`; guidance tensor shape `[1]` with value `3.5`; `FlowMatchEulerDiscreteScheduler` with `shift=3.0`, `use_dynamic_shifting=True`; sequence-length shift `mu=0.8466666667`.
- Explicit sampling follows the installed Diffusers `0.32.2` Pipeline path for prompt encoding, latent/image-ID preparation, dynamic-shifted timesteps, guidance tensor, Transformer calls, scheduler steps, unpacking, and VAE decode.
- The explicit run reused the exact saved initial latent. All 50 intermediate packed latent states matched the Pipeline exactly. Final packed latent max/mean absolute differences were `0.0 / 0.0`; decoded RGB image max/mean absolute differences were `0 / 0`, with exact pixel equality.
- Baseline runtime was `213.1 s`; explicit runtime was `214.9 s`; peak CUDA allocation was about `1384.7 / 1384.4 MiB`, with no OOM or abnormal memory growth.

Outputs:

- `outputs/baseline/flux_dev_baseline_seed-42_steps-50.png`
- `outputs/baseline/flux_dev_baseline_metrics.json`
- `outputs/baseline/flux_dev_baseline_state.pt` (initial latent and Pipeline step states)
- `outputs/explicit_sampling/flux_dev_explicit_seed-42_steps-50.png`
- `outputs/explicit_sampling/flux_dev_explicit_sampling_metrics.json`
- `outputs/explicit_sampling/flux_dev_explicit_state.pt`

Stage 1 is complete for this fixed configuration. The next stage is dev attention probing, rechecking token spans and image-query → text-key spatial signal before any dev guidance implementation.
