# Tight-box quantitative summary

All values are from existing Stage 1–7 artifacts. `box-inside ratio` means final block-18 target-box attention-mass ratio; it is not instance-mask IoU or a detector metric. `Coarse relocation` is the recorded visual judgement that both object centers/main bodies moved into the intended swapped regions.

| Case | Prompt | Seed | Object 1 ratio | Object 2 ratio | Coarse relocation | Baseline (s) | Guided (s) | Multiplier |
|---|---|---:|---:|---:|---|---:|---:|---:|
| apple_cup_seed42 | a red apple and a blue cup, realistic photo | 42 | apple 0.7389 | cup 0.6870 | yes | 213.07 | 430.99 | 2.023x |
| apple_cup_seed123 | a red apple and a blue cup, realistic photo | 123 | apple 0.7420 | cup 0.7139 | yes | 224.16 | 421.41 | 1.880x |
| apple_cup_seed2024 | a red apple and a blue cup, realistic photo | 2024 | apple 0.7486 | cup 0.7219 | yes | 228.56 | 386.98 | 1.693x |
| banana_bottle_seed31415 | a yellow banana and a green bottle, realistic photo | 31415 | banana 0.7641 | bottle 0.6259 | yes | 234.72 | 733.49 | 3.125x |

## Aggregate

- Eight-object mean box-inside ratio: **0.7178**.
- Eight-object minimum box-inside ratio: **0.6259** (bottle, seed 31415).
- Preliminary tested-case relocation count: **4/4**. This is not a formal benchmark success rate.
- Stage 7 three-case mean runtime overhead: **+284.81 s**, mean multiplier **2.233x**, or **+123.27%** versus each matching exact baseline.
- Four-case mean runtime multiplier (descriptive only): **2.180x**.
