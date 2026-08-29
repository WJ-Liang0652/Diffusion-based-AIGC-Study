# SDXL 与 FLUX.1-schnell：初级阶段实验记录

## 1. 实验目的

本阶段使用同一台 RTX 4090（24 GB）服务器和同一 Hugging Face cache，完成两个 Diffusers 文生图 baseline，并学习控制单个推理参数：

- **SDXL**：固定 prompt 与 seed，分别改变 `num_inference_steps` 和 `guidance_scale`。
- **FLUX.1-schnell**：固定 prompt、seed 和 `guidance_scale=0.0`，改变 `num_inference_steps`。

除非表格另有说明，时间均是各实验脚本通过 CUDA 同步后测得的纯 `pipe(...)` wall-clock time；模型加载时间不计入实验表格。

共同 prompt：

> A small red cabin beside a calm mountain lake at sunrise, realistic photograph

## 2. SDXL baseline

脚本：[sdxl_inference.py](sdxl/sdxl_inference.py)

| 项目 | 设置 / 已验证结果 |
| --- | --- |
| 模型 | `stabilityai/stable-diffusion-xl-base-1.0` |
| Pipeline | `StableDiffusionXLPipeline` |
| Prompt | 上述共同 prompt |
| Seed | 42（CUDA generator） |
| Steps | 30 |
| `guidance_scale` | 7.0 |
| dtype | `torch.float16` |
| 设备方式 | `pipe.to("cuda")` |
| 输出 | [sdxl_seed42.png](sdxl/outputs/sdxl_seed42.png) |

该最小 baseline 已成功生成图片。baseline 脚本当时没有记录纯推理时间或峰值显存，因此本文不为它填写这两项数据；这些指标以下面的单变量实验 CSV 为准。

### 输出图像

![SDXL baseline，seed 42](sdxl/outputs/sdxl_seed42.png)

## 3. SDXL：`num_inference_steps` 单变量实验

脚本：[steps_experiment.py](sdxl/experiments/steps_experiment.py)，原始记录：[steps_results.csv](sdxl/experiments/steps_results.csv)。固定 seed=42、`guidance_scale=7.0`、1024×1024 和 `torch.float16`，只改变 steps。

| Steps | 纯推理时间 | Peak allocated | Peak reserved | 输出 |
| ---: | ---: | ---: | ---: | --- |
| 10 | 1.866 s | 10720.3 MiB | 14116.0 MiB | [image](sdxl/outputs/steps_10_seed42.png) |
| 20 | 2.708 s | 10720.3 MiB | 14118.0 MiB | [image](sdxl/outputs/steps_20_seed42.png) |
| 30 | 3.928 s | 10720.3 MiB | 14118.0 MiB | [image](sdxl/outputs/steps_30_seed42.png) |
| 50 | 6.342 s | 10721.0 MiB | 14118.0 MiB | [image](sdxl/outputs/steps_50_seed42.png) |

从数值可直接看到，steps 增加使推理时间从 1.866 s 增至 6.342 s；峰值显存基本保持在约 10.72 GiB allocated、14.12 GiB reserved。该实验没有记录主观画质评分，因此不能只凭这些数值断言哪一组图像质量最好。

### 输出图像

<p>
  <img src="sdxl/outputs/steps_10_seed42.png" alt="SDXL steps 10, seed 42" width="48%" />
  <img src="sdxl/outputs/steps_20_seed42.png" alt="SDXL steps 20, seed 42" width="48%" />
</p>
<p>
  <img src="sdxl/outputs/steps_30_seed42.png" alt="SDXL steps 30, seed 42" width="48%" />
  <img src="sdxl/outputs/steps_50_seed42.png" alt="SDXL steps 50, seed 42" width="48%" />
</p>

## 4. SDXL：`guidance_scale` 单变量实验

脚本：[guidance_experiment.py](sdxl/experiments/guidance_experiment.py)，原始记录：[guidance_results.csv](sdxl/experiments/guidance_results.csv)。固定 seed=42、steps=30、1024×1024 和 `torch.float16`，只改变 guidance scale。

| Guidance scale | 纯推理时间 | Peak allocated | Peak reserved | 输出 |
| ---: | ---: | ---: | ---: | --- |
| 1.0 | 2.927 s | 10716.5 MiB | 14100.0 MiB | [image](sdxl/outputs/guidance_1_seed42.png) |
| 3.0 | 3.981 s | 10718.5 MiB | 14104.0 MiB | [image](sdxl/outputs/guidance_3_seed42.png) |
| 5.0 | 3.922 s | 10717.5 MiB | 14104.0 MiB | [image](sdxl/outputs/guidance_5_seed42.png) |
| 7.0 | 3.936 s | 10718.5 MiB | 14104.0 MiB | [image](sdxl/outputs/guidance_7_seed42.png) |
| 10.0 | 3.929 s | 10718.5 MiB | 14104.0 MiB | [image](sdxl/outputs/guidance_10_seed42.png) |

`guidance_scale=1.0` 的时间和显存略低；3.0、5.0、7.0、10.0 的实测时间约为 3.92–3.98 s、峰值显存几乎一致。此实验用于观察 prompt 条件强度的影响；没有记录人工或自动图像质量指标，因此不把它解释为“质量参数”的排名实验。

### 输出图像

<p>
  <img src="sdxl/outputs/guidance_1_seed42.png" alt="SDXL guidance 1, seed 42" width="32%" />
  <img src="sdxl/outputs/guidance_3_seed42.png" alt="SDXL guidance 3, seed 42" width="32%" />
  <img src="sdxl/outputs/guidance_5_seed42.png" alt="SDXL guidance 5, seed 42" width="32%" />
</p>
<p>
  <img src="sdxl/outputs/guidance_7_seed42.png" alt="SDXL guidance 7, seed 42" width="48%" />
  <img src="sdxl/outputs/guidance_10_seed42.png" alt="SDXL guidance 10, seed 42" width="48%" />
</p>

## 5. FLUX.1-schnell baseline

脚本：[flux_inference.py](flux/flux_inference.py)

| 项目 | 设置 / 已验证结果 |
| --- | --- |
| 模型 | `black-forest-labs/FLUX.1-schnell` |
| Pipeline | `FluxPipeline` |
| Prompt | 上述共同 prompt |
| Seed | 42（CPU generator） |
| Steps | 4 |
| `guidance_scale` | 0.0 |
| 分辨率 | 1024×1024 |
| dtype | `torch.bfloat16` |
| 显存策略 | `enable_model_cpu_offload()` |
| 纯推理时间 | 34.929 s |
| Peak allocated / reserved | 23110.0 / 23516.0 MiB |
| 输出 | [flux_schnell_seed42.png](flux/outputs/flux_schnell_seed42.png) |

### 输出图像

![FLUX.1-schnell baseline，steps 4，seed 42](flux/outputs/flux_schnell_seed42.png)

## 6. FLUX.1-schnell：`num_inference_steps` 单变量实验

脚本：[steps_experiment.py](flux/experiments/steps_experiment.py)，原始记录：[steps_results.csv](flux/experiments/steps_results.csv)。固定共同 prompt、seed=42、1024×1024、`guidance_scale=0.0`、`torch.bfloat16` 和 CPU offload，只改变 steps。

| Steps | 纯推理时间 | Peak allocated | Peak reserved | 输出 |
| ---: | ---: | ---: | ---: | --- |
| 1 | 31.388 s | 23109.5 MiB | 23514.0 MiB | [image](flux/outputs/flux_schnell_steps1_seed42.png) |
| 2 | 29.583 s | 23110.0 MiB | 23516.0 MiB | [image](flux/outputs/flux_schnell_steps2_seed42.png) |
| 4 | 28.226 s | 23110.0 MiB | 23516.0 MiB | [image](flux/outputs/flux_schnell_steps4_seed42.png) |
| 8 | 29.649 s | 23110.0 MiB | 23516.0 MiB | [image](flux/outputs/flux_schnell_steps8_seed42.png) |

四组峰值显存几乎不变，约为 23.11 GiB allocated、23.52 GiB reserved。总时间未随 steps 单调增长：这说明在当前 CPU offload 配置下，组件在 CPU 与 GPU 之间搬运及其他固定开销不可忽略。它不代表更多 steps 一定更快，也不构成模型本身速度的比较。

### 输出图像

<p>
  <img src="flux/outputs/flux_schnell_steps1_seed42.png" alt="FLUX steps 1, seed 42" width="48%" />
  <img src="flux/outputs/flux_schnell_steps2_seed42.png" alt="FLUX steps 2, seed 42" width="48%" />
</p>
<p>
  <img src="flux/outputs/flux_schnell_steps4_seed42.png" alt="FLUX steps 4, seed 42" width="48%" />
  <img src="flux/outputs/flux_schnell_steps8_seed42.png" alt="FLUX steps 8, seed 42" width="48%" />
</p>

## 7. 两个 Pipeline 的对照

| 方面 | SDXL | FLUX.1-schnell |
| --- | --- | --- |
| 本阶段典型 steps | baseline 为 30；测试 10、20、30、50 | baseline 为 4；测试 1、2、4、8 |
| 核心去噪网络 | UNet | Transformer |
| 本阶段 guidance 设置 | baseline 7.0；已测试 1、3、5、7、10 | baseline 0.0；未做 guidance ablation |
| 设备方式 | 整个 pipeline 放入 CUDA | CPU offload，在需要时把组件移到 GPU |
| 本阶段实验显存量级 | 约 10.72 GiB allocated / 14.12 GiB reserved | 约 23.11 GiB allocated / 23.52 GiB reserved |
| 已测实验时间量级 | SDXL steps 实验为 1.866–6.342 s | FLUX steps 实验为 28.226–31.388 s |

SDXL 的 `guidance_scale` 实验与 FLUX.1-schnell 的 `guidance_scale=0.0` **不能机械地直接类比**。前者实验的是 SDXL 的 Classifier-Free Guidance（CFG）强度；后者遵循 schnell 的当前 baseline 设置。本阶段**没有**进行专门的 FLUX guidance ablation。

同时，FLUX 使用了 CPU offload，而 SDXL 实验将 pipeline 常驻 GPU。因此上述 wall-clock time 受到不同显存策略、CPU↔GPU 数据搬运和固定开销影响，**不能作为 SDXL 与 FLUX 模型速度的严格 benchmark**。

## 8. 核心概念速记

- **Pipeline**：Diffusers 把文本编码、去噪、scheduler 更新和 VAE 解码封装成可调用的推理流程。
- **latent**：生成过程主要在较小的连续潜变量空间中进行，而不是直接在 RGB 像素上去噪。
- **scheduler**：给出每一步的噪声时间表和 latent 更新规则，决定如何从随机噪声逐步走向图像 latent。
- **VAE**：把最终 latent 解码成可保存、可查看的 RGB 图像。
- **seed**：初始化随机数生成器；固定模型、参数和随机数路径时，它用于复现相同的初始噪声条件。
- **`num_inference_steps`**：去噪 / sampling 的迭代次数。它通常影响采样轨迹和推理开销，但并不是单独的“质量保证”开关。
- **`guidance_scale`**：条件文本对生成轨迹的引导强度。在 SDXL 中这里对应 CFG 的强度；过大可能损害自然度或造成伪影。
- **dtype**：推理张量精度。本阶段 SDXL 使用 `float16`，FLUX 使用 `bfloat16`，以降低显存占用并适配 GPU 推理。
- **CPU offload**：不把所有 FLUX 组件同时常驻 GPU，而在使用某组件时将其移入 GPU，之后移回 CPU。它降低同时驻留的显存压力，但增加数据搬运和 wall-clock 时间。

两条调用链可概括为：

```text
SDXL: prompt -> text encoder -> latent -> UNet + scheduler -> VAE -> image
FLUX: prompt -> text encoder -> latent -> Transformer + scheduler -> VAE -> image
```
