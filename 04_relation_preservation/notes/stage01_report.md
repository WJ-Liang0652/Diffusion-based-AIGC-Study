# 任务 01：SDXL 单样本复查与 LMS 状态恢复验收

日期：2026-10-05（UTC）。本轮已完成并停止；没有进入两物体关系实验、参数 sweep、多 seed 或新方法设计。

**恢复验收通过**：从磁盘恢复后的第一步与最终 latent 最大绝对误差均为 0，`torch.equal=True`；最终图像逐像素一致。旧 guidance 的 50 次 latent 梯度均有限且非零，模型参数没有累计梯度。图像中小屋明显左移，但未整体进入目标框，并伴随其他内容变化。这个单样本不是“修复新约束破坏旧关系”的证据，也不是方法级完整复现。

## 代码与历史状态

开始时 `04_relation_preservation/` 不存在。先读取实际根 AGENTS.md、git status/diff、旧脚本 18 和 19、旧 LMS 指标与图像，以及本环境的 LMS 源码。根规则已是当前 SDXL 主线，未修改它。

基准 Git HEAD：`759d6efb27b1cd2ac64fd0c12c47ff34b27417cd`。开始时已有的用户工作全部保留：

```text
 M 03_flux_layout_control/src/18_flux_dev_paper_aligned_guidance.py
 M AGENTS.md
?? 03_flux_layout_control/outputs/strength_trajectory/
?? 03_flux_layout_control/src/22_flux_dev_strength_trajectory.py
```

本轮只新增以下目录内的文件：

- `04_relation_preservation/README.md`：限定任务与停止条件。
- `04_relation_preservation/.gitignore`：忽略 checkpoint `.pt`、运行 `.log` 与 Python 缓存；已用 `git check-ignore` 验证。
- `04_relation_preservation/src/attention_processor.py`：逐字复制旧脚本 19 的两种 processor 类，删除顶层实验入口；未 import/runpy 旧脚本。
- `04_relation_preservation/src/stage01_sdxl.py`：参数化采样、guidance、序列化/恢复和实际续采样比较。
- 本报告与 `outputs/` 内的配置、命令来源、标量、checkpoint 和图像。

没有修改历史脚本、历史输出、site-packages 或核心环境；没有下载、commit、push 或 pull。

执行前做了 `py_compile`、入口 `--help` 和 `git diff --check`。结束时再次检查 git 状态和差异。运行源码 SHA256 已保留；交付前 processor 仅去除文件尾一行多余空白，不改变可执行语句，另记录交付源码 SHA256，未因此重复 GPU 计算。具体哈希、该空白变更和启动时状态见 [provenance.json](../outputs/stage01_seed42_verified/provenance.json)。

## 实际环境与有效配置

| 项目 | 实际值 |
|---|---|
| Python | `/root/miniconda3/bin/python`，3.12.3 |
| 依赖 | torch 2.3.0+cu121、diffusers 0.32.2、transformers 4.48.3、accelerate 1.3.0 |
| CUDA | torch runtime 12.1；驱动 595.71.05（nvidia-smi 显示驱动支持 CUDA 13.2） |
| GPU | NVIDIA RTX 4090，24564 MiB 可见总显存 |
| 模型 | `stabilityai/stable-diffusion-xl-base-1.0`，本地缓存 snapshot `462165984030d82259a11f4367a4eed129e94a7b` |
| HF_HOME | `/root/autodl-tmp/huggingface`；`local_files_only=True`，HF/Transformers offline 环境变量均为 1 |
| 精度/尺寸 | 模型 FP16，1024×1024，latent `[1,4,128,128]`，选中 attention 网格 32×32 |
| batch | 图像与 conditional guidance batch=1；原实现 CFG forward 将无条件/有条件拼为 batch=2；分支顺序执行 |
| seed | 42，CUDA 显式 generator |
| 采样器 | LMSDiscreteScheduler，由模型原 Euler 配置构造；51 步，`order=4`，`leading` spacing，offset=1，epsilon prediction，非 Karras |
| CFG | 7.0，negative prompt=None，guidance_rescale=0 |
| 初始状态 | 单份初始 latent 分别 clone 给 baseline/guided；SHA256 `684a56c407477ea8a1f77ba8e1080016b065fdc1f8af534f12364b3876a166b2` |
| 初始 sigma | LMS `init_noise_sigma=11.073580741882324`；第一步 sigma=11.028335571289062 |
| 目标 | `cabin`，两个 tokenizer 索引均为 `[4]`；框 `(0.10,0.50,0.40,0.82)`，坐标归一化、原点左上 |
| 层 | `mid_block.attentions.0.transformer_blocks.0.attn2` 与 `up_blocks.0.attentions.0.transformer_blocks.0.attn2` |
| guidance | 最早 10 个 diffusion index（0–9），每步最多 5 次 inner update；loss_scale=30，E threshold=0.2，相对更新安全上限=0.1 |
| checkpoint | 完成 25 次 scheduler.step 后（index 24、t=495），next_index=25、next_t=476；还需 26 次更新 |

Prompt：

```text
A small red cabin beside a calm mountain lake at sunrise, realistic photograph
```

旧脚本开头的 `NUM_INFERENCE_STEPS=10` 被后续 `51` 覆盖；本轮采用有效值 51。旧脚本 18 用 Euler，19 仅改为 LMS；本轮只运行 LMS，不额外跑 Euler。

损失与更新沿用旧语义：先 conditional-only forward，取目标 token 的 head/token 均值；每层计算原始 attention 在框内的占比 `r`，`E=((1-r_mid)^2+(1-r_up)^2)/2`，更新 `z ← z−sigma²*grad(30E)`。early-stop 的下一轮判断使用上一轮更新前的 E。每个 diffusion index 最后重新 CFG forward，然后仅调用一次 scheduler.step；没有换损失或搜索参数。

完整有效配置、全部 timestep/sigma、processor 类型和环境见 [config.json](../outputs/stage01_seed42_verified/config.json)、[environment.json](../outputs/stage01_seed42_verified/environment.json)。模型参数冻结；conditioning 在 no_grad 下编码；text encoder 和 VAE 不参与 guidance 反向传播。解码前将 UNet/text encoders 移到 CPU，VAE 按历史逻辑 force_upcast；三张图使用相同解码路径。

## 实际运行命令

在 `/root/autodl-tmp/diffusion-reproduction` 执行：

```bash
set -o pipefail
HF_HOME=/root/autodl-tmp/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/root/miniconda3/bin/python -u 04_relation_preservation/src/stage01_sdxl.py \
  --scheduler lms --seed 42 --height 1024 --width 1024 --steps 51 \
  --cfg-scale 7 --target-word cabin --target-box 0.10 0.50 0.40 0.82 \
  --guided-steps 10 --inner-iters 5 --loss-scale 30 --loss-threshold 0.2 \
  --checkpoint-after 25 --output-dir 04_relation_preservation/outputs/stage01_seed42_verified \
  2>&1 | tee 04_relation_preservation/outputs/stage01_seed42_verified.log
```

入口要求输出目录为空。复跑时指定新的空输出目录，避免覆盖证据；本轮没有再执行。

首次启动使用输出目录 `outputs/stage01_seed42`，在初始 latent 生成前失败：diffusers 0.32.2 的 `prepare_latents` 要求 `torch.device`，传入字符串导致 `AttributeError: 'str' object has no attribute 'type'`。读取完整 traceback 后仅把该参数改为 `torch.device('cuda')`，随后以上命令成功退出。失败目录仅有 environment/provenance，未产生采样轨迹或 checkpoint；日志保留在 `outputs/stage01_seed42.log`。没有因为这个局部错误修改环境。

运行中出现一条 cuDNN execution plan 不支持的 warning，随后正常计算，没有 OOM 或非有限 latent。warning 没有被当作实验成功证据；恢复结果另行逐项比较。

## Guidance：attention 与最终几何分别判断

| 检查 | 结果 |
|---|---|
| inner 更新数 | 50（10×5），全部执行；没有触发 safety stop |
| latent 梯度 | 全部有限且非零；历史 FP16 范数范围 0.01318359375–0.03497314453125；CSV 另有 FP32 范数诊断 |
| 更新范数 | 0.62451171875–2.1015625；最大相对更新 0.0009069442749023438，低于 0.1 |
| 模型参数 | 全部冻结，`.grad` 始终不存在；无梯度累计 |
| attention processor | 每次捕获后在 finally 中恢复原对象，清空 captured 引用；普通采样与恢复均使用原 AttnProcessor2_0 |
| 同一 timestep 的 attention | 10 个引导步全部 E 下降，mid/up 框内占比均上升 |
| index 0 | E 0.8020865321→0.7982659340；mid 0.1018467695→0.1026703864；up 0.1069749147→0.1104318872 |
| index 9 | E 0.7823652029→0.7787323594；mid 0.1047670320→0.1050274447；up 0.1263361871→0.1302363276 |

这是每步相同 timestep、不同 latent 的更新前后比较，不能把不同 timestep 的趋势直接解释为 guidance 因果效果。只保存标量，没有保存完整 block/head/timestep attention。

人工查看本轮 baseline/guided 与目标框图：baseline 小屋位于右侧，guided 小屋明显移至左中部，大小、屋顶朝向也变化；小屋仍然存在。guided 小屋与目标框相交，但屋顶在框上方、右侧部分越过 x=0.40，因此**不能宣布整体入框成功**。本轮没有预注册“中心入框/整体入框/IoU”的最终判据，没有把近似位置当成检测器定量结果。坡面由偏黄绿变为明显红色，云层和湖岸也变化。这些内容变化只是辅助观察，没有测量“已有正确二维关系”的破坏。

baseline/guided 最终 latent 最大绝对差 3.6591796875；图像每通道平均绝对差 21.94786390（0–255），最大差 222；不同 RGB 像素比例 0.99986935。这些差异不能替代布局成功或旧关系保持指标。

与旧 LMS 输出只读比较：baseline 图逐像素完全一致；guided 图不完全一致，平均绝对差 1.51937358、最大差 132。10 行 energy_final 最大偏差 5.37634e-5，up_ratio_last 最大偏差 6.22272e-5，可见左移趋势一致。本轮保留了相同数学损失与更新，但新入口使用通用 `sum` 聚合两项，旧代码显式相加，计算图结构并非逐字一致；FP16 梯度累加次序是数值偏差的候选原因，**未通过额外诊断确认**，不声称历史 guided 的逐位复现。这个限制与本轮无 guidance 磁盘恢复误差为 0 的验收分开报告。没有挑 seed 或救参重跑。

原始记录见 [guided_inner_iteration_metrics.csv](../outputs/stage01_seed42_verified/guided_inner_iteration_metrics.csv)、[guided_step_metrics.csv](../outputs/stage01_seed42_verified/guided_step_metrics.csv)、[historical_scalar_comparison.json](../outputs/stage01_seed42_verified/historical_scalar_comparison.json)。

## Checkpoint 内容与恢复逻辑

`outputs/stage01_seed42_verified/baseline_checkpoint.pt` 为本地 torch 格式，1,794,042 bytes（约 1.71 MiB），已被本轮 `.gitignore` 忽略。保存前估算 latent/初始 latent/四项历史合计 786,432 bytes，另有 conditioning 和少量状态。没有权重、账号凭据或全量 attention。

保存内容：

- 当前 latent、共享初始 latent、边界字符串 `after scheduler.step`、next_index=25。
- 完整 scheduler config 与运行时实例状态（排除 `_internal_dict`，配置另存）：`_step_index`、`_begin_index`、`derivatives`、`timesteps`、`sigmas`、`num_inference_steps`、`is_scale_input_called`、`use_karras_sigmas`、`betas`、`alphas`、`alphas_cumprod`。
- 四项 derivatives 均为 `[1,4,128,128]`、FP16、原 device `cuda:0`；文件内 CPU clone 并记录原 device。
- conditional/CFG prompt embeddings、pooled text embeddings、SDXL time_ids。
- 有效配置、实际环境和所有 attention processor 类型描述。
- CPU/CUDA RNG、显式 CUDA generator 状态和 device，另保留 Python/NumPy RNG。LMS step 在当前配置下不使用 generator 或额外随机采样；仍保存状态供完整恢复。

baseline 主轨迹在第 25 次更新后保存 CPU 独立副本，然后保持原 scheduler 和 latent 不变继续采样，记录下一次更新后的 latent 与最终 latent。guided 使用另一个全新 scheduler 与初始 latent clone。随后真正从磁盘加载 checkpoint，为恢复分支构造新的 LMS 实例，将保存的全部 runtime fields 和各张量的原 dtype/device 恢复；**没有调用 set_timesteps 去跳到中间**。构造器的默认初始 schedule 随后被完整 snapshot 覆盖，不参与续采样。

恢复分支的 derivatives 列表和张量不与 baseline 共享；还检查 snapshot 与加载内容完全一致、conditioning/初始 latent/当前 latent 一致、CPU/CUDA/generator RNG 状态恢复、原 processor 正确安装。模型及原 processor 无权重变化；checkpoint 仅保存类型，利用同一 pipeline 的原 processor 对象恢复。本验收在同进程内进行，但分支的 latent、conditioning 和 scheduler 历史全部来自磁盘独立加载，不依赖 baseline 的活历史或重新编码 prompt。未声称跨硬件、跨版本或新进程等价。

| 比较项 | 最大绝对误差 | 完全一致 |
|---|---:|---|
| 恢复后的第一步：index 25，t=476，scheduler.step 后 | 0.0 | `torch.equal=True`，dtype 相同、均有限 |
| 51 步最终 latent | 0.0 | `torch.equal=True`，dtype 相同、均有限 |
| 最终解码图像 RGB 像素 | 0 | 逐像素一致 |

检查 `_step_index=next_index`，每步结束又检查其自增到 index+1，确保没有重复或漏步。以误差严格为零判定通过，没有放宽阈值。

## 成本与输出

| 阶段 | 秒 | 峰值 allocated MiB | 峰值 reserved MiB |
|---|---:|---:|---:|
| 模型加载 | 3.713 | 6732.70 | 6976 |
| baseline 51 步（含保存） | 6.373 | 7068.45 | 7540 |
| guided 51 步（50 inner forward/backward +10 final attention forward） | 13.489 | 11993.25 | 12352 |
| 磁盘加载和恢复 26 步 | 3.155 | 7078.24 | 12352 |
| 三张解码与图像输出 | 4.747 | 4175.56 | 7512 |

成功调用总耗时 34.752 秒（含初始化/记录/编码等）；首次启动失败的加载耗时约 6.60 秒，不包含在成功调用指标中。最大 allocated=11.712 GiB、reserved=12.0625 GiB，未 OOM。reserved 包含 PyTorch 沿用的 allocator 缓存，不能当成当前分支的活 tensor 大小。timing 均在阶段前后同步 CUDA，记录的是现场 `perf_counter`，没有 extrapolate 多样本成本。

核心图像：

- [baseline/guided 对照](../outputs/stage01_seed42_verified/baseline_guided_comparison.png)
- [baseline 目标框](../outputs/stage01_seed42_verified/baseline_with_target_box.png) / [guided 目标框](../outputs/stage01_seed42_verified/guided_with_target_box.png)
- [连续/恢复续采样对照](../outputs/stage01_seed42_verified/continuous_resumed_comparison.png)
- 原始图像 `baseline.png`、`guided.png`、`resumed.png` 位于同一目录。

机器可读验收、梯度、耗时、峰值显存和历史图比较见 [metrics.json](../outputs/stage01_seed42_verified/metrics.json)。

## 是否适合下一阶段

恢复与同初始状态对照的工程前提已通过，可提交设计/审查聊天评估是否进入**单例两物体关系试跑**。本轮 guidance 有可见空间变化但尚无精确框控制保证；下一阶段需要先定义旧关系合格条件、新旧约束几何兼容性和新约束修复成功判据。当前结果不证明关系退化、保持或保护方法有效，也不支持直接扩大 benchmark。按本轮授权范围停止，未自动执行下一阶段。
