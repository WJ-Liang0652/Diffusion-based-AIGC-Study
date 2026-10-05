# AGENTS.md

更新日期：2026-10-05。适用于 `/root/autodl-tmp/diffusion-reproduction`。

## 1. 当前研究主线

当前用户已决定：第一阶段先使用 SDXL，复用现有 4090 服务器和已有 layout guidance 实现，验证以下候选问题：

> 在生成过程中，修复一个尚未满足、且与已有目标兼容的布局约束时，是否会破坏已有正确的二维空间关系？如何在修复新约束的同时减少这种退化？

先证明可重复的失败现象，再评估简单保护基线；局部扰动稳定性是后续候选风险信号，只有证明其增量价值后才进入方法设计。当前不强制使用 attractor、memory 或 free-energy 理论解释。

此前“FLUX 优先、guidance strength → trajectory deviation → background disturbance”的任务作为历史探索保留。它不是当前第一阶段的执行计划。背景像素差异和轨迹偏移可以作为辅助诊断，不能代替新目标成功、旧关系保持和联合成功的评测。

本次 SDXL 主线已获得用户授权，不需要再次确认是否切换。后续新的用户指令优先于本文件。

## 2. 已有基础与目录

主要目录：

```text
01_diffusers_basics/        基础实验
02_layout_control/         SDXL 历史实现与结果，作为当前起点
03_flux_layout_control/    FLUX 历史实现、汇报测试与结果
04_relation_preservation/  当前新实验，按需创建
```

保留旧代码和旧结果，不直接覆盖。FLUX 目录中未提交的汇报测试先保留，除非影响当前任务，不要求清理或提交。

优先阅读 SDXL 的：

- `02_layout_control/src/18_paper_backward_guidance_full_generation.py`
- `02_layout_control/src/19_lms_scheduler_alignment_full_generation.py`
- 对应输出、指标和配置。

现有实现可作为开发基础，但“能出图”或“attention loss 下降”不等于完成方法级复现，也不等于最终物体几何控制成功。以现场代码与实际结果为准。

## 3. 模型与服务器环境

第一阶段模型：`stabilityai/stable-diffusion-xl-base-1.0`。FLUX 留作后续 backbone 验证，当前不主动运行 FLUX 批量实验。

2026-10-05 已检查环境：

```text
项目: /root/autodl-tmp/diffusion-reproduction
GPU: RTX 4090 24GB（约 23.5GiB）
CPU: 12 核
RAM: 90GB
数据盘: /root/autodl-tmp，150GB
Python: 3.12.3，/root/miniconda3/bin/python
torch: 2.3.0+cu121
diffusers: 0.32.2
transformers: 4.48.3
accelerate: 1.3.0
CUDA: 12.1
HF_HOME: /root/autodl-tmp/huggingface
```

优先复用当前环境和模型缓存，不新建近似环境、不重装 Codex、不重复下载模型、不随意升级或降级核心包。不修改 `site-packages`，使用项目内 processor、wrapper 或必要子类。

SDXL 先复用已可运行的 FP16、分辨率和采样配置，不将 FLUX 的 BF16/offload 配置直接套到 SDXL。batch=1，分支顺序执行；模型参数冻结，仅保留 guidance 所需 latent/state 梯度。text encoder 与 VAE 不参与本轮 attention guidance 的反向传播。

出现 OOM 时先检查计算图、attention 保存、旧引用与分支并发，再考虑降低开发分辨率。旧脚本有 32×32 attention 网格假设，修改分辨率时必须同步检查空间映射。不要因局部报错重装环境。

## 4. 当前唯一执行阶段：任务 01

本轮只做一个 prompt、一个 seed、一个 scheduler、一个恢复 checkpoint。

顺序：

```text
读取实际代码与 git 状态
→ 复查一个旧 SDXL baseline/guidance 案例
→ 中间状态保存与恢复
→ 连续续采样与恢复续采样一致性验收
→ 汇报并停止
```

本轮不做 strength sweep、多 seed 批量实验、大型检测器下载或多分支稳定性探测，不自动进入下一阶段。

新入口应参数化 prompt、seed、目标词/框、分辨率、采样器和输出路径。旧脚本存在顶层执行逻辑时，不通过 import/runpy 意外运行历史实验或覆盖旧输出。

## 5. 状态恢复验收

保存必要的 latent、采样步位置、scheduler 状态、conditioning、配置和随机数状态。明确 checkpoint 位于 scheduler.step 前还是后，避免重复或跳过一步。

LMS 是多步方法，需要保留历史 derivatives、内部步索引等实际必要状态。只存 latent/timestep，或重新 set_timesteps 后直接跳到中间位置，不足以保证正确恢复。

每个分支使用独立状态副本，禁止共享可变 scheduler 历史。确认 attention processor 的正确安装与恢复。

比较：

- 不中断的无 guidance 续采样；
- 保存到磁盘、加载、从相同位置恢复的无 guidance 续采样。

报告恢复后第一步与最终 latent 的最大绝对误差、是否完全一致，以及最终图像对照。优先追求完全一致；若有差异，定位原因并报告，不随意放宽阈值宣布通过。

## 6. 后续关系实验的原则

仅在任务 01 完成、结果审查后进入两物体关系试跑，再逐步扩大。

新旧约束必须几何兼容，且新目标确实需要修复。各方法使用同一批干预前合格样本；同时报告新目标成功、旧关系保持、联合成功、对象存活/消失、检测不确定和计算成本。

区分：

- 当前可访问的 attention/解码代理；
- 从中间状态无干预续采样的最终结果；
- 从同一状态干预续采样的最终结果。

无干预完整续采样属于有成本的离线诊断，不能当作零成本在线判断。attention-in-box 提升不等于最终关系成功；对象消失不能删除后再计算更高保持率。小样本可先人工复核。

如果 guidance 没有修复新目标，不能把旧关系保持归因于保护有效。比较保护方法时需要考虑相近新目标成功率和等计算预算。

## 7. 同条件可比性

baseline、guided 和其他干预分支应保持同一模型、prompt、初始 latent/noise、conditioning、scheduler、timestep 序列、步数、分辨率、精度和解码过程，除明确实验变量外不混入其他改变。

仅使用同一 seed 不足以保证相同初始状态；应明确共享初始 latent 或等价随机状态。guided 与 baseline 的 scheduler 历史必须各自独立维护。

## 8. 数据与实验记录

新结果写入 `04_relation_preservation/`，不覆盖旧 outputs。不保存全 timestep × block × head 的完整 attention，不在 checkpoint 中保存模型权重或账号凭据。

仅保存所需 checkpoint 和必要图像/标量，先估算较大 tensor 的磁盘占用；大 checkpoint、模型缓存和凭据保持本地，不加入 Git。

重要实验记录：

- 实际代码版本、未提交状态与运行命令；
- 环境版本、模型、精度、分辨率、seed、prompt、初始状态；
- scheduler、步数、目标词/框、层、干预步和 inner loop；
- 有限性、梯度范数、更新范数、模型参数梯度检查；
- 各比较项、恢复误差、图像路径、运行时间和峰值显存。

记录运行时实际有效配置，不直接照抄可能被后续常量覆盖的脚本开头配置。

## 9. 自主执行范围

当前已授权：读代码、git status/diff、只读环境检查、在新目录增加普通实验文件和日志、修复局部 Python/tensor/状态恢复问题，以及本轮限定的单样本试跑和一致性验收。

这些普通步骤可连续完成，无需逐步确认。遇到失败先读完整 traceback，定位最小根因，再做最小修改和复验。

当前不授权自动扩大为高成本 sweep/benchmark、切换新研究路线、下载大型模型/依赖、改变核心环境、修改第三方安装包、删除历史数据或破坏性 Git 操作。确有必要时先说明具体原因、范围和成本，并根据已有用户授权判断是否需要补充确认。

## 10. Git 与历史修改

修改前后检查 git status/diff，保留用户未提交工作；不执行未经授权的 `git reset --hard`、`git clean -fd` 或自动 git pull。

本轮不自动 commit/push。需要代码审查时，仅针对本轮代码、配置和报告按用户指令提交，不用 `git add .` 混入历史测试、大 tensor、缓存或凭据。

## 11. 每轮交付与停止条件

脚本成功退出不等于实验成功。汇报实际问题、改动文件、运行命令、有效配置、输出路径、梯度与可见几何效果、恢复误差、耗时和峰值显存、失败原因和可信性。

任务 01 建议交付 `04_relation_preservation/notes/stage01_report.md`、少量对照图和可读取的指标记录；按需创建文件，不创建大量空目录。

如果旧 guidance 没有明显几何控制效果，保留并报告事实，仍可独立完成恢复验收，不通过挑 seed 或大规模救参掩盖问题。任务 01 完成后停止汇报，由用户将结果交回设计/审查聊天，再确定下一阶段。

## 12. 当前优先级

```text
正确实验问题与设计
> 状态恢复正确、同条件可比
> 可审查的证据与可信结果
> 最小试跑
> 多样本与定量验证
> 方法改进
> 工程美观
```
