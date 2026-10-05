# 任务 02：SDXL 两物体关系干预最小试跑

日期：2026-10-05 UTC。基准提交：`7bc3ae9e8d08076d3335c770c4ef8cdf224b863f`（04-01）。

**本轮未找到合格 baseline，按预注册停止。** 固定候选 `[42,123,2024]` 全部保留：42 出现两个苹果（用户已确认），123 的苹果在杯右侧，2024 按包含杯把的 bbox 中心规则未满足左侧间隔。没有生成 guided 图，没有执行新的 checkpoint 恢复续采样验收。新目标修复、旧关系保持/退化、联合成功均为**未检验**，不是失败的干预结果，也不沿用 task01 的恢复通过结论。

## 预注册与代码版本

开始时 `git status --short` 为空，HEAD 与指定基准一致。读取实际根 AGENTS.md、stage01 报告/实现和模型缓存；任务02 文件原先不存在。根 AGENTS.md 仅局部更新过期的阶段、授权范围和停止条件，其余环境/数据/Git 保护规则保留。

在生成 baseline 前写定 [protocol](stage02_protocol.md) 和 [配置](../configs/stage02.json)。首次运行将两者和三个实际代码文件的 SHA256 写入 [protocol_lock.json](../outputs/stage02/protocol_lock.json)，锁定时间为 `2026-10-05 09:00:46 UTC`。后续 baseline 与 annotation 入口检查锁定哈希；没有在观察图片后改 prompt、seed 顺序、窗口、框、强度、阈值或采样实现。

复用 `stage01_sdxl.py` 的 Sampler、完整 scheduler state pack/unpack、RNG 与比较函数，以及 `attention_processor.py`。这两个文件与基准提交一致，没有修改。stage01 历史 guided 图差异原因仍未确认；本轮没有混用那些图片作对照。

新增 [stage02_relations.py](../src/stage02_relations.py)，包含三种入口：

- `baseline --seed`：检查固定顺序，采样并保存第 5 次更新后的完整状态、第一恢复步及最终 latent 参照。
- `annotate`：将人工主体框转换成归一化 bbox、中心与 dx/dy，执行固定资格/关系判据，保存带框图和筛查记录。
- `intervene`：仅允许已有合格 baseline 后进入；两分支各自从磁盘独立加载完整历史与 conditioning/RNG，先严格验收无 guidance 的第一步/最终 latent/图像，再执行配置中的唯一 guidance 窗口。支持保存的 next_index 和配置指定的半开 guided index 区间。**该入口本轮未调用，新的续采样/干预路径没有 GPU 运行验证。**

新增四项 CPU 决策规则测试 [test_stage02_annotations.py](../tests/test_stage02_annotations.py)，分别检查阈值边界、新目标已满足时不能当修复 baseline、对象缺失算失败、多实例保留原差值但判 uncertain；全部通过。新入口通过 py_compile 与 --help。结束时检查 git diff、代码锁定哈希、标注与原图哈希/几何计算、checkpoint 内容和 Git 忽略状态。没有改 site-packages、下载、FLUX 运行、清理、pull、commit 或 push。

## 实际有效配置与文本位置

固定 prompt，三个候选完全相同：

```text
A red apple to the left of a blue ceramic cup, against a plain neutral background, studio photograph, both objects fully visible.
```

A=红苹果；B=蓝色陶瓷杯。旧目标 A 在 B 左侧；新目标 A 在 B 上方，二维几何兼容。预注册资格：两个对象唯一、清晰、未明显截断，`dx>=0.06` 且 `dy<=0`。对应关系名义成功阈值 0.05，明确失败 <=0.04，明确成功 >=0.06，中间开区间 uncertain。

| 项目 | 实际值 |
|---|---|
| 环境 | `/root/miniconda3/bin/python` 3.12.3；torch 2.3.0+cu121、diffusers 0.32.2、transformers 4.48.3、accelerate 1.3.0 |
| GPU/CUDA | RTX 4090，可见 24564 MiB；torch CUDA runtime 12.1 |
| 模型 | `stabilityai/stable-diffusion-xl-base-1.0`；显式本地 revision `462165984030d82259a11f4367a4eed129e94a7b` |
| 缓存 | `HF_HOME=/root/autodl-tmp/huggingface`，HF/Transformers offline=1，local_files_only=True |
| 精度/尺寸 | FP16，1024×1024；latent `[1,4,128,128]`；attention 32×32 |
| 调度 | LMS 51 步、order=4、leading spacing、steps_offset=1、epsilon prediction、非 Karras |
| CFG/batch | CFG=7，negative prompt=None；图像与 conditional guidance batch=1，原 CFG forward batch=2；分支顺序执行 |
| checkpoint | 第 5 次 scheduler.step 后，完成 index 4、t=875，next_index=5、next_t=856 |
| 固定干预（未执行） | apple attention，框 `(0.0,0.10,1.0,0.35)` 横跨全宽；仅 index 5–9，loss_scale=30、最多5 inner、E threshold=0.2、sigma²更新、相对更新上限0.1 |
| 保护限制 | 无 cup 目标、旧关系 penalty/projection、保持损失或 prompt above 改写 |

三个候选中，两套 tokenizer 的 apple span 均为 `[[3]]`，token ID=3055、token=`apple</w>`；未取并集、未优化 cup 或特殊 token。完整 ids、token strings、special IDs 和目标 span 均保存到各 seed 的 `tokenizer_diagnostic.json`。SDXL 两套文本 hidden state 在特征维拼接，相同序列位置定义与所选 apple 索引一致。

全部模型参数冻结，baseline 全程 no_grad；conditioning 只编码一次并保存。baseline 均使用同一个实现、精度和解码路径。解码前将 UNet/text encoders 移到 CPU，VAE 按 stage01 force_upcast 逻辑解码，decoded tensor 检查全部有限；processor 均恢复原对象，模型参数没有梯度。没有执行 guidance，因此本轮没有梯度/更新范数或 attention 改善数据，不能借用 task01 的数值填充。

## 人工标注与筛查结果

在原始 1024×1024 图像目视标注轴对齐可见主体框。包含清晰苹果茎和杯把，排除阴影/倒影；不依赖 attention centroid 或检测器，不推断被遮挡的未知轮廓。像素框转换为归一化框和中心保存；小数位来自整数像素除以1024，不表示亚像素标注精度。完整图片与标注供用户复核。

| seed | 像素 bbox：A / B（x0,y0,x1,y1） | dx | dy | 旧/新关系 | 资格与理由 |
|---|---|---:|---:|---|---|
| 42 | A1=(442,506,673,764)，A2=(615,592,817,786)；B=(211,556,452,785) | null | null | uncertain / uncertain | 两个真实苹果；A 非唯一，用户已回复“确认两个苹果”；排除 |
| 123 | A=(639,499,954,814)；B=(42,384,631,810) | -0.44921875 | -0.05810547 | 明确失败 / 明确失败 | 对象唯一清晰且无边界截断，但苹果位于杯右侧；排除 |
| 2024 | A=(334,466,650,794)；B=(263,358,737,752) | 0.00781250 | -0.07324219 | 明确失败 / 明确失败 | 包含左杯把后中心几乎对齐，dx 未达0.06；排除 |

seed42 不将某个苹果强行指定为唯一 A；单个 dx/dy 留 null，保留两个实例的框/中心及 [各实例原始关系差值](../outputs/stage02/seed42/instance_relation_alternatives.json)：A1 相对 B 的 dx=-0.220703125、dy=0.034667969；A2 的 dx=-0.375488281、dy=-0.018066406。多实例关系仍按 protocol 记 uncertain，不据其作成功/退化结论。用户的确认核实了实例数量，没有改变资格规则。

seed2024 杯子被前景苹果局部遮挡，但身份、可见外轮廓极值和杯把清楚；记录 `occluded_by_A=true`，没有图像边界截断。这里“左侧失败”仅指预注册含杯把 bbox 中心差，不等同于苹果完全不在杯身左侧；也不代表整体分离或遮挡关系。**未在看图后改为排除杯把以获得合格样本。** 123/2024 的差值均不处于 uncertain 区间；主体身份与极值判断较清晰，但 bbox 仍是人工标注，用户可复核。

全部候选 A/B 都存在，没有对象消失；42 多实例没有被删除。没有选中 seed，不运行第四个候选、不改 prompt、不改判据、不以 guided 成败筛 seed。

可复核输出：

- [三候选原图对照](../outputs/stage02/baseline_screening_original.png) / [A/B 框对照](../outputs/stage02/baseline_screening_annotated.png)
- [seed42 原图](../outputs/stage02/seed42/baseline.png) / [标注图](../outputs/stage02/seed42/baseline_annotated.png) / [标注表](../outputs/stage02/seed42/baseline_annotation.json)
- [seed123 原图](../outputs/stage02/seed123/baseline.png) / [标注图](../outputs/stage02/seed123/baseline_annotated.png) / [标注表](../outputs/stage02/seed123/baseline_annotation.json)
- [seed2024 原图](../outputs/stage02/seed2024/baseline.png) / [标注图](../outputs/stage02/seed2024/baseline_annotated.png) / [标注表](../outputs/stage02/seed2024/baseline_annotation.json)
- [筛查 JSON](../outputs/stage02/screening.json) / [筛查 CSV](../outputs/stage02/screening.csv) / [关系结果](../outputs/stage02/relation_results.json)

## 状态保存与未执行项

三个 baseline 各保存一个本地 `checkpoint.pt`，约1.71 MiB；这是同一预注册步位置的三个候选状态，不是三个不同位置的恢复实验。保存前 latent/初始 latent/四项 derivative 合计估算786432 bytes；文件还包含 conditioning、RNG、配置和小量状态。无模型权重、凭据或 full attention；这些 `.pt` 和 `reference_latents.pt` 都被 `.gitignore` 忽略。

实际 checkpoint 保存：当前/初始 latent；边界 `after scheduler.step` 与 next_index=5；scheduler 完整 runtime state（四项 FP16 derivatives、`_step_index=5`、`_begin_index`、timesteps、sigmas、betas/alphas/alphas_cumprod、num_inference_steps、scale_input标志等）；conditional/CFG embeddings、pooled embeddings、time_ids；CPU/CUDA/显式 generator/Python/NumPy RNG；processor 类型；运行配置、实际环境、锁定源码/协议哈希。独立 reference 文件只保存第一恢复步和最终 latent，供合格样本的真实续采样验收。

| 必要验收/干预项 | 本轮状态 |
|---|---|
| checkpoint 边界/历史的只读文件审计 | 已检查 next_index=5、历史数4、dtype/device及实际timestep；这只是保存内容审计 |
| 无干预恢复后的第一步误差 | **未运行，null** |
| 无干预恢复后的最终 latent/图像误差 | **未运行，null** |
| fixed guided branch / inner gradient、更新、attention标量 | **未运行，无数据** |
| 新目标是否实际修复 | **未检验，null** |
| 旧关系是否保持或退化 | **未检验，null** |
| 联合成功/关系退化候选 | **未检验，null** |

没有把文件可读或 task01 验收代替本次新 checkpoint 的实际恢复验收。也没有声称 checkpoint latent 已具有可见正确关系。由于资格门槛未通过，不能就干预的修复能力、关系退化或保持得出结论。

## 实际命令、成本和文件

在项目根目录按序执行三次 baseline（无 loop 自动生成后续候选，每次先完成前一个 baseline 的人工标注和资格判定）：

```bash
set -o pipefail
HF_HOME=/root/autodl-tmp/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/root/miniconda3/bin/python -u 04_relation_preservation/src/stage02_relations.py baseline --seed 42 \
  2>&1 | tee 04_relation_preservation/outputs/stage02_baseline42.log
HF_HOME=/root/autodl-tmp/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/root/miniconda3/bin/python -u 04_relation_preservation/src/stage02_relations.py baseline --seed 123 \
  2>&1 | tee 04_relation_preservation/outputs/stage02_baseline123.log
HF_HOME=/root/autodl-tmp/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/root/miniconda3/bin/python -u 04_relation_preservation/src/stage02_relations.py baseline --seed 2024 \
  2>&1 | tee 04_relation_preservation/outputs/stage02_baseline2024.log
```

每个 seed 生成后执行标注入口，例如42（123/2024分别替换 seed 和路径）：

```bash
/root/miniconda3/bin/python 04_relation_preservation/src/stage02_relations.py annotate \
  --kind baseline --seed 42 \
  --annotation 04_relation_preservation/outputs/stage02/seed42/baseline_annotation_input.json
```

人工输入 JSON 已保存；42 的多实例框由 Pillow 叠加到 A1/A2 标注图，summary/CSV/对照图同样由保存的标注与运行指标生成，不调用生成模型。所有生成调用的实际 argv、环境、有效 scheduler 配置/序列及初始 latent SHA256 位于每个 seed 的 `run_metrics.json`、`effective_config.json`。

| seed | 同步采样循环秒数 | 生成调用内部总秒数 | 峰值 allocated MiB | 峰值 reserved MiB |
|---|---:|---:|---:|---:|
| 42 | 6.28343 | 12.20419 | 7067.71 | 7586 |
| 123 | 6.31318 | 12.56200 | 7060.07 | 7532 |
| 2024 | 6.30357 | 12.39028 | 7062.23 | 7568 |
| 合计/全程峰值 | **18.90018** | **37.15647** | **7067.71** | **7586** |

同步采样循环包含 CPU scheduler/序列化和 CUDA 同步，为现场预算使用的 wall time，不是纯 kernel 时间。三次模型加载合计9.25574秒、解码合计2.26236秒，已包含在生成调用内部总秒数中；37.15647秒不包括 Python import、人工标注、等待用户回复、报告和检查。protocol锁定至summary记录的实际墙钟时间342.23151秒，包含标注/交互，并非GPU采样。峰值 allocated **6.9021 GiB**、reserved **7.4082 GiB**。没有OOM；采样时间远未接近570秒停止边界或600秒上限。每次出现 cuDNN不支持某 execution plan 的 warning，随后采样与有限性检查正常完成；没有安全停止或 traceback。

最终 [stage02_summary.json](../outputs/stage02/stage02_summary.json) 汇总全部成本、存在/唯一性/uncertain标记，以及未运行项的 null。本轮新增 config/protocol/入口/测试/报告与 `outputs/stage02/`；唯一已跟踪修改是根 AGENTS.md 的局部阶段更新，stage01及历史数据保持不变。

## 停止结论

本轮完成了固定三候选的可审查筛查；**没有得到适合“修复新目标”诊断的无干预合格样本**，所以未验证新关系干预管线的完整能力。负/无效结果全部保留，未为制造退化或救结果扩大搜索。下一步应由用户将本报告及三候选图像交回审查聊天决定；当前没有自动进入下一阶段，也不证明跨prompt退化或保护方法有效。
