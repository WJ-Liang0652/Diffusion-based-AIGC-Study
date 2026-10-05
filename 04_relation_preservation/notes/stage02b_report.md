# 任务02B：有限起点筛查与一次真实关系干预

日期：2026-10-05 UTC。基准提交 `11432e791f7eeb05248040f6d1c8024dbfbaab68`。本轮目的为准备诊断起点和验证管线，**不是正式统计实验**。04-02的三个失败候选/protocol/config/锁保留；那轮起点未获得不表示guidance失败或研究假设被否定。

**找到合格P1/seed19并完成真实恢复及唯一固定干预。** 仅尝试2个新baseline（P1/7、P1/19），遇首个合格者提前停止；P2及其余14个候选未生成，不能称完整16例评测或据此估计成功率。新checkpoint恢复第一步/最终latent最大绝对误差均为0，最终图像逐像素一致。guided的苹果仍在杯子左下方：新上方目标明确未修复，旧左侧关系保持。这是预设解释**A**，不能评价成功修复后的保持/保护，也不是退化证据。

## 固定配置、源码与环境

开始时工作区干净，HEAD与指定基准一致。先读实际AGENTS.md、04-02报告/protocol/config/入口。根AGENTS.md仅局部更新02B当前阶段和范围；新增 [stage02b.json](../configs/stage02b.json)、[stage02b_protocol.md](stage02b_protocol.md)、[stage02b_relations.py](../src/stage02b_relations.py)、[队列测试](../tests/test_stage02b_queue.py)。旧三个源码文件 `stage02_relations.py`、`stage01_sdxl.py`、`attention_processor.py` 原样复用，未修改；没有解除或重写旧锁，历史结果与默认旧入口行为不变。

新入口只管理P1/P2队列、独立路径、跨prompt的累计预算、标注汇总和A/B/C/D解释；采样、LMS恢复、conditioning/RNG、guidance损失/更新、tokenizer诊断、解码、中心关系规则均调用旧入口的函数。P1/P2结果分别放在新输出目录下；独立锁只管本轮。七项CPU测试通过（旧四项判据测试+新三项队列顺序/首个合格即停止/对象缺失不包装退化），py_compile与git diff检查通过。

第一次新生成前锁定时间 `2026-10-05 09:29:05 UTC`，新配置/protocol与四个实际源码SHA256见 [protocol_lock.json](../outputs/stage02b/protocol_lock.json)。生成及标注入口均核对这些哈希；所有baseline/恢复/干预使用相同锁定代码，没有观察guided后改配置。

| 项目 | 本轮实际值 |
|---|---|
| Python/包 | `/root/miniconda3/bin/python`，3.12.3；torch2.3.0+cu121、diffusers0.32.2、transformers4.48.3、accelerate1.3.0 |
| GPU/CUDA | RTX4090，可见24564MiB；torch runtime CUDA12.1 |
| 模型 | `stabilityai/stable-diffusion-xl-base-1.0`；显式本地revision `462165984030d82259a11f4367a4eed129e94a7b` |
| 缓存与访问 | `HF_HOME=/root/autodl-tmp/huggingface`；HF/Transformers offline=1、local_files_only=True |
| 采样 | FP16、1024×1024、LMS51步、order4、leading、offset1、epsilon、非Karras；CFG7、negative prompt=None |
| batch | 图像/conditional guidance=1，原CFG拼接forward=2；全部分支顺序执行 |
| checkpoint | 第5次scheduler.step之后，index4/t875已完成；next_index5、next_t856；余46次更新 |
| 干预 | 仅apple attention；全宽框 `(0.0,0.10,1.0,0.35)`；仅index5–9；loss_scale30、每步最多5inner、E threshold0.2、sigma²更新、相对更新上限0.1 |
| 损失/层 | 当前Sampler的mid/up两层平均平方框外占比E，通用sum聚合保持当前计算图；层路径与task01/02一致 |
| 保护项 | 无cup约束、旧关系penalty/projection、保持损失或新损失 |

两条实际baseline及两续采样分支均使用P1原文，guided没有额外above文字：

```text
A red apple to the left of a blue ceramic cup, against a plain neutral background, studio photograph, both objects fully visible.
```

P2按用户给定文字保存于配置，但没有运行。它是另一种旧关系表达，不能作为同prompt成功率改善或泛化证据。候选顺序每prompt均为 `[7,19,101,314,777,1001,2048,4096]`，先P1再P2，实际分母2，在P1第二个候选停止。

P1中两套tokenizer apple span均为 `[[3]]`，tokenID3055、`apple</w>`，不是cup/特殊token，未取并集；两baseline与干预均保存完整诊断。所选seed19初始latent SHA256及全部scheduler实际序列见 [effective_config.json](../outputs/stage02b/P1/seed19/effective_config.json)。条件编码只生成一次，续采样使用磁盘conditioning。参数冻结；text encoder/VAE不参与attention反传。三最终图使用同一force_upcast解码路径，均检查decoded tensor有限。没有下载、安装、修改site-packages、FLUX、小屋重跑、额外参数搜索、pull/commit/push。

## 筛查与最终几何

人工在1024原图标轴对齐可见主体bbox，包含清晰苹果茎和杯把，排除阴影/倒影；不以attention centroid代替主体。像素除以1024得到归一化坐标与中心；保留整数像素输入、confidence、存活/唯一性/身份/遮挡/截断和额外对象记录。小数精度来自计算，不表示人工亚像素精度。

固定 `dx=cx(B)-cx(A)`、`dy=cy(B)-cy(A)`；对应差值<=0.04明确失败、>=0.06明确成功、中间开区间uncertain，名义阈值0.05。baseline还需dx>=0.06、dy<=0、对象唯一清晰、未明显截断且无标注歧义。中心规则不说明整体分离/遮挡或物体整体入框，未在看图后改变杯把规则。

| 图像 | A像素bbox / B像素bbox | dx | dy | 判定 |
|---|---|---:|---:|---|
| P1/7 baseline | (313,607,542,712) / (598,625,912,840) | 0.31982422 | 0.07128906 | 可见苹果已高于杯中心，新目标已经成功；不适合修复起点。苹果下部在额外蓝碗中，遮挡和蓝碗记录，不推断不可见轮廓 |
| P1/19 baseline | (160,637,385,875) / (483,580,836,864) | 0.37792969 | -0.03320313 | 首个合格；旧关系明确成功，新关系明确失败，两对象完整、唯一清晰 |
| P1/19 guided | (159,644,382,872) / (483,580,836,865) | 0.37988281 | -0.03466797 | 旧关系明确成功，新关系明确失败；联合成功=false |

所选baseline中心A=(0.26611328,0.73828125)，B=(0.64404297,0.70507813)；guided中心A=(0.26416016,0.74023438)，B=(0.64404297,0.70556641)。所选对照两张图A/B均存活、unique/identity_clear=true、无可见遮挡或图像边界截断；uncertain=false。dy始终为负，远离正向关系uncertain区间；不把几像素的框差当成几何移动效果。人工框供用户复核，没有虚构不可见坐标。

guided出现了透明瓶与高枝叶背景，均不作为第二个苹果或蓝陶瓷杯。这里记录为可见额外内容变化；没有测量因果attention归属，不能声称apple attention一定在追逐这条枝叶。新目标未修复，因此即便旧关系仍正确，也不能把保持归因于保护有效，或评价“成功修复新约束时”的保持性能。

机器结果采用解释 `A_new_target_not_repaired`；无退化候选、无联合成功。不能由本例外推所有条件的修复能力、跨prompt退化或保护方法有效。所有拒绝/负结果保留，没有换seed救结果。

## 真实恢复验收与分支独立性

所选 [baseline参考](../outputs/stage02b/P1/seed19/baseline.png) 的checkpoint为本地 `P1/seed19/checkpoint.pt`，1,795,170 bytes（约1.71MiB），以及必要 `reference_latents.pt`；均忽略Git。seed7也保留同一步位的候选checkpoint。保存前latent/初始latent/四项历史估算786432bytes，其他为conditioning/RNG/config小量数据；不含权重/凭据/full attention。

snapshot保存当前/初始latent；边界after scheduler.step/next_index5；完整LMS runtime fields与四项FP16 derivatives（原CUDA device，落盘CPU独立clone并保留原device描述）；实际timesteps/sigmas/内部索引/scale_input标志及训练系数；conditional/CFG embeddings/pooled/time_ids；CPU/CUDA/显式generator及Python/NumPy RNG；processor类型、实际配置/环境/源码锁。

连续baseline主轨迹从保存处正常继续，不中断或重置；保存下一步及最终latent作参照。新进程真实从磁盘加载无干预状态，新的LMS实例恢复完整字段，没有重新set_timesteps跳过历史；conditioning和RNG恢复、原processor安装。实际续采样46次，并逐步检查内部索引无重复/漏步。**这是02B现场验收，不是引用task01通过。**

| 比较 | 最大绝对误差 | 完全一致 |
|---|---:|---|
| 第一恢复步：index5/t856的scheduler.step后 | 0.0 | torch.equal=true，dtype相同，均有限 |
| 最终latent | 0.0 | torch.equal=true，dtype相同，均有限 |
| 最终图像 | 0 | 逐像素一致 |

无干预图像通过后才允许guided：重新从同一个checkpoint独立磁盘加载，检查未干预latent一致，scheduler/derivatives列表及tensor指针独立、初始snapshot一致、conditioning/RNG/processor一致；不是复制已经干预latent。无干预与guided在同一新加载pipeline中顺序运行；无干预解码后将模型设备及VAE初始精度恢复，再执行guided，最后按相同解码路径输出。模型权重冻结无变化。该验收只对现场相同环境/配置成立，不声称跨硬件版本等价。

无干预最终图是有成本的离线反事实诊断，不能声称checkpoint latent当时已可见地满足旧关系，也不是零成本在线判断。

## Guidance代理、安全性与成本

实际inner次数25（index5–9各5），无early-stop减少、无数值安全停止。梯度均有限且非零，历史FP16范数0.04394531–0.13708496；更新范数1.85351563–2.99023438；最大相对更新0.00246239<0.1。模型参数没有累计梯度，processor正确恢复，最终latent/decoded tensor均有限。每inner的FP32梯度诊断、范数及标量保存CSV。

五个guided timestep的更新前后E都降低，mid/up框内占比都上升；这些是同timestep不同latent的内部代理，不等于最终上方关系成功：

| index | E首次→更新后最终 | mid比例首次→最终 | up比例首次→最终 |
|---|---|---|---|
| 5 | 0.54720473→0.53211796 | 0.28317109→0.28445727 | 0.23805137→0.25687513 |
| 9 | 0.48589349→0.44964978 | 0.28703293→0.29067841 | 0.31921738→0.37058568 |

不把跨timestep E趋势当因果效果，不把框内attention提高解释为几何修复。当前中途干预条件并非论文默认配置。

| 阶段 | 同步采样秒 | 调用内部总秒 | 峰值allocated MiB | 峰值reserved MiB |
|---|---:|---:|---:|---:|
| P1/7 baseline51步 | 6.31619 | 12.66772 | 7067.33 | 7528 |
| P1/19 baseline51步 | 6.31745 | 12.51453 | 7065.44 | 7594 |
| seed19恢复46步+guided46步（25inner） | 15.24064 | 26.26458 | 11896.37 | 12044 |
| 本轮合计/峰值 | **27.87429** | **51.44684** | **11896.37** | **12044** |

恢复采样5.76451秒、guided采样9.47614秒。全程峰值allocated **11.6176GiB**、reserved **11.7617GiB**，无OOM。每阶段前后同步CUDA；“采样秒”包含CPU scheduler与保存开销，不是纯GPU kernel时间。“调用内部总秒”包含模型加载/编码/设备转移/解码等，但不含Python import、人工标注/报告和用户交互；这些是本轮实测，不挪用旧指标。预算远低于570秒停止边界/600秒上限。出现cuDNN某execution plan不支持的warning，随后正常计算并严格恢复通过；无traceback或安全停止。

## 命令与交付

项目根目录依次运行两个baseline，每次观察/标注后才继续，非一次批量扫候选：

```bash
set -o pipefail
HF_HOME=/root/autodl-tmp/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/root/miniconda3/bin/python -u 04_relation_preservation/src/stage02b_relations.py baseline --prompt-id P1 --seed 7 \
  2>&1 | tee 04_relation_preservation/outputs/stage02b_P1_seed7.log
HF_HOME=/root/autodl-tmp/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/root/miniconda3/bin/python -u 04_relation_preservation/src/stage02b_relations.py baseline --prompt-id P1 --seed 19 \
  2>&1 | tee 04_relation_preservation/outputs/stage02b_P1_seed19.log
```

baseline标注入口（7同理替换路径/seed），以及合格后唯一干预：

```bash
/root/miniconda3/bin/python 04_relation_preservation/src/stage02b_relations.py annotate \
  --prompt-id P1 --seed 19 --kind baseline \
  --annotation 04_relation_preservation/outputs/stage02b/P1/seed19/baseline_annotation_input.json
HF_HOME=/root/autodl-tmp/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/root/miniconda3/bin/python -u 04_relation_preservation/src/stage02b_relations.py intervene \
  2>&1 | tee 04_relation_preservation/outputs/stage02b_intervene.log
/root/miniconda3/bin/python 04_relation_preservation/src/stage02b_relations.py annotate \
  --prompt-id P1 --seed 19 --kind guided \
  --annotation 04_relation_preservation/outputs/stage02b/P1/intervention/guided_annotation_input.json
/root/miniconda3/bin/python 04_relation_preservation/src/stage02b_relations.py summary
```

原图、人工输入与评估框表均保留；实际argv/环境/源码哈希/逐阶段成本见各run_metrics.json。主要交付：

- [筛查带框汇总](../outputs/stage02b/screening_annotated.png)、[筛查JSON](../outputs/stage02b/screening.json)、[筛查CSV](../outputs/stage02b/screening.csv)、[干预前选样记录](../outputs/stage02b/selected_baseline.json)
- [baseline/guided带框对照](../outputs/stage02b/baseline_guided_annotated.png)、[原图对照](../outputs/stage02b/baseline_guided_original.png)、[baseline/恢复对照](../outputs/stage02b/baseline_restored_comparison.png)
- [关系JSON](../outputs/stage02b/relation_results.json)、[关系CSV](../outputs/stage02b/relations.csv)、[恢复JSON](../outputs/stage02b/restoration_metrics.json)、[恢复CSV](../outputs/stage02b/restoration_metrics.csv)
- [每inner记录](../outputs/stage02b/P1/intervention/guided_inner_iteration_metrics.csv)、[每步attention代理](../outputs/stage02b/P1/intervention/guided_step_metrics.csv)、[梯度汇总](../outputs/stage02b/guidance_metrics.json)
- [完整本轮summary](../outputs/stage02b/stage02b_summary.json)、[实际干预运行指标](../outputs/stage02b/P1/intervention/run_metrics.json)

任务02B已完成并停止。成功获得起点及验证真实恢复/单次干预管线，但**这个固定条件没有修复新关系**。尚不能评估成功修复后的旧关系保护，也无纯关系退化候选、跨prompt退化或保护有效证据。所有结果交回审查，不再运行剩余候选、改强度/损失或自动扩大方法/benchmark；无commit/push。
