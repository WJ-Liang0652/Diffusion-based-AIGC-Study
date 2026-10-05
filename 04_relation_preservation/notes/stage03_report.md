# 任务03：inner预算对照（incomplete）

2026-10-05 UTC，基准提交 `16f70132e316949e866e839192a45f417d222b03`。

**G20是否修复“苹果在杯子上方”：本轮无法判定，G20未启动。** U恢复通过，但G5在index5插桩guidance后的latent精确一致门槛失败，因此按用户要求停止。状态为 **incomplete / comparability gate failure**，不能包装成G20未修复的负结果、联合成功或关系退化候选。没有放宽阈值、改参数或换seed补救。

本次实际完成一个46步无干预续采样U；原Sampler探针执行index5的5次inner及一次scheduler.step；插桩G5执行index5的5次inner及更新后诊断forward，在执行正式index5 scheduler.step前停止。总inner为10，完整续采样为1。G5没有最终图或最终bbox；G20没有生成分支目录。交付只包含真实已生成产物，状态对照图中的空白栏明确标记未完成/未运行，没有使用02B历史guided替代本轮结果。

## 身份、实现与CPU检查

开始时工作区干净，HEAD等于指定基准。读取实际根AGENTS.md、02B报告/config/protocol/锁、checkpoint有效配置、原Sampler/attention processor、状态恢复/解码代码及逐inner/step CSV。旧锁全部六项fingerprint及checkpoint内fingerprints字典一致；checkpoint/reference存在，revision、prompt、seed、框、分辨率、采样与guidance设置匹配。旧锁、配置、历史源码、图像和大tensor未修改。

根 [AGENTS.md](../../AGENTS.md) 局部更新到任务03。新增 [配置](../configs/stage03.json)、[protocol](stage03_protocol.md)、[GPU wrapper](../src/stage03_inner_budget.py) 及 [CPU交付脚本](../src/stage03_cpu_delivery.py)。GPU wrapper继承原Sampler：conditional直接调用super，随后仅detach并复制32×32聚合热图到CPU；guidance直接调用super.guidance，原loss/反传/FP16更新/early-stop均由原方法执行。未改共用源码或site-packages。CPU交付脚本在失败后只做数组/CSV核对、已完成图标注、画面板和整理JSON，不执行采样。

CPU preflight首次因新wrapper误用了旧有效配置不存在的 `negative_prompt` / `lms_step_order` 字段而报KeyError，GPU尚未运行。读取实际记录后改为 `lms_order=4`；negative prompt未单独记录，核对旧锁定encode代码明确使用None，新配置明确保存None。旧状态加载的全部断言保留，没有绕过环境/processor/conditioning/RNG检查。

CPU检查通过：旧身份、packed latent FP16 `[1,4,128,128]`、next_index5/after scheduler.step、四项LMS历史、参考latent有限、130inner预算计算；py_compile、既有七项CPU判据/队列测试、git diff --check均通过。失败后的CPU审计再次逐项核对stage03锁内源码与输入文件哈希全部未变。

首次GPU前锁定于 **2026-10-05 10:55:01 UTC**，见 [protocol_lock.json](../outputs/stage03/protocol_lock.json)。锁包括新配置/protocol/wrapper/AGENTS及旧源码、旧锁、checkpoint/reference、参考图和CSV哈希。采样wrapper SHA256为 `535eb293ee96655e8ebaec80b06c34e4a681466a28b2d931c472f09bff2ce7f7`，运行后原样保留；checkpoint SHA256为 `9ab65c0ea94b32d3111b414cab5d5e2bc5b95931591f2a9c87543afccea3da6d`。后加CPU交付脚本单独记录自身哈希于failure_diagnostic，未重写运行锁。

## 实际配置与验收

模型为离线 `stabilityai/stable-diffusion-xl-base-1.0`，revision `462165984030d82259a11f4367a4eed129e94a7b`；4090、Python3.12.3 `/root/miniconda3/bin/python`、torch2.3.0+cu121、diffusers0.32.2、transformers4.48.3、accelerate1.3.0、CUDA12.1。FP16、1024²、batch1（原CFG forward batch2）、LMS51步/order4/leading/offset1/epsilon/非Karras、CFG7、negative=None；模型全部冻结、无累计参数梯度，text encoder/VAE不参与attention反传。

固定P1/seed19：

```text
A red apple to the left of a blue ceramic cup, against a plain neutral background, studio photograph, both objects fully visible.
```

两个tokenizer的apple位置均为3，cup均为11。完整实际timesteps/sigmas/scheduler config、初始latent哈希及分支配置见各effective_config与run_metrics。checkpoint在第5次scheduler.step后，next_index5、next_t856；从磁盘独立恢复完整scheduler runtime/四项derivatives、conditioning、RNG和processor，不重新set_timesteps跳过历史。

计划唯一变量为G5→G20每步inner上限5→20；indices5–9、apple框 `(0,.10,1,.35)`、mid/up层、原平方框外比例E、loss_scale30、threshold.2、sigma²更新、相对上限.1不变。early-stop仍使用更新前E，无cup/保护loss。因门槛失败，实际没有开展预算效应比较。

| 门槛 | 实测 | 结论 |
|---|---|---|
| U首个恢复step后 vs reference.first | max_abs=0，torch.equal=true | 通过 |
| U最终latent vs reference.final | max_abs=0，torch.equal=true | 通过 |
| U最终图 vs 02B baseline | 最大像素差0，逐像素一致 | 通过 |
| index5原Sampler探针 vs 插桩G5，guidance后 | max_abs=0.015625，mean_abs=0.0000974392460193485，torch.equal=false；dtype同且有限 | **失败，立即停止** |
| 原Sampler vs 插桩正式step后 | 插桩未执行该step | 未完成 |
| G5全部旧关键指标及最终guided图复现 | 仅index5局部标量，未续采样至终点 | 未通过验收/未完成 |
| G20 | 未启动 | 不可评测 |

独立分支load_branch检查全部通过：scheduler/conditioning精确、CPU/CUDA/显式generator RNG精确、processor恢复。U与G5的起始latent/derivatives数据指针不同，没有共享可变LMS历史。只读diagnostic forward前后完整恢复scheduler和RNG，并断言latent值未变；U最终精确恢复也验证这些诊断没有改变其正式结果。

## 失败定位与证据边界

完整traceback见 [run.log](../outputs/stage03/run.log) 与 [run_metrics.json](../outputs/stage03/run_metrics.json)，触发于wrapper第294行guidance latent的torch_equal断言，无OOM/非有限/参数梯度/相对更新越界。原Sampler与插桩结果出现真实数值差异，不能以差值小为理由通过。

仅CPU进一步定位见 [failure_diagnostic.json](../outputs/stage03/failure_diagnostic.json)：U与G5 index5更新前mid/up **raw数组逐元素完全一致**，mask均相同。对照02B旧inner CSV，G5 inner0的E、mid/up框内比例、loss_scaled和FP16范数一致；最早保存的不同标量是第一次反向传播的FP32梯度范数：

- 02B：`0.04422464966773987`；stage03插桩：`0.04423284903168678`，差 `0.000008199363946914673`。
- inner1起框内比例/E等开始出现差异，最终index5 E为 `.5321099758148193`，旧值 `.5321179628372192`。

这把历史复验的最早**可观测**偏差定位到第一次反向传播梯度诊断，排除了该次已保存前向热图/框/能量不一致；不能证明哪个底层算子、完整梯度tensor或内存分配导致差异。运行时 `deterministic_algorithms=false`、`cudnn_benchmark=false`、TF32 matmul=false；这些标志本身不构成本次根因证据。没有擅自改变kernel/确定性开关或第三方环境再试。

原Sampler探针的guidance/step latent及每inner指标仅在进程内保存，失败后没有落盘；保留了与插桩guidance的误差摘要，但因此无法事后完整比较探针每一轮梯度，也无法确认探针是否逐项复现旧CSV。这是本wrapper的取证缺口。已耗尽本轮允许的额外5次inner探针；没有为补齐证据额外运行GPU。底层数值根因尚未确定，报告保留此限制。

## 已有诊断、几何及资源

保存9个raw npz：U indices5–9 apple各一份、U index50最后step前apple/cup各一份、G5 index5前/最后更新后各一份。均含32×32 mid/up和目标mask，CPU审计全部有限；mask采用原floor/ceil网格规则，为9×32=288格，面积比例 **.28125**，不同于连续框面积.25，未更改原空间映射。只保留两个选定层的聚合图，不保存完整全head矩阵。

峰值位置/数值、归一化质心、自然对数空间熵和框内比例记录于 [heat_metrics.csv](../outputs/stage03/heat_metrics.csv)。原每inner梯度/更新范数、finite、参数无梯度和每步E沿用原输出结构。G5 index5已执行5次inner：更新范数1.955078125–2.072265625范围（详见CSV），最大相对更新 `.001201629638671875`，低于.1，梯度均有限非零、参数无累计梯度。

G5局部E `.5472047328948975→.5321099758148193`，mid比例 `.2831710875034332→.2844581604003906`，up比例 `.2380513697862625→.2568850815296173`。这些是**未通过一致性门槛的部分轨迹代理**，不代表有效G5/G20最终物体上移。index5面板所有分支/前后在同layer用同一色标；U窗口面板也按layer统一色标，具体范围见heat_panel_scales.json。cup仅只读后期诊断，没有进入正式loss。

U原图已人工复看；因与用户审查的02B baseline逐像素一致，复用其可见主体bbox（含苹果茎/杯把、排除阴影），两者存活、唯一清晰、无明显遮挡或边界截断，无额外对象。A=(160,637,385,875)，B=(483,580,836,864)，归一化中心A=(.26611328125,.73828125)、B=(.64404296875,.705078125)，**dx=.3779296875、dy=-.033203125**；旧左侧关系明确成功、新上方关系明确失败，联合=false、uncertain=false。G5/G20无最终图，不作bbox/对象缺失判定，机器结果为not_evaluated、对象存活null。

| 实际阶段 | 同步计时秒 |
|---|---:|
| 模型加载 | 3.20935 |
| U46步+所有U热图诊断 | 6.69970 |
| U解码 | 3.01597 |
| 恢复模型设备/精度 | 1.31341 |
| 原Sampler index5探针（5inner+step） | 1.18297 |
| G5部分guidance+热图与失败验收 | .77895 |
| 全部状态加载/重加载 | .04195 |
| 上述GPU工作计时合计 | **16.24229** |

整个run进程内部（包括CUDA初始化/计时外状态开销）的时间为 **16.58684秒**，可作为本轮全部GPU活动的保守计费上界；不含Python import、首次CPU检查、后续CPU画图/报告等非GPU工作。allocated峰值 **11898.3223MiB（11.6195GiB）**、reserved **12048MiB（11.7656GiB）**。所有失败尝试与forward已计入，未因预算触限停止；停止原因是精确验收失败。出现与02B相同的cuDNN execution-plan不支持warning，后续正常执行；没有证据证明该warning是误差根因。

## 实际命令与交付

项目根目录CPU命令（preflight使用相同离线变量）：

```bash
HF_HOME=/root/autodl-tmp/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/root/miniconda3/bin/python 04_relation_preservation/src/stage03_inner_budget.py preflight
/root/miniconda3/bin/python -m py_compile 04_relation_preservation/src/stage03_inner_budget.py
/root/miniconda3/bin/python -m unittest discover -s 04_relation_preservation/tests -v
git diff --check
```

唯一GPU命令：

```bash
set -o pipefail
HF_HOME=/root/autodl-tmp/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/root/miniconda3/bin/python -u 04_relation_preservation/src/stage03_inner_budget.py run \
  2>&1 | tee 04_relation_preservation/outputs/stage03/run.log
```

失败后的CPU交付：

```bash
HF_HOME=/root/autodl-tmp/huggingface \
/root/miniconda3/bin/python 04_relation_preservation/src/stage03_cpu_delivery.py
```

主要文件：

- [stage03_summary.json](../outputs/stage03/stage03_summary.json)、[run_metrics.json](../outputs/stage03/run_metrics.json)、[失败CPU诊断](../outputs/stage03/failure_diagnostic.json)、[CPU身份检查](../outputs/stage03/cpu_checks.json)、[独立锁](../outputs/stage03/protocol_lock.json)。
- [U原图](../outputs/stage03/U/final.png)、[U框图](../outputs/stage03/U/final_annotated.png)、[分支状态/bbox对照](../outputs/stage03/branch_status_bbox_comparison.png)、[关系JSON](../outputs/stage03/relation_results.json)、[关系CSV](../outputs/stage03/relations.csv)。
- [G5部分inner CSV](../outputs/stage03/G5/inner_metrics.csv)、[G5部分step CSV](../outputs/stage03/G5/step_metrics.csv)、[U step CSV](../outputs/stage03/U/step_metrics.csv)、[聚合热图CSV](../outputs/stage03/heat_metrics.csv)。
- [index5热图面板](../outputs/stage03/heat_index5.png)、[U窗口面板](../outputs/stage03/heat_U_window.png)、[U后期apple/cup面板](../outputs/stage03/heat_U_late.png)、[共同色标记录](../outputs/stage03/heat_panel_scales.json)；各U/G5目录raw npz。

大latent文件U/latents.pt仅留服务器并由已有Git规则忽略。没有下载、安装、环境变更、FLUX、新seed、保护方法、benchmark、pull/commit/push。旧记录与锁保持原样；新运行锁也没有解除/重写。任务03本次尝试按失败门槛停止并交回审查，**尚无可回答增加inner预算是否修复几何关系的有效结果**。
