# 任务03C：SDXL有限端到端效果验证

日期：2026-10-05。状态：完成，停止回传；没有自动进入保护方法、benchmark或FLUX运行。

## 1. 先回答效果问题

**G20修复了“苹果在蓝杯上方”，且按预先规定的目标对象中心判据保持了“苹果在蓝杯左侧”；G5未修复。** 这是固定P1/seed19上的单例几何联合成功，没有出现旧左右关系由明确成功变为明确失败。

但G20同时在原苹果位置生成了额外红白杯状物，并生成大量枝叶。场景共有两个杯状物，只有右侧主体是目标蓝色陶瓷杯。该结果的**场景内容保持失败**，不能称为干净的两物体控制成功、保护有效或跨样本有效。目标苹果与蓝杯在最终图中均清晰、完整且可辨认；跨时间的实例身份未被追踪。因此这里只确认最终目标类别/颜色对应物的中心关系，不宣称原苹果以身份保持的方式沿轨迹移动。

止损触发条件是G5/G20均未明确修复。本例G20明确修复，故该条件未触发，本轮没有将主线切到FLUX。完成这次有限试验后停止，由用户审查图像和上述质量限制；没有扩大实验范围。

[原图对照](../outputs/stage03c/U_G5_G20_original.png) · [bbox对照](../outputs/stage03c/U_G5_G20_annotated.png) · [机器可读summary](../outputs/stage03c/stage03c_summary.json)。

## 2. 输入与03B验收

本地HEAD为`92020a3c9c5ce35ecc96e24baee06ec8349d7872`；开始前git状态干净。读取了实际根AGENTS.md、03B报告/summary、确定性worker和确认结果、原Sampler/attention processor及历史02B配置和锁。GPU前重新在CPU加载03B原始trace：确定性A1/A2/A3与对应B1/B2/B3精确一致；原Sampler/D的5-inner guidance latent、一次step后latent及scheduler状态精确一致。没有重跑诊断矩阵。这些证据与实际开关均进入新锁。

旧02B checkpoint的文件哈希为`9ab65c0ea94b32d3111b414cab5d5e2bc5b95931591f2a9c87543afccea3da6d`，其嵌入指纹、after-step边界、next_index=5及旧配置逐项核验通过。初始latent直接取自该文件的已保存tensor，FP16形状`[1,4,128,128]`，tensor哈希：

```text
d314855341b482d81ae7ac11fd3854cb29d1ce342809e15f20f404b4fdf86b13
```

没有靠重设seed重画噪声。conditioning直接复用旧可信checkpoint内保存的tensor，negative_prompt=None不重新编码；旧环境加载验收保留，新增deterministic_algorithms=True的预期差异单独记录，未冒充历史非确定性环境。

原始入口/config/protocol首次GPU前锁定在`outputs/stage03c/protocol_lock.json`。唯一最小修复的新入口/config/protocol单独锁定在`outputs/stage03c/repair01/protocol_lock.json`。旧源码、锁、图片与tensor全部保留；根AGENTS仅局部更新当前阶段。CPU交付审计再次确认两套锁全部指纹未改变，旧输入与源码未改变。

## 3. 实际有效设置与实现

- SDXL `stabilityai/stable-diffusion-xl-base-1.0`，离线revision `462165984030d82259a11f4367a4eed129e94a7b`。
- P1/seed19；1024×1024，FP16，batch1；CFG推理内部conditional/unconditional批大小2；LMS51步/order4，CFG7，negative_prompt=None；模型参数全部冻结、无累计梯度。
- Prompt：`A red apple to the left of a blue ceramic cup, against a plain neutral background, studio photograph, both objects fully visible.`
- 全部正式进程从启动起设置`CUBLAS_WORKSPACE_CONFIG=:4096:8`；`cudnn.benchmark=False`、`cudnn.deterministic=True`、`torch.use_deterministic_algorithms(True, warn_only=False)`。
- 与03B保持一致：matmul TF32=False、cuDNN TF32=True；FP16/BF16 reduced_precision_reduction=True；flash/mem_efficient/math SDP开关均True；正式外层grad_enabled=True、autocast=False，采样step及只读诊断使用对应no_grad上下文。没有确定性算子静默降级。
- Python3.12.3（`/root/miniconda3/bin/python`），torch2.3.0+cu121，diffusers0.32.2，transformers4.48.3，accelerate1.3.0，CUDA12.1，RTX4090。复用`/root/autodl-tmp/huggingface`缓存，HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE及local_files_only开启；未安装、升级或修改site-packages。

各正式进程加载后核验四个模型实际权重内容哈希与03B相同：

| 模型 | SHA256 |
|---|---|
| UNet | `4f5380ff5c08fc0c32c2cb3a7a4251e4e2428fe4c8b9e0aa00fd385686e1135e` |
| text_encoder | `462aa8274a1dcbbe7477687f3cd17375ccda380a6b1ca71f1cb80f64184569cb` |
| text_encoder_2 | `1881faf704193c3b577c37925e41dee1b7e25e50332326576f4808c7146add5c` |
| VAE | `5e3c54629d4c485ab356341a8b1704eecca21fd5a3b3eaa446d02e72cb4a1053` |

新wrapper直接调用历史`Sampler.guidance`，保留原loss、FP16更新算术、sigma²、threshold检查与early-stop语义。仅G5/G20的inner上限5/20不同；窗口indices5–9、apple框`(0,.10,1,.35)`、mid/up层、loss_scale30、threshold.2、相对更新安全上限.1不变。没有cup或保护损失。

热图采用03B确认的反传后复制方案：conditional返回后仅暂存detach引用，`autograd.grad`完成后才转CPU并统计；最后更新后的图由原guidance已有的最终只读forward取得。仅index5/9的apple聚合32×32图留存，pending引用及时清空，不保存全层全head矩阵。没有另加后期forward、全程中间解码或GPU一致性探针。

## 4. 唯一失败与最小修复

原始prepare尝试在第一次采样step前终止：wrapper用原始scheduler config字典整体相等，错误地把`_use_default_values`的顺序当作数值身份。完整日志/traceback、失败metrics和旧锁保留在stage03c根输出。失败消耗12.837649秒，执行0采样step/0inner更新。

CPU复现确认唯一差异：历史metadata列表为`[use_exponential_sigmas, use_beta_sigmas]`，本次重建为反序。这是默认值来源成员列表，不是scheduler的数值参数。唯一修复为仅对该metadata列表排序后比较；其他键值仍全部精确比较，实际timestep/sigma序列另行精确验收。CPU局部复验确认反序metadata通过、真实beta参数变化仍拒绝。没有调整模型/环境、数值更新、身份哈希或恢复阈值。

新尝试另存`stage03c_sdxl_repair01.py`、config/protocol/lock和repair01输出；原入口及原锁未修改。修复依据见[repair_basis.json](../outputs/stage03c/repair_basis.json)，原/实际raw scheduler config均保存于修复prepare metrics。修复次数1，已用完允许额度。

## 5. U与R精确恢复

U用共享initial_latent完成51步；第5次scheduler.step后实际保存新checkpoint，next_index=5，4个LMS历史derivatives、内部step_index、conditioning、CPU/CUDA/generator RNG和新指纹全部保存。R独立从该磁盘文件加载，继续indices5–50共46步；scheduler及derivative存储没有共享可变引用。processor与新环境/指纹断言保留。

| 新设置内部的U/R验收 | 精确一致 | max绝对误差 |
|---|---|---:|
| 恢复首步latent | torch_equal=True | 0 |
| 最终latent | torch_equal=True | 0 |
| 最终RGB图 | 每像素相同 | 0 |

conditioning、scheduler状态、processor恢复与RNG验收均通过。CPU交付再次直接加载U reference与R final tensor，精确相等。恢复指标见[JSON](../outputs/stage03c/restoration_metrics.json)和[CSV](../outputs/stage03c/restoration_metrics.csv)。

新U经人工检查为合格起点：目标唯一清晰、无截断，dx=.377930≥.06，dy=-.033203≤0。新U恰好与历史02B baseline逐像素相同，所以复用历史人工bbox后重新目视核验；这只是实测历史参考，不是新设置对旧图的强制门槛。G5/G20不要求复现历史非确定性guided图片。U筛查记录在`repair01/baseline_selection_before_guidance.json`，先于guided分支执行。

## 6. 最终几何与对象质量

bbox像素坐标以1024原图为准，含苹果自身茎与杯把，排除阴影。G20苹果的短果柄计入，长支撑枝与叶作为额外对象，不计入果实bbox。目标A为red apple，B为blue ceramic cup；归一化中心定义dx=cx(B)-cx(A)、dy=cy(B)-cy(A)。差值≥.06明确成功、≤.04明确失败，中间uncertain；表中均远离阈值。

| 分支 | apple bbox(px) | blue cup bbox(px) | dx | dy | 左侧旧关系 | 上方新关系 | 中心联合成功 |
|---|---|---|---:|---:|---|---|---|
| U | (160,637,385,875) | (483,580,836,864) | .377930 | -.033203 | 明确成功 | 明确失败 | 否 |
| G5 | (159,644,383,872) | (483,580,836,865) | .379395 | -.034668 | 明确成功 | 明确失败 | 否 |
| G20 | (466,219,641,396) | (482,580,831,864) | .100586 | .404785 | 明确成功 | 明确成功 | 是，有额外对象 |

三张图的目标苹果及蓝杯均存活、目标类别/颜色唯一、身份可辨，主体无明显遮挡或截断。G5额外生成装有枝叶的透明瓶/花瓶。G20额外生成红白杯、灰蓝把手，bbox约(157,653,384,873)，以及长枝叶，枝条左边界有截断。G20杯类总数为2，目标蓝杯数为1，不能忽略额外杯来汇报场景纯净性。未测量跨时间的实例身份；类别/颜色标注不等价于实例跟踪。

相对U，G20最终苹果中心上移448.5px、右移281px，左侧间距显著缩小，但仍明确超过.06。此处为最终图bbox比较，不证明整个中间轨迹的身份保持。苹果并非整个bbox均处于(.10,.35)高度带，故也不宣布完整框内放置成功。标注仍待用户复核；原图、带框图、像素/归一化bbox、中心、存活/唯一/遮挡/截断、额外对象与关系字段均保留。机器结果见[relation_results.json](../outputs/stage03c/relation_results.json)和[relations.csv](../outputs/stage03c/relations.csv)。

## 7. Guidance与必要诊断

G5每步5次，共25次；G20每步20次，共100次。所有步均触及上限，没有在threshold=.2前early-stop；最后E仍大于.2。实际执行125次，没有额外inner反传。每inner tensor/标量有限，梯度非零，模型参数无累计梯度；相对更新远低于.1。

| 分支 | gradient norm范围 | update norm范围 | max相对更新 |
|---|---:|---:|---:|
| G5 | .043976–.137573 | 1.860352–3.000000 | .002470 |
| G20 | .039154–.200073 | 1.013672–5.324219 | .003588 |

| 分支/index | E前→最后更新后 | mid框内比例前→后 | up框内比例前→后 |
|---|---|---|---|
| G5/5 | .547205→.532122 | .283171→.284457 | .238051→.256870 |
| G5/9 | .485726→.449462 | .287027→.290699 | .319469→.370861 |
| G20/5 | .547205→.441595 | .283171→.293744 | .238051→.380006 |
| G20/9 | .335495→.288619 | .309260→.316741 | .559695→.667742 |

G5/G20在index5前5次的全部inner标量CSV行精确相同，是同一初始局部轨迹的额外交叉核验；这里只保存标量，不能把此项说成完整梯度tensor精确验收，完整tensor一致性来自先前03B确认。

8个raw npz仅含mid、up及mask，均有限。离散32×32 mask使用原floor/ceil规则，面积288/1024=.28125，连续目标框面积.25；没有改变mask。热图CSV含峰值、质心、空间熵和框比例，同一index/同一层在分支与前后阶段共用色标：[index5](../outputs/stage03c/heat_index5.png)、[index9](../outputs/stage03c/heat_index9.png)。raw文件在`repair01/G5/heat_*.npz`与`repair01/G20/heat_*.npz`。

G5的E下降却未改善最终上方关系，说明attention改善不能代替几何判定。G20的up框比例改善与最终上方苹果一致，但这仅是中间attention与最终bbox的跨时间诊断；瓶/枝叶/额外杯变化不能直接证明apple attention的因果归属。

## 8. 预算、命令与交付

累计GPU工作按从首次CUDA初始化前至进程完成的活动wall time计，包含模型加载/内容哈希、失败、计算、同步、CPU传输/落盘和解码。独立CPU准备/人工标注/最终绘图另计，不启动CUDA。

| 尝试 | GPU活动wall秒 | 采样/inner |
|---|---:|---|
| 原入口prepare失败 | 12.837649 | 0步/0inner |
| repair01 prepare：U+R | 34.725689 | 51+46步/0inner |
| repair01 guide：G5+G20 | 64.279018 | 46+46步/125inner |
| 总计 | **111.842356** | **4条正式轨迹，125inner，1次最小修复** |

峰值allocated=12035.684MiB（11.754GiB），reserved=12714MiB（12.416GiB）。分支时长U采样8.464秒、R采样7.157秒、G5采样含guidance14.778秒、G20采样含guidance30.852秒；各自解码约2.2–2.7秒。模型加载/内容哈希等纳入累计。未触及600秒/540秒新分支门槛；未运行FLUX、新seed、参数搜索或重复诊断矩阵。

实际GPU命令如下，均在项目根目录；第一条失败日志完整保留，后两条是唯一修复的新尝试：

```bash
set -o pipefail
CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HOME=/root/autodl-tmp/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /root/miniconda3/bin/python -u 04_relation_preservation/src/stage03c_sdxl.py prepare 2>&1 | tee 04_relation_preservation/outputs/stage03c/prepare.log
CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HOME=/root/autodl-tmp/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /root/miniconda3/bin/python -u 04_relation_preservation/src/stage03c_sdxl_repair01.py prepare 2>&1 | tee 04_relation_preservation/outputs/stage03c/repair01/prepare.log
CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HOME=/root/autodl-tmp/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /root/miniconda3/bin/python -u 04_relation_preservation/src/stage03c_sdxl_repair01.py guide 2>&1 | tee 04_relation_preservation/outputs/stage03c/repair01/guide.log
```

CPU preflight/局部scheduler复验记录见`cpu_checks.json`、`repair_basis.json`和两个prepare metrics。最终CPU核验/图表命令：

```bash
HF_HOME=/root/autodl-tmp/huggingface /root/miniconda3/bin/python 04_relation_preservation/src/stage03c_cpu_delivery.py
/root/miniconda3/bin/python -m py_compile 04_relation_preservation/src/stage03c_sdxl.py 04_relation_preservation/src/stage03c_sdxl_repair01.py 04_relation_preservation/src/stage03c_cpu_delivery.py
git diff --check
git status --short
```

主要执行文件哈希（完整输入/代码指纹以各锁为准）：

| 文件 | SHA256 |
|---|---|
| 原stage03c_sdxl.py | `db268026cb9d8f9f205e1b9902fefd49509fbe4811094bc39541020ffb4164b8` |
| repair01执行源码 | `4e5d5c09bcd00ac0196539e952a061d36ab9264e25fa2c506def7f56d3dc00e0` |
| repair01配置 | `ccbb3bf55f01fe0add7ba0282398fcac1dd1f0f36b95b9a7ad4b07bb22fe4000` |
| repair01 protocol | `f52aa9b7e0327adb8b887f6efe9a3b6567ad22f38c8452a62583ee80240049c5` |

修复锁时间`2026-10-05 13:04:21 UTC`，包含原失败锁与metrics的哈希。CPU交付代码及最新审计哈希见[artifact_audit.json](../outputs/stage03c/artifact_audit.json)。新大tensor全部留服务器并受Git忽略：U checkpoint/reference及R/G5/G20 final latent共5个pt，合计2,463,355bytes，未保存权重。旧历史tensor没有改动。

交付包括两套独立配置/protocol/锁/入口、CPU交付源码、本报告、summary、逐inner/step/heat CSV、恢复误差、三个原图与bbox对照、两幅热图面板及8个raw数组、运行命令/有效开关/指纹与失败完整记录。全部位于`04_relation_preservation/`；没有自动commit/push/pull。已只读查看FLUX current_state与Stage6E/7作为条件止损依据，但本轮止损条件未满足，因此未创建下一轮FLUX正式实验、未运行FLUX；历史左右交换也没有当成关系保持证据。
