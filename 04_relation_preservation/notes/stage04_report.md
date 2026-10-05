# 任务04：独立起点的固定G20验证

日期：2026-10-05。基准：`d0bb664e4915b9ab82bee60cc374a2b3c1b06a5f`。状态：完成并停止；未调参、未开发保护、未运行FLUX。

## 1. 结论与实际数量

**03C seed19的G20几何修复未在本轮新增合格起点上重复。** 固定队列生成6个baseline，取得2个合格起点（2048、4096），名单在任何guided图出现前冻结；两例都完成精确R验收和一次固定G20。

- seed2048：目标苹果与蓝杯存活、唯一、清晰；苹果仍在蓝杯左下方，新目标未修复，旧左侧关系保持。没有新增显著对象，但尺度/形状/反光有变化。
- seed4096：红苹果缺失；出现多个蓝杯/容器、红杯和上层架。无法可靠指定唯一蓝杯，更不能为苹果虚构bbox。该例是**对象缺失、多实例/身份不确定及内容保持失败**，不属于纯几何关系退化候选。

新增明确修复0；修复且保持旧关系0；修复且旧关系明确失败0。可可靠比较目标中心关系的新增案例1个，未修复；另一个失效案例完整保留，没有删除后计算更高成功率。按既有止损授权，**停止当前SDXL实现调参，主线转入已有FLUX单例准备**；配置、起点要求及历史成本已整理，本轮FLUX GPU工作为0。

这是一轮**合格起点条件下的小样本探索**，实际生成/合格/完成数分别为6/2/2，不估计总体成功率，不宣称SDXL在其他条件下必然失效。本轮没有新目标修复后旧关系明确退化的证据，关系保护问题尚无退化证据，不自动开发保护方法。

[机器summary](../outputs/stage04/stage04_summary.json) · [全部筛查图](../outputs/stage04/screening_annotated.png) · [2048 bbox对照](../outputs/stage04/seed2048/U_G20_annotated.png) · [4096 bbox对照](../outputs/stage04/seed4096/U_G20_annotated.png)。

## 2. 配置、身份与执行边界

开始时本地HEAD与指定基准一致，git状态干净。读取实际AGENTS、03C报告/summary、`stage03c_repair01.json`和repair01实际源码，以及其原Sampler、attention processor、恢复/解码与03B依赖。旧03C锁的每个源码/配置/输入指纹重新核验通过。

独立新增`configs/stage04.json`、`notes/stage04_protocol.md`、`src/stage04_repeat.py`及`outputs/stage04/`。GPU前锁定实际执行入口、候选规则、配置/protocol及历史依赖；锁为`outputs/stage04/protocol_lock.json`，保留AGENTS_at_lock快照。采样代码及配置/协议在首次GPU后未修改。CPU制表/审计源码另行保存，未参与GPU计算。历史代码、锁、结果不改。

完整冻结03C G20：SDXL离线revision `462165984030d82259a11f4367a4eed129e94a7b`、FP16、1024×1024、batch1、CFG7/negative_prompt=None、LMS51步/order4。CFG推理内部conditional/unconditional批大小2，不是两个独立样本。Prompt保持：

```text
A red apple to the left of a blue ceramic cup, against a plain neutral background, studio photograph, both objects fully visible.
```

Guidance仅apple mid/up原loss，框`(0,.10,1,.35)`，loss_scale30、threshold.2、sigma²、relative上限.1；indices5–9，每步最多20inner，原pre-update E early-stop语义。直接复用03C `PostBackwardSampler`与原`Sampler.guidance/step`，没有G5、cup/保护loss或额外诊断轨迹。所有参数冻结、梯度仅针对latent，processors按原规则安装/恢复。

全部GPU进程从启动起保持03B/03C已验收的同一设置：CUBLAS_WORKSPACE_CONFIG=:4096:8，cudnn.benchmark=False，cudnn.deterministic=True，torch.use_deterministic_algorithms(True,warn_only=False)；matmul TF32=False、cuDNN TF32=True；FP16/BF16 reduced_precision_reduction=True；flash/mem_efficient/math SDP=True；外层grad_enabled=True、autocast=False，原step与只读forward使用原no_grad。没有更改kernel或运行开关。

实际环境：Python3.12.3 `/root/miniconda3/bin/python`、torch2.3.0+cu121、diffusers0.32.2、transformers4.48.3、accelerate1.3.0、CUDA12.1、RTX4090。HF_HOME=`/root/autodl-tmp/huggingface`，HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE及local_files_only开启；无安装、下载或site-packages修改。

每个GPU进程加载后都核对四个模型实际权重内容哈希与03C一致；完整值在config及每次run metrics中。conditioning直接复用可信03C U checkpoint的已保存tensor，因为prompt/negative/分辨率完全相同；每个新U checkpoint单独保存它们与自己的RNG/LMS状态。CPU审计逐个确认conditioning精确一致。

每seed只调用一次prepare_latents，使用独立CUDA generator产生一个initial_latent；hash写入effective_config/机器表，tensor保存在该seed的checkpoint。不是仅靠seed在R/G20中重画噪声。六个initial hash互不相同，全部有磁盘输入身份。没有重跑seed19，只读其历史数据和conditioning。

## 3. 固定筛查与名单冻结

规则：红苹果与主要蓝色陶瓷杯唯一、清晰、无明显主体截断，dx≥.06且dy≤0；模糊/多蓝杯不可合格。逐个按[101,314,777,1001,2048,4096]生成一次baseline并人工标注。六个用完只取得两个，未补seed或改prompt。

| 顺序/seed | dx | dy | 结论与理由 |
|---|---:|---:|---|
| 1 / 101 | -.124023 | .286133 | 拒绝：苹果在杯上，旧左侧失败，且新上方已满足；苹果底部受杯沿轻微遮挡 |
| 2 / 314 | 不可判 | 不可判 | 拒绝：可见器皿为大而浅且左侧截断的蓝碗，没有可靠目标杯；B bbox留空 |
| 3 / 777 | -.202637 | -.026855 | 拒绝：苹果在主要蓝杯右侧；左侧小棕色塞状物另记 |
| 4 / 1001 | -.338867 | -.020020 | 拒绝：苹果在主要蓝杯右侧 |
| 5 / 2048 | .311523 | -.025879 | 合格：目标唯一完整、旧左侧成立、新上方未满足 |
| 6 / 4096 | .241211 | -.043945 | 合格：目标唯一完整、旧左侧成立、新上方未满足 |

每个候选原图、人工bbox/中心、对象数量、额外对象和拒绝理由均保留在`seed*/U/`及[screening.json](../outputs/stage04/screening.json)/[CSV](../outputs/stage04/screening.csv)。314没有虚构杯坐标；其目标杯缺失/器皿类别问题与单纯关系失败区分。

冻结名单为**[2048,4096]**，时间2026-10-05 13:52:40 UTC。`selection_lock.json`同时保存全部六个候选的图像、标注、effective config、checkpoint/reference及筛查文件哈希，记录frozen_before_any_guided_image=True。所有R/G20都验收此锁及输入哈希；CPU交付再次核验冻结文件未变，没有根据guided效果重选或删除案例。

## 4. 逐例磁盘恢复

每个U连续51步；第5次scheduler.step后保存next_index=5、4个LMS derivatives、实际内部步索引、conditioning、CPU/CUDA/generator RNG、processor身份和配置/代码指纹。R从磁盘完整恢复后续46步。通过首步/final/pixel精确门槛后，G20再次独立从同一磁盘状态恢复；scheduler、derivatives和latent存储不共享可变引用，没有set_timesteps跳步替代LMS历史。

| seed | R首步latent | R最终latent | R最终图 | max误差 |
|---|---|---|---|---:|
| 2048 | torch_equal=True | torch_equal=True | 每像素一致 | 0 / 0 / 0 |
| 4096 | torch_equal=True | torch_equal=True | 每像素一致 | 0 / 0 / 0 |

scheduler/conditioning/processor及CPU/CUDA/generator RNG验收均通过。逐例`restoration_metrics.json`和汇总[CSV](../outputs/stage04/restoration_metrics.csv)保留所有检查；CPU再次直接加载R/参考tensor与RGB核验。无恢复、身份或数值安全失败，无修复重试、kernel搜索或放宽精确门槛。

## 5. 新增结果与内容

标注基于1024原图：含苹果自身茎、杯把，排除阴影；长枝叶单列。A红苹果、B主要蓝杯；dx=cx(B)-cx(A)，dy=cy(B)-cy(A)，归一化，≥.06成功、≤.04失败、中间uncertain。不改成整体左右分离，不按期望结果挑杯子。

| seed/分支 | A bbox(px) | B bbox(px) | dx | dy | 旧左侧 | 新上方 | 中心联合 |
|---|---|---|---:|---:|---|---|---|
| 2048 U | (119,649,348,880) | (353,599,752,877) | .311523 | -.025879 | 成功 | 失败 | 否 |
| 2048 G20 | (145,675,348,880) | (352,596,750,874) | .297363 | -.041504 | 成功 | 失败 | 否 |
| 4096 U | (402,725,593,933) | (583,643,906,925) | .241211 | -.043945 | 成功 | 失败 | 否 |
| 4096 G20 | 苹果缺失，空 | 多蓝杯身份不清，空 | 不可判 | 不可判 | 对象失效/不可评 | 对象失效/不可评 | 否 |

2048的delta_dx=-.014160、delta_dy=-.015625；苹果中心由(233.5,764.5)变为(246.5,777.5)，向下13px，而非上移。两个目标仍清晰、唯一、无遮挡/截断。旧关系保持不能归因于保护，因为没有保护且新关系未修复。

4096未给不存在的苹果或任意蓝杯指定目标bbox；机器对象字段记录A present=False、B unique/identity_clear=False、geometry_evaluable=False。共享历史评测器在missing优先时关系标签为failure_object_missing；这不是“旧几何关系明确失败”。B身份及杯数不确定另行记录，不能用额外杯/架变化当作关系退化证据。

| seed/分支 | 红苹果数 | 蓝杯候选数 | 全部杯类数量 | 新增内容/对象 | 内容保持 |
|---|---:|---:|---|---|---|
| 2048 U/G20 | 1 / 1 | 1 / 1 | 1 / 1 | 无新增显著对象；苹果稍缩小、茎角度/杯反光形状变化 | 类别/数量/显著内容保持；非像素或实例保持 |
| 4096 U | 1 | 1 | 1 | 无额外显著对象 | baseline参考 |
| 4096 G20 | 0 | 2个可辨蓝色杯状器皿，唯一主要目标不明 | 至少3；另1圆蓝容器杯/碗类别不明，记录范围3–4 | 红杯、蓝色器皿/容器、上层架；右上蓝杯截断 | 失败 |

4096下方红杯和浅蓝杯的可见bbox仅作为额外对象框记录（约395,714,614,942与594,644,815,955），没有把它们改名为目标苹果或唯一目标B。右上有截断深蓝杯，另有圆蓝碗/容器；全部保留，杯类精确数量留uncertain。非排他诊断为苹果缺失1例、多蓝杯/目标身份不清1例、杯类数量uncertain1例、内容失败1例，均指同一个4096案例。

以上内容保持为人工显著对象/类别/数量/场景判断，不能声称外观逐像素保持。类别匹配也不证明跨时间实例身份。人工标注边界有少量像素误差但可判案例远离成功阈值；缺失/身份不明没有虚构坐标，最终标注待用户审查。

全部原图和bbox对照分别在`seed2048/`与`seed4096/`；[relation_results.json](../outputs/stage04/relation_results.json)、[对象/关系CSV](../outputs/stage04/relation_objects.csv)、[全部标注](../outputs/stage04/all_annotations.json)包含存活/唯一性/遮挡/截断、数量、新增对象、dx/dy变化及内容字段。

## 6. 原指标与有限热图

两个G20均在indices5–9各执行20次inner，共各100、总200。全部有限、梯度非零、update已执行、模型参数无累计梯度，relative<.1。所有step最终E仍>.2，没有达到early-stop门槛。未增加反传/完整诊断轨迹。

| seed | gradient norm范围 | update norm范围 | max relative |
|---|---:|---:|---:|
| 2048 | .034821–.363770 | 1.548828–7.937500 | .006569 |
| 4096 | .034851–.486084 | .995605–12.578125 | .009575 |

| seed/index | E前→后 | mid框比例前→后 | up框比例前→后 |
|---|---|---|---|
| 2048/5 | .542605→.483453 | .276804→.283805 | .250202→.326226 |
| 2048/9 | .366358→.314156 | .292189→.302079 | .518627→.624209 |
| 4096/5 | .554595→.476238 | .283535→.295614 | .228075→.324487 |
| 4096/9 | .340181→.273499 | .305637→.322967 | .554779→.702303 |

原每inner/step CSV保留在各G20目录并在根输出汇总。只记录indices5/9的apple mid/up更新前后raw图，每例4个npz，总8个，仅含聚合图/mask；detach引用在反传后才CPU复制，及时释放旧图。离散32×32 mask面积288/1024=.28125（原floor/ceil），连续框面积.25。峰值/质心/熵等沿原wrapper保留，没有全层全head矩阵。

[index5热图](../outputs/stage04/heat_index5.png)、[index9热图](../outputs/stage04/heat_index9.png)在同一index/层对两个seed和前后阶段共用色标，scale另存JSON。attention-in-box显著改善没有转化为这两例的可确认上方苹果；其热图与最终bbox只能作跨时间诊断，不能把上层器皿或背景变化直接解释为attention因果归属。

## 7. 03C历史另列

03C seed19：dx=.100586、dy=.404785，新旧中心关系均明确成功，同时额外红白杯/枝叶、内容保持失败。本轮没有再次生成/干预seed19；该历史案例不计入上述新增6/2/2或任何新增分母。它证明既有实现曾在一个固定案例产生几何修复，不消除本轮重复性与对象/内容问题。

## 8. 预算、实际命令与交付

累计GPU活动wall time包含各进程模型加载/权重哈希、采样/反传、状态保存/CPU转移、解码及所有尝试；独立人工标注/CPU审计绘图不启动CUDA。各进程记录previous累计并由CPU按真实队列复核，预算没有重置。

| GPU进程 | 活动秒 | 轨迹/inner |
|---|---:|---|
| U seed101 | 25.257673 | 1 / 0 |
| U seed314 | 25.453718 | 1 / 0 |
| U seed777 | 24.263977 | 1 / 0 |
| U seed1001 | 25.151644 | 1 / 0 |
| U seed2048 | 24.857489 | 1 / 0 |
| U seed4096 | 25.415853 | 1 / 0 |
| R+G20 seed2048 | 60.517104 | 2 / 100 |
| R+G20 seed4096 | 62.249346 | 2 / 100 |
| 合计 | **273.166805** | **10条完成轨迹 / 200inner** |

峰值allocated=11945.778MiB（约11.666GiB）、reserved=12068MiB（约11.785GiB）。2048的R/G20采样8.554/32.927秒，4096为8.088/34.277秒，各解码约2.5–2.7秒。配置上限为6U+3R+3G20、12轨迹/300inner/600秒，540秒后不启动新分支；未触限，完整运行，无失败/修复尝试。日志中cuDNN execution-plan候选不支持警告未导致分支失败，所有已记录确定性开关保持且两例R精确通过。

实际GPU命令按此顺序执行；每个baseline命令后人工标注，最后freeze，再运行两个intervene。下列统一环境对应各次实际argv/log，没有以循环预先生成后续候选：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
set -o pipefail
/root/miniconda3/bin/python -u 04_relation_preservation/src/stage04_repeat.py baseline --seed 101 2>&1 | tee 04_relation_preservation/outputs/stage04/baseline_seed101.log
/root/miniconda3/bin/python -u 04_relation_preservation/src/stage04_repeat.py baseline --seed 314 2>&1 | tee 04_relation_preservation/outputs/stage04/baseline_seed314.log
/root/miniconda3/bin/python -u 04_relation_preservation/src/stage04_repeat.py baseline --seed 777 2>&1 | tee 04_relation_preservation/outputs/stage04/baseline_seed777.log
/root/miniconda3/bin/python -u 04_relation_preservation/src/stage04_repeat.py baseline --seed 1001 2>&1 | tee 04_relation_preservation/outputs/stage04/baseline_seed1001.log
/root/miniconda3/bin/python -u 04_relation_preservation/src/stage04_repeat.py baseline --seed 2048 2>&1 | tee 04_relation_preservation/outputs/stage04/baseline_seed2048.log
/root/miniconda3/bin/python -u 04_relation_preservation/src/stage04_repeat.py baseline --seed 4096 2>&1 | tee 04_relation_preservation/outputs/stage04/baseline_seed4096.log
/root/miniconda3/bin/python -u 04_relation_preservation/src/stage04_repeat.py intervene --seed 2048 2>&1 | tee 04_relation_preservation/outputs/stage04/intervene_seed2048.log
/root/miniconda3/bin/python -u 04_relation_preservation/src/stage04_repeat.py intervene --seed 4096 2>&1 | tee 04_relation_preservation/outputs/stage04/intervene_seed4096.log
```

CPU入口实际使用`preflight`、`annotate --seed N --kind U/G20 --annotation outputs/stage04/manual_seedN_KIND.json`、`freeze`。完整人工输入文件保留；GPU前的preflight未初始化CUDA。最终CPU命令为`HF_HOME=/root/autodl-tmp/huggingface /root/miniconda3/bin/python 04_relation_preservation/src/stage04_cpu_delivery.py`，以及两个新源码的py_compile、git diff --check、git status --short。不存在外部安装/下载或自动Git提交。

| 冻结文件 | SHA256 |
|---|---|
| stage04_repeat.py | `dfc1c6c9302321770da072ce1c74ff2d04d4c5609ea784626e0e96f5fd27092a` |
| stage04.json | `3088ac7b2ec7ba6a76ab16ff11ffb152705bb4611f819cfd51ca4072629f89dd` |
| stage04_protocol.md | `89a215deab0b3a6c7da209856b7b2b39975e90bd9dcf93a21a9562e7ab5fdf2e` |
| protocol_lock.json | `c69724ac12c11ce9c55d02a7af90dd32297453dc1307a9954e4020131f5439ff` |

全部输入/权重/执行依赖哈希在锁和每次run metrics，六个初始tensor与checkpoint哈希在[initial_latent_identities.json](../outputs/stage04/initial_latent_identities.json)。16个pt合计13,197,398bytes，全部本地且Git忽略；没有保存模型权重。无G5、旧seed19重跑或额外诊断矩阵。根AGENTS在GPU前局部更新任务04，GPU后仅补写完成数量与按授权FLUX准备的止损决定；原锁快照保留不动。

[artifact_audit.json](../outputs/stage04/artifact_audit.json)汇总锁/输入保留、初始状态、条件一致性、精确恢复、预算及本地tensor检查。交付配置/protocol/锁/源码、全部筛查/冻结名单、逐例U/G20原图及带框对照、对象/关系表、恢复误差、原CSV、少量热图/raw及运行资源；没有自动pull/commit/push。

## 9. FLUX止损准备与停止

已只读核对FLUX current_state、Stage6E/7及native objective/update实现，整理[下一轮单例准备](stage04_flux_preparation.md)和[准备配置](../configs/stage04_flux_single_preparation.json)，历史输入/源码另存manifest。候选为已有FLUX seed42保存的initial latent与历史baseline，图像初审符合dx≥.06/dy≤0；下一轮仍须输入/完整状态/实际运行一致性验收。单例方案沿已有BF16/768/FlowMatch50/guidance3.5/offload，block18/indices0–9/K≤5/eta24.154335/threshold.2，apple单目标有限框上移、cup不参与loss。不直接调用旧交换实验main或覆盖其输出。

历史baseline约213秒，Stage6E guided431秒；Stage7 guided约387–733秒。一次replay+guided可能约600–950秒加加载/检查，不能接到本轮剩余预算，也不能保证apple-only新任务成本。下一轮需明确冻结预算/入口和运行开关，复用缓存，不重新搭环境。历史左右交换不作为关系保持证据；本轮FLUX运行数0。任务04完成后停止回传，不继续SDXL搜索、不自动跑FLUX批量或保护方法。
