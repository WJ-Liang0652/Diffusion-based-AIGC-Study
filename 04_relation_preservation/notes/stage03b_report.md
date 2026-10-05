# 任务03B：index5局部一致性诊断

2026-10-05 UTC。基准提交 `65fb21a6724fba8e062cc2887eaa0db5a477516f`（03A）。状态 **complete**：两套预定矩阵和唯一一对5-inner确认均完成。没有运行G20、完整续采样或新baseline。

**原设置下，原Sampler计算路径A自身不重复；确定性设置下，四条件三次重复及跨条件比较全部精确，原Sampler/D的5-inner+step确认也精确通过。** 因此，03A的差异不能直接归因于反传前的CPU热图复制。本轮观察到的原设置偏差始于完整梯度tensor，前向mid/up图、E/loss均一致；没有定位底层算子，也没有把多个确定性开关的联合作用拆开。改善只在本次记录设置、P1/seed19/index5局部上得到验证。

## 身份、范围与实现

读取实际AGENTS.md、03A report/run_metrics/failure_diagnostic/wrapper、原Sampler/processor/load_branch；初始工作区干净，HEAD为03A提交。03A锁的全部fingerprint在阶段更新前匹配；02B锁的六项原源码/config/protocol及旧checkpoint内fingerprints字典匹配，03A所锁checkpoint/reference/参考图/CSV等身份也一致。根AGENTS.md仅局部更新到03B，其历史hash仍留在03A旧锁中，03B记录这项授权阶段差异；没有修改/解除旧锁或旧源码。

新增 [config](../configs/stage03b.json)、[protocol](stage03b_protocol.md)、[GPU诊断入口](../src/stage03b_diagnostic.py)、[CPU审计入口](../src/stage03b_cpu_delivery.py) 与独立 `outputs/stage03b/`。首次GPU前锁定 **2026-10-05 11:52:04 UTC**，见 [独立锁](../outputs/stage03b/protocol_lock.json)。GPU源码SHA256 `97963f81b88ff675801f35f81833e269198f5767284237f0e6b92eaa69cad9c6`。锁包括新config/protocol/source/AGENTS、旧运行源码、02B输入及03A失败记录；CPU交付脚本在运行结束后新增，自身hash单列在artifact_audit，不改GPU运行锁。

固定P1/seed19，旧 `stage02b/P1/seed19/checkpoint.pt`，after第5次scheduler.step/next_index5/t856/sigma6.6685729。模型离线SDXL base revision `462165984030d82259a11f4367a4eed129e94a7b`；FP16、1024²、LMS51/order4、CFG7、negative=None、batch1（仅确认step的原CFG forward batch2）、冻结模型、apple token3/原mid-up框(0,.10,1,.35)、原E/loss_scale30/threshold.2/sigma²/相对上限.1不变。prompt仍是：

```text
A red apple to the left of a blue ceramic cup, against a plain neutral background, studio photograph, both objects fully visible.
```

环境保持4090、Python3.12.3 `/root/miniconda3/bin/python`、torch2.3.0+cu121、diffusers0.32.2、transformers4.48.3、accelerate1.3.0、CUDA12.1、HF_HOME原路径，local_files_only=True且离线变量=1。没有安装/升级/下载/改site-packages。

每次重新调用**未改动的旧load_branch**，完整验收旧fingerprints/environment、processor、LMS runtime及四项derivatives、conditioning、CPU/CUDA/显式generator RNG，独立加载latent和scheduler。每次先按旧开关验收，再应用本诊断指定开关；确定性worker启动前已有CUBLAS变量，旧environment原本不记录它，输入身份中明确另存该差异，没有冒充历史环境或删除断言。

每worker首尾对四个模型全部state_dict内容逐tensor计算SHA256；三worker的内容hash一致且首尾未变。每次restore核对所有parameter/buffer的dtype/device/shape/storage地址/version/requires_grad，模型参数均CUDA FP16且冻结；buffer按实际dtype单列。每次保存完整输入tensor/RNG，再做身份/安全判断。CPU审计确认28个恢复输入的latent/conditioning/LMS/RNG在worker内及跨worker完全相同（跨worker模型storage地址不同，单独以内容hash核对）。没有对象/几何评测，因为本轮没有最终生成图。

| 条件 | 单次计算路径及采集时序 |
|---|---|
| A | 原Sampler.conditional，反传前无热图复制/统计；原显式single-inner算术 |
| B | 03A DiagnosticSampler，context=None，保留其预算检查 |
| C | 03A DiagnosticSampler原context，conditional返回前在grad之前detach.cpu.numpy及CPU统计 |
| D | 仅暂存同图detach引用，grad完成后复制/统计并立即释放引用 |

顺序严格为 A/B/C/D × 3轮，每次同输入独立restore。所有条件共用资源计数、安全检查和反传后的证据保存；A指原Sampler计算路径。正式公式逐句保持原FP16计算：`loss=30*E; g=autograd.grad(loss,leaf); update=g.detach()*sigma.float().square(); result=leaf.detach()-update`；范数使用原CUDA FP16计算，另记FP32梯度范数。完整maps、E/loss、gradient、update、result均在任何数值比较断言前落盘，未额外forward重新获得A的原始图。

## 单次实现校验与原设置矩阵

先用2次反传分别执行原guidance(inner_iters=1)与显式single-inner。为了取原guidance的完整证据，仅在新模块中临时包装autograd.grad：调用真正grad后才读取调用帧的maps/leaf/loss/完整梯度并保存，finally恢复函数引用；不在旧Sampler中改代码，也不在原A反传前做热图采集。

两独立校验执行的前向图/E/loss精确一致，梯度最大差 `2.86102294921875e-6`，更新后latent最大差 `.0078125`，所以**这对原设置校验的跨运行精确一致没有通过**。每次根据自身完整梯度验证原更新公式都完全一致；按预注册规则，数值差异作为待诊断结果保留，并继续预定矩阵，没有放宽精确门槛或把它称为跨运行通过。之后确定性5-inner确认的逐轮证据补充验证显式D与真实原guidance的计算等价。

原设置：benchmark=False、cudnn.deterministic=False、use_deterministic_algorithms=False，CUBLAS_WORKSPACE_CONFIG未设置。12次矩阵反传全部完成；每个同条件有3个两两重复配对，每种A/B、A/C、A/D有3个按轮配对，共21对。

**全部21对的mid/up原图、E/loss都精确一致；全部21对的完整gradient、update、result都不一致。** 如下表，每行是3个配对的汇总，完整逐对mean/max/不同元素比例及dtype/finite见比较CSV/JSON。

| 比较 | gradient最大绝对差（3对最大） | gradient不同元素比例范围 | result最大绝对差（3对最大） |
|---|---:|---:|---:|
| A/A | 3.81469727e-6 | .888580–.902161 | .0078125 |
| B/B | 2.86102295e-6 | .894394–.903580 | .015625 |
| C/C | 3.81469727e-6 | .899292–.906281 | .015625 |
| D/D | 3.81469727e-6 | .899017–.903671 | .0078125 |
| A/B | 4.41074371e-6 | .899994–.904938 | .015625 |
| A/C | 3.81469727e-6 | .892654–.904419 | .015625 |
| A/D | 3.33786011e-6 | .891708–.904633 | .0078125 |

A1/A2/A3的FP16梯度范数均为 `.044219970703125`，但完整梯度tensor均不相等。A重复配对的最大mean_abs_error为 `3.307286533527076e-7`；不同元素比例较高与绝对差较小同时成立，不能用范数相同代替tensor一致。

这说明原计算路径本身已有可见的反向重复性问题。原设置的跨条件差异与同条件差异同时存在，三轮诊断不足以分离CPU复制是否增加了额外影响，不能把C的不同归因于复制，也不能宣称原设置下D已修复一致性。

## 确定性矩阵及唯一确认

原设置出现差异后，按授权在独立新子进程运行同一12次矩阵。启动前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，worker中明确benchmark=False、cudnn.deterministic=True、use_deterministic_algorithms(True,warn_only=False)。没有切换/搜索kernel，没有warn_only或静默降级；本次没有遇到不支持的确定性算子报错。

原matmul TF32=False、cuDNN TF32=True、FP16/BF16 reduced-precision-reduction=True、flash/mem-efficient/math SDP enable均True、autocast=False保持。多个确定性开关作为一组实验变量，不能据本结果判断是哪一项有效。

确定性矩阵的21对、每对7项完整证据 **全部torch_equal且dtype相同**；max_abs、mean_abs、不同元素比例全部0。A/B/C/D各自重复精确，A/B、A/C、A/D按轮精确，原C的反传前采集在此设置下也没有出现可测数值差异。不同worker的original-A1与deterministic-A1前向图/E/loss仍精确，但梯度/result不精确（result最大差.0078125）；新设置没有复现某一次历史反向结果，不能冒充旧02B/03A执行环境。

只有确定性设置满足A重复及A/D精确门槛，故只选择这一设置，在第三个独立worker执行**唯一一对**原Sampler/D的index5最多5inner，再各执行一次原scheduler.step。原Sampler保留真实guidance while逻辑，D保留原更新前E的early-stop语义；两者都执行5次inner。D正式latent始终保留GPU tensor，没有用CPU结果回灌轨迹。

| 确认项 | 实测 |
|---|---|
| inner0–4各自原mid/up图、E/loss、完整gradient/update/result | 全部torch_equal，所有误差/不同元素比例0 |
| 5-inner guidance后latent | torch_equal=true，max/mean=0，不同元素比例0 |
| guidance后完整packed scheduler runtime/history | 完全一致 |
| 一次index5 step后latent | torch_equal=true，max/mean=0，不同元素比例0 |
| step后完整packed scheduler runtime/history | 完全一致 |

逐inner证据均在最终比较前保存。确认只跑这一对，没有再尝试修复版本；没有把局部确认外推为完整51步历史guided图复现。它支持在本设置/样本/局部窗口上使用D作无数值改变的热图诊断，下一轮预算对照仍须另行验收整体可比性。

## 成本、取证与交付

共 **36次反传**：原single-inner校验2 + 原矩阵12 + 确定性矩阵12 + 唯一确认10。无失败/超限反传、无额外未计费重试。所有证据有限、模型参数无累计梯度、processor恢复，更新相对范数均低于.1；固定配置和安全阈值没有改变。

| worker | 反传数 | worker内部CUDA活动wall秒 | 保守子进程计费秒 | 峰值allocated MiB |
|---|---:|---:|---:|---:|
| original_matrix | 14 | 21.32977 | 26.22085 | 11992.88721 |
| deterministic_matrix | 12 | 23.72186 | 28.26695 | 12036.80908 |
| deterministic_confirm | 10 | 23.00231 | 27.67237 | 12035.39697 |
| 合计/峰值 | **36** | **68.05393** | **82.16017** | **12036.80908** |

保守计费包含子进程import、模型加载、CUDA初始化、全权重hash、保存和CPU比较，低于120秒；内部wall也包含加载/hash/记录等，不表示纯kernel耗时。总峰值allocated **11.75470GiB**，reserved **12.359375GiB（12656MiB）**，无OOM。原设置与确认日志出现cuDNN execution-plan不支持warning后正常执行；没有把它当成根因。运行结束后GPU进程退出，后续CPU审计没有初始化CUDA。

保存69个本地pt文件，共 **79,573,356 bytes（75.8871MiB）**，不含模型权重；全部被已有Git `*.pt`规则忽略。37份trace中有1份是validation_original父目录的重复证据（记录真实guidance result），对应36次独立反传。28份input_state保存完整latent/conditioning/LMS/RNG；4份guidance/step终点状态。小型identity/metrics/比较文件可以供Git审查；没有自动add/commit/push。

实际命令（项目根目录）：

```bash
HF_HOME=/root/autodl-tmp/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/root/miniconda3/bin/python 04_relation_preservation/src/stage03b_diagnostic.py preflight
/root/miniconda3/bin/python -m py_compile 04_relation_preservation/src/stage03b_diagnostic.py
git diff --check
set -o pipefail
HF_HOME=/root/autodl-tmp/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/root/miniconda3/bin/python -u 04_relation_preservation/src/stage03b_diagnostic.py run \
  2>&1 | tee 04_relation_preservation/outputs/stage03b_run.log
HF_HOME=/root/autodl-tmp/huggingface \
/root/miniconda3/bin/python 04_relation_preservation/src/stage03b_cpu_delivery.py
```

父入口没有初始化CUDA；实际三条子进程argv与CUBLAS变量完整记录在summary.commands，remaining timeout和previous-backwards/seconds通过argv传递。CPU另验证“相同范数、不同tensor”及“完全相同clone”的比较正确；py_compile、git diff --check及运行后锁/原输入完整审计通过。

主要交付：

- [stage03b_summary.json](../outputs/stage03b/stage03b_summary.json)、[protocol_lock.json](../outputs/stage03b/protocol_lock.json)、[cpu_checks.json](../outputs/stage03b/cpu_checks.json)、[完整CPU artifact_audit](../outputs/stage03b/artifact_audit.json)。
- [全部比较CSV](../outputs/stage03b/comparisons.csv)、[全部比较JSON](../outputs/stage03b/comparisons.json)、[配对汇总CSV](../outputs/stage03b/comparison_group_summary.csv)、[标量/范数CSV](../outputs/stage03b/scalar_metrics.csv)。
- [原worker记录](../outputs/stage03b/original_matrix/worker_metrics.json)、[原矩阵配对](../outputs/stage03b/original_matrix/comparisons.json)、[原guidance单次实现校验](../outputs/stage03b/original_matrix/implementation_validation.json)。
- [确定性worker记录](../outputs/stage03b/deterministic_matrix/worker_metrics.json)、[确定性矩阵配对](../outputs/stage03b/deterministic_matrix/comparisons.json)、[唯一5-inner/step确认](../outputs/stage03b/deterministic_confirm/confirmation.json)。
- [原矩阵日志](../outputs/stage03b/original_matrix.log)、[确定性矩阵日志](../outputs/stage03b/deterministic_matrix.log)、[确认日志](../outputs/stage03b/deterministic_confirm.log)；各trial的input_identity.json、metrics.json、原始本地input_state.pt/trace.pt留服务器。

03A缺少原探针完整证据的问题已在本轮通过先落盘得到补足；原设置反向不重复及确定性设置局部改善有直接tensor证据。CPU复制影响在原设置下尚未分离，底层算子与确定性开关各自贡献仍未定位；没有进行kernel搜索或超出预算的profiler试验。任务03B完成后停止回传，由下一轮审查决定是否恢复预算对照。
