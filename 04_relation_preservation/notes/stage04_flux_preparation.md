# 任务04止损后：已有FLUX单例准备（未运行）

本轮新增合格起点2个：2048的新目标未修复；4096目标苹果缺失且多蓝杯/容器，关系不可可靠评估。新增明确修复0，唯一可判几何关系的新增案例未修复。按用户授权停止当前SDXL实现调参，下一步主线转已有FLUX.1-dev单例“修复上方、检查左侧”。这不是总体成功率或跨模型优越性结论；本轮FLUX GPU工作为0。

配置草案：[stage04_flux_single_preparation.json](../configs/stage04_flux_single_preparation.json)。它标记preparation_only，**没有在本轮执行的入口、GPU预算或执行授权**。这里只整理既有实现、可审查的候选、单例目标及成本；下一轮应在GPU前冻结实际wrapper/运行开关和预算，不复用SDXL的FP16/LMS配置。

## 1. 可复用的基础

已读取`03_flux_layout_control/notes/current_state.md`及实际Stage6E/7源码、依赖的native attention/update路径和指标。有效路径为：`18_flux_dev_paper_aligned_guidance.py`的`run_guided/optimize_once`，Stage6E `19_flux_dev_tight_bbox_guidance.py`，Stage7 `20_flux_dev_repeatability.py`。历史文件只读，输入与源码哈希在`outputs/stage04/flux_preparation_manifest.json`。

FLUX.1-dev已有缓存revision `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`。既有配置为BF16、768×768、batch1、50步、guidance3.5、sequential CPU offload、transformer gradient checkpointing、冻结参数。FlowMatchEulerDiscreteScheduler动态shift，48×48 image-token网格、packed latent `[1,2304,64]`，mu=.8466666667。使用native joint-normalized image-query→target-text-key概率，Eq(2)，Double Stream block18；不将其称作SDXL cross-attention。

冻结继承Stage6E的indices0–9、每步最多5次inner、after-update E threshold=.2、eta_scale=.25、base_eta=96.61734058869797、eta=24.154335147174493、sigma²与FP32 gradient→BF16 update算术、relative safety=.1。本次准备的是**apple单目标上移**，cup仅作最终几何/内容评测，不加cup/保护loss；这是新任务定义，尚无apple-only上移实测有效性证据。

实现注意：历史`optimize_once`的记录遍历target_spans并索引loss对应objects，因此guidance探针/target_spans只传apple，并设`LAYOUTS['layout_b']={'apple':'apple_target'}`；不能把cup加进span后假定它不影响loss或记录。cup只读诊断如确有必要须另外明确实现、计成本。原Stage6E/7入口包含双目标swap及固定输出路径，**不能直接调用其main来执行新单例**；下一轮须在新目录用独立wrapper复用helpers，保留历史源码/锁/输出，不修改site-packages。

## 2. 固定单例候选与兼容目标

优先复用已有FLUX seed42，prompt保持其历史`a red apple and a blue cup, realistic photo`，不重新筛seed、不仅靠seed重画噪声。已有`outputs/baseline/flux_dev_baseline_state.pt`保存initial_latents、全部50个Pipeline step latent、final latent及latent_image_ids，initial tensor哈希`ae3c7bd3742597f7881546de96177481f24d92dfd30de63cd54f6ffc46cb2c01`。

**该历史state没有完整conditioning/RNG/scheduler，不能冒充任务04的可恢复LMS checkpoint。** 下一轮从已保存initial tensor重建独立conditioning/FlowMatch schedule/RNG并记录，先精确验收历史Pipeline轨迹/final/image对应输入；失败即停止，不靠kernel搜索或阈值救结果。若采用新的明确运行开关，必须独立记录预期差异并重新建立内部可信参考，不能说已复现历史环境。

本轮只在CPU检查历史原图：苹果含短自身茎bbox约(78,368,351,651)，蓝杯含把bbox约(388,314,758,665)，坐标基于768图；长附着叶单列为已有额外对象，排除阴影。dx=.466797、dy=-.026042，两个目标唯一清晰且完整，满足旧左侧/新上方待修复的起点条件。见`outputs/stage04/flux_candidate42_review.json`。这只是历史候选的图像审阅，下一轮仍须核验保存输入、运行一致性和实际起点；不合格或不可复现则单例停止，不私自换seed。

按这次可见主体规则取新source-grid `[4,23,22,41)`，提出apple target-grid `[4,3,22,21)`：x、宽高不变，仅上移20tokens，normalized `[.083333,.0625,.458333,.4375]`。其中心(.270833,.25)与现有蓝杯的左/上关系兼容，不要求整苹果都处于SDXL的(.10,.35)高度带。此为待下一轮冻结的有限框单例草案，**不是复用历史apple-right/cup-left交换框，也不是本轮SDXL调参**。

最终评测沿本轮可见bbox中心规则：dx/dy≥.06明确成功、≤.04失败、之间uncertain；目标缺失/多实例/多蓝杯与身份不清单列，拒绝按想要的结果选杯子。所有对象数、额外对象、背景/柄/形状等内容变化保留；长叶既已存在，不能把它算作guided新增。类别颜色匹配不是跨时间实例身份跟踪。

## 3. 历史成本与实际限制

| 历史任务 | inner次数 | 实测时长 | 峰值allocated |
|---|---:|---:|---:|
| FLUX seed42 baseline | 0 | 213.072秒 | 1384.690MiB |
| Stage6E seed42双目标有限框交换 | 22 | guided430.994秒，含加载437.086秒 | 2004.414MiB（reserved2140MiB） |
| Stage7 apple/cup seed123 | 21 | guided421.409秒 | 2004.695MiB |
| Stage7 apple/cup seed2024 | 18 | guided386.981秒 | 2004.695MiB |
| Stage7 banana/bottle | 48 | guided733.487秒 | 2004.695MiB |

上述为历史运行，不能保证新apple-only上移的成本或成功。以一次baseline/replay约213秒加一次guided约387–733秒估计，单例活动时间可能约600–950秒，额外加载/验收/保存另计；因此不能把它偷偷接到本轮SDXL600秒额度中。下一轮应明确总预算和停止阈值，再冻结新入口，不复用历史环境搭建或下载模型。

历史Stage6E/7展示的是粗略双目标搬移/左右交换，同时有杯把、背景、尺度与对象数量等fidelity失败，**不构成“修复上方时保持左侧”的证据**。本轮只准备一个兼容约束的正式单例，不自动进入保护方法或FLUX批量benchmark。
