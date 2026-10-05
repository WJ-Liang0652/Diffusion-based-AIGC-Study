# 任务 02 固定 protocol（生成及观察干预前写定）

本文件与 `configs/stage02.json` 在 baseline 生成前固定，并在运行 provenance 中记录 SHA256；本轮不在看图后修改定义。当前基准：7bc3ae9e8d08076d3335c770c4ef8cdf224b863f（04-01）。复用并冻结该提交的 stage01 Sampler、attention processor、loss/更新及 pack/unpack/RNG 逻辑；不混用历史 guided 图片。

固定 prompt：`A red apple to the left of a blue ceramic cup, against a plain neutral background, studio photograph, both objects fully visible.` A 为红苹果，B 为蓝色陶瓷杯。旧关系 A left-of B；新关系 A above B。prompt 始终不变，两种关系几何兼容。

按 `[42,123,2024]` 顺序生成无 guidance baseline，最多三个。每个 baseline 看图、标注、判定后才决定是否运行下一候选；仅选择第一个合格者。没有合格者则停止；多实例/身份模糊/无法可靠标注标 uncertain，保留图像和理由并交用户复核，不对其作关系结论。绝不依据 guided 成败选择 seed。

在最终图人工标 A/B 可见主体轴对齐 bbox，归一化到 [0,1]，原点左上；杯把清晰时包含，苹果可见茎归入主体；阴影/倒影不算主体或第二对象。保存像素框、归一化框、中心、带框原图和标注理由。人工标注主体身份、唯一性、截断和 confidence；确实不能识别时不填虚构 bbox。

`dx=cx(B)-cx(A)`、`dy=cy(B)-cy(A)`。两个关系分别按对应差值判断：名义成功 >=0.05；明确失败 <=0.04；明确成功 >=0.06；0.04<差值<0.06 为 uncertain。身份/多实例歧义会覆盖关系判定为 uncertain，但保留可计算的原始差值和名义结果。对象消失记录失败，不删除结果；明显截断标 invalid，禁止据此宣布联合成功。baseline 合格必须两个对象唯一清晰、无明显截断、dx>=0.06 且 dy<=0，判定没有歧义。中心规则不代表整体分离、遮挡、大小、距离或整体入框；本轮不外推这些属性。

SDXL base 1.0，本地固定 snapshot、FP16、1024×1024、LMS 51 步（order=4）、CFG=7、图像 batch=1、conditional guidance batch=1、原 CFG forward batch=2，各分支顺序执行。两套 tokenizer 必须对 apple 给出完全一致且非特殊 token 的位置；记录完整 ids/decoded tokens 和连续 span。若不一致，则诊断/报告映射阻塞，不取并集继续。

baseline 完成第 5 次 scheduler.step 后保存 checkpoint，next_index=5。同一 baseline 继续至最终图，并保存第一恢复步与最终 latent 作为严格验收参照。合格 seed 的两条后续分支均从同一个未干预 checkpoint 的独立磁盘加载副本出发；新 LMS 实例恢复全部实际 runtime state，包括四项 derivatives、内部索引、timestep/sigma 序列；恢复 conditioning 和 CPU/CUDA/显式 generator RNG，安装原 processor，禁止 reset schedule 后跳过历史。

先无 guidance 续采样，要求第一步、最终 latent 和最终图像完全一致；如有差异保存诊断并停止干预，不放宽阈值。然后唯一 guided 分支只在 index 5–9 执行 apple guidance，框 `(0.0,0.10,1.0,0.35)` 横跨整个宽度，不限制左半边，不对 cup 加保持目标，也没有旧关系 penalty/projection。沿用当前实现 E、loss_scale=30、最多五次 inner、threshold=0.2、sigma²更新、相对更新安全上限 0.1。这个中途配置是诊断条件，不声称等价论文默认方案。恢复入口按保存的 next_index 续采样，guidance 范围由固定配置指定。

保存每次 guidance 梯度有限/非零、参数无累计梯度、实际次数、更新范数、mid/up attention 比例和 E；检查 processor 恢复及 decoded tensor 有限性。安全停止时保留日志、最后完整状态的图像（如可解码）和失败原因，不改变配置重试救结果。未完成分支不得据其图像声称最终关系成功。

最终新关系仅按实际 A/B bbox dy；attention 入框不是最终上方关系成功。baseline 最终图的旧关系仅是无干预反事实参照，不声称 checkpoint latent 已有可见正确关系。

新关系明确成功且旧关系明确成功 = 联合成功；新关系成功而旧关系从 baseline 成功变成明确失败 = 退化候选；新关系明确失败 = 当前条件修复失败，不能把旧关系保持归因于保护。uncertain/invalid 单独报告。任何单例都不证明跨样本或跨 prompt 退化，也不证明保护方法有效。

预算：最多三个 baseline、一个合格 seed 的无 guidance 恢复分支、一个固定 guided 分支。记录实际本轮 GPU 同步计时和峰值显存，不引用 task01 成本。逐步检查累计采样时间，达到 570 秒时停止为 600 秒上限留余量，保留已完成结果和原因；不扩大搜索。checkpoint/必要 reference tensors 仅本地 `.pt` 且忽略 Git，不存权重/凭据/full attention。生成/判定结果及本轮汇报完成后停止，无自动 commit/push。
