# 任务03C：一次有限SDXL端到端有效性验证

2026-10-05，首次GPU前锁定新配置/源码/旧输入与03B验证证据哈希；保留历史源码/锁/结果。前置门槛为03B确定性A/A、A/B与5-inner guidance/step latent及scheduler完全一致，检查报告、机器记录和原始本地tensor。禁止重跑03B矩阵。

全部正式GPU进程启动前CUBLAS=:4096:8，从首个GPU计算起benchmark=False、cudnn.deterministic=True、use_deterministic_algorithms(True,warn_only=False)，TF32/FP16/SDP开关逐项匹配03B实际active_switches。固定P1/seed19、SDXL revision462165984030d82259a11f4367a4eed129e94a7b、FP16/1024/LMS51/order4/CFG7/negative=None/batch1、参数冻结。旧输入身份与新执行身份分别验收，不把新开关冒充旧环境。

严格使用旧02B checkpoint的initial_latent（哈希必须匹配旧有效配置），复用其可信conditioning/RNG；不调用prepare_latents重新抽noise、不更改prompt、不筛seed。U完整51步，在第5次step后保存新完整LMS runtime、四项derivatives、latent/initial、conditioning、RNG、switches/environment/源码身份的checkpoint，next_index5；不中断继续保存第一恢复step和最终latent参考。

第二条轨迹R实际从新磁盘独立加载恢复46步；first/final latent torch.equal且最终图逐像素一致才通过，仅对新设置内部U/R比较。旧baseline/guided仅历史参考不作为旧非确定性guidance精确门槛。U原图人工bbox沿02B规则，唯一清晰、未截断、dx>=.06且dy<=0才合格；不合格保留并停止，不换seed。

prepare进程只执行U/R后退出，CPU人工标注合格记录保存后才允许guide进程启动G5/G20。G5与G20各自从新checkpoint独立恢复完整scheduler/conditioning/RNG；除inner上限5→20外，indices5–9、apple框(0,.10,1,.35)、mid/up原loss、scale30、threshold.2、sigma²更新及相对上限.1完全一致。early-stop仍用更新前E，无cup/保护loss。

wrapper直接调用原Sampler.guidance，只在autograd.grad返回后flush已暂存的detached map（03B通过D方案），不在反传前CPU复制统计，不留图。仅index5/9保存apple首次更新前/最后更新后32×32聚合mid/up和原mask；无完整全层/全head矩阵。保存每inner/step的原E/ratio/梯度/更新范数、finite/次数/无参数累计grad，必要raw数组/面板统一同层同index色标。

最终U/G5/G20人工可见主体bbox含苹果茎/杯把、排除阴影；dx=cx(B)-cx(A)、dy=cy(B)-cy(A)，归一化坐标。>=.06明确成功、<=.04明确失败、其间uncertain。对象存活/唯一性/身份/遮挡/截断/额外对象单列，不能可靠标注留待复核。新目标未修复不能评价保护，attention改善不替代最终几何。

正常最多4条轨迹和125inner，顺序运行。累计GPU活动包含加载/哈希/diagnostics/decode/失败；使用两个GPU进程内部活动wall作为保守收费，CPU标注时间不收费，600秒硬边界/540秒后不启新分支，逐step/forward检查且按所剩额度安排。若出现明确wrapper/状态/设备错误，只允许一次最小修复及必要局部复验，新attempt另存源码快照/锁/输入/结果，旧锁不改；无明确错误不继续kernel/参数/seed搜索。

若G5/G20均未明确修复且无可在本范围解决的实现错误，结论为“当前SDXL控制实现未通过有效性验证”，停止SDXL搜索，主线按用户授权切FLUX。只读取既有current_state与Stage6E/7代码/实测指标，整理下一轮单例apple上方修复、左侧保持的配置与起点/成本，不自动运行FLUX或批量实验。若成功，仅报告单例联合成功/退化候选/对象不确定，停止不扩大。

交付report/summary、新config/protocol/lock/source、恢复误差、原图/人工bbox对照、CSV/少量heatmaps、命令/有效开关/hash/时长/显存与继续或止损结论。新大tensor估计<10MiB，留本地并Git忽略，不存模型权重。
