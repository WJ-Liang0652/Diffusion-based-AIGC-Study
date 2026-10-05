# Stage04 frozen protocol

基准d0bb664e4915b9ab82bee60cc374a2b3c1b06a5f，2026-10-05。任务是合格起点条件下的小样本探索，不估计总体成功率。03C seed19仅列历史，不运行或加入新增分母。

P1固定prompt及所有有效数值/确定性设置由configs/stage03c_repair01.json复用，冻结于configs/stage04.json；复用其PostBackwardSampler，不修改历史源码。SDXL离线revision462165984030d82259a11f4367a4eed129e94a7b，FP16/1024/batch1/LMS51 order4/CFG7/negativeNone。Guidance仅apple mid/up，box(0,.10,1,.35)，原loss scale30/threshold.2/sigma²/relative safety.1，indices5–9，每步最多20inner，原pre-update E的early-stop语义。

固定新seed队列[101,314,777,1001,2048,4096]，每seed仅一次U和一次initial_latent生成并保存tensor/hash。每张baseline人工bbox含自身茎/杯把、排除阴影；长枝叶另记。A红苹果、B主要蓝杯唯一清晰无明显截断且dx>=.06、dy<=0才合格；多实例/身份不清拒绝为uncertain。逐个生成和标注，取得第三个合格者即停止；最多6个。所有筛查和首三个合格名单在任何guided图出现前冻结，包含图像、标注、initial latent与checkpoint哈希。没有3个则运行已有合格者，不补seed。

U连续采样时第5次scheduler.step后存完整checkpoint，next_index5，包括4个LMS derivatives、步索引、conditioning、CPU/CUDA/generator RNG及processor指纹。按冻结名单，每seed先独立从磁盘R续采样46步，首步/final latent及最终图精确相等通过后才独立恢复G20。每个分支重新恢复完整scheduler，不共享可变状态，不调整kernel/阈值、不做修复重试。

仅保留原per-inner/per-step标量和indices5/9更新前后apple mid/up聚合热图。反传后复制CPU，及时释放计算图；不增加诊断轨迹、大attention矩阵、其他loss或G5。参数冻结、有限性/参数梯度/safety断言保留。

最终关系dx=cx(B)-cx(A)、dy=cy(B)-cy(A)，归一化，>=.06成功、<=.04失败、之间uncertain。不改成整体左右分离，不根据结果重新选杯子；多蓝杯/身份不清为uncertain，无可靠bbox留空。分别记录目标存活/唯一/遮挡/截断、苹果数/主要蓝杯数/全部杯数、相对U新增对象和内容保持；几何成功不替代内容质量或实例身份跟踪。

预算覆盖加载/哈希/计算/失败/传输保存/解码活动wall time；人工标注及独立CPU审计不启动CUDA。最多6U+3R+3G20=12轨迹/300inner；累计GPU600秒、540秒后不启动新完整分支。不重置计数。身份/恢复/数值安全失败立即保存证据停止；触限标incomplete，不救结果。

报告实际生成/合格/完成数、全部结果与分类。至少一个新增修复仅支持继续审查；修复均保持旧关系则关系保护问题仍无退化证据。全部有效新增均未修复时停止SDXL调参，按既有授权整理已有FLUX单例准备，不自动运行FLUX。无合格或未完成不可判定。完成停止，不benchmark或开发保护。
