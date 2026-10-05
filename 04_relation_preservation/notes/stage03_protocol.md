# 任务03：增加 inner 预算能否产生物体上移

锁定前预注册，2026-10-05。仅 P1/seed19，从02B第5次scheduler.step后(next_index=5)的独立磁盘状态顺序运行 U、G5、G20。不生成新起点。基准16f70132e316949e866e839192a45f417d222b03，完整实际配置见 configs/stage03.json。

唯一实验变量为每步inner上限5→20；indices5–9、apple全宽框(0,.10,1,.35)、原mid/up平方框外比例loss、scale30、threshold.2、sigma²与相对上限.1均不变。原early-stop用更新前E。无cup/旧关系保护项。

旧锁全部源码/config/protocol指纹与checkpoint指纹必须匹配；新锁含新配置/protocol/wrapper及全部依赖源码与旧输入文件哈希，首次GPU前建立。缺失或不符停止。完整原load_branch断言保留。U第一恢复步与最终latent必须torch.equal，最终图必须逐像素匹配旧baseline。G5 index5与原Sampler额外至多5更新的guidance/step结果完全一致；G5全部原inner/step指标及最终旧guided图必须完全一致。失败不启动G20，不放宽门槛。

wrapper调用原guidance；conditional后仅detach记录，每inner与每步E/ratio/范数沿用原CSV。只留首轮与最终更新后32×32聚合map，不留全head全层矩阵。U indices5–9只读forward各一份；index50最后step前只读apple/cup。额外forward前后保存/恢复完整scheduler runtime与RNG，断言正式latent、scheduler、RNG不变。cup仅诊断，绝不参与正式guidance loss。

raw npz含mid/up/mask；统计峰值坐标/数值、归一化像素中心质心、自然对数空间熵与mask面积比例。面板在同layer、同index和同phase across branches采用共同vmin=0/vmax，before/after也共同色标；不同index不强行比较绝对色值。热图与最终bbox只能作跨时间诊断，不宣称枝叶/瓶attention因果归属。

最终人工bbox沿02B：可见主体含苹果茎/杯把、排除阴影；归一化dx=cx(B)-cx(A)、dy=cy(B)-cy(A)。>=.06明确成功、<=.04明确失败、中间uncertain。记录存活/唯一性/身份/遮挡/截断与额外对象，标不准保留uncertain。G20未修复为负结果；修复且旧保持为单例联合成功；修复且旧明确失败才为待重复退化候选。对象缺失/模糊单列，不用attention判几何。

最多3完整续采样、130inner（G5≤25、G20≤100、原Sampler探针≤5）。所有GPU尝试/diagnostics计入累计570秒，540秒后不启动分支；逐forward/step检查上限，单个GPU操作为不可分割单元，触限保存已有产物并标incomplete。失败不换seed/参数补救。模型离线缓存revision462165984030d82259a11f4367a4eed129e94a7b，FP16/1024/LMS51-order4/CFG7/negative=None/batch1/冻结参数。无下载安装、FLUX、环境变更、共用源码改动、自动Git提交。

交付新配置/protocol/锁/wrapper、CPU检查、实际命令/环境/版本/哈希、门槛结果、inner/step/热图CSV、三个原图/框图、raw npz与少量面板、stage03_summary.json与notes/stage03_report.md。大latent文件只留服务器。完成停止，不自动扩展。
