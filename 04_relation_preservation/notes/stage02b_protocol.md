# 任务02B：生成前固定的起点筛查与单次干预 protocol

基准：11432e791f7eeb05248040f6d1c8024dbfbaab68。本轮是独立起点准备/管线验证，不是正式统计实验。任务02的42/123/2024均不合格，仅说明那轮没得到起点，不是guidance失败或研究假设被否定。历史配置/protocol/锁/结果全部保留。

本文件、`configs/stage02b.json`、新入口和所有复用源码在第一次新生成前写入独立 `outputs/stage02b/protocol_lock.json`。默认旧入口/旧任务保持原样；本轮不解锁旧protocol，不混用历史图片作分支对照。模型SDXL base1.0，本地revision `462165984030d82259a11f4367a4eed129e94a7b`，FP16、1024²、LMS51步order4、CFG7、图像/conditional batch1（CFG拼接batch2），顺序分支。复用现有环境与缓存，无检测器/依赖下载、无site-packages修改、无FLUX。

对象A红苹果，B蓝色陶瓷杯；旧关系A在B左侧，新关系A在B上方，两者几何兼容。固定prompt：

- P1：`A red apple to the left of a blue ceramic cup, against a plain neutral background, studio photograph, both objects fully visible.`
- P2：`On the left is one red apple. On the right is one blue ceramic cup. Two separate objects, against a plain neutral background, studio photograph, both objects fully visible.`

每prompt的seed均为`[7,19,101,314,777,1001,2048,4096]`。先P1全部，再P2全部，逐个生成、看图、人工标注、判定；首个合格即提前停止并选择，不看guided成败换起点。最多16个新baseline，不重跑旧seed。若全不合格或预算停止，保留所有结果停止，不换prompt/扩大池/松判据/预建立旧关系/换模型。报告实际筛查分母和提前停止；不称完整16例评测，不估计成功率，也不把P2作为同prompt成功率改善或方法泛化证据。

最终图人工标注可见主体轴对齐bbox，归一化[0,1]、左上原点；清晰杯把与苹果茎纳入，阴影/倒影不当对象。记录存活、唯一性、身份、遮挡、图像边界截断和标注confidence。无法可靠判断时uncertain，不造坐标；多实例保存各可辨实例的框，不能选一个强行当唯一A/B。必要时提供汇总图交用户复核。

中心差`dx=cx(B)-cx(A)`、`dy=cy(B)-cy(A)`。名义成功>=0.05；<=0.04明确失败，>=0.06明确成功，中间开区间uncertain。身份/多实例模糊使关系uncertain，但保留可计算原差值。对象消失算失败，明显截断单列invalid，禁止据其宣称联合成功。baseline合格：两对象唯一清晰、未明显截断且无明显标注歧义、dx>=0.06、dy<=0；看图后不排除杯把或更改门槛。中心规则不等于整体分离/无重叠或遮挡；本轮不外推这些属性。

所有baseline保存第5次scheduler.step之后的完整状态，next_index=5（完成index4/t875，下一步t856），以及第一恢复步和最终latent参照。合格seed才运行真实续采样：分别从同一未干预checkpoint的独立磁盘副本恢复latent、四项LMS derivatives、内部步索引、实际timestep/sigma等完整运行状态、conditioning、CPU/CUDA/显式generator及Python/NumPy RNG、原processor。禁止set_timesteps重置后跳历史，禁止共享可变scheduler或复制被干预latent。

先无guidance续采样，第一恢复步/最终latent最大绝对误差严格0且torch.equal，最终图像逐像素相同，才允许guided。没有把task01通过或文件可读当新验收通过。遇差异记录并定位，不放宽阈值；未通过则不执行干预。

唯一干预：所选prompt不变、不加above；仅apple attention、全宽目标框`(0.0,0.10,1.0,0.35)`，只index5–9，当前Sampler的两层原E与sigma²更新，loss_scale30、每步最多5inner、E threshold0.2、相对更新安全上限0.1。两套tokenizer apple span须一致且非特殊token，不盲目并集；保存完整span诊断。无cup约束、旧关系penalty/projection/保护项或新损失。看到guided图后不改窗口/强度/框/seed；本轮一合格seed一guided条件，不救参重跑。

记录所有inner梯度有限/非零、参数无累计梯度、更新范数、实际次数、mid/up原始attention比和E，processor恢复和decoded有限性。数值安全停止/未完成图与原因保留，不能当最终关系结果。最终guided用相同人工框规则，存活/多实例/遮挡/截断/uncertain单列。attention目标改善不替代dy几何判定。

无干预最终图是有成本的离线反事实参照，只说明最后旧关系合格，不表示checkpoint latent此时已有可见正确关系或零成本在线可判断。

固定解释：A，新目标明确未修复则当前干预不够有效，尚不能评价成功修复后的保持；B，新关系修复且旧关系保持则该例联合成功，无退化证据；C，新关系修复而旧关系由成功变成明确失败则单个退化候选，需要独立重复；D，对象消失/多实例/截断/身份或阈值模糊，单列失败/uncertain，不包装纯关系退化。所有单例均不证明跨prompt退化或保护有效。

累计本轮同步采样运行时间（含CPU scheduler/状态保存开销）逐步检查；570秒停止以给600秒上限留余量，保存已完成结果。不做额外benchmark、方法比较、保护loss、扰动rollout或小屋重跑。资源记录用本轮实测；所有尝试原图/标注/拒绝理由及负/无效结果保留，checkpoint/reference tensor本地忽略Git，不含权重/凭据/full attention。

交付新报告、配置/锁/命令/模型与源码版本、筛查汇总、baseline/guided原图及带框对照、关系与恢复JSON/CSV、梯度/资源记录。完成后停止交回审查，无自动commit/push或后续候选搜索。
