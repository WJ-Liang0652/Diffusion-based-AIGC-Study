# 任务03B：index5局部一致性诊断

2026-10-05。首次GPU前锁定；不改变历史源码、锁、结果。P1/seed19、02B checkpoint after scheduler.step/next_index5，FP16/1024/LMS51/order4/CFG7/negative=None、apple原框与原loss/threshold/sigma²/安全上限均固定。只运行index5局部，禁止完整续采样/G20/新baseline。

每次原load_branch按旧fingerprints/environment/processor/LMS/conditioning/RNG完整验收。确定性worker在旧设置下进行每次旧验收，随后应用记录的新开关；CUBLAS_WORKSPACE_CONFIG在子进程启动前设定，旧environment未包含该变量，因此明确记录差异，不冒充旧条件。每次保存起始latent/conditioning/scheduler/RNG与模型参数/buffer的dtype/device/storage/version签名；每worker开始/结束完整权重SHA256并核对，冻结无参数梯度。预估完整证据约<100MiB，不含权重，全tensor仅本地pt、Git忽略。

原设置先用最多2次反传核对显式single-inner与原guidance(inner_iters=1)。显式计算逐句复用原公式；原guidance取证在真实autograd.grad完成之后捕获调用帧的maps/loss/leaf与完整gradient，不在原A的反传前加入热图复制/统计。若两独立执行数值不同，不当作身份/公式错误，保存比较并完成矩阵；各次用其自身gradient精确验证更新公式。原guidance本身及旧Sampler不改。

矩阵按A/B/C/D依次执行，重复3轮，每次独立磁盘恢复。A使用原Sampler.conditional；B用03A DiagnosticSampler但context=None，保留预算检查；C使用03A原context采集，在grad前detach.cpu.numpy+stats；D暂存同一map的detach引用，在grad完成后再CPU复制/stats并及时释放。所有条件从相同输入执行单次原FP16更新；记录原maps/E/loss/完整gradient/update/result、norm/flags和输入身份，再做比较断言。不额外forward重新取A的maps。数值不一致是结果，身份错误/非有限/参数grad/相对上限异常立即停。

比较所有同条件三轮两两A/A、B/B、C/C、D/D，以及每轮对应A/B、A/C、A/D。逐tensor报告torch.equal、max/mean_abs_error、different_element_ratio、dtype/finite；范数相同不作为tensor一致证据。

若原设置任何重复或跨条件tensor差异，独立子进程运行同矩阵；启动前CUBLAS=:4096:8，benchmark=False，cudnn.deterministic=True，use_deterministic_algorithms(True,warn_only=False)。TF32/FP16与其他设置沿用原值。确定性不支持则保存完整traceback/算子位置停止该设置，不切kernel或静默降级。

矩阵全部完成后，若某设置A三轮精确且D与对应A精确（图/能量/gradient/update/result均精确），按原设置优先选择首个合格设置，至多执行一对原Sampler/D的5-inner与一次step。原Sampler通过post-grad取证，D经post-grad flush，逐inner完整证据先落盘，随后最终guidance/step latent及完整scheduler状态精确比较。只一个确认条件，不尝试多个版本。确认新子进程须核对模型内容hash与矩阵相同。

最多36反传=单次校验2+两矩阵24+确认10。反传调用前计数，失败计入。父进程累计所有GPU子进程wall time（含import、模型加载、哈希、保存与CPU比较，作为保守GPU收费）上限120秒；子进程逐forward/grad/权重哈希检查剩余额度，父进程设置总剩余timeout，触限保留此前证据并incomplete。不放宽精确门槛，不运行完整续采样。所有真实命令/版本/hash/cost保存。

交付notes/stage03b_report.md、stage03b_summary.json、比较JSON/CSV、新config/protocol/lock/source与raw本地tensor；区分原实现不重复、插桩差异、确定性设置效果和未定位原因。结束交回，不自动恢复03预算对照。
