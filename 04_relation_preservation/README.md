# SDXL 任务 01

当前目标：单样本复查旧 SDXL layout guidance，并验收 LMS 中间状态保存/恢复。
复用脚本 19 的有效配置、原 attention energy 和 sigma² latent 更新；不执行旧脚本。
只运行一个 prompt、seed、scheduler、checkpoint。验收后停止，等待结果审查。
根 AGENTS.md、历史代码与未提交 FLUX 工作保持原样。

入口：`src/stage01_sdxl.py`。默认 checkpoint 在第 25 次 scheduler.step 后，下一步索引为 25（从 0 开始）。
输出目录要求为空，避免覆盖已有结果；checkpoint 仅适用于记录的环境和相同本地模型。
checkpoint 保存 scheduler 完整实例状态（不含权重）、conditioning、配置、processor 类型和 RNG；
恢复不会调用 set_timesteps 覆盖历史。张量保留原 dtype/device，分支各自获得独立副本。
