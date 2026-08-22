# Diffusion Reproduction

本项目研究 diffusion-based AIGC 的推理与科研复现流程。

第一阶段使用 Hugging Face Diffusers 分别调用 SDXL 与 FLUX，完成可复现的文生图推理和基础参数实验。第二阶段将在 Diffusers 版本的 SDXL 上复现 *Training-Free Layout Control with Cross-Attention Guidance*。

两个阶段共用同一套 Python、PyTorch/CUDA 与 Diffusers 环境，并共用数据盘上的 Hugging Face 缓存：`/root/autodl-tmp/huggingface`。模型与实验输出按阶段目录分别管理。
