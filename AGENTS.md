# AGENTS.md

## 1. 项目总目标

这是一个面向扩散模型 / 文生图模型的科研复现项目。

当前服务器将连续完成两个阶段的任务：

1. **初级阶段：Diffusers 推理与代码理解**
   - 使用 Hugging Face Diffusers 调用 **SDXL**
   - 使用 Hugging Face Diffusers 调用 **FLUX**
   - 完成文生图推理、固定随机种子、基础参数实验
   - 理解两种 Pipeline 的基本结构与主要差异

2. **进阶阶段：论文复现**
   - 在 **Diffusers 版本的 Stable Diffusion XL** 上复现：
     **Training-Free Layout Control with Cross-Attention Guidance**
   - 该阶段按“无现成参考代码”的方式推进
   - 需要从论文方法出发，在现有 Diffusers SDXL 推理流程中定位并实现 Cross-Attention Guidance / Layout Control
   - 完成基础功能、实验验证、结果记录与代码整理

项目目标不是让 Codex 黑盒式把任务做完，而是让用户逐步掌握科研代码复现流程。

---

## 2. 当前服务器配置

当前开发环境为租用的 Linux GPU 服务器，通过 SSH + Codex CLI 进行开发。

已知配置：

- OS: Ubuntu 22.04
- GPU: NVIDIA RTX 4090 24GB × 1
- CPU: 12 vCPU, Intel Xeon Platinum 8352V @ 2.10GHz
- RAM: 90GB
- CUDA: 12.1
- 镜像预装 PyTorch: 2.3.0
- 镜像预装 Python: 3.12
- 系统盘: 30GB
- 数据盘:
  - 免费 50GB SSD
  - 已额外扩容 100GB
  - 当前按约 **150GB 数据盘空间**规划使用

注意：

- 当前项目目录以 Codex CLI 启动时的项目根目录为准。
- 不要假设固定绝对路径，首次运行时使用 `pwd`、`df -h`、`lsblk` 确认。
- 大型模型、Hugging Face cache、实验输出优先放数据盘。
- 不要把 FLUX / SDXL 模型缓存到 30GB 系统盘。
- 在大型下载前检查磁盘空间。

---

## 3. 总体目录结构

整个复现项目建议使用一个根目录，并把两个阶段分开：

```text
diffusion-reproduction/
│
├── AGENTS.md
├── README.md
├── requirements.txt
├── .gitignore
│
├── 01_diffusers_basics/
│   ├── sdxl/
│   │   ├── sdxl_inference.py
│   │   ├── experiments/
│   │   ├── outputs/
│   │   └── notes.md
│   │
│   ├── flux/
│   │   ├── flux_inference.py
│   │   ├── experiments/
│   │   ├── outputs/
│   │   └── notes.md
│   │
│   └── comparison.md
│
├── 02_layout_control/
│   ├── paper_notes/
│   ├── src/
│   ├── experiments/
│   ├── outputs/
│   ├── configs/
│   └── reproduction_notes.md
│
└── shared/
    ├── utils/
    └── assets/
```

原则：

- **阶段分目录，环境尽量共用。**
- 不要为了第二阶段提前复制一整套环境。
- 不要把同一个模型重复下载多份。
- 不要把同一套基础代码复制很多份；真正共用的工具才放 `shared/`。
- 初级阶段代码保持简单，不要过早抽象。

---

## 4. 环境策略：尽量只维护一套

### 4.1 默认策略

优先使用一个统一 Conda / venv 环境，例如：

```text
diffusion-repro
```

两个阶段都共用：

- Python
- PyTorch
- CUDA runtime
- diffusers
- transformers
- accelerate
- safetensors
- huggingface_hub
- numpy
- Pillow
- matplotlib
- 其他确有必要的依赖

不要为：

```text
01_diffusers_basics
02_layout_control
```

分别创建两套几乎相同的环境。

---

### 4.2 Python 版本

服务器镜像预装 Python 3.12。

处理原则：

1. 先检查现有 Python 3.12 + PyTorch 2.3.0 是否能稳定运行当前 Diffusers。
2. 如果没有兼容问题，可以沿用。
3. 如果出现明确的包兼容问题，再统一创建 Python 3.10 或 3.11 环境。
4. 不要一开始就重装整个 CUDA / PyTorch 环境。

---

### 4.3 requirements 管理

项目根目录优先维护：

```text
requirements.txt
```

初级阶段跑通后，再把验证可用的版本固定下来。

如果第二阶段只增加少量额外依赖，优先继续使用同一环境并更新根目录 requirements。

只有出现**明确的版本冲突**时，才考虑：

```text
requirements-base.txt
requirements-layout-control.txt
```

或单独环境。

不要因为“可能冲突”提前拆环境。

---

## 5. Hugging Face 与模型缓存策略

SDXL 和 FLUX 都会占用较大磁盘空间。

整个项目应共用一个 Hugging Face cache，例如数据盘中的：

```text
<DATA_DISK>/huggingface/
```

如果环境中已经设置：

```bash
echo $HF_HOME
```

则优先沿用现有设置。

如果没有设置：

1. 先确认数据盘实际挂载路径。
2. 再设置统一 `HF_HOME`。
3. 不要把模型缓存放系统盘。
4. 不要为阶段 1 和阶段 2 分别下载一份 SDXL。

需要检查：

```bash
df -h
du -sh "$HF_HOME" 2>/dev/null || true
```

模型下载前：

- 检查剩余磁盘空间
- 检查 Hugging Face 登录状态
- 检查模型是否 gated
- 不要把 token 写进代码
- 不要把 token 提交 Git

---

# Part I：初级复现

## 6. 阶段 1A：环境检查

进入项目后先执行只读检查：

```bash
pwd
ls -la
df -h
lsblk
nvidia-smi
python --version
which python
pip --version
git status
```

Python 检查：

```python
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("cuda:", torch.version.cuda)
```

确认以下包：

```text
torch
diffusers
transformers
accelerate
safetensors
huggingface_hub
```

原则：

- 先检查，再安装。
- 不要一次安装大量无关依赖。
- 不要随意升级 / 降级 PyTorch。
- 出现环境问题时先解释根因。

---

## 7. 阶段 1B：SDXL 最小 Baseline

目标模型：

```text
stabilityai/stable-diffusion-xl-base-1.0
```

目标：

- 使用 Diffusers 官方 Pipeline。
- 在 RTX 4090 上完成一次文生图推理。
- 固定随机种子。
- 保存生成结果。
- 记录关键参数。
- 代码尽量贴近官方最小示例。

建议入口：

```text
01_diffusers_basics/sdxl/sdxl_inference.py
```

输出：

```text
01_diffusers_basics/sdxl/outputs/
```

第一版代码不要加入：

- Web UI
- Gradio
- FastAPI
- Hydra
- 分布式
- 数据库
- 多进程框架
- 复杂类封装
- 与当前任务无关的工程重构

需要理解：

- `StableDiffusionXLPipeline`
- `from_pretrained`
- `torch_dtype`
- `.to("cuda")`
- `prompt`
- `negative_prompt`
- `generator`
- `seed`
- `num_inference_steps`
- `guidance_scale`
- `height`
- `width`
- `.images[0]`

---

## 8. 阶段 1C：SDXL 基础实验

Baseline 跑通后进行少量单变量实验。

要求固定：

- prompt
- seed
- 其他参数

一次只修改一个变量。

至少完成一种：

### 实验 A：steps

例如：

```text
10
20
30
50
```

观察：

- 生成质量
- 细节
- 推理耗时

### 实验 B：guidance scale

例如：

```text
1
3
5
7
10
```

观察：

- prompt 对齐程度
- 图像自然度
- 过强 guidance 的影响

记录：

- prompt
- seed
- steps
- guidance
- resolution
- dtype
- runtime
- 输出路径
- 简要观察

---

## 9. 阶段 1D：FLUX 最小 Baseline

优先使用：

```text
black-forest-labs/FLUX.1-schnell
```

完成后，如确有需要，再尝试：

```text
black-forest-labs/FLUX.1-dev
```

入口建议：

```text
01_diffusers_basics/flux/flux_inference.py
```

输出：

```text
01_diffusers_basics/flux/outputs/
```

目标：

- 使用 Diffusers `FluxPipeline`
- 固定随机种子
- 完成一次文生图
- 保存结果
- 记录参数
- 理解 FLUX 与 SDXL 在调用层面和架构层面的主要差异

由于 RTX 4090 只有 24GB 显存：

- 优先按当前 Diffusers 官方推荐方式处理显存。
- 必要时使用 CPU offload。
- 不要机械套用所有优化选项。
- OOM 时先分析显存占用。
- 不要因为一次 OOM 就重装整个环境。

---

## 10. 阶段 1E：SDXL 与 FLUX 对比

完成两个 Baseline 后，在：

```text
01_diffusers_basics/comparison.md
```

记录至少以下内容：

### 相同点

```text
Prompt
  ↓
Text Encoding
  ↓
Latent Representation
  ↓
Iterative Generation
  ↓
VAE Decode
  ↓
Image
```

### SDXL

重点理解：

```text
Tokenizer / Text Encoder
UNet
Scheduler
VAE
CFG
```

### FLUX

重点理解：

```text
Tokenizer / Text Encoder
Transformer
Flow / Scheduler
VAE
Guidance
```

需要能解释：

- Pipeline 是什么
- latent 是什么
- scheduler 的作用
- VAE 的作用
- SDXL 为什么使用 UNet
- FLUX 为什么核心是 Transformer
- seed 的作用
- inference steps 的作用
- guidance 的作用
- dtype 的作用
- CPU offload 为什么能降低显存占用

---

## 11. 初级阶段完成标准

以下全部完成后，才进入进阶论文复现：

- [ ] CUDA / PyTorch / Diffusers 环境正常
- [ ] SDXL 能生成并保存图片
- [ ] SDXL 固定 seed 可重复实验
- [ ] 至少完成一次 SDXL 单变量实验
- [ ] FLUX.1-schnell 能生成并保存图片
- [ ] FLUX 固定 seed 可重复实验
- [ ] 能解释 SDXL / FLUX Pipeline 的基本差异
- [ ] 能解释 UNet 与 Transformer 的角色
- [ ] 能解释 seed / steps / guidance / dtype / offload
- [ ] 初级阶段代码已由 Git 管理
- [ ] 关键实验参数有记录

---

# Part II：进阶论文复现

## 12. 进阶任务

目标：

在 Hugging Face Diffusers 版本的 SDXL 上复现：

```text
Training-Free Layout Control with Cross-Attention Guidance
```

当前按：

```text
无官方 / 无可直接使用参考代码
```

的方式处理。

核心目标不是重新训练 SDXL，而是在现有预训练 SDXL 推理过程中实现 **training-free layout control**。

---

## 13. 进阶阶段总体路线

不要一开始就直接改 Diffusers 大量源码。

按以下顺序推进：

```text
论文方法重新梳理
        ↓
明确算法输入 / 输出
        ↓
把论文公式映射到 SDXL 推理流程
        ↓
定位 Cross-Attention
        ↓
读取 / 保存 Attention Map
        ↓
验证 Attention Map 是否正确
        ↓
实现 Layout Condition / Region Objective
        ↓
实现 Cross-Attention Guidance
        ↓
把 Guidance 接入 denoising loop
        ↓
最小样例测试
        ↓
调参数
        ↓
复现论文定性结果
        ↓
整理实验
```

---

## 14. 阶段 2A：论文方法拆解

先在：

```text
02_layout_control/paper_notes/
```

记录论文核心内容。

至少明确：

1. 论文解决什么问题
2. 输入是什么
3. 输出是什么
4. 是否需要训练
5. Layout 条件如何表示
6. Cross-Attention Map 如何获得
7. Guidance loss / objective 如何定义
8. Guidance 在 denoising 的哪一步生效
9. 哪些 timestep 使用 guidance
10. guidance strength 如何控制
11. 是否需要修改 latent
12. 是否需要修改 attention
13. 是否修改模型参数
14. 推理额外开销是什么

如果论文公式无法直接映射到代码，先停止实现并解释缺口。

---

## 15. 阶段 2B：建立干净的 SDXL 进阶 Baseline

不要直接在初级脚本上堆大量修改。

在：

```text
02_layout_control/src/
```

建立一个可控、可调试的 SDXL 推理 baseline。

它应与初级 SDXL 使用：

- 相同模型 cache
- 相同主要 Python 环境
- 相同 GPU
- 相同 Diffusers 安装

但代码可以适当拆开，以便：

- 获取 prompt embedding
- 初始化 latent
- 获取 timesteps
- 调用 UNet
- 执行 scheduler step
- VAE decode
- 插入 attention hook / processor
- 插入 guidance

---

## 16. 阶段 2C：定位 SDXL Cross-Attention

目标不是立刻实现论文，而是先回答：

```text
SDXL 的 Cross-Attention 在 Diffusers 当前版本中具体在哪里？
```

需要定位：

- UNet 中 attention block
- self-attention 与 cross-attention 的区别
- query / key / value 来自哪里
- text condition 如何进入 attention
- attention processor 机制
- 当前 Diffusers 版本可否通过 processor / hook 获取 attention map

原则：

- 优先使用公开、稳定的 Diffusers 扩展点。
- 不要一开始直接修改 site-packages 中的 Diffusers 源码。
- 如果必须 patch library，先说明原因。
- 优先把自定义逻辑放项目自己的 `src/` 中。

---

## 17. 阶段 2D：Attention Map 可视化验证

在实现 guidance 前，先验证是否能可靠得到 Cross-Attention Map。

需要：

- 使用简单 prompt
- 选取明确 token
- 获取不同 timestep / layer 的 attention
- 保存或可视化 attention map
- 确认空间尺寸、token 对应关系、batch / CFG 维度处理正确

输出放：

```text
02_layout_control/outputs/attention_maps/
```

只有 attention map 验证正确后，才进入 guidance 实现。

---

## 18. 阶段 2E：实现 Layout 表示

根据论文定义，实现 layout 输入。

可能包括：

- bounding box
- spatial mask
- region
- token-to-region mapping

但必须以论文为准，不要凭空设计。

配置建议放：

```text
02_layout_control/configs/
```

例如：

```text
prompt
token
box / mask
seed
steps
guidance strength
target layers
target timesteps
```

---

## 19. 阶段 2F：实现 Cross-Attention Guidance

根据论文公式实现 guidance。

要求：

1. 明确 objective / loss。
2. 明确 loss 对什么变量求梯度。
3. 明确每个 timestep 是否都使用。
4. 明确 guidance strength。
5. 明确 CFG 与 layout guidance 的关系。
6. 保持模型参数冻结。
7. 不进行训练。
8. 防止梯度无意累计到模型参数。

实现时优先拆成可测试函数。

例如逻辑层面可类似：

```text
attention map
    ↓
layout objective
    ↓
loss
    ↓
gradient
    ↓
update latent / guided state
    ↓
continue denoising
```

实际实现必须严格以论文为准。

---

## 20. 阶段 2G：最小功能测试

不要一开始复现复杂论文图。

先设计最简单的布局：

```text
"a red apple and a blue cup"
```

例如：

- apple 在左侧
- cup 在右侧

或者论文中最简单的示例。

先验证：

```text
无 Layout Guidance
vs.
有 Layout Guidance
```

是否出现明显空间控制差异。

---

## 21. 阶段 2H：实验设计

每次实验至少记录：

```text
model
prompt
negative prompt
seed
resolution
num_inference_steps
CFG scale
layout condition
target token
target region
layout guidance scale
guidance timestep range
attention layers
runtime
VRAM
output path
```

不要同时修改很多变量。

优先单变量实验：

- guidance scale
- guidance timestep range
- attention layer selection
- box size / region
- seed

---

## 22. 阶段 2I：复现论文结果

根据论文中可获得的信息逐步复现：

1. 定性图片
2. 相同 / 相近 prompt
3. 相同 / 相近 layout
4. 相同随机种子条件（如论文提供）
5. 相同推理步数
6. 相同 guidance 参数
7. 论文中使用的评价指标（如果适合当前时间和算力）

如果无法完全复现：

- 明确哪些参数论文未公开
- 明确 Diffusers / SDXL 版本差异
- 明确实现假设
- 不要伪装成完全一致复现

---

## 23. 进阶阶段目录建议

```text
02_layout_control/
│
├── paper_notes/
│   └── method.md
│
├── src/
│   ├── baseline.py
│   ├── attention_control.py
│   ├── layout_guidance.py
│   └── utils.py
│
├── configs/
│   └── examples/
│
├── experiments/
│   ├── exp_001/
│   ├── exp_002/
│   └── ...
│
├── outputs/
│   ├── baseline/
│   ├── attention_maps/
│   └── guided/
│
└── reproduction_notes.md
```

这是建议结构，不要为了匹配结构而创建空的复杂框架。

按实际进展逐步创建。

---

## 24. 进阶阶段完成标准

最低完成：

- [ ] 能稳定运行自己的 SDXL denoising baseline
- [ ] 能定位并获取 Cross-Attention
- [ ] Attention Map 可视化验证正确
- [ ] 能输入一个明确 layout
- [ ] 实现论文 Cross-Attention Guidance 核心逻辑
- [ ] 模型无需重新训练
- [ ] 简单布局实验能观察到控制效果
- [ ] 有 baseline vs guidance 对比
- [ ] 关键参数和结果有记录
- [ ] 代码由 Git 管理
- [ ] 能解释实现与论文公式的对应关系
- [ ] 明确记录未完全复现或存在假设的部分

如果时间允许，再做：

- 多 prompt
- 多 layout
- 多 seed
- 参数消融
- 定量评价
- 论文图表级复现

---

# 25. 两个阶段如何共用配置

必须优先复用：

### 共用 GPU / CUDA / PyTorch

```text
RTX 4090
CUDA
PyTorch
```

### 共用 Python 环境

默认：

```text
diffusion-repro
```

### 共用 Hugging Face cache

SDXL 不重复下载。

### 共用基础依赖

```text
torch
diffusers
transformers
accelerate
safetensors
huggingface_hub
numpy
Pillow
matplotlib
```

### 共用 Git repository

两个阶段在同一个 Git repo 中。

### 不建议共用的内容

实验输出分目录：

```text
01_diffusers_basics/.../outputs
02_layout_control/outputs
```

实验记录分目录。

避免第二阶段实验污染初级 baseline。

---

# 26. Codex 工作方式要求

## 26.1 用户是科研初学者

不要默认用户已经熟悉：

- Diffusers 内部源码
- PyTorch autograd
- attention processor
- hook
- CUDA OOM 排查
- Git 高级操作

出现这些内容时应简要解释。

---

## 26.2 不要一次完成所有步骤

本轮只做用户明确指定的任务。

例如：

“检查环境”

则不要顺便：

- 安装全部依赖
- 下载 SDXL
- 下载 FLUX
- 写完两个阶段全部代码

---

## 26.3 修改前说明计划

涉及以下操作前先说明：

- 新建 / 修改文件
- pip / conda install
- 升级 / 降级包
- 修改环境变量
- 下载大型模型
- 删除文件
- 大规模重构
- 运行高成本 GPU 实验

只读命令可直接执行。

---

## 26.4 报错处理

遇到问题：

1. 读取完整报错
2. 分类
3. 解释根因
4. 最小修改
5. 重新验证

优先检查：

```text
Python / package
CUDA / PyTorch
VRAM
RAM
disk
Hugging Face auth
network
model config
tensor shape
autograd
attention implementation
```

不要因为局部报错直接重装整个环境。

---

## 26.5 不要擅自修改 Diffusers 安装包

默认不要直接编辑：

```text
site-packages/diffusers/
```

优先：

- subclass
- attention processor
- hook
- wrapper
- 项目内自定义实现

只有确认公开扩展点无法满足论文实现时，才考虑 patch，并先说明。

---

## 26.6 保持实验可复现

所有关键实验尽量固定：

- seed
- model id
- diffusers version
- torch version
- prompt
- steps
- guidance
- resolution
- layout parameters

必要时记录：

```bash
pip freeze
```

但不要每次实验都重复生成庞大依赖文件。

---

# 27. Git 要求

项目使用一个 Git repository。

建议 checkpoint：

```text
initial project
environment ready
SDXL baseline
FLUX baseline
basic experiments complete
layout-control baseline
attention extraction
layout guidance MVP
paper reproduction experiments
```

原则：

- 修改前检查 `git status`
- 修改后优先 `git diff`
- 不覆盖用户未提交改动
- 不擅自使用破坏性命令
- 禁止未经允许执行：

```bash
git reset --hard
git clean -fd
```

---

# 28. 每次任务完成后的汇报格式

完成任务后简要报告：

1. 做了什么
2. 修改了哪些文件
3. 执行了哪些关键命令
4. 当前结果
5. 输出文件位置
6. 是否有警告 / 风险
7. 下一步建议

如果失败：

- 明确当前阻塞点
- 不要假装任务已经完成

---

# 29. 核心原则

这是一个学习型科研复现项目。

优先级：

```text
理解 > 黑盒自动化
可复现 > 偶然跑通
最小 baseline > 复杂工程
单变量实验 > 同时修改很多东西
先验证 > 再优化
复用环境 > 重复配置
复用模型 cache > 重复下载
项目内扩展 > 直接修改第三方库
保留已有工作 > 大规模重构
```

Codex 的职责是帮助用户完成科研代码复现，但同时必须让用户知道：

- 为什么这样做
- 修改了什么
- 代码与论文如何对应
- 当前实验能说明什么
