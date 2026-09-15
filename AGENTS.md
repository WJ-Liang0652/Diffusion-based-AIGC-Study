# AGENTS.md

## 1. 当前主线

当前项目主线不是继续系统学习 FLUX，而是：

> 在 FLUX 上完成 Training-Free Layout Control / Backward Guidance 的方法级完整复现，并分析 block、timestep、attention 机制对 layout control 的影响。

实验优先，源码按需阅读。不要为了完整理解 FLUX 工程实现而延迟实验。

## 2. 当前已有状态

项目根目录：

```text
diffusion-reproduction/
```

已有：

```text
01_diffusers_basics/
02_layout_control/
```

后续新建：

```text
03_flux_layout_control/
```

### SDXL 当前状态

`02_layout_control/` 已完成较长实验链，包括 baseline、explicit denoising、cross-attention probe、attention heatmap、layout objective、gradient probe、latent update、iterative optimization、backward guidance、full generation、scheduler alignment。

当前应视为：

> 可运行、肉眼能看到一定控制效果，但尚未完成系统消融和定量评价的半成品 baseline/reference。

因此：
- 不继续优先补 SDXL；
- 不把 SDXL 称为完整复现；
- 后续把它作为 FLUX 的算法参考；
- `02_layout_control/` 默认保留，不大规模修改。

## 3. 模型策略

### 开发 / Debug

使用：

```text
black-forest-labs/FLUX.1-schnell
```

用途：
- baseline
- explicit sampling
- attention probe
- token-space mapping
- layout loss
- autograd
- gradient probe
- backward guidance MVP
- 快速 debug

### 正式实验

使用：

```text
black-forest-labs/FLUX.1-dev
```

用途：
- 正式 qualitative results
- 多 seed
- timestep / block / layer 分析
- guidance strength sweep
- quantitative evaluation
- SDXL vs FLUX comparison

正确顺序：

```text
schnell 跑通机制
→ dev baseline
→ dev gradient probe
→ dev backward guidance
→ dev 系统实验
```

## 4. FLUX.1-dev 下载规则

允许下载 `FLUX.1-dev`。

但大型模型下载前必须先向用户确认，并先检查：

```bash
pwd
df -h
echo "$HF_HOME"
du -sh "$HF_HOME" 2>/dev/null || true
```

同时确认：
- Hugging Face 登录状态
- gated model 权限
- License / access 条件
- 剩余磁盘空间

优先通过 Diffusers：

```python
FluxPipeline.from_pretrained(...)
```

下载实际需要的组件。

不要无筛选下载完整仓库，避免同时保存：

```text
单文件 checkpoint
+
Diffusers transformer shards
```

当前约 100GB 空闲空间足够后续工作，但必须控制大型中间数据。

## 5. 显存策略

服务器：

```text
GPU: RTX 4090 24GB
RAM: 90GB
```

### 普通 FLUX.1-dev 推理

优先：
- batch size = 1
- BF16
- CPU offload
- 必要时 sequential / group offload

不要默认把所有组件同时 `.to("cuda")`。

### Backward Guidance

Backward Guidance 需要 autograd，因此显存风险明显高于普通 inference。

必须逐级验证：

```text
dev baseline
→ attention probe
→ layout loss
→ single-step gradient
→ short guidance
→ full guidance
```

优先策略：
1. 冻结模型参数；
2. 只对必要 latent / state 求梯度；
3. text encoder 不参与 backward；
4. VAE 不参与 guidance backward；
5. 限制 target block / layer；
6. 不保存全部 raw attention；
7. inner loop 后及时释放 graph；
8. 检查无意的 `retain_graph=True`；
9. 必要时降低开发分辨率；
10. 再考虑更强 offload / checkpointing；
11. 量化只作为后手，正式结果尽量保持 BF16。

不要因为一次 OOM 就重装环境或更换服务器。

## 6. FLUX 实验目录

全部 FLUX 新工作放：

```text
03_flux_layout_control/
```

建议逐步形成：

```text
03_flux_layout_control/
├── README.md
├── src/
├── configs/
├── experiments/
├── outputs/
│   ├── baseline/
│   ├── attention/
│   ├── guidance/
│   ├── sweeps/
│   └── evaluation/
└── notes/
```

不要一次创建大量空文件。

## 7. 实验主路线

严格围绕：

```text
只读环境检查
→ schnell baseline
→ explicit sampling
→ attention / token-space 最小验证
→ spatial mapping
→ layout loss
→ gradient probe
→ backward guidance MVP
→ dev baseline
→ dev backward guidance
→ system ablation
→ quantitative evaluation
→ SDXL vs FLUX comparison
```

### Baseline
要求固定 model、seed、prompt、resolution、steps、guidance，并记录 runtime / VRAM。

### Explicit Sampling
先与官方 Pipeline 尽量对齐；在对齐前不要加入 layout guidance。

### Attention / Token-Space
只回答当前实验所需问题：
- text token 如何进入 Transformer；
- image token 如何形成；
- Double Stream 如何交互；
- Single Stream 如何交互；
- Q/K/V 如何组织；
- image token 如何映射为空间网格；
- 哪个 attention quantity 适合构造 layout objective。

只做最小必要验证，不遍历所有 block。

### Layout Loss
先验证：
- 数值稳定；
- token index 正确；
- bbox / mask 正确；
- spatial reshape 正确；
- 不同 target region 能产生合理差异。

### Gradient Probe
必须确认：
- gradient 非 None；
- gradient 非全零；
- model parameter 不累计 gradient；
- 单步 update 有可解释变化；
- 无明显 graph 泄漏。

### Backward Guidance MVP
先在 schnell 上完成：

```text
baseline
vs
guided
```

要求看到稳定的空间位置变化，而不是只发生颜色、亮度或纹理变化。

### dev 正式实验

顺序：

```text
dev baseline
→ dev attention probe
→ dev layout loss
→ dev gradient probe
→ dev guidance
```

随后再做：
- strength sweep
- timestep sweep
- block / layer sweep
- prompt
- layout
- seed
- multi-object

## 8. 研究目标

最终至少回答：

1. SDXL 原方法依赖的 cross-attention 在 FLUX 中如何重新映射？
2. FLUX 中哪个 text-image spatial signal 最适合构造 layout objective？
3. Double Stream 和 Single Stream 哪些 block 更适合 layout guidance？
4. 哪些 timestep 对 layout 最敏感？
5. schnell 与 dev 的 guidance 行为有什么差异？
6. 哪些部分仍属于原论文机制？
7. 哪些部分是 FLUX-specific redesign？
8. 控制效果、稳定性、runtime 和 VRAM 开销如何？

未经实验验证，不要声称“完全等价”。

## 9. Backward Guidance 优先级

主线：

```text
Backward Guidance
```

Forward Guidance：
- 不与主线并行开发；
- 不阻塞当前实验；
- Backward Guidance 完整跑通后再决定是否补；
- 如后续用于消融或论文对照，再实现。

## 10. 源码阅读规则

源码学习只服务于实验。

优先关注：

```text
FluxPipeline
FluxTransformer2DModel
FluxTransformerBlock
FluxSingleTransformerBlock
FluxAttention / FluxAttnProcessor
```

只有遇到当前实验问题时才继续深入。

当前服务器实际安装版本优先于 GitHub `main`。

## 11. Diffusers 修改规则

默认：

```text
site-packages/diffusers/
```

只读。

优先使用：
- hook
- attention processor
- wrapper
- subclass
- 项目内自定义实现

如果必须 patch Diffusers：
1. 先说明原因；
2. 说明改动位置；
3. 说明风险；
4. 等用户确认。

不要为了快速跑通直接修改第三方安装包。

## 12. Attention 数据保存

默认不保存：

```text
所有 timestep
× 所有 block
× 所有 head
× 完整 attention matrix
```

优先保存：
- selected token map
- head-averaged heatmap
- layer / timestep statistics
- layout loss
- gradient norm
- bbox mass ratio
- 少量必要 tensor

保存大型 tensor 前先估算磁盘占用。

## 13. 实验记录

每个重要实验至少记录：

```text
model
diffusers version
torch version
dtype
offload strategy
prompt
seed
resolution
steps
guidance
layout
target token
target block / layer
target timestep
layout guidance strength
inner-loop count
runtime
peak VRAM
output path
```

正式实验不要只保存图片。

## 14. Codex CLI 权限

### 可以直接连续执行

无需每一步确认：
- 读取代码
- 只读环境检查
- 创建普通实验脚本
- 运行低成本实验
- 修普通 Python bug
- 修 tensor shape bug
- 查看 Diffusers 源码
- 新建 `03_flux_layout_control/` 中普通文件
- 保存日志
- `git status`
- `git diff`

普通流程可以：

```text
读代码
→ 写脚本
→ 运行
→ 修 bug
→ 重跑
→ 保存结果
→ 汇报
```

### 必须先确认

以下操作必须先向用户说明：
- 下载大型模型
- 下载 FLUX.1-dev
- 安装大型依赖
- 升级 / 降级 PyTorch
- 升级 / 降级 Diffusers
- 修改 CUDA 环境
- 修改 `site-packages`
- 删除历史文件
- 大规模移动旧文件
- 覆盖 SDXL 已有结果
- 高成本批量 GPU 实验
- 大规模 benchmark
- 使用量化作为正式方案
- 更换 GPU / 服务器
- 出现明显不同的算法路线需要选择
- 破坏性 Git 操作

## 15. Git 规则

每次修改前：

```bash
git status
```

修改后：

```bash
git diff
git status
```

禁止未经允许执行：

```bash
git reset --hard
git clean -fd
```

不要覆盖用户未提交改动。

## 16. 环境与报错规则

优先复用现有环境。先检查，再安装。

不要因为局部报错直接：
- 重装 CUDA
- 重装 PyTorch
- 新建多套近似环境
- 升降级核心包

报错处理顺序：
1. 看完整 traceback；
2. 分类；
3. 找最小根因；
4. 做最小修改；
5. 重新验证。

优先检查：

```text
Python
package version
CUDA / PyTorch
VRAM / RAM / disk
HF auth / network
dtype / device
tensor shape
autograd
offload
attention
scheduler
token index
spatial reshape
```

## 17. 通过条件

### Baseline
- 可重复；
- fixed seed 稳定；
- 参数完整记录。

### Explicit Sampling
- 与 Pipeline 尽量对齐。

### Attention Probe
- shape 正确；
- token index 正确；
- spatial reshape 正确；
- heatmap 有合理语义。

### Layout Loss
- 数值稳定；
- 能区分目标区域。

### Gradient Probe
- gradient 存在且非零；
- 参数无意梯度不存在；
- 单步 update 有合理变化。

### Backward Guidance MVP
- baseline vs guided 有稳定 layout 差异；
- 不只是颜色 / 亮度变化；
- 多 seed 下具有一定重复性。

### 方法级完整复现
至少包括：
- 多 prompt；
- 多 layout；
- 多 seed；
- strength / timestep / block 消融；
- 代表性定量评价；
- SDXL / FLUX 比较；
- 完整实验记录。

## 18. 当前研究优先级

```text
正确的实验问题
>
方法级完整复现
>
可复现
>
最小可验证实验
>
系统消融
>
定量评价
>
源码按需理解
>
工程美观
```

同时坚持：

```text
证据 > 猜测
实验 > 为学习而学习
单变量 > 多变量混杂
最小修改 > 大规模重构
项目内扩展 > 修改第三方库
保留已有工作 > 重写历史代码
```

## 19. 每轮汇报格式

Codex 每轮完成后简要报告：

1. 当前验证的问题
2. 做了什么
3. 修改了哪些文件
4. 执行了哪些关键命令
5. 结果是否通过
6. 输出路径
7. runtime / VRAM 是否异常
8. 当前风险 / 假设
9. 下一步建议

失败时明确阻塞点，不把未验证结果描述成成功。

## 20. 已确定、不重复询问的方向

除非实验事实迫使修改，否则以下方向已经确定：

- FLUX layout control 是当前主线
- schnell 用于开发 / debug
- dev 用于正式实验
- 允许后续下载 dev
- Backward Guidance 主线
- Forward Guidance 后补
- 方法级完整复现 + 代表性定量实验
- 不要求一开始重跑全部 benchmark
- 研究 block / timestep / attention 差异
- SDXL 当前只是半成品 reference
- `02_layout_control/` 保留
- `03_flux_layout_control/` 用于全部 FLUX 新实验
- Codex 可自主完成普通开发流程
- 大模型下载 / 核心环境变化 / 高成本实验必须确认
- dev 普通推理优先 BF16 + offload
- dev Backward Guidance 按高 OOM 风险处理
- 100GB 空闲磁盘足够，但限制 raw attention 数据
