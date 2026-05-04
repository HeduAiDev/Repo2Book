# Research Brief: Self-Attention — 从数学公式到 vLLM Backend Dispatch

**Researcher**: researcher
**Date**: 2026-05-04
**For**: Writer — Ch01 v5 叙事参考
**本轮调研重点**：演进史、源码考古（git log）、竞品对比、关键设计决策

---

## 演进时间线

| 时间 | 事件 | 论文/PR | 关键贡献 |
|------|------|---------|---------|
| 2017.06 | Transformer 论文 | "Attention Is All You Need" (Vaswani et al.) | Scaled Dot-Product Attention：`softmax(QK^T/√d_k)V`，MHA |
| 2019 | MQA | "Fast Transformer Decoding" (Shazeer) | 所有 Q 头共享一个 KV，显存节省 ~H 倍 |
| 2021.05 | ALiBi | Press et al. | 用偏置替代位置编码 |
| 2022.05 | FlashAttention | Dao et al. | IO-aware tiling + online softmax，O(N²d)→O(N) 显存 |
| 2023.05 | GQA | Ainslie et al. | MHA↔MQA 的 sweet spot，Llama 3/Mistral 标配 |
| 2023.06 | vLLM PagedAttention | Kwon et al. (SOSP'23) | 分页 KV cache，<4% 显存浪费 |
| 2023.07 | FlashAttention-2 | Dao 2023 | Seq-length 维度并行，2x 提速 |
| 2024.03 | **vLLM 首次分离 backend** | PR #3005 | `Separate attention backends` — dispatch 模式诞生 |
| 2024.05 | MLA (DeepSeek-V2) | DeepSeek | 低秩 KV 压缩，KV cache 减少 10x+ |
| 2024.07 | FlashAttention-3 | Shah et al. | H100 异步 GEMM + FP8，3x 加速 |
| 2025.03 | Triton backend 扩展至 CUDA | f8a08cb | 原 ROCm-only 的 Triton attn 支持 NVIDIA GPU |
| 2025.05 | ROCm 拆分为独立 backend | 175811e | Triton / ROCm 关注分离 |
| 2025.06 | FlexAttention | PyTorch 2.5 | 灵活 mask+bias 组合，torch.compile 友好 |
| 2025-2026 | FlashInfer MLA, TurboQuant, etc. | 多个 PR | 社区驱动的大量 backend 涌现 |
| 2026.01 | **V1 attention 重构** | PR #31916 | 文件重整 + 接口标准化，**当前架构的起点** |
| 2026.02-04 | DFlash, Sparse MLA, FP8 KV | 持续迭代 | Perf + 新模型支持 |
| 2026.04 | TurboQuant backend | PR #38479 | 2-bit KV cache，4x 容量 |

---

## vLLM 源码考古（Git Log 深度追踪）

### 关键文件的引入时间线

```
2023 (v0.1):
  vllm/model_executor/layers/attention.py     ← 最初的 Attention 层
    注意：git log 显示最早的 attention.py 已不在当前树中
    commit 71bcaf99e: "Enable GQA support" (2023)
    commit 0580aab02: "ROCm support without flash-attn" (2023)
    commit 9090bf02e: "FP8-E5M2 KV Cache" (2023)
    commit d10f8e1d4: "Prefix Caching" (2023)

2024.03 — **第一次重大拆分**（V0 时代）:
  commit 2daf23ab0: "Separate attention backends (#3005)"
    → 这是 backend dispatch 模式的雏形！
    → 将 FlashAttn / Xformers / ROCm 分成独立 backend

2025.03 — Triton 跨平台:
  commit f8a08cb: "Enable Triton(ROCm) Attention backend for Nvidia GPUs (#14071)"
    → triton_attn.py 从 ROCm-only 扩展到 CUDA

2025.05 — ROCm 独立:
  commit 175811e: "Split triton_attn in triton-only and rocm specific backends (#24648)"
    → rocm_aiter_fa.py / rocm_attn.py 拆分出来

2026.01 — **V1 架构确立**（当前架构的起点）:
  commit 2612ba928: "[1/N][Attention] Restructure attention: move files (#31916)"
    → vllm/v1/attention/ 目录创建
    → backend.py, selector.py, backends/ 文件标准化

2026.01-04 — 快速扩张期:
  commit f2c47886f: "FlashInfer Sparse MLA backend (#33451)"
  commit f4b42df04: "TurboQuant: 2-bit KV cache (#38479)"
  commit 914d0464c: "Unify 2D/3D kernels in triton_unified_attention (#40631)"
  commit fd74c90d9: "Independent drafter attention backend selection (#39930)"
  commit b7a260502: "Make Attention Backend Auto-Selection Batch-Invariance-Aware (#40193)"

2026.04-05 — 最新:
  commit f3fef1235: "Abstract the MLA prefill backends and eliminate cuDNN (#32623)"
  commit 4d51588e2: "DeepSeek V4 Rebased (#40860)" — 复杂的 attention dispatch
```

### Backend 数量演变

```
V0 早期 (2023):  1 backend  = 直接 FlashAttention
V0 后期 (2024):  3 backends = FlashAttn + Xformers + ROCm
V1 初期 (2026):  ~15 backends = 完整 MLA 矩阵 + Mamba + Hybrid
V1 当前 (2026.05): 20+ backends (含 sparse MLA / Mamba / linear / GDN / short-conv)
```

### 重要发现：Backend Dispatch 不是"设计出来的"

从 git log 看，这是一个**有机演化过程**：
1. PR #3005 (2024.03): "我们需要支持多个 attention 后端" → 引入抽象基类
2. PR #14071 (2025.03): "Triton kernel 在 NVIDIA 上也能跑" → 跨平台需求驱动力
3. PR #24648 (2025.05): "ROCm 用户需要自己的优化路径" → 按硬件拆分
4. PR #31916 (2026.01): "文件太多了，需要重整" → V1 目录结构
5. 之后每次新硬件/新模型 → 加一个新 backend → registry 持续膨胀

**核心驱动力是社区多样性，不是架构哲学。**

---

## 关键设计决策

### 决策 1: 为什么 vLLM 把 attention 拆成 Backend Dispatch 模式？

```
问题：如何同时支持？
├── 3 种硬件: NVIDIA CUDA / AMD ROCm / Intel CPU+XPU
├── 5+ 种 kernel 库: FlashAttention 2/3/4, FlashInfer, Triton, AITER, FlexAttention
├── 10+ 种注意力变体: MHA, GQA, MQA, MLA, Sparse MLA, Sliding Window, ALiBi, ...
└── 4 种量化: FP16/BF16, FP8 KV, TurboQuant (2-bit), NVFP4

不可能用一个 kernel 支持所有组合 → 矩阵太大
```

**vLLM 的三层 dispatch 架构**：

```
Layer 1 — 平台层 (platform.get_attn_backend_cls):
  CUDA → FLASH_ATTN (default)
  ROCm → ROCM_ATTN or ROCM_AITER_FA
  CPU  → CPU_ATTN

Layer 2 — 匹配层 (selector.py → AttentionSelectorConfig):
  输入: (head_size, dtype, kv_cache_dtype, block_size, use_mla, has_sink, ...)
  逻辑: 逐个 backend 调用 validate_configuration() 过滤
  输出: 第一个完全匹配的 backend

Layer 3 — 注册层 (AttentionBackendEnum + register_backend):
  20+ 内置 backends
  第三方可热注册覆盖（装饰器 @register_backend）
```

**核心 trade-off**：
- 优点：每个 backend 自描述"我支持什么"，新增 backend 不影响已有代码；硬件厂商（AMD, Intel）可以提交自己的 backend 而不修改核心逻辑
- 缺点：`validate_configuration` 有 10+ 个参数，每新增一个特性就要所有 backend 更新；selector 逻辑日益复杂

### 决策 2: 为什么 scale 是构造函数参数而非计算值？

**直接原因**: 不是所有模型都用 `head_dim ** -0.5`。

```
标准 MHA:    scale = head_dim ** -0.5           (Transformer 原始公式)
ALiBi:       scale = head_dim ** -0.5 + bias    (slope 加到 attention score)
Gemma:       scale = head_dim ** -0.5, logits_soft_cap = 50.0
DeepSeek V3: scale 有自己的公式（与 qk_head_dim 相关）
GQA:         scale 理论上可以不同，实践中通常不变
```

**代码流追踪**（从最外层到最内层）：

```python
# 1. 模型定义层 (model_executor/models/gemma.py:L156)
self.scaling = self.head_dim**-0.5     # 模型知道自己的 head_dim

# 2. vLLM Attention 层 (model_executor/layers/attention/attention.py:L193)
def __init__(self, ..., scale: float, ...):  # 从模型定义接受 scale
    ...

# 3. 传给具体实现 (attention.py:L344-348)
impl_cls = self.attn_backend.get_impl_cls()
self.impl = impl_cls(num_heads, head_size, scale, ...)

# 4. Backend 存储 (triton_attn.py:L469)
self.scale = float(scale)               # 所有 backend 都这样存储
```

**为什么不在 backend 内部计算？**
1. **单一职责**: kernel 只负责"以这个 scale 执行 attention"，不负责"这个模型应该用什么 scale"
2. **模型定制**: 某些模型（如 DeepSeek）需要特殊 scale 公式，kernel 不应该知道所有模型的细节
3. **一次性计算**: `scale` 是常量（不会在推理中变化），在 `__init__` 计算一次即可，不需要每次 forward 重算
4. **资源分离**: 修改 scale（如模型 fine-tune 后改配置）只需改模型层，不需要修改 backend 代码

---

## 竞品对比

### TensorRT-LLM attention 实现

| 维度 | vLLM | TensorRT-LLM |
|------|------|-------------|
| **Backend 选择** | 运行时 selector + 注册表 | 编译时 graph capture → TRT engine |
| **Attention kernel** | 多种 kernel 共存（FA/FlashInfer/Triton） | 统一 TRT plugin，编译时融合 |
| **扩展方式** | 注册新 backend（Python 代码） | 写 TRT plugin（C++/CUDA） |
| **KV Cache** | PagedAttention (虚拟分页) | 类似 paged KV cache |
| **Prefill/Decode** | Unified kernel (2D/3D) 或分离 | TRT engine 自动调度 |
| **硬件** | CUDA + ROCm + CPU + XPU | NVIDIA only |
| **灵活性** | 高（运行时切换、热注册） | 低（编译后固定） |
| **性能** | 接近 peak（hand-tuned Triton/CUDA） | 接近 peak（TRT 编译优化） |
| **Chunked Prefill** | 支持（chunked_prefill_paged_decode.py） | 类似机制 |

**关键差异**: TRT-LLM 用编译时优化（graph optimization + kernel auto-tuning），vLLM 用运行时灵活性（selector 可以按 layer 选不同 backend）。TRT-LLM 对 NVIDIA 硬件有极致优化但封闭；vLLM 更开放但有 dispatch 开销。

### SGLang attention 实现

| 维度 | vLLM | SGLang |
|------|------|-------------|
| **Backend 选择** | AttentionBackendEnum 注册表 | FlashAttention backend 为主 + sgl-kernel |
| **KV Cache** | PagedAttention | **RadixAttention**（前缀树共享） |
| **Prefill/Decode** | 统一 AttentionLayer | **分离**：extend/prefill/decode 不同路径 |
| **Tree Attention** | 通过 TreeAttn backend | 原生支持 tree attention |
| **代码结构** | 3 层抽象 (Backend/Builder/Impl) | 2 层抽象 (Backend/Impl)，更扁平 |
| **MLA 支持** | 6 个 MLA backend | sgl-kernel 中的 MLA |

**关键差异**: SGLang 的 RadixAttention 是 KV cache 管理的不同哲学：vLLM 用虚拟内存分页，SGLang 用前缀树自动共享公共前缀。对于有 system prompt 的场景，SGLang 的共享更自然。vLLM 也有 prefix caching 但基于 hash 匹配而非树结构。

### vLLM 相比竞品的独特取舍

1. **社区驱动 > 极致性能**: 放弃编译时优化换取了多厂商贡献的便利性（AMD/Intel/XPU 都可以加 backend）
2. **接口富表达 > 接口简洁**: `validate_configuration` 有 10+ 个参数，但能精确描述每个 backend 的能力边界
3. **KV Cache 统一抽象 > 专用优化**: PagedAttention 所有 backend 共享同一套 block table 接口，但牺牲了某些专用 kernel 的可能优化

---

## 本章范围界定

### 应该深度覆盖的内容
1. Scaled Dot-Product Attention 的完整数学推导 + √d_k 方差分析
2. MHA/GQA/MQA 的压缩比量化分析（配合显存计算）
3. 从朴素 attention → online softmax → tiled attention 的渐进推导
4. vLLM 的 Backend Dispatch 模式：为什么有 20+ backend，selector 如何工作
5. Attention Mask 类型（Causal/Padding/Sliding Window）

### 应该点到为止的内容（留给后续章节）
- Block table / PagedAttention 的原理（Ch03 主题）
- KV Cache 的分配/管理（Ch02 主题）
- FlashAttention-2/3 的架构差异（Ch03 覆盖）
- Batch-level 调度（Ch04 覆盖）

### 不应涉及的内容
- MLA (Multi-head Latent Attention) — 属于 Part 4 专家模型主题
- Cascade Attention / DCP — 高级优化，Part 5
- Speculative Decode 中的 attention — Part 5
- Mamba/Linear Attention 等非 Transformer attention — Part 5

---

## 给 Writer 的建议

### 最重要的 intuition
Self-attention 不是一个"已经解决的算法"——它在持续演进。vLLM 的 backend dispatch 模式不是哲学偏好的产物，而是**被社区多样性驱动的生存策略**：20+ backend，4 种硬件，无数模型变体。读者应该理解：这 20 个 backend 的内核是同一个数学原理，区别只在于在什么硬件上、用什么技巧来加速。

### 最适合零基础读者的切入角度
从**规模问题**讲起：
1. 假设 seq_len=2048, d_head=128, fp16 → 一个 head 的 attention matrix = 8MB
2. 40 层 × 32 头 → **10.24 GB**（仅 attention matrices！远超过 HBM）
3. 这就是为什么需要 IO-aware computing → FlashAttention 的 tiling
4. 自然的引出：vLLM 不是造 attention kernel，而是**选择+管理**最优 kernel

### 容易混淆的概念
1. **attention 的 softmax_scale vs KV quant 的 _k_scale**: 前者是 `1/√d_k`（数学），后者是 FP8 量化缩放因子（工程）。两个都叫 "scale" 但完全无关
2. **Backend vs Implementation**: `AttentionBackend` 是类型工厂（"我是 FlashAttn"），`AttentionImpl` 是实例（"这一层的具体参数"）。一对多关系
3. **Metadata vs MetadataBuilder**: Builder 是工厂，batch 调用 `build()` 生成 batch-specific metadata
