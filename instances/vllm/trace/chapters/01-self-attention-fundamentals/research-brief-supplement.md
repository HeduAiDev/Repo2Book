# Supplementary Research: Self-Attention 演进史 "为什么不那样做"

**To**: Writer (writer-2)
**From**: Researcher
**Date**: 2026-05-04
**Builds on**: research-brief.md (primary)

---

## 1. 概念起源：Attention 的三种计算方式及其命运

### 1.1 Bahdanau Attention (2014) — 加性注意力

```
score(s_{t-1}, h_j) = v^T tanh(W_a s_{t-1} + U_a h_j)
context_t = Σ_j α_{tj} h_j  (softmax over scores)
```

**核心问题**：RNN seq2seq 的 encoder 隐状态是变长向量，decoder 需要一个定长 context。
Bahdanau 的解决方案是"学习如何加权"——用一个小型 FFN（`v^T tanh(W·s + U·h)`）来算对齐分数。

**为什么后来被点积取代？**
- **速度**：加性 attention 需要计算一个二次的矩阵乘法（decoder 隐状态 × encoder 隐状态），计算量是 O(T_s × T_t × d²)。点积只需要 O(T_s × T_t × d)。
- **并行性**：加性 attention 无法直接用 matmul 实现（需要每对 (i,j) 单独过 FFN），而点积直接是 Q@K^T。
- **精度**：当 d 较小时，加性 attention 略优于点积（因为可学习的参数更多）；但 d 增大后差距消失。

**vLLM 完全不考虑加性 attention**，因为其全部模型都是 Transformer decoder，且 GPU 上矩阵乘法是最高效的操作（Tensor Core 可达数十 TFLOPS）。

### 1.2 Luong Attention (2015) — 点积/双线性/加性三选一

Luong 在同一篇论文中比较了三种 score：
```
dot:      score(h_t, h_s) = h_t^T h_s
general:  score(h_t, h_s) = h_t^T W_a h_s    (双线性, W_a 可学习)
concat:   score(h_t, h_s) = v^T tanh(W_a [h_t; h_s])  (加性)
```

**结论**：对于"全局 attention"（对全部 encoder states），dot 最快且效果足够好。"local attention"（窗口）则需要 general 来弥补窗口限制。

**与 vLLM 的关系**：
- vLLM 本质使用 dot-product（Q@K^T），无 W_a 矩阵。这是 Transformer 的标准选择。
- W_a 可以被吸收到 Q 或 K 的投影矩阵中（如果 Q,K 共享同一个投影矩阵的话，W_a 无额外表达能力）。
- 注意：vLLM 的 QKV 是分开投影的（`QKVParallelLinear` 输出 [Q|K|V] concatenated），所以 W_Q 与 W_K 各自独立，双线性的表达能力已隐含在内。

### 1.3 Transformer (Vaswani et al., 2017) — Scaled Dot-Product

```
Attention(Q,K,V) = softmax(QK^T / √d_k) V
```

**为什么是 √d_k？——三层次的递进式理解**

**层次 1: 方差直觉**（论文 3.2.1 脚注）：
假设 Q 和 K 的每个元素独立同分布，均值为 0，方差为 1。则 Q·K（两个 d_k 维向量的内积）的方差为 d_k。除以 √d_k 后，方差归 1。这样 softmax 的输入不会因为 d_k 增大而出现极大值（→ gradient vanishing）。

**层次 2: 精确推导**：
```
设 q_i, k_i ~ N(0,1) 独立，则:
(Q@K^T)_{ij} = Σ_{m=1}^{d_k} q_{i,m} * k_{j,m}
Var(每一项) = Var(q) * Var(k) = 1
Var(内积) = d_k * 1 = d_k
Var(内积 / √d_k) = 1
```
所以当 softmax 输入的标准差 ≈ 1 时，输出的分布才"合理"（不会退化为 one-hot）。

**层次 3: 如果不用 √d_k 会怎样？**
- d_k=64 时，softmax 输入的元素绝对值可达 ~10-15（无缩放下）。exp(15) ≈ 3.3×10^6，softmax 几乎变成 one-hot → gradient 为零 → 训练失败。
- 一些早期实现用 `1/d_k` 而非 `1/√d_k`（如归一化到 1/d_k），但这会导致 softmax 输入过小，分布退化为均匀分布 → 无注意力效果。
- `1/√d_k` 是唯一让 softmax 输入方差为 1 的归一化系数。

**vLLM 代码体现**：
```python
# llama.py:~152
self.scaling = self.head_dim**-0.5
# 传递到 Attention.__init__(scale=self.scaling)
# 最终: softmax_scale=self.scale 传给 FA/Triton kernel
```

---

## 2. MHA → MQA → GQA 的演进："不是那样的"逻辑链

### 2.1 MHA (Multi-Head Attention, Vaswani 2017)

```
MHA 的线性代数本质：将 d_model 维空间切成 h 个子空间，
在每个子空间独立做 attention，最后拼接投影。

可视为 h 个并行的低秩 attention（秩 ≈ d_model/h）
```

**问题**：在推理（自回归生成）时，每个 token 都需要缓存所有头的 K 和 V。对于 Llama 1 65B：80 层 × 64 头 × 128 维 × fp16 = 约 1.3 GB/token。2048 token 上下文 → 2.6 TB KV cache —— 不可行。

### 2.2 MQA (Multi-Query Attention, Shazeer 2019)

```
所有 Q 头共享 1 对 KV 头
KV cache: 从 H × (d_k + d_v) 降到 1 × (d_k + d_v)
显存节省: ~H 倍
```

**论文的核心洞察（需要 Narrative 传达）**：
Shazeer 发现——在自回归推理中，KV cache 的显存是瓶颈，而 attention 计算的开销相对较小。MQA 用"所有 Q 头共享 KV"来减少显存，但牺牲的是多头注意力的表达能力（每个头看到的是相同的 K、V，只是 query 不同）。

**为什么精度损失"太大"？**
MQA 的直觉是"多个 Q 头可以有略微不同的关注模式"。共享 KV 后，所有头被迫关注相同位置——精度损失约 2-5%（在 MMLU 等 benchmark 上），但这对于高性能模型不可接受。

### 2.3 GQA (Grouped Query Attention, Ainslie et al. 2023)

```
把 Q 头分成 G 组，每组共享一对 KV 头
G = 1 → MQA; G = H → MHA
实际: G ∈ [4, 8] 是最常见的区间
```

**核心洞察**：Ainslie 等人通过一种训练策略——从 MHA checkpoint 开始，将 KV 头按组"平均池化"来初始化 GQA——证明 GQA 可以接近 MHA 的质量，同时仅需 MQA 的 1/G 的 KV cache。

**为什么是分组，不是插值？**
如果在 MHA 和 MQA 之间做线性插值（比如混合 MHA 的 KV 和 MQA 的单 KV），效果很差——因为不连续。分组是唯一能保持"每个 KV 头都有训练信号"的结构化压缩方式。

### 2.4 主流模型的 GQA 选择

| 模型 | num_heads | num_kv_heads | 分组数 | 压缩比 | 原因 |
|------|-----------|-------------|--------|--------|------|
| Llama 1 (65B) | 64 | 64 | 1 (MHA) | 1:1 | 推出时 GQA 尚未流行 |
| Llama 2 (70B) | 64 | 8 | 8 | 8:1 | 每个 KV 头服务 8 个 Q 头 |
| Llama 3 (8B) | 32 | 8 | 4 | 4:1 | 小模型可用更低压缩比 |
| Llama 3 (70B) | 64 | 8 | 8 | 8:1 | 大模型加速节省更显著 |
| Mistral (7B) | 32 | 8 | 4 | 4:1 | 与滑动窗口配合 |
| Gemma 2 (27B) | 32 | 16 | 2 | 2:1 | 高精度，低压缩 |

**趋势**：大模型倾向更高压缩比（KV cache 显存与层数成正比），小模型可用低压缩比（本身 KV cache 已经较小）。

**vLLM 代码中的 GQA 处理**（triton_decode_attention.py）：
```python
# L746: kv_group_num = q.shape[1] // v_buffer.shape[-2]
# kv_group_num == 1 → 走 MHA 路径（简单）
# kv_group_num > 1  → 走 GQA/MQA 路径（需要 grouped 特殊处理）
```

---

## 3. Attention 计算的工程化演进

### 3.1 FlashAttention (Dao et al., 2022) — 硬件意识

**解决的问题**：GPU 的瓶颈不是 FLOPS，是内存带宽。
```
HBM bandwidth: ~1.5 TB/s (H100)
SRAM (shared memory): ~19 TB/s
朴素 attention: QK^T 完整存储在 HBM → O(n²) HBM reads
FlashAttention: 逐块计算，保持在 SRAM → O(n) HBM reads
```

**为什么 "exact" 是关键词？** Dao 证明 tiled softmax 在数学上完全等于朴素 softmax（不是近似）。证明依赖于 online softmax 的更新公式的精确性。

### 3.2 FlashAttention-2 (Dao, 2023) — 更好的并行

主要改进：
1. **减少非 matmul FLOPS**：重新排列循环嵌套（Q outer loop → KV outer loop），让每个 thread block 处理更长的序列
2. **Sequence 维度并行化**：每个 thread block 处理序列的一部分
3. **Warp-level 通信优化**：减少 intra-block reduction 的开销

### 3.3 FlashAttention-3 (Shah et al., 2024) — Hopper 架构专用

```
H100 SM 的新特性：
- wgmma (warp group matrix multiply accumulate): 异步 GEMM
- TMA (tensor memory accelerator): 硬件 DMA
- 较低的 FP8 tensor core 延迟
FA3 利用 wgmma 实现 3x decode 加速 (vs FA2)
```

### 3.4 vLLM 的 backend 抽象 —— 为什么是这个架构？

```
vllm/v1/attention/
├── backend.py         # AttentionBackend 基类
├── selector.py        # get_attn_backend() 运行时选择
├── backends/
│   ├── flash_attn.py  # FA2/FA3/FA4 (CUDA, diverse hardware)
│   ├── triton_attn.py # Triton (cross-platform)
│   ├── flashinfer.py  # FlashInfer (specialized decode)
│   ├── flex_attention.py  # FlexAttention (PyTorch 2.0+)
│   └── ...             # ROCm, XPU, CPU backends
└── ops/
    ├── triton_unified_attention.py  # ★ 关键：2D/3D 统一 kernel
    ├── triton_decode_attention.py   # PagedAttention decode
    └── ...
```

**vLLM 的选择**：不选"最优 kernel"——选"最优 kernel 选择器"。
- 原因 1: CUDA FA3 在 H100 上最快，但不支持 AMD
- 原因 2: Triton 通用性最强，但不如手工 FA3
- 原因 3: FlashInfer 专精 decode 场景
- 结果: 运行时 selector 根据 hardware + dtype + head_size + 其他特征 自动选择

这是典型的"策略模式"——每种 backend 都实现 `AttentionImpl.forward()` 接口。

---

## 4. vLLM 的独特设计选择 —— "为什么不那样做"

### 4.1 为什么 scale 是构造参数而非 Attention 内部计算？

**vLLM 的做法**：
```python
# llama.py:~152 — 模型文件中计算
self.scaling = self.head_dim**-0.5

# attention.py:~348 — 通过构造函数传入
self.impl = impl_cls(..., scale, ...)

# flash_attn.py:~613 — backend 保存
self.scale = float(scale)
```

**为什么不直接在 Attention.forward() 中算？**
1. **scale 是模型的属性，不是 attention 算子的属性**。不同模型可能使用不同的 scale 公式（e.g., DeepSeek 的某些变体可能修改 scale）。
2. **性能**：构造函数中计算一次（float），forward 中直接使用。如果每次 forward 都算 `head_dim**-0.5`，每秒数十万次的调用会有不可忽视的 CPU 开销。
3. **分离关注**：Attention 类只负责"执行 attention"——它不关心 QKV 是如何来的，也不关心 scale 是如何定义的。这符合单一职责原则。

### 4.2 为什么 QKV 投影在模型文件中而非 Attention 类？

**vLLM 的做法**：
```python
# llama.py — LlamaAttention.forward()
def forward(self, positions, hidden_states):
    qkv, _ = self.qkv_proj(hidden_states)    # ← 在模型文件中
    q, k, v = qkv.split([...], dim=-1)       # ← split 也在模型文件中
    q, k = self.rotary_emb(positions, q, k)   # ← RoPE 也在模型文件中
    attn_output = self.attn(q, k, v)           # ← 只把 projected QKV 传给 Attention
    output, _ = self.o_proj(attn_output)       # ← output projection 也在模型文件中
```

**为什么不把 QKV projection 放进 Attention？**
1. **不同模型有不同的投影策略**：
   - Llama: QKV 合并为 `QKVParallelLinear`（一次 matmul 同时输出 Q,K,V）
   - DeepSeek V3: MLA 需要压缩投影 + RoPE 分离
   - Gemma: 可能有不同的 head_dim
2. **TP (Tensor Parallelism) 由模型控制**：不同的模型有不同方式的 TP 切分。`QKVParallelLinear` 负责在初始化时正确处理 TP。
3. **backbone-model 分离**：Attention 类是完全 backbone-agnostic 的——它接收 (num_tokens, num_heads, head_size) 的 Q,K,V 张量，执行 attention，返回 output。这种设计让任意模型（Llama, Mistral, Gemma, Phi...）都可以复用同一个 Attention 核心。
4. **对比 PyTorch 的 nn.MultiheadAttention**：PyTorch 内置的 MHA 包含 `in_proj_weight`（将输入同时投影到 Q,K,V），这让模型文件更简洁——但一旦模型有特殊要求（如 GQA 的不同 K 维度），就需要绕过 PyTorch 的默认实现。vLLM 吸取了这个教训——把投影交给模型，保持 Attention 干净。

### 4.3 为什么 vLLM 从不创建显式 mask tensor？

**vLLM 的做法**（triton_attention_helpers.py:L284）：
```python
# Inside the Triton kernel — NEVER as a PyTorch tensor
seq_mask = seq_offset[None, :] <= query_abs_pos  # causal
seq_mask &= ((query_abs_pos - seq_offset) < SLIDING_WINDOW)  # sliding window
```

**为什么不分配 (seq_len, seq_len) 的 mask tensor？**
1. **显存灾难**：seq_len=131072 时，mask tensor 是 131072² = 17B elements × fp16 = 34 GB。即使 bool 也是 17 GB。而 vLLM 的目标是"batch_size=256, seq_len>128K"——mask 本身就会占满 H100 的 80GB。
2. **不必要的 HBM 访问**：mask 在 attention kernel 中读取一次就丢弃。在 kernel 内部用算术生成 mask（只需 seq_offset 和 query_abs_pos 两个向量）比从 HBM 读一个全尺寸 mask tensor 更快。
3. **fused 操作**：mask 的生成和 attention 的计算在同一个 kernel 中，可以被编译器融合优化。

**对比 PyTorch 的常见做法**：
```python
# ❌ PyTorch 常见做法 — vLLM 永远不会这样做
causal_mask = torch.triu(
    torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1
)  # 全 materix in HBM!

# ❌ 即使使用 boolean mask 也是 ~seq_len² 内存
# ❌ `torch.nn.functional.scaled_dot_product_attention` 的 is_causal=True
#    也要在 C++ kernel 内部生成 mask，但 API 层面已经避免了显式 tensor
```

**vLLM 的 mask 策略总结**：
| Mask 类型 | 实现方式 | 显存 |
|-----------|---------|------|
| Causal | `seq_offset <= query_abs_pos` (kernel 内算术) | O(1) |
| Sliding Window | `query_abs_pos - seq_offset < W` (kernel 内算术) | O(1) |
| Padding | `query_start_loc` + `seq_lens` 间接控制循环范围 | O(batch) |
| PrefixLM (bidirectional) | `mm_prefix_range_ptr` 只存储范围边界 | O(batch × ranges) |
| Chunked | CHUNK_LOOKBACK + CHUNK_SIZE 算术 | O(1) |

所有 mask 都是 **O(1)** 或 **O(batch)** 空间，而不是 **O(seq²)**。

---

## 5. 关键叙事建议：本补充中最重要的 "WHY NOT" 链条

Writer 应在叙事中建立以下因果链（每个环节回答"为什么不那样"）：

```
问题 1: 为什么要 attention，不全连接？
→ 全连接参数固定依赖序列长度，attention 是参数无关的。

问题 2: 为什么点积不是加性？
→ 加性无法用 matmul 加速，GPU 上慢一个数量级。

问题 3: 为什么要除 √d_k 而不是直接 softmax？
→ 不缩放的话 softmax 饱和成 one-hot，梯度消失。

问题 4: 为什么多头而不是单头维度更大？
→ 多头 = 多个低秩子空间并行注意力。单头高维需要更大的 softmax 矩阵（二次增长），而多头线性增长。

问题 5: 为什么 GQA 而不是 MHA（显存）或 MQA（精度）？
→ MHA KV cache 太大，MQA 质量损失不可接受。GQA 是 sweet spot。

问题 6: 为什么需要 tiling（FlashAttention）？
→ GPU HBM 带宽是瓶颈，全矩阵 O(n²) 读写不可接受。

问题 7: 为什么 vLLM 不直接用 PyTorch 的 scaled_dot_product_attention？
→ vLLM 需要 block_table（PagedAttention）、FP8 scales、sinks 等专用功能，PyTorch 的通用 API 不提供。

问题 8: 为什么 mask 要在 kernel 内计算而不是预先构建 tensor？
→ O(n²) 的 mask tensor 在长序列下本身比 attention 计算还要耗内存。
```

---

## 6. 与 Narrative 各 Cell 的建议映射

| Cell | 本补充中最相关的章节 | 重点 |
|------|-------------------|------|
| Cell 2 (Hook) | §1.3 "层次1" | 用方差直觉吸引读者：d_k=64 时 softmax 输入可到 ±8 |
| Cell 3 (Problem) | §4.3 | "如果你用 PyTorch 写 attention，你会创建一个 seq² 的 mask — 然后 128K 上下文时显存就炸了" |
| Cell 4 (Theory) | §1.3 "层次2-3" | 完整的方差推导 + "如果不用 √d_k" 的对比 |
| Cell 5 (Walkthrough) | §4.1, §4.2 | 展示 vLLM 中 scale 的传递链 + QKV 投影的分离 |
| Cell 6 (Implementation) | §4.3 | Show `compute_kv_seq_mask` 作为 mask 策略的案例 |
| Cell 7 (Numerical) | §2.4 | GQA 压缩比的量化表格 |
| Cell 9 (Source Mapping) | §3.4 | vLLM backend 文件树 |
