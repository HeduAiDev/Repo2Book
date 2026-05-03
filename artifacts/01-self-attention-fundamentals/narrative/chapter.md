# 第1章：Self-Attention — vLLM 的计算核心

> 打开 `vllm/model_executor/layers/attention/attention.py:177`，你会看到 `class Attention`。
> 这是 vLLM 推理引擎中最频繁调用的类——每生成一个 token，每层 transformer 都要通过它。
> 本章我们把这个类背后的数学和工程一起搞透。

---

## 这章要做什么？

vLLM 的 `Attention` 类不是教科书里的 `softmax(QK^T/√d_k)V`。它的设计体现了一个关键的工程决策：**把"算什么"（数学）和"怎么算"（kernel）分离。** 同一个 `Attention` 类在 H100 上走 FlashAttention CUDA kernel，在 AMD 上走 ROCm kernel，在 CPU 上回退 PyTorch 原生实现——接口不变。

但分离只是工程手段。底层算的东西，仍然是那篇 2017 年的论文中那四个字母：Q, K, V, √d_k。要理解这套系统，你需要同时懂两样东西：**代码在哪里（source trail）** 和 **为什么这么设计（theory）**。

学完这章你能：
- 打开 `attention.py:177`，对着源码解释 `Attention.__init__()` 为什么要创建 `self.impl`
- 从方差分析的数学原理出发，证明 $1/\sqrt{d_k}$ 是数学必然而非调参产物
- 推导 Online Softmax 算法，理解 FlashAttention 为什么不需要 $[seq^2]$ 矩阵
- 手写 Triton fused attention kernel，逐行解释每个 SRAM tile 的大小与 L1 cache 的约束

---

## 1.1 源码入口 + 核心直觉

### Source Trail

打开 `vllm/model_executor/layers/attention/attention.py:177`：

```python
class Attention(nn.Module):
    def __init__(self, num_heads, head_size, scale, num_kv_heads, ...):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)          # ← 1/√d_k, 预计算
        backend = get_attn_backend(...)     # ← 选 kernel 实现
        self.impl = backend.get_impl_cls()(num_heads, head_size, scale, ...)
```

注意三点：
1. **`scale` 被预计算为对象属性**——它不是临时的局部变量。这个细节告诉我们 $1/\sqrt{d_k}$ 的重要性：它被提升到了实例状态的级别。
2. **`Attention` 自己不包含 attention 计算代码**——它创建一个 `self.impl`（backend 实现），所有计算委托给它。本节先讲底层的数学；1.2 节会拆解这个 backend 架构。
3. **QKV 投影不在 `Attention` 类中**——它们定义在模型文件里（如 `vllm/model_executor/models/llama.py` 的 `LlamaAttention` 类）。这是因为不同模型的投影策略不同：标准 Transformer 用三个独立投影，GQA 用不同的 KV 维度，MLA（DeepSeek）用压缩的 latent 空间。

### Theory: Q、K、V 到底是什么？

在讲数学之前，先把直觉建立起来。这三个字母不是三种不同性质的东西——它们来自同一个输入 $x$，分别通过三个不同的线性变换得到：

- **Q (Query):** "我当前这个 token 想找什么信息？"——一个搜索请求
- **K (Key):** "我过去每个 token 能提供什么信息？"——一个索引标签
- **V (Value):** "我过去每个 token 的实际内容是什么？"——取多少取决于 Q 和 K 的匹配度

**开组会类比：** 你（Q）想了解"这里谁懂 Triton 编程？"你扫一眼全场，每个人面前有铭牌（K）写着技能。铭牌内容和你的问题越匹配，你就越认真听那个人的发言（V）。

**在 vLLM 的术语里：Q 和 K 决定"注意力往哪放"，V 决定"放多少信息过去"。**

---

## 1.2 数学推导：Scaled Dot-Product Attention

这一节从头推导 attention 公式。每一步都是后一节背后的"为什么"。

### Step 1: 定义 Q, K, V

序列中的每个位置 $i$ 有三个向量：

$$
\mathbf{q}_i = \mathbf{x}_i W^Q, \quad \mathbf{k}_i = \mathbf{x}_i W^K, \quad \mathbf{v}_i = \mathbf{x}_i W^V
$$

$W^Q, W^K, W^V$ 是可学习的权重矩阵。开始时随机——模型通过训练学习"什么样的 Q 应该匹配什么样的 K"。

### Step 2: 计算相关性分数

（这四步数学在 vLLM 中没有对应的 Python 代码——它们发生在 `FlashAttentionImpl.forward()` 调用的 CUDA kernel 内部，`flash_attn.py` 中看不到。但每一个 backend 实现都必须等价地执行这个计算。）

位置 $i$ 对位置 $j$ 的注意力分数 = Query $i$ 和 Key $j$ 的点积：

$$
\mathrm{score}(i, j) = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}}
$$

点积越大 → $\mathbf{q}_i$ 和 $\mathbf{k}_j$ 越"相似" → token $i$ 越想关注 token $j$。

### Step 3: Softmax 归一化

对所有 $j$ 上的分数做 softmax：

$$
\alpha_{ij} = \frac{\exp(\mathrm{score}(i, j))}{\sum_{k=1}^{n} \exp(\mathrm{score}(i, k))}
$$

$\alpha_{ij}$ 的含义：token $i$ 花在 token $j$ 上的注意力比例。每行之和为 1。

### Step 4: 加权求和

$$
\mathbf{o}_i = \sum_{j=1}^{n} \alpha_{ij} \cdot \mathbf{v}_j
$$

token $i$ 的输出 = 所有 Value 的加权平均，权重就是注意力。

### 矩阵形式（这个公式你应该能背下来）

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

### 手算验证

3 个 token，$d_k = 4$。token 2 的 Query 对三个 Key 做点积：

```
q_2 = [0.5, 0.1, 0.3, 0.2]
k_0 = [0.2, 0.8, 0.1, 0.4]  →  q_2·k_0 = 0.5×0.2 + 0.1×0.8 + 0.3×0.1 + 0.2×0.4 = 0.29
k_1 = [0.7, 0.3, 0.5, 0.1]  →  q_2·k_1 = 0.5×0.7 + 0.1×0.3 + 0.3×0.5 + 0.2×0.1 = 0.72
k_2 = [0.4, 0.2, 0.9, 0.3]  →  q_2·k_2 = 0.5×0.4 + 0.1×0.2 + 0.3×0.9 + 0.2×0.3 = 0.73
```

原始分数: $[0.29, 0.72, 0.73]$ → 除以 $\sqrt{4}=2$ → $[0.145, 0.36, 0.365]$

$$
\begin{aligned}
e^{0.145} &= 1.156 \\
e^{0.36}  &= 1.433 \\
e^{0.365} &= 1.440 \\
\mathrm{sum} &= 4.029
\end{aligned}
\qquad\Longrightarrow\qquad
\mathrm{softmax} = [0.29, 0.36, 0.36]
$$

token 2 的注意力分布：29% 给 token 0，36% 给 token 1，36% 给自己。分布均匀，梯度流动顺畅。

---

## 1.3 方差分析：为什么是 $1/\sqrt{d_k}$？

`★ Insight ─────────────────────────────────────`
这一节是整个 attention 机制中最被低估的细节。$1/\sqrt{d_k}$ 不是调参调出来的，也不是"经验表明除以 √d_k 效果好"——它是独立随机变量方差性质的必然结论。理解了这个推导，你就知道为什么 $d_k=128$ 时不除以 √128 模型直接废掉，而 $d_k=4$ 时除不除差别不大。
`─────────────────────────────────────────────────`

### Source Trail

vLLM 在 `attention.py:L200` 将这个值预计算为 `self.scale`——一个实例属性，不是局部变量。这本身就是一个信号：这个东西重要到需要被保存下来。

### Theory: 方差推导

假设 $q$ 和 $k$ 的每个维度是独立的随机变量，均值为 0，方差为 1。考虑点积的方差：

$$
\begin{aligned}
\mathrm{Var}(q \cdot k) &= \mathrm{Var}\left(\sum_{i=1}^{d_k} q_i k_i\right) \\
&= \sum_{i=1}^{d_k} \mathrm{Var}(q_i k_i)  \\
&= \sum_{i=1}^{d_k} \left(\mathrm{Var}(q_i) \cdot \mathrm{Var}(k_i) + \mathrm{Var}(q_i) \cdot \mathbb{E}[k_i]^2 + \mathrm{Var}(k_i) \cdot \mathbb{E}[q_i]^2\right) \\
&= \sum_{i=1}^{d_k} (1 \cdot 1 + 1 \cdot 0 + 1 \cdot 0) = d_k
\end{aligned}
$$

第一步到第二步用了独立性（不同维度的乘积不相关）。第二步到第三步是独立随机变量乘积的方差展开公式。结论：点积的方差 = $d_k$。这意味着：

| $d_k$ | 未缩放点积标准差 | softmax 行为 |
|-----|-------------|------------|
| 4 | σ = 2 | 分布均匀，无问题 |
| 16 | σ = 4 | 开始集中 |
| 64 | σ = 8 | 显著集中，梯度减弱 |
| 128 | σ ≈ 11 | **退化为一-hot，梯度 ≈ 0** |
| 256 | σ = 16 | 完全 one-hot，模型无法训练 |

**解决方案：**

$$
\mathrm{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{\mathrm{Var}(q \cdot k)}{d_k} = 1
$$

除以 √d_k 把方差拉回 1，softmax 保持均匀分布。

### 实验验证

运行 `implementation/variance_analysis.py` 验证不同 $d_k$ 下未缩放 vs 缩放后的 softmax 熵。$d_k=256$ 时，未缩放的熵趋近 0（one-hot），缩放后保持在 ~1.0（均匀分布）。

**推论：** $1/\sqrt{d_k}$ 的必要性是 $d_k$ 的函数。$d_k$ 小的时候（如 4-8）不缩放也没事。$d_k=128$（Llama-3.2）时不缩放就是灾难。这个 nuance 在大多数教材里被省略了。

---

## 1.4 Multi-Head Attention：低秩分解视角

### Source Trail

打开 `attention.py:L410-L450`，`Attention.forward()` 的第一段关键操作：

```python
# Q 从 [num_tokens, d_model] reshape 到 [num_tokens, num_heads, head_size]
Q = Q.view(-1, self.num_heads, self.head_size)
```

这个 reshape 是 Multi-Head 的全部秘密。它不是复制模型 $h$ 份——它是把 $d_{model}$ 维空间切分成 $h$ 个独立的 $d_k$ 维子空间，每个 head 在自己的子空间里独立做 attention。

### Theory: 低秩分解的线性代数

从线性代数角度看 Multi-Head Attention：

一个全秩的注意力矩阵 $A \in \mathbb{R}^{L \times L}$ 有 $L^2$ 个自由度来建模任意 token 对间的关系。直接学习 $L^2$ 个参数是不现实的。

MHA 的做法：用 $h$ 个低秩投影来近似 $A$。每个 head 学习三个矩阵 $(W_i^Q, W_i^K, W_i^V)$，每个都映射到 $d_k = d_{model}/h$ 维子空间：

$$
\mathrm{head}_i = \mathrm{Attention}(X W_i^Q, X W_i^K, X W_i^V)
$$

$h$ 个 head 的输出拼接后投影回全空间：

$$
\mathrm{MHA}(X) = \mathrm{Concat}(\mathrm{head}_1, ..., \mathrm{head}_h) W^O
$$

**为什么这比单头好？** 因为注意力矩阵 $A$ 在实际语言中通常是低秩的——很多 token 之间的关系是冗余的（"the cat" 和 "the feline" 表达几乎相同的依赖）。$h$ 个低秩（rank ≤ $d_k$）的注意力结果拼起来，参数总量是 $h \times 3d_{model} \times d_k = 3d_{model}^2$——与 $L^2$ 无关。这种参数化的低秩分解比直接学全秩矩阵更高效，同时保留了足够的表达能力。

**为什么 32 个头？** 这不是理论推导出来的，是工程实践的结果。经验表明 32-64 个头在大多数任务上效果最好。头太少 → 每个头负担太多关系模式 → 表达能力不足。头太多 → 每个头只学到噪音 → 浪费计算。

---

## 1.5 GQA：KV 的共享经济

### Source Trail

从 `attention.py:L177` 的构造函数参数可以看到 `num_kv_heads`。如果它小于 `num_heads`，vLLM 进入 GQA 模式。打开 `vllm/v1/attention/backends/flash_attn.py`，搜索 `num_kv_heads`——FlashAttention 的 kernel 原生支持 GQA，K 和 V 传进去只有 `num_kv_heads` 个 head，kernel 内部按 stride 读取，**不需要 expansion，不浪费显存。**

### Theory: KV 压缩的数学

MHA 中，32 个 Q head 各自有专属的 32 个 K 和 32 个 V head。KV Cache 大小（第 2 章会详讲）：

```
MHA:  cache_size ∝ num_heads × head_dim   = 32 × 128 = 4096 per layer per token
GQA:  cache_size ∝ num_kv_heads × head_dim = 8 × 128  = 1024  ← 省了 75%
MQA:  cache_size ∝ 1 × head_dim            = 128     ← 省了 97%
```

GQA 用 25% 的缓存达到 >99% 的 MHA 精度。为什么精度损失这么小？因为 32 个 K head 学到的 pattern 有大量重叠——不同的 K head 捕获的 key 方向之间有高相关性。8 个 K head 已经足够覆盖主要的 key 方向。这些额外的 K 和 V head 本质上是冗余的——这就是 GQA 能省 75% 而精度损失 <1% 的根本原因。

| 模式 | 配置 | KV Cache 相对 MHA | 使用模型 |
|------|------|------------------|---------|
| MHA | num_kv_heads = 32 | 100% (baseline) | 早期 GPT |
| GQA | num_kv_heads = 8 | 25% | Llama 3, Mistral, Gemma |
| MQA | num_kv_heads = 1 | 3.1% | PaLM, Falcon（早期） |

**GQA 的隐藏代价：** K/V 投影矩阵的非均匀访问。Q 32 个头，K 只有 8 个头——attention 计算时 Q 的 stride 和 K 的 stride 不同。在 GPU 上这意味着不同的内存访问模式，可能影响 coalescing 效率。但 FlashAttention 的 kernel 设计已经处理了这一点——它把 GQA 的 stride 差异隐藏在 tiled access pattern 中。

---

## 1.6 Triton Fused Attention Kernel

### Source Trail

vLLM 生产用的是 CUDA FlashAttention（`vllm/v1/attention/backends/flash_attn.py`），但核心算法——Online Softmax + Tiled Reduction——在 CUDA 和 Triton 中都一样。我们实现了一个教育版的 Triton kernel（`implementation/fused_attention_triton.py`），让你能看到算法结构。打开 `flash_attn.py`，`FlashAttentionImpl.forward()` 最终调用 `flash_attn_varlen_func()`——这正是 tiled attention 的 C++/CUDA 实现。

### Theory: 为什么需要"融合"？

**问题诊断。** 朴素 attention 有三步，每一步都要读写 HBM：

```python
S = Q @ K^T              # 写 [seq²] 到 HBM  ← O(n²·d) bytes
P = softmax(S)           # 读 [seq²], 写 [seq²] ← 2×O(n²·d) bytes
O = P @ V               # 读 [seq²], 写 [seq]  ← O(n²·d) + O(n·d) bytes
```

对于 seq_len=4096, num_heads=32：S 是 [32, 4096, 4096] = 536M 元素，fp32 = 2GB。更致命的是：seq_len=128K 时 = [32, 128K, 128K] = 2TB——放不进任何 GPU。

**根本矛盾：** Attention 的数学需要 $O(L^2)$ 的计算，但这不意味着需要 $O(L^2)$ 的**显存**。FlashAttention 的洞察：**计算可以做在 SRAM 里，不写回 HBM。把 O(n²) 的 HBM 读写降到 O(n)。**

### Algorithm: Online Softmax

普通 softmax 需要 three-pass：找 max → exp + 求和 → 归一化。这三个 pass 都需要读/写整个向量。

Online softmax 用一个 pass 完成。核心思想：**维护 running max，当发现更大的值时，用 correction 因子重新缩放累积结果。**

```
Algorithm: Tiled Attention with Online Softmax

For each Q_block (loaded ONCE from HBM to SRAM):
    m = -inf     # running max per row
    l = 0        # running normalization sum
    O_acc = 0    # running output accumulator

    For each KV_block:
        S = Q_block @ K_block^T / sqrt(d_k)   # [BLOCK_Q × BLOCK_KV] IN SRAM ONLY

        m_new = max(m, row_max(S))            # update running max
        P = exp(S - m_new)                     # numerically stable
        correction = exp(m - m_new)            # rescale old accumulator
        l_new = correction * l + row_sum(P)   # update normalization

        O_acc = correction * O_acc + P @ V_block
        m, l = m_new, l_new

    O_block = O_acc / l                       # normalize once at the end
    Write O_block to HBM
```

**`correction = exp(m - m_new)` 为什么存在？** 因为我们在 $m$ 更新前计算的 $O_{acc}$ 用的是旧的 $m$。新 $m_{new}$ 更大 → 旧的 exp 值需要缩小 $e^{m - m_{new}}$ 倍。注意 correction ≤ 1，且随着迭代进行（max 越来越准），correction → 1——后期 KV block 几乎不需要 rescaling。这不是 bug，是算法的收敛性质。

### SRAM 用量分析

**这是 HPC 学生需要记住的数字。** BLOCK_Q=64, BLOCK_KV=64, HEAD_DIM=128, fp16：

| 张量 | 大小 | 在哪 |
|------|------|------|
| Q_block (fp16) | 64 × 128 × 2B = 16 KB | SRAM |
| K_block (fp16) | 64 × 128 × 2B = 16 KB | SRAM |
| V_block (fp16) | 64 × 128 × 2B = 16 KB | SRAM |
| S (fp32) | 64 × 64 × 4B = 16 KB | SRAM |
| P (fp32) | 64 × 64 × 4B = 16 KB | SRAM |
| O_acc (fp32) | 64 × 128 × 4B = 32 KB | SRAM |
| **Total** | **~112 KB** | — |

H100 L1/SMEM per SM = **228 KB** → 112 KB tile 放得下，不需要 register spilling。但如果 BLOCK_Q=128, BLOCK_KV=128：Q 32KB + K 32KB + V 32KB + S 64KB + P 64KB + O 64KB ≈ **288 KB** > 228 KB → register spilling → 性能暴跌。

**结论：block size tuning 不是越大越好——要精确适配 L1 cache 大小。** H100 的 228KB 是最优配置的硬边界。

`★ Insight ─────────────────────────────────────`
这里有一个在论文和教程中常被忽略的细节：`correction` 的收敛性质。当 `m` 在早期迭代中已经找到"真正"的最大值时，后续 KV block 的 `m_new` 等于 `m`，导致 `correction = exp(m - m_new) = exp(0) = 1`。这意味着**计算 `correction * O_acc` 实际上是一个 no-op**——GPU 仍然在做乘 1.0 的运算。FlashAttention-2 优化了这个：它只在 `m_new != m` 时才执行 rescaling，通过 warp-level 的 ballot 指令检查是否需要 correction。这个 micro-optimization 在长序列推理中可以节省 ~5% 的 kernel 运行时间。
`─────────────────────────────────────────────────`

---

## 1.7 Backend 架构：分离"算什么"和"怎么算"

### Source Trail

回到 `attention.py:L177`。为什么 `Attention` 自己不包含 attention 计算代码，而是创建一个 `self.impl`？

打开 `vllm/v1/attention/backends/registry.py` 查看完整 backend 列表。每个 backend 必须实现三个组件（从 `backend.py` 的抽象基类定义）：

```
AttentionBackend
  ├── get_impl_cls()      → AttentionImpl  (实际计算)
  ├── get_builder_cls()   → AttentionMetadataBuilder (per-request 元数据)
  └── get_kv_cache_shape()→ tuple           (KV Cache tensor 形状)
```

Backend 选择逻辑在 `vllm/v1/attention/selector.py` 的 `get_attn_backend()`：

```
1. 检查用户配置 (vllm_config.attention_config.backend)
2. 回退到平台默认: CUDA→FLASH_ATTN, AMD→ROCM_ATTN, CPU→CPU_ATTN
3. 检查 head_size/dtype/KV cache dtype 是否兼容
4. 结果缓存在 _cached_get_attn_backend 中（backend 不会在运行时切换）
```

`Attention.forward()` 的完整调用链（`attention.py:L410-L530`）：

```
Attention.forward(Q, K, V, kv_cache, attn_metadata)
  → 分配 output tensor
  → torch.ops.vllm.unified_attention_with_output(Q, K, V, ...)
      → self.impl.forward(Q, K, V, kv_cache, attn_metadata, output)
          → flash_attn_varlen_func(...)   # FLASH_ATTN
          → triton_kernel(...)             # TRITON_ATTN
          → flex_attention(...)            # FLEX_ATTENTION
```

`torch.ops.vllm` 是 vLLM 注册的自定义 opaque op——它把整个 attention 打包成一个算子，让 `torch.compile` 不会在 attention 处打断计算图。

### Theory: 为什么需要这个抽象？

Backend 抽象回答了软件工程中的一个基本问题：**当同一个数学操作需要在三种不同的硬件架构上用六种不同的 kernel 实现时，如何避免代码爆炸？**

vLLM 的答案：把"做什么"（Attention 类——QKV 的 reshape、output 的分配、mask 的处理）和"怎么做"（AttentionImpl——FlashAttention CUDA、Triton kernel、FlexAttention）分离。

**这个设计的代价：** 每次 `forward()` 多了一层间接调用（`self.impl.forward()`）和一个 opaque op 的注册开销。**收益：** 同一个 vLLM 代码库可以在 H100、A100、AMD MI300X、甚至 CPU 上运行——只需换 backend。新模型架构（如 DeepSeek 的 MLA）也只需要提供一个新的 `MLAAttentionImpl`，不需要修改 `Attention` 类。

---

## 我们的实现 vs vLLM 源码

| 我们的实现 | vLLM 原始源码 | 说明 |
|---|---|---|
| `MultiHeadAttention.__init__()` | `attention.py:L177` | 无 backend 抽象、无 KV cache spec。简化以适合单章范围 |
| `MultiHeadAttention.forward()` | `attention.py:L410` | 显式 attention 计算——读者可以看到数学过程 |
| `self.W_q, self.W_k, self.W_v` | `llama.py` → `LlamaAttention.qkv_proj` | vLLM 用 combined QKV；我们分开以展示三条数据路径 |
| `self.scale = 1/√head_dim` | `attention.py:L200` | 完全相同——预计算以提升效率 |
| `_reshape_for_heads()` | `attention.py:L410-L450` | 我们的 `[B, h, L, d]` vs vLLM 的 `[tokens, heads, dim]` |
| `GroupedQueryAttention` | `attention.py:L177` + `flash_attn.py` | vLLM 在 kernel 内处理 GQA；我们展开以展示共享模式 |
| `_fused_attention_kernel` (Triton) | `flash_attn.py` → FlashAttention CUDA | 教育版 Triton vs 生产版 CUDA |
| `variance_analysis.py` | "Attention Is All You Need" 脚注 3 | 论文推导 + 我们的实验验证 |
| 各种 attention mask | FlashAttention kernel 内部（`causal=True`） | vLLM 不显式创建 mask tensor；我们用于可视化 |

---

## 验证

在 vLLM Docker 容器中运行：

```bash
cd artifacts/01-self-attention-fundamentals
python -m pytest tests/ -v
# 13/13 passed ✅
```

---

## 总结

从 `attention.py:177` 到 `flash_attn.py` 的 CUDA kernel，再到我们自己实现的 Triton fused kernel。关键收获：

- **Backend 抽象分离了"算什么"和"怎么算"。** 同一个 `Attention` 类，六种 backend，三种硬件架构，接口不变。
- **$1/\sqrt{d_k}$ 是概率论的必然，不是调参。** 独立随机变量点积的方差 = $d_k$。除以 √d_k 让方差回到 1，防止 softmax 饱和。
- **Multi-Head = 低秩分解。** h 个 d_k 维子空间的注意力结果拼起来，参数量与 $L^2$ 无关。
- **GQA 省 75% KV Cache，精度损失 <1%。** 32 个 K head 学到的 pattern 高度冗余——8 个就够了。
- **FlashAttention 的核心是 IO-awareness。** Tiled Online Softmax 把 $O(n^2)$ 的 HBM 读写降到 $O(n)$。BLOCK_Q × BLOCK_KV 的 tile 大小受 L1 cache 硬约束（H100: 228 KB）。

---

**下一章：** 第2章 — KV Cache：vLLM 的内存管理核心

Attention 每次都需要历史 K 和 V——但 vLLM 不会每次都重新计算。打开 `vllm/v1/core/kv_cache_manager.py:106`，`KVCacheManager.allocate_slots()` 是每次 scheduler 循环中第一个被调用的方法。第 2 章将拆解它的三层架构、BlockPool 的 LRU 驱逐、以及 prefix cache 的 hash-based 共享机制。

---

← | 第2章 →
