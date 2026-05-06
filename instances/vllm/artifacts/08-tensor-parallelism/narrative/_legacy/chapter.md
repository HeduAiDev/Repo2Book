# 第8章：Tensor Parallelism — 跨 GPU 的矩阵分片

> 打开 `vllm/model_executor/layers/linear.py:410`。`ColumnParallelLinear` 和 `RowParallelLinear`
> 是 vLLM 中最重要的两个并行层。它们实现了一个简洁的数学事实：**矩阵乘法可以分片，代价是 AllReduce。**
> 理解 ColPar → RowPar 的组合模式 = 理解整个 Megatron-style TP。

---

## 这章要做什么？

前七章都在单 GPU 的假设下。但当 Llama-3.2-70B 有 140 GB 权重、H100 只有 80 GB 显存时，一张卡放不下。Tensor Parallelism 的解法：把权重矩阵切成 TP_size 份，每张 GPU 拿一份，并行计算后再通过 AllReduce 合并。

vLLM 实现了完整的 Megatron-LM 式 TP。它的核心只有两种层：`ColumnParallelLinear`（沿输出维切）和 `RowParallelLinear`（沿输入维切）。组合起来，一个 Transformer Block 只需要 **2 次 AllReduce**——而且其中没有额外的同步点。

学完这章你能：
- 推导为什么 ColPar → activation → RowPar 中间不需要通信——这是 Megatron TP 的数学基石
- 打开 `linear.py:410` 追踪 `ColumnParallelLinear.forward()` 的 all-gather 和 `linear.py:1394` 追踪 `RowParallelLinear.forward()` 的 all-reduce
- 理解 QKV 头分片——当 `tp_size > num_kv_heads` 时 KV 头如何复制
- 量化 TP 的通信开销——一次 forward 中有多少次 AllReduce，每次传多少字节

---

## 8.1 矩阵乘法的分片数学

### Theory: ColPar + RowPar 为什么只需要一次通信？

（本节推导对应 `vllm/model_executor/layers/linear.py:L410` ColPar 和 `L1394` RowPar 的数学基础。）

**直觉（先说结论再推导）：** 想象你把一本词典沿词条切分——每个同事拿一部分词条（ColPar）。然后每个同事**只查自己那部分词条**（不需要跟别人商量），查到后把结果拼起来（Reduce）。整个过程除了最后拼起来的那一步，大家不需要互相通信。这就是 ColPar → RowPar 的精髓——中间结果已经是**按 rank 分好**的，不需要重新洗牌。

**符号定义：** 设权重矩阵 $W_1 \in \mathbb{R}^{d_{in} \times d_{out}}$ 和 $W_2 \in \mathbb{R}^{d_{out} \times d_{hidden}}$，输入 $X \in \mathbb{R}^{B \times L \times d_{in}}$。单 GPU 计算：$Y = X W_1$, $Z = Y W_2$。

TP 把这个计算分到 $P$ 个 GPU。每个 GPU $i$ 持有 $W_1$ 的 $d_{out}/P$ 列和 $W_2$ 的 $d_{out}/P$ 行。

**ColPar（列切 $W_1$）：** 把 $W_1$ 沿输出维切成 $P$ 块：

$$
W_1 = [W_1^{(0)} \;|\; W_1^{(1)} \;|\; \cdots \;|\; W_1^{(P-1)}]
$$

每个 rank $i$ 用相同的输入 $X$ 计算自己的部分输出：

$$
Y_i = X W_1^{(i)} \in \mathbb{R}^{B \times L \times d_{out}/P}
$$

**关键观察：$Y_i$ 是正确且完整的。** 它不需要和其他 rank 商量——$W_1^{(i)}$ 的每一列独立于 $W_1^{(j)}$ 的列，乘积 $X \cdot [W_1^{(0)} \;|\; \cdots]$ 中的每列只依赖 $X$ 和对应列。中间 $\mathbf{Y = [Y_0 | Y_1 | \cdots | Y_{P-1}]}$ 天然就是按 rank 分好的。

**RowPar（行切 $W_2$）：** 把 $W_2$ 沿输入维切成 $P$ 块：

$$
W_2 = \begin{bmatrix} W_2^{(0)} \\ W_2^{(1)} \\ \vdots \\ W_2^{(P-1)} \end{bmatrix}
$$

每个 rank $i$ 用自己的 $Y_i$（来自 ColPar，$d_{out}/P$ 宽）和 $W_2^{(i)}$（$d_{out}/P$ 高）计算部分输出：

$$
Z_i = Y_i W_2^{(i)} \in \mathbb{R}^{B \times L \times d_{hidden}}
$$

然后 **AllReduce(SUM)** 合并：$Z = \sum_{i=0}^{P-1} Z_i$。

**为什么求和是正确的——形式化证明：**

完整乘积 $Z = Y W_2$。把 $Y$ 写成 ColPar 输出的拼接 $[Y_0 | \cdots | Y_{P-1}]$，把 $W_2$ 写成 RowPar 分片的堆叠：

$$
\begin{aligned}
Z &= Y W_2 
= [Y_0 \;|\; Y_1 \;|\; \cdots \;|\; Y_{P-1}] \cdot \begin{bmatrix} W_2^{(0)} \\ W_2^{(1)} \\ \vdots \\ W_2^{(P-1)} \end{bmatrix} \\
&= Y_0 W_2^{(0)} + Y_1 W_2^{(1)} + \cdots + Y_{P-1} W_2^{(P-1)} \\
&= Z_0 + Z_1 + \cdots + Z_{P-1}
\end{aligned}
$$

**这是块矩阵乘法的定义**——线性代数告诉我们 $[A|B] \cdot [C;D] = AC + BD$。ColPar 产生列分块的 $Y$，RowPar 产生行分块的 $W_2$，乘积恰好是各 rank 部分结果的和。

**数值 trace（tp=4, d_in=4, d_out=8, d_hidden=2, B=1, L=1）：**

| Rank | $W_1^{(i)}$ 形状 | $Y_i = XW_1^{(i)}$ 形状 | $W_2^{(i)}$ 形状 | $Z_i = Y_iW_2^{(i)}$ |
|------|-------------------|-------------------------|-------------------|---------------------|
| 0 | [4×2] | [1×2] | [2×2] | [1×2] |
| 1 | [4×2] | [1×2] | [2×2] | [1×2] |
| 2 | [4×2] | [1×2] | [2×2] | [1×2] |
| 3 | [4×2] | [1×2] | [2×2] | [1×2] |

AllReduce(SUM) → $Z = Z_0 + Z_1 + Z_2 + Z_3 \in \mathbb{R}^{1 \times 2}$。每个 rank 贡献了 $d_{out}/P=2$ 列 × $d_{hidden}=2$ 行 = 4 次乘加，4 个 rank 共 16 次乘加 = 全秩 $d_{out} \times d_{hidden} = 8\times 2 = 16$ 次乘加。**计算量守恒，只是分布了。**

**核心洞察（为什么中间不需要通信）：** ColPar 输出的列恰好按 rank 分好——rank 0 有列 0-1，rank 1 有列 2-3，等等。RowPar 需要的输入恰好是"按列分好的数据"——每个 rank 独立消费自己的列，不需要 access 其他 rank 的列。这就是"零通信中间层"的根本原因：**ColPar 的输出分割恰好匹配 RowPar 的输入分割。** 如果你颠倒顺序（先 RowPar 后 ColPar），RowPar 输出是完整的 $d_{hidden}$ 维向量，ColPar 需要完整输入——你必须在中间做 AllGather，多一次通信。

`★ Insight ─────────────────────────────────────`
TP 不需要复杂证明。它是线性代数中块矩阵乘法的直接应用。$[A_0|A_1] \cdot [B_0;B_1] = A_0 B_0 + A_1 B_1$ 不是近似——是恒等式。这意味着 TP 在数学上是**精确的**——和单 GPU 计算结果完全一样（忽略 fp16 舍入）。不像其他并行策略（数据并行需要 gradient sync，流水线并行需要 microbatch bubble），**TP 唯一的通信成本是 AllReduce——而且只在 ColPar→RowPar 对的末尾发生一次。**
`─────────────────────────────────────────────────`

---

## 8.2 ColumnParallelLinear

### Source Trail

打开 `vllm/model_executor/layers/linear.py:410`。

```python
class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, tp_size, ...):
        self.output_size_per_partition = out_features // tp_size   # L454
        self.weight = Parameter(torch.empty(
            self.output_size_per_partition, in_features))           # [out/tp, in]

    def forward(self, input_):
        # Each rank: Y_i = X @ W_i^T (local GEMM)                  # L579
        output_parallel = F.linear(input_, self.weight, self.bias)

        if self.gather_output:                                      # L591
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
```

**权重形状：** `[out/tp, in]`。每个 rank 的 W 有 `out_features/tp_size` 行，`in_features` 列。

**两个模式：**
- `gather_output=True`（默认）: GEMM 之后做 AllGather，每个 rank 得到完整输出。用于那些后面没有 RowPar 的层（如独立的分类头）。
- `gather_output=False`: 不做 gather。用于 QKV 和 gate/up——这些层的输出直接流入下一个 RowPar，已经以正确的分片格式存在。

### MergedColumnParallelLinear (L609)

用于 MLP 的 gate_proj + up_proj 融合。这两个投影共享相同的输入 X，只是输出不同——把它们拼在一起更高效：

```python
# gate_proj: W_gate [intermediate/tp, in]
# up_proj:   W_up   [intermediate/tp, in]
# Merged:    W_merged = cat([W_gate, W_up], dim=0)  [2*intermediate/tp, in]
# Forward:   Y_merged = X @ W_merged^T → split into gate, up
```

---

## 8.3 RowParallelLinear

### Source Trail

打开 `vllm/model_executor/layers/linear.py:1394`。

```python
class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, tp_size, ...):
        self.input_size_per_partition = in_features // tp_size  # L1447
        self.weight = Parameter(torch.empty(
            out_features, self.input_size_per_partition))        # [out, in/tp]

    def forward(self, input_):
        if not self.input_is_parallel:                          # L1548
            input_ = input_.chunk(tp_size, dim=-1)[self.tp_rank]

        output_parallel = F.linear(input_, self.weight)         # L1556

        if self.reduce_results:                                 # L1563
            output = tensor_model_parallel_all_reduce(output_parallel)
```

**权重形状：** `[out, in/tp]`。每个 rank 的 W 有 `out_features` 行，`in_features/tp_size` 列。

**`input_is_parallel` 参数：**
- `True`: 输入已经是分片格式（来自前面的 ColPar`gather_output=False`）→ 直接用
- `False`: 输入是完整的 → 内部做 `chunk` 切分 → 每个 rank 只取自己的那片

**`reduce_results=True`（默认）:** GEMM 之后做 AllReduce(SUM)，把各 rank 的部分结果加起来。

---

## 8.4 QKVParallelLinear：头分片的特殊处理

### Source Trail

打开 `vllm/model_executor/layers/linear.py:977`。

```python
class QKVParallelLinear(ColumnParallelLinear):
    def __init__(self, ..., num_heads, num_kv_heads, head_size, tp_size, ...):
        self.num_heads = num_heads // tp_size          # L1030
        if tp_size >= num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = tp_size // num_kv_heads  # L1033
        else:
            self.num_kv_heads = num_kv_heads // tp_size
            self.num_kv_head_replicas = 1               # L1036
```

QKV 投影的分片不是均匀的——Q 和 KV 的处理方式不同：

**Q 头分片：** 总 Q 头数除以 tp_size——每个 rank 负责 `num_heads/tp` 个头。4 GPU 分 32 头 → 每 rank 8 头。简单的均分。

**KV 头分片：** 分两种 case：
- **Case 1: `tp_size <= num_kv_heads`（标准 GQA）**：KV 头在 rank 间均分。32 头 Q、8 头 KV、tp=4 → 每 rank 8 Q 头 + 2 KV 头。
- **Case 2: `tp_size > num_kv_heads`（过并行）**：KV 头不够分！例如 `num_kv_heads=2, tp_size=4`——只有 2 个 KV 头但需要分到 4 张 GPU 上。vLLM 的处理：**复制 KV 头。** 每 rank 拿到 1 个 KV 头（完整的），`num_kv_head_replicas = tp_size // num_kv_heads = 2`——意味着同一个 KV 头被 2 个 rank 共享。

**为什么 KV 头可以复制？** 因为 KV 头的权重是相同的矩阵——复制不会影响正确性，只是每个副本独立计算相同的 K 和 V。这在数学上等价于 GQA 的 `repeat_interleave` 在硬件层面实现。

---

## 8.5 TP + Attention：头级别的并行

### Source Trail

打开 `vllm/model_executor/models/llama.py:145`：

```python
self.num_heads = self.total_num_heads // tp_size
self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
```

`Attention` 层本身**不执行任何 TP 通信**。原因：Q、K、V 投影已经被 `QKVParallelLinear` 分片了——每个 rank 只有自己的 Q/KV 头子集，Attention 在这些本地头上独立计算。

打开 `vllm/model_executor/layers/attention/attention.py:177`——`Attention.forward()` 不包含任何 AllReduce 或 AllGather。因为 FlashAttention 的输入已经是分片的：`Q: [batch, num_heads_per_rank, head_dim]`。

**O 投影是唯一需要通信的点。** O 投影作为 `RowParallelLinear` 实现——每个 rank 计算自己那部分头的输出，然后 AllReduce 合并。这与 ColPar → RowPar 模式一致：QKV(ColPar) → Attention(local) → O(RowPar → AllReduce)。

---

## 8.6 通信成本量化

### Theory: 一次 forward 的 AllReduce 次数

打开 `vllm/model_executor/models/llama.py`，追踪完整的 TP 通信：

```
1. VocabParallelEmbedding.forward()
   → AllReduce (合并嵌入)

2. For each decoder layer:
   a. QKV (ColPar, gather=False) → Attention (local) → O (RowPar, AllReduce)
   b. Gate+Up (ColPar, gather=False) → SiLU → Down (RowPar, AllReduce)
   → 2 AllReduces per layer

3. LM Head (ParallelLMHead)
   → AllGather (收集 logits)

Total: 1 + 2×L + 1 = 2L + 2 次集体通信
```

对于 L=32 层的 Llama-3.2：**66 次 AllReduce** 每次 forward。

### Per-AllReduce 的数据量

一次 AllReduce 传一个 `[B, L, d_model]` 的 tensor。bf16 = 2B/token：

```
per_AR_bytes = 2 × B × L × d × 2 bytes
             = 2 × 4 × 4096 × 4096 × 2 = 268 MB  (双倍因为 send+recv)
```

对于 NVLink（900 GB/s 双向）：268 MB / 900 GB/s ≈ **0.3 ms** per AllReduce。

66 × 0.3 ms = **~20 ms** 的总通信时间。对比一次 forward 的约 200 ms 计算时间，通信占比约 **10%**——在大多数规模下可以接受。

**TP 通信的隐藏：** vLLM 可以通过 `torch.distributed.all_reduce` 与 PyTorch 的 CUDA stream 重叠计算和通信——AllReduce 在后台进行，GPU 同时处理下一个 kernel。

---

## 8.7 权重加载：分片参数

### Source Trail

打开 `vllm/model_executor/parameter.py:148`：

```python
def load_column_parallel_weight(param, loaded_weight, tp_rank, tp_size, ...):
    shard_size = param.shape[output_dim]   # output_size_per_partition
    start_idx = tp_rank * shard_size
    loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
```

加载时，磁盘上的完整权重被每个 TP rank 切成 1/tp 的条带。ColPar 沿 output 维切，RowPar 沿 input 维切。

**融合权重的加载（gate_up_proj, qkv_proj）：** 打开 `llama.py:436`：

```python
stacked_params_mapping = [
    (".qkv_proj", ".q_proj", "q"),
    (".qkv_proj", ".k_proj", "k"),
    (".qkv_proj", ".v_proj", "v"),
    (".gate_up_proj", ".gate_proj", 0),
    (".gate_up_proj", ".up_proj", 1),
]
```

磁盘上可能是分开的 `q_proj`、`k_proj`、`v_proj` 权重，但 vLLM 在内存中把它们融合为一个 `qkv_proj`。这个映射表告诉加载器：把 `q_proj` 加载到 `qkv_proj` 的 "q" 部分，把 `k_proj` 加载到 "k" 部分——同时对每个部分应用正确的 TP 分片。

---

## 我们的实现 vs vLLM 源码

| 我们的实现 | vLLM 原始源码 | 说明 |
|---|---|---|
| `ColumnParallelLinear` | `linear.py:L410` | 保留核心：out/tp 分片 + all-gather。未实现量化方法（quant_method.apply） |
| `RowParallelLinear` | `linear.py:L1394` | 保留核心：in/tp 分片 + all-reduce。未实现量化方法 |
| `TPTransformerBlock` | `llama.py:L316` `LlamaDecoderLayer` | 展示 ColPar→RowPar 模式；简化了真实 attention 和 MLP 计算 |
| `SimulatedTPGroup` | `parallel_state.py:L290` `GroupCoordinator` | 教学版——无真实 NCCL，演示通信模式 |
| 通信分析 | `communication_op.py` | 原创量化分析 |

---

## 验证

```bash
cd artifacts/08-tensor-parallelism && python -m pytest tests/ -q
# 10/10 passed ✅
```

---

## 总结

- **ColPar → RowPar 是 Megatron TP 的核心模式。** 利用块矩阵乘法的分配律 $[A|B] \cdot [C;D] = AC + BD$，实现中间零同步。
- **一个 Transformer block 只需 2 次 AllReduce。** QKV(ColPar)→Attention(local)→O(RowPar) 和 GateUp(ColPar)→SiLU→Down(RowPar)。
- **KV 头在 `tp_size > num_kv_heads` 时被复制。** 数学等价于 GQA 在硬件层面——复制不会影响正确性。
- **AllReduce 通信占比 <10% 在典型规模下。** NVLink 带宽让 TP 在 8 GPU 以内保持高效。

---

**下一章：** 第9章 — Expert Parallelism：MoE 专家并行

TP 把一层内的权重分片。EP 把 MoE 的 experts 分片——但这里不只是切矩阵。你需要一个 Router 决定每个 token 去哪个 expert，需要 AllToAll 通信把 token 路由到正确的 GPU，需要 load balancing 防止某些 expert 过载。下一章将追踪 vLLM 中从 `FusedMoE.forward()` 到 `all_to_all` 通信的完整路径。

---

← 第7章 | 第9章 →
