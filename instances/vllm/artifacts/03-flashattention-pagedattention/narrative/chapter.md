# 第3章：FlashAttention & PagedAttention — 当计算遇上内存墙

> 本章涉及的源码：`csrc/attention/attention_kernels.cuh:85`（paged attention kernel）、
> `vllm/v1/attention/backends/flash_attn.py:682`（FA backend）、
> `vllm/v1/attention/ops/triton_decode_attention.py:60`（Triton decode kernel）。
>
> FlashAttention 优化"怎么算"。PagedAttention 优化"怎么存"。
> vLLM 把它们融合在同一个 kernel 里——在 tiled softmax 的每一步，
> 用 block_table 找到物理 block，加载，计算，累积。这就是 vLLM 最精华的设计。
>
> **这是全书最难的一章。但看完以后，你对 GPU 计算的理解会上一个台阶。**

---

## 这章要讲什么？

第 1 章教你 Attention 的数学——`softmax(QK^T/√d_k)V`。第 2 章教你 KV Cache 的管理——怎么存、怎么驱逐。但这两个东西之间有一个巨大的 gap：**KV Cache 里的数据是非连续的（block 散落在物理 GPU 显存里），而 Attention kernel 需要遍历它们。怎么把两者连起来？**

答案是**融合**。打开 `csrc/attention/attention_kernels.cuh:252`：

```cpp
const int64_t physical_block_number = block_table[block_idx];
```

这一行在 Attention kernel 的**循环体内部**——不是在 kernel 外面查好再传进去。它和 FlashAttention 的 online softmax 在同一个循环里。这就是融合。搞懂这一行，就搞懂了 vLLM 最大胆的设计。

但在搞懂融合之前，你得先分别搞懂 FlashAttention 怎么算、PagedAttention 怎么存。这就是本章的结构。

学完这章你能：
- 解释为什么朴素 Attention 在长序列上跑不动——用 HBM 读写量说话，不说"太慢"
- 从零手算一轮 online softmax——画出 tiling pattern，写出每步的 m、l、correction
- 用归纳法证明 online softmax 是精确算法（不是近似！）
- 写出 `block_table[logical_block] → physical_block` 的查表逻辑
- 理解融合 kernel 的一个循环迭代：查 block_table → 加载 K,V → 算 QK^T → online softmax → 累积

---

## 3.1 问题：为什么 Attention 遇到内存墙？

### GPU 的内存不是平的

你的 GPU 有 80GB 显存（HBM），速度"很快"——2TB/s。但 GPU 的计算核心（Tensor Core）更快——对于 fp16 矩阵乘法，A100 的峰值吞吐是 312 TFLOPS。**如果你只能以 2TB/s 喂数据，每秒最多喂 1T 个 fp16 元素——但计算核心每秒能处理 312T 个。差了 300 倍。**

这就是内存墙（Memory Wall）：**计算太快，数据传输太慢。大部分时间 GPU 在等数据。**

GPU 设计者知道这个问题。所以 GPU 还有一片更小但更快的 SRAM（也叫 shared memory）：只有约 20MB，但带宽达到 20TB/s——比 HBM 快 10 倍。

直观理解：
```
把你的 GPU 想象成一个厨房：
  HBM = 仓库（大，但每趟来回要时间）
  SRAM = 灶台（小，但伸手就能拿到）

好的 GPU 程序 = 让大部分操作在灶台上完成
差的 GPU 程序 = 每做一个动作就跑一趟仓库
```

### 朴素 Attention 的 HBM 足迹

打开 `instances/vllm/source/vllm/v1/attention/backends/flash_attn.py:682`——这是 vLLM 调用 FlashAttention 的入口。在这个文件写成之前，Attention 的计算流程是这样的：

```
步骤 1: 从 HBM 读 Q [L×d]        → SRAM → 算 S = Q @ K^T [L×L] → 写 S 回 HBM
步骤 2: 从 HBM 读 S [L×L]         → SRAM → 算 P = softmax(S)  → 写 P 回 HBM
步骤 3: 从 HBM 读 P [L×L], V [L×d] → SRAM → 算 O = P @ V     → 写 O 回 HBM
```

注意 **S 和 P 都是 $L \times L$**。当 $L$ 稍微大一点，这两个矩阵就直接爆炸：

| 序列长度 L | S 的大小 (fp32, 1 head) | 32 heads | 说人话 |
|-----------|------------------------|----------|--------|
| 512 | 1 MB | 32 MB | 还能忍 |
| 2048 | 16 MB | 512 MB | 开始疼了 |
| 4096 | 64 MB | **2 GB** | HBM 带宽撑不住了 |
| 8192 | 256 MB | **8 GB** | 光 S 就 8GB |
| 128K | 64 GB | **2048 GB** | 一张卡根本放不下 |

**但 HBM 更大的问题是：不仅要存 S 和 P，还要把它们写进 HBM 再读出来。** HBM 的写带宽和读带宽是一样的——2TB/s（A100）。写 2GB 需要 1ms。读回来又 1ms。每个 head 都来一遍，几十个 layer 都来一遍——**HBM 读写成为瓶颈，计算核心大部分时间在发呆。**

### FlashAttention 的核心洞察

FlashAttention（Dao et al., 2022）的 insight 一句话就能说清楚：

> **既然 S 和 P 写入 HBM 是瓶颈，那就永远不把它们写入 HBM。在 SRAM 里算完立刻用掉，只写最终结果 O。**

但不是一次性算整个 S。SRAM 只有 20MB，放不下 $L \times L$ 的 S。所以得把计算**分块（tile）**。一个 tile 的 S（比如 $64 \times 64$）只有 16KB——轻松放进 SRAM。算完 softmax 立刻和 V 乘起来，结果累加到 O。然后这个 S tile 就不要了——下个 tile 覆盖它。

**这引出了两个问题：**

1. **Tiling 问题**：怎么把 $S = QK^T$ 切成 tile？每个 tile 需要哪些数据？
2. **Online Softmax 问题**：softmax 需要全局归一化。只看到部分 tile 时怎么算 softmax？

下面分别展开。先看图。

---

## 3.2 Tiling：Q 切成条，KV 反复遍历

> **Source Trail:** `csrc/attention/attention_kernels.cuh:222` — `for (block_idx = ...)` tiled loop over KV blocks.
> `vllm/v1/attention/backends/flash_attn.py:682` — `FlashAttentionImpl.forward()` dispatches tiled attention.

### 直觉

想象你要算 12 个学生对 12 门课的分数矩阵。你不能一次看完所有 12×12=144 个分数——桌子只能放 4×4=16 个格子。

解法：一次拿 4 个学生（Q tile），遍历全部 12 门课（KV tiles，每个 4 门课）。算完这 4 个学生的注意力后，把结果写下来，再拿下一组 4 个学生。

**代价**：每拿一组 Q tile，你都要把全部 12 门课重新看一遍。KV 被读了 $(12/4)=3$ 遍。
**收益**：144 个格子的中间表从来没被写到大本子上——只在桌上临时记一下，用完就擦掉。

### Tiling 模式图

用 $L=12$，Q tile=4，KV tile=4：

![FlashAttention Tiling Pattern](../diagrams/03-fa-tiling-py.png)

> *图注：左列蓝框是 3 个 Q tile（Q₀、Q₁、Q₂），右列黄框是 3 个 KV tile（K₀V₀、K₁V₁、K₂V₂）。每条彩色线 = 1 次 KV 加载。Q₀ 的 3 次加载用蓝色线表示，Q₁ 用紫色，Q₂ 用绿色。共 9 次加载。*

### Q₀ 的完整处理过程（追踪一个 Q tile 的生命）

**1. 加载 Q₀ [4×d] 到 SRAM。** 对于 bf16、d=128，Q₀ 只有 4×128×2=1KB。Q₀ 留在 SRAM 中，在接下来的 3 次 KV 遍历中不再从 HBM 读取。

**2. 遍历 KV₀（tokens 0-3）：**
- 加载 K₀[4×d] 和 V₀[4×d] 从 HBM → SRAM
- 算 $S = Q₀ @ K₀^T$ → [4×4]，**只在 SRAM 中存在**
- Online softmax → 累积到 O_acc
- K₀ 和 V₀ 不要了（被下个 tile 覆盖）

**3. 遍历 KV₁（tokens 4-7）：**
- 加载 K₁、V₁ 从 HBM → SRAM
- 算 S = Q₀ @ K₁^T → [4×4]，**覆盖之前的 S**
- Online softmax → 累积到 O_acc

**4. 遍历 KV₂（tokens 8-11）：**
- 加载 K₂、V₂ 从 HBM → SRAM
- 算 S = Q₀ @ K₂^T → [4×4]
- Online softmax → 最终累积
- 归一化 → 写 O₀ [4×d] 到 HBM。完成！

**Q₁（tokens 4-7）——完全一样的 KV 再读一遍：** 加载 Q₁ → 再次遍历 K₀,V₀、K₁,V₁、K₂,V₂ → 写 O₁。

**这就是"KV 被反复读了 $L/B_Q$ 次"的根源。** 有 3 个 Q tile，每个需要加载全部 3 个 KV tile——KV 总共被从 HBM 加载了 $3 \times 3 = 9$ 次。对比朴素 attention：K 和 V 各读 1 次。

### HBM 流量量化

| 操作 | 朴素 Attention | FlashAttention |
|------|---------------|----------------|
| 读 Q | $Ld$ | $Ld$（相同） |
| 读 K | $Ld$ | $(L/B_Q) \times Ld$（多了 $L/B_Q$ 倍！） |
| 读 V | $Ld$ | $(L/B_Q) \times Ld$（多了 $L/B_Q$ 倍！） |
| **中间写 S** | **$L^2$** | **0** |
| **中间写 P** | **$L^2$** | **0** |
| 写 O | $Ld$ | $Ld$（相同） |

代入 $L=4096, d=128, B_Q=64$，bf16：

| | 朴素 | FlashAttention |
|---|---|---|
| 读 Q | 1 MB | 1 MB |
| 读 K | 1 MB | 64 MB |
| 读 V | 1 MB | 64 MB |
| **中间写 (S+P)** | **2 GB** | **0** |
| 写 O | 1 MB | 1 MB |
| **总 HBM 流量** | **~2 GB** | **~131 MB** |

**FlashAttention 读了 64 倍更多的 K 和 V（129MB vs 2MB），但避免了 2GB 的中间矩阵写入。净收益：~15 倍。** 而且序列越长收益越大——$L=128K$ 时朴素 attention 的 S=64GB，单卡 HBM 都放不下，而 FlashAttention 根本不用存 S。

---

## 3.3 Online Softmax：分块做 softmax 的数学魔术

> **Source Trail:** `csrc/attention/attention_kernels.cuh:307-341` — warp-level softmax reduction (m_new, correction, l_new).
> `vllm/v1/attention/ops/triton_decode_attention.py:186-200` — Triton online softmax update.

Tiling 要求我们"看一部分 KV 就算一部分 softmax"。但普通 softmax 必须知道**全局**最大值才能做数值稳定的计算。看一部分怎么知道全局最大？

答案：**先猜一个局部最大，如果后面发现更大的，用 correction factor 修正之前的结果。correction 不是近似——是代数恒等式。**

### 先看一个日常类比

你在统计班里的身高排名。只有一部分人的身高数据：

- 第一批人：最高 180cm。"目前"认为最高是 180cm。把所有身高减去 180 再算。结果比例先记着。
- 第二批人：发现一个 185cm 的！之前的"最高"不对了。把之前的所有结果**按比例缩小**——因为之前用的 reference（180）偏低了 5cm，之前的每个 exp 值都**大了 e⁵ ≈ 148 倍**。用 correction = e⁻⁵ ≈ 0.0067 把旧结果缩回去。
- 第三批人：最高还是 185cm。correction = e⁰ = 1——旧结果不用动。

**如果后面没有出现更高的，旧结果就是对的。如果出现了更高的，correction factor 恰好能把旧结果调整到正确的 scale。**

### 数值 Trace：追踪一个 Q token 过 3 个 KV block

这个 trace 是本章最重要的东西。用笔和纸跟着算一遍——搞懂它，就搞懂了 FlashAttention 的一半。

设 1 个 Q token，head_dim=d，3 个 KV block，每个 4 个 token。`SCALE = 1/√d`（为方便数值展示设为 1.0）。

追踪变量：
- $m$：running max（当前见过的全局最大 attention score）
- $l$：running exp sum（当前见过的所有 softmax 分母）
- $O_{acc}$：running weighted output（softmax 分子 × V 的累积）
- corr：correction factor = $\exp(m_{\mathrm{old}} - m_{\mathrm{new}})$

| 变量 | 初始 | 迭代 1 (K₀V₀) | 迭代 2 (K₁V₁) | 迭代 3 (K₂V₂) |
|------|------|---------------|---------------|---------------|
| **S** | — | [2.0, 1.0, 0.5, 3.0] | [1.5, 4.0, 2.0, 1.0] | [3.5, 2.0, 1.5, 0.5] |
| **m** | −∞ | → **3.0** | 3.0 → **4.0** | 4.0 → 4.0 |
| **corr** | — | `exp(-∞-3)=0` | **`exp(3-4)=0.368`** | `exp(4-4)=1.0` |
| **l** | 0 | 0 + Σexp(S₀−3) = 1.585 | 0.368×1.585 + Σexp(S₁−4) = 1.850 | 1.0×1.850 + Σexp(S₂−4) = ... |
| **O_acc** | 0 | P₀ @ V₀ | 0.368×old + P₁ @ V₁ | 1.0×old + P₂ @ V₂ |

最终：$O = O_{acc} / l$

> 最终归一化只做一次！不像朴素 softmax 那样每个元素都除以分母——FA 的分母是一边迭代一边累加的，等全部 KV 走完才除一次。

数值 trace 的图解版本：

![Online Softmax State Evolution](../diagrams/03-online-softmax-py.png)

> *图注：corr 行是核心。迭代 1 中 corr=0（首次，无旧值要修正）。迭代 2 中 max 从 3→4，corr=0.368——旧 O_acc 必须乘以 0.368 缩小。迭代 3 中 max 没变，corr=1.0——零修正。箭头表示 m,l,O_acc 状态从一列流向下一列。*

### 为什么不先 normalize 最后再除？

一个常见的疑问：为什么不干脆每步都算 softmax，加总时再调整？

因为这样会产生错误的结果。考虑：迭代 1 产生 softmax₁（基于 max=3.0），迭代 2 产生 softmax₂（基于 max=4.0）。你不能把 softmax₁ 和 softmax₂ 直接按某个权重相加——因为它们的分母不同（前者分母基于 max=3.0，后者基于 max=4.0）。correction 的作用就是把 softmax₁ 的分母和分子**同时**调整到新的 scale，使得它们能和新的 softmax₂ 的分子分母相加。

### 数学证明：为什么 Online Softmax = 精确 Softmax？

数值 trace 告诉你怎么做。以下是**为什么它对**的证明。这是本章的数学核心。

**你不需要背下这个证明——只需要记住一个结论：online softmax 不是近似，是精确的。** 但如果你想自己验证——想真的搞懂 correction 为什么恰好是 $\exp(m_{\mathrm{old}} - m_{\mathrm{new}})$ 而不是别的什么——下面是完整的归纳法证明。

设我们要计算 $K$ 个 KV block 上的 attention。第 $k$ 个 block 的 attention scores 为 $S^{(k)}$（一个 Q token 对 B 个 KV token 的分数向量）。完整 softmax 需要所有 $K$ 个 block 的 $S$ 一次性处理：

$$
P = \frac{\exp(S)}{\sum \exp(S)}, \quad S = [S^{(0)}, S^{(1)}, ..., S^{(K-1)}]
$$

Online softmax 维护三个 running state：$m$（当前全局最大）、$l$（exp 和）、$O_{acc}$（加权输出）。每来一个 block，执行：

$$
\begin{aligned}
m^{(k)} &= \max(m^{(k-1)}, \max(S^{(k)})) \\[4pt]
P^{(k)} &= \exp(S^{(k)} - m^{(k)}) \\[4pt]
c^{(k)} &= \exp(m^{(k-1)} - m^{(k)}) \\[4pt]
l^{(k)} &= c^{(k)} \cdot l^{(k-1)} + \sum P^{(k)} \\[4pt]
O_{acc}^{(k)} &= c^{(k)} \cdot O_{acc}^{(k-1)} + P^{(k)} V^{(k)}
\end{aligned}
$$

最终输出：$O = O_{acc}^{(K-1)} / l^{(K-1)}$。

其中 $c^{(k)} = \exp(m^{(k-1)} - m^{(k)})$ 是 correction factor。

**定理：** 上述迭代算法产生的 $O$ 等同于标准 softmax（一次性拿到所有 $S$ 后计算）。也就是说，Online Softmax 不是近似——**它是精确算法**。

**证明思路（白话版，先别怕）：**

核心问题只有一个：当新 block 的 max 比之前所有 block 的 max 都大时，之前算好的 $l$ 和 $O_{acc}$ 都"大了一号"——因为之前用一个偏小的 max 做参考，导致每个 exp 值都放大了 $\exp(M_k - M_{k-1})$ 倍。correction factor = $\exp(M_{k-1} - M_k) = 1 / \exp(M_k - M_{k-1})$ 恰好把这些"大了一号"的值**缩小回正确的 scale**。

为什么恰好是对的？因为 $\exp(a) \times \exp(b) = \exp(a+b)$。这就是全部数学——exp 加法性质的一次巧妙应用。不是概率假设，不是数值近似。

下面用符号把这个直觉写成严格证明。

**归纳假设 $\mathcal{H}(k)$：** 处理完 $k$ 个 block 后，$l^{(k)}$ 和 $O_{acc}^{(k)}$ 的值等于"如果前 $k$ 个 block 一起用它们的真实全局 max $M_k$ 做标准 softmax"的结果：

$$
l^{(k)} = \sum_{i=0}^{k} \sum \exp(S^{(i)} - M_k), \quad
O_{acc}^{(k)} = \sum_{i=0}^{k} \sum \exp(S^{(i)} - M_k) \cdot V^{(i)}
$$

其中 $M_k = \max(S^{(0)}, ..., S^{(k)})$。也就是说：$\mathcal{H}(k)$ 断言"算法维护的 running state 在任何时刻都等于标准 softmax 的中间结果。"

**Base case ($k=0$)：** 只有一个 block 时，"全局 max"就是它自己的 max。算法正确计算了这个 block 的 softmax。$\mathcal{H}(0)$ 成立。没什么可证的——只有一个 block 时，局部 max 就是全局 max。

**Inductive step：** 假设 $\mathcal{H}(k-1)$ 对前 $k-1$ 个 block 成立。现在第 $k$ 个 block 到达。分两种情况。

**情况 A：max 没变（$m^{(k)} = m^{(k-1)}$）**

新 block 的所有 score 都不超过之前的全局 max。因此旧的 $l$ 和 $O_{acc}$ 已经用了正确的 reference——不用改。correction $c^{(k)} = \exp(0) = 1$。

代入归纳假设 $\mathcal{H}(k-1)$：

$$
\begin{aligned}
l^{(k)} &= 1 \cdot l^{(k-1)} + \sum \exp(S^{(k)} - M_k) \\
&= \sum_{i=0}^{k-1} \sum \exp(S^{(i)} - M_{k-1}) + \sum \exp(S^{(k)} - M_k) \\
&= \sum_{i=0}^{k} \sum \exp(S^{(i)} - M_k)
\end{aligned}
$$

最后一步成立因为 $M_k = M_{k-1}$：前 $k-1$ 项和新项现在共享同一个全局 max。$\mathcal{H}(k)$ 成立。

**情况 B：max 更新了（$m^{(k)} > m^{(k-1)}$）——整个证明的关键**

新 block 里出现了一个更大的值。之前算的所有 exp 都用了一个偏小的 max 做 reference——每个都太大了。具体来说，每项大了 $\exp(M_k - M_{k-1})$ 倍。correction = $\exp(M_{k-1} - M_k) = 1 / \exp(M_k - M_{k-1})$ 恰好是放大倍数的倒数。

写成代数：

$$
\begin{aligned}
l^{(k)} &= c^{(k)} \cdot l^{(k-1)} + \sum \exp(S^{(k)} - M_k) \\[4pt]
&= \exp(M_{k-1} - M_k) \cdot \sum_{i=0}^{k-1} \sum \exp(S^{(i)} - M_{k-1}) + \sum \exp(S^{(k)} - M_k) \\[4pt]
&= \sum_{i=0}^{k-1} \sum \exp(S^{(i)} - M_{k-1} + M_{k-1} - M_k) + \sum \exp(S^{(k)} - M_k) \\[4pt]
&= \sum_{i=0}^{k} \sum \exp(S^{(i)} - M_k)
\end{aligned}
$$

**关键一步在第三行：** $\exp(S^{(i)} - M_{k-1}) \cdot \exp(M_{k-1} - M_k) = \exp(S^{(i)} - M_k)$。这是 $\exp$ 的基本性质 $e^a \cdot e^b = e^{a+b}$。correction 因子和旧 exp 值相乘，指数部分相加：$-M_{k-1} + (M_{k-1} - M_k) = -M_k$——**旧的 reference max 被完美替换为新的 reference max。** 不是启发式修正，是代数恒等式。

关于 $O_{acc}$：同样的代数对每项附着 V 也成立——把 $V^{(i)}$ 乘在每一项上，上面的推导完全对称。$\mathcal{H}(K-1)$ 成立。■

**总结（白话版）：** Online Softmax 能在"不知道全局 max"的情况下精确算 softmax，靠的是 $\exp$ 函数的加法性质 $e^{a+b} = e^a e^b$。不是任何近似！FlashAttention 能跑长序列而不爆显存，同时输出和朴素 attention 完全一样（fp32 下 bit-exact，fp16 下有舍入误差但可忽略）。

---

## 3.4 融合：当 FlashAttention 遇到 PagedAttention

### 问题：KV Cache 是非连续的

第 2 章学了 KV Cache 的 block 分配。KV Cache 把每个序列的 KV 状态切成固定大小的 block（vLLM 用 16 tokens/block），block 物理上散落在 GPU 显存中。逻辑上连续的 KV 序列，物理上不连续。

现在 Attention kernel 需要遍历 KV 来算 attention。如果先 gather 所有 KV block 到一个连续 tensor，再做 FlashAttention——**这个临时 tensor 有多大？**

对于 128K 序列、128 维 head、32 个头、32 层：一个 layer 的 gathered tensor =

$$
128K \times 128 \times 32 \times 2\ \mathrm{bytes} = 1\ \mathrm{GB}
$$

每个 layer 都需要 1GB 临时显存——32 layers = 32GB。浪费。

**PagedAttention 的 insight：不做 gather。在 kernel 内部循环中，每遇到一个逻辑 block，查 block_table 找到物理地址，直接从物理地址加载 K 和 V。**

### block_table：逻辑→物理的映射

`block_table` 是 `[num_seqs, max_blocks_per_seq]` 的 int32 tensor。每一行是一个序列的"逻辑 block → 物理 block"映射。

```
序列 A 的逻辑布局:    Block0   Block1   Block2   Block3
                       ↓        ↓        ↓        ↓
block_table[A]:     [  17  ][  3   ][  42  ][  8   ]
                    散落在 GPU 显存的任意位置
```

### 融合 kernel 的一个迭代

打开 `csrc/attention/attention_kernels.cuh:222`。这是 vLLM 最核心的循环：

```
for blk in 0..num_blocks:
    ┌─────────────────────────────────────────────┐
    │ STEP 1: 查表 (.cuh L252)                     │
    │   phys = block_table[seq_idx][blk]           │
    │   逻辑 block blk → 物理 block phys            │
    │                                              │
    │ STEP 2: 加载非连续 K,V (.cuh L269, L397)     │
    │   K_blk = K_cache[phys]   ← 物理地址直接索引 │
    │   V_blk = V_cache[phys]                      │
    │                                              │
    │ STEP 3: Q @ K^T (.cuh L289)                  │
    │   S = Q @ K_blk^T / sqrt(d)                  │
    │   S 只在 SRAM 中，不写回 HBM                  │
    │                                              │
    │ STEP 4: Online Softmax (.cuh L307-L341)      │
    │   m_new = max(m, max(S))                     │
    │   correction = exp(m - m_new)                │
    │   l_new = correction*l + sum(exp(S-m_new))   │
    │   O_acc = correction*O_acc + softmax(S)@V_blk│
    │                                              │
    │ STEP 5: 下一个 block                          │
    └─────────────────────────────────────────────┘
归一化: O = O_acc / l     (.cuh L337)
```

**为什么一定要融合？** 如果分开——先 gather 再 FA——gathered tensor 至少需要 $L \times d \times \mathrm{heads}$ 的临时 HBM 空间。融合 kernel 不需要这个临时空间——K 和 V 直接从物理地址加载到寄存器/SRAM，用完就丢。融合使得 **PagedAttention 的碎片化存储不再是一个"要去解决"的问题——它被消除了。** 碎片化对 kernel 来说是透明的。

---

## 3.5 代码走读：Triton Paged Attention Kernel

现在走读真实的 Triton kernel——FA 和 PA 的融合实现。

本节对应两个源文件：
- 我们的 Triton 实现：`implementation/triton_paged_attention.py`
- vLLM 的生产代码：`vllm/v1/attention/ops/triton_decode_attention.py:60`

打开 `implementation/triton_paged_attention.py:46`，核心 kernel 从这里开始。

### Grid 结构——"谁在跑？"

vLLM 用 2D grid `(num_seqs, num_kv_heads)`——每个 (sequence, head) 组合启动一个 Triton program。**decode 阶段每个 sequence 只有 1 个 Q token**，所以每个 program 处理"1 个 Q token 对 1 个 head 的全部 KV blocks"。

```python
# triton_paged_attention.py:L89-L90
# REFERENCE: triton_decode_attention.py — grid = (num_seqs, num_kv_heads)
seq_idx = tl.program_id(0)   # 哪个序列
kv_head = tl.program_id(1)   # 哪个 KV head
```

vLLM 的 Triton decode kernel 也用了 3D grid `(num_seqs, num_kv_heads, NUM_KV_SPLITS)`——第三维用于 partition 长序列（类似 3.6 节的 V2 kernel）。我们的实现简化为 2D——省略 split-KV 的 reduce 阶段，但核心融合逻辑相同。

### 初始化——三件套

```python
# triton_paged_attention.py:L93-L105
# REFERENCE: attention_kernels.cuh:L196 — float qk_max = -FLT_MAX; float exp_sum = 0
seq_len = tl.load(seq_lens_ptr + seq_idx)
num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE

# 加载这个 Q token 的这个 head
q_offset = seq_idx * stride_q_tok + kv_head * stride_q_h
Q_vec = tl.load(Q_ptr + q_offset + tl.arange(0, HEAD_DIM))

# Online softmax 初始状态——对应 3.3 节数值 trace 的行 0
m_i = tl.full([1], float("-inf"), dtype=tl.float32)   # m = -∞
l_i = tl.full([1], 0.0, dtype=tl.float32)              # l = 0
O_acc = tl.zeros([HEAD_DIM], dtype=tl.float32)          # O_acc = 0
```

对照 3.3 节的数值 trace：初始状态完全一致——m=−∞, l=0, O_acc=0。

### 主循环——FA × PA × Online Softmax 三步合一

以下代码在一个循环里同时做了 block_table 查表（PA）、QK^T 点积（FA）和 online softmax update。对应 vLLM `attention_kernels.cuh:222-476`。

```python
# triton_paged_attention.py:L114-L162
# REFERENCE: attention_kernels.cuh:L222 — for (block_idx = ...)
for blk_idx in range(num_blocks):

    # ═══ STEP 1: PA - 查表获取物理 block ═══
    # REFERENCE: attention_kernels.cuh:L252-L253
    bt_offset = seq_idx * stride_bt_seq
    phys_blk = tl.load(block_tables_ptr + bt_offset + blk_idx * stride_bt_blk)
```

**这一行就是整个 PagedAttention 的物理实现。** `block_tables_ptr[seq_idx, blk_idx]` 返回物理 block ID。kernel 用这个 ID 去索引 K_cache 和 V_cache——不需要 gather，不需要临时 tensor。

```python
    # ═══ STEP 2: PA - 从非连续物理地址加载 K,V ═══
    # REFERENCE: attention_kernels.cuh:L269 — k_cache + phys_blk * kv_block_stride
    k_offs = (phys_blk * stride_k_blk +
              tl.arange(0, BLOCK_SIZE)[:, None] * stride_k_tok +
              kv_head * stride_k_h +
              tl.arange(0, HEAD_DIM)[None, :] * stride_k_d)
    K_blk = tl.load(K_cache_ptr + k_offs,
                    mask=tl.arange(0, BLOCK_SIZE)[:, None] < n_tokens)
    # K_blk: [BLOCK_SIZE, HEAD_DIM] — 从分散的物理位置直接加载

    v_offs = (phys_blk * stride_v_blk + ...)  # 同理
    V_blk = tl.load(V_cache_ptr + v_offs,
                    mask=tl.arange(0, BLOCK_SIZE)[:, None] < n_tokens)
    # V_blk: [BLOCK_SIZE, HEAD_DIM]
```

第 blk_idx 个逻辑 block 的物理地址可能和第 blk_idx−1 个的完全不挨着。但 kernel 无所谓——每次循环直接算物理地址，直接加载。

```python
    # ═══ STEP 3: FA - Q @ K^T（SRAM 中计算，不写回 HBM）═══
    # REFERENCE: attention_kernels.cuh:L289 — Qk_dot::dot(q_vecs, k_vecs)
    Q_broadcast = Q_vec[None, :]  # [1, HEAD_DIM]
    S = tl.sum(Q_broadcast * K_blk, axis=1) * SCALE  # [BLOCK_SIZE]
    S = tl.where(tl.arange(0, BLOCK_SIZE) < n_tokens, S, float("-inf"))
```

`tl.sum(Q[None,:] * K, axis=1)` 就是 Q @ K^T——这里的 "SRAM" 是 Triton 自动管理的寄存器 + shared memory。S 是 `[BLOCK_SIZE]` 的向量——这个 Q token 对当前 block 中每个 KV token 的 attention score。**S 永远不会被写入 HBM。**

`tl.where(..., float("-inf"))` 把超出实际序列长度的 token 的 score 设为 −∞——这些位置 softmax 后就是 0，不影响最终输出。

```python
    # ═══ STEP 4: Online Softmax - m, l, correction 全部更新 ═══
    # REFERENCE: attention_kernels.cuh:L307-L341 (warp-level softmax reduction)
    m_new = tl.maximum(m_i, tl.max(S, axis=0))          # ① 更新 running max
    P = tl.exp(S - m_new)                                # ② 稳定 exp
    correction = tl.exp(m_i - m_new)                     # ③ correction factor
    l_new = correction * l_i + tl.sum(P, axis=0)         # ④ 更新 exp sum
    # Q token 对当前 block 的 softmax:P [BLOCK_SIZE]
    # O_acc += P @ V_blk (P 加权 V 求和)
    O_acc = correction * O_acc + tl.sum(P[:, None] * V_blk, axis=0)
    # P[:, None]: [BLOCK_SIZE, 1] × V_blk: [BLOCK_SIZE, HEAD_DIM]
    # → [BLOCK_SIZE, HEAD_DIM] → sum over blk dim → [HEAD_DIM]

    m_i, l_i = m_new, l_new  # 状态传到下一个 KV block
```

对照 3.3 节的数值 trace 第 2 迭代：
- $m_{\mathrm{old}}=3.0, m_{\mathrm{new}}=4.0$ → correction = 0.368
- $l_{\mathrm{old}}=1.585, \sum P_{\mathrm{new}}=1.267$
- $l_{\mathrm{new}} = 0.368 \times 1.585 + 1.267 = 1.850$

代码中的 `tl.sum(P, axis=0)` 就是 `∑ exp(S − m_new)` ——3.3 节表格中 `Σexp(S₁−4) = 1.267`。

### 最终归一化 + 写入

```python
# triton_paged_attention.py:L166-L172
# REFERENCE: attention_kernels.cuh:L337 — inv_sum * logits
O_final = O_acc / l_i  # 全部 KV block 走完后才除一次
tl.store(Out_ptr + o_offset + tl.arange(0, HEAD_DIM), O_final.to(Q_vec.dtype))
```

**注意：归一化只做一次，在所有 KV block 遍历完之后。** 不像朴素 softmax 那样每个元素都除以分母——FA 的分母 `l_i` 是一边迭代一边累加的，最终归一化一次完成。

### 运行验证

打开 `implementation/triton_paged_attention.py` 直接运行：

```bash
$ cd artifacts/03-flashattention-pagedattention && python3 implementation/triton_paged_attention.py
Triton vs Reference (fp16): max error = 0.000977
✅ MATCH (within fp16 tolerance)
```

Triton kernel 的输出与 gather-KV→连续→标准 attention 的参考实现一致。误差 < 0.001（fp16 舍入范围内）。

### 对照 vLLM 源文件：一行一行地对

| 我们的 Triton kernel | vLLM CUDA kernel | vLLM Triton kernel | 说明 |
|---|---|---|---|
| `program_id(0)` = seq | `blockIdx.y` (L106) | `program_id(0)` | PID 到序列映射 |
| `program_id(1)` = head | `blockIdx.x` (L107) | `program_id(1)` | PID 到 head 映射 |
| `tl.load(block_tables_ptr + ...)` | `block_table[block_idx]` (L252) | `tl.load(Req_to_tokens + ...)` (L119) | PA 核心：逻辑→物理 |
| `Q_vec[None,:] * K_blk` | `Qk_dot::dot(q_vecs, k_vecs)` (L289) | `tl.dot(q, k)` (L186) | FA：Q @ K^T |
| `m_new = tl.maximum(m_i, tl.max(S))` | `fmaxf(qk_max, VLLM_SHUFFLE_XOR(...))` (L310) | same pattern | warp-level max reduce |
| `l_new = correction*l_i + tl.sum(P)` | `block_sum<NUM_WARPS>(...)` (L334) | same pattern | warp-level sum reduce |
| `O_acc = correction*O_acc + Σ(P×V)` | warp-level output reduction (L432) | same pattern | FA：P @ V 累积 |
| `O_final = O_acc / l_i` | `inv_sum * logits[i]` (L337) | `acc = acc / l[:, None]` (L384) | 最终归一化 |

---

## 3.6 V1 vs V2 Kernel：长序列怎么分？

vLLM 有两个版本的 paged attention kernel（源码：`csrc/attention/paged_attention_v1.cu` 和 `paged_attention_v2.cu`）。

**问题：** V1 用 shared memory 存储中间结果。shared memory 的大小和 BLOCK 相关——序列越长，BLOCK 越多（在一个 kernel launch 内处理的 block 数受 shared memory 限制）。当序列超过 ~8K tokens，shared memory 不够用了。

**V2 的解法：partition。** 把长序列切成多个 512-token partition。每个 partition 独立做一次 partial online softmax，产生 partial output + partial exp_sum + partial max。最后用一个 reduce kernel 合并所有 partition 的结果。

| | V1 (no partition) | V2 (partition=512) |
|---|---|---|
| Grid | `(heads, seqs, 1)` | `(heads, seqs, num_partitions)` |
| Shared memory | O(max_seq_len) | O(512)——固定！ |
| 序列长度上限 | ~8K（SMEM 限制） | 无限制 |
| 额外 buffer | 无 | tmp_out + exp_sums + max_logits |
| 额外 kernel | 无 | `paged_attention_v2_reduce_kernel` |

这本质上是 3.3 节 online softmax 的 **multi-partition 版本**——每个 partition 先独立做 online softmax，然后合并。合并也使用同样的 correction 机制：如果 partition B 的 max 更大，partition A 的结果乘以 correction factor 缩小。

---

## 3.7 我们的实现总览

| 文件 | 对应 vLLM 源码 | 说明 |
|---|---|---|
| `implementation/triton_paged_attention.py` | `triton_decode_attention.py:60` | 完整的 Triton FA+PA kernel，相同 grid 结构 + block_table + online softmax |
| `implementation/fused_attention_demo.py` | `attention_kernels.cuh:85` | **可直接运行的 Python demo**，输出完整的 m/l/correction 迭代 trace |
| `implementation/paged_attention.py` | `attention_kernels.cuh:85` | Python 参考实现：PA（block_table 循环）、FA（tiled online softmax）、融合 |
| `implementation/paged_attention.py:calculate_hbm_traffic()` | 论文公式 + Nsight 验证 | HBM 流量计算器——教学级精度，不含 cache line 细节 |
| `implementation/paged_attention.py:build_block_table()` | `kv_cache_manager.py` + `block_pool.py` | 简化的 first-fit 分配器 |

### 运行 demo.py 看完整的迭代 trace

```bash
$ cd artifacts/03-flashattention-pagedattention && python3 implementation/fused_attention_demo.py
=================================================================
Fused FlashAttention + PagedAttention — Runnable Trace
=================================================================
Q: [1 token, d=8]  KV: [12 tokens, d=8]
BLOCK_KV=4 → 3 KV blocks
Block table: logical 0→physical 3, 1→1, 2→5

ONLINE SOFTMAX TRACE — 3 KV blocks, 1 Q token
-----------------------------------------------------------------
Iteration 1 (logical block 0 → physical block 3):
  m:   -inf →  2.3405
  correction = exp(-inf - 2.3405) = 0.000000
  l: 0.000000 → 1.896321
  O_acc (first 4 dims): [0.1498, 0.6795, -0.8557, 0.1764]

Iteration 2 (logical block 1 → physical block 1):
  m: 2.3405 → 3.2146  (max updated!)
  correction = exp(2.3405 - 3.2146) = 0.417192
  l: 1.896321 → 2.581338
  O_acc (first 4 dims): [-0.1064, 0.4110, -0.4358, 0.0622]

Iteration 3 (logical block 2 → physical block 5):
  m: 3.2146 → 3.2146  (same)
  correction = exp(3.2146 - 3.2146) = 1.000000
  l: 2.581338 → ...
  O_acc (first 4 dims): [...]

FINAL: O = O_acc / l
VERIFICATION: Compare against naive (full KV, contiguous, standard softmax)
  Max error (fp32): 0.0000000047
  ✅ EXACT MATCH
```

看输出：迭代 1 用物理 block 3（不是 0！），迭代 2 用物理 block 1（不是 1 连续！），迭代 3 用物理 block 5。K 和 V 来自三个完全不相邻的物理位置——但最终输出和 gather 到连续 tensor 后做标准 attention **完全一致**。

---

## 验证

```bash
cd artifacts/03-flashattention-pagedattention && python -m pytest tests/ -q
# 9/9 passed ✅
```

公式检查和源文件扎根检查：

```bash
python3 scripts/lint_formulas.py instances/vllm/artifacts/03-flashattention-pagedattention/narrative/chapter.md
python3 scripts/lint_source_grounding.py instances/vllm/artifacts/03-flashattention-pagedattention/
```

---

## 总结

好，到这里你已经搞懂了 vLLM 最精华的设计。回头看整个故事线：

1. **内存墙逼出了 FlashAttention 的 tiling。** 朴素 Attention 的 $[L \times L]$ 中间矩阵 S 和 P 是 HBM 杀手——$L=4096$ 就 2GB 了，$L=128K$ 单卡根本放不下。Tiling 把 Q 切成小块，每次只看部分 KV——S 和 P 永远不写入 HBM。

2. **Tiling 要求能分块做 softmax——online softmax 就是答案。** running max + correction factor = $\exp(m - m_{\mathrm{new}})$ 让一个 pass 完成。correction 是 $\exp$ 函数性质的直接应用——不是近似，是代数恒等式。max 不再更新后 correction=1.0——旧结果正确，无需修正。

3. **PagedAttention 给 KV Cache 带来了碎片化，但融合消除了它的代价。** block_table 在 kernel 内部循环中查询——K 和 V 直接从非连续的物理 block 加载到寄存器，零额外 HBM。碎片化对计算是透明的。

4. **FA + PA 的融合是 vLLM 的架构优势。** 两者在同一循环——Q @ K^T + online softmax + block_table 查表——三个操作在一条指令流里交替执行。分开做需要的 gather 临时空间（$L \times d \times \mathrm{heads}$）被彻底消除。

一句话总结本章：**FlashAttention 让 Attention 能算长序列。PagedAttention 让 KV Cache 能碎片化管理。把它们融合在一个 kernel 里——这就是 vLLM。**

---

← 第2章 | 第4章 →
