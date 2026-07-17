# ch42 论文包 —《FlashAttention：在线 softmax 与分块遍历》

> 本章定位：**全书最后一个原理章（primer），实战原理高潮**。前面第 2 章立起「访存 vs 算力」三把性能判据尺，第 27 章讲透了 MMA / Tensor Core 的布局，第 29 章讲透了软件流水线怎么用异步 load 藏延迟，第 34 章讲共享内存。到本章，把这些零件拼成 Triton 官方的旗舰 kernel——`python/tutorials/06-fused-attention.py` 里那个 FlashAttention 前向核。它要回答的核心问题是：**朴素 attention 要物化一个 $`N\times N`$ 的注意力打分矩阵（$`O(N^2)`$ 显存，长序列直接爆），FlashAttention 怎么做到既不物化整张矩阵、又算出数学上完全相同的结果，还更快？** 答案是两件事叠在一起：**在线 softmax（online softmax）**——一遍扫过、边扫边更新 running max / running sum，不用先算完整一行再做 softmax；**分块遍历（tiling）**——把 K/V 切块，外层锁定一块 Q、内层逐块遍历 K/V，每来一块就用在线 softmax **增量更新**输出累加器，关键动作是 **rescale**：新块把 running max 抬高了，就把旧累加器整体乘 $`e^{m_{\mathrm{old}}-m_{\mathrm{new}}}`$ 补偿回来。看懂 `tutorials/06` 内层循环里 `m_i` / `l_i` / `acc` / `alpha` 这套记账，就看懂了 FlashAttention。这是第 43 章收官实战（把这个 fused-attention 核从 `tl.*` 一路走到 PTX）的原理地基。
>
> **本章的主真相源是 pin v3.2.0 源码（A 档）**：`python/tutorials/06-fused-attention.py`（641 行）是 **Triton 官方 FlashAttention v2 kernel 的真实实现**——FlashAttention 算法就逐行印在这段代码里，是本 primer 最权威、逐字可核的抓手。前向核的在线 softmax 递推与 rescale（`_attn_fwd_inner` 内层循环 `:46–77`、`_attn_fwd` 的初始化与收尾 `:154–189`）**逐字可核**；文件顶的 docstring（`:5`、`:11–12`）**逐字写出**了 FA2 / FA1 / Rabe-Staats 的论文链接，是 source-cited 出处。四篇论文（FlashAttention arXiv:2205.14135、FlashAttention-2 arXiv:2307.08691、online-softmax Milakov & Gimelshein arXiv:1805.02867、lazy-softmax Rabe & Staats arXiv:2112.05682）只作**学术出处（C 档）**。
>
> **红线：只写已核实内容。** A 档一切引文都能在标注的 `06-fused-attention.py:行号` 处逐字核对。**在线 softmax 递推与 FlashAttention 的 rescale 恒等性——这两件事可以从 `tutorials/06` 代码逐行逐字推出，属可核**（本章的数学推导以源码为准，论文只作学术出处）。而论文里的**具体定理编号、复杂度证明、FA-2 相对 FA-1 的具体加速比数字、Rabe-Staats 的 $`O(\log N)`$ 内存证明细节**——本包组装环境**无网络 / WebFetch**，无法逐字核实 arXiv 论文内文，凡涉及一律**标「待核·回指 arXiv:xxxx」，绝不编造论文里的具体定理 / 公式编号 / 加速比**（承 exp-0715-1）。docstring 里逐字写出的论文 URL 属 source-cited，可照引其存在与定位；论文内文细节待核。

---

## 0. 来源层级表（防越档编造）

| 档 | 含义 | 本章用到的具体来源 | 用法 |
|---|---|---|---|
| **A** | pin 源码逐字（最高权威，本章主真相源） | `python/tutorials/06-fused-attention.py`：**顶部 docstring**（`:5` 逐字「This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao」+ URL；`:11–12` 逐字 FA1 arXiv:2205.14135 与 Rabe-Staats arXiv:2112.05682 链接）；**`_attn_fwd_inner`**（`:27–77`）——内层循环遍历 K/V 块：`qk = tl.dot(q, k)`（`:50`）、running max `m_ij = tl.maximum(m_i, tl.max(qk,1)*qk_scale)`（`:57`）、减 running max `qk = qk*qk_scale - m_ij[:,None]`（`:58`）、`p = tl.math.exp2(qk)`（`:59`）、块内和 `l_ij = tl.sum(p,1)`（`:60`）、**rescale 因子** `alpha = tl.math.exp2(m_i - m_ij)`（`:62`）、running sum 更新 `l_i = l_i*alpha + l_ij`（`:63`）、**累加器 rescale** `acc = acc*alpha[:,None]`（`:65`）、增量矩阵乘 `acc = tl.dot(p, v, acc)`（`:72`）、`m_i = m_ij`（`:74`）；causal 分块 mask（`:51–55`）；**`_attn_fwd`** 初始化 `m_i=-inf` / `l_i=1.0` / `acc=0`（`:158–160`）、`qk_scale *= 1.44269504`（`:163`）、`q` 常驻 SRAM（`:165`）、causal 两趟 STAGE 拆分（`:169–183`）、**epilogue** `m_i += tl.math.log2(l_i)` + `acc = acc / l_i[:,None]`（`:185–186`）；`forward` wrapper 的 grid（`:458`）与 `stage = 3 if causal else 1`（`:451`）；**朴素参考实现**（test，物化 $`N\times N`$：`:536–541`） | 所有核心论断——「在线 softmax 一遍过 running max/sum」「分块遍历 + rescale 恒等于全矩阵 softmax」「不物化 $`N\times N`$ 只留 $`O(\mathrm{block})`$ running 状态」「exp2 + 1/ln2 预乘」「epilogue 的 log-sum-exp 归一化」——**逐字引 `06-fused-attention.py:行号`**；这是本章基石 |
| **C** | 学术出处（论文，标出处不引内文） | **Dao, T., et al. (2022).** *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* **arXiv:2205.14135**（tiling + rescaling + 反向重算奠基）。**Dao, T. (2023/ICLR 2024).** *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* **arXiv:2307.08691**（并行度改进：把并行从 batch·head 扩到序列块，`tutorials/06` 实现的正是 v2）。**Milakov, M., Gimelshein, N. (2018).** *Online normalizer calculation for softmax.* **arXiv:1805.02867**（running max/sum 一遍过的在线 softmax 递推）。**Rabe, M. N., Staats, C. (2021).** *Self-attention Does Not Need $`O(n^2)`$ Memory.* **arXiv:2112.05682**（lazy-softmax，$`O(n)`$/$`O(\log n)`$ 内存前身）。 | 为「在线 softmax / tiling / rescale / $`O(N)`$ 内存」提供**学术出处**。`:5` 与 `:11–12` 的 arXiv 链接由 docstring 逐字坐实（source-cited）。**论文内文的定理编号、复杂度证明、FA-2 具体加速比数字未联网逐字核实——一律标「待核·回指 arXiv」**，正文的数学以 `tutorials/06` 代码 + 教科书级共识直觉为准 |

> 红线复述：本包只登记**已核实**内容。A 档所有引文可在 `06-fused-attention.py:行号` 处逐字核对；在线 softmax 递推与 rescale 恒等性从代码逐行推出（可核）；C 档论文的存在与 URL 由 docstring 坐实（source-cited），其**内文定理 / 公式编号 / 加速比数字**标「待核·回指 arXiv」，绝不编造。

---

## 1. 动机：朴素 attention 要物化 N×N 打分矩阵，长序列直接爆显存

先看**朴素 attention 长什么样**。`tutorials/06` 自带的参考实现（在 `test_op` 里，就是一行数学翻译）把它写得清清楚楚（`06-fused-attention.py:536–541`，A 档逐字）：

```python
# reference implementation
M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
p = torch.matmul(q, k.transpose(2, 3)) * sm_scale     # ← 物化整张 N×N 打分矩阵 p
if causal:
    p[:, :, M == 0] = float("-inf")
p = torch.softmax(p.float(), dim=-1).half()           # ← 在 N×N 上做 softmax
ref_out = torch.matmul(p, v)                           # ← 再乘 V
```

数学定义就是：

```math
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V
```

注意中间那个 `p`：它的形状是 $`N\times N`$（`N_CTX × N_CTX`），$`N`$ 是序列长度。

**问题就出在这个 $`p`$。** 朴素实现必须把整张 $`N\times N`$ 的打分矩阵**完整算出来、写进显存（HBM），再读回来做 softmax，再读回来乘 V**。代价有两层：

- **显存**：$`p`$ 占 $`O(N^2)`$。序列长 8K 时，单个 head 的打分矩阵就是 $`8192^2 \approx 6.7\times10^7`$ 个元素；序列长 32K、64K 时平方增长直接把显存撑爆。这是长上下文模型的头号拦路虎。
- **访存带宽**：$`p`$ 在 HBM 上被**写一遍、读三遍**（算 QKᵀ 写、softmax 读写、乘 V 读）。attention 本质是**访存受限（memory-bound）** 的——瓶颈不是算力，是把这张大矩阵在 HBM 和计算单元之间来回搬（第 2 章的访存判据尺）。

于是一个自然的追问：**softmax 的每一行只依赖那一行的打分**，我们真的需要把整张 $`N\times N`$ 一次性都摆在显存里吗？能不能**一行 Q 对所有 K 的打分，边算边消化，算完一块就扔，永远不物化整行、更不物化整张矩阵**？

这正是 FlashAttention 的出发点，也是它的两个孪生子问题：

1. softmax 天然要「先看完整行、找到最大值、再逐个 exp 归一化」——**怎么可能边扫边算、不等整行看完**？→ §2 **在线 softmax**。
2. 就算能边扫边算，$`QK^\top`$ 本身还是个大矩阵——**怎么把它切块、让每块只在片上（SRAM / 寄存器）过一遍**？→ §3 **分块遍历 + rescale**。

---

## 2. 核心：在线 softmax——一遍过，边扫边更新 running max / running sum

### 2.1 为什么 softmax 天生「要看完整行」

数值稳定的 softmax，标准写法是**先减去该行最大值**再取指数（否则 $`e^{x}`$ 溢出）：

```math
\mathrm{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_{k} e^{x_k - m}}, \qquad m = \max_k x_k
```

难点在 $`m`$：它是**整行的最大值**，得先扫完一整行才知道。所以教科书 softmax 是**三遍扫描**——第一遍找 max，第二遍算分母 $`\sum e^{x_k-m}`$，第三遍算每个 $`e^{x_i-m}`$ 除以分母。要「一遍过」，就得在**还没看完整行**的时候就动手，等后面出现更大的值再回头修正。

### 2.2 在线 softmax 递推：running max + running sum 一趟扫完

Milakov & Gimelshein（arXiv:1805.02867）的关键观察：**running max 和 running sum 可以在一遍扫描里联合维护**。扫到第 $`j`$ 个元素 $`x_j`$ 时——

```math
m_j = \max(m_{j-1},\, x_j)
```
```math
d_j = d_{j-1}\, e^{\,m_{j-1}-m_j} + e^{\,x_j - m_j}
```

$`m_j`$ 是「到目前为止的最大值」，$`d_j`$ 是「到目前为止、以当前 $`m_j`$ 为基准的分母」。**精髓在 $`d_j`$ 里那个 $`e^{m_{j-1}-m_j}`$ 因子**：当新元素刷新了最大值（$`m_j > m_{j-1}`$），之前累积的 $`d_{j-1}`$ 是以旧基准 $`m_{j-1}`$ 算的、整体偏大了，就乘上 $`e^{m_{j-1}-m_j}\le 1`$ **把旧账「降标度」到新基准**——这一步就是后面 FlashAttention 里的 **rescale**。扫完最后一个元素，$`d_N`$ 就是正确的 softmax 分母，一遍搞定。

> 数学出处：上述递推的**思想与形式**属教科书级共识，可从下一节 `tutorials/06` 代码逐行逐字推出（可核）；作为学术首出，回指 Milakov & Gimelshein **arXiv:1805.02867**。论文内**定理编号与误差分析细节未联网核实，标「待核·回指 arXiv:1805.02867」**——本章不引其内文，只用能被代码坐实的递推式。

### 2.3 attention 版：softmax 的分子也一起 online 更新

attention 不只要 softmax 的分母，还要 $`\mathrm{softmax}(S)\cdot V`$。所以除了 running max $`m`$、running sum $`\ell`$，还要维护一个**输出累加器** $`O`$（就是代码里的 `acc`）。把上面的递推从「标量元素」升级到「一块 K/V」：设已经处理到第 $`j-1`$ 块、现在来了第 $`j`$ 块，块打分 $`S^{(j)} = q\,K_j^\top\cdot\mathrm{scale}`$，则三件套联合更新——

```math
m^{(j)} = \max\!\big(m^{(j-1)},\ \mathrm{rowmax}(S^{(j)})\big), \qquad \alpha = e^{\,m^{(j-1)} - m^{(j)}}
```
```math
\tilde P^{(j)} = e^{\,S^{(j)} - m^{(j)}}, \qquad \ell^{(j)} = \alpha\,\ell^{(j-1)} + \mathrm{rowsum}(\tilde P^{(j)})
```
```math
O^{(j)} = \alpha\,O^{(j-1)} + \tilde P^{(j)} V_j
```

**三样东西（$`\ell`$ 分母、$`O`$ 分子累加器）在 max 抬高时都乘同一个 $`\alpha`$ rescale**——这保证了无论分几块、每块多大，最后的结果和「一次性全矩阵 softmax」**逐位相等**（这是 FlashAttention「exact attention」的含义：不是近似，是恒等重排）。扫完所有块，最后一步归一化：

```math
O = O^{(\mathrm{last})} \big/ \ell^{(\mathrm{last})}
```

下一节直接把这几行数学和 `tutorials/06` 的真实 Triton 代码**逐行对上**。

---

## 3. 展开：分块遍历 + rescale——tutorials/06 逐段对上

### 3.1 骨架：外层锁一块 Q，内层遍历 K/V 块

`_attn_fwd` 是前向核。每个 program（第 2 章的 program 概念）负责**一块 Q**（`BLOCK_M` 行），先把三件套初始化好，再进内层循环遍历所有 K/V 块。初始化（`06-fused-attention.py:158–165`，A 档逐字）：

```python
# initialize pointer to m and l
m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")   # running max，初始 -∞
l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0            # running sum，初始 1.0
acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)        # 输出累加器 O，初始 0
# load scales
qk_scale = sm_scale
qk_scale *= 1.44269504  # 1/log(2)
# load q: it will stay in SRAM throughout
q = tl.load(Q_block_ptr)
```

三个对应关系一目了然：`m_i` = running max $`m`$、`l_i` = running sum $`\ell`$、`acc` = 输出累加器 $`O`$。两个工程细节要点透——

- **`qk_scale *= 1.44269504`（= $`1/\ln 2`$）**：代码全程用 `exp2`（2 为底的指数，是 GPU 的原生快指令）而非自然指数 `exp`。靠恒等式 $`e^{x}=2^{x/\ln 2}`$，把打分预乘 $`1/\ln 2`$，之后 `exp2` 出来的就等于 `exp`。这是纯性能优化，不改数学。
- **`q` 常驻 SRAM**：注释 `it will stay in SRAM throughout` 点破 FlashAttention「都在片上算」的关键——这一块 Q 一次 load 进来，整个内层循环反复用，不回 HBM（回指第 34 章共享内存）。

### 3.2 内层循环：running max / rescale / 增量累加，逐行对上数学

内层循环体 `_attn_fwd_inner`——每次迭代吃**一块 K/V**，把 §2.3 的三个更新式落地。核心段（`06-fused-attention.py:46–74`，A 档逐字，精简掉 causal 分支与 fp8 分支）：

```python
# loop over k, v and update accumulator
for start_n in range(lo, hi, BLOCK_N):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    # -- compute qk ----
    k = tl.load(K_block_ptr)
    qk = tl.dot(q, k)                                       # S^(j) = q · Kjᵀ（未 scale）
    m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)       # m^(j) = max(m^(j-1), rowmax(S))
    qk = qk * qk_scale - m_ij[:, None]                     # S - m^(j)（减 running max，稳定）
    p = tl.math.exp2(qk)                                   # P̃ = exp2(S - m^(j))
    l_ij = tl.sum(p, 1)                                    # rowsum(P̃)：本块贡献的分母
    # -- update m_i and l_i
    alpha = tl.math.exp2(m_i - m_ij)                       # α = exp2(m^(j-1) - m^(j)) ← RESCALE 因子
    l_i = l_i * alpha + l_ij                               # ℓ^(j) = α·ℓ^(j-1) + rowsum(P̃)
    # -- update output accumulator --
    acc = acc * alpha[:, None]                             # O 先整体 rescale：α·O^(j-1)
    # update acc
    v = tl.load(V_block_ptr)
    p = p.to(tl.float16)
    acc = tl.dot(p, v, acc)                                # O^(j) = α·O^(j-1) + P̃·Vj（融进 dot 的累加）
    # update m_i and l_i
    m_i = m_ij                                             # 滚动 running max
    V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))    # 下一块 K/V
    K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
```

**逐行对上 §2.3 的数学**（这是本章最该盯住的对照表）：

| 代码 | 行 | 数学 | 含义 |
|---|---|---|---|
| `qk = tl.dot(q, k)` | :50 | $`S^{(j)}=q\,K_j^\top`$ | 本块打分，只 $`\mathrm{BLOCK\_M}\times\mathrm{BLOCK\_N}`$，**从不物化整行** |
| `m_ij = tl.maximum(m_i, tl.max(qk,1)*qk_scale)` | :57 | $`m^{(j)}=\max(m^{(j-1)},\mathrm{rowmax}(S^{(j)}))`$ | running max 抬升 |
| `qk = qk*qk_scale - m_ij[:,None]` | :58 | $`S^{(j)}-m^{(j)}`$ | 减 running max，数值稳定 |
| `p = tl.math.exp2(qk)` | :59 | $`\tilde P^{(j)}=e^{S^{(j)}-m^{(j)}}`$ | 未归一化的注意力权重 |
| `l_ij = tl.sum(p, 1)` | :60 | $`\mathrm{rowsum}(\tilde P^{(j)})`$ | 本块分母贡献 |
| `alpha = tl.math.exp2(m_i - m_ij)` | :62 | $`\alpha=e^{m^{(j-1)}-m^{(j)}}`$ | **rescale 因子** |
| `l_i = l_i*alpha + l_ij` | :63 | $`\ell^{(j)}=\alpha\ell^{(j-1)}+\mathrm{rowsum}(\tilde P^{(j)})`$ | 分母 rescale 后累加 |
| `acc = acc*alpha[:,None]` | :65 | $`\alpha\,O^{(j-1)}`$ | **累加器 rescale** |
| `acc = tl.dot(p, v, acc)` | :72 | $`O^{(j)}=\alpha O^{(j-1)}+\tilde P^{(j)}V_j`$ | 增量矩阵乘累加（`tl.dot` 第三参 = 加到 acc 上） |
| `m_i = m_ij` | :74 | 滚动 | 为下一块准备 |

**关键就是 `alpha`（`:62`、`:65`）**：新块可能带来更大的打分 → running max 从 `m_i` 抬到 `m_ij` → 之前累积的 `acc` 和 `l_i` 都是以旧基准算的、偏大了 → 全部乘上 $`\alpha=e^{m_i-m_{ij}}\le 1`$ **降标度到新基准**，再把新块的贡献加进去。这一步让「分块增量」和「一次性全矩阵 softmax」**严格恒等**——这就是 §2.2 那个 $`e^{m_{j-1}-m_j}`$ 因子在 attention 里的化身。（注：`l_i` 初值取 `1.0` 而非 `0`，第一块时 `m_i=-inf` 使 `alpha = exp2(-inf) = 0`，`l_i*alpha` 把初值清零，故不影响正确性。）

### 3.3 收尾：延迟归一化 + log-sum-exp

内层循环只累加**未归一化**的 `acc` 和分母 `l_i`——**归一化推迟到所有块都扫完**，只做一次除法（`06-fused-attention.py:185–189`，A 档逐字）：

```python
# epilogue
m_i += tl.math.log2(l_i)                    # log-sum-exp（base-2），存给反向用
acc = acc / l_i[:, None]                    # O = O / ℓ ：一次性归一化
m_ptrs = M + off_hz * N_CTX + offs_m
tl.store(m_ptrs, m_i)
tl.store(O_block_ptr, acc.to(Out.type.element_ty))
```

- `acc = acc / l_i[:,None]` 就是 §2.3 的 $`O=O^{(\mathrm{last})}/\ell^{(\mathrm{last})}`$——**整个过程只在最后除一次**，省掉了朴素实现每行都归一化的开销。
- `m_i += tl.math.log2(l_i)` 把 running max 和 log(分母) 合成一个 **log-sum-exp** 值存进 `M`——反向传播重算 softmax 时只需这一个标量（对应 FlashAttention 论文的「反向重算不存 $`N\times N`$」思想，回指 arXiv:2205.14135，具体反向推导待核）。

### 3.4 causal mask：分块处理，对角块单独走一趟

因果 attention（query 只能看它之前的 key）在分块框架下有个漂亮的优化：**整块在对角线以下的 K/V 块，query 全都能看到，不用逐元素 mask**；只有**对角线所在的那一块**需要逐元素 mask。`tutorials/06` 用 `STAGE` 把两种块**拆成两趟循环**（`06-fused-attention.py:169–183`，A 档逐字，精简）：

```python
# stage 1: off-band —— 对角线严格以下的整块，无需 mask，直接全算
if STAGE & 1:
    acc, l_i, m_i = _attn_fwd_inner(..., 4 - STAGE, ...)
# stage 2: on-band —— 对角线那一块，需要逐元素 mask
if STAGE & 2:
    acc, l_i, m_i = _attn_fwd_inner(..., 2, ...)
```

对角块的逐元素 mask 在 `_attn_fwd_inner` 的 `STAGE == 2` 分支（`:51–55`，A 档逐字）：把被 mask 掉的位置的打分设成 $`-10^6`$（`exp2` 后趋近 0），其余不变：

```python
if STAGE == 2:
    mask = offs_m[:, None] >= (start_n + offs_n[None, :])   # query 位置 >= key 位置才可见
    qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)          # 不可见处打成 -1e6
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
```

外层 wrapper 用 `stage = 3 if causal else 1` 决定走几趟（`:451`）——causal 时 `STAGE=3`（两趟都走：先 off-band 整块、再 on-band 对角块），非 causal 时 `STAGE=1`（一趟扫完整个 $`N`$）。**整块跳过 mask 是分块框架白捡的便宜**：因果结构天然只需算下三角，分块让「跳过整个上三角块」变得零成本。

---

## 4. 落地：为什么省显存、为什么快

把前三节接起来，FlashAttention 的两个收益都能落到代码上说清：

**为什么省显存——只存 $`O(\mathrm{block})`$ 的 running 状态，永不物化 $`N\times N`$。** 整个前向核里，一块 Q 对应的中间状态只有：`m_i`（`[BLOCK_M]`）、`l_i`（`[BLOCK_M]`）、`acc`（`[BLOCK_M, HEAD_DIM]`）——全是 $`O(\mathrm{block})`$、和序列长度 $`N`$ **无关**的常数大小，全程待在寄存器 / 共享内存里。对比 §1 朴素实现的 $`p`$（$`N\times N`$，`:536`），FlashAttention **从头到尾没有任何一个 $`N\times N`$ 的张量落地**。这就是长序列不爆显存的根因（对应 Rabe & Staats 的 $`O(n)`$ 内存思想，回指 arXiv:2112.05682，其 $`O(\log n)`$ 证明细节待核）。

**为什么快——不落回 HBM，都在 SRAM / 寄存器。** attention 是访存受限的（§1）。FlashAttention 把 QKᵀ、softmax、乘 V **融成一个 kernel（fused）**：`q` 一次 load 进 SRAM 常驻（`:165`），内层循环里 K/V 分块流过、打分和累加全在片上寄存器完成，**中间的打分矩阵从不写回 HBM**。省掉的正是 §1 里「$`N\times N`$ 写一遍读三遍」那笔巨额 HBM 流量——把 IO 从 $`O(N^2)`$ 降到 $`O(N)`$ 级别。再叠加第 29 章的软件流水线（`num_stages` 用异步 load 把 K/V 分块搬运藏到 dot 计算背后）、第 27 章的 Tensor Core 布局（`tl.dot` 直接落到 MMA），这个 kernel 就是「tile 抽象 + 布局 + 流水线」三件套的集大成——这也是它作为 Triton 旗舰示例的意义。

**FA-2 的改进（一句带过）**：`tutorials/06` 实现的是 **FlashAttention-2**（docstring `:5` 逐字坐实）。相对 FA-1，v2 的主要改进是**工作划分（work partitioning）**：把并行度从「batch × head」进一步扩展到**序列块**维度（每个 Q 块一个 program，见 grid `:458`），并减少非矩阵乘的冗余 rescale。**具体的并行策略细节与相对 FA-1 的加速比数字待核·回指 arXiv:2307.08691**（本包无网络，不引其内文数字）。

**本章到此为止**：我们从「朴素 attention 为何爆显存」出发，用在线 softmax 解决「softmax 要看完整行」的矛盾，用分块遍历 + rescale 把它落成 `tutorials/06` 里逐行可核的 `m_i`/`l_i`/`acc`/`alpha` 记账，最后说清省显存（不物化 $`N\times N`$）与快（不落 HBM）的机理。至于**把这个 kernel 从 `tl.*` 一路编译到 PTX、看每一层 IR 怎么变**，留给第 43 章收官实战逐层拆——本章把 FlashAttention 的原理地基交给你。

---

## 附 A：A 档源码锚点清单（逐条可核，pin v3.2.0，`python/tutorials/06-fused-attention.py`）

| # | 行号 | 内容 | 用在 |
|---|---|---|---|
| A1 | `:5` | docstring 逐字：「This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao」+ flash2 URL | 红线 / §4（实现的是 FA-2，source-cited） |
| A2 | `:11–12` | docstring 逐字：FA1 arXiv:2205.14135 + Rabe-Staats arXiv:2112.05682 链接 | §0 / §4（论文出处 source-cited） |
| A3 | `:27–33` | `_attn_fwd_inner` 签名：`acc, l_i, m_i, q, K_block_ptr, V_block_ptr, ...` 三件套入参 | §3.2 |
| A4 | `:35–44` | `STAGE` 决定 K/V 遍历区间 `lo, hi`（causal off-band / on-band / 全程）+ advance 指针 | §3.4 |
| A5 | `:46–50` | 内层循环 `for start_n in range(lo, hi, BLOCK_N)`：load K、`qk = tl.dot(q, k)` | §3.1 / §3.2（分块遍历骨架） |
| A6 | `:51–55` | `STAGE==2` causal 对角块逐元素 mask（`>=`、`-1.0e6`）+ running max | §3.4 |
| A7 | `:57–60` | `m_ij = maximum(m_i, max(qk,1)*qk_scale)`、`qk = qk*qk_scale - m_ij`、`p = exp2(qk)`、`l_ij = sum(p,1)` | §3.2（running max + 减 max + exp2 + 块内和） |
| A8 | `:62–63` | **`alpha = exp2(m_i - m_ij)`**（rescale 因子）、`l_i = l_i*alpha + l_ij` | §2.3 / §3.2（rescale 核心） |
| A9 | `:65–74` | `acc = acc*alpha[:,None]`（累加器 rescale）、`acc = tl.dot(p, v, acc)`（增量累加）、`m_i = m_ij` | §3.2（增量更新 + 恒等性） |
| A10 | `:158–160` | 初始化 `m_i=-inf`、`l_i=1.0`、`acc=0` | §3.1 |
| A11 | `:162–165` | `qk_scale *= 1.44269504`（1/log2）、`q = tl.load(...)` 常驻 SRAM | §3.1（exp2 优化 + q 驻 SRAM） |
| A12 | `:169–183` | causal 两趟 STAGE 拆分：off-band 整块（STAGE&1）+ on-band 对角块（STAGE&2） | §3.4 |
| A13 | `:185–189` | epilogue：`m_i += log2(l_i)`（log-sum-exp）、`acc = acc / l_i`（延迟归一化）、store | §3.3 |
| A14 | `:451`、`:458` | wrapper：`stage = 3 if causal else 1`、grid = `(cdiv(N_CTX,BLOCK_M), Z*H, 1)`（每 Q 块一 program） | §3.4 / §4（FA-2 序列块并行） |
| A15 | `:536–541` | test 里的**朴素参考实现**：物化 `p = matmul(q,kᵀ)*scale`（$`N\times N`$）+ `softmax` + `matmul(p,v)` | §1（对照：要物化的 $`N\times N`$） |

---

## 附 B：C 档论文核实记录

| 概念 | 源码坐实？ | 学术出处 | 核实状态 |
|---|---|---|---|
| **在线 softmax 递推**（running max/sum 一遍过、$`e^{m_{j-1}-m_j}`$ 降标度） | ✅ 递推可从 `:57–63` 逐行推出（`m_ij`/`l_i`/`alpha`） | Milakov & Gimelshein **arXiv:1805.02867** | 递推形式由代码坐实（可核）；论文首出定位 web-verified（meta）。**论文内定理编号 / 误差分析待核·回指 arXiv:1805.02867** |
| **tiling + rescale 恒等性**（分块增量 = 全矩阵 softmax） | ✅ `:62–72` 三件套同乘 `alpha` 逐字坐实 | FlashAttention **arXiv:2205.14135** | rescale 机制由代码坐实（可核）；URL 由 docstring `:11` 逐字坐实（source-cited）。**论文内 IO 复杂度定理 / 反向重算证明待核·回指 arXiv:2205.14135** |
| **反向重算 / log-sum-exp 存储**（`M` 存 `m_i+log2(l_i)`） | ✅ `:185` 逐字 | FlashAttention arXiv:2205.14135 | `M` 存 log-sum-exp 由代码坐实；反向核 `_attn_bwd*`（`:192–437`）在源码内可核，但**反向数学推导超出本 primer 范围、且论文证明待核·回指 arXiv:2205.14135** |
| **FA-2 工作划分**（并行扩到序列块） | ✅ docstring `:5` 逐字「Flash Attention v2」+ grid `:458` 序列块并行 | FlashAttention-2 **arXiv:2307.08691** | 「实现的是 v2」+「每 Q 块一 program」由代码坐实（source-cited + 可核）。**v2 相对 v1 的具体并行策略细节 / 加速比数字待核·回指 arXiv:2307.08691** |
| **$`O(n)`$ / $`O(\log n)`$ 内存** | ✅ running 状态 `:158–160` 全为 $`O(\mathrm{block})`$、无 $`N\times N`$ 落地 | Rabe & Staats **arXiv:2112.05682** | 「不物化 $`N\times N`$」由代码坐实（可核）；URL 由 docstring `:12` 逐字坐实（source-cited）。**$`O(\log n)`$ 内存证明细节待核·回指 arXiv:2112.05682** |

> 核实边界声明：本包组装环境**无网络 / WebFetch**。四篇论文的 arXiv id 与定位：其中 FA-2（`:5`）、FA-1 与 Rabe-Staats（`:11–12`）由 `06-fused-attention.py` docstring **逐字写出**（source-cited）；online-softmax（Milakov & Gimelshein）为 meta web-verified 的 uncited 学术前身。**四篇论文的具体定理、复杂度证明、公式编号、FA-2 加速比数字均未联网逐字核实，凡涉及一律标「待核·回指 arXiv:xxxx」**。本章正文的**每一条机制论断都能落到 `06-fused-attention.py:行号`**——在线 softmax 递推与 rescale 恒等性从代码逐行推出（可核），论文只作学术出处，**绝不编造论文内文**。

---

## 附 C：key_figures（交 illustrator 重绘）

1. **在线 softmax 一遍过：running max / running sum 递推图**（§2.2 target）——画一条数据流从左到右扫过若干元素 $`x_1, x_2, \dots`$；每步显示 `m_j = max(m_{j-1}, x_j)` 抬升 running max、`d_j = d_{j-1}·e^{m_{j-1}-m_j} + e^{x_j-m_j}` 累积分母；**高亮某一步 $`x_j`$ 刷新了最大值 → 旧的 `d_{j-1}` 被 $`e^{m_{j-1}-m_j}<1`$ 因子「降标度」** 的那一刻（rescale 的雏形）。对比朴素三遍扫描（先找 max、再算分母、再归一化）。**这是全章讲清「softmax 为何能一遍过」的核心图**。锚点：`06-fused-attention.py:57–63` + Milakov & Gimelshein arXiv:1805.02867。
2. **FlashAttention 分块遍历 + rescale 时序图**（§3.2 target）——外层锁定一块 Q（`BLOCK_M` 行常驻 SRAM），内层水平铺开 K/V 块 $`K_1V_1, K_2V_2, \dots`$ 逐块流过；每块画出四步：`qk=q·Kjᵀ` → `m_ij` 抬 running max → `alpha=exp2(m_i-m_ij)` 把旧 `acc`/`l_i` **整体乘 α 降标度**（用一个「缩放」箭头醒目标出）→ `acc = α·acc + P̃·Vj` 增量累加；右侧标注三件套 `m_i`/`l_i`/`acc` 都是 $`O(\mathrm{block})`$、**全程无 $`N\times N`$ 落地**；末尾 epilogue `acc/l_i` 一次归一化。**这是全章把「分块 + rescale 恒等于全矩阵 softmax」画实的落点图**。锚点：`06-fused-attention.py:46–74`（内层循环）+ `:185–186`（归一化）+ FlashAttention arXiv:2205.14135。
