# Accelerating Speculative Decoding with Block Diffusion Draft Trees (DDTree)

- **arXiv**: 2604.12989v1（提交 2026-04-14）
- **作者**: Liran Ringel, Yaniv Romano（Technion）
- **来源**: https://arxiv.org/html/2604.12989v1（HTML 全文）；项目页 https://liranringel.github.io/ddtree；代码 https://github.com/liranringel/ddtree
- **抓取日期**: 2026-07-11

> **本文件是 DFlash（arXiv:2602.06036，见 `paper.md`）的后续工作摘录，不含 KV 注入的原始推导**——DDTree 直接复用 DFlash 的块扩散起草器与 KV 注入机制作为"黑盒草稿器"，本文只解决"一次块扩散前向产出的每位置边际分布，如何比'只验证一条轨迹'更充分地利用"这一个问题。收录范围：Abstract、§1 引言节选、§3 背景（块扩散起草的两种分布对比）、§4 方法（Overview / Surrogate objective / Algorithm 1 / 验证与缓存更新）、§5 实验节选。附录 A 的 Proposition 1-3 完整证明未收录，只保留结论陈述；References 全删。

---

## Abstract

# PAPER Abstract

> Speculative decoding accelerates autoregressive language models by using a lightweight drafter to propose multiple future tokens, which the target model then verifies in parallel. DFlash shows that a block diffusion drafter can generate an entire draft block in a single forward pass and achieve state-of-the-art speculative decoding performance, outperforming strong autoregressive drafters such as EAGLE-3. Vanilla DFlash, however, still verifies only a single drafted trajectory per round, potentially limiting its acceptance length. We introduce DDTree (Diffusion Draft Tree), a method that constructs a draft tree directly from the per-position distributions of a block diffusion drafter. Under a fixed node budget, DDTree uses a simple best-first heap algorithm to select the continuations that are most likely to match the target model according to a surrogate defined by the draft model's output. The resulting tree is verified efficiently in a single target model forward pass using an ancestor-only attention mask. Because DDTree builds on DFlash, a leading draft model for speculative decoding, these gains place DDTree among the leading approaches to speculative decoding.

中文旁注：这段摘要本身就交代了 DDTree 与 DFlash 的关系——"vanilla DFlash 每轮只验证一条轨迹"是被改进的起点，DDTree 加的是"从同一次块扩散前向的输出里，多挑几条轨迹一起验证"这一层，起草器本身（块扩散 + KV 注入）原封不动地继承自 DFlash。

---

## 1 Introduction（节选）

# PAPER §1

> Block diffusion is especially attractive in this setting as it can generate an entire draft block in a single forward pass. DFlash shows the promise of this approach: it uses a small block diffusion drafter that leverages features derived from the larger target model, achieving state-of-the-art speculative decoding performance. [...] These results establish block diffusion as a powerful foundation for speculative decoding. At the same time, these promising results spark a key challenge:
>
> **How should we make the best use of the information that the block diffusion draft model produces?**
>
> Currently, DFlash verifies only one drafted trajectory per round, even though a single block diffusion forward pass produces a distribution over tokens at each future position. As such, DFlash does not utilize the many plausible continuations that block diffusion produces. Naturally, exploring more continuations could increase the probability that the target model continues along a drafted path. On the other hand, naively using multiple continuations increases the verifier cost and can erase the latency benefit. The challenge, therefore, is to use the draft model's per-position distributions to choose the continuations that are most worthwhile to verify.

> We address this challenge with DDTree (Diffusion Draft Tree). Our method (i) constructs a draft tree directly from the per-position distributions produced by a block diffusion drafter, (ii) selects a compact set of promising continuations under a specified tree-node budget, and (iii) verifies them in a single target model forward pass with tree attention.

中文旁注：这是本章"接受率-延迟权衡"要点的关键出处——DDTree 的动机不是"块扩散起草得不够好"，而是"块扩散一次前向其实产出了比单条轨迹丰富得多的信息（每个位置的完整边际分布），DFlash 只用了其中一条路径就扔掉了剩下的"。

### 相关工作定位（§2 节选）

# PAPER §2

> OPT-Tree by Wang et al. constructs adaptive trees for autoregressive drafters by maximizing an approximate expected acceptance length under a node budget. In that autoregressive setting, tree construction still requires one drafter's forward pass per tree depth (i.e., per token position), and thus it has higher computational overhead compared to our DDTree approach.
>
> Vanilla DFlash, however, explores only one continuation per round. A very recent work, DART, also constructs draft trees from one-pass parallel logits. Still, it relies on continuity-aware tree pruning with an external [scoring model]. By contrast, our proposed DDTree keeps the same one-pass DFlash drafter and constructs the tree directly from the per-position probabilities produced by that pass. This avoids auxiliary external scoring and gives an explicit surrogate objective that our best-first construction provably maximizes.

中文旁注：与 OPT-Tree 的区别是"树的信息来自单次块扩散前向，而不是多次自回归前向"；与 DART 的区别是"树的打分直接用块扩散自身的每位置概率，不引入外部打分模型"。

---

## 3 Background: Block Diffusion Drafting — 两种分布的区别

# PAPER §3

> Under the target model, the continuation distribution factorizes autoregressively as
>
> $$p(y_{1:L}\mid c,b)=\prod_{i=1}^{L}p(y_{i}\mid c,b,y_{1:i-1}). \qquad (1)$$
>
> A one-pass block diffusion drafter does not expose these continuation-conditioned factors. Instead, it provides only per-position marginals [...] The distribution associated with a one-pass block diffusion drafter is the factorized distribution
>
> $$Q(y_{1:L}\mid c,b):=\prod_{i=1}^{L}q_{i}(y_{i}\mid c,b). \qquad (2)$$
>
> Thus, the target model provides a path-conditioned autoregressive distribution, whereas the drafter provides only a factorized distribution over the next $L$ positions.

中文旁注（本章"块内并行去噪"到底并行的是什么，这里是最精确的数学交代）：Eq.(1) 里 target 的 $p(y_i\mid c,b,y_{1:i-1})$ 依赖已生成的 $y_{1:i-1}$，是路径条件分布；Eq.(2) 里 draft 的 $q_i(y_i\mid c,b)$ 只依赖上下文 $c$ 和 bonus token $b$，**不依赖块内其他位置的采样结果**——这正是"块扩散一次前向能把整块同时算出来"的代数原因：每个位置的分布互相独立，谁都不用等谁。也正因为如此，$Q$ 只是对 $p$ 的一个近似（$q_i$ 忽略了块内位置间的相关性），DDTree §4 要解决的就是"在明知 $Q\neq p$ 的前提下，怎么用 $Q$ 挑出最值得验证的候选集合"。

---

## 4 The Proposed Method: Diffusion Draft Tree

### 4.1 Overview

# PAPER §4.1 + Figure 2

**Figure 2**（key_figures 收录）caption 大意：DDTree 一轮解码示意，(a) 单次块扩散前向对位置 1/2/3 产出边际分布、以 bonus token 为根构建候选树；(b) 验证器沿树行走，两次匹配后在首个不匹配处产生下一个 bonus token。

> DDTree builds a draft tree under a node budget $B$ [that ideally would maximize expected acceptance length under the target model $p$, Eq.(1)] needed to optimize the expected target-model acceptance length directly. We therefore can only optimize a surrogate objective, which is the expected acceptance length under the drafter's factorized approximation (2).

四步流程（原文列举）：

1. Run the block diffusion drafter once to obtain per-position distributions for the next $L$ positions.
2. Build a draft tree with $B$ nodes from those distributions.
3. Compile the tree into input tensors for the target model and run one target-model forward pass with tree attention.
4. Walk the tree as illustrated in Figure 2: at each step, use the target model's decoding rule to choose the next token and check whether it matches a child in the tree; accept the matched drafted path, and carry the first unmatched target token (the next bonus token) to the next round.

### 4.2 Surrogate objective for draft-tree selection

# PAPER §4.2 Eq.(3)-(8)

定义匹配长度 $\alpha_T$：

$$\alpha_{T}(y_{1:L})=\max\{d:y_{1:d}\in T\}, \qquad (4)$$

理想目标（不可行，因为需要 target 的路径条件概率）：

$$\max_{T:\,|T|\leq B,\;T\text{ valid}}\mathbb{E}_{Y_{1:L}\sim p(\cdot\mid c,b)}[\alpha_{T}(Y_{1:L})], \qquad (5)$$

替代目标（用 draft 的分解分布 $Q$ 代替 $p$）：

$$\max_{T:\,|T|\leq B,\;T\text{ valid}}\mathbb{E}_{Y_{1:L}\sim Q(\cdot\mid c,b)}[\alpha_{T}(Y_{1:L})]. \qquad (6)$$

前缀概率（在 $Q$ 下，一个长度为 $|u|$ 的候选延续 $u$ 的概率）：

$$q(u\mid c,b)=\prod_{i=1}^{|u|}q_{i}(u_{i}\mid c,b). \qquad (7)$$

# PAPER Proposition 1

> For any valid draft tree $T$:
>
> $$\mathbb{E}_{Y_{1:L}\sim Q(\cdot\mid c,b)}[\alpha_{T}(Y_{1:L})]=\sum_{u\in T}q(u\mid c,b). \qquad (8)$$

中文旁注：Eq.(8) 把"期望接受长度"这个看似要枚举所有可能延续求期望的量，化简成"把树里每个节点（前缀）的概率直接加起来"——这一步是后面"贪心/堆算法能求出最优树"的代数基础：目标函数是可加的，不需要联合考虑树的组合结构。

# PAPER Proposition 2

> Let $u^{(1)},u^{(2)},\dots$ be all nonempty prefixes of length at most $L$, ordered so that $q(u^{(1)}\mid c,b)\geq q(u^{(2)}\mid c,b)\geq\dots$, with ties broken arbitrarily. Define $T_{B}=\{u^{(1)},\dots,u^{(B)}\}$. [Then $T_B$ maximizes the surrogate objective in (6) under node budget $B$.]

> The above result is analogous to OPT-Tree's expected acceptance length objective, with the crucial difference that in our case all required probabilities are obtained from a single block diffusion drafter forward pass rather than from multiple autoregressive passes.

中文旁注：Proposition 2 说"最优树 = 概率最高的 $B$ 个前缀"，逻辑很直白（Eq.(8) 是可加的，直接贪心取前 $B$ 大即可），但"前 $B$ 个前缀是否自动构成一棵合法的树（即每个节点的所有祖先前缀也必须在树里）"这件事本身也需要证明——原文在 Proposition 2 证明中指出：因为分解分布下延续越长概率只会单调不增（$q(u_{1:d+1})=q(u_{1:d})\cdot q_{d+1}(\cdot)\leq q(u_{1:d})$），所以任何前缀的概率不小于它的任何延伸，"取前 B 大"这个操作自动满足前缀闭合，不需要额外判断树合法性。

### 4.3 Efficient and optimal tree construction — best-first 堆算法

# PAPER §4.3

> Proposition 2 reveals that an optimal draft tree is obtained by taking the $B$ highest-probability prefixes. The remaining challenge is therefore algorithmic: how can we recover these prefixes efficiently, without enumerating the exponentially many possible prefixes up to depth $L$? [...] this can be done with a simple best-first search procedure.
>
> At each depth $i$, let $v_{i}^{(1)},v_{i}^{(2)},\dots$ be the tokens ordered so that $q_{i}(v_{i}^{(1)}\mid c,b)\geq q_{i}(v_{i}^{(2)}\mid c,b)\geq\dots$, and let $q_{i}^{(k)}=q_{i}(v_{i}^{(k)}\mid c,b)$. We index prefixes by token ranks rather than by vocabulary ids. [A rank tuple $\rho=(\rho_1,\ldots,\rho_d)$ denotes the depth-$d$ prefix $(v_1^{(\rho_1)},\ldots,v_d^{(\rho_d)})$, and its log-probability is $\sigma(\rho)=\sum_{i=1}^{d}\log q_i^{(\rho_i)}$.]

# PAPER Lemma 1

> There exists an optimal valid draft tree that maximizes the surrogate expected acceptance length in (6), such that every node in the tree lies in [the top-$K$ rank space $\mathcal{S}_K$, where $K=\min(B,|\mathcal{V}|)$].

# PAPER Algorithm 1

> **Algorithm 1** Best-first draft-tree construction from one block diffusion drafter pass
>
> 1: Top-$K$ tokens $\{v_{i}^{(k)}\}_{i=1,k=1}^{L,K}$ and their probabilities $\{q_{i}^{(k)}\}_{i=1,k=1}^{L,K}$; node budget $B$
> 2: Initialize max-heap $H\leftarrow\{((1),\sigma((1)))\}$
> 3: Initialize draft tree $T\leftarrow\varnothing$
> 4: **while** $|T|<B$ and $H\neq\varnothing$ **do**
> 5:   Pop the rank tuple $\rho$ with largest score $\sigma(\rho)$
> 6:   Add prefix $(v_1^{(\rho_1)},\ldots,v_d^{(\rho_d)})$ to $T$
> 7:   **if** $\rho_d+1\leq K$ **then**
> 8:     Push next sibling $(\rho_1,\ldots,\rho_{d-1},\rho_d+1)$ with score $\sigma(\rho)-\log q_d^{(\rho_d)}+\log q_d^{(\rho_d+1)}$
> 9:   **end if**
> 10:   **if** $d<L$ **then**
> 11:     Push first child $(\rho_1,\ldots,\rho_d,1)$ with score $\sigma(\rho)+\log q_{d+1}^{(1)}$
> 12:   **end if**
> 13: **end while**
> 14: **return** draft tree $T$

# PAPER Proposition 3 + Remark 2

> Algorithm 1 returns an optimal valid draft tree for the surrogate objective in (6) under node budget $B$.
>
> [Remark 2:] The heap stage costs $O(B\log B)$.

中文旁注：这个算法直觉是"每次从堆里弹出当前分数最高的候选前缀，加进树，然后往两个方向扩展候选：①同一深度换成排名下一位的兄弟节点（sibling），②往下延伸一层、取该层排名第一的孩子节点（first child）"——因为每换一个 rank 或每往下扩一层都只让 log 概率减少，堆顶弹出的顺序天然就是全局最优的非增序，不需要提前枚举所有 $K^L$ 种组合。这是本章④"接受率/加速比"的进阶延伸：DFlash 本体只用块扩散边际分布采样出一条轨迹验证，DDTree 在同一堆边际分布上跑一次 $O(B\log B)$ 的堆算法就能拿到可证明最优的候选树。

### 4.4 Efficient verification and cache update

# PAPER §4.4

> To verify the selected draft tree in one target-model forward pass, we flatten it into a sequence of token ids rooted at the bonus token $b$. We assign position ids by tree depth so that the verifier applies the correct positional encoding. We then use **tree attention**, under which each drafted node attends to the past context through the KV cache and, within the drafted tree, **only to the root, its ancestors, and itself**.
>
> Verification then follows the target model's own decoding rule, whether greedy or temperature-based sampling. [...] Starting from the bonus token $b$, we check whether the token selected by the target model at the current node matches one of that node's children in the draft tree. If it does, that child is accepted and the walk continues. [...] If no child matches, the walk stops. The accepted path is appended to the output sequence, the first unmatched target-model token becomes the bonus token for the next round, and the KV cache is compacted to retain only the accepted path.

中文旁注（"ancestor-only attention mask"术语落地）：这句"each drafted node attends [...] only to the root, its ancestors, and itself"就是本章 outline 与 meta.json 里提到的"ancestor-only attention mask"的准确定义——树上任意节点只看得到从根到自己这条链，看不到树上的兄弟分支，因此一次 target 前向能同时给树上所有候选节点打分而不互相污染。

---

## 5 Experiments（节选，均为论文自报，未独立复现）

# PAPER §5.1

> We evaluate three target models, Qwen3-4B, Qwen3-8B, and Qwen3-Coder-30B-A3B-Instruct, each paired with its corresponding DFlash checkpoint [...] Our benchmark suite spans reasoning tasks such as MATH-500, GSM8K, AIME 2024, and AIME 2025; code tasks such as HumanEval, MBPP, LiveCodeBench, and SWE-bench Lite; and general instruction and dialogue tasks such as MT-Bench and Alpaca. We run the benchmark on 8 H200 GPUs at temperatures 0.0 and 1.0 [...] All runs use block size 16, DDTree node budgets [16, 32, 64, 128, 256, 512, 1024].

Table 1 节选（Temperature = 0.0，格式 "DFlash Speedup / DFlash+DDTree Speedup"）：

| Dataset | Qwen3-4B | Qwen3-8B | Qwen3-Coder-30B-A3B |
|---|---|---|---|
| GSM8K | 4.77× → 6.51× | 4.78× → 6.57× | 4.00× → 5.18× |
| HumanEval | 4.81× → 6.62× | 4.84× → 6.61× | （原文表格该行 Coder 列在抓取时被截断，未收录，不杜撰） |
| AIME 2024 | 5.56× → 7.54× | 5.38× → 7.46× | 3.95× → 5.16× |
| AIME 2025 | 5.33× → 7.37× | 5.32× → 7.39× | 3.98× → 5.13× |
| Alpaca | 2.03× → 3.11× | 2.07× → 3.12× | 1.53× → 2.22× |

# PAPER §5（结果段落）

> Figure 1 and Table 1 summarize the main benchmark results. [...] All 60 entries (10 datasets × 3 models × 2 temperatures) show consistent DDTree improvements.

# PAPER Figure 3 caption + 正文

> Budget tradeoff on MATH-500 with Qwen3-8B at temperature 0.0. Acceptance length increases steadily with the DDTree node budget, while speedup peaks at an intermediate budget once verifier cost becomes dominant.
>
> As our DDTree node budget grows, acceptance length increases steadily, and the end-to-end speedup improves until it peaks around budgets of 256 to 512. Pushing the budget to 1024 increases acceptance length further, but the tradeoff is no longer favorable as the additional overhead of verifying more drafted tokens outweighs the gain from the longer accepted prefix. [...] This highlights the importance of a front-heavy tree that does not waste budget on low-probability trajectories.

# PAPER Figure 4 caption

> Acceptance length distribution on MATH-500 with Qwen3-8B at temperature 0.0. The DDTree histogram uses the best speedup budget [...]

中文旁注：Figure 3 是本章"接受率-延迟权衡"最直接的数值证据——树预算不是越大越好，256-512 是这组实验里的甜蜜点，再往上验证开销（target 单次前向要打分的节点数）反超了接受长度收益。

**数字来源核实结论**：以上 DDTree 数字均出自 arXiv:2604.12989 论文本体（非厂商博客），但同样**未见第三方独立复现**；"60 组实验全部提升 30-60%" 是论文自陈的统计口径，本文件只摘了 Table 1 中 GSM8K/HumanEval/AIME/Alpaca 几行代表性数字，MBPP/LiveCodeBench/SWE-bench/MT-Bench 及 Temperature=1.0 数据未逐行摘录（避免整表照搬），如需完整数字应查阅原文 Table 1。
