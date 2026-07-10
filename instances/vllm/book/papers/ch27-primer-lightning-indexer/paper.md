> **Source note**: 本文件为策展摘录（curated excerpt），非整篇论文原文顺抄。§一 谱系铺垫改写自 arXiv:2502.11089《Native Sparse Attention》(Yuan et al., DeepSeek-AI, 2025)摘要/§1；§二～§四 逐段摘录自 arXiv:2512.02556《DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models》(DeepSeek-AI, 2026)§2.1「DeepSeek Sparse Attention」及其子节「Prototype of DSA」「Instantiate DSA Under MLA」「2.1.1 Continued Pre-Training」「2.3 Inference Costs」，公式编号(1)-(4)与原文一致（对应源文件 `paper-dsa.md` 的 Eq.(1)-(4)）。IndexCache 布局细节与 FP4 量化变体在本报告中覆盖不足（本报告的索引器仅提到 FP8），已用 arXiv:2606.19348《DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence》(DeepSeek-AI, 2026)补齐，见同目录 `paper-v4.md`（§2.3.1 Eq.(9)-(19)、§2.3.4、§5.2.1）。全文引用号务必以两份原文为准，本文件不新增编号、不改写公式。
---

# 一、谱系铺垫：从 NSA 到 DSA（一段概述，详见 ch23）

稀疏注意力削减 $O(L^2)$ 注意力税的思路，最早由 arXiv:2502.11089《Native Sparse Attention (NSA)》系统化：压缩(compression)、选择(selection)、滑窗(sliding window)三条支路并行打分，训练期与推理期共用一套稀疏模式（"natively trainable"）。DeepSeek-V3.2 的 **DeepSeek Sparse Attention (DSA)** 是这条谱系在生产模型上的落地——舍弃了 NSA 的三支路结构，只保留"打分 + top-k 选择"一支，用一个专门的小模块（lightning indexer）代替 NSA 里的压缩/选择联合打分。NSA 的三支路设计、训练期稀疏与 Triton 核实现细节不是本章重点，完整谱系见 `instances/vllm-ascend/book/papers/ch23-primer-sparse-attention/paper.md`；本章只取 DSA 之后"索引器怎么打分、索引器自己的缓存怎么管、量化到多低精度"这条线，自包含往下讲。

# 二、DSA 的原型：lightning indexer + top-k 选择

> 摘自 arXiv:2512.02556 §2.1 "DeepSeek Sparse Attention" → "Prototype of DSA."

The prototype of DSA primarily consists of two components: a lightning indexer and a fine-grained token selection mechanism.

The lightning indexer computes the index score $I_{t,s}$ between the query token $\mathbf{h}_t \in \mathbb{R}^d$ and a preceding token $\mathbf{h}_s \in \mathbb{R}^d$, determining which tokens to be selected by the query token:

$$
I_{t,s} = \sum_{j=1}^{H^I} w_{t,j}^I \cdot \mathrm{ReLU}\left(\mathbf{q}_{t,j}^I \cdot \mathbf{k}_s^I\right)
\tag{1}
$$

where $H^I$ denotes the number of indexer heads; $\mathbf{q}_{t,j}^I \in \mathbb{R}^{d^I}$ and $w_{t,j}^I \in \mathbb{R}$ are derived from the query token $\mathbf{h}_t$; and $\mathbf{k}_s^I \in \mathbb{R}^{d^I}$ is derived from the preceding token $\mathbf{h}_s$. We choose ReLU as the activation function for throughput consideration. Given that the lightning indexer has a small number of heads and can be implemented in FP8, its computational efficiency is remarkable.

Given the index scores $\{I_{t,s}\}$ for each query token $\mathbf{h}_t$, our fine-grained token selection mechanism retrieves only the key-value entries $\{\mathbf{c}_s\}$ corresponding to the top-k index scores. Then, the attention output $\mathbf{u}_t$ is computed by applying the attention mechanism between the query token $\mathbf{h}_t$ and the sparsely selected key-value entries $\{\mathbf{c}_s\}$:

$$
\mathbf{u}_t = \mathrm{Attn}\left(\mathbf{h}_t, \left\{\mathbf{c}_s \mid I_{t,s} \in \mathrm{Top}\text{-}k(I_{t,:})\right\}\right)
\tag{2}
$$

**（图见 Figure 2, arXiv:2512.02556——见本包 `meta.json` key_figures 第一项，标出 DSA 在 MLA 之下如何用 indexer 挑出 top-k 的 latent KV 条目。）**

> 摘自同节 "Instantiate DSA Under MLA."

For the consideration of continued training from DeepSeek-V3.1-Terminus, we instantiate DSA based on MLA (deepseekV2) for DeepSeek-V3.2. At the kernel level, each key-value entry must be shared across multiple queries for computational efficiency (yuan-etal-2025-native). Therefore, we implement DSA based on the MQA (MQA) mode of MLA, where each latent vector (the key-value entry of MLA) will be shared across all query heads of the query token.

**为何用独立小头（本章重点之一，直接源于上述两段原文）**：indexer 的 $H^I$ 与主注意力头数无关，是一组独立、小规模、可 FP8 化的头——它只负责打分排序，不参与最终数值计算（数值计算仍由主 MQA/MLA 完成）。这个设计把"决定看哪里"和"看完之后怎么算"彻底解耦：indexer 便宜到可以对每个 query 都对全部历史 token 算一遍分（$O(L^2)$，见下文§四），而真正的注意力数值计算只用在被选中的 $k$ 个条目上。

# 三、索引器怎么训练：让打分对齐主注意力

> 摘自 arXiv:2512.02556 §2.1.1 "Continued Pre-Training" → "Dense Warm-up Stage."

We first use a short warm-up stage to initialize the lightning indexer. In this stage, we keep dense attention and freeze all model parameters except for the lightning indexer. To align the indexer outputs with the main attention distribution, for the $t$-th query token, we first aggregate the main attention scores by summing across all attention heads. This sum is then L1-normalized along the sequence dimension to produce a target distribution $p_{t,:} \in \mathbb{R}^t$. Based on $p_{t,:}$, we set a KL-divergence loss as the training objective of the indexer:

$$
\mathcal{L}^I = \sum_t \mathbb{D}_{KL}\left(p_{t,:} \parallel \mathrm{Softmax}(I_{t,:})\right)
\tag{3}
$$

> 摘自同节 "Sparse Training Stage."

Following indexer warm-up, we introduce the fine-grained token selection mechanism and optimize all model parameters to adapt the model to the sparse pattern of DSA. In this stage, we also keep aligning the indexer outputs to the main attention distribution, but considering only the selected token set $\mathcal{S}_t = \{s \mid I_{t,s} \in \mathrm{Top}\text{-}k(I_{t,:})\}$:

$$
\mathcal{L}^I = \sum_t \mathbb{D}_{KL}\left(p_{t,\mathcal{S}_t} \parallel \mathrm{Softmax}(I_{t,\mathcal{S}_t})\right)
\tag{4}
$$

It is worth noting that we detach the indexer input from the computational graph for separate optimization. The training signal of the indexer is from only $\mathcal{L}^I$, while the optimization of the main model is according to only the language modeling loss.

**推理期视角**：训练期的 KL 对齐（式(3)(4)）解释了 lightning indexer 的打分为何"可信"——它被单独一条损失监督去逼近主注意力真实分配的注意力质量，而不是随便一个 heuristic 打分器。这也是"为何独立小头"这一设计选择在训练侧的根据：indexer 有自己的参数、自己的梯度来源（`detach`），推理时才能作为一个独立、廉价、但语义对齐的路由器使用。

# 四、复杂度账：indexer 本身仍是 $O(L^2)$

> 摘自 arXiv:2512.02556 §2.3 "Inference Costs"

DSA reduces the core attention complexity of the main model from $O(L^2)$ to $O(Lk)$, where $k$ ($\ll L$) is the number of selected tokens. Although the lightning indexer still has a complexity of $O(L^2)$, it requires much less computation compared with MLA in DeepSeek-V3.1-Terminus. Combined with our optimized implementation, DSA achieves a significant end-to-end speedup in long-context scenarios.

**这是本章要讲透的"诚实账"**：DSA 并没有把 $O(L^2)$ 项消灭——它只是把 $O(L^2)$ 项从"贵"（全头 MLA 数值计算）换成"便宜"（少头 indexer 打分，可 FP8/FP4 化、可跳过反向）。indexer 自己的打分仍然要扫过全部历史 token，这正是它需要一份**独立于主 KV cache 的自己的缓存**（IndexCache，见 `paper-v4.md` §2.3.1 关于 $K^{\mathrm{IComp}}$ 的描述）——否则每步 decode 都要重新计算全部历史 token 的 indexer key，代价与主注意力的 KV cache 缺失同构。

# 五、指路：IndexCache 布局与 FP4 量化变体见 `paper-v4.md`

本报告（arXiv:2512.02556）对 DSA 的描述停在"indexer 可以用 FP8 实现"这一句概述性论断，没有给出索引器自身缓存的具体张量布局，也没有 FP4 变体。落地到 vLLM 的 `vllm/model_executor/layers/deepseek_v4_attention.py` 与 `vllm/v1/attention/backends/mla/indexer.py` 时，索引器缓存的布局与 FP4 量化变体来自 DeepSeek-V4 技术报告（arXiv:2606.19348）——该报告把 DSA 的 lightning indexer 纳入其 Compressed Sparse Attention (CSA) 框架并给出了压缩索引键 $K^{\mathrm{IComp}}$ 的显式构造（§2.3.1）、混合精度存储与 indexer 注意力计算改用 FP4 的效率账（§2.3.4），以及 indexer QK 路径的 FP4 量化感知训练（§5.2.1，含 2× top-k 选择器加速、99.7% 召回率的实测数字）。详见同目录 `paper-v4.md`。
