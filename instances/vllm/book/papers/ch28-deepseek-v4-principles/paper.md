> **Source note**（本包是**策展摘录**，非原文顺抄；所有公式**保持源论文原始编号**，本包不新增编号、不改写公式）
>
> 本包服务 **ch28《【primer】DeepSeek-V4 原理：七件套怎么还两本账》**（v3 重编号后的原理章；其后紧邻 ch29 装配实战）。按「七件套」组织：①两本账动机 ②CSA 压缩稀疏 ③HCA 重压缩 ④滑窗支路 ⑤mHC 多流残差 + Sinkhorn ⑥MoE 与 hash 路由 ⑦MTP。各节来源（每节正文首行再标一次本节局部来源与抓取日期）：
>
> - **主料 arXiv:2606.19348《DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence》**（DeepSeek-AI + 318 作者；v1 = 2026-04-26，本包 2026-09-16 复核 abs 页仍为 v1 唯一版本）：§2.1（MoE / hash 路由 / MTP 沿用）、§2.2（mHC，Eq.(1)-(8)）、§2.3 引言、§2.3.2（HCA，Eq.(20)-(26)）、§2.3.3（滑窗与位置细节）、§2.3.4（效率账）、§5.2.1（FP4 QAT，只取与 MoE 相关一段）。**其中 §2.3 引言、§2.3.1（CSA，Eq.(9)-(19)）、§2.3.4、§5.2.1 整段采用本书 ch27 论文包 `papers/ch27-primer-lightning-indexer/paper-v4.md` 的既有摘录（同一 URL 的 2026-07-10 抓取版），逐字未改、编号未动**——本章 §二/§三/§六 直接引用它们。
> - **谱系锚 arXiv:2512.02556《DeepSeek-V3.2》§2.1（DSA / lightning indexer）**：只借 Eq.(1)(2) 两式说明「V4 的索引器打分函数从哪来」，完整摘录见 `papers/ch27-primer-lightning-indexer/paper.md`。
> - **mHC 前身 arXiv:2409.19606《Hyper-Connections》**（Defa Zhu et al.；v1 2024-09-29 / v3 2025-03-18）：摘要 + Eq.(1)-(9)(14)，2026-09-16 抓 https://ar5iv.labs.arxiv.org/html/2409.19606（ar5iv 渲染页，非官方 HTML——编号已与摘要页核对，公式细节建议以 PDF 复核）。
> - **mHC 本尊 arXiv:2512.24880《Manifold-Constrained Hyper-Connections》**（Zhenda Xie et al.，20 作者；v1 2025-12-31 / v2 2026-01-05）：摘要逐字 + Eq.(3)(5)(6)(7)(8)(9) + §4.1 关键句，2026-09-16 抓 https://arxiv.org/html/2512.24880v2。
> - **MoE arXiv:2401.06066《DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models》**（Damai Dai et al.；v1 2024-01-11）：§3.1-§3.3 的 Eq.(3)-(17)，双渲染交叉核对（官方 https://arxiv.org/html/2401.06066v1 与 ar5iv 两处编号一致）。
> - **hash 路由 arXiv:2106.04426《Hash Layers For Large Sparse Models》**（Stephen Roller et al.；v1 2021-06-08 / v3 2021-07-20）：摘要 + 路由式 + 结论句，2026-09-16 抓 abs 页与 https://ar5iv.labs.arxiv.org/html/2106.04426。
> - **MTP arXiv:2412.19437v2《DeepSeek-V3 Technical Report》§2.2**：Eq.(21)-(25) 与超参口径（D=1、λ=0.3），2026-09-16 抓 https://arxiv.org/html/2412.19437v2。
> - **数字口径（非论文）**：HuggingFace 官方 config `deepseek-ai/DeepSeek-V4-Flash`（本批 2026-09-16 直抓），逐字段见 §八。**论文普遍只给符号（m、m′、n_win、n_hc），具体数字全在 config**——正文引用数字时必须标 config，引用机制必须标论文。
>
> **抓取方式说明**：arXiv 的 `/html/` 页经 WebFetch 摘要器转写，长段落只能给出短引文（≤ 125 字符/条）与公式行。凡标「**原文摘录**」者为逐字引用（含公式行）；标「**据 … 转述**」者为摘要器转述（已尽量保留原短语，未逐字核）；标「**未获取到原文**」者为该页未提供。§九 汇总覆盖度与缺口。

---

# 一、两本账动机：1M 上下文的 FLOPs 账与 KV 账

> 来源：arXiv:2606.19348 摘要 + §2.3.4（**原文摘录**，转引自 `papers/ch27-primer-lightning-indexer/paper-v4.md`，2026-07-10 抓取；2026-09-16 复核 arXiv abs 页，数字未变）。

**摘要（Abstract，节选）**：

We present a preview version of DeepSeek-V4 series, including two strong Mixture-of-Experts (MoE) language models — DeepSeek-V4-Pro with 1.6T parameters (49B activated) and DeepSeek-V4-Flash with 284B parameters (13B activated) — both supporting a context length of one million tokens. DeepSeek-V4 series incorporate several key upgrades in architecture and optimization: (1) a hybrid attention architecture that combines Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA) to improve long-context efficiency; (2) Manifold-Constrained Hyper-Connections (mHC) that enhance conventional residual connections; (3) and the Muon optimizer for faster convergence and greater training stability. [...] In the one-million-token context setting, DeepSeek-V4-Pro requires only 27% of single-token inference FLOPs and 10% of KV cache compared with DeepSeek-V3.2.

**§2.3.4（节选）**：

Taking BF16 GQA8 [ainslie2023gqa] with a head dimension of 128 as the baseline — one of the common configurations of LLM attention — the KV cache size of DeepSeek-V4 series can be dramatically reduced to approximately 2% times of that baseline in the 1M-context setting. Moreover, even when compared with DeepSeek-V3.2 [dsv32] — already an efficient baseline — DeepSeek-V4 series still exhibits substantial advantages in efficiency. A comparison of their inference FLOPs and KV cache size is provided in the right part of Figure 1.

**两本账的读法（本章注解，非原文）**：

- **三个数字、三个分母**：`27% FLOPs` / `10% KV` 的分母是 **DeepSeek-V3.2**（同为 1M 场景、单 token 推理）；`~2% KV` 的分母是 **BF16 GQA8、head_dim=128 这一"常见注意力配置"基线**。正文引用任何一个数字都必须带分母，否则是错误陈述。
- **两笔账在机制上各有归属**：FLOPs 账由「压缩（条数↓）+ 稀疏（每 query 看的条数↓）+ 混合精度计算」三件还；KV 体积账由「压缩条数↓ + 混合精度存储（RoPE 维 BF16、其余 FP8，体积近半，§2.3.4）」还。滑窗是**加账**（额外保真段），不是省账手段。
- **论文只给相对比例**：绝对量纲（字节数、FLOPs 数）论文不给，需按 §八 的 config 自算（43 层 / hidden 4096 / head_dim 512 / 窗口 128 / top-k 512）。
- **Figure 1（右）**：V4 系列 vs V3.2 的推理 FLOPs 与 KV cache 对账柱状图——见 `papers/ch27-primer-lightning-indexer/meta.json` key_figures 第三项。
- 摘要三项升级里的 **Muon 优化器不在本包范围**（本章七件套不含训练优化器）。

# 二、CSA：压缩稀疏注意力（V4 §2.3.1，Eq.(9)-(19)）

> 来源：arXiv:2606.19348 §2.3 引言与 §2.3.1，**整段采用** `papers/ch27-primer-lightning-indexer/paper-v4.md` 的摘录（**原文摘录**，2026-07-10 抓取）：

## 2.0 谱系锚：这套选择机制从哪来（arXiv:2512.02556 §2.1，Eq.(1)(2)）

> 来源：DeepSeek-V3.2 报告（**原文摘录**，完整上下文见 `papers/ch27-primer-lightning-indexer/paper.md`）。V4 的 CSA 就是把这套「打分 + top-k」从 **token 级**搬到 **压缩块级**。

The lightning indexer computes the index score $I_{t,s}$ between the query token $\mathbf{h}_t \in \mathbb{R}^d$ and a preceding token $\mathbf{h}_s \in \mathbb{R}^d$, determining which tokens to be selected by the query token:

$$
I_{t,s} = \sum_{j=1}^{H^I} w_{t,j}^I \cdot \mathrm{ReLU}\left(\mathbf{q}_{t,j}^I \cdot \mathbf{k}_s^I\right)
\tag{1}
$$

$$
\mathbf{u}_t = \mathrm{Attn}\left(\mathbf{h}_t, \left\{\mathbf{c}_s \mid I_{t,s} \in \mathrm{Top}\text{-}k(I_{t,:})\right\}\right)
\tag{2}
$$

## 2.1 压缩：每个条目吸收 $2m$ 个 token 的加权信息（Eq.(9)-(12)）

> 摘自 arXiv:2606.19348 §2.3 "Hybrid Attention with CSA and HCA" 引言与 §2.3.1 "Compressed Key-Value Entries"。

As the context length reaches extreme scales, the attention mechanism emerges as the dominant computational bottleneck in a model. For DeepSeek-V4, we design two efficient attention architectures — Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA) — and employ their **interleaved hybrid configuration**, which substantially reduces the computational cost of attention in long-text scenarios. CSA integrates both compression and sparse attention strategies: it first compresses the Key-Value (KV) cache of every $m$ tokens into one entry, and then applies DeepSeek Sparse Attention (DSA) [dsv32] where each query token attends to only $k$ compressed KV entries. HCA aims for extreme compression by consolidating the KV cache of every $m'$ ($\gg m$) tokens into a single entry.

**Figure 3**（arXiv:2606.19348）图注：Core architectures of CSA. It compresses the number of KV entries to $\frac{1}{m}$ times, and then applies DeepSeek Sparse Attention for further acceleration. Additionally, a small set of sliding window KV entries is combined with the selected compressed KV entries to enhance local fine-grained dependencies. — 见 `papers/ch27-primer-lightning-indexer/meta.json` key_figures 第二项。

Let $H \in \mathbb{R}^{n \times d}$ be a sequence of input hidden states, where $n$ is the sequence length and $d$ is the hidden size. CSA first computes two series of KV entries $C^a, C^b \in \mathbb{R}^{n \times c}$ and their corresponding compression weights $Z^a, Z^b \in \mathbb{R}^{n \times c}$, where $c$ is the head dimension:

$$
C^a = H \cdot W^{aKV}, \quad C^b = H \cdot W^{bKV}
\tag{9}
$$

$$
Z^a = H \cdot W^{aZ}, \quad Z^b = H \cdot W^{bZ}
\tag{10}
$$

where $W^{aKV}, W^{bKV}, W^{aZ}, W^{bZ} \in \mathbb{R}^{d \times c}$ are trainable parameters. Next, each $m$ KV entries in $C^a$ and $C^b$ will be compressed into one entry according to their compression weights and learnable positional biases $B^a, B^b \in \mathbb{R}^{m \times c}$, producing $C^{\mathrm{Comp}} \in \mathbb{R}^{\frac{n}{m} \times c}$. Each compressed entry $C^{\mathrm{Comp}}_i \in \mathbb{R}^c$ is computed by

$$
[S^a_{mi:m(i+1)-1}; S^b_{m(i-1):mi-1}] = \mathrm{Softmax}_{\mathrm{row}}\left([Z^a_{mi:m(i+1)-1} + B^a; Z^b_{m(i-1):mi-1} + B^b]\right)
\tag{11}
$$

$$
C^{\mathrm{Comp}}_i = \sum_{j=mi}^{m(i+1)-1} S^a_j \odot C^a_j + \sum_{j=m(i-1)}^{mi-1} S^b_j \odot C^b_j
\tag{12}
$$

where $\odot$ denotes the Hadamard product; $\mathrm{Softmax}_{\mathrm{row}}(\cdot)$ denotes the softmax operation along the row dimension, which performs normalization across the total of $2m$ elements from both $Z^a$ and $Z^b$. When $i=0$, $Z^b_{m(i-1):mi-1}$ is padded with negative infinity and $C^b_{m(i-1):mi-1}$ is padded with zeros. Note that each $C^{\mathrm{Comp}}_i$ is derived from $2m$ KV entries, but the indexes of $C^b$ used for $C^{\mathrm{Comp}}_i$ and the indexes of $C^a$ used for $C^{\mathrm{Comp}}_{i-1}$ are overlapped. Therefore, CSA in fact compresses the sequence length to $\frac{1}{m}$ times.

**读式 (11)(12) 的三个要点（本章注解）**：

- **「软池化」= 学习加权的加权和**：每个条目不是被「摘要」出来的，而是 $2m$ 个 KV 向量按 softmax 权重逐元素加权求和；权重来自可学习投影 $W^{aZ}/W^{bZ}$ 加可学习位置偏置 $B^a/B^b$——没有生成、没有语言，是纯线性混合。
- **窗宽 $2m$ 与「重叠」**：软归一化跨 $2m$ 个元素（当前窗 $m$ 个 $Z^a$ + 前一窗 $m$ 个 $Z^b$），相邻条目共享一半输入，所以序列恰好压到 $1/m$、边界处信息被平滑而非截断。**config 口径 $m=4$ ⇒ 窗宽 $2m=8$**（V4-Flash `compress_ratios` 中 CSA 层的 4）；本书 ch26 正文已立的「交叠窗宽 8 = 2×压缩比 4」即此式。
- **$i=0$ 的边界处理**是唯一特例（$Z^b$ 填 $-\infty$、$C^b$ 填 0）——实现时等价于「第一个窗口只看自己」。

## 2.2 选择：压缩块上的 indexer 打分与 top-k（Eq.(13)-(17)）

> 摘自同节 "Lightning Indexer for Sparse Selection"。

After obtaining the compressed KV entries $C^{\mathrm{Comp}}$, CSA applies the DSA strategy to select top-k compressed KV entries for core attention. First, CSA performs the same compression operation used for $C^{\mathrm{Comp}}$ to get **compressed indexer keys** $K^{\mathrm{IComp}} \in \mathbb{R}^{\frac{n}{m} \times c^I}$, where $c^I$ is the indexer head dimension. Then, for a query token $t$, we produce the indexer queries $\{\mathbf{q}_{t,1}^I; \mathbf{q}_{t,2}^I; ...; \mathbf{q}_{t,n_h^I}^I\}$ in a low-rank manner:

$$
\mathbf{c}_t^Q = \mathbf{h}_t \cdot W^{DQ}
\tag{13}
$$

$$
[\mathbf{q}_{t,1}^I; \mathbf{q}_{t,2}^I; ...; \mathbf{q}_{t,n_h^I}^I] = \mathbf{q}_t^I = \mathbf{c}_t^Q \cdot W^{IUQ}
\tag{14}
$$

where $\mathbf{h}_t \in \mathbb{R}^d$ is the input hidden state of the query token $t$; $\mathbf{c}_t^Q \in \mathbb{R}^{d_c}$ is the compressed latent vector for queries; $d_c$ denotes the query compression dimension; $n_h^I$ denotes the number of indexer query heads; $W^{DQ} \in \mathbb{R}^{d \times d_c}$ and $W^{IUQ} \in \mathbb{R}^{d_c \times c^I n_h^I}$ are the down-projection and up-projection matrices for indexer queries, respectively. Next, the index score $I_{t,s} \in \mathbb{R}$ between the query token $t$ and a preceding compressed block $s$ (with $s < \mathrm{Floor}(t/m)$) is computed by

$$
[w_{t,1}^I; w_{t,2}^I; ...; w_{t,n_h^I}^I] = \mathbf{w}_t^I = \mathbf{h}_t \cdot W^w
\tag{15}
$$

$$
I_{t,s} = \sum_{h=1}^{n_h^I} w_{t,h}^I \cdot \mathrm{ReLU}\left(\mathbf{q}^I_{t,h} \cdot K^{\mathrm{IComp}}_s\right)
\tag{16}
$$

where $W^w \in \mathbb{R}^{d \times n_h^I}$ is a learnable matrix; $w_{t,h}^I \in \mathbb{R}$ is the weight of the $h$-th indexer head. For a query token $t$, given its index scores $I_{t,:}$, we employ a top-k selector to selectively retain a subset of compressed KV entries $\mathcal{C}_t^{\mathrm{SprsComp}}$ for subsequent core attention:

$$
\mathcal{C}_t^{\mathrm{SprsComp}} = \left\{C^{\mathrm{Comp}}_s \;\middle|\; I_{t,s} \in \mathrm{Top}\text{-}k(I_{t,:})\right\}
\tag{17}
$$

**与 DSA 的两个差别（本章注解）**：

- **打分对象从 token 换成压缩块**：式 (16) 与 V3.2 的 Eq.(1) 同形（同是「多头 $w$ 加权 + ReLU 内积」），但 $\mathbf{k}^I_s$ 换成 $K^{\mathrm{IComp}}_s$——索引器自己也有一份**压缩后的键缓存**（构造方式 = 对索引键做与 $C^{\mathrm{Comp}}$ 相同的压缩），因此打分成本随 $n/m$ 而非 $n$ 增长。这正是 ch27 论文包 `paper-v4.md` §2.3.1 里 IndexCache 的来历。
- **因果约束 $s < \mathrm{Floor}(t/m)$**：query 只能选**严格在自己所在块之前**的完整块（块内 token 谁先谁后无法在块级表达）。

## 2.3 核心注意力：共享 KV 的 MQA（Eq.(18)(19)）

> 摘自同节 "Shared Key-Value MQA"。

After selecting the sparse KV entries, CSA then performs core attention in a Multi-Query Attention (MQA) [mqa] manner, where each compressed KV entry in $\mathcal{C}_t^{\mathrm{SprsComp}}$ serves as both attention key and value. To be specific, for a query token $t$, we first produce attention queries $\{\mathbf{q}_{t,1}; \mathbf{q}_{t,2}; ...; \mathbf{q}_{t,n_h}\}$ from the compressed latent vector $\mathbf{c}_t^Q$:

$$
[\mathbf{q}_{t,1}; \mathbf{q}_{t,2}; ...; \mathbf{q}_{t,n_h}] = \mathbf{q}_t = \mathbf{c}_t^Q \cdot W^{UQ}
\tag{18}
$$

where $n_h$ denotes the number of query heads; $W^{UQ} \in \mathbb{R}^{d_c \times c n_h}$ is the up-projection matrices for queries. Note that the latent query vector $\mathbf{c}_t^Q$ is shared with that used for the indexer queries. Next, we perform MQA on $\{\mathbf{q}_{t,i}\}$ and $\mathcal{C}_t^{\mathrm{SprsComp}}$:

$$
\mathbf{o}_{t,i} = \mathrm{CoreAttn}\left(\mathrm{query} = \mathbf{q}_{t,i}, \mathrm{key} = \mathcal{C}_t^{\mathrm{SprsComp}}, \mathrm{value} = \mathcal{C}_t^{\mathrm{SprsComp}}\right)
\tag{19}
$$

**IndexCache 要点（对应上段原文）**：indexer 自己维护的缓存不是主 KV cache 的旁支，而是一份形状独立的张量——$K^{\mathrm{IComp}} \in \mathbb{R}^{n/m \times c^I}$，行数与主压缩 KV cache $C^{\mathrm{Comp}}$ 同步增长（都以 $n/m$ 个压缩块为单位），但列宽是 indexer 专属的头维度 $c^I$（config：`index_head_dim=128`），与主注意力头维度 $c$（config：`head_dim=512`）无关——这正是「独立小头」在缓存层面的体现。

# 三、HCA：重压缩注意力（V4 §2.3.2，Eq.(20)-(26)）

> 来源：arXiv:2606.19348 §2.3.2（**公式行为原文摘录**；本节 O 段 prose 本批只取到首句/尾句/图注/关键句，其余为「**据转述**」——见 §九）。**注意：ch27 论文包的 `paper-v4.md` 明确省略了 §2.3.2（当时主题是索引器），本节是本包新补的内容。**

**首句（原文摘录）**：

The core architecture of HCA is illustrated in Figure 4, which compresses the KV cache in a heavier manner, but does not employ sparse attention.

**Figure 4** 图注（原文摘录）：Core architectures of HCA. It performs heavier compression, where the KV entries of $m'$ ($\gg m$) tokens will be consolidated into one.

**压缩（Eq.(20)-(23)，公式原文摘录）**：

$$
C = H \cdot W^{KV},
\tag{20}
$$

$$
Z = H \cdot W^{Z},
\tag{21}
$$

$$
S_{m'i:m'(i+1)-1} = \operatorname{Softmax}_{\text{row}}\left(Z_{m'i:m'(i+1)-1} + B\right),
\tag{22}
$$

$$
C^{\text{Comp}}_{i} = \sum_{j=m'i}^{m'(i+1)-1} S_{j} \odot C_{j}.
\tag{23}
$$

**共享 KV 的 MQA（Eq.(24)-(26)，公式原文摘录）**：

$$
\mathbf{c}_{t}^{Q} = \mathbf{h}_{t} \cdot W^{DQ},
\tag{24}
$$

$$
[\mathbf{q}_{t,1};\mathbf{q}_{t,2};...;\mathbf{q}_{t,n_{h}}]=\mathbf{q}_{t} = \mathbf{c}_{t}^{Q}\cdot W^{UQ},
\tag{25}
$$

$$
\mathbf{o}_{t,i}=\operatorname{CoreAttn}(\texttt{query=}\mathbf{q}_{t,i},\texttt{key}=C^{\text{Comp}},\texttt{value}=C^{\text{Comp}}),
\tag{26}
$$

**尾句（原文摘录）**：Finally, HCA projects the intermediate output $[o^{G'}_{t,1};o^{G'}_{t,2};\dots;o^{G'}_{t,g}] \in \mathbb{R}^{d_g g}$ to the final attention output $\hat{o}_{t} \in \mathbb{R}^{d}$.

**关键差异（原文摘录 + 本章注解）**：

- 「employs a larger compression rate $m'$ ($\gg m$)」「HCA compresses the sequence length to $\frac{1}{m'}$ times」（**据转述**，原短语保留）；**不做稀疏选择**（"does not employ sparse attention"，原文摘录）。
- **与 CSA 共用同一套压缩器机制**：把 Eq.(20)-(23) 与 Eq.(9)-(12) 并排看，机制同构（两组投影 → 权重投影 → 块内 softmax 行归一 → 加权求和），差别只在 **HCA 只有一组** $C/Z$（CSA 有两组 $a/b$、靠重叠窗平滑）、**窗宽是 $m'$ 而非 $2m$**、**并且没有 indexer**（此对比是公式层面的观察，非论文原话）。
- **「压完全看不挑」**：query 在因果允许范围内看**全部**压缩块（Eq.(26) 的 key/value 是整份 $C^{\text{Comp}}$，不是某个 top-k 子集）——压到条目足够少（config 口径 $m'=128$ ⇒ 1M 序列只剩 $\approx 7{,}812$ 块）之后，稠密注意力本身已足够便宜，选择器的开销与盲区都不划算。
- **config 口径**：HCA 层 = `compress_ratios` 里取 128 的层（V4-Flash 43 层主干中 20 层为 128，其余为 4 或 0）。
- Eq.(26) 之后的分组输出投影（尾句）是 HCA/CSA 共有的收尾件；论文在 CSA 侧未单列此式。

# 四、滑窗支路：因果、块内失明与「攒批时间差」（V4 §2.3.3）

> 来源：arXiv:2606.19348 §2.3.3 "Other Details"（**原文摘录**，本批抓取；每条均 ≤ 125 字符短引文）。

**论文口径（原文摘录）**：

- In order to strictly preserve causality in CSA and HCA, each query attends to only preceding compressed KV blocks.
- a query cannot access information from other tokens within its own compressed block.
- recent tokens usually possess greater relevance to the query token in language modeling.
- we additionally produce $n_{\text{win}}$ uncompressed KV entries corresponding to the recent $n_{\text{win}}$ tokens.
- these KV entries in the sliding window will be used along with the compressed KV entries.

**Figure 3 图注里的对应句（原文摘录，转引自 `paper-v4.md`）**：Additionally, a small set of sliding window KV entries is combined with the selected compressed KV entries to enhance local fine-grained dependencies.

**位置与数值细节（原文摘录，同节）**：

- RoPE 只作用于最后 64 维；核心注意力输出侧做反向旋转：we also apply RoPE with position $-i$ on the last 64 dimensions of each $o_{t,i}$。
- HCA 层另有 attention sink（可学习 sink logit）：Exp($z'_h$) will be added to the denominator of the attention score；allows each query head to adjust its total attention scores to be not equal to 1, and even to be near 0。
- 论文**未给 $n_{\text{win}}$ 的数值**；config 口径 `sliding_window=128`（两型号一致）。

**三层理由（本章注解；前两层属论文原话，第三层是实现层口径）**：

1. **严格因果**：压缩块是「整块」单位，块内 token 的相对前后在块级无法表达 ⇒ 第 $t$ 个 query 只能看严格在前的完整块，当前块内更早的 token 会「失明」。
2. **近处更相关**：语言建模里最近的 token 通常更该被精确看到，而它们要么在未完成的块里、要么刚被压进一个混了 $m'$ 个 token 的块里。
3. **攒批时间差（实现层，非论文原话）**：压缩机每 $m$（或 $m'$）个 token 才产出一个整块，decode 时最新不满一块的那几十个 token 还没有对应压缩条目——**只有滑窗能看见它们的原样 KV**。本书 ch26 正文已立此口径（`artifacts-v3/ch26-deepseek-indexer-nsa-dsa/narrative/chapter.md`：「pos 3 只攒出 1 个压缩块，buffer 行选 [0] 其余全 -1——同一拍里缓存先建、选块极简」）；第三方参考实现 lc2 讲义亦同（dhcode-cpp/DeepSeek-V4-mini，confidence: medium）。
4. **ratio=0 的层**：`compress_ratios` 中取 0 的层没有压缩机，滑窗支路即该层的全部注意力（V4-Flash 前两层 0、0；末位 0 对应 MTP 槽）。

# 五、mHC：多流残差 + Sinkhorn 投影

## 5.1 前身：Hyper-Connections（arXiv:2409.19606）

> 来源：摘要页（**原文摘录**，2026-09-16 抓 https://arxiv.org/abs/2409.19606）+ ar5iv 正文公式（**公式行原文摘录**，同批抓 https://ar5iv.labs.arxiv.org/html/2409.19606）。Defa Zhu, Hongzhi Huang, Zihao Huang, Yutao Zeng, Yunyao Mao, Banggu Wu, Qiyang Min, Xun Zhou；v1 2024-09-29 / v3 2025-03-18。

**摘要（分句摘录）**：

- We present hyper-connections, a simple yet effective method that can serve as an alternative to residual connections.
- it specifically addresses common drawbacks observed in residual connection variants, such as the seesaw effect between gradient vanishing and representation collapse.
- hyper-connections allow the network to adjust the strength of connections between features at different depths and dynamically rearrange layers.
- We conduct experiments focusing on the pre-training of large language models, including dense and sparse models, where hyper-connections show significant performance improvements over residual connections.

**机制（公式行原文摘录，§2；编号为 HC 论文原始编号）**：

$$
\mathcal{HC}=\begin{pmatrix}\mathbf{0}_{1\times 1}&\mathbf{B}\\ \mathbf{A_{m}}&\mathbf{A_{r}}\end{pmatrix}\in\mathbb{R}^{(n+1)\times(n+1)}
\tag{1}
$$

$$
\mathbf{\hat{H}}=\mathbf{B}^{\intercal}\mathcal{T}(\mathbf{H}^{\intercal}\mathbf{A_{m}})^{\intercal}+\mathbf{A_{r}}^{\intercal}\mathbf{H}
\tag{2}
$$

$$
\mathbf{h}_{0}^{\intercal}=\mathbf{A_{m}}^{\intercal}\mathbf{H}
\tag{3}
$$

$$
\mathbf{H^{\prime}}=\mathbf{A_{r}}^{\intercal}\mathbf{H}
\tag{4}
$$

$$
\mathcal{HC}(\mathbf{H})=\begin{pmatrix}\mathbf{0}_{1\times 1}&\mathcal{B}(\mathbf{H})\\ \mathcal{A}_{m}(\mathbf{H})&\mathcal{A}_{r}(\mathbf{H})\end{pmatrix}
\tag{8}
$$

$$
\mathbf{\hat{H}}=\mathcal{HC}(\mathbf{H})(\mathcal{T},\mathbf{H})
\tag{9}
$$

**要点（原文摘录 + 注解）**：残差流从 1 条扩成 $n$ 条——初始状态是输入「replicated $n$ times」：$\mathbf{H}^{0}=(\mathbf{h}^{0}\ \mathbf{h}^{0}\ \dots\ \mathbf{h}^{0})^{\intercal}\in\mathbb{R}^{n\times d}$；实验用 expansion rate 最高 $n=8$、**默认 $n=4$（DHC）**；论文指出 Pre-Norm / Post-Norm 都是它的「non-trainable forms」；静态版（Eq.(1)-(4)）与动态版（Eq.(8)(9)，矩阵本身是 $\mathbf{H}$ 的函数）分开给。**V4 的 config 口径 `hc_mult=4` 正对应这里的默认 $n=4$。**

## 5.2 mHC 本尊（arXiv:2512.24880）

> 来源：摘要逐字 + 正文公式行（**原文摘录**，2026-09-16 抓 https://arxiv.org/html/2512.24880v2）。Zhenda Xie, Yixuan Wei, Huanqi Cao, Chenggang Zhao, Chengqi Deng, Jiashi Li, Damai Dai, Huazuo Gao, Jiang Chang, Kuai Yu, Liang Zhao, Shangyan Zhou, Zhean Xu, Zhengyan Zhang, Wangding Zeng, Shengding Hu, Yuqing Wang, Jingyang Yuan, Lean Wang, Wenfeng Liang；v1 2025-12-31 / v2 2026-01-05。

**摘要（逐字）**：

Recently, studies exemplified by Hyper-Connections (HC) have extended the ubiquitous residual connection paradigm established over the past decade by expanding the residual stream width and diversifying connectivity patterns. While yielding substantial performance gains, this diversification fundamentally compromises the identity mapping property intrinsic to the residual connection, which causes severe training instability and restricted scalability, and additionally incurs notable memory access overhead. To address these challenges, we propose Manifold-Constrained Hyper-Connections (mHC), a general framework that projects the residual connection space of HC onto a specific manifold to restore the identity mapping property, while incorporating rigorous infrastructure optimization to ensure efficiency. Empirical experiments demonstrate that mHC is effective for training at scale, offering tangible performance improvements and superior scalability. We anticipate that mHC, as a flexible and practical extension of HC, will contribute to a deeper understanding of topological architecture design and suggest promising directions for the evolution of foundational models.

**更新式（Eq.(3)，原文摘录，§1）**——mHC 用 $H^{pre}/H^{post}/H^{res}$ 三映射，不用 A/B/C：

$$
x_{l+1} = H_l^{res} x_l + H_l^{post \top} F(H_l^pre x_l, W_l)
\tag{3}
$$

其中 $H_l^{pre}, H_l^{post} \in \mathbb{R}^{1\times n}$（读入/写回）、$H_l^{res} \in \mathbb{R}^{n\times n}$（流间混合）；Eq.(4) 给出跨层复合形式 $x_L = \left(\prod_{i=1}^{L-l} H_{L-i}^{res}\right) x_l + \Sigma\ldots$（**抓取片段，求和项被截断**——需要完整式请以 PDF 为准）。

**参数化（Eq.(5)，原文摘录，§3 Preliminary）**——raw 参数由「输入相关」生成：

$$
\begin{aligned}
\tilde{x}_l &= \mathrm{RMSNorm}(x_l)\\
H_l^{pre} &= \alpha_l^{pre} \cdot \tanh(\theta_l^{pre} \tilde{x}_l^\top) + b_l^{pre}\\
H_l^{post} &= \alpha_l^{post} \cdot \tanh(\theta_l^{post} \tilde{x}_l^\top) + b_l^{post}\\
H_l^{res} &= \alpha_l^{res} \cdot \tanh(\theta_l^{res} \tilde{x}_l^\top) + b_l^{res}
\end{aligned}
\tag{5}
$$

**流形约束（Eq.(6) 与 §4.1，原文摘录）**——$H_l^{res}$ 被投影到 Birkhoff 多面体（双随机矩阵集合）：

$$
P_{M^{res}}(H_l^{res}) := \left\{ H_l^{res} \in \mathbb{R}^{n\times n} \;\middle|\; H_l^{res} \mathbf{1}_n = \mathbf{1}_n,\; \mathbf{1}_n^\top H_l^{res} = \mathbf{1}_n^\top,\; H_l^{res} \geqslant 0 \right\}
\tag{6}
$$

§4.1 列明该约束的三条性质（**据转述 + 短引原文**）：(1) 范数保持——$\|H_l^{res}\|_2 \le 1$，论文原短语 "is bounded by 1"；(2) 恢复恒等映射性质（identity mapping）；(3) 该集合对乘法封闭（composition 仍在此集合内）。

**参数化与投影（Eq.(7)(8)(9)，原文摘录，§4.2 "Parameterization and Manifold Projection"）**——flattened 变体：

$$
\begin{aligned}
\vec{x}'_l &= \mathrm{RMSNorm}(\vec{x}_l)\\
\tilde{H}_l^{pre} &= \alpha_l^{pre} \cdot (\vec{x}'_l \varphi_l^{pre}) + b_l^{pre}\\
\tilde{H}_l^{post} &= \alpha_l^{post} \cdot (\vec{x}'_l \varphi_l^{post}) + b_l^{post}\\
\tilde{H}_l^{res} &= \alpha_l^{res} \cdot \mathrm{mat}(\vec{x}'_l \varphi_l^{res}) + b_l^{res}
\end{aligned}
\tag{7}
$$

$$
H_l^{pre} = \sigma(\tilde{H}_l^{pre}), \qquad H_l^{post} = 2\sigma(\tilde{H}_l^{post}), \qquad H_l^{res} = \mathrm{Sinkhorn\text{-}Knopp}(\tilde{H}_l^{res})
\tag{8}
$$

$$
M^{(t)} = T_r(T_c(M^{(t-1)}))
\tag{9}
$$

迭代起点 $M^{(0)} = \exp(\tilde{H}_l^{res})$，$T_r / T_c$ 为行/列归一化算子；论文：converges to a doubly stochastic matrix；**We choose $t_{\text{max}}=20$.**（即 $H_l^{res} = M^{(t_{\text{max}})}$。）

## 5.3 V4 报告的适配形式（§2.2，Eq.(1)-(8)）——本章正文采用的口径

> 来源：arXiv:2606.19348 §2.2（**公式行为原文摘录**，2026-09-16 抓；符号命名 A/B/C 是该报告自己的写法，对应 mHC 原论文的 $H^{pre}/H^{res}/H^{post}$）。

**引入（原文摘录）**：DeepSeek-V4 uses "Manifold-Constrained Hyper-Connections (mHC) to strengthen the conventional residual connections"；naive HC 的病：training will frequently exhibit numerical instability when stacking multiple layers。

**残差状态与更新式（Eq.(1)）**——残差流是 $X \in \mathbb{R}^{n_{hc} \times d}$（宽度扩 $n_{hc}$ 倍），每半层做「读入 → 子层 → 写回」：

$$
X_{l+1}=B_{l}X_{l}+C_{l}\mathcal{F}_{l}(A_{l}X_{l}),
\tag{1}
$$

其中 $A_l$ 读入（把 $n_{hc}$ 条流聚成子层输入）、$B_l$ 流间混合（残差自带）、$C_l$ 写回（把子层输出散回 $n_{hc}$ 条流）（**据转述**；三者的角色名是该报告 §2.2 表述，符号本身为 Eq.(1) 原文）。

**双随机约束（Eq.(2)）**——$B_l$ 被关进 Birkhoff 多面体：

$$
B_{l}\in\mathcal{M}\coloneq\{M\in\mathbb{R}^{n\times n}\mid M\mathbf{1}_{n}=\mathbf{1}_{n},\;\mathbf{1}_{n}^{T}M=\mathbf{1}_{n}^{T},\;M\geqslant 0\}.
\tag{2}
$$

（抓取片段在最后一个条件处截断，此处按 mHC 原论文 Eq.(6) 的同构式补齐收尾；论文原句：the spectral norm of the mapping matrix $\|B_{l}\|_{2}$ is bounded by 1 —— 并指出双随机集合对乘法封闭 ⇒ 深堆叠也稳。）

**动态参数化（Eq.(3)(4)(5)，raw 参数）**——先算 RMSNorm 展平态 $\hat{X}_l = \mathrm{RMSNorm}(\mathrm{vec}(X_l)) \in \mathbb{R}^{1\times n_{hc}d}$：

$$
\tilde{A}_l = \alpha_{l}^{\mathrm{pre}}\cdot(\hat{X}_{l}W^{\mathrm{pre}}_{l})+S_{l}^{\mathrm{pre}},
\tag{3}
$$

$$
\tilde{B}_l = \alpha_{l}^{\mathrm{res}}\cdot\operatorname{Mat}(\hat{X}_{l}W^{\mathrm{res}}_{l})+S_{l}^{\mathrm{res}},
\tag{4}
$$

$$
\tilde{C}_l = \alpha_{l}^{\mathrm{post}}\cdot(\hat{X}_{l}W^{\mathrm{post}}_{l})^{T}+S_{l}^{\mathrm{post}},
\tag{5}
$$

符号（原文摘录）：

- $\alpha_l^{\mathrm{pre}}, \alpha_l^{\mathrm{res}}, \alpha_l^{\mathrm{post}} \in \mathbb{R}$ are learnable gating factors（初始化很小）；
- $S_l^{\mathrm{pre}} \in \mathbb{R}^{1\times n_{hc}}, S_l^{\mathrm{post}} \in \mathbb{R}^{n_{hc}\times 1}$, and $S_l^{\mathrm{res}} \in \mathbb{R}^{n_{hc}\times n_{hc}}$ are learnable static biases；
- $W_l^{\mathrm{pre}}, W_l^{\mathrm{post}} \in \mathbb{R}^{n_{hc}d\times n_{hc}}$ and $W_l^{\mathrm{res}} \in \mathbb{R}^{n_{hc}d\times n_{hc}^{2}}$ are learnable parameters；
- The parameters of three linear mappings are dynamically generated, which are decomposed into a dynamic (input-dependent) component and a static (input-independent) component.
- $\operatorname{Mat}(\cdot)$ = 把 $1\times n_{hc}^2$ 的行向量重排成 $n_{hc}\times n_{hc}$ 矩阵。

**约束与投影（Eq.(6)(7)(8)）**：

$$
A_{l} = \sigma(\tilde{A}_{l}),
\tag{6}
$$

$$
C_{l} = 2\sigma(\tilde{C}_{l}).
\tag{7}
$$

$$
M^{(t)}=\mathcal{T}_{r}(\mathcal{T}_{c}(M^{(t-1)})),
\tag{8}
$$

以及原句：**We choose $t_{\text{max}}=20$ as a practical value.**（$B_l = M^{(t_{\text{max}})}$，迭代起点 $M^{(0)}=\exp(\tilde{B}_l)$。）

**符号对照（本章注解，三方对齐用）**：

| 角色 | V4 报告 §2.2 | mHC 原论文 | config（V4-Flash） |
|---|---|---|---|
| 读入（聚合成子层输入） | $A_l$，$A_l=\sigma(\tilde{A}_l)$ | $H_l^{pre}$，$\sigma(\tilde{H}_l^{pre})$ | `hc_mult=4`（流数 $n_{hc}$） |
| 流间混合（残差） | $B_l$ ∈ Birkhoff 多面体 | $H_l^{res}$ = Sinkhorn-Knopp$(\tilde{H}_l^{res})$ | `hc_sinkhorn_iters=20`、`hc_eps=1e-06` |
| 写回（散回 $n_{hc}$ 条流） | $C_l$，$C_l=2\sigma(\tilde{C}_l)$ | $H_l^{post}$，$2\sigma(\tilde{H}_l^{post})$ | — |
| 迭代上限 | $t_{\text{max}}=20$ | $t_{\text{max}}=20$ | `hc_sinkhorn_iters=20`（互证） |

## 5.4 支撑：Sinkhorn-Knopp 与双随机（为什么是 20 次）

- **双随机矩阵集合 = Birkhoff 多面体**（Eq.(6)/Eq.(2) 的定义即其定义）：非负、每行和为 1、每列和为 1。置换矩阵是该集合的**顶点**（标准线性代数事实，非论文原话——本包补充）。
- **非扩张性**：$\|B\|_2 \le 1$（论文原句 "bounded by 1"）——任何一次流间混合都不放大信号；双随机集合对乘法封闭 ⇒ 连续多层混合后范数仍 ≤ 1。这是 mHC「恢复恒等映射性质」的数学核心：**约束而不是放弃多流**。
- **Sinkhorn-Knopp 迭代**：从 $M^{(0)}=\exp(\tilde{B})$ 出发，交替做列归一化 $T_c$ 与行归一化 $T_r$（Eq.(8)/(9)），收敛到双随机矩阵；V4/mHC 只迭代 $t_{\text{max}}=20$ 次，**是精度与开销的实用折中，不是精确投影**（论文原句 "as a practical value"）。
- 一般背景（非 V4/mHC 论文断言）：Sinkhorn 定理（1964）——任何全正矩阵经正对角缩放可变为双随机；交替归一迭代即其构造性证明。参考 Wikipedia「Sinkhorn's theorem」（confidence: medium，仅背景，未逐字核对原始文献）。

# 六、MoE 与 hash 路由

## 6.1 DeepSeekMoE 的两板斧（arXiv:2401.06066 §3.1-§3.3，Eq.(3)-(17)）

> 来源：**公式行原文摘录**，2026-09-16 双渲染交叉核对（官方 https://arxiv.org/html/2401.06066v1 与 https://ar5iv.labs.arxiv.org/html/2401.06066 编号一致）。Damai Dai et al.，v1 2024-01-11。

**标准 MoE 层（Eq.(3)(4)(5)，§3.1）**：

$$
h_t^l = \sum_{i=1}^{N} g_{i,t} FFN_i(u_t^l) + u_t^l
\tag{3}
$$

$$
g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \mathrm{Topk}(\{s_{j,t} \mid 1 \le j \le N\}, K)\\ 0, & \text{otherwise} \end{cases}
\tag{4}
$$

$$
s_{i,t} = \mathrm{Softmax}_i\left({u_t^l}^\top e_i^l\right)
\tag{5}
$$

**板斧一：细粒度专家切分（Eq.(6)(7)(8)）**——原文：we segment each expert FFN into $m$ smaller experts by reducing the FFN intermediate hidden dimension to $1/m$ times its original size；and increase the number of activated experts to $m$ times to keep the same computation cost：

$$
h_t^l = \sum_{i=1}^{mN} g_{i,t} FFN_i(u_t^l) + u_t^l
\tag{6}
$$

$$
g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \mathrm{Topk}(\{s_{j,t} \mid 1 \le j \le mN\}, mK)\\ 0, & \text{otherwise} \end{cases}
\tag{7}
$$

$$
s_{i,t} = \mathrm{Softmax}_i\left({u_t^l}^\top e_i^l\right)
\tag{8}
$$

（摘要口径：finely segmenting the experts into $mN$ ones and activating $mK$ from them ⇒ a more flexible combination of activated experts。）

**板斧二：共享专家隔离（Eq.(9)(10)(11)，§3.2）**——原文：isolate $K_s$ experts to serve as shared experts，其职责是 capturing and consolidating common knowledge across varying contexts，从而 mitigate redundancy among other routed experts：

$$
h_t^l = \sum_{i=1}^{K_s} FFN_i(u_t^l) + \sum_{i=K_s+1}^{mN} g_{i,t} FFN_i(u_t^l) + u_t^l
\tag{9}
$$

$$
g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \mathrm{Topk}(\{s_{j,t} \mid K_s+1 \le j \le mN\}, mK - K_s)\\ 0, & \text{otherwise} \end{cases}
\tag{10}
$$

$$
s_{i,t} = \mathrm{Softmax}_i\left({u_t^l}^\top e_i^l\right)
\tag{11}
$$

**读法（本章注解）**：前 $K_s$ 个专家**无条件计算**（式 (9) 第一项没有 gate），路由只在前者之后的 $mN-K_s$ 个专家里挑 $mK-K_s$ 个。这就是 V4 config 里 `n_shared_experts=1`（无条件算 1 个）+ `n_routed_experts=256`、`num_experts_per_tok=6`（挑 6 个）的直接出处机制。

**负载均衡损失（Eq.(12)-(14) 专家级 / Eq.(15)-(17) 设备级，§3.3，原文摘录）**：

$$
L_{ExpBal} = \alpha_1 \sum_{i=1}^{N'} f_i P_i
\tag{12}
$$

$$
f_i = \frac{N'}{K'T}\sum_{t=1}^{T} \mathds{1}(\text{Token } t \text{ selects Expert } i)
\tag{13}
$$

$$
P_i = \frac{1}{T}\sum_{t=1}^{T} s_{i,t}
\tag{14}
$$

$$
L_{DevBal} = \alpha_2 \sum_{i=1}^{D} f_i' P_i'
\tag{15}
$$

$$
f_i' = \frac{1}{|\mathcal{E}_i|}\sum_{j\in\mathcal{E}_i} f_j
\tag{16}
$$

$$
P_i' = \sum_{j\in\mathcal{E}_i} P_j
\tag{17}
$$

（V4 不再用这两个损失作主力：见 6.3 的 auxiliary-loss-free，Eq.(12)-(17) 在这里是 DeepSeekMoE 的原始设计，作谱系背景。）

## 6.2 Hash Layers：查表派单（arXiv:2106.04426）

> 来源：**原文摘录**，2026-09-16 抓 abs 页与 https://ar5iv.labs.arxiv.org/html/2106.04426。Stephen Roller, Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston；v1 2021-06-08 / v3 2021-07-20。

**摘要（分句摘录）**：

- We investigate the training of sparse layers that use different parameters for different inputs based on hashing in large Transformer models.
- Specifically, we modify the feedforward layer to hash to different sets of weights depending on the current token, over all tokens in the sequence.
- We show that this procedure either outperforms or is competitive with learning-to-route mixture-of-expert methods such as Switch Transformers and BASE Layers, while requiring **no routing parameters or extra terms in the objective function such as a load balancing loss, and no sophisticated assignment algorithm**.
- We study the performance of different hashing techniques, hash sizes and input features, and show that **balanced and random hashes focused on the most local features work best**, compared to either learning clusters or using longer-range context.

**路由式与机制（原文摘录）**：

$$
\mathbf{h}^l_t=\mbox{FFN}_{\mbox{hash}(x_t)}(\bar{\mathbf{h}}^l_t), \qquad t=1,\dots,T.
$$

- our routing function uses the **original input token** $x_t$ rather than the hidden state.
- by hashing the tokens into a fixed number of buckets, each bucket corresponding to an expert.
- 变体：Bigram Hash（当前 + 前一 token）、Previous Token Hash、MultiHash（$N$ 个独立 hash 函数，各自选一段 FFN 权重后拼接）。
- 结论细节：balanced 与 random 的指派「perform similarly well」；learned clustering 比 random「clearly worse」；Switch 型方案需要 a load balancing term in the objective function，hash 方案不需要。

## 6.3 V4 的口径（§2.1 原文摘录 + config）

> 来源：arXiv:2606.19348 §2.1（**原文摘录**，2026-09-16 抓）+ HuggingFace `deepseek-ai/DeepSeek-V4-Flash` config（2026-09-16 直抓）。

**§2.1 原文摘录（短引）**：

- V4 的 FFN 沿用 DeepSeekMoE：which sets **fine-grained routed experts and shared experts**。
- 亲和分激活函数换掉：we change the activation function that computes the affinity scores from Sigmoid(·) into **Sqrt(Softplus(·))**。
- 负载均衡：For load balancing, we also employ the **auxiliary-loss-free strategy**, augmented by a slight sequence-wise balance loss that prevents extreme imbalance within individual sequences。
- hash 路由：we replace the dense FFN layers in the initial several Transformer blocks with MoE layers that employ **Hash routing**；The Hash routing strategy determines the target experts of each token according to a **predefined hash function with regard to the input token ID**。
- （据转述）该报告还提到取消了「路由目标节点数」的上限并重设了并行策略（与本章原理无关，留给 ch29/ch35）。

**config 口径（2026-09-16 直抓，逐字段）**：

| 字段 | 值 | 含义 |
|---|---|---|
| `n_routed_experts` | 256 | 路由专家数（V4-Pro 为 384） |
| `n_shared_experts` | 1 | 无条件计算的共享专家数 |
| `num_experts_per_tok` | 6 | 每 token 激活的路由专家数 |
| `num_hash_layers` | 3 | 最前 3 层的路由由 hash 表决定 |
| `moe_intermediate_size` | 2048 | 单个专家的 FFN 中间维（hidden 4096 的一半 ⇒ 「切得细」） |
| `hidden_size` | 4096 | — |
| `scoring_func` | `sqrtsoftplus` | 与 §2.1 的 Sqrt(Softplus) 对应 |
| `topk_method` | `noaux_tc` | auxiliary-loss-free 的选择法 |

**注解**：论文 §2.1 **未给任何专家数**（256/6/384 等全在 config）；「为什么只有前 3 层用 hash」论文未说明——正文不要替论文编理由（本书 ch29 已立此纪律）。

# 七、MTP：多 token 预测（V3 §2.2 Eq.(21)-(25)；V4 原样沿用）

> 来源：arXiv:2412.19437v2《DeepSeek-V3 Technical Report》§2.2 "Multi-Token Prediction"（**公式行与短句原文摘录**，2026-09-16 抓）+ arXiv:2606.19348 §2.1（沿用声明）。

**模块结构（Eq.(21)(22)(23)）**：

$$
h'^k_i = M_k\left[\mathrm{RMSNorm}(h^{k-1}_i); \mathrm{RMSNorm}(\mathrm{Emb}(t_{i+k}))\right]
\tag{21}
$$

$$
h^k_{1:T-k} = \mathrm{TRM}_k(h'^k_{1:T-k})
\tag{22}
$$

$$
P^k_{i+k+1} = \mathrm{OutHead}(h^k_i)
\tag{23}
$$

**损失（Eq.(24)(25)）**：

$$
L^k_{MTP} = \mathrm{CrossEntropy}(P^k_{2+k:T+1}, t_{2+k:T+1})
\tag{24}
$$

$$
L_{MTP} = \frac{\lambda}{D}\sum_{k=1}^{D} L^k_{MTP}
\tag{25}
$$

**关键短句（原文摘录）**：

- its embedding layer is shared with the main model；its output head is shared with the main model（$M_k$ 与输出头复用主模型件）。
- our MTP implementation uses $D$ **sequential** modules to predict $D$ additional tokens；keep the complete causal chain at each prediction depth——并明确与「which parallelly predicts $D$ additional tokens using independent output heads」（Gloeckle 等的并行多头）区分。
- 超参：The multi-token prediction depth $D$ is set to 1；The MTP loss weight $\lambda$ is set to 0.3。
- 推理期两条路：we can directly discard the MTP modules；或 repurpose these MTP modules for speculative decoding。

**V4 沿用（原文摘录，arXiv:2606.19348 §2.1）**：we adopt the same strategy for DeepSeek-V4 series **without modification**。（该句在 §2.1 中紧接 MTP 的表述；config 口径 `num_nextn_predict_layers=1` 与 V3 的 $D=1$ 一致。）

# 八、数字口径总表：V4-Flash config（2026-09-16 直抓）

来源：https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main/config.json（**原文摘录**，逐字段）。论文正文只给符号，**正文引用下列数字时必须写「config」**。

| 字段 | 值 | 本包哪一节要用 |
|---|---|---|
| `num_hidden_layers` | 43 | 整机层数（+1 MTP） |
| `num_nextn_predict_layers` | 1 | §七 MTP 层数（$D=1$） |
| `max_position_embeddings` | 1048576 | §一 1M 上下文 |
| `compress_ratios` | `[0, 0, 4, 128, 4, 128, …, 4, 128, 4, 0]`（44 项：前两项 0、末项 0，中段 4/128 严格交替） | §二 $m=4$ / §三 $m'=128$ / §四 ratio=0 纯滑窗层 |
| `sliding_window` | 128 | §四 $n_{\text{win}}$（论文无数字） |
| `index_topk` | 512 | §二 每 query 选中的压缩块数 $k$（V4-Pro 为 1024） |
| `index_n_heads` | 64 | §二 $n_h^I$ |
| `index_head_dim` | 128 | §二 $c^I$ |
| `head_dim` | 512 | §二 主注意力头维 $c$ |
| `hc_mult` | 4 | §五 残差流条数 $n_{hc}$（= HC 默认 $n=4$） |
| `hc_sinkhorn_iters` | 20 | §五 $t_{\text{max}}=20$（互证） |
| `hc_eps` | 1e-06 | §五 Sinkhorn 数值保护 |
| `n_routed_experts` / `n_shared_experts` / `num_experts_per_tok` | 256 / 1 / 6 | §六 256 挑 6 + 1 共享 |
| `num_hash_layers` | 3 | §六 最前三层 hash 路由 |
| `scoring_func` / `topk_method` | `sqrtsoftplus` / `noaux_tc` | §六 §2.1 口径 |
| `moe_intermediate_size` / `hidden_size` | 2048 / 4096 | §六 专家「切细」 |
| `architectures` / `model_type` | `DeepseekV4ForCausalLM` / `deepseek_v4` | 拉源码时的入口名 |
| `vocab_size` | 129280 | hash 表 $[\text{vocab}, \text{topk}]$ 的形状上限 |

# 九、覆盖度与缺口（诚实清单）

**逐字获取（原文摘录级）**：
- §一：V4 摘要（节选）、§2.3.4 两段——经 ch27 包转引 + 2026-09-16 abs 页复核。
- §二：V4 §2.3 引言、§2.3.1 全部（Eq.(9)-(19)）——转引自 ch27 包 `paper-v4.md`（2026-07-10 抓取，逐字）；V3.2 Eq.(1)(2) 同一来源的 ch27 包 `paper.md`。
- §三：V4 §2.3.2 的 Eq.(20)-(26)、首句、尾句、Figure 4 图注——2026-09-16 直抓。
- §四：V4 §2.3.3 的 5 条关键句 + RoPE/attention sink 两句——2026-09-16 直抓。
- §五：HC 摘要分句 + Eq.(1)(2)(3)(4)(8)(9) + 初始态式 + Eq.(14)（ar5iv 渲染抓取）；mHC 摘要逐字 + Eq.(3)(5)(6)(7)(8)(9) + §4.1 短句（官方 HTML 抓取）；V4 §2.2 的 Eq.(1)-(8) 公式行与符号定义短句（官方 HTML 抓取）。
- §六：DeepSeekMoE Eq.(3)-(17) 公式行 + 两条定义句（官方 HTML 与 ar5iv 双渲染互核）；Hash Layers 摘要与路由式、结论句（abs 页 + ar5iv）；V4 §2.1 五条短句。
- §七：V3 §2.2 的 Eq.(21)-(25) 与全部短句；V4「without modification」句。
- §八：config 全字段（直抓 JSON）。

**转述级（未逐字核；已在正文标「据转述」）**：
- V4 §2.2 中 A/B/C 的**角色描述**（"A_l 读入、B_l 残差混合、C_l 写回"）与「参数动态生成、门控小初始化、HC 堆叠不稳定」三句——摘要器转述，短引已给。
- V4 §2.3.2 的「$m'\gg m$」「压到 $1/m'$ 倍」「分组输出投影」周边表述。
- HC 论文的「seesaw」段与「adjust the strength of connections / dynamically rearrange layers」为摘要页分句引文；HC 正文的 Eq.(6)(7)（depth/width-connection 分解）与 Eq.(14)（初始化为 Pre-Norm 等价）为 ar5iv 摘录，**未与官方 PDF 逐字核对**。
- mHC §4.1 的三条性质除 $\|H^{res}\|_2 \le 1$ 有原短语外，其余两条为转述。

**未获取到原文（正文引用需另行核实或按缺口处理）**：
- V4 §2.2 Eq.(2) 的**尾部**（抓取截断在第三条件处；已按 mHC Eq.(6) 同构式补齐并标注）。
- V4 §2.2 Eq.(3)(4)(5) 的**左端符号**经二次抓取确认（$\tilde{A}_l /= \tilde{B}_l /= \tilde{C}_l =$）；但 W 矩阵的**维度再确认**仅一次抓取，如需精确请复核 PDF。
- mHC 论文 Eq.(4) 的**求和项**（抓取截断为 $\Sigma\ldots$）。
- V4 §2.3.2、§2.3.3 的**完整段落 prose**（摘要器只给关键句）；§2.3.1 与 §2.3.4 例外（ch27 包逐字摘录）。
- V4 §5.2.1 的 **MoE 权重 FP4→FP8 无损反量化论证**：本包只取「QK 路径 FP4、score 降至 BF16、top-k 2× 加速、99.7% 召回」一句（见 ch27 包 `paper-v4.md` §5.2.1 全文），MoE 权重那半段的完整推导未在本包复现。
- **V4-Pro config** 未在本批直抓（V4-Pro 数字 61 层 / 384 路由专家 / `index_topk=1024` 来自 ch29 研究文件的既有记录，2026-09-16 本批由队友抓取，confidence: medium；正文如要写 Pro 数字建议由 Lead 或 writer 复核一次）。

**版本提示**：V4 报告本批复核仍为 **v1（2026-04-26）唯一版本**；mHC 为 **v2（2026-01-05）**；HC 为 **v3（2025-03-18）**；Hash Layers 为 **v3（2021-07-20）**；V3 报告为 **v2**；DeepSeekMoE 官方 HTML 仅有 **v1**。凡「现在怎样」类表述，正文请锚定上述版本日期。

# 十、落点（vLLM 侧，七件套 → 源码路径）

> 本节的路径均在本书 pin 的 `vllm/` 源码树中逐一核实存在（2026-09-16，`instances/vllm/source/`）；**映射关系供 writer/插图定位，机制解释仍以前九节论文为准**（本节不新增论文断言）。

| 七件套 | 落点（vllm 路径） |
|---|---|
| ① 两本账 / 层排布 | `vllm/models/deepseek_v4/attention.py`（层型与 `compress_ratios` 消费）、`vllm/models/deepseek_v4/nvidia/model.py` |
| ② CSA（压缩 + 块级 top-k） | `vllm/models/deepseek_v4/compressor.py`、`vllm/models/deepseek_v4/sparse_mla.py`、`vllm/v1/attention/backends/mla/indexer.py`（索引器缓存/打分）、`vllm/v1/attention/backends/mla/compressor_utils.py` |
| ③ HCA（128:1 全看不挑） | `vllm/models/deepseek_v4/compressor.py`（同一压缩机机制）、`vllm/models/deepseek_v4/attention.py` |
| ④ 滑窗支路 | `vllm/v1/attention/backends/mla/sparse_swa.py`、`vllm/models/deepseek_v4/nvidia/flashmla.py`（滑窗与压缩块合进一次核调用） |
| ⑤ mHC + Sinkhorn | `vllm/model_executor/layers/mhc.py`（`MHCPreOp`/`MHCPostOp`/`HCHeadOp`/`MHCFusedPostPreOp`）、`vllm/model_executor/kernels/mhc/tilelang.py`（含 sinkhorn 核） |
| ⑥ MoE + hash 路由 | `vllm/models/deepseek_v4/nvidia/model.py`（gate 的 `tid2eid` 查表分支）、`vllm/model_executor/layers/fused_moe/routed_experts.py` |
| ⑦ MTP | `vllm/models/deepseek_v4/nvidia/mtp.py`、`vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py` |

**config 字段名（§八 表，会出现在插图/正文里的字形）**：`compress_ratios`、`sliding_window`、`index_topk`、`index_n_heads`、`index_head_dim`、`hc_mult`、`hc_sinkhorn_iters`、`hc_eps`、`n_routed_experts`、`n_shared_experts`、`num_experts_per_tok`、`num_hash_layers`、`moe_intermediate_size`、`scoring_func=sqrtsoftplus`、`topk_method=noaux_tc`。
