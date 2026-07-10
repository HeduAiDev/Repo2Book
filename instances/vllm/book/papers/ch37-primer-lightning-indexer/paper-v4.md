> **Source note**: Excerpt of arXiv:2606.19348, "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence" (DeepSeek-AI, 2026), fetched 2026-07-10 from https://arxiv.org/html/2606.19348v1. This is the supplementary source for this primer pack's IndexCache-layout and FP4-quantization coverage — the DSA/V3.2 report (arXiv:2512.02556, `paper.md` in this same directory) states only that the indexer "can be implemented in FP8" without giving the indexer's own cache tensor layout or an FP4 variant; DeepSeek-V4 promotes the same lightning-indexer mechanism into its Compressed Sparse Attention (CSA) design and is the paper that actually specifies the compressed-indexer-key construction, the mixed-precision storage scheme, and the FP4 quantization-aware training applied to the indexer's Q/K path — this is the mechanism `vllm/model_executor/layers/deepseek_v4_attention.py` and `vllm/v1/attention/backends/mla/indexer.py` implement (`use_fp4_indexer_cache`, MXFP4 packing). Kept verbatim: §2.3.1 "Compressed Sparse Attention" (paragraphs "Compressed Key-Value Entries" and "Lightning Indexer for Sparse Selection," Eq.(9)-(19), plus the "Shared Key-Value MQA" paragraph needed to close the pipeline); §2.3.4 "Efficiency Discussion" (in full); §5.2.1 "FP4 Quantization-Aware Training" (in full). Omitted for pack focus: §2.3.2 Heavily Compressed Attention (HCA, a sibling mechanism that does not use the indexer), §2.3.3 Other Details (RoPE/attention-sink tricks not specific to the indexer), §2.4 Muon optimizer, all of §3/§4/§5.2.2-5.2.5/§5.3 (training infra and benchmark tables unrelated to the indexer) — see full arXiv source for these.
---

# DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence

DeepSeek-AI · research@deepseek.com

## Abstract (excerpt)

We present a preview version of DeepSeek-V4 series, including two strong Mixture-of-Experts (MoE) language models — DeepSeek-V4-Pro with 1.6T parameters (49B activated) and DeepSeek-V4-Flash with 284B parameters (13B activated) — both supporting a context length of one million tokens. DeepSeek-V4 series incorporate several key upgrades in architecture and optimization: (1) a hybrid attention architecture that combines Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA) to improve long-context efficiency; (2) Manifold-Constrained Hyper-Connections (mHC) that enhance conventional residual connections; (3) and the Muon optimizer for faster convergence and greater training stability. [...] In the one-million-token context setting, DeepSeek-V4-Pro requires only 27% of single-token inference FLOPs and 10% of KV cache compared with DeepSeek-V3.2.

**Figure 1** (arXiv:2606.19348): Left: benchmark performance of DeepSeek-V4-Pro-Max and its counterparts. Right: inference FLOPs and KV cache size of DeepSeek-V4 series and DeepSeek-V3.2. — 见本包 `meta.json` key_figures 第三项（右半张：FLOPs/KV cache 的复杂度对账柱状图）。

## 2.3 Hybrid Attention with CSA and HCA

As the context length reaches extreme scales, the attention mechanism emerges as the dominant computational bottleneck in a model. For DeepSeek-V4, we design two efficient attention architectures — Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA) — and employ their interleaved hybrid configuration, which substantially reduces the computational cost of attention in long-text scenarios. CSA integrates both compression and sparse attention strategies: it first compresses the Key-Value (KV) cache of every $m$ tokens into one entry, and then applies DeepSeek Sparse Attention (DSA) [dsv32] where each query token attends to only $k$ compressed KV entries. HCA aims for extreme compression by consolidating the KV cache of every $m'$ ($\gg m$) tokens into a single entry.

**Figure 3** (arXiv:2606.19348): Core architectures of CSA. It compresses the number of KV entries to $\frac{1}{m}$ times, and then applies DeepSeek Sparse Attention for further acceleration. Additionally, a small set of sliding window KV entries is combined with the selected compressed KV entries to enhance local fine-grained dependencies. — 见本包 `meta.json` key_figures 第二项。

### 2.3.1 Compressed Sparse Attention

The core architecture of CSA is illustrated in Figure 3, which first compresses the KV cache of each $m$ tokens into one entry, and then applies DeepSeek Sparse Attention for further acceleration.

**Compressed Key-Value Entries.**

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

**Lightning Indexer for Sparse Selection.**

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

**Shared Key-Value MQA.**

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

where $\mathbf{o}_{t,i} \in \mathbb{R}^c$ is the core attention output of the $i$-th head at the $t$-th token; $\mathrm{CoreAttn}(\cdot)$ denotes the core attention operation.

**IndexCache 要点（对应本段原文）**：indexer 自己维护的缓存不是主 KV cache 的旁支，而是一份形状独立的张量——$K^{\mathrm{IComp}} \in \mathbb{R}^{n/m \times c^I}$，行数与主压缩 KV cache $C^{\mathrm{Comp}}$ 同步增长（都以 $n/m$ 个压缩块为单位），但列宽是 indexer 专属的头维度 $c^I$，与主注意力头维度 $c$ 无关、通常远小于 $c$。这正是"独立小头"在缓存层面的体现：indexer 不复用主 KV cache 的存储，也不需要主 KV cache 的精度/维度约束。

### 2.3.4 Efficiency Discussion

Due to the employment of hybrid CSA and HCA, together with low-precision computation and storage, the attention module of DeepSeek-V4 series achieves remarkable efficiency in both attention FLOPs and KV cache size, especially in long-context scenarios. First, we adopt a mixed storage format for KV entries: BF16 precision is used for the rotary positional embedding (RoPE) dimensions, while FP8 precision is applied to the remaining dimensions. This hybrid representation reduces the KV cache size by nearly half compared with pure BF16 storage. **Second, attention computation within the lightning indexer is performed in FP4 precision, which accelerates the attention operation under extremely long contexts.** Third, relative to DeepSeek-V3.2, a smaller attention top-k is chosen in DeepSeek-V4 series, thereby improving model efficiency on short- and medium-length texts. Finally, and most importantly, compressed attention and hybrid attention techniques substantially reduce both the KV cache size and the computational FLOPs.

Taking BF16 GQA8 [ainslie2023gqa] with a head dimension of 128 as the baseline — one of the common configurations of LLM attention — the KV cache size of DeepSeek-V4 series can be dramatically reduced to approximately 2% times of that baseline in the 1M-context setting. Moreover, even when compared with DeepSeek-V3.2 [dsv32] — already an efficient baseline — DeepSeek-V4 series still exhibits substantial advantages in efficiency. A comparison of their inference FLOPs and KV cache size is provided in the right part of Figure 1.

## 5.2.1 FP4 Quantization-Aware Training

To achieve inference acceleration and reducing memory traffic at deployment, we introduce Quantization-Aware Training (QAT) [QAT] during the post-training stage, enabling the model, including those of teacher and reference models, to adapt to the precision degradation introduced by quantization. **We apply FP4 (MXFP4) quantization [OCP_MXFormat] to two components: (1) MoE expert weights, which are a major source of GPU memory occupancy [gpt_oss], and (2) the Query-Key (QK) path in the indexer of CSA, where QK activations are cached, loaded, and multiplied entirely in FP4, accelerating attention score computation in long-context scenarios.** In addition, we further quantize the index scores $I_{:,:}$ from FP32 to BF16 during this QAT process. **This optimization achieves a 2× speedup for the top-k selector, while preserving a 99.7% recall rate of KV entries.**

For MoE expert weights, following the common practice of QAT, the FP32 master weights maintained by the optimizer are first quantized to FP4, then dequantized back to FP8 for computation. Notably, our FP4-to-FP8 dequantization is lossless. This is because FP8 (E4M3) has 2 additional exponent bits compared with FP4 (E2M1), offering a larger dynamic range. Consequently, as long as the ratio between the maximum and minimum scale factors of the FP4 sub-blocks ($1 \times 32$ tiles) within each FP8 quantization block ($128 \times 128$ tiles) does not exceed a certain threshold, the fine-grained scale information can be fully absorbed by the extended dynamic range of FP8. We empirically verify that current weights satisfy this condition. This allows the entire QAT pipeline to fully reuse the existing FP8 training framework without any modification. In the backward pass, gradients are computed with respect to the same FP8 weights in the forward pass and directly propagated back to the FP32 master weights, equivalent to applying the Straight-Through Estimator (STE) through the quantization operation. This also avoids the need to re-quantize transposed weights.

During the inference and rollout phases of RL training, which do not involve backward passes, we directly use native FP4 quantized weights instead of simulated quantization. This ensures that model behavior during sampling is fully consistent with online deployment, while also reducing kernel memory loading for actual speedup and significantly lowering memory consumption. **We process the QK path in the indexer of CSA similarly.**

**为何要 FP4 变体（对应本段原文的落地推理）**：indexer 的复杂度仍是 $O(L^2)$（见 `paper.md` §四），意味着它的打分开销随上下文长度平方增长——当上下文推到百万 token 级别时，即便 indexer 本身"轻量"，$O(L^2)$ 这一项也会重新变贵。FP4/MXFP4 把 indexer 的 QK 路径（含它自己的 IndexCache 读取）压到比 FP8 更低的位宽，用 QAT 保证精度不塌（0.997 召回率），换来 top-k 选择器 2 倍加速——这是"indexer 廉价"这个假设在百万 token 场景下继续成立的具体工程手段，而不是通用的模型量化操作。
