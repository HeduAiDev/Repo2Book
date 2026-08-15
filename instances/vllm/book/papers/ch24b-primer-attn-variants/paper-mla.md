# MLA 论文摘编:DeepSeek-V2 §2.1 Multi-head Latent Attention(+ 附录 B/C/D)+ DeepSeek-V3 §2.1.1 重述注记

> 主出处:DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model", arXiv:2405.04434v3,抓取于 2026-08-16(arxiv.org/html)。
> 附注出处:DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv:2412.19437v1,抓取于 2026-08-16(arxiv.org/html)。
> 裁剪:只保留注意力变体数学(MHA/GQA/MQA 基线、MLA 全部公式、KV cache 对比、消融结论);MoE、训练、对齐、评测部分全部裁掉。
> 数学清理说明:arXiv HTML 的 MathML 渲染在抓取中产生「Unicode 数学 + LaTeX 源码」双重伪影,以下公式均从抓取文本中的 LaTeX 源逐条恢复,式号与原文一一对应(1–19、37–47);prose 为逐句清理伪影后的原文。Table 1/8/9 的表体在抓取中丢失,已按 caption+prose 原句重建并明标。

---

# DeepSeek-V2(注意力部分)

###### Abstract(仅 MLA/KV cache 相关句)

MLA guarantees efficient inference through significantly compressing the KV cache into a latent vector. Compared with DeepSeek 67B, DeepSeek-V2 achieves significantly stronger performance, and meanwhile saves 42.5% of training costs, reduces the KV cache by 93.3%, and boosts the maximum generation throughput to 5.76 times.

## 1 Introduction(MLA 相关段落摘录)

In the context of attention mechanisms, the Key-Value (KV) cache of the Multi-Head Attention (MHA) poses a significant obstacle to achieving high inference efficiency. Various approaches have been explored to reduce the KV cache, including Grouped-Query Attention (GQA) (Ainslie et al., 2023) and Multi-Query Attention (MQA) (Shazeer, 2019). However, these methods often compromise performance in their attempt to reduce the KV cache. In order to achieve the best of both worlds, we introduce MLA, an attention mechanism equipped with low-rank key-value joint compression. Empirically, MLA achieves superior performance compared with MHA, and meanwhile significantly reduces the KV cache during inference, thus boosting the inference efficiency.

## 2.1 Multi-Head Latent Attention: Boosting Inference Efficiency

Conventional Transformer models usually adopt Multi-Head Attention (MHA) for sequence modeling, but during generation, its heavy Key-Value (KV) cache will become the bottleneck that limits the inference efficiency. In order to reduce the KV cache, Multi-Query Attention (MQA) (Shazeer, 2019) and Grouped-Query Attention (GQA) (Ainslie et al., 2023) are proposed. They require a smaller magnitude of KV cache, but their performance does not match MHA (we provide the ablation of MHA, GQA and MQA in Appendix D.1). For DeepSeek-V2, we design an innovative attention mechanism called Multi-head Latent Attention (MLA). Equipped with low-rank key-value joint compression, MLA achieves better performance than MHA, but requires a significantly smaller amount of KV cache.

### 2.1.1 Preliminaries: Standard Multi-Head Attention

The standard MHA mechanism is formulated as follows: first, the hidden state of each token is projected onto query, key, and value respectively through three matrices:

\[\mathbf{q}_{t} = W^{Q}\mathbf{h}_{t} \tag{1}\]

\[\mathbf{k}_{t} = W^{K}\mathbf{h}_{t} \tag{2}\]

\[\mathbf{v}_{t} = W^{V}\mathbf{h}_{t} \tag{3}\]

where \(\mathbf{h}_{t}\in\mathbb{R}^{d}\) denotes the hidden state of the t-th token; \(W^{Q},W^{K}\in\mathbb{R}^{d_{h}n_{h}\times d}\) are the projection matrices for queries and keys; and \(W^{V}\in\mathbb{R}^{d_{h}n_{h}\times d}\) is the projection matrix for values. \(n_{h}\) and \(d_{h}\) respectively denote the number of attention heads and the dimension per attention head. Then, the queries, keys, and values will be split into \(n_{h}\) heads to compute the attention score respectively:

\[[\mathbf{q}_{t,1};\ldots;\mathbf{q}_{t,n_{h}}]=\mathbf{q}_{t}, \tag{4}\]

\[[\mathbf{k}_{t,1};\ldots;\mathbf{k}_{t,n_{h}}]=\mathbf{k}_{t}, \tag{5}\]

\[[\mathbf{v}_{t,1};\ldots;\mathbf{v}_{t,n_{h}}]=\mathbf{v}_{t}, \tag{6}\]

\[\mathbf{o}_{t,i}=\sum_{j=1}^{t}\operatorname{Softmax}_{j}\left(\frac{\mathbf{q}_{t,i}^{T}\mathbf{k}_{j,i}}{\sqrt{d_{h}}}\right)\mathbf{v}_{j,i}, \tag{7}\]

\[\mathbf{u}_{t}=W^{O}[\mathbf{o}_{t,1};\ldots;\mathbf{o}_{t,n_{h}}], \tag{8}\]

where \(\mathbf{q}_{t,i},\mathbf{k}_{t,i},\mathbf{v}_{t,i}\in\mathbb{R}^{d_{h}}\) denote the query, key, and value of the i-th attention head, respectively; \(W^{O}\in\mathbb{R}^{d\times d_{h}n_{h}}\) denotes the output projection matrix. During inference, all keys and values need to be cached to accelerate inference, so MHA needs to cache \(2n_{h}d_{h}l\) elements for each token. In model deployment, this heavy KV cache is a large bottleneck that limits the maximum batch size and sequence length.

[注:式号与 arXiv v3 原文 §2.1.1 严格一致——(1)-(3) 投影、(4)(5)(6) 三条独立编号的逐头切分等式、(7) 逐头注意力得分、(8) 输出投影。]

### 2.1.2 Low-Rank Key-Value Joint Compression

The core of MLA is the low-rank joint compression for keys and values to reduce KV cache:

\[\mathbf{c}_{t}^{KV}=W^{DKV}\mathbf{h}_{t}, \tag{9}\]

\[\mathbf{k}_{t}^{C}=W^{UK}\mathbf{c}_{t}^{KV}, \tag{10}\]

\[\mathbf{v}_{t}^{C}=W^{UV}\mathbf{c}_{t}^{KV}, \tag{11}\]

where \(\mathbf{c}_{t}^{KV}\in\mathbb{R}^{d_{c}}\) is the compressed latent vector for keys and values; \(d_{c}(\ll d_{h}n_{h})\) denotes the KV compression dimension; \(W^{DKV}\in\mathbb{R}^{d_{c}\times d}\) is the down-projection matrix; and \(W^{UK},W^{UV}\in\mathbb{R}^{d_{h}n_{h}\times d_{c}}\) are the up-projection matrices for keys and values, respectively. During inference, MLA only needs to cache \(\mathbf{c}_{t}^{KV}\), so its KV cache has only \(d_{c}l\) elements, where \(l\) denotes the number of layers. In addition, during inference, since \(W^{UK}\) can be absorbed into \(W^{Q}\), and \(W^{UV}\) can be absorbed into \(W^{O}\), we even do not need to compute keys and values out for attention. Figure 3 intuitively illustrates how the KV joint compression in MLA reduces the KV cache.

Moreover, in order to reduce the activation memory during training, we also perform low-rank compression for the queries, even if it cannot reduce the KV cache:

\[\mathbf{c}_{t}^{Q}=W^{DQ}\mathbf{h}_{t}, \tag{12}\]

\[\mathbf{q}_{t}^{C}=W^{UQ}\mathbf{c}_{t}^{Q}, \tag{13}\]

where \(\mathbf{c}_{t}^{Q}\in\mathbb{R}^{d_{c}^{\prime}}\) is the compressed latent vector for queries; \(d_{c}^{\prime}(\ll d_{h}n_{h})\) denotes the query compression dimension; \(W^{DQ}\in\mathbb{R}^{d_{c}^{\prime}\times d}\) is the down-projection matrix for queries; and \(W^{UQ}\in\mathbb{R}^{d_{h}n_{h}\times d_{c}^{\prime}}\) is the up-projection matrix for queries.

### 2.1.3 Decoupled Rotary Position Embedding

Following DeepSeek 67B, we intend to use the Rotary Position Embedding (RoPE) for DeepSeek-V2. However, RoPE is incompatible with low-rank KV compression. To be specific, RoPE is position-sensitive for both keys and queries. If we apply RoPE for the keys \(\mathbf{k}_{t}^{C}\), \(W^{UK}\) in Equation 10 will be coupled with a position-sensitive RoPE matrix. In this way, \(W^{UK}\) cannot be absorbed into \(W^{Q}\) any more during inference, since a RoPE matrix related to the currently generating token will lie between \(W^{Q}\) and \(W^{UK}\) and matrix multiplication does not obey a commutative law. As a result, we must recompute the keys for all the prefix tokens during inference, which will significantly hinder the inference efficiency.

As a solution, we propose the decoupled RoPE strategy that uses additional multi-head queries \(\mathbf{q}_{t,i}^{R}\in\mathbb{R}^{d_{h}^{R}}\) and a shared key \(\mathbf{k}_{t}^{R}\in\mathbb{R}^{d_{h}^{R}}\) to carry RoPE, where \(d_{h}^{R}\) denotes the per-head dimension of the decoupled queries and key. Equipped with the decoupled RoPE strategy, MLA performs the following computation:

\[[\mathbf{q}_{t,1}^{R};\ldots;\mathbf{q}_{t,n_{h}}^{R}]=\mathbf{q}_{t}^{R}=\operatorname{RoPE}(W^{QR}\mathbf{c}_{t}^{Q}), \tag{14}\]

\[\mathbf{k}_{t}^{R}=\operatorname{RoPE}(W^{KR}\mathbf{h}_{t}), \tag{15}\]

\[\mathbf{q}_{t,i}=[\mathbf{q}_{t,i}^{C};\mathbf{q}_{t,i}^{R}], \tag{16}\]

\[\mathbf{k}_{t,i}=[\mathbf{k}_{t,i}^{C};\mathbf{k}_{t}^{R}], \tag{17}\]

\[\mathbf{o}_{t,i}=\sum_{j=1}^{t}\operatorname{Softmax}_{j}\left(\frac{\mathbf{q}_{t,i}^{T}\mathbf{k}_{j,i}}{\sqrt{d_{h}+d_{h}^{R}}}\right)\mathbf{v}_{j,i}^{C}, \tag{18}\]

\[\mathbf{u}_{t}=W^{O}[\mathbf{o}_{t,1};\ldots;\mathbf{o}_{t,n_{h}}], \tag{19}\]

where \(W^{QR}\in\mathbb{R}^{d_{h}^{R}n_{h}\times d_{c}^{\prime}}\) and \(W^{KR}\in\mathbb{R}^{d_{h}^{R}\times d}\) are matrices to produce the decoupled queries and key, respectively; \(\operatorname{RoPE}(\cdot)\) denotes the operation that applies RoPE matrices; and \([\cdot;\cdot]\) denotes the concatenation operation. During inference, the decoupled key should also be cached. Therefore, DeepSeek-V2 requires a total KV cache containing \((d_{c}+d_{h}^{R})l\) elements.

### 2.1.4 Comparison of Key-Value Cache

We demonstrate a comparison of the KV cache per token among different attention mechanisms in Table 1. MLA requires only a small amount of KV cache, equal to GQA with only 2.25 groups, but can achieve stronger performance than MHA.

Table 1: Comparison of the KV cache per token among different attention mechanisms. \(n_{h}\) denotes the number of attention heads, \(d_{h}\) denotes the dimension per attention head, \(l\) denotes the number of layers, \(n_{g}\) denotes the number of groups in GQA, and \(d_{c}\) and \(d_{h}^{R}\) denote the KV compression dimension and the per-head dimension of the decoupled queries and key in MLA, respectively. The amount of KV cache is measured by the number of elements, regardless of the storage precision. For DeepSeek-V2, \(d_{c}\) is set to \(4d_{h}\) and \(d_{h}^{R}\) is set to \(d_{h}/2\). So, its KV cache is equal to GQA with only 2.25 groups, but its performance is stronger than MHA.

| Attention | KV Cache (per token) |
|---|---|
| MHA | \(2n_{h}d_{h}l\) |
| MQA | \(2d_{h}l\) |
| GQA | \(2n_{g}d_{h}l\) |
| MLA | \((d_{c}+d_{h}^{R})l\) |

[注:表体为重建——ar5iv/arXiv HTML 抓取只保留了 caption;MHA 行 \(2n_{h}d_{h}l\) 出自 §2.1.1 原文「MHA needs to cache \(2n_{h}d_{h}l\) elements for each token」,MLA 行 \((d_{c}+d_{h}^{R})l\) 出自 §2.1.3 原文,GQA/MQA 行为 n_g/n=1 个 KV 头的直推(与 GQA 论文「reducing the key-value cache by a factor of H」一致)。「2.25 groups」=(d_c+d_h^R)/(2d_h)=(4d_h+d_h/2)/(2d_h)=2.25,由 caption 原句给出。]

![Figure 3: Illustration of (a) MHA, (b) GQA, (c) MQA, and (d) MLA. For MLA, only the blue-boxed vectors (i.e., the compressed latent vector \(\mathbf{c}_{t}^{KV}\) and the decoupled key \(\mathbf{k}_{t}^{R}\) carrying RoPE) need to be cached during generation. Through the low-rank joint compression, MLA significantly reduces the KV cache into a latent vector.](https://arxiv.org/html/2405.04434v3/x4.png)

## MLA 超参数(摘自 §3.1.2 Model Hyper-Parameters)

DeepSeek-V2 in total contains 236B parameters, of which 21B are activated for each token. We set the number of Transformer layers to 60 and the hidden dimension to 5120. In MLA, we set the number of attention heads \(n_{h}\) to 128 and the per-head dimension \(d_{h}\) to 128. The KV compression dimension \(d_{c}\) is set to 512, and the query compression dimension \(d_{c}^{\prime}\) is set to 1536. For the decoupled queries and key, we set the per-head dimension \(d_{h}^{R}\) to 64.

Due to the co-impact of the MoE and MLA architectures, the output scale of each layer exhibits a greater magnitude than that of a standard Transformer. Considering the potential convergence risk imposed by large propagation scale, in practice, we employ additional RMS Norm layers after the compressed latent vectors, and multiply additional scaling factors at the width bottlenecks (i.e., the compressed latent vectors and the intermediate hidden states of routed experts) to ensure stable training.

## §3.1.4 Long Context Extension(仅 RoPE 相关句)

We set the RoPE base frequency \(\theta\) to 10000 for pre-training. YaRN (Peng et al., 2023) is a context extension method based on NTK-aware interpolation and attention entropy. [...] YaRN was specifically applied to the decoupled shared key \(\mathbf{k}_{t}^{R}\) as it is responsible for carrying RoPE.

## §3.2.3 Inference Efficiency(MLA 相关)

In order to efficiently deploy DeepSeek-V2 for service, we first convert its parameters into the precision of FP8. In addition, we also perform KV cache quantization for DeepSeek-V2 to further compress each element in its KV cache into 6 bits on average. Benefiting from MLA and these optimizations, actually deployed DeepSeek-V2 requires significantly less KV cache than DeepSeek 67B, and thus can serve a much larger batch size. We evaluate the generation throughput under effective input and output lengths of 4K. [...] On a single node with 8 H800 GPUs, DeepSeek-V2 achieves a generation throughput exceeding 50K tokens per second, which is 5.76 times the maximum generation throughput of DeepSeek 67B.

## Appendix B.1 DeepSeek-V2-Lite(MLA 配置差异)

DeepSeek-V2-Lite has 27 layers and a hidden dimension of 2048. It also employs MLA and has 16 attention heads, where each head has a dimension of 128. Its KV compression dimension \(d_{c}\) is 512, but slightly different from DeepSeek-V2, it does not compress the queries. For the decoupled queries and key, it has a per-head dimension \(d_{h}^{R}\) of 64. For other configurations, DeepSeek-V2-Lite is exactly the same as DeepSeek-V2.

## Appendix C Full Formulas of MLA

In order to demonstrate the complete computation process of MLA, we provide its full formulas in the following:

\[\mathbf{c}_{t}^{Q}=W^{DQ}\mathbf{h}_{t}, \tag{37}\]

\[\mathbf{c}_{t}^{KV}=W^{DKV}\mathbf{h}_{t}, \tag{38}\]

\[[\mathbf{q}_{t,1}^{R};\ldots;\mathbf{q}_{t,n_{h}}^{R}]=\mathbf{q}_{t}^{R}=\operatorname{RoPE}(W^{QR}\mathbf{c}_{t}^{Q}), \tag{39}\]

\[\mathbf{k}_{t}^{R}=\operatorname{RoPE}(W^{KR}\mathbf{h}_{t}), \tag{40}\]

\[\boxed{\mathbf{c}_{t}^{KV}},\quad\boxed{\mathbf{k}_{t}^{R}} \tag{41}\]

\[\mathbf{q}_{t}^{C}=W^{UQ}\mathbf{c}_{t}^{Q}, \tag{42}\]

\[[\mathbf{k}_{t,1}^{C};\ldots;\mathbf{k}_{t,n_{h}}^{C}]=\mathbf{k}_{t}^{C}=W^{UK}\boxed{\mathbf{c}_{t}^{KV}}, \tag{43}\]

\[[\mathbf{v}_{t,1}^{C};\ldots;\mathbf{v}_{t,n_{h}}^{C}]=\mathbf{v}_{t}^{C}=W^{UV}\boxed{\mathbf{c}_{t}^{KV}}, \tag{44}\]

\[\mathbf{q}_{t,i}=[\mathbf{q}_{t,i}^{C};\mathbf{q}_{t,i}^{R}], \tag{45}\]

\[\mathbf{k}_{t,i}=[\mathbf{k}_{t,i}^{C};\mathbf{k}_{t}^{R}], \tag{46}\]

\[\mathbf{o}_{t,i}=\sum_{j=1}^{t}\operatorname{Softmax}_{j}\left(\frac{\mathbf{q}_{t,i}^{T}\mathbf{k}_{j,i}}{\sqrt{d_{h}+d_{h}^{R}}}\right)\mathbf{v}_{j,i}^{C},\quad \mathbf{u}_{t}=W^{O}[\mathbf{o}_{t,1};\ldots;\mathbf{o}_{t,n_{h}}] \tag{47}\]

[注:式 (41) 在原文中不是一个独立等式,而是把 (38)/(40) 的两个「需缓存向量」\(\mathbf{c}_{t}^{KV}\) 与 \(\mathbf{k}_{t}^{R}\) 用蓝框单独标出(原文 boxed in blue);此处以 boxed 保留该语义。(43)/(44) 中的 boxed 表示 k/v 上投影的输入即被缓存的 \(\mathbf{c}_{t}^{KV}\)。]

During inference, the naive formula needs to recover \(\mathbf{k}_{t}^{C}\) and \(\mathbf{v}_{t}^{C}\) from \(\mathbf{c}_{t}^{KV}\) for attention. Fortunately, due to the associative law of matrix multiplication, we can absorb \(W^{UK}\) into \(W^{UQ}\), and \(W^{UV}\) into \(W^{O}\). Since this optimization is related to only model parameters, it can be completed offline at once. Through this optimization, we avoid the computational overhead for recomputing \(\mathbf{k}_{t}^{C}\) and \(\mathbf{v}_{t}^{C}\) during inference.

## Appendix D Ablations(注意力相关结论,prose)

### D.1 Ablation of Attention Mechanisms

We show the evaluation results for 7B dense models with MHA, GQA, and MQA on four hard benchmarks in Table 8. All of these three models are trained on 1.33T tokens, and share the same architecture except for the attention mechanisms. In addition, for a fair comparison, we align the number of parameters of them to around 7B by adjusting the number of layers. From the table, we can find that MHA demonstrates significant advantages over GQA and MQA on these benchmarks.

### D.2 Ablation of MLA

In Table 9, we show the evaluation results for MoE models equipped with MLA and MHA. Two small MoE models comprise about 16B total parameters, where 2.4B are activated for each token, and we train them on 1.33T tokens. Two large MoE models comprise about 250B total parameters, where 21B are activated for each token, and we train them on 420B tokens. All of these models are trained on the same pre-training corpus as DeepSeek-V2. From the table, we can observe that MLA shows better performance than MHA. More importantly, MLA requires a significantly smaller amount of KV cache (14% for small MoE models and 4% for large MoE models) than MHA.

[注:Table 8/9 表体在抓取中丢失,如需逐格数值请查原文 PDF;上述 prose 结论句均为原文逐句。]

---

# 附注:DeepSeek-V3 Technical Report 的 MLA(§2.1.1 重述 + §4.2 超参)

V3 报告将 MLA 压缩为一节重述,数学与 V2 完全一致(仅记号排布略异),并把「需要缓存什么」用蓝框直接写进公式——这一版重述对「KV cache 装的是什么」比 V2 原文更醒目:

The core of MLA is the low-rank joint compression for attention keys and values to reduce Key-Value (KV) cache during inference:

\[\boxed{\mathbf{c}_{t}^{KV}=W^{DKV}\mathbf{h}_{t}} \tag{V3-1}\]

\[[\mathbf{k}_{t,1}^{C};\ldots;\mathbf{k}_{t,n_{h}}^{C}]=\mathbf{k}_{t}^{C}=W^{UK}\mathbf{c}_{t}^{KV} \tag{V3-2}\]

\[\boxed{\mathbf{k}_{t}^{R}=\operatorname{RoPE}(W^{KR}\mathbf{h}_{t})} \tag{V3-3}\]

\[\mathbf{k}_{t,i}=[\mathbf{k}_{t,i}^{C};\mathbf{k}_{t}^{R}] \tag{V3-4}\]

\[[\mathbf{v}_{t,1}^{C};\ldots;\mathbf{v}_{t,n_{h}}^{C}]=\mathbf{v}_{t}^{C}=W^{UV}\mathbf{c}_{t}^{KV} \tag{V3-5}\]

\[\mathbf{c}_{t}^{Q}=W^{DQ}\mathbf{h}_{t} \tag{V3-6}\]

\[[\mathbf{q}_{t,1}^{C};\ldots;\mathbf{q}_{t,n_{h}}^{C}]=\mathbf{q}_{t}^{C}=W^{UQ}\mathbf{c}_{t}^{Q} \tag{V3-7}\]

\[[\mathbf{q}_{t,1}^{R};\ldots;\mathbf{q}_{t,n_{h}}^{R}]=\mathbf{q}_{t}^{R}=\operatorname{RoPE}(W^{QR}\mathbf{c}_{t}^{Q}) \tag{V3-8}\]

\[\mathbf{q}_{t,i}=[\mathbf{q}_{t,i}^{C};\mathbf{q}_{t,i}^{R}] \tag{V3-9}\]

\[\mathbf{o}_{t,i}=\sum_{j=1}^{t}\operatorname{Softmax}_{j}\left(\frac{\mathbf{q}_{t,i}^{T}\mathbf{k}_{j,i}}{\sqrt{d_{h}+d_{h}^{R}}}\right)\mathbf{v}_{j,i}^{C}, \tag{V3-10}\]

\[\mathbf{u}_{t}=W^{O}[\mathbf{o}_{t,1};\ldots;\mathbf{o}_{t,n_{h}}] \tag{V3-11}\]

where \(\mathbf{h}_{t}\in\mathbb{R}^{d}\) denotes the input hidden state of the t-th token, \(\mathbf{c}_{t}^{KV}\) and \(\mathbf{c}_{t}^{Q}\) respectively denote the compressed latent vectors for keys-values and queries, \(\operatorname{RoPE}(\cdot)\) denotes the application of RoPE matrices, and \(W^{(\cdot)}\) are projection matrices. Note that for MLA, only the blue-boxed vectors (i.e., \(\mathbf{c}_{t}^{KV}\) and \(\mathbf{k}_{t}^{R}\)) need to be cached during generation, which results in significantly reduced KV cache while maintaining performance comparable to standard Multi-Head Attention (MHA).

![Figure 2: The basic architecture of DeepSeek-V3. MLA (Multi-head Latent Attention) and DeepSeekMoE comprise each Transformer layer of the model.](https://arxiv.org/html/2412.19437v1/assets/basic_arch.png)

### V3 超参(§4.2,MLA 相关句)

In terms for the attention mechanism, DeepSeek-V3 adopts the MLA architecture. [...] In MLA, we set the number of attention heads \(n_{h}\) to 128 and the per-head dimension \(d_{h}\) to 128. The KV compression dimension \(d_{c}\) is set to 512, and the query compression dimension \(d_{c}^{\prime}\) is set to 1536. For the decoupled queries and key, we set the per-head dimension \(d_{h}^{R}\) to 64. [...] We set the number of layers to 61 and the hidden dimension to 7168. [...] As DeepSeek-V2, DeepSeek-V3 also employs additional RMSNorm layers after the compressed latent vectors, and multiplies additional scaling factors at the width bottlenecks.

### V3 相对 V2 的注意力侧变化(核读)

- **MLA 数学与超参完全沿用 V2**:\(n_{h}=128,d_{h}=128,d_{c}=512,d_{c}^{\prime}=1536,d_{h}^{R}=64\)。V3 报告 §2.1.1 的 MLA 一节是重述而非改版;报告全文对 MLA 的表述为 "DeepSeek-V3 adopts the MLA architecture, which allows for more efficient inference"(无任何结构性改动的声明)。
- V3 报告「Compared with DeepSeek-V2, an exception is that we additionally introduce an auxiliary-loss-free load balancing strategy for DeepSeekMoE」——即**架构层显式声明的变化都在 MoE 侧**(辅助损失无关的负载均衡、无 token 丢弃策略),注意力侧无变化声明。
- [附]V3 §3.2.3 训练侧提到 "Recomputation of RMSNorm and MLA Up-Projection: We recompute all RMSNorm operations and MLA up-projections during back-propagation to reduce memory consumption during training"——训练期显存技巧,不改变推理侧 MLA 数学,与本章正题无关,备引。

---

# 研究注记(researcher 注,非论文原文;正文引用时不得作为论文出处)

1. **「576 维潜向量」的算术出处**:V2/V3 的 MLA 每 token 每层需缓存的元素数 = \(d_{c}+d_{h}^{R}=512+64=576\)(V2 §2.1.3 原文句 + §3.1.2 超参;V3 §4.2 同值)。这 576 = 512 维 KV 压缩潜向量 \(\mathbf{c}_{t}^{KV}\) + 64 维共享解耦 key \(\mathbf{k}_{t}^{R}\)(后者是所有头共享的、仅承载 RoPE 的分量,不分头、不压入潜向量)。对比 V2/V3 自己的 MHA 配置(\(n_{h}=128,d_{h}=128\)):每 token 每层 \(2n_{h}d_{h}=32768\) 元素,MLA 为其 \(576/32768\approx1.75\%\)。注意:abstract 里「reduces the KV cache by 93.3%」是相对 DeepSeek 67B(不同头配置)的口径,别与 1.75% 混用;附录 D.2 的「14%/4%」是 MoE 模型整体(含多因素)的实测口径。
2. **「展开 vs 吸收」两种形态的论文侧依据**(工程侧的 prefill-MHA/decode-MQA 分野属 vLLM 实现叙事,ch25 讲,论文只给了这三块数学事实):
   - **naive/展开形态**:附录 C 原句 "During inference, the naive formula needs to recover \(\mathbf{k}_{t}^{C}\) and \(\mathbf{v}_{t}^{C}\) from \(\mathbf{c}_{t}^{KV}\) for attention"——即用 \(W^{UK}/W^{UV}\) 把潜向量展开回每头 K/V(MHA 形状),预填充批量算划算。
   - **吸收形态**:附录 C 原句 "we can absorb \(W^{UK}\) into \(W^{UQ}\), and \(W^{UV}\) into \(W^{O}\). Since this optimization is related to only model parameters, it can be completed offline at once"——权重侧预乘成 \(\widehat{W}^{UKQ}=(W^{UQ})^{T}W^{UK}\) 型矩阵,推理时直接拿 \(\mathbf{c}_{t}^{KV}\) 当「单头 KV」参与 attention(MQA 形状),解码期免去逐 token 展开。
   - **为什么两者并存(吸收不能全用)**:§2.1.3 原文解释了 RoPE 与低秩压缩的不兼容——RoPE 矩阵是位置敏感的、夹在 \(W^{Q}\) 与 \(W^{UK}\) 之间,结合律被破坏,所以位置分量必须解耦成共享的 \(\mathbf{k}_{t}^{R}\)(不参与吸收)。
   - **术语口径提醒**:V2 正文 §2.1.2 说吸收进 \(W^{Q}\),附录 C 说吸收进 \(W^{UQ}\)——同一件事的简写与精确写(带查询压缩时 query 路径为 \(W^{UQ}W^{DQ}\)),写作时统一用附录 C 的精确口径。
3. **表体重建声明**:V2 Table 1/8/9 与 GQA Table 1 的表体在 arXiv HTML 抓取中丢失(只剩 caption 与 prose),本文档内的表格/数值凡标「重建」均由 prose 原句直推;未标注处均为抓取原文。逐格数值如需引用,请对 PDF 核。
4. **V4 未覆盖**:本包止于 V3 技术报告。v3 书 pin 的是 DeepSeek-V4 系(见 INSTANCE),V4 若有 MLA 结构变化,以 pin dossier 为准——本包不含 V4 论文。
