# DFlash: Block Diffusion for Flash Speculative Decoding

- **arXiv**: 2602.06036 (ICML 2026, camera-ready 版 2026-05-28)
- **作者**: Jian Chen, Yesheng Liang, Zhijian Liu（Z Lab）
- **来源**: https://arxiv.org/html/2602.06036（HTML 全文，另有项目页 https://z-lab.ai/projects/dflash/ 与代码 https://github.com/z-lab/dflash）
- **抓取日期**: 2026-07-11

> 本文件按 `# PAPER §x.y` 锚点摘录原文关键段落与公式（原文英文，锚点后中文旁注仅解释、不改写公式本身）。参考文献列表未收录；附录 A.1/A.2（训练超参细节）、A.5 及之后未收录。

---

## Abstract

# PAPER Abstract

> Autoregressive large language models (LLMs) deliver strong performance but require inherently sequential decoding, leading to high inference latency and poor GPU utilization. Speculative decoding mitigates this bottleneck by using a fast draft model whose outputs are verified in parallel by the target LLM. However, existing methods still rely on autoregressive drafting, which remains sequential and constrains practical speedups. Diffusion LLMs offer a promising alternative by enabling parallel generation, but current diffusion models typically underperform compared with autoregressive models. In this paper, we introduce DFlash, a speculative decoding framework that employs a lightweight block diffusion model for parallel drafting. We show that speculative decoding provides a natural and effective setting for diffusion models. By generating draft tokens in a single forward pass, DFlash enables efficient drafting, and by conditioning the draft model on context features extracted from the target model, it achieves high-quality drafts with higher acceptance rates. Experiments show that DFlash achieves over 6× lossless acceleration [...], delivering up to 2.5× higher speedup than the state-of-the-art speculative decoding method EAGLE-3.

中文旁注：一句话摘要——**块扩散做草稿、目标模型特征做条件、单次前向出整块**，三件事叠加得到"论文自报"6× 无损加速、比 EAGLE-3 快 2.5×。

---

## 1 Introduction（节选）

# PAPER §1

> Speculative decoding [...] has emerged as a primary solution to this bottleneck. This paradigm employs a lightweight draft model to speculate a sequence of future tokens, which are then verified in parallel by the large target model. While this approach achieves lossless acceleration [...], state-of-the-art methods like EAGLE-3 [...] still rely on autoregressive drafting. This serial drafting process is not only inherently inefficient but also susceptible to error accumulation, which effectively caps achievable speedups at approximately 2–3×.

> Is there truly "no free lunch"? Can we build a diffusion drafter that is both lightweight and highly accurate?

> In this paper, we introduce DFlash [...] Our key insight is simple: **the target knows best**. [...] large autoregressive LLMs' hidden features implicitly contain information about multiple future tokens. DFlash utilizes these hidden features as context, conditioning the draft model to predict future blocks of tokens in parallel. [...] Instead of requiring a tiny draft model to reason from scratch, DFlash fuses the reasoning capabilities of the target model with the parallel generation speed of a small diffusion drafter.

> As shown in Figure 1, DFlash achieves up to a 6.1× speedup on Qwen3-8B [...], and is nearly 2.5× faster than the state-of-the-art EAGLE-3 across most benchmarks.

**Figure 1**（原文配图，描述）：Qwen3-8B 上 DFlash / EAGLE-3 / 自回归解码三者的加速比对比柱状图（Transformers 后端）。原文明确写"DFlash achieves more than 2.5× higher speedup than EAGLE-3"。——本图不收进 key_figures（属于总览性能图，非机制图），机制图见下文 Figure 2/3/4。

失败前例（正文明确点名，交代"为什么不能直接拿扩散模型当草稿器"）：

# PAPER §1（续）

> Methods such as DiffuSpec [...] and SpecDiff-2 [...] utilize massive (e.g., 7B parameter) draft models. This significant memory footprint is often prohibitively expensive for real-world serving. [...] the high drafting latency limits their practical speedups to a modest ~3×. [PARD (An et al., 2025)] trains small autoregressive models to mimic diffusion-style parallel generation [...] the resulting small models lack the modeling capacity of the target LLMs, leading to limited acceptance lengths and a speedup ceiling of approximately 3×.

中文旁注：这段是本章"为什么不是简单粗暴地上一个扩散模型"的直接出处——大扩散草稿器（DiffuSpec/SpecDiff-2）撞在"起草延迟"上，小型自回归模仿扩散（PARD）撞在"建模能力不足"上，DFlash 要同时躲开这两个坑。

---

## 2 Related Work（节选，仅摘与本章直接相关的定位句）

# PAPER §2.1

> The EAGLE series [...] further improves speculative decoding by exploiting feature-level context from the frozen target model. EAGLE-1 predicts future hidden-state distributions to boost acceptance rates, EAGLE-2 introduces adaptive drafting trees, and EAGLE-3 refines training objectives to scale speedups.
> Despite these advances, most existing methods rely on autoregressive drafting, which remains inherently sequential, limiting their speedups.

# PAPER §2.2

> Block diffusion models [...] address these issues by denoising sequences block-by-block, blending parallelism with autoregressive structure. [...] Nevertheless, existing dLLMs generally underperform state-of-the-art autoregressive models and often require many denoising steps, which limits their practical inference speed.

# PAPER §2.3

> DiffuSpec [...] and SpecDiff-2 [...] employ large pre-trained dLLMs as speculative drafters [...] these approaches rely on massive drafters (e.g., 7B parameters), incurring substantial memory and latency overhead.

中文旁注：DFlash 相对 EAGLE 系列的定位是"draft 不再逐 token 自回归"；相对 DiffuSpec/SpecDiff-2 的定位是"draft 模型极轻量（5 层量级）而非 7B"。

---

## 3 Preliminaries — 块扩散为什么在起草延迟上赢自回归

### 3.1 Speculative Decoding Speedup

# PAPER §3.1 Eq.(1)

> Following Sadhukhan et al. (2025), the average per-token latency is
>
> $$L=\frac{T_{\text{draft}}+T_{\text{verify}}}{\tau}, \qquad (1)$$
>
> where $T_{\text{draft}}$ is the time spent generating draft tokens, $T_{\text{verify}}$ is the cost of verification, and $\tau\in[1,\gamma+1]$ is the expected number of accepted tokens per cycle, including the bonus token produced by the target model. Let $L_{\text{target}}$ denote the autoregressive per-token latency of $\mathcal{M}_{t}$; the resulting speedup is $\eta=L_{\text{target}}/L$.

中文旁注：加速比 $\eta$ 由两处杠杆决定——分子的起草成本 $T_{\text{draft}}$ 越低越好，分母的期望接受长度 $\tau$ 越高越好。DFlash 的两个招（块扩散并行起草、KV 注入）分别对应压低分子和抬高分母，见 §3.2 与 §4.1。

### 3.2 Autoregressive vs. Diffusion Drafting

# PAPER §3.2 Eq.(2)(3)

> Autoregressive drafters generate tokens sequentially, incurring a drafting cost
>
> $$T_{\text{draft}}=\gamma\cdot t_{\text{step}}, \qquad (2)$$
>
> where $t_{\text{step}}$ is the latency of a single forward pass. Drafting costs therefore grow linearly with the speculation budget $\gamma$.
>
> To keep latency manageable, autoregressive drafters are constrained to very shallow architectures (e.g., a single transformer layer in EAGLE-3). This severely limits the draft quality [...]
>
> Diffusion drafters generate all $\gamma$ tokens in parallel within a single forward pass, yielding
>
> $$T_{\text{draft}}=t_{\text{parallel}}, \qquad (3)$$
>
> where $t_{\text{parallel}}$ denotes the latency of block generation. Modern GPUs execute such parallel operations far more efficiently than multiple sequential passes, making $t_{\text{parallel}}\ll\gamma\cdot t_{\text{step}}$.

> This parallelism fundamentally changes the design space. Because drafting cost no longer scales with the number of generated tokens, diffusion drafters can afford deeper, more expressive architectures without sacrificing latency. [...] Empirically, a five-layer DFlash draft model generating 16 tokens achieves both lower latency (Figure 3) and higher acceptance length than EAGLE-3 generating 8 tokens, placing DFlash on a more favorable Pareto frontier between draft quality and drafting cost.

**Figure 3**（key_figures 收录，机制图 1）：1/3/5 层 DFlash 与 1 层 EAGLE-3 的起草延迟（draft cost）对比曲线——横轴生成 token 数、纵轴延迟；DFlash 各层数曲线几乎水平，EAGLE-3 单层曲线随 token 数线性上升。

中文旁注：Eq.(2) 里 $\gamma$（自回归草稿的块大小）直接乘进延迟，逼得 EAGLE-3 只能用 1 层——这是"逐 token 自回归"这个控制流本身的代价，不是工程没做好；Eq.(3) 把 $\gamma$ 移出了延迟表达式，代价挪到了"单次前向"的常数项 $t_{\text{parallel}}$ 里，于是层数可以做深而不再线性拖慢起草。

---

## 4 Method — KV 注入与并行起草

**Figure 2**（key_figures 收录，机制图 2，正文置于 §3 开头处）：

# PAPER Figure 2 caption

> DFlash Inference Design. Hidden context features extracted from the target model are fused and injected into each draft layer's Key-Value cache to enable conditional speculation.

### 4.1 Inference

# PAPER §4.1（"Context features from the target model" 段）

> Prior work like An et al. (2025) naively applied a small diffusion model as a speculative drafter, which leads to poor acceptance length and limited speedups. To validate this, we train a five-layer block diffusion draft model without any conditioning from the target model and evaluate it on several math benchmarks. As the results shown in the Table 10, the resulting speedups are modest, typically around 3× from scratch.
>
> In contrast, the hidden representations of large autoregressive target models encode substantially more information than token-level logits. These features capture long-range dependencies and task-specific semantics, and—crucially—implicitly encode information about future token predictions, as also observed by Samragh et al. (2025).
>
> In DFlash, given an input prompt, the target model first performs a standard prefill pass to generate the first token. During this pass, we extract hidden representations from a fixed set of layers uniformly sampled from shallow to deep. These hidden states are concatenated and passed through a lightweight projection layer to fuse cross-layer information into a compact **target context feature**, which is then used to condition the draft model.

中文旁注：这段是"3× 天花板"的直接出处——不带 target 条件的裸块扩散起草器，速度上限就卡在 3× 左右（对应 meta.json 里"厂商自报"数字之一）。

# PAPER §4.1（"Conditioning via KV injection enables acceptance scaling" 段——EAGLE-3 对比 + DFlash 做法）

> Existing methods such as EAGLE-3 also leverage hidden features from the target model, but they fuse these features with the draft model's token embeddings and feed them only as inputs to the draft model. As the draft model depth increases, the information from target model becomes more and more diluted, resulting in diminishing gains in acceptance length when adding more draft layers.
>
> DFlash adopts a fundamentally different strategy. We treat the fused target context feature as persistent contextual information and directly inject it into the Key and Value projections of every draft model layer. The projected features are stored in the draft model's KV cache and reused across drafting iterations. [...] This design provides strong and consistent conditioning throughout the draft model, enabling acceptance length to scale effectively with the number of draft layers.

> Another key contributor to DFlash's speed is its low drafting latency. [...] DFlash predicts the next token block using a block-level diffusion process. All masked positions within a block are decoded in parallel in a single forward pass.

### 附录 A.3：KV 注入的精确算子形式（补 4.1 节没写出的公式）

# PAPER Appendix A.3

> DFlash uses KV injection to condition the diffusion drafter on target-model features. We first concatenate hidden states from selected target layers and project them once into the draft hidden dimension:
>
> $$\mathbf{H}_{t}=\mathrm{RMSNorm}\left(W_{c}[\mathbf{H}^{(l_{1})};\ldots;\mathbf{H}^{(l_{5})}]\right).$$
>
> The projected target features are shared by all draft layers. At layer $i$, draft tokens produce queries, while both target features and draft tokens are projected into keys and values:
>
> $$\mathbf{Q}_{i}=W_{i}^{Q}\mathbf{H}_{d},$$
> $$\mathbf{K}_{i}=[W_{i}^{K}\mathbf{H}_{t};\,W_{i}^{K}\mathbf{H}_{d}]_{\mathrm{seq}},$$
> $$\mathbf{V}_{i}=[W_{i}^{V}\mathbf{H}_{t};\,W_{i}^{V}\mathbf{H}_{d}]_{\mathrm{seq}}.$$
>
> Thus, target features only serve as additional KV entries for the masked-block draft tokens. They bypass the draft model's $Q$ projection, output projection, self-attention update, and FFN.
>
> The memory overhead is small. The only extra parameterized component is the shared projection $W_{c}\in\mathbb{R}^{D\times 5D}$ [...] negligible compared with the roughly 70 GB target model. [...] for batch size 1 and sequence length 2048, the projection input and output require about 40 MB and 8 MB, respectively. During decoding with block size 16, the temporary activation is below 400 KB.

中文旁注（术语澄清——"交叉注意力"是本章对机制的归纳描述，非论文原词）：论文正文没有出现 "cross-attention" 这个词，但 $\mathbf{K}_i,\mathbf{V}_i$ 的构造——**draft 自己的隐藏态 $\mathbf{H}_d$ 只贡献 Q**，**K/V 由 $[\mathbf{H}_t;\mathbf{H}_d]_{\mathrm{seq}}$ 拼接而成，其中 $\mathbf{H}_t$ 来自 target**——在效果上就是"draft 的 query 对 target 特征做交叉注意力 + 对自己做自注意力"合并在同一次 attention 里完成，且 $\mathbf{H}_t$ 逐层都注入（不是只在输入层），这正是与 EAGLE-3（$\mathbf{H}_t$ 只在输入层与 token embedding 融合、越往深层信号越稀释）的结构性差异所在。$l_1,\ldots,l_5$ 具体是"从 target 模型第 2 层到倒数第 3 层之间均匀采样的 5 层"（见 §5 Implementation 段）。

### 4.2 Training

**Figure 4**（key_figures 收录，机制图 3）：

# PAPER Figure 4 caption

> DFlash training attention. The target model provides context features (blue) that condition the draft model. The input consists of clean prompt tokens [and masked response tokens].

# PAPER §4.2

> DFlash draft models are trained to align block-level diffusion predictions with the outputs of a frozen autoregressive target model. [...]
>
> **KV injection.** Following the inference pipeline, given a sequence consisting of a prompt and its response, we first pass the entire clean sequence through the target model to extract and fuse the hidden features for all tokens. The hidden features are then injected into the draft model as Key and Value projections, as illustrated in Figure 4.
>
> **Random sampling of masked blocks.** In standard block diffusion training, the response is uniformly divided into blocks and random positions within each block are masked, with the model trained to denoise the masked tokens.
>
> DFlash instead tailors block construction to the speculative decoding setting. We randomly sample anchor tokens from the response, use each anchor as the first position of a block, and mask the remaining positions. The draft model is trained to predict the next $\text{block\_size}-1$ tokens in parallel. This directly matches inference-time behavior, where the draft model always conditions on a clean token produced by the target model (i.e., the bonus token from the previous verification step). Randomizing anchor positions also exposes the draft model to more diverse target context features, improving data efficiency and coverage.
>
> During training, all blocks are concatenated into a single sequence and processed jointly using a sparse attention mask as shown in Figure 4. Tokens attend bidirectionally within the same block and to the corresponding injected target context features, while attention across different blocks is disallowed. This design enables multiple draft blocks to be trained efficiently within a single forward and backward pass using Flex Attention.

# PAPER §4.2 Eq.(4)（位置加权损失）

> In speculative decoding, not all tokens are equal. Errors at early positions within a draft block invalidate all subsequent tokens. This makes early predictions disproportionately important for acceptance length. We reflect this asymmetry by weighting the cross-entropy loss to emphasize earlier token positions during training. Specifically, for a token at position $k$ we apply an exponentially decaying weight
>
> $$w_{k}=\exp\!\left(-\frac{k-1}{\gamma}\right), \qquad (4)$$

> **Shared embedding and LM head.** To improve training efficiency, the draft model shares the token embedding layer and language modeling head with the target model and keeps them frozen during training. Only the draft Transformer layers are updated.

中文旁注：Eq.(4) 说明块内位置 $k=1$ 权重为 1，越往后权重按 $\exp(-(k-1)/\gamma)$ 指数衰减——训练目标本身就承认"块扩散虽然并行出 token，但块内位置 1 的预测质量比位置 16 更值钱"，因为一旦位置 1 被拒绝，位置 2-16 全部作废（接受是最长前缀匹配，不是集合匹配）。这与"块内并行去噪"容易望文生义地以为"每个位置同等重要"恰好相反，是本章"块扩散不是多轮迭代去噪，而是单次前向内的掩码预测 + 位置加权训练"这一澄清的直接出处。

---

## 5 Experiments — 数值表（均为论文自报，未独立复现）

### 5.1 Instruct Models（Table 1 节选，Temperature = 0，格式 "Speedup / τ(平均接受长度)"）

# PAPER Table 1（节选）

| Model | Method | GSM8K | HumanEval | MT-Bench | Avg. |
|---|---|---|---|---|---|
| Qwen3-4B | EAGLE-3 (16) | 1.99 / 3.30 | 1.84 / 3.05 | 1.74 / 3.02 | 1.81 / 3.05 |
| Qwen3-4B | DFlash (16) | 5.15 / 6.53 | 5.21 / 6.64 | 2.85 / 4.35 | 4.91 / 6.54 |
| Qwen3-8B | EAGLE-3 (16) | 1.94 / 3.23 | 1.89 / 3.17 | 1.63 / 2.83 | 1.76 / 2.96 |
| Qwen3-8B | DFlash (16) | 5.15 / 6.54 | 5.14 / 6.50 | 2.75 / 4.24 | 4.86 / 6.49 |

> DFlash achieves an average speedup of 4.86× on Qwen3-8B, corresponding to a 2.76× improvement over EAGLE-3 (16)（由上表 Avg. 列 4.86/1.76 相除得出，原文未直接给出这个比值句，此处是本文档计算，仅供交叉核对）。

（论文自报；未独立复现。原表还含 MATH-500/AIME25/MBPP/LCB 列及 Temperature=1、EAGLE-3(60) 行，此处只摘 GSM8K/HumanEval/MT-Bench/Avg. 四列 × 两个 Temperature=0 模型，用于呼应本章 outline 点名的 GSM8K/HumanEval/MT-Bench 三个 benchmark。）

### 5.2 Reasoning Models（Table 2 节选，thinking 开启，格式同上）

# PAPER Table 2（节选，Temperature=0）

| Model | MATH-500 | AIME25 |
|---|---|---|
| Qwen3-4B | 4.59 / 5.74 | 4.39 / 5.54 |
| Qwen3-8B | 4.64 / 5.82 | 4.51 / 5.74 |

### 5.5.5 KV Injection vs. Input Fusion（消融，Qwen3-4B，5 层草稿器，block size 8）

# PAPER §5.5.5 + Table 9

> This ablation studies whether target features should be injected only once at the input layer, as in EAGLE-3 style input fusion, or injected into every draft layer as KV entries. [...]

格式 "τ(平均接受长度) / speedup"：

| Variant | Injection 方式 | GSM8K | HumanEval | MT-Bench |
|---|---|---|---|---|
| EAGLE-3-5L（自回归起草） | Input（仅输入层） | 4.2 / 2.1× | 4.3 / 2.2× | 3.1 / 1.4× |
| DFlash-AR（自回归起草，仅换成 KV 注入） | KV（逐层） | 4.8 / 2.4× | 4.6 / 2.3× | 3.4 / 1.5× |
| DFlash（块扩散起草） | Input（仅输入层） | 3.5 / 2.9× | 3.5 / 2.9× | 2.6 / 2.0× |
| DFlash（块扩散起草） | KV（逐层，完整方案） | 4.2 / 3.3× | 4.0 / 3.2× | 3.0 / 2.2× |

> The results show that KV injection is more effective than input fusion. In autoregressive drafting, DFlash-AR achieves higher acceptance length than EAGLE-3-5L on all tasks. In block-diffusion drafting, DFlash with KV injection also improves acceptance length over DFlash with input fusion on all tasks. [...] DFlash achieves acceptance length comparable to EAGLE-3-5L, but obtains much higher speedup because block diffusion drafts multiple tokens in parallel. Therefore, DFlash benefits from both stronger conditioning through KV injection and faster parallel drafting.

中文旁注：这张 2×2 消融表把"块扩散起草"和"KV 注入"两个变量拆开单独验证——**只换起草方式**（EAGLE-3-5L → DFlash 自回归版，都用 Input）：接受长度小涨（4.2→4.8）、速度小涨；**只换条件注入方式**（Input → KV，都用块扩散）：接受长度从 3.5 涨到 4.2、速度从 2.9× 涨到 3.3×。两个改动方向一致、都有效，且叠加后（完整 DFlash）达到接受长度持平 EAGLE-3-5L 但速度显著更高——因为块扩散是并行出块、EAGLE-3 是逐 token。

### 5.3 Performance on Serving Frameworks（节选）

# PAPER §5.3

> [Table 3 reports] Throughput (tok/s), speedup over baseline, and average acceptance length [on SGLang].

（论文正文此处的具体吞吐数字表格式在 HTML 抽取中未能完整保留数值，仅确认 Table 3 存在且统计口径为"吞吐/加速比/接受长度"；不在此处杜撰数字——生产级 SGLang 吞吐数字改用下方 LMSYS 博客的 Qwen3.5-397B-A17B 实测。）

---

## 附：LMSYS 博客数字（生产实现，非论文本体，标"厂商/合作方博客自报"）

来源：https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/（Z Lab / Modal / SGLang 三方联合博客，2026-06-15）

# PAPER（博客）"DFlash: Parallel drafting with KV injection"

> Our new, jointly-released DFlash model for Qwen 3.5 397B-A17B achieves higher throughput than both the baseline model and native MTP speculation in all the settings we benchmarked. At concurrency 1 on the HumanEval coding dataset, it achieves **>4.3x** the throughput of baseline and **1.5x** the throughput of MTP.
>
> Workload: Qwen 3.5 397B-A17B (BF16), HumanEval. Settings: greedy decoding, thinking enabled, max new tokens 4096. Hardware: 8xB200 on Modal. [...] Draft token/block counts selected for maximum throughput (MTP: 7 steps; DFlash: block size 16).

> For methods like EAGLE, the draft KV cache is fully private to the draft model, calculated based on KV projection of the draft's own latents. In DFlash, the latents of the target model are instead passed through a KV projection by the draft model.
>
> We don't want to store those latents and cut into precious KV cache space [...] So we run the draft KV projection ahead of the rest of the draft forward pass — **immediate materialization**. That needs to be fast, so we added a layer-batched linear projection and a fused Triton kernel for the norm+RoPE post-processing.

> Under V2 [engine, with overlap scheduling] [...] performance improved by over 33%, from ~11.4 ktok/s to ~15.3 ktok/s, when running Qwen 3-8B on a single B200 at concurrency 32.

中文旁注（"immediate materialization"术语落地）：这段是本章 outline 点名的 `precompute_and_store_context_kv`（vllm-ascend 源码里的实际函数名）在工程叙事层面的出处——"提前把 target 特征投影好、存进 KV cache，而不是等 draft 前向跑到每一层再现算"，对应论文里"projected features are stored in the draft model's KV cache and reused across drafting iterations"（§4.1）在生产实现里的具体做法（一次 layer-batched GEMM + 融合 Triton kernel，而非每层单独投影）。

**数字来源核实结论**（务必标注厂商/博客自报，未独立复现）：
- "Qwen3-8B 上 6× 无损加速 / 比 EAGLE-3 快 2.5×"——出自论文 Abstract + §1（arXiv:2602.06036），非独立复现。
- "GSM8K 3.3×/HumanEval 3.2×/MT-Bench 2.2×"（Qwen3-4B，5 层 DFlash vs EAGLE-3-5L）——出自 LMSYS 博客消融表，与论文 Table 9（DFlash/KV 行）数字一致（3.3×≈论文 GSM8K 4.2/3.3、3.2×≈HumanEval 4.0/3.2、2.2×≈MT-Bench 3.0/2.2），互相印证，但两处均为厂商自报。
- "Qwen3.5-397B-A17B：>4.3× vs baseline，1.5× vs MTP"——仅见于 LMSYS 博客，特定工作负载（HumanEval、并发 1、8×B200），非通用结论。
- "3× 起草天花板"（裸块扩散无条件 / PARD 类小型自回归模仿扩散）——出自论文 §1 与 §4.1，两处独立提及但数值均为约数（"approximately 3×"/"typically around 3× from scratch"），非精确复现实验对照组。
