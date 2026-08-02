# DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation
  Xin Cheng1,2,∗      Xingkai Yu2,∗      Chenze Shao2,∗      Jiashi Li2,∗      Yunfan Xiong2,∗ Yi Qian2      Jiaqi Zhu2      Shirong Ma2      Xiaokang Zhang2      Jiasheng Ye2      Qinyu Chen2      Chengqi Deng2      Jiping Yu2      Damai Dai2      Zhengyan Zhang2      Yixuan Wei2      Yixuan Tan2      Wenkai Yang2      Runxin Xu2      Yu Wu2      Zhean Xu2      Xuanyu Wang2      Muyang Chen2      Rui Tian2      Xiao Bi2      Zhewen Hao2      Shaoyuan Chen2      Huanqi Cao2      Wentao Zhang2      Anyi Xu2      Huishuai Zhang1      Dongyan Zhao1      Wenfeng Liang2  
1Peking University    2DeepSeek-AI  
{chengxin      xingkai      shaochenze      js.li      yunfanxiong}@deepseek.com   

###### Abstract
Speculative decoding accelerates Large Language Model (LLM) inference by decoupling draft generation from target verification. While recent parallel drafters efficiently propose long token sequences in a single forward pass, they suffer from rapid acceptance decay due to a lack of inter-token dependencies. Furthermore, indiscriminately verifying these extended blocks wastes critical batch capacity on tokens with high rejection risks, severely degrading throughput in high-concurrency serving systems. We introduce DSpark, a speculative decoding framework that unifies high-throughput parallel generation with adaptive, load-aware verification. To maintain draft quality, DSpark utilizes a semi-autoregressive architecture—coupling a parallel backbone with a lightweight sequential module—to introduce intra-block dependency modeling and mitigate suffix decay. To optimize system efficiency, DSpark employs confidence-scheduled verification, dynamically tailoring the verification length for each request based on estimated prefix survival probabilities and engine-specific throughput profiles. On offline benchmarks across diverse domains, DSpark substantially improves the accepted length over state-of-the-art autoregressive and parallel drafters. When deployed within the DeepSeek-V4 serving system under live user traffic, DSpark successfully mitigates verification waste. Compared to the established production baseline (MTP-1), DSpark accelerates per-user generation speeds by 60%–85% at matched throughput levels. More importantly, by preventing severe throughput degradation under strict interactivity constraints, it enables performance tiers that were previously unattainable, shifting the Pareto frontier of our serving system. To facilitate community progress, we open-source the [DSpark checkpoints](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark/tree/main) alongside [DeepSpec](https://github.com/deepseek-ai/DeepSpec), an algorithm-driven training repository for speculative decoding.

\*\*footnotetext: Equal contribution.

## 1 Introduction
Large Language Models (LLMs) generate text autoregressively: each new token requires a full forward pass conditioned on all preceding tokens, making inference latency proportional to the output length. The resulting low GPU utilization and high user-perceived waiting time constitute a primary bottleneck in production LLM serving, particularly for latency-sensitive scenarios such as real-time conversational assistants and multi-turn agentic workflows. Speculative decoding \[Leviathan et al., [2023](#bib.bib46), Chen et al., [2023](#bib.bib11)\] offers a principled solution: a lightweight draft model proposes a block of candidate tokens, and the full-size target model verifies the entire block in a single forward pass via rejection sampling, accepting the longest prefix consistent with the target distribution and appending one bonus token. Because verification is parallel and the acceptance rule preserves the target distribution exactly, speculative decoding accelerates generation without any quality loss.

The design of the draft model governs the trade-off between drafting latency and acceptance rate. Early drafters are autoregressive \[Li et al., [2024b](#bib.bib52), Cheng et al., [2024](#bib.bib14)\], conditioning each position on previously sampled tokens. However, their drafting latency grows linearly with the block size, forcing these methods to use short blocks and shallow architectures. To break this sequential bottleneck, parallel drafters \[Cai et al., [2024](#bib.bib9), Liu et al., [2026a](#bib.bib58), Chen et al., [2026](#bib.bib12)\] have emerged as a compelling alternative: all draft positions are produced in a single forward pass, making drafting latency nearly independent of block size. This structural advantage theoretically allows parallel drafters to efficiently generate substantially longer draft blocks.

However, fully unlocking the potential of large parallel draft blocks introduces two critical bottlenecks—one in generation quality, and the other in system efficiency. First, because parallel drafters predict each position independently, they cannot model inter-token dependencies within a block. This independence leads to multi-modal collisions and rapid acceptance decay at later positions \[Gu et al., [2018](#bib.bib28), Huang et al., [2022b](#bib.bib37)\]. Second, determining the optimal verification length remains a challenge. While parallel generation easily produces long draft blocks, indiscriminately verifying all proposed tokens degrades system throughput, particularly under high-concurrency workloads \[Liu et al., [2024c](#bib.bib62), Hu et al., [2026b](#bib.bib34)\]. The ideal verification length varies along two axes. On the data side, structured requests like code naturally sustain higher acceptance rates than open-ended chat \[Xia et al., [2024](#bib.bib92), Abramovich et al., [2026](#bib.bib1)\]. On the system side, verifying extra tokens is nearly free under light loads. Under heavy loads, however, verifying tokens with a high rejection risk occupies critical batch capacity that could otherwise serve other active requests \[Liu et al., [2024b](#bib.bib61), Wu et al., [2025](#bib.bib90)\].

To address these bottlenecks, we introduce DSpark, a speculative decoding framework that unifies high-throughput parallel generation with adaptive, load-aware verification. At its core, DSpark is designed to resolve the inherent trade-offs in draft generation and verification through two complementary mechanisms.

  - •
    
    First, to overcome the lack of inter-token dependencies, DSpark adopts a semi-autoregressive architecture. It keeps the computationally expensive draft backbone fully parallel, appending only a lightweight serial output head to inject local transition information. This design preserves the drafting speed of parallel models while significantly mitigating suffix decay.
    
  - •
    
    Second, to resolve the system-level bottleneck, DSpark employs confidence-scheduled verification. By coupling a confidence head—which estimates per-position prefix survival probabilities—with a hardware-aware scheduler, DSpark dynamically tailors the verification length for each request. This scheduler leverages real-time engine throughput profiles to route target verification budget only toward tokens with the highest expected return.
    

We extensively evaluate DSpark across both controlled offline benchmarks and production-scale online deployments. On controlled offline benchmarks—spanning mathematical reasoning, code generation, and daily chat—DSpark consistently outperforms strong baselines. Specifically, across the Qwen3-4B, 8B, and 14B target models \[Yang et al., [2025](#bib.bib97)\], it improves the macro-average accepted length over the autoregressive Eagle3 \[Li et al., [2026b](#bib.bib54)\] by 30.9%, 26.7%, and 30.0%, and over the parallel DFlash \[Chen et al., [2026](#bib.bib12)\] by 16.3%, 18.4%, and 18.3%, respectively. Beyond top-line metrics, our fine-grained position-wise analysis reveals the distinct generation characteristics of different drafters, empirically demonstrating how DSpark successfully combines the high initial-token capacity of parallel models with the suffix coherence of autoregressive models.

Beyond offline evaluation, we deployed DSpark within the DeepSeek-V4 \[DeepSeek-AI, [2026](#bib.bib18)\] serving system to assess its performance under live user traffic. Compared to the prior MTP-1 production baseline \[DeepSeek-AI, [2024](#bib.bib17)\], DSpark significantly broadens the system’s operational envelope. Specifically, it consistently accelerates per-user generation speeds by 60%–85% (V4-Flash) and 57%–78% (V4-Pro) at matched aggregate throughput capacities. Furthermore, under strict Service Level Agreements (SLAs) where the baseline’s capacity deteriorates severely—such as 120 TPS for Flash and 50 TPS for Pro—DSpark mitigates verification overhead to maintain robust throughput. By overcoming this performance cliff, DSpark unlocks strict interactivity tiers that were previously unattainable, effectively shifting the Pareto frontier of LLM serving.

To foster collective advancement within the open-source community, we are making our artifacts publicly available. Specifically, we release the trained [DSpark checkpoints](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark/tree/main) for both the DeepSeek-V4-Flash (preview) and DeepSeek-V4-Pro (preview) models. Furthermore, we open-source [DeepSpec](https://github.com/deepseek-ai/DeepSpec), an algorithm-driven training repository, including Eagle3, DFlash and DSpark. These artifacts are intended to support further research on efficient LLM serving.

## 2 Background
### 2.1 Speculative Decoding
Autoregressive language models generate one token per forward pass, making inference latency proportional to output length. Speculative decoding \[Ge et al., [2022](#bib.bib24), Leviathan et al., [2023](#bib.bib46), Chen et al., [2023](#bib.bib11)\] accelerates the inference of a target model \(M_{t}\) using a lightweight draft model \(M_{d}\). At each decoding cycle, the draft model proposes \(\gamma\) candidate tokens \(x_{1},\ldots,x_{\gamma}\). The target model verifies all candidates in a single forward pass, accepting the longest prefix consistent with its own distribution.

Concretely, at each draft position \(k\), the target model computes its own distribution \(p_{k}^{t}\) and compares it against the draft distribution \(p_{k}^{d}\). The token \(x_{k}\) is accepted with probability \(\min{(1,{{{p_{k}^{t}\hspace{0pt}{(x_{k})}}/p_{k}^{d}}\hspace{0pt}{(x_{k})}})}\). Verification proceeds left to right: the first rejection at position \(k\) discards all subsequent tokens \(x_{k + 1},\ldots,x_{\gamma}\), regardless of their quality.

Let \(\tau\) denote the number of accepted tokens per cycle, and let \(T_{\text{draft}}\) and \(T_{\text{verify}}\) be the wall-clock times of the drafting and verification passes, respectively. The average latency per generated token is:

|  |                                                              |  |                                                                   |
|  | ------------------------------------------------------------ |  | ----------------------------------------------------------------- |
|  | \[{L = \frac{T_{\text{draft}} + T_{\text{verify}}}{\tau}}.\] |  | (1) |

Improving speedup therefore reduces to three levers: lowering \(T_{\text{draft}}\) (draft faster), raising \(\tau\) (draft better), or reducing the effective \(T_{\text{verify}}\) (verify smarter).

### 2.2 Drafter Architectures
The design of the draft model determines how \(T_{\text{draft}}\) and \(\tau\) trade off. Existing approaches fall into two categories.

##### Autoregressive drafters.
Autoregressive drafters generate draft tokens sequentially, conditioning each position on previously sampled tokens \[DeepSeek-AI, [2024](#bib.bib17), Li et al., [2024c](#bib.bib53), [b](#bib.bib52), [2026b](#bib.bib54), Zhang et al., [2025](#bib.bib101)\]. This explicit dependency gives strong modeling capacity, but the drafting cost grows linearly with block size: \(T_{\text{draft}} \propto \gamma\), which forces autoregressive drafters to use small \(\gamma\) and shallow architectures to keep \(T_{\text{draft}}\) low. To compensate for the short block, tree-based verification \[Miao et al., [2024](#bib.bib65)\] expands candidates into a tree and verifies multiple paths via tree attention, but the large number of verification tokens reduces overall serving throughput.

##### Parallel drafters.
Parallel drafters produce all \(\gamma\) draft tokens in a single forward pass, making \(T_{\text{draft}}\) nearly independent of the block size \[Cai et al., [2024](#bib.bib9), Chen et al., [2026](#bib.bib12), Liu et al., [2026a](#bib.bib58), Li et al., [2025a](#bib.bib47), Sandler et al., [2026](#bib.bib73)\]. This allows substantially larger blocks (e.g., \(\gamma = 16\)) without proportionally increasing latency.

Among them, DFlash \[Chen et al., [2026](#bib.bib12)\] is a state-of-the-art parallel drafter, which conditions its draft model on rich context features extracted from the target model (KV injection). During prefill, hidden states from a set of target layers \(\{ l_{1},\ldots,l_{m}\}\) are concatenated and projected into the draft hidden space:

|  |                                                                                                                                  |  |                                                                   |
|  | -------------------------------------------------------------------------------------------------------------------------------- |  | ----------------------------------------------------------------- |
|  | \[{H_{\text{ctx}} = {{RMSNorm}\hspace{0pt}\left( {W_{c}\hspace{0pt}{\lbrack H^{(l_{1})};\ldots;H^{(l_{m})}\rbrack}} \right)}},\] |  | (2) |

where \(W_{c} \in {\mathbb{R}}^{{d \times m}\hspace{0pt}d}\) is a shared projection. These context features are injected into every draft layer by concatenating them with the draft block representations along the sequence dimension of keys and values:

|  |                                                                                                                                                                                                  |  |                                                                   |
|  | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |  | ----------------------------------------------------------------- |
|  | \[{{K_{i} = {\lbrack{W_{i}^{K}\hspace{0pt}H_{\text{ctx}}};{W_{i}^{K}\hspace{0pt}H_{d}}\rbrack}},{V_{i} = {\lbrack{W_{i}^{V}\hspace{0pt}H_{\text{ctx}}};{W_{i}^{V}\hspace{0pt}H_{d}}\rbrack}}}.\] |  | (3) |

All positions within a block attend bidirectionally to each other and to the injected target context.

The draft model shares the target model’s embedding layer and language modeling head (both frozen). It takes as input the embedding of an anchor token111We use the terms anchor token and bonus token interchangeably in this paper to denote the final token generated by the target model in the previous decoding round. followed by \(\gamma\) mask token embeddings, and produces logits for all mask positions in a single forward pass. Since drafting requires only a single forward pass regardless of block size, DFlash can afford deeper architectures and larger blocks than autoregressive drafters under the same latency budget.

## 3 Architecture
The overview of DSpark is shown in [Figure 1](#S3.F1 "Figure 1 ‣ 3 Architecture ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation"). Recall from [Equation 1](#S2.E1 "1 ‣ 2.1 Speculative Decoding ‣ 2 Background ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation") that the per-token latency of speculative decoding is \(L = {{({T_{\text{draft}} + T_{\text{verify}}})}/\tau}\). Autoregressive drafters achieve high \(\tau\) but pay \(T_{\text{draft}} \propto \gamma\); parallel drafters collapse \(T_{\text{draft}}\) to a single pass but sacrifice \(\tau\) because each position is predicted independently. Meanwhile, fixed-length verification wastes \(T_{\text{verify}}\) on low-confidence suffix tokens that are almost certain to be rejected. DSpark addresses these limitations with two complementary components:

  - •
    
    Semi-autoregressive generation ([Section 3.1](#S3.SS1 "3.1 Semi-Autoregressive Generation ‣ 3 Architecture ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation")). A parallel backbone handles the bulk of draft computation, which keeps \(T_{\text{draft}}\) nearly independent of \(\gamma\). A lightweight sequential block then injects dependency among draft tokens, improving \(\tau\) at minimal additional latency.
    
  - •
    
    Confidence-scheduled verification ([Section 3.2](#S3.SS2 "3.2 Confidence-Scheduled Verification ‣ 3 Architecture ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation")). A confidence head estimates per-position acceptance probabilities, and a hardware-aware scheduler uses these estimates to prune low-confidence suffix tokens, cutting unnecessary verification compute.
    

![Figure 1:  The DSpark architecture and decoding cycle. Given prompt tokens ABC, the target model executes one step to generate the next token D, which serves as the anchor for the drafting phase. Using D as the input, DSpark employs a heavy parallel backbone and a lightweight sequential head to generate draft tokens EFGH along with their corresponding confidence scores \(c_{1}\)–\(c_{4}\). The Hardware-Aware Prefix Scheduler then evaluates these scores to retain the prefix EFG and drop the low-confidence token H. Finally, the target model verifies the scheduled prefix in parallel. As illustrated, E and F are accepted while G is rejected, prompting the model to generate a corrected token G∗ to complete the current round. ](2607.05147v1/x1.png)

### 3.1 Semi-Autoregressive Generation
A parallel drafter produces all \(\gamma\) draft logits in one forward pass, so each prediction cannot condition on tokens sampled elsewhere in the block. When the context admits multiple plausible continuations, e.g., “of course” and “no problem”, a parallel drafter may produce incoherent combinations such as “of problem” or “no course”, because each position marginalizes over all possible predecessors rather than conditioning on the one actually sampled \[Gu et al., [2018](#bib.bib28), Huang et al., [2022a](#bib.bib36)\]. Acceptance rate thus decays rapidly along the block, wasting both draft and verification compute. We therefore adopt a semi-autoregressive structure that splits draft generation into two stages:

##### Parallel stage.
A parallel backbone (in our instantiation, DFlash \[Chen et al., [2026](#bib.bib12)\]) runs a single forward pass over the entire block, producing hidden states \(h_{1},\ldots,h_{\gamma}\) and base logits \(U_{1},\ldots,U_{\gamma}\). We make only a minor modification to the original DFlash backbone: instead of feeding an anchor token plus \(\gamma\) mask tokens and predicting only the mask positions, we treat the anchor itself as the first prediction position, so \(\gamma\) input tokens (anchor \(+\) \(\gamma - 1\) masks) yield \(\gamma\) draft logits. This reduces draft computation while maintaining similar draft quality.

##### Sequential stage.
The sequential stage supplements the base logits with a prefix-dependent transition bias \(B_{k}\hspace{0pt}{(x_{0},x_{(4) |

Here, \(x_{0}\) denotes the anchor token from the previous verification cycle, \(U_{k}\) is the base logit vector produced by the parallel backbone at position \(k\), and \(\mathcal{V}\) is the vocabulary. At inference time, the sequential block samples left to right according to \(p_{k}{( \cdot \mid x_{0},x_{

  - •
    
    Markov head. The simplest instantiation restricts \(B_{k}\) to depend only on the immediately preceding token, reducing it to a first-order transition \(B\hspace{0pt}{(x_{k - 1},x_{k})}\). In principle this is a full \(V \times V\) matrix \(B\); we approximate it with a low-rank factorization \(B = {W_{1}\hspace{0pt}W_{2}}\), where \(W_{1} \in {\mathbb{R}}^{V \times r}\) and \(W_{2} \in {\mathbb{R}}^{r \times V}\). Given the preceding token \(x_{k - 1}\), the transition bias for position \(k\) is:
    |  |                                                                                                                                   |  |                                                                   |
    |  | --------------------------------------------------------------------------------------------------------------------------------- |  | ----------------------------------------------------------------- |
    |  | \[{{B\hspace{0pt}{(x_{k - 1}, \cdot )}} = {W_{1}\hspace{0pt}{\lbrack x_{k - 1}\rbrack}\hspace{0pt}W_{2}} \in {\mathbb{R}}^{V}},\] |  | (5) |
    where \(W_{1}\) serves as an embedding lookup table and \(W_{2}\) as a logit projection. The low-rank factorization (\(r = 256\) by default) keeps both storage and per-step compute small, making the sequential loop efficient even for large vocabularies. Returning to the earlier example: once position 1 samples “of”, the Markov head boosts “course” and suppresses “problem” at position 2, which mitigates the cross-mode collision.
    
  - •
    
    RNN head. The Markov head is memoryless beyond one step—position \(k\) cannot access tokens before \(x_{k - 1}\). The RNN head relaxes this by maintaining a recurrent state \(s_{k}\) that accumulates the full prefix history within a block. At each step, the module concatenates the current state \(s_{k - 1} \in {\mathbb{R}}^{r}\), the previous token embedding \({W_{1}\hspace{0pt}{\lbrack x_{k - 1}\rbrack}} \in {\mathbb{R}}^{r}\), and the backbone hidden \(h_{k} \in {\mathbb{R}}^{d}\) into an input vector \(z_{k} = {\lbrack s_{k - 1};{W_{1}\hspace{0pt}{\lbrack x_{k - 1}\rbrack}};h_{k}\rbrack} \in {\mathbb{R}}^{{2\hspace{0pt}r} + d}\), then applies a single gated update:
    

### 3.2 Confidence-Scheduled Verification
The semi-autoregressive architecture enables DSpark to generate large draft blocks efficiently. However, producing more draft tokens does not automatically translate to higher end-to-end speedups. Indiscriminately verifying the full draft block can actually degrade overall system throughput, especially in high-concurrency scenarios \[Liu et al., [2024c](#bib.bib62), Hu et al., [2026b](#bib.bib34)\].

This performance bottleneck stems from two interacting factors. First, on the data side, draft acceptance rates inherently vary across domains: structured text like code naturally yields high acceptance, whereas open-ended chat has significantly lower acceptance \[Xia et al., [2024](#bib.bib92), Abramovich et al., [2026](#bib.bib1)\]. Second, on the system side, the actual cost of verifying an extra token depends strictly on the engine load. Under light system load, an extra verification incurs minimal penalty even if rejected. However, under high-concurrency deployments, every unnecessary verification occupies target model batch capacity that could otherwise serve other active requests \[Liu et al., [2024b](#bib.bib61), Wu et al., [2025](#bib.bib90)\].

Therefore, fully unlocking the potential of large draft blocks requires a unified mechanism that routes target model compute only toward tokens with a positive expected return. DSpark achieves this by coupling a confidence head ([Section 3.2.1](#S3.SS2.SSS1 "3.2.1 Confidence Head ‣ 3.2 Confidence-Scheduled Verification ‣ 3 Architecture ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation")) that predicts prefix survival probabilities, with a hardware-aware prefix scheduler ([Section 3.2.2](#S3.SS2.SSS2 "3.2.2 Hardware-Aware Prefix Scheduler ‣ 3.2 Confidence-Scheduled Verification ‣ 3 Architecture ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation")) that dynamically determines the optimal verification lengths based on current system load.

#### 3.2.1 Confidence Head
Drawing inspiration from Huang et al. \[[2024](#bib.bib39)\], Wang et al. \[[2026b](#bib.bib86)\], the confidence head outputs a scalar \(c_{k} \in {(0,1)}\) for each draft position \(k\). Crucially, \(c_{k}\) models the conditional probability that the draft token at position \(k\) will survive target verification, given that all preceding tokens in the block have been accepted. The architecture features a lightweight linear projection followed by a sigmoid function:

|  |                                                                                                                                              |  |                                                                   |
|  | -------------------------------------------------------------------------------------------------------------------------------------------- |  | ----------------------------------------------------------------- |
|  | \[{c_{k} = {\sigma\hspace{0pt}\left( {w^{\top}\hspace{0pt}{\lbrack h_{k};{W_{1}\hspace{0pt}{\lbrack x_{k - 1}\rbrack}}\rbrack}} \right)}},\] |  | (7) |

where \(h_{k}\) is the hidden state of the backbone and \(W_{1}\hspace{0pt}{\lbrack x_{k - 1}\rbrack}\) is the Markov Embedding from the previous draft token. We supervise \(c_{k}\) using the analytical acceptance rate per-step \(c_{k}^{\ast}\). This rate is determined by the total variation distance between the draft distribution \(p_{k}^{d}\) and the target distribution \(p_{k}^{t}\):

|  |                                                                                        |  |                                                                   |
|  | -------------------------------------------------------------------------------------- |  | ----------------------------------------------------------------- |
|  | \[{c_{k}^{\ast} = {1 - {\frac{1}{2}\hspace{0pt}{\|{p_{k}^{d} - p_{k}^{t}}\|}_{1}}}}.\] |  | (8) |

##### Post-hoc Calibration.
Unlike threshold-based verification heuristics \[Huang et al., [2024](#bib.bib39), Li et al., [2024b](#bib.bib52), Zhang et al., [2026b](#bib.bib102)\], which only require confidence scores to correctly rank draft token qualities, our hardware-aware scheduling approach (detailed in [Section 3.2.2](#S3.SS2.SSS2 "3.2.2 Hardware-Aware Prefix Scheduler ‣ 3.2 Confidence-Scheduled Verification ‣ 3 Architecture ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation")) precisely requires the absolute magnitudes of the cumulative acceptance probabilities to compute the expected acceptance length \(\tau\). Because neural confidence estimates are often overconfident \[Guo et al., [2017](#bib.bib30), Ovadia et al., [2019](#bib.bib67)\], using the raw confidence scores directly would distort the throughput estimation, leading to suboptimal scheduling.

To address this, we introduce Sequential Temperature Scaling (STS). Because each \(c_{i}\) models a conditional probability, the chain rule dictates that the joint probability of a draft prefix being accepted factorizes into the cumulative product \(\prod_{i \leqslant k}c_{i}\). Using a held-out validation set, STS calibrates this joint probability consecutively from left to right. Specifically, at each position \(k \in {\{ 1,\ldots,\gamma\}}\), we perform a simple 1D grid search to find the optimal temperature scalar that minimizes the Expected Calibration Error (ECE) \[Naeini et al., [2015](#bib.bib66)\] of the cumulative product, keeping the already-calibrated scores of all preceding positions fixed. Crucially, temperature scaling is an order-preserving transformation: it rectifies the predicted probabilities to match empirical acceptance rates without disrupting the relative draft token rankings learned by the confidence head.

#### 3.2.2 Hardware-Aware Prefix Scheduler
Algorithm 1  Hardware-Aware Prefix Scheduler

0: Active requests \(r \in {\{ 1,\ldots,R\}}\); confidence sequence \(c_{r,1},\ldots,c_{r,\gamma}\) per request; profiled step curve \(\text{SPS}\hspace{0pt}{(B)}\)

0: Selected per-request prefix lengths \(\ell_{1}^{\ast},\ldots,\ell_{R}^{\ast}\)

1: for \(r = 1\) to \(R\) do

2:  Compute prefix survival probabilities: \(a_{r,j}\leftarrow{\prod_{i \leqslant j}c_{r,i}}\) for \(j = {1,\ldots,\gamma}\)

3: end for

4: Construct candidate space \(\mathcal{E}\leftarrow{\{{(r,j)}\mid{a_{r,j} > 0}\}}\) and sort descending by \(a_{r,j}\)

5: Initialize states: \(\ell_{r}\leftarrow 0\) for all \(r\); Batch size \(B\leftarrow R\); Expected accepts \(\tau^{\ast}\leftarrow R\)

6: Initialize tracking: \(\Theta_{\text{best}}\leftarrow{{R \cdot \text{SPS}}\hspace{0pt}{(R)}}\); Selected lengths \(\ell_{r}^{\ast}\leftarrow 0\) for all \(r\)

7: for each \({(r,j)} \in \mathcal{E}\) in sorted order do

8:  \(\ell_{r}\leftarrow j\); \(B\leftarrow{B + 1}\); \(\tau^{\ast}\leftarrow{\tau^{\ast} + a_{r,j}}\)

9:  Current throughput \(\Theta\leftarrow{{\tau^{\ast} \cdot \text{SPS}}\hspace{0pt}{(B)}}\)

10:  if \(\Theta > \Theta_{\text{best}}\) then

11:   \(\Theta_{\text{best}}\leftarrow\Theta\); Update selected lengths \(\ell_{r}^{\ast}\leftarrow\ell_{r}\)

12:  else

13:   break

14:  end if

15: end for

16: return \((\ell_{1}^{\ast},\ldots,\ell_{R}^{\ast})\) achieving \(\Theta_{\text{best}}\)

Prior methods \[Huang et al., [2024](#bib.bib39), Li et al., [2024b](#bib.bib52)\] typically apply a static threshold to confidence scores to determine verification length. While effective under isolated, single-request assumptions, static thresholds can be suboptimal in high-concurrency production systems, where the utility of verifying a draft token depends heavily on the current system load.

To address this, we formulate verification length selection as a global throughput maximization problem ([Algorithm 1](#alg1 "Algorithm 1 ‣ 3.2.2 Hardware-Aware Prefix Scheduler ‣ 3.2 Confidence-Scheduled Verification ‣ 3 Architecture ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation")). Consider a batch of \(R\) active requests. For request \(r\), let \(c_{r,1},\ldots,c_{r,\gamma}\) be the per-position confidence estimates, and let \(\ell_{r} \in {\{ 0,\ldots,\gamma\}}\) denote the scheduled verification length. Because speculative decoding dynamically accepts draft tokens only as a continuous prefix, the survival probability of a token at position \(j\) is the cumulative product \(a_{r,j} = {\prod_{i \leqslant j}c_{r,i}}\).

In a single verification step, the total batch size (measured in tokens) sent to the target model is \(B = {\sum_{r = 1}^{R}{({1 + \ell_{r}})}}\), and the expected number of successfully accepted tokens is \(\tau = {\sum_{r = 1}^{R}\left( {1 + {\sum_{j = 1}^{\ell_{r}}a_{r,j}}} \right)}\). Under a simplifying assumption222In practical serving scenarios, average context lengths remain well below extremes (e.g., 1M tokens), making their impact on decode latency marginal for highly optimized architectures like DeepSeek-V4. Moreover, in prefill-decode disaggregated deployments, decode load balancers keep both request counts and total context lengths roughly balanced across data-parallel (DP) ranks. This effectively amortizes sequence-length variance, allowing us to safely assume that engine throughput depends predominantly on the verification batch size \(B\)., let \(\text{SPS}\hspace{0pt}{(B)}\) denote the engine throughput, measured in steps per second, for a given forward-pass batch size \(B\). Crucially, this capacity curve is profiled once during engine initialization and stored as a lightweight cost table. Our scheduler then aims to maximize the expected system-wide token throughput \(\Theta = {{\tau \cdot \text{SPS}}\hspace{0pt}{(B)}}\) by dynamically selecting verification lengths \(\ell_{1},\ldots,\ell_{R}\).

Although finding the global maximum of \(\Theta\) appears to be a combinatorial search, the objective structure allows for an efficient greedy solution. Because \(a_{r,j}\) is monotonically non-increasing with respect to \(j\) (i.e., \(a_{r,j} \leq a_{r,{j - 1}}\)), the marginal gain in expected accepted tokens for extending request \(r\)’s verification length from \(j - 1\) to \(j\) is exactly \(a_{r,j}\). This monotonicity ensures that sorting candidate tokens globally by \(a_{r,j}\) naturally respects intra-block prefix dependencies. Consequently, if the total verification batch size \(B\) were fixed, the optimal allocation \(\{\ell_{r}\}\) would be determined by greedily selecting the draft tokens with the highest survival probabilities from the global pool of all \(\{ a_{r,j}\}\).

Building on this insight, the optimization can be evaluated along this greedy admission path. We first globally sort all valid prefix extensions in descending order of survival probability. To dynamically determine the optimal target batch size \(B\), we incrementally admit tokens from this sorted pool, updating the expected throughput \(\Theta\) via an lookup from the cost table.

Lossless speculative decoding strictly requires the non-anticipating property: admission decisions must not depend on future candidate tokens \[Leviathan et al., [2023](#bib.bib46), Chen et al., [2023](#bib.bib11)\]. Because our confidence head relies on the Markov feature of the previously sampled token, computing the next survival probability \(a_{r,{k + 1}}\) explicitly requires the instantiated candidate \(x_{r,k}\). A retrospective global search would thus inadvertently leak \(x_{r,k}\) into the admission decision for step \(k\), introducing selection bias (we provide a concrete counterexample demonstrating this theoretical violation in [Appendix A](#A1 "Appendix A Counterexample: Selection Bias Without Early-Stopping ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation")).

To enforce strict causality, the scheduler ([Algorithm 1](#alg1 "Algorithm 1 ‣ 3.2.2 Hardware-Aware Prefix Scheduler ‣ 3.2 Confidence-Scheduled Verification ‣ 3 Architecture ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation")) employs an early-stopping mechanism. By breaking the greedy search immediately when the throughput drops (\(\Theta \leq \Theta_{\text{best}}\)), the truncation decision relies solely on the prefix processed up to that exact step. This isolates the admission event from future tokens, ensuring exact target-distribution recovery. Note that this stepwise early-stopping yields the global maximum throughput if and only if the objective \(\Theta\) is unimodal, which implicitly assumes a smoothly decaying hardware capacity curve. We address the engineering adaptations required for real-world, non-smooth SPS characteristics and asynchronous system pipelines in [Section 5.2](#S5.SS2 "5.2 Hardware-Aware Prefix Scheduler in Practice ‣ 5 Real-World Deployment of DSpark ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation").

### 3.3 Training
During training, we randomly sample multiple anchor positions from each target sequence to form \(\gamma\)-token blocks as training data. The target model is frozen throughout training; the draft model shares its embedding layer and language modeling head and keeps them frozen, updating only the backbone drafter, sequential block, and confidence head.

The training objective consists of three terms: a cross-entropy loss \(\mathcal{L}_{\text{ce}}\), a distribution-matching loss \(\mathcal{L}_{\text{tv}}\), and a confidence loss \(\mathcal{L}_{\text{conf}}\). All three are position-weighted by \(w_{k} = {\exp{({- {{({k - 1})}/\gamma}})}}\) \[Chen et al., [2026](#bib.bib12)\], which emphasizes earlier block positions that contribute more to the expected acceptance length under prefix-based verification. The cross-entropy loss \(\mathcal{L}_{\text{ce}}\) trains the drafter to predict the correct next token:

|  |                                                                                                                                    |  |                                                                   |
|  | ---------------------------------------------------------------------------------------------------------------------------------- |  | ----------------------------------------------------------------- |
|  | \[{\mathcal{L}_{\text{ce}} = {- {\sum\limits_{k = 1}^{\gamma}{w_{k}\hspace{0pt}{\log p_{k}^{d}}\hspace{0pt}{(x_{k}^{\ast})}}}}},\] |  | (9) |

where \(x_{k}^{\ast}\) is the ground-truth token and \(p_{k}^{d}\) is the draft distribution. The distribution-matching loss \(\mathcal{L}_{\text{tv}}\) penalizes the total variation distance between the draft and target distributions:

|  |                                                                                                                     |  |                                                                    |
|  | ------------------------------------------------------------------------------------------------------------------- |  | ------------------------------------------------------------------ |
|  | \[{\mathcal{L}_{\text{tv}} = {\sum\limits_{k = 1}^{\gamma}{w_{k}\hspace{0pt}{\|{p_{k}^{d} - p_{k}^{t}}\|}_{1}}}}.\] |  | (10) |

Since the total variation distance is a direct proxy for the acceptance rate: the per-step acceptance probability equals \(1 - {\frac{1}{2}\hspace{0pt}{\|{p^{d} - p^{t}}\|}_{1}}\) \[Leviathan et al., [2023](#bib.bib46)\], minimizing \(\mathcal{L}_{\text{tv}}\) directly maximizes the expected acceptance rate.

The confidence loss \(\mathcal{L}_{\text{conf}}\) is a binary cross-entropy that trains the confidence head to predict the soft acceptance label \(c_{k}^{\ast}\) from [Equation 8](#S3.E8 "8 ‣ 3.2.1 Confidence Head ‣ 3.2 Confidence-Scheduled Verification ‣ 3 Architecture ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation"):

|  |                                                                                                                                                                                                                         |  |                                                                    |
|  | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |  | ------------------------------------------------------------------ |
|  | \[{\mathcal{L}_{\text{conf}} = {- {\sum\limits_{k = 1}^{\gamma}{w_{k}\hspace{0pt}\left\lbrack {{c_{k}^{\ast}\hspace{0pt}{\log c_{k}}} + {{({1 - c_{k}^{\ast}})}\hspace{0pt}{\log{({1 - c_{k}})}}}} \right\rbrack}}}}.\] |  | (11) |

The overall objective is a weighted combination of the three terms (with default weights \(\alpha_{\text{ce}} = 0.1\), \(\alpha_{\text{tv}} = 0.9\), \(\alpha_{\text{conf}} = 1.0\)):

|  |                                                                                                                                                                                                     |  |                                                                    |
|  | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |  | ------------------------------------------------------------------ |
|  | \[\mathcal{L} = {{\alpha_{\text{ce}}\hspace{0pt}\mathcal{L}_{\text{ce}}} + {\alpha_{\text{tv}}\hspace{0pt}\mathcal{L}_{\text{tv}}} + {\alpha_{\text{conf}}\hspace{0pt}\mathcal{L}_{\text{conf}}}}\] |  | (12) |

## 4 Experiments
In this section, we validate the draft quality of DSpark using offline benchmarks and report the effectiveness of confidence scheduler under online production traffic in [Section 5](#S5 "5 Real-World Deployment of DSpark ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation"). The experimental setup is described in [Section 4.1](#S4.SS1 "4.1 Experimental Setup ‣ 4 Experiments ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation"), main results in [Section 4.2](#S4.SS2 "4.2 Experimental Results ‣ 4 Experiments ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation"), and additional analyses are included in [Section 4.3](#S4.SS3 "4.3 Experimental Analysis ‣ 4 Experiments ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation").

### 4.1 Experimental Setup
##### Target and draft models.
We evaluate DSpark on four target models spanning different scales and model families: Qwen3-{4B, 8B, 14B} \[Yang et al., [2025](#bib.bib97)\], and Gemma4-12B \[Google DeepMind, [2026](#bib.bib27)\]. For draft models, we compare DSpark with two representative drafters: DFlash \[Chen et al., [2026](#bib.bib12)\], a state-of-the-art parallel drafter, and Eagle3 \[Li et al., [2026b](#bib.bib54)\], an autoregressive drafter based on Training-Time Test (TTT). For strict and fair comparison, we retrain all drafters in the same [training framework](https://github.com/deepseek-ai/DeepSpec) and on the same data333To facilitate future research, we release all [checkpoints](https://huggingface.co/collections/deepseek-ai/deepspec) we trained, including Eagle3, DFlash and DSpark.. We align Eagle3’s TTT horizon (7) with the block size (7) used by DFlash and DSpark, and we use the same target-model feature layers for all drafters. For the number of draft model layers, we set 1 for Eagle3 and 5 for DSpark and DFlash \[Chen et al., [2026](#bib.bib12)\]. Unless otherwise stated, DSpark denotes the Markov-head variant; we study the RNN-head variant in [Section 4.3.2](#S4.SS3.SSS2 "4.3.2 A Little Autoregression Goes a Long Way ‣ 4.3 Experimental Analysis ‣ 4 Experiments ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation").

##### Training data.
We use [Open-PerfectBlend](https://huggingface.co/datasets/mlabonne/open-perfectblend), an open-sourced version of PerfectBlend \[Xu et al., [2024](#bib.bib95)\] consisting of 1.3 million samples. It is a general-purpose instruction dataset containing chat (17.6%), math (39.4%), code (38.9%), and instruction-following data (4.1%). We only use the prompts from Open-PerfectBlend; responses are regenerated by each target model with recommended sampling parameters. Each drafter is trained for 10 epochs to ensure full convergence. For data generation and evaluation, we adopt the non-thinking mode.

##### Evaluation protocol.
We evaluate the performance of different algorithms on three domains:

1.  1.
    
    Mathematical Reasoning, including GSM8K \[Cobbe et al., [2021](#bib.bib15)\], MATH500 \[Lightman et al., [2024](#bib.bib56)\] and AIME25 \[Zhang and Math-AI, [2025](#bib.bib103)\].
    
2.  2.
    
    Code Generation, including MBPP \[Austin et al., [2021b](#bib.bib8)\], HumanEval \[Chen et al., [2021](#bib.bib13)\] and Live-CodeBench \[Jain et al., [2025](#bib.bib43)\].
    
3.  3.
    
    Daily Chat, including MT-Bench \[Zheng et al., [2023](#bib.bib107)\], Alpaca \[Taori et al., [2023](#bib.bib82)\] and Arena-Hard \[Li et al., [2024a](#bib.bib49), [2025b](#bib.bib50)\].
    

For all benchmarks, we use standard speculative decoding \[Leviathan et al., [2023](#bib.bib46), Chen et al., [2023](#bib.bib11)\] with the sampling temperature set to \(1.0\). We report the accepted length (\(\tau\)) per decoding round444For clarity, unless otherwise stated, all reported metrics for accepted length and acceptance rate include the target-generated bonus token.. For all drafters, we use chain-based drafting.

Table 1: Main speculative decoding results. We report accepted length (\(\tau\)) per decoding round (higher is better) for different target models and domains. Bold marks the best results.

 

Target

Drafter

Math

Code

Chat

GSM8K

MATH

AIME25

MBPP

HumanEval

LCB

MT-Bench

Alpaca

Arena-Hard

Qwen3-4B

Eagle3

5.14

4.62

3.92

3.69

4.16

3.77

2.39

2.26

2.55

DFlash

5.40

4.85

4.15

4.40

4.74

4.18

3.07

2.96

2.83

DSpark

6.11

5.70

4.89

5.13

5.38

4.86

3.64

3.54

3.29

Qwen3-8B

Eagle3

5.30

4.77

3.91

3.96

4.33

4.17

2.66

2.54

2.54

DFlash

5.33

4.91

4.07

4.36

4.64

4.39

3.11

2.98

2.81

DSpark

6.17

5.78

5.01

5.16

5.52

5.17

3.72

3.58

3.21

Qwen3-14B

Eagle3

5.24

4.60

3.71

3.81

4.14

4.01

2.62

2.47

2.48

DFlash

5.41

4.84

3.98

4.44

4.59

4.33

3.10

2.94

2.72

DSpark

6.21

5.74

4.94

5.26

5.43

5.02

3.70

3.58

3.13

Gemma4-12B

Eagle3

5.87

5.46

4.83

4.72

5.37

4.16

3.19

3.06

2.72

DFlash

5.45

5.04

4.22

4.39

4.95

3.70

2.98

2.84

2.59

DSpark

6.05

5.78

5.12

5.11

5.64

4.51

3.49

3.35

2.92

### 4.2 Experimental Results
To isolate the raw draft quality from system-level scheduling policies, our offline evaluation disables the confidence scheduler, forcing all drafters to propose a fixed block of tokens. The main results, measured by the average accepted length (\(\tau\)) per round, are reported in [Table 1](#S4.T1 "Table 1 ‣ Evaluation protocol. ‣ 4.1 Experimental Setup ‣ 4 Experiments ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation").

DSpark consistently outperforms both the autoregressive baseline (Eagle3) and the parallel baseline (DFlash) across all evaluated target models and benchmark domains. Specifically, across the Qwen3-4B, 8B, and 14B models, DSpark improves the macro-average accepted length over Eagle3 by 30.9%, 26.7%, and 30.0%, respectively. Similarly, compared to DFlash, DSpark yields relative improvements of 16.3%, 18.4%, and 18.3% across the three scales. Crucially, this advantage generalizes across model families, as demonstrated by the consistent performance gains on the Gemma4-12B target.

Beyond the average improvements, [Table 1](#S4.T1 "Table 1 ‣ Evaluation protocol. ‣ 4.1 Experimental Setup ‣ 4 Experiments ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation") reveals a strong domain effect: the accepted length is naturally higher on structured tasks (e.g., 5.57 on math and 5.12 on code for Qwen3-4B) than on open-ended chat (3.49). This inherent variance in data predictability means a static verification length often wastes compute on trailing tokens that are highly likely to be rejected. This directly motivates our confidence-scheduled verification, which dynamically prunes the draft block based on expected acceptance.

### 4.3 Experimental Analysis
#### 4.3.1 Why Can Parallel Generation Outperform Autoregression?
![Figure 2: Position-wise conditional acceptance. We report the empirical conditional acceptance rate for each draft position, averaged across benchmarks within each domain using the Qwen3-4B target model. Unlike standard prefix survival, this metric isolates the baseline predictive quality at position \(k\) by removing the penalty of previous rejections. Notice that the autoregressive drafter (Eagle3) remains stable or trends upward, while the parallel drafter (DFlash) suffers suffix decay.](2607.05147v1/x2.png)

[Table 1](#S4.T1 "Table 1 ‣ Evaluation protocol. ‣ 4.1 Experimental Setup ‣ 4 Experiments ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation") presents a counter-intuitive observation: the parallel drafter (DFlash) and the semi-autoregressive drafter (DSpark) often yield longer accepted lengths than the fully autoregressive drafter (Eagle3). This finding contrasts with the standard expectation that step-by-step autoregression produces higher-quality sequences than parallel models \[Ren et al., [2020](#bib.bib69), Israel et al., [2026](#bib.bib42), Zheng et al., [2025](#bib.bib106)\].

To analyze this behavior, we examine performance beyond the macro-level accepted length. Using the Qwen3-4B target model and the benchmark sets described in [Section 4.1](#S4.SS1 "4.1 Experimental Setup ‣ 4 Experiments ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation"), we introduce position-wise conditional acceptance tracked during actual speculative decoding rollouts. Specifically, for a given draft position \(k\), the evaluation denominator counts only the instances where the target model successfully verifies and accepts all preceding draft tokens from \(1\) to \(k - 1\). The metric then calculates the proportion of these valid instances where the token at position \(k\) is also accepted. This approach ensures that the evaluation of position \(k\) is not penalized by earlier prefix errors, revealing the underlying predictive quality at each specific step. [Figure 2](#S4.F2 "Figure 2 ‣ 4.3.1 Why Can Parallel Generation Outperform Autoregression? ‣ 4.3 Experimental Analysis ‣ 4 Experiments ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation") details these measurements, demonstrating clear behavioral differences across the architectures.

##### The Capacity Advantage at Position 1.
At the first draft position, both architectures predict the next token based solely on the target context. The performance divergence here stems strictly from architectural capacity: autoregressive models like Eagle3 are constrained to shallow networks due to their \(O\hspace{0pt}{(\gamma)}\) latency, whereas \(O\hspace{0pt}{(1)}\) parallel drafters can afford much deeper networks. This structural gap yields a substantial accuracy margin at position 1, with DFlash starting noticeably higher than Eagle3 (e.g., 0.88 vs. 0.81 on Math, and 0.72 vs. 0.53 on Chat). Because speculative decoding operates as a strict prefix-matching survival process, the first token carries the highest leverage—a rejection here immediately invalidates the entire block. Consequently, this initial capacity advantage disproportionately boosts the final accepted length, explaining why parallel drafters ultimately outperform autoregressive ones globally despite rapid acceptance decay at later positions.

##### The Limitation of Independence at Later Positions.
Examining the tail of the curves (positions 2 through 7) exposes the inherent limitation of independent parallel generation. As earlier tokens lock in a specific semantic path, subsequent tokens naturally become more predictable. Autoregressive models like Eagle3 effectively leverage this conditional certainty, maintaining or even increasing conditional acceptance deeper into the block (e.g., from 0.53 to 0.74 on Chat). In contrast, DFlash suffers from rapid acceptance decay, dropping from 0.87 to 0.78 on Code and 0.72 to 0.63 on Chat. Because each parallel position marginalizes over all possible prior tokens rather than conditioning on an exact sampled prefix, the model frequently proposes inconsistent suffix combinations—a mode known as multi-modal collision \[Gu et al., [2018](#bib.bib28), Stern et al., [2018](#bib.bib79)\].

##### Mitigating Suffix Decay with Semi-Autoregression.
The preceding analysis highlights a clear architectural objective: combining the high capacity of a parallel backbone for the initial token with the dependency modeling of an autoregressive model for subsequent tokens. This directly motivates DSpark’s semi-autoregressive design. As shown in [Figure 2](#S4.F2 "Figure 2 ‣ 4.3.1 Why Can Parallel Generation Outperform Autoregression? ‣ 4.3 Experimental Analysis ‣ 4 Experiments ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation"), DSpark inherits the high initial acceptance of the deep parallel drafter (e.g., starting at 0.93 on Math). Simultaneously, its lightweight sequential head mitigates the rapid acceptance decay typical of parallel generation. By resolving this trade-off, DSpark maintains a high and stable conditional acceptance rate throughout the entire draft block.

#### 4.3.2 A Little Autoregression Goes a Long Way
Building on the insights from [Section 4.3.1](#S4.SS3.SSS1 "4.3.1 Why Can Parallel Generation Outperform Autoregression? ‣ 4.3 Experimental Analysis ‣ 4 Experiments ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation"), we explore the architectural design space of DSpark along two dimensions: drafter depth (number of transformer layers) and proposal length (block size \(\gamma\)). Unless otherwise stated, all experiments in this section use Qwen3-4B as the target model and follow the evaluation protocol detailed in [Section 4.1](#S4.SS1 "4.1 Experimental Setup ‣ 4 Experiments ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation").

![Figure 3: Effect of drafter depth. With proposal length fixed, DSpark’s performance improves as drafter layers are added. Notably, a shallow 2-layer DSpark outperforms a deeper 5-layer DFlash baseline, highlighting the parameter efficiency of sequential modeling.](2607.05147v1/x3.png)

![Figure 4: Effect of proposal length and latency overhead. DSpark consistently outperforms DFlash across various block sizes (left three panels). The rightmost panel demonstrates that the sequential head introduces minimal latency overhead during serving.](2607.05147v1/x4.png)

##### Drafter Depth.
Increasing the number of transformer layers naturally expands a draft model’s predictive capacity. To isolate this effect, we fix the block size to 7 and vary the number of DSpark layers from 1 to 5, comparing it against a 5-layer DFlash baseline. [Figure 3](#S4.F3 "Figure 3 ‣ 4.3.2 A Little Autoregression Goes a Long Way ‣ 4.3 Experimental Analysis ‣ 4 Experiments ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation") aggregates the accepted lengths across the math, code, and chat domains. As expected, DSpark’s performance improves monotonically with depth, with the steepest marginal gain occurring from one to two layers. Notably, a 2-layer DSpark outperforms the 5-layer DFlash baseline across all domains. This demonstrates that injecting local auto-regression via a lightweight sequential head offers a highly favorable accuracy-parameter trade-off, achieving better sequence coherence than simply stacking deeper parallel layers.

##### Proposal Length.
Next, we fix the drafter depth to 5 layers and scale the draft length (proposal length \(\gamma\) plus one anchor token) across \(\{ 4,8,12,16\}\) to evaluate performance on longer draft blocks. For DSpark, we evaluate both the default Markov head and the RNN head. The first three panels of [Figure 4](#S4.F4 "Figure 4 ‣ 4.3.2 A Little Autoregression Goes a Long Way ‣ 4.3 Experimental Analysis ‣ 4 Experiments ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation") show that DSpark consistently outperforms DFlash at every proposal length. More importantly, the performance gap steadily widens as \(\gamma\) increases. Because pure parallel generation (DFlash) suffers from rapid acceptance decay ([Figure 2](#S4.F2 "Figure 2 ‣ 4.3.1 Why Can Parallel Generation Outperform Autoregression? ‣ 4.3 Experimental Analysis ‣ 4 Experiments ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation")), its marginal utility diminishes for long blocks. DSpark mitigates this decay, causing its relative gain over DFlash to grow. For instance, at \(\gamma = 7\), DSpark improves the accepted length by 16% on math, 15% on code, and 18% on chat; at \(\gamma = 15\), these gains expand to 30%, 26%, and 22%, respectively. Also, RNN head provides only marginal additional gains over the Markov head, mainly at longer proposal lengths. Given its higher implementation complexity and less favorable deployment properties, we use the Markov head as the default.

##### Latency Overhead.
We quantify the overhead of the sequential generation loop in DSpark. The rightmost panel of [Figure 4](#S4.F4 "Figure 4 ‣ 4.3.2 A Little Autoregression Goes a Long Way ‣ 4.3 Experimental Analysis ‣ 4 Experiments ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation") reports the per-round engine latency—comprising one target verification pass, the parallel draft block forward, and the serial sampling loop—measured at a batch size of 128. To prevent sequence-length bias, the reported latency represents the arithmetic mean across varying context lengths (\({\{ 512,1024,2048,4096\}}\hspace{0pt}\text{tokens}\)). Since the target model dominates the verification compute time at this batch size, the sequential block’s latency overhead is negligible. Consequently, scaling the draft length from 4 to 16 adds a marginal 0.2% to 1.3% to the full-round latency over the DFlash baseline, despite delivering up to a 30% improvement in accepted length.

#### 4.3.3 Verify Smarter, Not Longer: The Role of Confidence Head
While DSpark sustains high acceptance over long draft blocks, verifying the entire proposal remains inefficient \[Huang et al., [2024](#bib.bib39), Hu et al., [2026b](#bib.bib34)\]. Due to the inherent domain variance noted in [Section 4.2](#S4.SS2 "4.2 Experimental Results ‣ 4 Experiments ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation"), trailing tokens in open-ended chat still face high rejection risks, making blind verification a waste of target compute. To evaluate whether the confidence head can effectively prune these unpromising suffixes, we conduct an offline threshold sweep using Qwen3-4B. We validate the estimator in isolation here, reserving the hardware-aware prefix scheduler ([Section 3.2.2](#S3.SS2.SSS2 "3.2.2 Hardware-Aware Prefix Scheduler ‣ 3.2 Confidence-Scheduled Verification ‣ 3 Architecture ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation")) for live production evaluation in [Section 5](#S5 "5 Real-World Deployment of DSpark ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation").

![Figure 5: Confidence threshold sweep. A threshold of 0 corresponds to standard fixed-length verification. As the threshold increases, the overall acceptance rate steadily rises because the confidence head effectively prunes tokens that would ultimately be rejected (hashed bars).](2607.05147v1/x5.png)

![Figure 6: The Reliability Diagram on Alpaca Dataset. While the raw confidence estimator achieves strong discrimination, its predictions are inherently overconfident. Applying post-hoc calibration helps to align the prefix survival probabilities with empirical acceptance rates. The shaded background histogram represents the frequency distribution of sample counts across different confidence bins.](2607.05147v1/x6.png)

##### Diagnostic: Static Threshold Sweep.
[Figure 5](#S4.F5 "Figure 5 ‣ 4.3.3 Verify Smarter, Not Longer: The Role of Confidence Head ‣ 4.3 Experimental Analysis ‣ 4 Experiments ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation") plots the average tokens per step (bars) and the overall acceptance rate (line) across confidence thresholds. As the threshold increases, the acceptance rate steadily rises because the estimator filters out tokens that would ultimately be rejected (hashed bars). This suggests that the confidence head can identify lower-value suffix tokens and this pruning is most pronounced on chat workloads, where higher-entropy token distributions limit the efficiency of fixed-length verification. In the Chat subplot, raising the threshold significantly reduces rejected tokens, increasing the acceptance rate from 45.7% to 95.7%. In contrast, structured tasks (Math and Code) experience milder pruning and retain more draft tokens, with acceptance rates rising from 76.9% to 92.5% and 67.6% to 92.0%, respectively.

##### From Static Thresholds to Calibrated Scheduling.
While useful for diagnostics, a static threshold is sub-optimal in dynamic serving environments because it ignores system load: verifying low-confidence tokens incurs minimal opportunity cost under low concurrency, but wastes critical batch capacity under high concurrency. This load dependency motivates the hardware-aware prefix scheduler. As formulated in [Section 3.2](#S3.SS2 "3.2 Confidence-Scheduled Verification ‣ 3 Architecture ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation"), maximizing system-level throughput requires the confidence model to exhibit both strong predictive discrimination and precise calibration to accurately estimate cumulative survival probabilities. The reliability diagram ([Figure 6](#S4.F6 "Figure 6 ‣ 4.3.3 Verify Smarter, Not Longer: The Role of Confidence Head ‣ 4.3 Experimental Analysis ‣ 4 Experiments ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation")) demonstrates that while the raw model achieves strong discrimination (ROC-AUC \[Hanley and McNeil, [1982](#bib.bib31)\] ranging from 0.81 to 0.90), it is overly confident (ECE 3%–8%). Applying post-hoc STS ([Section 3.2.1](#S3.SS2.SSS1 "3.2.1 Confidence Head ‣ 3.2 Confidence-Scheduled Verification ‣ 3 Architecture ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation")) mitigates this overconfidence, reducing the average ECE to \(\sim\)1% and yielding reliable survival estimates.

## 5 Real-World Deployment of DSpark
While [Section 4](#S4 "4 Experiments ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation") establishes the algorithmic gains of DSpark on offline benchmarks, deploying it alongside large-scale models like DeepSeek-V4 \[DeepSeek-AI, [2026](#bib.bib18)\] introduces additional system-level challenges across both training and inference. In this section, we present the end-to-end production pipeline of DSpark. We detail our scalable training mechanisms, the system-level optimizations necessary to deploy the hardware-aware prefix scheduler ([Section 3.2.2](#S3.SS2.SSS2 "3.2.2 Hardware-Aware Prefix Scheduler ‣ 3.2 Confidence-Scheduled Verification ‣ 3 Architecture ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation")), and the framework’s end-to-end performance under live user traffic.

### 5.1 Scalable and Flexible Training
The DSpark draft models are co-deployed with the preview versions of DeepSeek-V4-Flash and DeepSeek-V4-Pro \[DeepSeek-AI, [2026](#bib.bib18)\]. The parallel backbone comprises three MoE layers \[Dai et al., [2024](#bib.bib16)\] with mHC \[Xie et al., [2026](#bib.bib94)\] and a sliding window attention of 128. We configure the maximum block size to \(\gamma = 5\) and utilize the Markov head for sequential modeling. Furthermore, the confidence head is trained end-to-end alongside the draft model and subsequently calibrated via STS to provide reliable scheduling signals.

Training the draft model requires the target model’s output distributions for supervision. Evaluating both models over the full document context incurs substantial memory footprints and inter-worker communication overhead. To address these bottlenecks, we implement two system-level optimizations within our internal training framework (HAI-LLM)555:

  - •
    
    Hidden state communication. Transferring the target model’s full-vocabulary logits (\(V \approx 10^{5}\)) across parallel workers creates a significant bandwidth bottleneck. Instead, we temporarily cache the target model’s forward-pass activations and communicate only the hidden states immediately preceding the language modeling (LM) head. The LM head projection is then executed locally on the draft model’s workers only for the sampled target positions. This reduces the per-token communication complexity to \(O\hspace{0pt}{(d)}\), where \(d\) is the hidden dimension.
    
  - •
    
    Anchor-bounded sequence packing. To decouple the draft model’s computational cost from the target model’s context length, we sample a fixed number of draft anchors from the training sequence and pack these isolated prediction blocks into dense training batches. We manage this packing via token-level attention indices rather than standard 2D masks. This maintains exact causal masking across multiple independent sequences and anchors, avoiding the computational and memory overhead associated with standard padding.
    

### 5.2 Hardware-Aware Prefix Scheduler in Practice
In [Section 3.2.2](#S3.SS2.SSS2 "3.2.2 Hardware-Aware Prefix Scheduler ‣ 3.2 Confidence-Scheduled Verification ‣ 3 Architecture ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation"), [Algorithm 1](#alg1 "Algorithm 1 ‣ 3.2.2 Hardware-Aware Prefix Scheduler ‣ 3.2 Confidence-Scheduled Verification ‣ 3 Architecture ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation") provides a theoretically sound and lossless scheduling mechanism. However, directly deploying this algorithm into a production environment exposes two fundamental conflicts with real-world infrastructure. First, the algorithm assumes a smooth, unimodal capacity curve, whereas the true hardware capacity \(\text{SPS}\hspace{0pt}{(B)}\) is inherently discrete, exhibiting a jagged, step-wise degradation \[Yan et al., [2020](#bib.bib96)\]. Second, the algorithm requires scheduling of dynamic draft tokens per step, which clashes with continuous CUDA graph replay \[Fireworks AI, [2023](#bib.bib23)\] and Zero-Overhead Scheduling (ZOS) \[Zhu et al., [2025](#bib.bib110), Zheng et al., [2024](#bib.bib108)\].

To navigate the trade-offs among system compatibility, throughput, and algorithmic correctness, we adapt the scheduler to operate asynchronously. Because ZOS requires the batch size for the next step to be known before the current step completes, synchronous scheduling would inevitably stall the GPU pipeline. Instead, we approximate the upcoming verification capacity using the confidence head outputs from two steps prior. Mechanically, the candidate tokens in the current step are still strictly sorted by their actual, up-to-date cumulative confidence scores; the historical prediction from two steps prior is used solely to determine the dynamic truncation length (i.e., the batch capacity limit \(K\)). This effectively casts the admission process as a dynamic top-\(K\) selection. While approximating the capacity \(K\) introduces a slight temporal offset, the selection mechanism is fundamentally rank-preserving: the most confident draft tokens are always prioritized for verification. This adaptation fully hides scheduling latency and ensures seamless ZOS integration.

Building on this asynchronous pipeline, we resolve the hardware utilization bottleneck. To prevent the scheduler from being trapped in local minima by jagged SPS cliffs, we remove the early-stopping break, enabling an unconstrained global search. Ordinarily, this retrospective search would leak future token information and violate the lossless guarantee ([Appendix A](#A1 "Appendix A Counterexample: Selection Bias Without Early-Stopping ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation")). However, our ZOS-driven adaptation naturally prevents this. Because the unconstrained search evaluates only historical predictions from two steps prior, the admission decision is isolated from the realization of the current token \(x_{r,k}\). The truncation length inherently depends only on information available from two steps prior. Thus, asynchronous design forms a causal barrier, maximizing physical throughput across hardware cliffs while preserving the exact target distribution.

### 5.3 High-Throughput and Low-Latency Inference
During decoding, production serving systems must simultaneously optimize two competing objectives: per-request latency and aggregate throughput \[Kwon et al., [2023](#bib.bib45), Zhong et al., [2024](#bib.bib109), Zhao et al., [2025a](#bib.bib104)\]. The former governs the quality of service for individual users—a factor increasingly critical in agent-based workloads \[Tiwari et al., [2026](#bib.bib83)\]—while the latter determines the total number of concurrently served users. Because speculative decoding inevitably incurs wasted verification compute, it inherently navigates this trade-off, trading extra system compute for faster per-request generation.

In our deployment setting, however, the number of requests processed per step is frequently constrained by resource limits (e.g., fixed KV-cache capacity per request) and the pool of available user traffic (e.g., RL long-tail loads). Consequently, the effective batch size persistently remains well below the GPU’s compute-saturating threshold. Under this regime, the traditional trade-off simplifies: given a fixed concurrency limit, maximizing per-GPU total token throughput and maximizing the generation speed per user (tok/s/user) become highly correlated objectives rather than competing ones.

To achieve this maximum throughput, the asynchronous scheduler ([Section 5.2](#S5.SS2 "5.2 Hardware-Aware Prefix Scheduler in Practice ‣ 5 Real-World Deployment of DSpark ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation")) actively routes idle compute toward the most promising draft tokens. However, executing this dynamic routing introduces a severe challenge at the physical execution layer: the inference framework must efficiently support variable-length queries within a single batch. Standard decode kernels are heavily optimized for fixed query lengths; naively processing variable-length verified prefixes leads to severe GPU under-utilization due to padding and uneven workload distribution. We resolve this by decoupling physical execution from logical sequence tracking. In our compute kernels, all tokens across different requests are flattened and processed identically as independent elements. The complex intra-sequence dependencies are then strictly conveyed via a marker tensor integrated into our sparse attention implementation. Specifically on the DeepSeek-V4 architecture, only the index-attention and compress kernels require modification to support this variable-length routing, allowing the dynamic scheduler to operate seamlessly without introducing low-level execution overhead.

### 5.4 Performance under Live User Traffic
![Figure 7:  Throughput vs. TPS. Aggregate output token throughput against per-request generation speed (tok/s/user) under live traffic. In our production deployment, DSpark improves the observed throughput–interactivity frontier relative to the MTP-1 baseline under the measured traffic and engine configurations. ](2607.05147v1/x7.png)

We evaluate DSpark-5 (configured with a maximum draft length of \(\gamma = 5\)) against the MTP-1 \[DeepSeek-AI, [2024](#bib.bib17)\] baseline within the production serving engines of DeepSeek-V4-Flash (preview) and DeepSeek-V4-Pro (preview). MTP-1 represents the former production setup, having been superseded by DSpark two weeks following the DeepSeek-V4-preview release. This single-token setup was historically maintained in production because deploying a static multi-token drafter (e.g., MTP-3/5) strictly degrades aggregate throughput under high concurrency due to excessive verification overhead. Therefore, comparing DSpark against this established baseline directly demonstrates its ability to safely unlock the performance potential of larger draft blocks in dynamic serving environments. In all figures, the scatter points represent raw telemetry data sampled directly from live user traffic, capturing complex, real-world request distributions, while the solid lines represent the fitted performance frontiers.

##### The Serving Pareto Frontier.
[Figure 7](#S5.F7 "Figure 7 ‣ 5.4 Performance under Live User Traffic ‣ 5 Real-World Deployment of DSpark ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation") illustrates the trade-off between aggregate system throughput and per-user generation speed (interactivity). To quantify DSpark’s behavior under practical deployment constraints, we evaluate the system at several interactivity SLA anchors. Here, an SLA (Service Level Agreement) specifies the minimum per-user generation speed (in tokens per second) that the system must guarantee.

For the V4-Flash engine, we evaluate the system at SLA anchors of 80 and 120 tok/s/user. At the moderate 80 tok/s/user SLA, DSpark improves aggregate throughput by 51% over the MTP-1 baseline. The stricter 120 tok/s/user SLA represents a qualitatively different regime: under this constraint, the single-token MTP-1 baseline approaches its operational boundary and can sustain only a very small concurrent batch. Consequently, the relative throughput ratio at this point is numerically large, with DSpark achieving a nominal 661% higher aggregate throughput. We therefore interpret this high-SLA point primarily as evidence that DSpark extends the feasible interactivity frontier, rather than as a representative multiplicative speedup over a well-utilized baseline. At matched practical throughput levels, which provide a more stable comparison, DSpark accelerates per-user generation speeds by 60% to 85%.

The V4-Pro deployment shows the same pattern. At the moderate 35 tok/s/user SLA, DSpark improves aggregate throughput by 52%. At the stricter 50 tok/s/user SLA, MTP-1 again enters a low-concurrency regime, yielding a nominal 406% relative throughput advantage for DSpark. As with V4-Flash, we treat this point as an indication that DSpark sustains useful throughput under an interactivity target that the baseline cannot efficiently support. At matched system capacities, DSpark delivers 57% to 78% faster per-user generation. Overall, these results show that DSpark shifts the observed throughput–interactivity frontier outward: it improves throughput in moderate-SLA regimes and, more importantly, preserves non-degenerate serving capacity under strict interactivity constraints.

##### Throughput Dynamics under Load.
![Figure 8: Load-adaptive throughput and verification budgets. Top row (a, b): Aggregate output throughput across varying levels of system concurrency. Bottom row (c, d): The average target verification budget allocated per request. As concurrent load increases, the dynamic scheduler automatically restricts the per-request verification length to prevent resource contention.](2607.05147v1/x8.png)

[Figure 8](#S5.F8 "Figure 8 ‣ Throughput Dynamics under Load. ‣ 5.4 Performance under Live User Traffic ‣ 5 Real-World Deployment of DSpark ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation") analyzes the underlying mechanism driving these gains by plotting aggregate throughput (top row) and the dynamic verification budget (bottom row) against system concurrency.

  - •
    
    Under the moderate concurrency regimes typical of our production deployment (fewer than 200 concurrent requests for V4-Flash and 150 for V4-Pro), the hardware-aware scheduler leverages available target compute capacity by allocating longer verification budgets, expanding from MTP-1’s static 2 tokens to roughly 4–6 tokens per request. This extended verification yields more accepted tokens per forward pass, directly contributing to the throughput gains observed on the Pareto frontier.
    
  - •
    
    As system concurrency scales and target capacity saturates, the scheduler dynamically restricts this budget. The average verification length decreases smoothly with load, ensuring that low-confidence draft tokens are pruned before they consume critical batch capacity. This load-aware behavior stabilizes production deployment: DSpark maximizes the utility of idle compute under light traffic, while effectively preserving critical batch capacity under heavy traffic.
    

##### Limitations.
Although the prefix scheduler minimizes wasted target-model verification, DSpark still incurs a fixed draft-side cost to generate the initial \(\gamma\)-token block via the parallel backbone. For complex queries with inherently low acceptance rates, this upfront drafting compute is unrecoverable. Future optimizations could introduce difficulty-aware early exiting within the draft model, enabling such requests to bypass full-block generation.

## 6 Related Work
##### Speculative Decoding Algorithms.
Speculative decoding accelerates autoregressive generation by decoupling token proposal from verification. Building on early blockwise methods \[Stern et al., [2018](#bib.bib79), Sun et al., [2021](#bib.bib80), Ge et al., [2022](#bib.bib24), Xia et al., [2023](#bib.bib91)\], modern approaches employ rejection sampling to exactly preserve the target model’s distribution \[Chen et al., [2023](#bib.bib11), Leviathan et al., [2023](#bib.bib46)\]. Because inference speedup directly depends on the drafter’s efficiency and accuracy, extensive research has focused on optimizing its architecture. Beyond using standalone small language models \[Chen et al., [2023](#bib.bib11), Leviathan et al., [2023](#bib.bib46)\], subsequent work integrates multi-token heads or feature extrapolators directly into the target model \[Cai et al., [2024](#bib.bib9), Ankner et al., [2024](#bib.bib5), Li et al., [2024c](#bib.bib53), [b](#bib.bib52), [2026b](#bib.bib54), Zhang et al., [2025](#bib.bib101), Gloeckle et al., [2024](#bib.bib26), DeepSeek-AI, [2024](#bib.bib17), Cai et al., [2025](#bib.bib10), Eldenk et al., [2026](#bib.bib21)\]. Other strategies include self-speculation via early exits \[Zhang et al., [2024](#bib.bib99), Elhoushi et al., [2024](#bib.bib22), Liu et al., [2024a](#bib.bib57), Xia et al., [2025](#bib.bib93)\], dynamic vocabulary compression \[Zhao et al., [2025b](#bib.bib105), Williams et al., [2026](#bib.bib89)\], prompt lookup \[Saxena, [2023](#bib.bib74), Somasundaram et al., [2025](#bib.bib78)\], suffix automata \[Hu et al., [2025](#bib.bib35)\], and retrieval \[He et al., [2023](#bib.bib32), Shen et al., [2026](#bib.bib77)\]. To remove the sequential bottleneck of the drafting itself, a line of research proposes parallel or blockwise generation, including Medusa \[Cai et al., [2024](#bib.bib9)\], P-EAGLE \[Hui et al., [2026](#bib.bib41)\], PARD \[An et al., [2026a](#bib.bib2), [b](#bib.bib3)\], DART \[Liu et al., [2026a](#bib.bib58)\] and DFlash \[Chen et al., [2026](#bib.bib12)\]. DDTree, TAPS and JetSpec then extend the draft chain to verifiable trees \[Ringel and Romano, [2026](#bib.bib70), Wang et al., [2026a](#bib.bib85), Hu et al., [2026a](#bib.bib33)\]. Concurrent efforts include: Domino \[Huang et al., [2026a](#bib.bib38)\] introduces a CausalEncoder conceptually similar to our RNN Head; DFlare \[Zhang et al., [2026a](#bib.bib100)\] addresses conditioning bottlenecks via layer-wise fusion.

##### System-Aware Scheduling for Speculative Decoding.
Beyond drafter architecture, another line of work focuses on determining the optimal number of speculative tokens to generate or verify in each round. To this end, various approaches adapt draft lengths on the fly using confidence heuristics \[Mamou et al., [2024](#bib.bib64), Li et al., [2024b](#bib.bib52), Du et al., [2024](#bib.bib20), Liu et al., [2026c](#bib.bib60), Wen and Feng, [2026](#bib.bib87)\], learned acceptance predictors \[Huang et al., [2024](#bib.bib39), Zacks917, [2026](#bib.bib98)\], or bandit-style policies \[Liu et al., [2026b](#bib.bib59)\]. Furthermore, recognizing speculative decoding as inherently a system-level scheduling problem, recent works optimize overall goodput and latency by adjusting speculation budgets according to real-time system load and request priority \[AngelSlim Team, [2026](#bib.bib4), Liu et al., [2024c](#bib.bib62), Huang et al., [2026b](#bib.bib40), Li et al., [2026a](#bib.bib48), Wu et al., [2025](#bib.bib90), Hu et al., [2026b](#bib.bib34), Miao et al., [2024](#bib.bib65), Sadhukhan et al., [2025](#bib.bib71)\].

##### Parallel Generation.
Models that generate tokens in parallel offer a decoding latency nearly independent of output length, making them an attractive alternative to autoregressive decoding. Non-Autoregressive Transformers \[NATs, Gu et al., [2018](#bib.bib28)\] pioneered this direction by predicting all positions independently in a single pass. However, this forces the model to average over all plausible modes, often producing outputs that mix fragments from different valid sequences. Two broad lines of work have emerged to address this limitation. One direction retains the single-pass architecture but changes what the model sees or how it is trained: introducing latent variables as conditioning input to steer all positions toward a consistent output \[Gu et al., [2018](#bib.bib28), Kaiser et al., [2018](#bib.bib44), Ma et al., [2019](#bib.bib63)\], or relaxing the training objective so that the model focuses on producing a single coherent output rather than modeling the full distribution over all valid alternatives \[Shao et al., [2021](#bib.bib75), Du et al., [2021](#bib.bib19), Qian et al., [2021](#bib.bib68), Shao et al., [2023](#bib.bib76)\]. The other direction reintroduces limited sequential dependency through iterative re-prediction \[Ghazvininejad et al., [2019](#bib.bib25), Austin et al., [2021a](#bib.bib7), Li et al., [2022](#bib.bib51)\], block-level autoregression \[Wang et al., [2018](#bib.bib84), Arriola et al., [2025](#bib.bib6)\], or structured output layers such as CRF \[Sun et al., [2019](#bib.bib81)\], CTC \[Libovický and Helcl, [2018](#bib.bib55), Saharia et al., [2020](#bib.bib72)\], HMM \[Huang et al., [2022b](#bib.bib37)\], and PCFG \[Gui et al., [2023](#bib.bib29)\].

Speculative decoding places a further demand that the drafter must provide exact per-token probabilities for the rejection sampling rule. Most techniques above cannot readily provide such probabilities due to iterative refinement, latent marginalization, or global normalization. For instance, in a design closely related to ours, CRF-NAT \[Sun et al., [2019](#bib.bib81)\] also places a sequential module over parallel hidden states, but its globally normalized partition function prevents exact per-token probability computation. Similarly, when adapting the CTC output layer to parallel speculative decoding, CTC-drafter \[Wen et al., [2024](#bib.bib88)\] is restricted to greedy verification due to the latent marginalization of alignment paths. DSpark circumvents these limitations by keeping the sequential correction local, so per-token probabilities remain exact softmax evaluations.

## 7 Conclusion
In this paper, we present DSpark, a speculative decoding framework designed to overcome the structural and system-level bottlenecks of large language model inference in high-concurrency production environments. Algorithmically, DSpark introduces a semi-autoregressive generation paradigm—coupling a computationally heavy parallel backbone with a lightweight sequential head—to mitigate the rapid suffix decay of independent parallel drafters. At the system level, we formulate verification length selection as a global throughput maximization problem, employing a hardware-aware prefix scheduler that dynamically tailors the target model’s verification budget based on calibrated survival probabilities and real-time engine load. Extensive offline evaluations demonstrate that DSpark substantially outperforms state-of-the-art autoregressive and parallel baselines across diverse domains. Furthermore, its real-world deployment within the DeepSeek-V4 validates its practical value in production serving: by intelligently managing verification overhead, DSpark sustains robust concurrency under heavy load, consistently accelerates per-user generation speeds, and shifts the Pareto frontier of LLM serving outward.

## References
  - Abramovich et al. \[2026\]  T. Abramovich, M. Ashkenazi, I. Putterman, B. Chislett, T. Mitra, B. D. Rouhani, R. Zilberstein, and Y. Geifman.  Speed-bench: A unified and diverse benchmark for speculative decoding.  *arXiv preprint arXiv:2604.09557*, 2026. 
  - An et al. \[2026a\]  Z. An, H. Bai, Z. Liu, D. Li, and E. Barsoum.  PARD: Accelerating LLM inference with low-cost PARallel draft model adaptation.  In *The Fourteenth International Conference on Learning Representations*, 2026a.  URL . 
  - An et al. \[2026b\]  Z. An, T. Liu, Z. Liu, D. Li, R. Liu, and E. Barsoum.  Pard-2: Target-aligned parallel draft model for dual-mode speculative decoding.  *arXiv preprint arXiv:2605.08632*, 2026b. 
  - AngelSlim Team \[2026\]  AngelSlim Team.  D-Cut: Adaptive verification depth pruning for speculative decoding, 2026.  URL . 
  - Ankner et al. \[2024\]  Z. Ankner, R. Parthasarathy, A. Nrusimha, C. Rinard, J. Ragan-Kelley, and W. Brandon.  Hydra: Sequentially-dependent draft heads for medusa decoding.  In *First Conference on Language Modeling*, 2024.  URL . 
  - Arriola et al. \[2025\]  M. Arriola, S. S. Sahoo, A. Gokaslan, Z. Yang, Z. Qi, J. Han, J. T. Chiu, and V. Kuleshov.  Block diffusion: Interpolating between autoregressive and diffusion language models.  In *The Thirteenth International Conference on Learning Representations*, 2025.  URL . 
  - Austin et al. \[2021a\]  J. Austin, D. D. Johnson, J. Ho, D. Tarlow, and R. van den Berg.  Structured denoising diffusion models in discrete state-spaces.  In A. Beygelzimer, Y. Dauphin, P. Liang, and J. W. Vaughan, editors, *Advances in Neural Information Processing Systems*, 2021a.  URL . 
  - Austin et al. \[2021b\]  J. Austin, A. Odena, M. Nye, M. Bosma, H. Michalewski, D. Dohan, E. Jiang, C. Cai, M. Terry, Q. Le, et al.  Program synthesis with large language models.  *arXiv preprint arXiv:2108.07732*, 2021b. 
  - Cai et al. \[2024\]  T. Cai, Y. Li, Z. Geng, H. Peng, J. D. Lee, D. Chen, and T. Dao.  Medusa: Simple LLM inference acceleration framework with multiple decoding heads.  In R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett, and F. Berkenkamp, editors, *Proceedings of the 41st International Conference on Machine Learning*, volume 235 of *Proceedings of Machine Learning Research*, pages 5209–5235. PMLR, 21–27 Jul 2024.  URL . 
  - Cai et al. \[2025\]  Y. Cai, X. Liang, X. Wang, J. Ma, H. Liang, J. Luo, X. Zuo, L. Duan, Y. Yin, and X. Chen.  Fastmtp: Accelerating llm inference with enhanced multi-token prediction, 2025.  URL . 
  - Chen et al. \[2023\]  C. Chen, S. Borgeaud, G. Irving, J.-B. Lespiau, L. Sifre, and J. Jumper.  Accelerating large language model decoding with speculative sampling.  *arXiv preprint arXiv:2302.01318*, 2023. 
  - Chen et al. \[2026\]  J. Chen, Y. Liang, and Z. Liu.  Dflash: Block diffusion for flash speculative decoding.  *arXiv preprint arXiv:2602.06036*, 2026. 
  - Chen et al. \[2021\]  M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. Knight, M. Brundage, M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and W. Zaremba.  Evaluating large language models trained on code, 2021. 
  - Cheng et al. \[2024\]  Y. Cheng, A. Zhang, X. Zhang, C. Wang, and Y. Wang.  Recurrent drafter for fast speculative decoding in large language models, 2024.  URL . 
  - Cobbe et al. \[2021\]  K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman.  Training verifiers to solve math word problems.  *arXiv preprint arXiv:2110.14168*, 2021. 
  - Dai et al. \[2024\]  D. Dai, C. Deng, C. Zhao, R. Xu, H. Gao, D. Chen, J. Li, W. Zeng, X. Yu, Y. Wu, et al.  Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models.  In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1280–1297, 2024. 
  - DeepSeek-AI \[2024\]  DeepSeek-AI.  Deepseek-v3 technical report.  *arXiv preprint arXiv:2412.19437*, 2024. 
  - DeepSeek-AI \[2026\]  DeepSeek-AI.  Deepseek-v4: Towards highly efficient million-token context intelligence, 2026.  URL . 
  - Du et al. \[2021\]  C. Du, Z. Tu, and J. Jiang.  Order-agnostic cross entropy for non-autoregressive machine translation.  In M. Meila and T. Zhang, editors, *Proceedings of the 38th International Conference on Machine Learning*, volume 139 of *Proceedings of Machine Learning Research*, pages 2849–2859. PMLR, 18–24 Jul 2021.  URL . 
  - Du et al. \[2024\]  C. Du, J. Jiang, X. Yuanchen, J. Wu, S. Yu, Y. Li, S. Li, K. Xu, L. Nie, Z. Tu, and Y. You.  GliDe with a CaPE: A low-hassle method to accelerate speculative decoding.  In R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett, and F. Berkenkamp, editors, *Proceedings of the 41st International Conference on Machine Learning*, volume 235 of *Proceedings of Machine Learning Research*, pages 11704–11720. PMLR, 21–27 Jul 2024.  URL . 
  - Eldenk et al. \[2026\]  D. Eldenk, P. Mohapatra, Y. Comlek, K. Oktay, H. Zhang, and S. Xia.  Attention drift: What autoregressive speculative decoding models learn, 2026.  URL . 
  - Elhoushi et al. \[2024\]  M. Elhoushi, A. Shrivastava, D. Liskovich, B. Hosmer, B. Wasti, L. Lai, A. Mahmoud, B. Acun, S. Agarwal, A. Roman, A. Aly, B. Chen, and C.-J. Wu.  Layerskip: Enabling early exit inference and self-speculative decoding.  In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, page 12622–12642. Association for Computational Linguistics, 2024.  [10.18653/v1/2024.acl-long.681](https:/doi.org/10.18653/v1/2024.acl-long.681).  URL . 
  - Fireworks AI \[2023\]  Fireworks AI.  Speed, Python: Pick Two. How CUDA Graphs Enable Fast Python Code for Deep Learning.  , Aug. 2023.  Accessed: 2026-06-22. 
  - Ge et al. \[2022\]  T. Ge, H. Xia, X. Sun, S.-Q. Chen, and F. Wei.  Lossless acceleration for seq2seq generation with aggressive decoding.  *arXiv preprint arXiv:2205.10350*, 2022. 
  - Ghazvininejad et al. \[2019\]  M. Ghazvininejad, O. Levy, Y. Liu, and L. Zettlemoyer.  Mask-predict: Parallel decoding of conditional masked language models.  In K. Inui, J. Jiang, V. Ng, and X. Wan, editors, *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 6112–6121, Hong Kong, China, Nov. 2019. Association for Computational Linguistics.  [10.18653/v1/D19-1633](https:/doi.org/10.18653/v1/D19-1633).  URL . 
  - Gloeckle et al. \[2024\]  F. Gloeckle, B. Youbi Idrissi, B. Roziere, D. Lopez-Paz, and G. Synnaeve.  Better & faster large language models via multi-token prediction.  In R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett, and F. Berkenkamp, editors, *Proceedings of the 41st International Conference on Machine Learning*, volume 235 of *Proceedings of Machine Learning Research*, pages 15706–15734. PMLR, 21–27 Jul 2024.  URL . 
  - Google DeepMind \[2026\]  Google DeepMind.  Gemma 4 model card.  , 2026.  Accessed: 2026-06-11. 
  - Gu et al. \[2018\]  J. Gu, J. Bradbury, C. Xiong, V. O. Li, and R. Socher.  Non-autoregressive neural machine translation.  In *International Conference on Learning Representations*, 2018.  URL . 
  - Gui et al. \[2023\]  S. Gui, C. Shao, Z. Ma, X. Zhang, Y. Chen, and Y. Feng.  Non-autoregressive machine translation with probabilistic context-free grammar.  In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023.  URL . 
  - Guo et al. \[2017\]  C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger.  On calibration of modern neural networks.  In *International conference on machine learning*, pages 1321–1330. PMLR, 2017. 
  - Hanley and McNeil \[1982\]  J. A. Hanley and B. J. McNeil.  The meaning and use of the area under a receiver operating characteristic (roc) curve.  *Radiology*, 143(1):29–36, 1982.  [10.1148/radiology.143.1.7063747](https:/doi.org/10.1148/radiology.143.1.7063747). 
  - He et al. \[2023\]  Z. He, Z. Zhong, T. Cai, J. D. Lee, and D. He.  Rest: Retrieval-based speculative decoding, 2023. 
  - Hu et al. \[2026a\]  L. Hu, Z. Feng, Y. Wu, H. Yuan, Y. Zhao, Y.-Y. Qian, B. Wang, P. Zhao, D. Jiang, Y. Zhu, T. Rosing, and H. Zhang.  Jetspec: Breaking the scaling ceiling of speculative decoding with parallel tree drafting, 2026a.  URL . 
  - Hu et al. \[2026b\]  X. Hu, Y. Shen, B. Zhang, H. Zhang, J. Dai, S. Ge, L. Chen, Y. Li, and M. Wan.  Echo: Elastic speculative decoding with sparse gating for high-concurrency scenarios.  *arXiv preprint arXiv:2604.09603*, 2026b. 
  - Hu et al. \[2025\]  Y. Hu, K. Wang, X. Zhang, F. Zhang, C. Li, H. Chen, and J. Zhang.  SAM decoding: Speculative decoding via suffix automaton.  In W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, editors, *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 12187–12204, Vienna, Austria, July 2025. Association for Computational Linguistics.  ISBN 979-8-89176-251-0.  [10.18653/v1/2025.acl-long.595](https:/doi.org/10.18653/v1/2025.acl-long.595).  URL . 
  - Huang et al. \[2022a\]  F. Huang, T. Tao, H. Zhou, L. Li, and M. Huang.  On the learning of non-autoregressive transformers.  In K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvari, G. Niu, and S. Sabato, editors, *Proceedings of the 39th International Conference on Machine Learning*, volume 162 of *Proceedings of Machine Learning Research*, pages 9356–9376. PMLR, 17–23 Jul 2022a.  URL . 
  - Huang et al. \[2022b\]  F. Huang, H. Zhou, Y. Liu, H. Li, and M. Huang.  Directed acyclic transformer for non-autoregressive machine translation.  In K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvari, G. Niu, and S. Sabato, editors, *Proceedings of the 39th International Conference on Machine Learning*, volume 162 of *Proceedings of Machine Learning Research*, pages 9410–9428. PMLR, 17–23 Jul 2022b.  URL . 
  - Huang et al. \[2026a\]  J. Huang, Y. Zhang, Q. Zhang, H. Lin, H. Xu, and L. Zhang.  Domino: Decoupling causal modeling from autoregressive drafting in speculative decoding, 2026a.  URL . 
  - Huang et al. \[2024\]  K. Huang, X. Guo, and M. Wang.  Specdec++: Boosting speculative decoding via adaptive candidate lengths.  *arXiv preprint arXiv:2405.19715*, 2024. 
  - Huang et al. \[2026b\]  K. Huang, H. Wu, Z. Shi, H. Zou, M. Yu, and Q. Shi.  Adaspec: Adaptive speculative decoding for fast, slo-aware large language model serving.  In *Proceedings of the 2025 ACM Symposium on Cloud Computing*, SoCC ’25, page 361–374, New York, NY, USA, 2026b. Association for Computing Machinery.  ISBN 9798400722769.  [10.1145/3772052.3772239](https:/doi.org/10.1145/3772052.3772239).  URL . 
  - Hui et al. \[2026\]  M. Hui, X. Huang, J. C. Salas, Y. Sun, N. Pemberton, X. Song, A. Khetan, and G. Karypis.  P-eagle: Parallel-drafting eagle with scalable training, 2026.  URL . 
  - Israel et al. \[2026\]  D. Israel, G. Van den Broeck, and A. Grover.  Accelerating diffusion llms via adaptive parallel decoding.  *Advances in neural information processing systems*, 38:52870–52888, 2026. 
  - Jain et al. \[2025\]  N. Jain, A. Gu, W.-D. Li, F. Yan, T. Zhang, S. Wang, A. Solar-Lezama, K. Sen, and I. Stoica.  Livecodebench: Holistic and contamination free evaluation of large language models for code.  In *International Conference on Learning Representations*, volume 2025, pages 58791–58831, 2025. 
  - Kaiser et al. \[2018\]  L. Kaiser, S. Bengio, A. Roy, A. Vaswani, N. Parmar, J. Uszkoreit, and N. Shazeer.  Fast decoding in sequence models using discrete latent variables.  In J. Dy and A. Krause, editors, *Proceedings of the 35th International Conference on Machine Learning*, volume 80 of *Proceedings of Machine Learning Research*, pages 2390–2399. PMLR, 10–15 Jul 2018.  URL . 
  - Kwon et al. \[2023\]  W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. Gonzalez, H. Zhang, and I. Stoica.  Efficient memory management for large language model serving with pagedattention.  In *Proceedings of the 29th symposium on operating systems principles*, pages 611–626, 2023. 
  - Leviathan et al. \[2023\]  Y. Leviathan, M. Kalman, and Y. Matias.  Fast inference from transformers via speculative decoding.  In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, *Proceedings of the 40th International Conference on Machine Learning*, volume 202 of *Proceedings of Machine Learning Research*, pages 19274–19286. PMLR, 23–29 Jul 2023.  URL . 
  - Li et al. \[2025a\]  G. Li, Z. Fu, M. Fang, Q. Zhao, M. Tang, C. Yuan, and J. Wang.  Diffuspec: Unlocking diffusion language models for speculative decoding, 2025a.  URL . 
  - Li et al. \[2026a\]  R. Li, Z. Zhang, L. Zhang, H. Wang, X. Fu, and Z. Lai.  Nightjar: Dynamic adaptive speculative decoding for large language models serving, 2026a.  URL . 
  - Li et al. \[2024a\]  T. Li, W.-L. Chiang, E. Frick, L. Dunlap, B. Zhu, J. E. Gonzalez, and I. Stoica.  From live data to high-quality benchmarks: The arena-hard pipeline, April 2024a.  URL . 
  - Li et al. \[2025b\]  T. Li, W.-L. Chiang, E. Frick, L. Dunlap, T. Wu, B. Zhu, J. E. Gonzalez, and I. Stoica.  From crowdsourced data to high-quality benchmarks: Arena-hard and benchbuilder pipeline.  In A. Singh, M. Fazel, D. Hsu, S. Lacoste-Julien, F. Berkenkamp, T. Maharaj, K. Wagstaff, and J. Zhu, editors, *Proceedings of the 42nd International Conference on Machine Learning*, volume 267 of *Proceedings of Machine Learning Research*, pages 34209–34231. PMLR, 13–19 Jul 2025b.  URL . 
  - Li et al. \[2022\]  X. L. Li, J. Thickstun, I. Gulrajani, P. Liang, and T. Hashimoto.  Diffusion-LM improves controllable text generation.  In A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho, editors, *Advances in Neural Information Processing Systems*, 2022.  URL . 
  - Li et al. \[2024b\]  Y. Li, F. Wei, C. Zhang, and H. Zhang.  EAGLE-2: Faster inference of language models with dynamic draft trees.  In Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, editors, *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, pages 7421–7432, Miami, Florida, USA, Nov. 2024b. Association for Computational Linguistics.  [10.18653/v1/2024.emnlp-main.422](https:/doi.org/10.18653/v1/2024.emnlp-main.422).  URL . 
  - Li et al. \[2024c\]  Y. Li, F. Wei, C. Zhang, and H. Zhang.  EAGLE: Speculative sampling requires rethinking feature uncertainty.  In R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett, and F. Berkenkamp, editors, *Proceedings of the 41st International Conference on Machine Learning*, volume 235 of *Proceedings of Machine Learning Research*, pages 28935–28948. PMLR, 21–27 Jul 2024c.  URL . 
  - Li et al. \[2026b\]  Y. Li, F. Wei, C. Zhang, and H. Zhang.  EAGLE-3: Scaling up inference acceleration of large language models via training-time test.  In *The Thirty-ninth Annual Conference on Neural Information Processing Systems*, 2026b.  URL . 
  - Libovický and Helcl \[2018\]  J. Libovický and J. Helcl.  End-to-end non-autoregressive neural machine translation with connectionist temporal classification.  In E. Riloff, D. Chiang, J. Hockenmaier, and J. Tsujii, editors, *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 3016–3021, Brussels, Belgium, Oct.-Nov. 2018. Association for Computational Linguistics.  [10.18653/v1/D18-1336](https:/doi.org/10.18653/v1/D18-1336).  URL . 
  - Lightman et al. \[2024\]  H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker, T. Lee, J. Leike, J. Schulman, I. Sutskever, and K. Cobbe.  Let’s verify step by step.  In *The Twelfth International Conference on Learning Representations*, 2024.  URL . 
  - Liu et al. \[2024a\]  F. Liu, Y. Tang, Z. Liu, Y. Ni, D. Tang, K. Han, and Y. Wang.  Kangaroo: Lossless self-speculative decoding for accelerating llms via double early exiting.  In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, *Advances in Neural Information Processing Systems*, volume 37, pages 11946–11965. Curran Associates, Inc., 2024a.  [10.52202/079017-0381](https:/doi.org/10.52202/079017-0381).  URL . 
  - Liu et al. \[2026a\]  F. Liu, X. Li, K. Zhao, Y. Gao, Z. Zhou, Z. Zhang, Z. Wang, W. Dou, S. Zhong, and C. Tian.  Dart: Diffusion-inspired speculative decoding for fast llm inference, 2026a.  URL . 
  - Liu et al. \[2026b\]  H. Liu, J. Huang, Z. Jia, Y. Park, and Y.-X. Wang.  Not-a-bandit: Provably no-regret drafter selection in speculative decoding for llms, 2026b.  URL . 
  - Liu et al. \[2026c\]  T. Liu, Q. Lv, Y. Shen, X. Sun, and X. Sun.  Talon: Confidence-aware speculative decoding with adaptive token trees.  *arXiv preprint arXiv:2601.07353*, 2026c. 
  - Liu et al. \[2024b\]  X. Liu, C. Daniel, L. Hu, W. Kwon, Z. Li, X. Mo, A. Cheung, Z. Deng, I. Stoica, and H. Zhang.  Optimizing speculative decoding for serving large language models using goodput.  *arXiv e-prints*, pages arXiv–2406, 2024b. 
  - Liu et al. \[2024c\]  X. Liu, J. Park, L. Hu, W. Kwon, Z. Li, C. Zhang, K. Du, X. Mo, K. You, A. Cheung, et al.  Turbospec: Closed-loop speculation control system for optimizing llm serving goodput.  *arXiv preprint arXiv:2406.14066*, 2024c. 
  - Ma et al. \[2019\]  X. Ma, C. Zhou, X. Li, G. Neubig, and E. Hovy.  FlowSeq: Non-autoregressive conditional sequence generation with generative flow.  In K. Inui, J. Jiang, V. Ng, and X. Wan, editors, *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 4282–4292, Hong Kong, China, Nov. 2019. Association for Computational Linguistics.  [10.18653/v1/D19-1437](https:/doi.org/10.18653/v1/D19-1437).  URL . 
  - Mamou et al. \[2024\]  J. Mamou, O. Pereg, D. Korat, M. Berchansky, N. Timor, M. Wasserblat, and R. Schwartz.  Dynamic speculation lookahead accelerates speculative decoding of large language models.  In M. Rezagholizadeh, P. Passban, S. Samiee, V. Partovi Nia, Y. Cheng, Y. Deng, Q. Liu, and B. Chen, editors, *Proceedings of The 4th NeurIPS Efficient Natural Language and Speech Processing Workshop*, volume 262 of *Proceedings of Machine Learning Research*, pages 456–467. PMLR, 14 Dec 2024.  URL . 
  - Miao et al. \[2024\]  X. Miao, G. Oliaro, Z. Zhang, X. Cheng, Z. Wang, Z. Zhang, R. Y. Y. Wong, A. Zhu, L. Yang, X. Shi, et al.  Specinfer: Accelerating large language model serving with tree-based speculative inference and verification.  In *Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3*, pages 932–949, 2024. 
  - Naeini et al. \[2015\]  M. P. Naeini, G. Cooper, and M. Hauskrecht.  Obtaining well calibrated probabilities using bayesian binning.  In *Proceedings of the AAAI conference on artificial intelligence*, volume 29, 2015. 
  - Ovadia et al. \[2019\]  Y. Ovadia, E. Fertig, J. Ren, Z. Nado, D. Sculley, S. Nowozin, J. Dillon, B. Lakshminarayanan, and J. Snoek.  Can you trust your model’s uncertainty? evaluating predictive uncertainty under dataset shift.  *Advances in neural information processing systems*, 32, 2019. 
  - Qian et al. \[2021\]  L. Qian, H. Zhou, Y. Bao, M. Wang, L. Qiu, W. Zhang, Y. Yu, and L. Li.  Glancing transformer for non-autoregressive neural machine translation.  In C. Zong, F. Xia, W. Li, and R. Navigli, editors, *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 1993–2003, Online, Aug. 2021. Association for Computational Linguistics.  [10.18653/v1/2021.acl-long.155](https:/doi.org/10.18653/v1/2021.acl-long.155).  URL . 
  - Ren et al. \[2020\]  Y. Ren, J. Liu, X. Tan, Z. Zhao, S. Zhao, and T.-Y. Liu.  A study of non-autoregressive model for sequence generation.  In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 149–159, 2020. 
  - Ringel and Romano \[2026\]  L. Ringel and Y. Romano.  Accelerating speculative decoding with block diffusion draft trees.  *arXiv preprint arXiv:2604.12989*, 2026. 
  - Sadhukhan et al. \[2025\]  R. Sadhukhan, J. Chen, Z. Chen, V. Tiwari, R. Lai, J. Shi, I. E.-H. Yen, A. May, T. Chen, and B. Chen.  Magicdec: Breaking the latency-throughput tradeoff for long context generation with speculative decoding.  In *The Thirteenth International Conference on Learning Representations*, 2025.  URL . 
  - Saharia et al. \[2020\]  C. Saharia, W. Chan, S. Saxena, and M. Norouzi.  Non-autoregressive machine translation with latent alignments.  In B. Webber, T. Cohn, Y. He, and Y. Liu, editors, *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 1098–1108, Online, Nov. 2020. Association for Computational Linguistics.  [10.18653/v1/2020.emnlp-main.83](https:/doi.org/10.18653/v1/2020.emnlp-main.83).  URL . 
  - Sandler et al. \[2026\]  J. Sandler, J. Christopher, T. Hartvigsen, and F. Fioretto.  Specdiff-2: Scaling diffusion drafter alignment for faster speculative decoding.  In *Ninth Conference on Machine Learning and Systems*, 2026.  URL . 
  - Saxena \[2023\]  A. Saxena.  Prompt lookup decoding, November 2023.  URL . 
  - Shao et al. \[2021\]  C. Shao, Y. Feng, J. Zhang, F. Meng, and J. Zhou.  Sequence-level training for non-autoregressive neural machine translation.  *Computational Linguistics*, 47(4):891–925, Dec. 2021.  [10.1162/coli\_a\_00421](https:/doi.org/10.1162/coli_a_00421).  URL . 
  - Shao et al. \[2023\]  C. Shao, Z. Ma, M. Zhang, and Y. Feng.  Beyond MLE: Convex learning for text generation.  In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023.  URL . 
  - Shen et al. \[2026\]  Y. Shen, T. Liu, X. Hu, Q. Kong, B. Zhang, J. Dai, J. Zhang, S. Ge, L. Chen, Y. Li, M. Wan, and C. Wang.  Draft less, retrieve more: Hybrid tree construction for speculative decoding, 2026.  URL . 
  - Somasundaram et al. \[2025\]  S. Somasundaram, A. Phukan, and A. Saxena.  PLD+: Accelerating LLM inference by leveraging language model artifacts.  In L. Chiruzzo, A. Ritter, and L. Wang, editors, *Findings of the Association for Computational Linguistics: NAACL 2025*, pages 6090–6104, Albuquerque, New Mexico, Apr. 2025. Association for Computational Linguistics.  ISBN 979-8-89176-195-7.  [10.18653/v1/2025.findings-naacl.338](https:/doi.org/10.18653/v1/2025.findings-naacl.338).  URL . 
  - Stern et al. \[2018\]  M. Stern, N. Shazeer, and J. Uszkoreit.  Blockwise parallel decoding for deep autoregressive models.  In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, *Advances in Neural Information Processing Systems*, volume 31. Curran Associates, Inc., 2018.  URL . 
  - Sun et al. \[2021\]  X. Sun, T. Ge, F. Wei, and H. Wang.  Instantaneous grammatical error correction with shallow aggressive decoding.  In C. Zong, F. Xia, W. Li, and R. Navigli, editors, *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 5937–5947, Online, Aug. 2021. Association for Computational Linguistics.  [10.18653/v1/2021.acl-long.462](https:/doi.org/10.18653/v1/2021.acl-long.462).  URL . 
  - Sun et al. \[2019\]  Z. Sun, Z. Li, H. Wang, D. He, Z. Lin, and Z. Deng.  Fast structured decoding for sequence models.  In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, *Advances in Neural Information Processing Systems*, volume 32. Curran Associates, Inc., 2019.  URL . 
  - Taori et al. \[2023\]  R. Taori, I. Gulrajani, T. Zhang, Y. Dubois, X. Li, C. Guestrin, P. Liang, and T. B. Hashimoto.  Stanford alpaca: An instruction-following llama model.  , 2023. 
  - Tiwari et al. \[2026\]  S. Tiwari, T. Chugh, N. Rickert, S. Peter, R. Mahajan, and H. Shen.  Cachewise: Understanding workloads and optimizing kvcache management for efficiently serving llm coding agents.  *arXiv preprint arXiv:2606.16824*, 2026. 
  - Wang et al. \[2018\]  C. Wang, J. Zhang, and H. Chen.  Semi-autoregressive neural machine translation.  In E. Riloff, D. Chiang, J. Hockenmaier, and J. Tsujii, editors, *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 479–488, Brussels, Belgium, Oct.-Nov. 2018. Association for Computational Linguistics.  [10.18653/v1/D18-1044](https:/doi.org/10.18653/v1/D18-1044).  URL . 
  - Wang et al. \[2026a\]  Z. Wang, J. Huang, and X. Chen.  Taps: Target-aware prefix tree selection for diffusion-drafted speculative decoding, 2026a.  URL . 
  - Wang et al. \[2026b\]  Z. Wang, D. Ma, X. Huang, D. Cai, T. Lan, J. Xu, H. Mi, X. Tang, and Y. Wang.  THE END OF MANUAL DECODING: TOWARDS TRULY END-TO-END LANGUAGE MODELS.  In *The Fourteenth International Conference on Learning Representations*, 2026b.  URL . 
  - Wen and Feng \[2026\]  Z. Wen and Y. Feng.  Specbound: Adaptive bounded self-speculation with layer-wise confidence calibration, 2026.  URL . 
  - Wen et al. \[2024\]  Z. Wen, S. Gui, and Y. Feng.  Speculative decoding with ctc-based draft model for llm inference acceleration.  In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, *Advances in Neural Information Processing Systems*, volume 37, pages 92082–92100. Curran Associates, Inc., 2024.  [10.52202/079017-2923](https:/doi.org/10.52202/079017-2923).  URL . 
  - Williams et al. \[2026\]  M. Williams, Y. D. Kwon, R. Li, A. Kouris, and S. I. Venieris.  Speculative decoding with a speculative vocabulary.  *arXiv preprint arXiv:2602.13836*, 2026. 
  - Wu et al. \[2025\]  Z. Wu, Z. Zhou, A. Verma, A. Prakash, D. Rus, and B. K. H. Low.  TETRIS: Optimal draft token selection for batch speculative decoding.  In W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, editors, *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 33329–33345, Vienna, Austria, July 2025. Association for Computational Linguistics.  ISBN 979-8-89176-251-0.  [10.18653/v1/2025.acl-long.1598](https:/doi.org/10.18653/v1/2025.acl-long.1598).  URL . 
  - Xia et al. \[2023\]  H. Xia, T. Ge, P. Wang, S.-Q. Chen, F. Wei, and Z. Sui.  Speculative decoding: Exploiting speculative execution for accelerating seq2seq generation.  In H. Bouamor, J. Pino, and K. Bali, editors, *Findings of the Association for Computational Linguistics: EMNLP 2023*, pages 3909–3925, Singapore, Dec. 2023. Association for Computational Linguistics.  [10.18653/v1/2023.findings-emnlp.257](https:/doi.org/10.18653/v1/2023.findings-emnlp.257).  URL . 
  - Xia et al. \[2024\]  H. Xia, Z. Yang, Q. Dong, P. Wang, Y. Li, T. Ge, T. Liu, W. Li, and Z. Sui.  Unlocking efficiency in large language model inference: A comprehensive survey of speculative decoding.  In L.-W. Ku, A. Martins, and V. Srikumar, editors, *Findings of the Association for Computational Linguistics ACL 2024*, pages 7655–7671, Bangkok, Thailand and virtual meeting, Aug. 2024. Association for Computational Linguistics.  [10.18653/v1/2024.findings-acl.456](https:/doi.org/10.18653/v1/2024.findings-acl.456).  URL . 
  - Xia et al. \[2025\]  H. Xia, Y. Li, J. Zhang, C. Du, and W. Li.  SWIFT: On-the-fly self-speculative decoding for LLM inference acceleration.  In *The Thirteenth International Conference on Learning Representations*, 2025.  URL . 
  - Xie et al. \[2026\]  Z. Xie, Y. Wei, H. Cao, C. Zhao, C. Deng, J. Li, D. Dai, H. Gao, M. Xu, K. Yu, L. Zhao, S. Zhou, Z. Xu, Z. Zhang, W. Zeng, S. Hu, Y. Wang, J. Yuan, L. Wang, and W. Liang.  mHC: Manifold-constrained hyper-connections.  In *Forty-third International Conference on Machine Learning*, 2026.  URL . 
  - Xu et al. \[2024\]  T. Xu, E. Helenowski, K. A. Sankararaman, D. Jin, K. Peng, E. Han, S. Nie, C. Zhu, H. Zhang, W. Zhou, et al.  The perfect blend: Redefining rlhf with mixture of judges.  *arXiv preprint arXiv:2409.20370*, 2024. 
  - Yan et al. \[2020\]  D. Yan, W. Wang, and X. Chu.  Demystifying tensor cores to optimize half-precision matrix multiply.  *2020 IEEE International Parallel and Distributed Processing Symposium (IPDPS)*, pages 634–643, 2020.  URL . 
  - Yang et al. \[2025\]  A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, et al.  Qwen3 technical report.  *arXiv preprint arXiv:2505.09388*, 2025. 
  - Zacks917 \[2026\]  Zacks917.  AutoMTP\_vLLM: Adapt vllm to automtp (early stop for multi-token prediction).  , 2026.  Accessed: 2026-06-21. 
  - Zhang et al. \[2024\]  J. Zhang, J. Wang, H. Li, L. Shou, K. Chen, G. Chen, and S. Mehrotra.  Draft& verify: Lossless large language model acceleration via self-speculative decoding.  In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, page 11263–11282. Association for Computational Linguistics, 2024.  [10.18653/v1/2024.acl-long.607](https:/doi.org/10.18653/v1/2024.acl-long.607).  URL . 
  - Zhang et al. \[2026a\]  J. Zhang, Z. Yu, S. Liu, E. J. Yu, Z. Li, D. Zhu, J. Duo, W. Xiong, Y. Song, G. Yu, J. Zhu, and S. Li.  Dflare: Scaling up draft capacity for block diffusion speculative decoding, 2026a.  URL . 
  - Zhang et al. \[2025\]  L. Zhang, X. Wang, Y. Huang, and R. Xu.  Learning harmonized representations for speculative sampling, 2025.  URL . 
  - Zhang et al. \[2026b\]  S. Zhang, Y. Zhang, Z. Zhu, H. Wang, D. Ma, D. Zhang, L. Chen, and K. Yu.  Pacer: Blockwise pre-verification for speculative decoding with adaptive length.  *arXiv preprint arXiv:2602.01274*, 2026b. 
  - Zhang and Math-AI \[2025\]  Y. Zhang and T. Math-AI.  American invitational mathematics examination (aime) 2025, 2025. 
  - Zhao et al. \[2025a\]  C. Zhao, C. Deng, C. Ruan, D. Dai, H. Gao, J. Li, L. Zhang, P. Huang, S. Zhou, S. Ma, et al.  Insights into deepseek-v3: Scaling challenges and reflections on hardware for ai architectures.  In *Proceedings of the 52nd Annual International Symposium on Computer Architecture*, pages 1731–1745, 2025a. 
  - Zhao et al. \[2025b\]  W. Zhao, T. Pan, X. Han, Y. Zhang, S. Ao, Y. Huang, K. Zhang, W. Zhao, Y. Li, J. Zhou, et al.  Fr-spec: Accelerating large-vocabulary language models via frequency-ranked speculative sampling.  In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 3909–3921, 2025b. 
  - Zheng et al. \[2025\]  K. Zheng, Y. Chen, H. Mao, M.-Y. Liu, J. Zhu, and Q. Zhang.  Masked diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling.  In *The Thirteenth International Conference on Learning Representations*, 2025.  URL . 
  - Zheng et al. \[2023\]  L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. Xing, et al.  Judging llm-as-a-judge with mt-bench and chatbot arena.  *Advances in neural information processing systems*, 36:46595–46623, 2023. 
  - Zheng et al. \[2024\]  L. Zheng, L. Yin, Z. Xie, C. Sun, J. Huang, C. H. Yu, S. Cao, C. Kozyrakis, I. Stoica, J. E. Gonzalez, C. Barrett, and Y. Sheng.  Sglang: Efficient execution of structured language model programs.  In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, *Advances in Neural Information Processing Systems*, volume 37, pages 62557–62583. Curran Associates, Inc., 2024.  [10.52202/079017-2000](https:/doi.org/10.52202/079017-2000).  URL . 
  - Zhong et al. \[2024\]  Y. Zhong, S. Liu, J. Chen, J. Hu, Y. Zhu, X. Liu, X. Jin, and H. Zhang.  \(\{\)DistServe\(\}\): Disaggregating prefill and decoding for goodput-optimized large language model serving.  In *18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24)*, pages 193–210, 2024. 
  - Zhu et al. \[2025\]  K. Zhu, Y. Gao, Y. Zhao, L. Zhao, G. Zuo, Y. Gu, D. Xie, T. Tang, Q. Xu, Z. Ye, K. Kamahori, C.-Y. Lin, Z. Wang, S. Wang, A. Krishnamurthy, and B. Kasikci.  Nanoflow: towards optimal large language model serving throughput.  In *Proceedings of the 19th USENIX Conference on Operating Systems Design and Implementation*, OSDI ’25, USA, 2025. USENIX Association.  ISBN 978-1-939133-47-2. 

## Appendices
## Appendix A Counterexample: Selection Bias Without Early-Stopping
We provide a simple counterexample to illustrate how an offline global search, i.e., operating without the break condition in [Algorithm 1](#alg1 "Algorithm 1 ‣ 3.2.2 Hardware-Aware Prefix Scheduler ‣ 3.2 Confidence-Scheduled Verification ‣ 3 Architecture ‣ DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation"), violates the non-anticipating property required by lossless speculative decoding. Formally, the admission event for the \(k\)-th draft token, \(\ell_{r} \geq k\), must be determined by scheduler-visible information available before the token \(x_{r,k}\) is sampled. It must not depend on the realization of \(x_{r,k}\) itself. Consider a scenario with a single request (\(R = 1\)) and maximum draft length (\(\gamma = 2\)). Suppose the pre-token confidence for the first position is \(a_{1} = 0.8\), and the profiled capacity curve is

|  |                                                                                                              |  |
|  | ------------------------------------------------------------------------------------------------------------ |  |
|  | \[{{{{SPS}\hspace{0pt}{(1)}} = 1.0},{{{{SPS}\hspace{0pt}{(2)}} = 0.5},{{{SPS}\hspace{0pt}{(3)}} = 0.45}}}.\] |  |

The expected throughputs for verifying \(0\) and \(1\) draft tokens are

|  |                |                                                               |  |
|  | -------------- | ------------------------------------------------------------- |  |
|  | \(\Theta_{0}\) | \({= {{1 \cdot {SPS}}\hspace{0pt}{(1)}} = 1.0},\)             |  |
|  | \(\Theta_{1}\) | \({= {{{({1 + 0.8})} \cdot {SPS}}\hspace{0pt}{(2)}} = 0.9}.\) |  |

Without early-stopping, the scheduler proceeds to evaluate \(\Theta_{2}\) before committing any admission decisions. Because the Markov confidence head uses the previously sampled token, the next confidence score \(c_{2}\) explicitly depends on the realization of \(x_{1}\). Consequently, the second-prefix survival probability

|  |                                      |  |
|  | ------------------------------------ |  |
|  | \[a_{2} = {a_{1}\hspace{0pt}c_{2}}\] |  |

also depends on \(x_{1}\). Consider two possible realizations of \(x_{1}\):

  - •
    
    Case 1 (\(x_{1}\) yields a high \(c_{2}\)): Suppose \(x_{1}\) results in \(c_{2} = 0.9\). Then
    |  |                                        |  |
    |  | -------------------------------------- |  |
    |  | \[{a_{2} = {0.8 \times 0.9} = 0.72}.\] |  |
    The expected throughput for length \(2\) is
    |  |                                                                |  |
    |  | -------------------------------------------------------------- |  |
    |  | \[{\Theta_{2} = {{({1 + 0.8 + 0.72})} \times 0.45} = 1.134}.\] |  |
    Since \(\Theta_{2}\) is the global maximum among \(\{ 1.0,0.9,1.134\}\), the scheduler returns \(\ell = 2\). The first token \(x_{1}\) is admitted into the verification prefix.
    
  - •
    
    Case 2 (\(x_{1}\) yields a low \(c_{2}\)): Suppose \(x_{1}\) results in \(c_{2} = 0\). Then
    |  |                  |  |
    |  | ---------------- |  |
    |  | \[{a_{2} = 0}.\] |  |
    The expected throughput for length \(2\) is
    |  |                                                            |  |
    |  | ---------------------------------------------------------- |  |
    |  | \[{\Theta_{2} = {{({1 + 0.8 + 0})} \times 0.45} = 0.81}.\] |  |
    Here, the global maximum remains \(\Theta_{0} = 1.0\), so the scheduler returns \(\ell = 0\). The first token \(x_{1}\) is not admitted into the verification prefix.
    

Thus, the admission of the first draft token dynamically depends on the value of the first draft token itself. This retrospective dependence introduces selection bias: the scheduler favors tokens that lead to highly confident continuations, even though the admission decision for \(x_{1}\) should have been made before observing \(x_{1}\). We now make the distributional bias explicit. Let the vocabulary be \(\{ A,B\}\), and consider the target and draft distributions at the first position:

|  |                                                                          |  |
|  | ------------------------------------------------------------------------ |  |
|  | \[{{{p_{t}\hspace{0pt}{(A)}} = 0.7},{{p_{t}\hspace{0pt}{(B)}} = 0.3}},\] |  |

|  |                                                                          |  |
|  | ------------------------------------------------------------------------ |  |
|  | \[{{{p_{d}\hspace{0pt}{(A)}} = 0.5},{{p_{d}\hspace{0pt}{(B)}} = 0.5}}.\] |  |

The standard speculative acceptance probability at the first position is

|  |                                                                                                                                                               |  |
|  | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |  |
|  | \[{{\sum\limits_{x \in {\{ A,B\}}}{\min\left( {p_{t}\hspace{0pt}{(x)}},{p_{d}\hspace{0pt}{(x)}} \right)}} = {{\min{(0.7,0.5)}} + {\min{(0.3,0.5)}}} = 0.8},\] |  |

matching the assumed value (\(a_{1} = 0.8\)). Suppose the retrospective scheduler behaves as above: \(x_{1} = A\) yields a high continuation confidence and hence \(\ell = 2\), while \(x_{1} = B\) yields a low continuation confidence and hence \(\ell = 0\). Then the first output token is distributed as follows. If \(x_{1} = A\), the draft token is admitted and accepted with probability

|  |                                                                                                                                  |  |
|  | -------------------------------------------------------------------------------------------------------------------------------- |  |
|  | \[{{\min\left( 1,\frac{p_{t}\hspace{0pt}{(A)}}{p_{d}\hspace{0pt}{(A)}} \right)} = {\min\left( 1,\frac{0.7}{0.5} \right)} = 1},\] |  |

so the output token is \(A\). If \(x_{1} = B\), the draft token is not admitted; the target model instead generates a fresh token from \(p_{t}\). Therefore,

|  |                                                                                                                                                      |  |
|  | ---------------------------------------------------------------------------------------------------------------------------------------------------- |  |
|  | \[{{\Pr{({Y = A})}} = {{{\Pr{({x_{1} = A})}} \cdot 1} + {{{\Pr{({x_{1} = B})}} \cdot p_{t}}\hspace{0pt}{(A)}}} = {0.5 + {0.5 \times 0.7}} = 0.85},\] |  |

and hence

|  |                                |  |
|  | ------------------------------ |  |
|  | \[{{\Pr{({Y = B})}} = 0.15}.\] |  |

This output distribution (\((0.85,0.15)\)) differs from the target distribution (\((0.7,0.3)\)), proving that the retrospective scheduler is not lossless. The early-stopping mechanism prevents this issue in the causal greedy scheduler. Since \(\Theta_{1} 

Experimental support, please [view the build logs](./2607.05147v1/__stdout.txt) for errors. Generated by [ L A T E  xml ![\[LOGO\]](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](https://math.nist.gov/~BMiller/LaTeXML/) .

## Instructions for reporting errors
We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the methods listed below:

  - Click the "Report Issue" (
    ) button, located in the page header.

**Tip:** You can select the relevant text first, to include it in your report.

Our team has already identified [the following issues](https://github.com/arXiv/html_feedback/issues). We appreciate your time reviewing and reporting rendering errors we may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability should not be a barrier to accessing research. Thank you for your continued support in championing open access for all.

Have a free development cycle? Help support accessibility at arXiv\! Our collaborators at LaTeXML maintain a [list of packages that need conversion](https://github.com/brucemiller/LaTeXML/wiki/Porting-LaTeX-packages-for-LaTeXML), and welcome [developer contributions](https://github.com/brucemiller/LaTeXML/issues).

We gratefully acknowledge support from our **major funders**, [**member institutions**](https://info.arxiv.org/about/ourmembers.html), ****, and all contributors.

[About](https://info.arxiv.org/about) · [Help](https://info.arxiv.org/help) · [Contact](https://info.arxiv.org/help/contact.html) · [Subscribe](https://info.arxiv.org/help/subscribe) · [Copyright](https://info.arxiv.org/help/license/index.html) · [Privacy](https://info.arxiv.org/help/policies/privacy_policy.html) · [Accessibility](https://info.arxiv.org/help/web_accessibility.html) · [Operational Status (opens in new tab)](https://status.arxiv.org)

Major funding support from

[![Simons Foundation](/static/base/1.0.1/images/funders/simons-foundation.png)](https://www.simonsfoundation.org/) [![Simons Foundation International](/static/base/1.0.1/images/funders/simons-foundation-international.png)](https://www.sfi.org.bm/) [![Schmidt Sciences](/static/base/1.0.1/images/funders/schmidt-sciences.png)](https://www.schmidtsciences.org/)

