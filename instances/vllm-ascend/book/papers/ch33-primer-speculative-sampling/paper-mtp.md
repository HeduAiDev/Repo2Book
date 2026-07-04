> **Source note**: Excerpt of arXiv:2412.19437, "DeepSeek-V3 Technical Report" (DeepSeek-AI, 2024), §2.2 Multi-Token Prediction only (MTP Modules / MTP Training Objective / MTP in Inference). This is the supplementary MTP source for the primer chapter; the main paper.md in this directory is the foundational speculative-decoding paper (Leviathan et al., arXiv:2211.17192). See full arXiv source at https://arxiv.org/abs/2412.19437.
---

### 2.2 Multi-Token Prediction
Inspired by Gloeckle et al. ([2024](https://arxiv.org/html/2412.19437v2#bib.bib26)), we investigate and set a Multi-Token Prediction (MTP) objective for DeepSeek-V3, which extends the prediction scope to multiple future tokens at each position. On the one hand, an MTP objective densifies the training signals and may improve data efficiency. On the other hand, MTP may enable the model to pre-plan its representations for better prediction of future tokens. Figure [3](https://arxiv.org/html/2412.19437v2#S2.F3 "Figure 3 ‣ No Token-Dropping. ‣ 2.1.2 DeepSeekMoE with Auxiliary-Loss-Free Load Balancing ‣ 2.1 Basic Architecture ‣ 2 Architecture ‣ DeepSeek-V3 Technical Report") illustrates our implementation of MTP. Different from Gloeckle et al. ([2024](https://arxiv.org/html/2412.19437v2#bib.bib26)), which parallelly predicts \(D\) additional tokens using independent output heads, we sequentially predict additional tokens and keep the complete causal chain at each prediction depth. We introduce the details of our MTP implementation in this section.
##### MTP Modules.
To be specific, our MTP implementation uses \(D\) sequential modules to predict \(D\) additional tokens. The \(k\)-th MTP module consists of a shared embedding layer \({Emb}{( \cdot )}\), a shared output head \({OutHead}{( \cdot )}\), a Transformer block \({TRM}_{k}{( \cdot )}\), and a projection matrix \(M_{k} \in {\mathbb{R}}^{{d \times 2}d}\). For the \(i\)-th input token \(t_{i}\), at the \(k\)-th prediction depth, we first combine the representation of the \(i\)-th token at the \(({k - 1})\)-th depth \(\mathbf{h}_{i}^{k - 1} \in {\mathbb{R}}^{d}\) and the embedding of the \(({i + k})\)-th token \({Emb{(t_{i + k})}} \in {\mathbb{R}}^{d}\) with the linear projection:

|  |                                                                                                                                     |  |                                                                    |
|  | ----------------------------------------------------------------------------------------------------------------------------------- |  | ------------------------------------------------------------------ |
|  | \[{\mathbf{h}_{i}^{\prime k} = {M_{k}{\lbrack{{RMSNorm}{(\mathbf{h}_{i}^{k - 1})}};{{RMSNorm}{({{Emb}{(t_{i + k})}})}}\rbrack}}},\] |  | (21) |

where \(\lbrack \cdot ; \cdot \rbrack\) denotes concatenation. Especially, when \(k = 1\), \(\mathbf{h}_{i}^{k - 1}\) refers to the representation given by the main model. Note that for each MTP module, its embedding layer is shared with the main model. The combined \(\mathbf{h}_{i}^{\prime k}\) serves as the input of the Transformer block at the \(k\)-th depth to produce the output representation at the current depth \(\mathbf{h}_{i}^{k}\):

|  |                                                                                      |  |                                                                    |
|  | ------------------------------------------------------------------------------------ |  | ------------------------------------------------------------------ |
|  | \[{\mathbf{h}_{1:{T - k}}^{k} = {{TRM}_{k}{(\mathbf{h}_{1:{T - k}}^{\prime k})}}},\] |  | (22) |

where \(T\) represents the input sequence length and <sub>i:j</sub> denotes the slicing operation (inclusive of both the left and right boundaries). Finally, taking \(\mathbf{h}_{i}^{k}\) as the input, the shared output head will compute the probability distribution for the \(k\)-th additional prediction token \(P_{i + 1 + k}^{k} \in {\mathbb{R}}^{V}\), where \(V\) is the vocabulary size:

|  |                                                              |  |                                                                    |
|  | ------------------------------------------------------------ |  | ------------------------------------------------------------------ |
|  | \[{P_{i + k + 1}^{k} = {{OutHead}{(\mathbf{h}_{i}^{k})}}}.\] |  | (23) |

The output head \({OutHead}{( \cdot )}\) linearly maps the representation to logits and subsequently applies the \({Softmax}{( \cdot )}\) function to compute the prediction probabilities of the \(k\)-th additional token. Also, for each MTP module, its output head is shared with the main model. Our principle of maintaining the causal chain of predictions is similar to that of EAGLE (Li et al., [2024b](https://arxiv.org/html/2412.19437v2#bib.bib51)), but its primary objective is speculative decoding (Xia et al., [2023](https://arxiv.org/html/2412.19437v2#bib.bib99); Leviathan et al., [2023](https://arxiv.org/html/2412.19437v2#bib.bib46)), whereas we utilize MTP to improve training.
##### MTP Training Objective.
For each prediction depth, we compute a cross-entropy loss \(\mathcal{L}_{\text{MTP}}^{k}\):

|  |                                                                                                                                                                                                      |  |                                                                    |
|  | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |  | ------------------------------------------------------------------ |
|  | \[{\mathcal{L}_{\text{MTP}}^{k} = {{CrossEntropy}{(P_{{2 + k}:{T + 1}}^{k},t_{{2 + k}:{T + 1}})}} = {- {\frac{1}{T}{\sum\limits_{i = {2 + k}}^{T + 1}{{\log P_{i}^{k}}{\lbrack t_{i}\rbrack}}}}}},\] |  | (24) |

where \(T\) denotes the input sequence length, \(t_{i}\) denotes the ground-truth token at the \(i\)-th position, and \(P_{i}^{k}{\lbrack t_{i}\rbrack}\) denotes the corresponding prediction probability of \(t_{i}\), given by the \(k\)-th MTP module. Finally, we compute the average of the MTP losses across all depths and multiply it by a weighting factor \(\lambda\) to obtain the overall MTP loss \(\mathcal{L}_{\text{MTP}}\), which serves as an additional training objective for DeepSeek-V3:

|  |                                                                                                            |  |                                                                    |
|  | ---------------------------------------------------------------------------------------------------------- |  | ------------------------------------------------------------------ |
|  | \[{\mathcal{L}_{\text{MTP}} = {\frac{\lambda}{D}{\sum\limits_{k = 1}^{D}\mathcal{L}_{\text{MTP}}^{k}}}}.\] |  | (25) |
##### MTP in Inference.
Our MTP strategy mainly aims to improve the performance of the main model, so during inference, we can directly discard the MTP modules and the main model can function independently and normally. Additionally, we can also repurpose these MTP modules for speculative decoding to further improve the generation latency.
