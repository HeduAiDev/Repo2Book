> **Source note**: Excerpt of arXiv:2412.19437, "DeepSeek-V3 Technical Report" (DeepSeek-AI, 2024). Kept: §2.1.2 DeepSeekMoE with Auxiliary-Loss-Free Load Balancing (the balancing *objective* — bias-term update rule, complementary sequence-wise auxiliary loss, node-limited routing), and §3.4 Inference and Deployment (the redundant-experts *deployment* strategy: how the balancing plan is realized as a physical placement across prefill/decode). Omitted: everything else in the technical report — see full arXiv source at https://arxiv.org/abs/2412.19437. Appended below: the deepseek-ai/EPLB reference implementation README (not part of the arXiv paper; open-source code release describing the concrete hierarchical / global load-balancing packing policies referenced by the technical report's prose but not spelled out there as pseudocode).
---

#### 2.1.2 DeepSeekMoE with Auxiliary-Loss-Free Load Balancing
##### Basic Architecture of DeepSeekMoE.
For Feed-Forward Networks (FFNs), DeepSeek-V3 employs the DeepSeekMoE architecture (Dai et al., [2024](https://arxiv.org/html/2412.19437v2#bib.bib13)). Compared with traditional MoE architectures like GShard (Lepikhin et al., [2021](https://arxiv.org/html/2412.19437v2#bib.bib45)), DeepSeekMoE uses finer-grained experts and isolates some experts as shared ones. Let \(\mathbf{u}_{t}\) denote the FFN input of the \(t\)-th token, we compute the FFN output \(\mathbf{h}_{t}^{\prime}\) as follows:

|  |                             |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |  |                                                                    |
|  | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |  | ------------------------------------------------------------------ |
|  | \(\mathbf{h}_{t}^{\prime}\) | \({= {\mathbf{u}_{t} + {\sum\limits_{i = 1}^{N_{s}}{{FFN}_{i}^{(s)}\left( \mathbf{u}_{t} \right)}} + {\sum\limits_{i = 1}^{N_{r}}{g_{i,t}{{FFN}_{i}^{(r)}\left( \mathbf{u}_{t} \right)}}}}},\)                                                                                                                                                                                                                                                                                                                                                          |  | (12) |
|  | \(g_{i,t}\)                 | \({= \frac{g_{i,t}^{\prime}}{\sum_{j = 1}^{N_{r}}g_{j,t}^{\prime}}},\)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |  | (13) |
|  | \(g_{i,t}^{\prime}\)        | \(= \begin{cases}
{s_{i,t},} & {{s_{i,t} \in {{Topk}{(\left. \{ s_{j,t} \middle| {1 \leqslant j \leqslant N_{r}}\} \right.,K_{r})}}},} \\
{0,} & {\text{otherwise},} \\
\end{cases}\)                                                                                                                                                                                                                                                                                                                                                                   |  | (14) |
|  | \(s_{i,t}\)                 | =Sigmoid⁡(𝐮tT⁢𝐞i),absentSigmoidsuperscriptsubscript𝐮𝑡𝑇subscript𝐞𝑖\\displaystyle=\\operatorname{Sigmoid}\\left({\\mathbf{u}\_{t}}^{T}\\mathbf{e}\_{i}% \\right),= roman\_Sigmoid ( bold\_u start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT italic\_T end\_POSTSUPERSCRIPT bold\_e start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) , |  | (15) |

where \(N_{s}\) and \(N_{r}\) denote the numbers of shared experts and routed experts, respectively; \({FFN}_{i}^{(s)}{( \cdot )}\) and \({FFN}_{i}^{(r)}{( \cdot )}\) denote the \(i\)-th shared expert and the \(i\)-th routed expert, respectively; \(K_{r}\) denotes the number of activated routed experts; \(g_{i,t}\) is the gating value for the \(i\)-th expert; \(s_{i,t}\) is the token-to-expert affinity; \(\mathbf{e}_{i}\) is the centroid vector of the \(i\)-th routed expert; and \({Topk}{( \cdot ,K)}\) denotes the set comprising \(K\) highest scores among the affinity scores calculated for the \(t\)-th token and all routed experts. Slightly different from DeepSeek-V2, DeepSeek-V3 uses the sigmoid function to compute the affinity scores, and applies a normalization among all selected affinity scores to produce the gating values.
##### Auxiliary-Loss-Free Load Balancing.
For MoE models, an unbalanced expert load will lead to routing collapse (Shazeer et al., [2017](https://arxiv.org/html/2412.19437v2#bib.bib81)) and diminish computational efficiency in scenarios with expert parallelism. Conventional solutions usually rely on the auxiliary loss (Fedus et al., [2021](https://arxiv.org/html/2412.19437v2#bib.bib21); Lepikhin et al., [2021](https://arxiv.org/html/2412.19437v2#bib.bib45)) to avoid unbalanced load. However, too large an auxiliary loss will impair the model performance (Wang et al., [2024a](https://arxiv.org/html/2412.19437v2#bib.bib93)). To achieve a better trade-off between load balance and model performance, we pioneer an auxiliary-loss-free load balancing strategy (Wang et al., [2024a](https://arxiv.org/html/2412.19437v2#bib.bib93)) to ensure load balance. To be specific, we introduce a bias term \(b_{i}\) for each expert and add it to the corresponding affinity scores \(s_{i,t}\) to determine the top-K routing:

|  |                      |                                                                                                                                                                                                          |  |                                                                    |
|  | -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |  | ------------------------------------------------------------------ |
|  | \(g_{i,t}^{\prime}\) | \(= \begin{cases}
{s_{i,t},} & {{{s_{i,t} + b_{i}} \in {{Topk}{(\left. \{{s_{j,t} + b_{j}} \middle| {1 \leqslant j \leqslant N_{r}}\} \right.,K_{r})}}},} \\
{0,} & {\text{otherwise}.} \\
\end{cases}\) |  | (16) |

Note that the bias term is only used for routing. The gating value, which will be multiplied with the FFN output, is still derived from the original affinity score \(s_{i,t}\). During training, we keep monitoring the expert load on the whole batch of each training step. At the end of each step, we will decrease the bias term by \(\gamma\) if its corresponding expert is overloaded, and increase it by \(\gamma\) if its corresponding expert is underloaded, where \(\gamma\) is a hyper-parameter called bias update speed. Through the dynamic adjustment, DeepSeek-V3 keeps balanced expert load during training, and achieves better performance than models that encourage load balance through pure auxiliary losses.
##### Complementary Sequence-Wise Auxiliary Loss.
Although DeepSeek-V3 mainly relies on the auxiliary-loss-free strategy for load balance, to prevent extreme imbalance within any single sequence, we also employ a complementary sequence-wise balance loss:

|  |                                                                       |                                                                                                                          |  |                                                                    |
|  | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |  | ------------------------------------------------------------------ |
|  | \(\mathcal{L}_{Bal}\)                                                 | \({= {\alpha{\sum\limits_{i = 1}^{N_{r}}{f_{i}P_{i}}}}},\)                                                               |  | (17) |
|  | \(f_{i} = {\frac{N_{r}}{K_{r}T}{\sum\limits_{t = 1}^{T}\mathbb{1}}}\) | \(\left( {s_{i,t} \in {{Topk}{(\left. \{ s_{j,t} \middle| {1 \leqslant j \leqslant N_{r}}\} \right.,K_{r})}}} \right),\) |  | (18) |
|  | \(s_{i,t}^{\prime}\)                                                  | \({= \frac{s_{i,t}}{\sum_{j = 1}^{N_{r}}s_{j,t}}},\)                                                                     |  | (19) |
|  | \(P_{i}\)                                                             | \({= {\frac{1}{T}{\sum\limits_{t = 1}^{T}s_{i,t}^{\prime}}}},\)                                                          |  | (20) |

where the balance factor \(\alpha\) is a hyper-parameter, which will be assigned an extremely small value for DeepSeek-V3; \(\mathbb{1}{( \cdot )}\) denotes the indicator function; and \(T\) denotes the number of tokens in a sequence. The sequence-wise balance loss encourages the expert load on each sequence to be balanced.
##### Node-Limited Routing.
Like the device-limited routing used by DeepSeek-V2, DeepSeek-V3 also uses a restricted routing mechanism to limit communication costs during training. In short, we ensure that each token will be sent to at most \(M\) nodes, which are selected according to the sum of the highest \(\frac{K_{r}}{M}\) affinity scores of the experts distributed on each node. Under this constraint, our MoE training framework can nearly achieve full computation-communication overlap.
##### No Token-Dropping.
Due to the effective load balancing strategy, DeepSeek-V3 keeps a good load balance during its full training. Therefore, DeepSeek-V3 does not drop any tokens during training. In addition, we also implement specific deployment strategies to ensure inference load balance, so DeepSeek-V3 also does not drop tokens during inference.
![Figure 3:  Illustration of our Multi-Token Prediction (MTP) implementation. We keep the complete causal chain for the prediction of each token at each depth.](/html/2412.19437v2/x3.png)

> [intervening sections omitted for primer excerpt — see full arXiv source]

### 3.4 Inference and Deployment
We deploy DeepSeek-V3 on the H800 cluster, where GPUs within each node are interconnected using NVLink, and all GPUs across the cluster are fully interconnected via IB. To simultaneously ensure both the Service-Level Objective (SLO) for online services and high throughput, we employ the following deployment strategy that separates the prefilling and decoding stages.
#### 3.4.1 Prefilling
The minimum deployment unit of the prefilling stage consists of 4 nodes with 32 GPUs. The attention part employs 4-way Tensor Parallelism (TP4) with Sequence Parallelism (SP), combined with 8-way Data Parallelism (DP8). Its small TP size of 4 limits the overhead of TP communication. For the MoE part, we use 32-way Expert Parallelism (EP32), which ensures that each expert processes a sufficiently large batch size, thereby enhancing computational efficiency. For the MoE all-to-all communication, we use the same method as in training: first transferring tokens across nodes via IB, and then forwarding among the intra-node GPUs via NVLink. In particular, we use 1-way Tensor Parallelism for the dense MLPs in shallow layers to save TP communication.
To achieve load balancing among different experts in the MoE part, we need to ensure that each GPU processes approximately the same number of tokens. To this end, we introduce a deployment strategy of redundant experts, which duplicates high-load experts and deploys them redundantly. The high-load experts are detected based on statistics collected during the online deployment and are adjusted periodically (e.g., every 10 minutes). After determining the set of redundant experts, we carefully rearrange experts among GPUs within a node based on the observed loads, striving to balance the load across GPUs as much as possible without increasing the cross-node all-to-all communication overhead. For the deployment of DeepSeek-V3, we set 32 redundant experts for the prefilling stage. For each GPU, besides the original 8 experts it hosts, it will also host one additional redundant expert.
Furthermore, in the prefilling stage, to improve the throughput and hide the overhead of all-to-all and TP communication, we simultaneously process two micro-batches with similar computational workloads, overlapping the attention and MoE of one micro-batch with the dispatch and combine of another.
Finally, we are exploring a dynamic redundancy strategy for experts, where each GPU hosts more experts (e.g., 16 experts), but only 9 will be activated during each inference step. Before the all-to-all operation at each layer begins, we compute the globally optimal routing scheme on the fly. Given the substantial computation involved in the prefilling stage, the overhead of computing this routing scheme is almost negligible.
#### 3.4.2 Decoding
During decoding, we treat the shared expert as a routed one. From this perspective, each token will select 9 experts during routing, where the shared expert is regarded as a heavy-load one that will always be selected. The minimum deployment unit of the decoding stage consists of 40 nodes with 320 GPUs. The attention part employs TP4 with SP, combined with DP80, while the MoE part uses EP320. For the MoE part, each GPU hosts only one expert, and 64 GPUs are responsible for hosting redundant experts and shared experts. All-to-all communication of the dispatch and combine parts is performed via direct point-to-point transfers over IB to achieve low latency. Additionally, we leverage the IBGDA (NVIDIA, [2022](https://arxiv.org/html/2412.19437v2#bib.bib61)) technology to further minimize latency and enhance communication efficiency.
Similar to prefilling, we periodically determine the set of redundant experts in a certain interval, based on the statistical expert load from our online service. However, we do not need to rearrange experts since each GPU only hosts one expert. We are also exploring the dynamic redundancy strategy for decoding. However, this requires more careful optimization of the algorithm that computes the globally optimal routing scheme and the fusion with the dispatch kernel to reduce overhead.
Additionally, to enhance throughput and hide the overhead of all-to-all communication, we are also exploring processing two micro-batches with similar computational workloads simultaneously in the decoding stage. Unlike prefilling, attention consumes a larger portion of time in the decoding stage. Therefore, we overlap the attention of one micro-batch with the dispatch+MoE+combine of another. In the decoding stage, the batch size per expert is relatively small (usually within 256 tokens), and the bottleneck is memory access rather than computation. Since the MoE part only needs to load the parameters of one expert, the memory access overhead is minimal, so using fewer SMs will not significantly affect the overall performance. Therefore, to avoid impacting the computation speed of the attention part, we can allocate only a small portion of SMs to dispatch+MoE+combine.

---

## Appendix: deepseek-ai/EPLB reference implementation (README, not part of the arXiv paper)

# Expert Parallelism Load Balancer (EPLB)

When using expert parallelism (EP), different experts are assigned to different GPUs. Because the load of different 
experts may vary depending on the current workload, it is important to keep the load of different GPUs balanced. 
As described in the DeepSeek-V3 paper, we adopt a **redundant experts** strategy that duplicates heavy-loaded experts. 
Then, we heuristically pack the duplicated experts to GPUs to ensure load balancing across different GPUs. Moreover, 
thanks to the **group-limited expert routing** used in DeepSeek-V3, we also attempt to place the experts of the same 
group to the same node to reduce inter-node data traffic, whenever possible.

To facilitate reproduction and deployment, we open-source our deployed EP load balancing algorithm in `eplb.py`. 
The algorithm computes a balanced expert replication and placement plan based on the estimated expert loads. Note 
that the exact method to predict the loads of experts is out of this repo's scope. A common method is to use 
moving average of historical statistics. 

## The Algorithm

The load balancing algorithm comes with two policies used for different cases.

### Hierarchical Load Balancing

When the number of server nodes divides the number of expert groups, we use the hierarchical load balancing policy to
harness the group-limited expert routing. We first pack the expert groups to nodes evenly, ensuring the loads of 
different nodes are balanced. Then, we replicate the experts within each node. Finally, we pack the replicated experts 
to individual GPUs to ensure different GPUs are load-balanced. The hierarchical load balancing policy can be used in 
prefilling stage with a smaller expert-parallel size.

### Global Load Balancing

In other cases, we use the global load balancing policy that replicates the experts globally regardless of expert 
groups, and pack the replicated experts to individual GPUs. This policy can be adopted in decoding stage with a larger 
expert-parallel size.

## Interface and Example

The main function of the load balancer is `eplb.rebalance_experts`.

The following code illustrates an example of a two-layer MoE model, and each layer contains 12 experts. We introduce 4 redundant experts per layer, and the total 16 replicas are placed on 2 nodes, and each node contains 4 GPUs.

``` python
import torch
import eplb

weight = torch.tensor([[ 90, 132,  40,  61, 104, 165,  39,   4,  73,  56, 183,  86],
                       [ 20, 107, 104,  64,  19, 197, 187, 157, 172,  86,  16,  27]])

num_replicas = 16
num_groups = 4
num_nodes = 2
num_gpus = 8

phy2log, log2phy, logcnt = eplb.rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)
print(phy2log)

# Output:
# tensor([[ 5,  6,  5,  7,  8,  4,  3,  4, 10,  9, 10,  2,  0,  1, 11,  1],
#         [ 7, 10,  6,  8,  6, 11,  8,  9,  2,  4,  5,  1,  5,  0,  3,  1]])
```

The output, generated by the hierarchical load balancing policy, indicates the following 
expert replication and placement plan.

![](example.png)


## License

This code repository is released under the MIT License.
