# ch37 参考实现笔记 — DFlash 块扩散并行起草（primer 论文忠实实现，非 subtract-only）

本章豁免「只做减法」：`implementation/` 是**论文忠实的小型参考实现**（纯 CPU torch/NumPy），
不是 vllm-ascend/vllm 源码的精简版。每个 def/class 标 `# PAPER: §x Eq.y`（对标 `# SOURCE:` 的角
色），目的是让读者用调试器逐行跑通 Eq.(1)-(4) 与 Appendix A.3 的算子形式、DDTree 的 Algorithm 1，
并用小张量数值验证「块扩散起草不随块大小线性变慢」「KV 注入的融合 GEMM 与逐层投影数值等价」
「早位置权重更高」「best-first 堆算法拿到可证明最优的树」这几条论文断言，而不是重建
vllm-ascend/patch/worker/patch_qwen3_dflash.py 或 vllm/model_executor/models/qwen3_dflash.py 的生
产级向量化/aclgraph/int32 缓冲细节。

## 1:1 Paper Map

| 参考实现 | 论文出处 | 对应真实落地代码（锚点，非本文件对标对象） |
|---|---|---|
| `latency_model.py::per_token_latency` / `speedup` | paper.md §3.1 Eq.(1)（L=(T_draft+T_verify)/tau，eta=L_target/L） | 无直接落地代码（分析量） |
| `latency_model.py::autoregressive_draft_cost` | paper.md §3.2 Eq.(2)（T_draft=gamma·t_step） | `vllm_ascend/spec_decode/eagle_proposer.py` 逐 token 起草循环（本章不精简，仅原理对照） |
| `latency_model.py::diffusion_draft_cost` | paper.md §3.2 Eq.(3)（T_draft=t_parallel，与 gamma 无关） | `vllm_ascend/spec_decode/dflash_proposer.py:L81` `num_query_per_req=1+num_speculative_tokens`（一次前向覆盖整块） |
| `latency_model.py::speedup_for_mode` | paper.md §3.1 Eq.(1) + §3.2 Eq.(2)/(3)（组合对照） | 同上两处 |
| `position_weighted_loss.py::position_weights` / `position_weighted_cross_entropy` | paper.md §4.2 Eq.(4)（w_k=exp(-(k-1)/gamma)） | 无落地代码（训练期损失，vllm-ascend 只含推理侧） |
| `position_weighted_loss.py::sample_anchor_blocks` | paper.md §4.2（"Random sampling of masked blocks" 段） | `vllm_ascend/ops/triton/spec_decode/utils.py:L129-L136`（推理期 bonus+mask 构造与训练期锚点+掩码同构，见正文对照） |
| `position_weighted_loss.py::build_training_attention_mask` | paper.md §4.2 + Figure 4（块内双向、跨块隔离、context 列恒可见） | 无落地代码（训练期 sparse mask） |
| `kv_injection.py::rms_norm` / `fuse_target_context_features` | paper.md Appendix A.3（H_t=RMSNorm(W_c[H^(l1);...;H^(l5)])） | `vllm/model_executor/models/qwen3_dflash.py:L275-L278`（`self.fc`/`hidden_norm`，昇腾对应 `vllm_ascend/patch/worker/patch_qwen3_dflash.py:L22` `self.hidden_norm(context_states)`） |
| `kv_injection.py::build_fused_kv_weight` / `precompute_layer_kv_fused` | paper.md Appendix A.3（K_i=W_i^K H_t 等式的融合工程实现） | `vllm/model_executor/models/qwen3_dflash.py:L301-L303` `_build_fused_kv_buffers`；`vllm_ascend/patch/worker/patch_qwen3_dflash.py:L22-L43` `precompute_and_store_context_kv` |
| `kv_injection.py::precompute_layer_kv_looped` | paper.md Appendix A.3（逐层朴素读法，供与融合版数值对照） | 同上（本参考实现独有的等价性验证对照组，源码里没有逐层版本——生产代码直接用融合版） |
| `kv_injection.py::apply_rope` | Appendix A.3 未显式写出（工程支撑：给注入的 context K 一致的位置编码，才能与 query 自己的 K 在同一 attention 里比较） | `vllm_ascend/patch/worker/patch_qwen3_dflash.py:L40-L43`（`rotary_emb(positions_repeated, ...)`），基座对照 `ops.rotary_embedding`（`vllm/model_executor/models/qwen3_dflash.py:L403-L418`） |
| `kv_injection.py::dflash_layer_attention` | paper.md Appendix A.3（Q_i=W_i^Q H_d；K_i,V_i=[H_t;H_d]_seq 拼接，非因果） | `vllm/model_executor/models/qwen3_dflash.py:L125-L150` `DFlashQwen3Attention.forward`；非因果元数据见 `vllm_ascend/spec_decode/dflash_proposer.py:L144` `cad.causal=False` |
| `dflash_draft_model.py::TinyDflashDraftLayer` / `TinyDflashDraftModel` | paper.md §3.2 Eq.(3)、§4.1、§4.2（KV 注入一次 + 整块单次前向） | `vllm/model_executor/models/qwen3_dflash.py:L216-L455`（`DFlashQwen3Model`，含 `_build_fused_kv_buffers`/`precompute_and_store_context_kv`/`forward`）；昇腾调用点 `vllm_ascend/spec_decode/dflash_proposer.py:L250-L264` `build_model_inputs_first_pass` |
| `dflash_draft_model.py::count_forward_calls_diffusion` / `count_forward_calls_autoregressive` | paper.md §3.2 Eq.(2)/(3)（结构化复述："一次调用出整块" vs "每 token 一次调用"） | 同 latency_model.py 两处锚点 |
| `ddtree.py::prefix_log_prob` | paper-ddtree.md §4.2 Eq.(7) | 无落地代码（DDTree 是 DFlash 之上未在 vllm-ascend 落地的延伸论文，见正文"延伸讨论"） |
| `ddtree.py::best_first_tree` | paper-ddtree.md §4.3 Algorithm 1（Proposition 2/3：top-B 前缀=最优树，堆算法可 O(B log B) 恢复） | 同上 |
| `ddtree.py::expected_acceptance_length_surrogate` | paper-ddtree.md §4.2 Proposition 1 / Eq.(8) | 同上 |

**刻意省略**（论文推导之外、纯工程/生产细节，不属于本章 primer 推导范围）：
张量并行/EP 切分、量化（quant_config）、aclgraph 捕获与 dummy_run 显存预估、
`num_rejected_tokens` 截断（截断有效前缀是推理期"被拒 token"记账，非 Eq.(1)-(4) 或 Appendix
A.3 的一部分，正文落地一节单独讲）、int32/int64 缓冲区 dtype 选择（NPU vs CUDA 差异，纯工程）、
DDTree 的目标模型树验证（tree attention 的 ancestor-only mask 本身，§4.4）——本参考实现只到"构树"
为止（Algorithm 1），不模拟验证步骤，因为验证步骤依赖具体 transformer 的前向实现而非独立可测的
数值程序。

## 测试映射（`tests/`）

| 测试类 | 验证的论文断言 |
|---|---|
| `TestPerTokenLatencyEq1` | L=(T_draft+T_verify)/tau 与 eta=L_target/L 的直接代数核对；tau<=0 拒绝（§3.1） |
| `TestAutoregressiveCostEq2` | T_draft 随 gamma 线性增长（gamma 翻倍则代价翻倍）（§3.2 Eq.2） |
| `TestDiffusionCostEq3` | T_draft 与 gamma 无关（多组 gamma 得到同一常数）（§3.2 Eq.3） |
| `TestSpeedupForMode` | gamma=1 时两种起草模式代价相等；gamma 增大后扩散起草反超自回归起草（Fig.3 的定性claim）（§3.1-3.2） |
| `TestPositionWeightsEq4` | w_1=1；闭式解 exp(-(k-1)/gamma) 精确匹配；单调递减（§4.2 Eq.4） |
| `TestPositionWeightedCrossEntropy` | 同等误差下，早位置犯错的加权损失 > 晚位置犯错（§4.2 Eq.4 的动机）；全对时损失≈0 |
| `TestSampleAnchorBlocks` | 锚点采样落在 [0, response_len-block_size]；response 过短时拒绝（§4.2 锚点采样段） |
| `TestTrainingAttentionMask` | context 列恒可见；块内可见、跨块隔离（块对角结构）（Figure 4） |
| `TestRMSNorm`（kv_injection） | RMSNorm 输出（weight=1 时）单位 RMS |
| `TestFuseTargetContextFeatures` | H_t 形状正确；5 个被选层任一扰动都会改变 H_t（Appendix A.3，防止"5 层"名不副实） |
| `TestFusedVsLoopedKvEquivalence` | 融合 GEMM 与逐层投影数值 allclose；融合权重形状正确（"一次 GEMM 算全层"工程claim的数值验证） |
| `TestDflashLayerAttention` | 输出形状正确；输出对注入的 context K/V 敏感（KV 注入确有因果效应）；Q 只依赖 H_d（函数签名层面核实：无 target_hidden_states 参数）；块内位置双向可见 |
| `TestSingleForwardProducesWholeBlock` | 单次 forward 调用对任意 block_size 返回整块 logits；`count_forward_calls_diffusion` 恒为 1、`count_forward_calls_autoregressive` = block_size（Eq.2 vs Eq.3 的结构化对照） |
| `TestBlockIsNonCausal` | 扰动某掩码位会影响块内其它位置（含 bonus 位）的输出，证明非因果注意力覆盖整个 query 段 |
| `TestContextConditioning` | 换掉 target 隐藏特征会改变 draft 输出（KV 注入确有条件化效应） |
| `TestPrefixLogProb` | log q(u\|c,b)=sum log q_i(u_i)，直接求和核对（Eq.7） |
| `TestBestFirstTreeOptimality` | 小规模实例下 best_first_tree 与暴力枚举 top-B 前缀完全一致（集合相等）（Proposition 2/3）；返回节点数 ≤ budget；根节点恒在树中；树前缀闭合（每个非根节点的父前缀也在树中） |
| `TestExpectedAcceptanceLengthSurrogate` | 单链树的 surrogate 值 = 链上各前缀概率之和（Eq.8）；budget 增大时 surrogate 值单调不减 |

运行：`cd implementation && PYTHONPATH=. python3 -m pytest ../tests/ -q`（或从章目录
`PYTHONPATH=implementation python3 -m pytest tests/ -q`）——host 纯 CPU torch/numpy 即可跑，
无需容器；39 passed。

## 收工前自检
`python3 scripts/lint_paper_grounding.py <chapter_dir> --expect-primer` → 0 BLOCKING
（narrative/chapter.md 尚未写就时，仅剩「narrative 尚不存在」这条非阻断 WARNING 属正常；写作阶段
完成后需重跑本 lint 确认公式锚点、arXiv 引用与 key_figures（Figure 2/3/4 + DDTree Figure 2）图注
对应仍为 0 BLOCKING）。
