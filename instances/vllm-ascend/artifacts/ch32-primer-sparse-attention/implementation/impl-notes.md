# ch32 implementation notes —— 稀疏注意力谱系(NSA→DSA)原理精读：论文忠实的小型参考实现

本章是 **primer 原理章**（豁免"只做减法"）：`implementation/` 不是任何真实代码仓的精简版，
而是把 NSA 论文（arXiv:2502.11089，`book/papers/ch32-primer-sparse-attention/paper.md`）与
DeepSeek-V3.2 DSA 论文（arXiv:2512.02556，同目录 `paper-dsa.md`）的公式**逐条**变成可跑的
NumPy 代码。落地时回指 [第 21 章 稀疏注意力实现（SFA/DSA）](../ch21-sparse-attention-sfa-dsa/narrative/chapter.md)
的真实代码 `vllm_ascend/attention/sfa_v1.py` / `dsa_v1.py`。

## 文件划分

| 文件 | 覆盖论文小节 | 内容 |
|---|---|---|
| `standard_attention.py` | NSA §3.1 Eq.1-2 | 标准因果注意力 + O(L²) 点积计数：动机段的算式来源 |
| `nsa_framework.py` | NSA §3.2 Eq.3-6 | 多支路重映射总框架，N_t<<t 稀疏比定义 |
| `nsa_selection.py` | NSA §3.3.2 Eq.8-12 | 复用压缩注意力分数诱导块重要性 + top-n 块选择 |
| `lightning_indexer.py` | DSA §2.1 Eq.1 | Lightning Indexer 打分函数（ReLU 加权求和） |
| `dsa_topk_selection.py` | DSA §2.1 Eq.2 | top-k 细粒度选择 + 稀疏注意力 |
| `training_coadapt.py` | DSA §2.1.1 Eq.3-4 | KL 对齐损失 + "对齐程度 → top-k 质量召回"数值实验 |
| `cost_model.py` | DSA §2.3 | 成本模型：单 decode 步账 + 整条 prefill 累加账，数值推演 |
| `indexcache.py` | 落地 IndexCache/skip_topk | 层间复用 top-k 索引（supporting，论文未展开） |

## Paper Map（公式 ↔ 函数 ↔ 落地代码锚点）

| 论文公式 | 参考实现函数 | 对应落地代码 |
|---|---|---|
| NSA Eq.1-2（标准因果注意力，α_{t,i}=softmax(q_t^Tk_i/√d_k)） | `standard_attention.causal_attention_scores`/`causal_attention_output` | 对照组：不对应任何单一落地函数——是"没有稀疏化"的基线，动机段引用 |
| NSA §3.1 文字（O(L²) 注意力税，64K 解码占 70-80% 延迟） | `standard_attention.quadratic_dot_product_count`/`quadratic_attention_flops` | `vllm_ascend/attention/dsa_v1.py:L1574-L1649`（`_forward_prefill` 里稠密路径，稀疏化前的对照） |
| NSA Eq.3-4（K~_t/V~_t 重映射，单支路 Attn） | `nsa_framework.branch_attention_output` | 对照组：NSA 框架本身未直接落地于 vllm_ascend（vllm_ascend 落地的是 DSA 简化版） |
| NSA Eq.5-6（多支路门控求和，N_t<<t） | `nsa_framework.gated_multi_branch_output`/`total_remapped_size`/`sparsity_ratio` | `vllm_ascend/attention/dsa_v1.py:L2135-L2184`（`_forward_prefill` 里 cmp/win 分支的并行执行，DSA 版本只保留两条路） |
| NSA Eq.8（复用压缩注意力分数 p_t^cmp） | `nsa_selection.compression_attention_scores` | `vllm_ascend/attention/dsa_v1.py:L2110-L2133`（重要性打分入口，DSA 简化为独立 indexer 打分） |
| NSA Eq.9-10（块粒度映射 + GQA 组内求和） | `nsa_selection.block_importance_from_compression`/`gqa_group_importance` | 同上；DSA 落地不再需要块粒度映射（token 级选择） |
| NSA Eq.11-12（top-n 块选择 + 拼接） | `nsa_selection.topn_block_selection`/`gather_selected_blocks` | 概念前身，见下方 DSA Eq.2 的 token 级版本 |
| DSA Eq.1（lightning indexer 打分 I_{t,s}） | `lightning_indexer.indexer_score`/`indexer_scores_for_query` | `vllm_ascend/attention/dsa_v1.py:L1443-L1462`（indexer 参数装配：indexer_heads=H^I=64、inderxer_dim=d^I=128、weights_proj=w^I）+ `L2735`（`weights = self.weights_proj(x) * scale`）+ `L2683-2704`（`npu_quant_lightning_indexer` 融合算子）；`sfa_v1.py:L961-L1071`（`indexer_select_pre_process`/`indexer_select_post_process` 造 k^I/q^I/w^I） |
| DSA Eq.2（top-k 选择 + 稀疏注意力 u_t） | `dsa_topk_selection.topk_select`/`sparse_attention_output`/`indexer_then_sparse_attention` | `vllm_ascend/attention/dsa_v1.py:L2660-L2704`（`_indexer_qli` 返回 `topk_idxs`）+ `L2110-L2159`（`attn_op(..., cmp_sparse_indices=compress_topk_idxs, ...)`）；`sfa_v1.py:L1328-1347`（`topk_indices` → `_execute_sparse_flash_attention_process`） |
| DSA Eq.3（dense warm-up KL 损失） | `training_coadapt.aggregate_main_attention`/`dense_warmup_kl` | 无对应推理期代码（这是训练期损失，落地代码只跑推理；本节解释"为何 index_topk=512 能不掉点"） |
| DSA Eq.4（sparse stage KL 损失，限定 S_t） | `training_coadapt.sparse_stage_kl`/`topk_mass_recall`/`simulate_indexer_logits` | 同上 |
| DSA §2.3（成本模型 O(L·d_idx+k·d)） | `cost_model.speedup_accounting`/`vllm_ascend_deployment_numbers`/`paper_training_numbers` | `vllm_ascend/attention/dsa_v1.py:L829-831`（`index_n_heads`/`index_head_dim`/`index_topk # 512` 真实数字来源） |
| 落地 IndexCache/skip_topk（论文未展开） | `indexcache.get_cached_topk_indices`/`layer_topk_indices` | `sfa_v1.py:L1073-1091,L1328-1347`（`skip_topk`/`_get_indexcache_topk_indices`）；`dsa_v1.py:L1476-1485`（`use_index_cache`） |

## 关键设计取舍

1. **indexer key 只有一份，被 H^I 个 query 头共享**：`lightning_indexer.py` 里 `k_s`/`k_seq`
   都只有 `d^I` 一个维度（没有头维度），而 `q_t` 是 `(H^I, d^I)`——这直接照抄落地代码的形状
   （`indexer_select_pre_process` 只产出一份 `k_li`，`indexer_select_post_process` 产出
   `(n_head, head_dim)` 的 `q_li`），对应论文脚注"DSA instantiated under MLA 的 MQA 模式，
   每个 latent 被全部 query 头共享"。**不是简化，是论文和代码原本就这样设计。**
2. **ReLU 而非 softmax**：`indexer_score`/`indexer_scores_for_query` 严格用
   `np.maximum(dots, 0.0)`，不做归一化——`test_relu_zeroes_out_negative_dot_products` 专门
   验证这一点：大权重乘负点积应该被 ReLU 清零，而不是被保留成一个大的负贡献。这是论文
   §2.1 明确写的"为吞吐考量"的设计选择，不能悄悄换成更"合理"的 softmax。
3. **训练损失只做度量,不做梯度下降**：`training_coadapt.py` 没有实现反向传播——论文本身
   没有给出可复现的训练代码/超参细节到能跑梯度下降的程度，硬造一个训练循环会越过"论文
   忠实的小型参考实现"的边界，变成发明论文没有的东西。改用一个"对齐程度"旋钮
   （`simulate_indexer_logits` 的 `alpha`）插值出不同质量的 indexer 打分，用 Eq.3/4 的损失
   函数本身去度量它、并演示"低 KL ⟺ 高 top-k 质量召回"这一因果关系，这正是 Eq.3/4 存在的
   目的，也是"top-k 稀疏为何不掉点"的定量证据。
4. **单 decode 步 vs 整条 prefill 的两套成本函数**：`cost_model.py` 同时提供
   `decode_step_*`（单个 query 在满上下文 context_len=L 处的成本，`main_only_speedup` 精确
   等于 `context_len/k`，与 dossier 数值推演的 256×/64× 直接对应）和 `prefill_total_*`
   （对 t=1..L 累加，得到论文所说"indexer 仍是 O(L²)"这句话背后的真实标度）。两套账都
   实现，是为了不让读者把"主注意力降 256 倍"误当成"总复杂度降 256 倍"——`speedup_accounting`
   的 `end_to_end_speedup`（~8.7×）才是把 indexer 开销算进去后更诚实的数字。
5. **数值锚全部来自真实落地代码/论文原文，不是随手取的**：`index_n_heads=64`、
   `index_head_dim=128`、`index_topk=512` 取自 `dsa_v1.py:L829-831` 的真实注释；
   `n_h=128、d_c=512、d_h^R=64` 取自 ch31 primer 已核实的 DeepSeek-V2 数字（同一批 MLA
   记号在 ch31 已验证，ch32 承接）；`L=131072` 取自 paper-dsa.md §2.1.1 "context length has
   been extended to 128K"。`k=2048` 取自 paper-dsa.md §2.1.1"select 2048 key-value tokens"。

## 测试

`tests/`（51 例，host `python3 -m pytest`）覆盖：
- 标准因果注意力的 softmax 归一化性质 + O(L²) 点积计数随 L 二次增长（`test_standard_attention.py`）；
- NSA 多支路门控求和与稀疏比定义（`test_nsa_framework.py`）；
- NSA 块重要性映射（Eq.9 一般情形手算校验）+ GQA 组内求和 + top-n 选块 + 拼块
  （`test_nsa_selection.py`）；
- lightning indexer 打分函数逐对/批量一致性 + ReLU 清零负点积（`test_lightning_indexer.py`）；
- top-k 选择 + 稀疏注意力在 k=全序列时退化为稠密注意力、k<全序列时确实只用选中 KV
  （`test_dsa_topk_selection.py`）；
- KL 对齐损失在完美对齐时为 0、在错位时显著为正；top-k 质量召回在对齐 indexer 下最大化；
  **旗舰测试**：对齐程度旋钮从 0 扫到 1，平均 KL 单调上升、平均 top-k 质量召回单调下降
  （`test_training_coadapt.py`）；
- 成本模型：单 decode 步主注意力降幅精确等于 `context_len/k`（256×/64× 与 dossier 数值
  推演一致）；端到端加速账（含 indexer）严格比只看主注意力更保守；整条 prefill 累加账
  与单步成本逐项求和一致（`test_cost_model.py`）；
- IndexCache/skip_topk：命中缓存时绝不调用真实 indexer 计算路径，未命中时正确报错
  （`test_indexcache.py`）。

全部通过。
