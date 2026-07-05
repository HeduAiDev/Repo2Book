# ch36 implementation notes —— DeepSeek-V4 CSA/HCA 混合注意力（论文精读）：论文忠实的小型参考实现

本章是 **primer 原理章**（豁免"只做减法"）：`implementation/` 不是任何真实代码仓的精简版，
而是把 DeepSeek-V4 论文（arXiv:2606.19348，`book/papers/ch36-primer-v4-csa-hca/paper.md`）
§2.2（mHC）、§2.3（CSA/HCA 混合注意力）的公式**逐条**变成可跑的 NumPy 代码。落地时回指
`vllm_ascend/models/deepseek_v4.py`（Compressor/Indexer/DeepseekV4Attention/DeepseekV2DecoderLayer/
DeepseekV4Model）、`vllm_ascend/utils.py`（get_dsv4_compress_ratio）、
`vllm_ascend/worker/kvcomp_utils.py`（KVCompConfig/HashEncoder/KVCompMetaData）、
`vllm_ascend/models/layer/attention/layer.py`（DSAAttention.get_kv_cache_spec）。

本章是 ch31（MLA，压 KV 的**维度**）→ ch32（DSA，稀疏**选块**）演进线的收束章：CSA/HCA
把两条线合流——先压**序列长度**(每 m 个 token 合一)，CSA 层再叠加 DSA 式 top-k 稀疏选择。

## 文件划分

| 文件 | 覆盖论文小节 | 内容 |
|---|---|---|
| `csa_compression.py` | §2.3.1 Eq.9-12 / §2.3.2 Eq.20-23 | CSA(重叠 2m 窗口)与 HCA(不重叠 m')共用的 KV 压缩算子；`overlap_transform` 是落地同名函数的对应物 |
| `lightning_indexer_csa.py` | §2.3.1 Eq.13-17 | lightning indexer：indexer query/权重/打分/top-k 选块，复用 `csa_compression` 压出 indexer key |
| `shared_mqa_grouped_output.py` | §2.3.1 Eq.18-19 / §2.3.2 Eq.24-26 | 共享 KV 的 MQA 核注意力 + 分组输出投影(CSA/HCA 同构，合一实现) |
| `attention_extras.py` | §2.3.3 + Eq.27 | Query/KV RMSNorm、部分 RoPE、滑窗支线、注意力 sink |
| `hybrid_layer.py` | §1 + §2.3 | 混合开关表(get_dsv4_compress_ratio)+ 单层装配(LayerSpec)+ 模型层序列交错 + KV 缓存路由 |
| `mhc_sinkhorn.py` | §2.2 Eq.1-8 | Manifold-Constrained Hyper-Connections：动态参数化 + Sigmoid/Sinkhorn-Knopp 约束 |
| `kvcomp_hash.py` | 落地近似 §2.3.1 Eq.16-17 | KVComp：LSH 哈希 + 汉明距离 top-k 选块(运行期工程近似，非论文正文公式) |
| `efficiency_account.py` | §1 + §2.3.4 | 数值推演账本模型：CSA/HCA/稠密三种层的 KV 存量与 FLOPs 账 + 混合精度存储账 |

## Paper Map（公式 ↔ 参考实现函数 ↔ 落地代码锚点）

| 论文公式 | 参考实现函数 | 对应落地代码 |
|---|---|---|
| Eq.1-8（mHC：HC 残差更新 + 双随机矩阵约束 + Sinkhorn-Knopp） | `mhc_sinkhorn.hc_residual_update`/`dynamic_raw_params`/`apply_constraints`/`sinkhorn_knopp` | `vllm_ascend/models/deepseek_v4.py:L984-1003`（`DeepseekV2DecoderLayer.forward` 的 `hc_pre`/`hc_post` 融合算子，内含 `hc_sinkhorn_iters`） |
| Eq.9-10（CSA：C^a/C^b/Z^a/Z^b 投影） | `csa_compression.csa_project_kv_entries` | `vllm_ascend/models/deepseek_v4.py:L610-621`（`Compressor.__init__` 的 `wkv`/`wgate`，`coff=2` 打包两段） |
| Eq.11-12（CSA：重叠 2m 窗口 softmax 加权压缩） | `csa_compression.overlap_transform`/`softmax_over_positions`/`weighted_pool`/`csa_compress_sequence` | `vllm_ascend/models/deepseek_v4.py:L668-674`（`overlap_transform`） |
| Eq.13-15（CSA：indexer query/权重的低秩生成） | `lightning_indexer_csa.low_rank_query_latent`/`indexer_queries`/`indexer_head_weights` | `vllm_ascend/models/deepseek_v4.py:L544-556`（`Indexer.__init__` 的 `wq_b`/`weights_proj`） |
| Eq.16-17（CSA：lightning indexer 打分 + top-k 选块） | `lightning_indexer_csa.index_score`/`index_scores_for_query`/`topk_sparse_selection`/`indexer_compressed_keys` | `vllm_ascend/models/deepseek_v4.py:L590-592`（`Indexer.compressor`，复用 Compressor 压 indexer key）；`vllm_ascend/worker/kvcomp_utils.py`（运行期用 hash 近似，见下） |
| Eq.18-19（CSA：共享 KV 的 MQA 核注意力 + 分组输出投影） | `shared_mqa_grouped_output.attention_queries`/`mqa_core_attention`/`grouped_output_projection` | `vllm_ascend/models/deepseek_v4.py:L774-789`（`wo_a`/`wo_b`/`n_groups`/`o_lora_rank`） |
| Eq.20-23（HCA：不重叠压缩） | `csa_compression.hca_project_kv_entries`/`hca_compress_sequence` | `vllm_ascend/models/deepseek_v4.py:L598-666`（同一个 `Compressor` 类，`overlap=False` 分支，`coff=1`） |
| Eq.24-26（HCA：共享 KV MQA，与 CSA 同构） | `shared_mqa_grouped_output.attention_output_pipeline`（HCA 调用时传全部压缩块而非稀疏子集） | 同 Eq.18-19 落地锚点 |
| §2.3.3 首段（Query/KV RMSNorm） | `attention_extras.rms_norm` | `vllm_ascend/models/deepseek_v4.py`（`self.norm`/`kv_norm`/`q_norm` 系列） |
| §2.3.3（部分 RoPE：最后 64 维 + 输出 -i 反向旋转） | `attention_extras.apply_partial_rope`/`apply_output_relative_rope` | `vllm_ascend/models/deepseek_v4.py:L792-810`（`ComplexExpRotaryEmbedding` 装配） |
| §2.3.3（滑窗支线） | `attention_extras.sliding_window_recent_kv` | `vllm_ascend/models/deepseek_v4.py:L737`（`window_size`） |
| Eq.27（注意力 sink） | `attention_extras.attention_sink_scores`/`sink_absorbed_mass` | `vllm_ascend/models/deepseek_v4.py:L743-744`（`attn_sink`） |
| §1 + §2.3（混合开关表 + 单层装配 + 层序列交错） | `hybrid_layer.get_dsv4_compress_ratio`/`build_layer_spec`/`build_model_layers`/`kv_cache_spec_for_layer` | `vllm_ascend/utils.py:L105-110`；`vllm_ascend/models/deepseek_v4.py:L790-834,L1044-1048`；`vllm_ascend/models/layer/attention/layer.py:L174-192` |
| 落地近似 Eq.16-17（KVComp：hash + 汉明 top-k） | `kvcomp_hash.HashEncoder.compute_hash`/`select_topk_blocks_by_hamming`/`must_select_indices_for` | `vllm_ascend/worker/kvcomp_utils.py:L368-458,L491-580` |
| §1 + §2.3.4（KV/FLOPs 数值推演账本） | `efficiency_account.csa_layer_cost`/`hca_layer_cost`/`hybrid_stack_average_cost`/`relative_efficiency`/`mixed_precision_kv_bytes`/`worked_example_efficiency` | 无单一落地函数对应——这是论文结论性数字("27% FLOPs / 10% KV")背后的账本模型 |

## 关键设计取舍

1. **`overlap_transform` 与落地代码同构，不是重新发明**：`csa_compression.py` 的
   `overlap_transform(a_seq, b_seq, m, pad_value)` 与 `vllm_ascend/models/deepseek_v4.py:L668-674`
   的同名方法做的是同一件事——每个压缩块 i 的窗口 = [本块自己的 a 值; 上一块借来的 b 值]，
   `test_overlap_transform_shape_and_borrow_pattern` 手算验证了这个错位借用关系。落地代码把
   a/b 打包进同一个 `coff*head_dim` 线性层输出里（`wkv`/`wgate`），本文件为了让读者对照
   论文 Eq.9 的两个独立矩阵 W^aKV/W^bKV，选择保留了分离的投影函数
   （`csa_project_kv_entries` 返回四个独立数组），但窗口拼接的**索引逻辑**与落地代码逐位对应。
2. **Eq.11 的 Softmax_row 是逐通道归一化，不是标量注意力**：`softmax_over_positions` 沿
   "位置"这一维（axis=1）独立地对每个通道 softmax——`test_weighted_pool_matches_manual_computation`
   与两个 `*_soft_argmax_limit` 测试专门验证这一点（用极端 bias 逼近 one-hot，确认压缩输出
   趋近于被选中位置的 C 值），避免读者把 Eq.11-12 的 Hadamard 积误读成普通的标量加权求和。
3. **CSA 退化为 DSA-only 基线（m=1），不用另造对照函数**：`efficiency_account.csa_layer_cost`
   在 `m=1` 时就是"不压缩、只做 top-k 稀疏"，与论文 §2.3 首段"CSA...then applies DSA"的
   陈述一致；`test_csa_with_m_equals_1_matches_dsa_only_semantics` 验证了这一退化。用它做
   DeepSeek-V3.2 DSA 基线，避免重复实现一套独立的成本模型（ch32 primer 已有一份更完整的
   DSA 成本模型，两者精神一致但不跨章导入——每章实现自成一体）。
4. **不假装复现论文的 27%/10%**：论文只给出结论性百分比，没有给出能重推出这两个数字的
   完整发行版配置（逐层 compress_ratios 数组、每层 k、indexer 头数等）。
   `efficiency_account.worked_example_efficiency` 用示意性参数（数量级参考 code_spine 的线索，
   如滑窗/index_topk 的**存在性**，但不是 DeepSeek 未公开的确切数字）跑一遍账本模型，只验证
   "hybrid 比两个基线都更省"这一定性结论，`impl-notes` 与源码注释都显式声明这一范围限制——
   这是"不发明论文没有的数字"铁律的直接体现。
5. **`apply_output_relative_rope` 显式标注论文记号的歧义**：§2.3.3 原文写"apply RoPE with
   position -i on the last 64 dimensions of each o_{t,i}"，其中"i"在同一段落里同时被用作
   "第 i 个注意力头"的下标（呼应 Eq.18-19 的 o_{t,i} 记号）。字面理解会冲突（头索引不该是
   一个"位置"）。本实现选择更符合"输出携带相对位置信息"这一说明的读法——把 `-i` 理解为
   query 自身 token 位置的负值（`-t`）——并在函数 docstring 与本文件顶部注释里显式标出这一
   选择，供 writer/reviewer 在写正文时决定如何措辞（可以直接引用原文并注明这一记号重叠，
   不必假装消歧义已经确定）。
6. **KVComp 是"落地近似"，不是论文正文公式**：`kvcomp_hash.py` 整个文件对应的是
   `vllm_ascend/worker/kvcomp_utils.py` 的工程实现，用 LSH 哈希 + 汉明距离近似 §2.3.1
   Eq.16-17 的 lightning indexer 打分——这是本章 dossier `code_spine` 明确列出的落地机制，
   但论文本身没有描述这个具体的哈希方案。文件顶部与每个函数的 `# PAPER:` 锚都明确写"落地
   近似"而非直接引用论文公式编号，避免读者误以为这是论文原文推导。
7. **mHC 的 F_l（真正的 attn/mlp 子层）不在本章重新实现**：`mhc_sinkhorn.hc_residual_update`
   把 `F_l_out` 作为外部传入的参数，不在内部调用任何 attention/MLP 实现——论文 Eq.1 本身
   就是"残差流怎么把子层输出重新混回去"，子层内部实现是另一件事（CSA/HCA 已经是本章的
   主角），强行内联会越过"论文忠实的小型参考实现"的边界。

## 测试

`tests/`（89 例，host `python3 -m pytest`）覆盖：
- CSA 重叠窗口拼接的错位借用关系手算校验 + 因果 padding(-inf)不产生 NaN；HCA 不重叠压缩
  的 shape 与退化情形；两者的 softmax 逐通道归一化性质与"soft-argmax 极限"数值验证
  （`test_csa_compression.py`）；
- lightning indexer 打分函数 ReLU 清零负点积 + 批量/逐对一致性；top-k 选择的因果约束、
  k 超过候选数、空因果窗口等边界情形（`test_lightning_indexer_csa.py`）；
- 共享 KV 的 MQA 核注意力（含单条目退化为直接返回该条目、空 kv 返回全零）+ 分组输出投影
  手算校验 + 组数不整除时报错（`test_shared_mqa_grouped_output.py`）；
- RMSNorm 单位 RMS 性质；部分 RoPE 保范数、零位置为恒等、rope_dims 超界报错；输出反向
  RoPE 与"负位置"直接调用等价；滑窗支线的正常/边界(序列起点)截断；注意力 sink 在 sink
  主导/可忽略两种极端下的正确行为（`test_attention_extras.py`）；
- 混合开关表读取(含 None/越界回退稠密)；单层装配的三种 kind(CSA/HCA/dense)及其
  compressor/indexer 挂载规则；模型层交错序列；KV 缓存路由(SWA vs MLA(compress_ratio=.))
  （`test_hybrid_layer.py`）；
- mHC 的 HC 残差更新公式手算校验；双随机矩阵判定；Sinkhorn-Knopp 收敛性(含"迭代越多越
  收敛"的单调性检验)；Sigmoid 约束的值域与中点值；动态参数化到约束施加的端到端流程
  （`test_mhc_sinkhorn.py`）；
- KVCompConfig 默认值；chunk 代表方法(max/min/sum)；QR 正交化哈希权重的正交性；
  HashEncoder 的打包/解包一致性 + 汉明距离正确计数；must_select_blocks 的 sink+recent
  语义映射；汉明 top-k 选块正确强制并入 must-select 集合（`test_kvcomp_hash.py`）；
- CSA/HCA/稠密三种层成本随压缩率倒数缩放；CSA 在 m=1 时退化为 DSA-only 基线的语义校验；
  混合层栈平均账；混合精度 KV 字节数少于纯 BF16；示意性数值推演产出"hybrid 更省"的
  一致结论，且明确不冒充论文原文百分比（`test_efficiency_account.py`）。

全部通过。

## 收工前自检

`python3 scripts/lint_paper_grounding.py <chapter_dir> --expect-primer` 无 BLOCKING（6 条
`paper_ref` WARNING 是 dossier `paper_origin.sections` 使用范围记号如 `"Eq.(9)-(12)"`、
linter 按字面子串匹配论文包文本导致的误报——本章公式确实在 paper.md 里逐条可查，
`# PAPER:` 锚已覆盖每个 def/class；这是 dossier 字段书写风格问题，非实现缺陷）。
