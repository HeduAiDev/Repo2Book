# impl-notes — ch37 primer《Lightning Indexer 与 IndexCache》

本章 `kind: primer`——不是目标代码仓的减法精简版，而是**论文忠实的小型参考实现**
（NumPy，纯 CPU，小参数即可全量跑完，用于配合 explainer 产出可示教的数值轨迹）。
每个 `def`/`class` 用 `# PAPER: §x Eq.y` 锚定论文出处（替代普通章节的 `# SOURCE:`）。
两代论文并存：arXiv:2512.02556（DeepSeek-V3.2 DSA，Eq.(1)-(4)）与
arXiv:2606.19348（DeepSeek-V4 CSA，Eq.(9)-(19) 及 §5.2.1 FP4 QAT）。

## 文件清单

- `lightning_indexer.py` —— arXiv:2512.02556 §2.1 Eq.(1)(2)：打分函数
  `I_{t,s}=Σ w·ReLU(q·k)` 与 top-k 选块；`IndexerConfig` 落地"独立小头"设计（§2.1
  "Instantiate DSA Under MLA"）。
- `kl_alignment.py` —— arXiv:2512.02556 §2.1.1 Eq.(3)(4)：Dense Warm-up / Sparse
  Training 两阶段 KL 对齐损失，以及"detach 独立优化"的语义化落地。
- `complexity.py` —— arXiv:2512.02556 §2.3：把"主注意力 O(L²)→O(Lk)、indexer 自身
  仍 O(L²)"这句诚实账换成可代入具体 L、k 的逐元素计数。
- `csa.py` —— arXiv:2606.19348 §2.3.1 Eq.(9)-(19)：V4/CSA 版压缩键构造
  （C^Comp / K^IComp 并行产出）、低秩 indexer query、压缩块上的打分选择（复用
  `lightning_indexer` 的核）、共享 latent 的 MQA 核心注意力。
- `index_cache.py` —— 落地 arXiv:2606.19348 §2.3.1 "IndexCache 要点"：indexer 专属
  KV 缓存的量化写入/收集，显式建模其与主 KV cache 完全独立分配这一性质（不持有任何
  主缓存引用）。
- `mxfp4_quant.py` —— arXiv:2606.19348 §5.2.1：indexer QK 路径的 MXFP4 量化
  （block=32，4-bit 定点）与 index 分数 FP32→BF16 截断；可验证的"top-k 召回率"效应。
- `wiring.py` —— arXiv:2512.02556 §2.1 "Instantiate DSA Under MLA"：indexer 写
  共享 `topk_indices_buffer`（纯副作用）、主注意力从同一 buffer 读取 top-k 索引再算
  数值——对应 `mla.py:L168-169` 的调用序拓扑。

## 1:1 Paper Map（参考实现符号 ↔ 论文出处 ↔ 说明）

| 参考实现符号 | 论文出处 | 对应关系 / 取舍 |
|---|---|---|
| `lightning_indexer.IndexerConfig` | arXiv:2512.02556 §2.1 "Instantiate DSA Under MLA" | H^I/d^I 独立配置，与主注意力头数/头维无关——独立小头设计的字面证据 |
| `lightning_indexer.index_score` | arXiv:2512.02556 §2.1 Eq.(1) | 逐头点积→ReLU→逐头权重加权求和；k 不分头（MQA 式，跨头共享），与"each latent vector will be shared across all query heads"一致 |
| `lightning_indexer.topk_select` | arXiv:2512.02556 §2.1 Eq.(2) | 保留 top-k 索引，`argsort(kind='stable')` 保证并列时索引小者优先，对应 vllm `top_k_per_row_prefill` 的稳定选择行为 |
| `kl_alignment.main_attention_target_distribution` | arXiv:2512.02556 §2.1.1 "Dense Warm-up Stage" | 各头求和后沿序列维 L1 归一化，构造目标分布 p_{t,:} |
| `kl_alignment.dense_warmup_loss` | arXiv:2512.02556 §2.1.1 Eq.(3) | 全序列 KL 对齐 |
| `kl_alignment.sparse_training_loss` | arXiv:2512.02556 §2.1.1 Eq.(4) | 只在选中集 S_t 上对齐；p_{t,S_t} 限制到子集不重新归一化（照字面公式的直接后果），Softmax(I_{t,S_t}) 在子集内重新 softmax |
| `kl_alignment.detach` | arXiv:2512.02556 §2.1.1 "we detach the indexer input ... for separate optimization" | 本参考实现无 autograd，`detach` 是把这句训练协议语义化为可读代码契约的恒等拷贝，不是真正的梯度阻断机制 |
| `complexity.main_attention_ops` | arXiv:2512.02556 §2.3 | 因果稠密 O(L²) vs 稀疏 O(Lk) 的逐 query 精确计数（非仅渐近符号） |
| `complexity.indexer_ops` / `speedup_ratio` | arXiv:2512.02556 §2.3 | indexer 自身仍 O(L²)，`cost_ratio` 把"常数远小"换成可乘比例；`speedup_ratio` 供 worked example 代入 L≈128k、k≈2048 |
| `csa.project_kv_and_gates` | arXiv:2606.19348 §2.3.1 Eq.(9)(10) | C^a/C^b/Z^a/Z^b 四路投影；同一函数既产主 KV 的 C^Comp 输入也产 indexer 的 K^IComp 输入（论文原句"CSA performs the same compression operation"） |
| `csa.compress` | arXiv:2606.19348 §2.3.1 Eq.(11)(12) | 每 m 个条目按联合 softmax 权重压成 1 个；i=0 时 Z^b 侧填 -inf、C^b 侧填 0（论文显式约定） |
| `csa.indexer_query_low_rank` | arXiv:2606.19348 §2.3.1 Eq.(13)(14) | 下投影出 c_t^Q，再上投影出 n_h^I 个 indexer query 头 |
| `csa.indexer_head_weights` | arXiv:2606.19348 §2.3.1 Eq.(15) | 逐头标量权重 w_t^I，与 Eq.(1) 的 w_{t,j}^I 同构 |
| `csa.csa_index_score` / `csa_topk_select` | arXiv:2606.19348 §2.3.1 Eq.(16)(17) | 直接复用 `lightning_indexer.index_score`/`topk_select`——打分核同构，s 从单 token 换成压缩块 |
| `csa.main_query_low_rank` | arXiv:2606.19348 §2.3.1 Eq.(18) | 主注意力 query 上投影，与 indexer query 共享同一个 c_t^Q（不同的上投影矩阵） |
| `csa.core_attention_sparse` | arXiv:2606.19348 §2.3.1 Eq.(19) | Shared Key-Value MQA：只在 Eq.(17) 选中的压缩块集合上做标准缩放点积注意力，key=value 共享 |
| `index_cache.IndexCache` | arXiv:2606.19348 §2.3.1 "IndexCache 要点" | 量化写入 + 分块收集；显式不持有任何主 KV cache 引用，落地"K^IComp 列宽 c^I 与主头维 c 无关"这条独立性 |
| `mxfp4_quant.quantize_mxfp4` / `dequantize_mxfp4` | arXiv:2606.19348 §5.2.1 | MXFP4：block=32 共享 scale，4-bit 定点（[-7,7]），显式建模"电平数骤减"而非位级打包编码 |
| `mxfp4_quant.quantize_scores_bf16` | arXiv:2606.19348 §5.2.1 | index 分数 FP32→BF16：截断 float32 位模式低 16 位尾数，保留指数范围 |
| `mxfp4_quant.topk_recall` | arXiv:2606.19348 §5.2.1 "99.7% recall rate" | 验证"量化误差越大、top-k 召回率越低"的方向性结论；不复刻具体 99.7% 数字（该数字来自真实训练权重的经验测量） |
| `wiring.TopkIndicesBuffer` | 落地 `mla.py:L168-169` / `deepseek_v2.py:L1217-1227` | 全模型共享 buffer，-1 填充无效位，与 vllm 填充约定一致 |
| `wiring.v32_indexer_step` | arXiv:2512.02556 §2.1 "Instantiate DSA Under MLA" | indexer 打分+选 top-k 是纯副作用调用（写 buffer），返回值仅供示教 |
| `wiring.main_attention_from_buffer` | 落地 `mla.py:L168-169` | 主注意力只从共享 buffer 读 top-k 索引，不接触 indexer 的 q/k/权重 |

## 取舍说明（不发明论文没有的机制）

- **不实现真正的反向传播 / 训练循环**：`kl_alignment.py` 的两条损失（Eq.3/4）是训练期
  概念，参考实现只算损失值本身（供 worked example 数值验证"对齐时 KL≈0"），不建模
  optimizer/梯度更新——`detach` 因此是语义占位而非真实梯度阻断。
- **量化不复刻位级编码**：`index_cache.py`/`mxfp4_quant.py` 都用"定点整数+per-block
  float scale"模拟 FP8/MXFP4 的数值行为（有界量化误差、召回率下降方向），不去逐位实现
  IEEE FP8(E4M3)/MXFP4(E2M1) 的位模式打包（`sparse_attn_indexer.py` 的
  `kv_cache_as_quant_view` 位级布局留给正文引用真实源码讲解，参考实现聚焦"量化引入的
  数值效应"这条可验证的主线）。
- **不实现 CUDA/ROCm/XPU 分派、真实 kernel（`mqa_logits`/`top_k_per_row`/
  `persistent_topk`）**：`lightning_indexer.index_score`/`topk_select` 与
  `wiring.py` 只用 NumPy 复现这些 kernel 对外可观察的数值行为（打分公式 + top-k
  语义），不复刻 GPU 并行实现细节。
- **`csa.compress` 只处理 `n` 能被 `m` 整除的情形**：论文 Eq.(11)(12) 本身只在
  "每 m 个条目压成 1 个"的规整网格上定义；不规整边界（padding/最后一个不完整块）是
  vLLM 实现层面的工程细节，不属于本章要讲的论文机制，故显式断言排除。
- **`complexity.py` 用"每 query-key 对一次核算"精确计数，不是真实 FLOPs**：论文
  §2.3 本身只给渐近符号（O(L²)/O(Lk)），本参考实现把它换成可代入具体 L、k 的整数计数
  （供 worked example），不建模乘加次数、内存带宽等真实成本模型。

## 测试

`tests/test_lightning_indexer.py`（7）、`tests/test_kl_alignment.py`（6）、
`tests/test_index_cache.py`（5）、`tests/test_csa.py`（6）、
`tests/test_mxfp4_quant.py`（6）、`tests/test_wiring.py`（4）、
`tests/test_complexity.py`（5）——TDD：均先写断言目标行为（手算小例子对拍论文公式），
再补/验证实现。`conftest.py` 把 `implementation/` 加入 `sys.path`，host
`python3 -m pytest tests/` 即可跑（纯 CPU/NumPy，无需进容器）。

跑法：
```
cd instances/vllm/artifacts/ch37-primer-lightning-indexer
python3 -m pytest tests/ -q
```
39 passed（无 xfail/skip）。
