"""arXiv:2512.02556 (DeepSeek-V3.2, "DeepSeek Sparse Attention") §2.1 "Prototype of DSA"
—— lightning indexer 的打分函数 Eq.(1) 与细粒度 token 选择 Eq.(2)；以及 §2.1
"Instantiate DSA Under MLA" 里"为何用独立小头"的字面依据：indexer 的头数/头维是一组
独立于主注意力头数/头维的配置，不共享参数、不共享维度。
"""
from dataclasses import dataclass

import numpy as np


# 独立于主注意力头数/头维的配置——字面对应 vllm 的 config.index_n_heads /
# config.index_head_dim（与主注意力 num_attention_heads / qk_head_dim 无关的独立字段）。
# indexer 只负责打分排序、不参与最终数值计算，因此可以又小又低精度。
# PAPER: §2.1 "Instantiate DSA Under MLA" —— H^I(indexer 头数)、d^I(indexer 头维)
@dataclass(frozen=True)
class IndexerConfig:
    n_heads: int  # H^I
    head_dim: int  # d^I


# 逐头点积 -> ReLU -> 逐头标量权重加权求和。ReLU（而非 softmax）是论文明确写出的吞吐
# 考虑；k_s^I 不分头（MQA 式，跨头共享同一份 key），与"each latent vector will be
# shared across all query heads"一致。
# PAPER: §2.1 Eq.(1) —— I_{t,s} = sum_j w_{t,j}^I * ReLU(q_{t,j}^I . k_s^I)
def index_score(q: np.ndarray, k: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    q: [T, H^I, d^I]  T 个 query token 的 indexer query（逐头）
    k: [S, d^I]       S 个历史 token 的 indexer key（跨头共享，MQA 式）
    w: [T, H^I]       逐头标量权重

    returns I: [T, S] 索引分数
    """
    assert q.ndim == 3 and k.ndim == 2, "q 须为 [T,H^I,d^I]，k 须为 [S,d^I]"
    assert w.shape == q.shape[:2], "w 须为 [T,H^I]，与 q 的 T/H^I 对齐"
    assert q.shape[-1] == k.shape[-1], "indexer head_dim 须一致"
    dots = np.einsum("thd,sd->ths", q, k)  # q_{t,j}^I . k_s^I，逐头点积
    relu = np.maximum(dots, 0.0)  # ReLU
    return np.einsum("th,ths->ts", w, relu)  # sum_j w_{t,j}^I * relu(...)


# PAPER: §2.1 Eq.(2) —— fine-grained token selection: 只保留 I_{t,:} 中
# Top-k(I_{t,:}) 对应的 key-value 条目参与主注意力，其余丢弃。
def topk_select(scores: np.ndarray, k: int) -> np.ndarray:
    """
    scores: [T, S] 索引分数 I_{t,:}
    returns idx: [T, k] 每行 top-k 索引（降序；并列时索引小的优先，
    与 vllm top_k_per_row_prefill 的稳定选择行为一致——argsort(kind='stable') 保证）
    """
    assert scores.ndim == 2
    _, s = scores.shape
    k = min(k, s)
    order = np.argsort(-scores, axis=-1, kind="stable")
    return order[:, :k]
