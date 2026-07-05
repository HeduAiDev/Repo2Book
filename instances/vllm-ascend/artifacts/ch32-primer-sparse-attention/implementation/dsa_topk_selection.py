"""ch32 §2.1 Eq.(2)(paper-dsa.md) —— top-k 细粒度 token 选择 + 稀疏注意力。

给定 indexer 打分 I_{t,:}(Eq.1 的结果,见 lightning_indexer.py),只取 top-k 个索引对应的
KV 条目 {c_s} 参与主注意力——这是把 O(L^2) 主注意力砍到 O(L.k) 的关键一步。落地对应
sfa_v1.py 的 `topk_indices = indexer_select_post_process(...)` -> `_execute_sparse_flash_
attention_process(...)`,以及 dsa_v1.py 的 `topk_idxs, _ = torch.ops._C_ascend.
npu_quant_lightning_indexer(..., sparse_count=index_topk)` -> `attn_op(..., cmp_sparse_
indices=topk_idxs, ...)`。
"""
import numpy as np

from standard_attention import causal_attention_scores


# PAPER: §2.1 Eq.(2) —— {s | I_{t,s} in Top-k(I_{t,:})},取 indexer 打分最高的 k 个前驱下标
def topk_select(index_scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(index_scores))
    return np.argsort(-index_scores, kind="stable")[:k]


# PAPER: §2.1 Eq.(2) —— u_t = Attn(h_t, {c_s | I_{t,s} in Top-k(I_{t,:})}),
# 只在选中 KV 上算主注意力(其余前驱直接跳过,不进入 softmax 归一化范围)
def sparse_attention_output(
    q_t: np.ndarray, k_seq: np.ndarray, v_seq: np.ndarray, topk_indices: np.ndarray, d_k: int
) -> np.ndarray:
    k_selected = k_seq[topk_indices]
    v_selected = v_seq[topk_indices]
    alpha = causal_attention_scores(q_t, k_selected, d_k)
    return alpha @ v_selected


# PAPER: §2.1 Eq.(1)+Eq.(2) 端到端装配 —— 给定 query token 的 indexer 打分,
# 先 top-k(Eq.2)再稀疏注意力(Eq.2 的 Attn(...)),对应落地代码 "打分 -> topk_indices ->
# 稀疏 flash attention" 这条主链(sfa_v1.py forward L1328-1347)
def indexer_then_sparse_attention(
    q_t: np.ndarray,
    indexer_scores: np.ndarray,
    k_seq: np.ndarray,
    v_seq: np.ndarray,
    k: int,
    d_k: int,
):
    topk_indices = topk_select(indexer_scores, k)
    output = sparse_attention_output(q_t, k_seq, v_seq, topk_indices, d_k)
    return output, topk_indices
