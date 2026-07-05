"""ch32 §3.2 Eq.(3)-(6) —— NSA 总框架:用更紧凑的 K~_t/V~_t 替换原始 k_{:t}/v_{:t}。

三条支路 C={cmp,slc,win}(压缩/选择/滑窗)各自产出一份 (K~_t^c, V~_t^c),经门控 g_t^c
加权求和(Eq.5)。核心不等式 N_t << t(Eq.6)是"稀疏为何省"的定量定义——只要保留的
key/value 条目总数远小于原序列长度,就能省下大部分注意力计算量。DSA(见 lightning_indexer.py/
dsa_topk_selection.py)是这个框架的简化后裔:去掉 cmp/win 两条支路,只留一条基于 indexer
打分的选择支路。
"""
import numpy as np

from standard_attention import causal_attention_scores


# PAPER: §3.2 Eq.(4) —— o_t* = Attn(q_t, K~_t, V~_t),单支路注意力:用 K~_t/V~_t 代替原始
# k_{:t}/v_{:t}(K~_t 由 Eq.3 的映射函数 f_K 产出)
def branch_attention_output(q_t: np.ndarray, k_tilde: np.ndarray, v_tilde: np.ndarray, d_k: int) -> np.ndarray:
    if k_tilde.shape[0] == 0:
        return np.zeros(v_tilde.shape[-1] if v_tilde.ndim == 2 else 0)
    alpha = causal_attention_scores(q_t, k_tilde, d_k)
    return alpha @ v_tilde


# PAPER: §3.2 Eq.(5) —— o_t* = Sum_{c in C} g_t^c . Attn(q_t, K~_t^c, V~_t^c),
# 门控加权求和多支路(cmp/slc/win)输出
def gated_multi_branch_output(
    q_t: np.ndarray,
    branch_k: dict,
    branch_v: dict,
    gates: dict,
    d_k: int,
) -> np.ndarray:
    total = None
    for branch_name, gate in gates.items():
        out = branch_attention_output(q_t, branch_k[branch_name], branch_v[branch_name], d_k)
        contribution = gate * out
        total = contribution if total is None else total + contribution
    return total


# PAPER: §3.2 Eq.(6) —— N_t = Sum_{c in C} size[K~_t^c],多支路保留的 key/value 条目总数
def total_remapped_size(branch_k: dict) -> int:
    return sum(k.shape[0] for k in branch_k.values())


# PAPER: §3.2 文字("We maintain a high sparsity ratio by ensuring N_t << t") —— 稀疏比定义,
# N_t/t 越小,稀疏程度越高
def sparsity_ratio(n_t: int, t: int) -> float:
    if t == 0:
        return 0.0
    return n_t / t
