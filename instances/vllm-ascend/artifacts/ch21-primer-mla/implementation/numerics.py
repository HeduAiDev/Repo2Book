"""ch31 — 小工具：数值稳定 softmax / causal mask，供其余模块共用。
不是论文里的独立公式，只是把 Eq.7 / Eq.18 里反复出现的 Softmax_j(...) 与因果截断抽出来一份。
"""
import numpy as np


# PAPER: §2.1.1 Eq.7 / §2.1.3 Eq.18 —— 两处公式共用的 Softmax_j(...) 算子（数值稳定版）
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# PAPER: §2.1.1 Eq.7 —— "sum_{j=1}^{t}" 的因果上界，只是把求和范围显式变成掩码
def causal_mask(num_query: int, num_key: int) -> np.ndarray:
    """returns bool (num_query, num_key)，True = 该位置需要被掩掉（j 严格晚于 t，即 key 在 query 之后）。

    支持 num_query != num_key（decode 增量场景：query 只有 1 个新 token，但 key 历史更长）。
    约定：第 t 个 query 对应绝对位置 (num_key - num_query + t)，只能看到 j <= 该绝对位置的 key。
    """
    offset = num_key - num_query
    t_idx = np.arange(num_query)[:, None]
    j_idx = np.arange(num_key)[None, :]
    return j_idx > (t_idx + offset)


# PAPER: §2.1.1 Eq.7 —— causal_mask + softmax 的组合，供 mha_baseline/decoupled_rope 复用
def masked_softmax_scores(scores: np.ndarray) -> np.ndarray:
    """scores: (T_q, T_k)，对每行做因果掩码 + softmax。"""
    m = causal_mask(scores.shape[0], scores.shape[1])
    masked = np.where(m, -np.inf, scores)
    return softmax(masked, axis=-1)
