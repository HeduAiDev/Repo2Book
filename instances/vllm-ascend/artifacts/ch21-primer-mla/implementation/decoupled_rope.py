"""ch31 §2.1.3 —— 解耦 RoPE：本章要补的 ch20 最大认知悬崖。

DeepSeek-V2 论文 arXiv:2405.04434 §2.1.3 原文论证（这是本文件要变成可运行代码的核心段落）：

    "If we apply RoPE for the keys k_t^C, W^{UK} in Equation (10) will be coupled with a
    position-sensitive RoPE matrix. In this way, W^{UK} cannot be absorbed into W^Q any more
    during inference, since a RoPE matrix related to the currently generating token will lie
    between W^Q and W^{UK} and matrix multiplication does not obey a commutative law. As a
    result, we must recompute the keys for all the prefix tokens during inference."

即：若直接对压缩后的 key 施加 RoPE，打分展开为
    q_{t}^T k_{j}^{C,rope} = h_t^T (W^Q)^T R_t^T R_j W^{UK} c_j = h_t^T (W^Q)^T R_{j-t} W^{UK} c_j
中间夹了一个依赖相对位置 (j-t) 的旋转矩阵 R_{j-t}——它把 (W^Q)^T 和 W^{UK} 隔开了。矩阵乘不满足
交换律，R_{j-t} 无法被搬到最外侧和 (W^Q)^T W^{UK} 合并成一个与位置无关的静态矩阵；换句话说，
"打分中间那块矩阵" M(delta)=(W^Q)^T R_delta W^{UK} 本身就是 delta 的函数，每个相对位置一个值，
根本没有"先离线算好、线上复用"的空间——这就是 low_rank_mla.py 里 W~（与位置无关的静态量）
之所以能被吸收、而这里不能的根本区别。

解法（Eq.14-19）：不要把 RoPE 加在共享的 k^C 上，而是拆出一小撮专门扛位置信息的维度
q_t^R / k_t^R（d_h^R 维），c^{KV} 主体保持位置无关从而继续可吸收；q^R/k^R 拼接到 nope 部分
之后一起参与打分（Eq.16-18）。
"""
from dataclasses import dataclass

import numpy as np

from mha_baseline import split_heads
from numerics import masked_softmax_scores


# ---------------------------------------------------------------------------
# RoPE 本体：显式旋转矩阵（小维度下直接构造矩阵，便于对"矩阵乘不交换"做数值验证）
# ---------------------------------------------------------------------------

# PAPER: §2.1.3 Eq.14-15 "RoPE(.)" 算子的显式矩阵形式（标准 RoPE 定义，论文引用 Su et al. 2024）
def rope_rotation_matrix(pos: float, dim: int, base: float = 10000.0) -> np.ndarray:
    """构造论文 RoPE(.) 使用的旋转矩阵 R_pos（dim x dim，dim 必须是偶数）。

    按标准 RoPE 定义：把 dim 维向量看成 dim/2 个二维子空间，第 i 个子空间按角度
    pos * theta_i 旋转，theta_i = base^{-2i/dim}。R_pos 是分块对角的正交矩阵。
    """
    assert dim % 2 == 0, "RoPE 维度必须是偶数（成对旋转）"
    half = dim // 2
    inv_freq = base ** (-(2.0 * np.arange(half)) / dim)
    angles = pos * inv_freq
    R = np.zeros((dim, dim))
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    for i in range(half):
        R[2 * i, 2 * i] = cos_a[i]
        R[2 * i, 2 * i + 1] = -sin_a[i]
        R[2 * i + 1, 2 * i] = sin_a[i]
        R[2 * i + 1, 2 * i + 1] = cos_a[i]
    return R


# PAPER: §2.1.3 Eq.14-15 "RoPE(.)" 算子——批量施加到整段序列
def apply_rope(x_seq: np.ndarray, positions, base: float = 10000.0) -> np.ndarray:
    """对序列逐 token 施加 RoPE：out[t] = R_{positions[t]} @ x_seq[t]。"""
    out = np.zeros_like(x_seq)
    for t in range(x_seq.shape[0]):
        R = rope_rotation_matrix(positions[t], x_seq.shape[-1], base)
        out[t] = R @ x_seq[t]
    return out


# PAPER: §2.1.3 文字 "a RoPE matrix related to the currently generating token" —— R_t^T R_j = R_{j-t} 性质
def verify_relative_position_property(pos_query: float, pos_key: float, dim: int, base: float = 10000.0):
    """RoPE 的标准性质：R_query^T @ R_key == R_{key-query}（旋转矩阵的转置=逆=负角旋转，
    两次旋转叠加=角度相加）。返回 (lhs, rhs) 供调用方自行比较——这条性质本身没有错，
    错的是"能否把 R_{key-query} 从 (W^Q)^T 与 W^{UK} 中间搬出去"，见 effective_middle_matrix。
    """
    r_q = rope_rotation_matrix(pos_query, dim, base)
    r_k = rope_rotation_matrix(pos_key, dim, base)
    lhs = r_q.T @ r_k
    rhs = rope_rotation_matrix(pos_key - pos_query, dim, base)
    return lhs, rhs


# ---------------------------------------------------------------------------
# 认知悬崖演示：假设"直接对 k^C 加 RoPE"这条错误路线，量化展示它为何不可吸收
# ---------------------------------------------------------------------------

# PAPER: §2.1.3 文字 "If we apply RoPE for the keys k_t^C..." —— 反证用的假设路线打分
def rope_on_compressed_key_score(h_t, pos_t, c_j, pos_j, w_q_head, w_uk_head, base: float = 10000.0) -> float:
    """假设路线（论文明确否决的路线）：对物化出的 k^C=W^{UK}c 直接加 RoPE 再打分。

    q_t = R_{pos_t}(W^Q h_t)，k_j^{C,rope} = R_{pos_j}(W^{UK} c_j)，score = q_t . k_j^{C,rope}。
    这需要在每一步都重新对每个历史 j 物化 k_j^C 再旋转——恰是论文说的"must recompute the
    keys for all the prefix tokens"。
    """
    q_t = rope_rotation_matrix(pos_t, w_q_head.shape[0], base) @ (w_q_head @ h_t)
    k_j = rope_rotation_matrix(pos_j, w_uk_head.shape[0], base) @ (w_uk_head @ c_j)
    return float(q_t @ k_j)


# PAPER: §2.1.3 文字 "matrix multiplication does not obey a commutative law" —— 夹在中间、随位置变化的矩阵
def effective_middle_matrix(w_q_head: np.ndarray, w_uk_head: np.ndarray, delta: float, base: float = 10000.0) -> np.ndarray:
    """把 rope_on_compressed_key_score 展开后夹在中间的矩阵 M(delta) = (W^Q)^T R_delta W^{UK}。

    score = h_t^T (W^Q)^T R_{pos_j-pos_t} W^{UK} c_j == h_t^T @ effective_middle_matrix(delta) @ c_j
    随 delta 变化——不存在一个与 delta 无关的静态矩阵可供预计算/复用，这就是"不可吸收"的代数含义。
    """
    d_h = w_q_head.shape[0]
    r_delta = rope_rotation_matrix(delta, d_h, base)
    return w_q_head.T @ r_delta @ w_uk_head


# ---------------------------------------------------------------------------
# 论文实际选择的解法：解耦 RoPE（Eq.14-19）
# ---------------------------------------------------------------------------

# PAPER: §2.1.3 Eq.14-15 —— W^{QR}/W^{KR} 的容器
@dataclass
class DecoupledRopeWeights:
    W_QR: np.ndarray  # (n_h*d_h_r, d_c_q)  c^Q -> 各头的解耦 query
    W_KR: np.ndarray  # (d_h_r, d)          h   -> 共享的解耦 key


# PAPER: §2.1.3 Eq.14-15 —— 随机初始化 W^{QR}/W^{KR}
def init_decoupled_rope_weights(d: int, n_h: int, d_h_r: int, d_c_q: int, seed: int = 0) -> DecoupledRopeWeights:
    rng = np.random.default_rng(seed)
    return DecoupledRopeWeights(
        W_QR=rng.normal(scale=1.0 / np.sqrt(d_c_q), size=(n_h * d_h_r, d_c_q)),
        W_KR=rng.normal(scale=1.0 / np.sqrt(d), size=(d_h_r, d)),
    )


# PAPER: §2.1.3 Eq.14 —— q_t^R = RoPE(W^{QR} c_t^Q)，逐头切开后各自旋转
def decoupled_query_rope(c_q_seq: np.ndarray, W_QR: np.ndarray, positions, n_h: int, d_h_r: int, base: float = 10000.0) -> np.ndarray:
    pre = c_q_seq @ W_QR.T                       # (T, n_h*d_h_r)
    pre_heads = split_heads(pre, n_h, d_h_r)      # (n_h, T, d_h_r)
    out = np.stack([apply_rope(pre_heads[i], positions, base) for i in range(n_h)])
    return out  # (n_h, T, d_h_r)


# PAPER: §2.1.3 Eq.15 —— k_t^R = RoPE(W^{KR} h_t)，n_h 个头共享同一份
def decoupled_key_rope(h_seq: np.ndarray, W_KR: np.ndarray, positions, base: float = 10000.0) -> np.ndarray:
    pre = h_seq @ W_KR.T           # (T, d_h_r)
    return apply_rope(pre, positions, base)  # (T, d_h_r)，广播到各头由调用方处理


# PAPER: §2.1.3 Eq.16-17 —— 拼接 nope + rope
def concat_nope_rope_query(q_c_heads: np.ndarray, q_r_heads: np.ndarray) -> np.ndarray:
    """(n_h,T,d_h) + (n_h,T,d_h_r) -> (n_h,T,d_h+d_h_r)"""
    return np.concatenate([q_c_heads, q_r_heads], axis=-1)


# PAPER: §2.1.3 Eq.17 —— k_{t,i} = [k_{t,i}^C ; k_t^R]（k^R 全头共享，广播后拼接）
def concat_nope_rope_key(k_c_heads: np.ndarray, k_r_seq: np.ndarray) -> np.ndarray:
    """(n_h,T,d_h) + (T,d_h_r，共享）-> (n_h,T,d_h+d_h_r)"""
    n_h = k_c_heads.shape[0]
    k_r_b = np.broadcast_to(k_r_seq, (n_h,) + k_r_seq.shape)
    return np.concatenate([k_c_heads, k_r_b], axis=-1)


# PAPER: §2.1.3 Eq.18 —— o_{t,i}=sum_j Softmax_j(q_{t,i}^T k_{j,i}/sqrt(d_h+d_h^R)) v_{j,i}^C
def decoupled_attention_scores(q_full_heads: np.ndarray, k_full_heads: np.ndarray) -> np.ndarray:
    """返回逐头的注意力权重（已做因果 softmax），形状 (n_h, T_q, T_k)。不在这里乘 value——
    调用方（mla_reference.py）要用这份权重分别去乘物化的 v^C（prefill 路径）或直接乘缓存的
    c^{KV}（decode 吸收路径），二者的等价性正是 low_rank_mla.attention_in_latent_space 的内容。
    """
    n_h, t_q, dim = q_full_heads.shape
    t_k = k_full_heads.shape[1]
    scale = np.sqrt(dim)
    out = np.zeros((n_h, t_q, t_k))
    for i in range(n_h):
        scores = q_full_heads[i] @ k_full_heads[i].T / scale
        out[i] = masked_softmax_scores(scores)
    return out
