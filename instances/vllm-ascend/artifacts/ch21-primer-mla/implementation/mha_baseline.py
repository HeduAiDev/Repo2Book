"""ch31 §2.1.1 —— 标准 Multi-Head Attention（MLA 的对照基线）。

DeepSeek-V2 论文 arXiv:2405.04434 §2.1.1 用它做背景：先建立"每 token 缓存 2·n_h·d_h·l 个
K/V 元素"这个后面所有压缩方案都要对抗的基线。本文件只实现这一节（Eq.1-8），不含任何压缩。

记号（与论文一致）：d=模型维，n_h=头数，d_h=每头维，T=序列长度。权重按 nn.Linear 惯例存成
(out_dim, in_dim)，对一批 token 用 h_seq @ W.T 等价于逐 token 的列向量乘法 W @ h_t。
"""
from dataclasses import dataclass

import numpy as np

from numerics import masked_softmax_scores


# PAPER: §2.1.1 Eq.1-3, Eq.8 —— W^Q/W^K/W^V/W^O 的容器
@dataclass
class MHAWeights:
    W_Q: np.ndarray  # (n_h*d_h, d)
    W_K: np.ndarray  # (n_h*d_h, d)
    W_V: np.ndarray  # (n_h*d_h, d)
    W_O: np.ndarray  # (d, n_h*d_h)


# PAPER: §2.1.1 Eq.1-3, Eq.8 —— 随机初始化 W^Q/W^K/W^V/W^O（论文未规定具体初始化，仅需可跑的小参数）
def init_mha_weights(d: int, n_h: int, d_h: int, seed: int = 0) -> MHAWeights:
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(d)
    return MHAWeights(
        W_Q=rng.normal(scale=scale, size=(n_h * d_h, d)),
        W_K=rng.normal(scale=scale, size=(n_h * d_h, d)),
        W_V=rng.normal(scale=scale, size=(n_h * d_h, d)),
        W_O=rng.normal(scale=scale, size=(d, n_h * d_h)),
    )


# PAPER: §2.1.1 Eq.1-3 —— q_t=W^Q h_t, k_t=W^K h_t, v_t=W^V h_t
def project_qkv(h_seq: np.ndarray, weights: MHAWeights):
    q_seq = h_seq @ weights.W_Q.T
    k_seq = h_seq @ weights.W_K.T
    v_seq = h_seq @ weights.W_V.T
    return q_seq, k_seq, v_seq


# PAPER: §2.1.1 Eq.4-6 —— 切成 n_h 个头
def split_heads(x_seq: np.ndarray, n_h: int, d_h: int) -> np.ndarray:
    """(T, n_h*d_h) -> (n_h, T, d_h)"""
    t = x_seq.shape[0]
    return x_seq.reshape(t, n_h, d_h).transpose(1, 0, 2)


# PAPER: §2.1.1 Eq.4-6 的逆操作 —— 供 Eq.8 的拼接 [o_{t,1};...;o_{t,n_h}] 复用
def merge_heads(x_heads: np.ndarray) -> np.ndarray:
    """(n_h, T, d_h) -> (T, n_h*d_h)，split_heads 的逆操作。"""
    n_h, t, d_h = x_heads.shape
    return x_heads.transpose(1, 0, 2).reshape(t, n_h * d_h)


# PAPER: §2.1.1 Eq.7 —— o_{t,i} = sum_j Softmax_j(q_{t,i}^T k_{j,i} / sqrt(d_h)) v_{j,i}
def scaled_dot_product_attention(q_heads: np.ndarray, k_heads: np.ndarray, v_heads: np.ndarray) -> np.ndarray:
    n_h, t_q, d_h = q_heads.shape
    t_k = k_heads.shape[1]
    o_heads = np.zeros((n_h, t_q, v_heads.shape[-1]))
    scale = np.sqrt(d_h)
    for i in range(n_h):
        scores = q_heads[i] @ k_heads[i].T / scale  # (T_q, T_k)
        weights = masked_softmax_scores(scores)
        o_heads[i] = weights @ v_heads[i]
    return o_heads


# PAPER: §2.1.1 Eq.8 —— u_t = W^O [o_{t,1};...;o_{t,n_h}]
def output_projection(o_heads: np.ndarray, W_O: np.ndarray) -> np.ndarray:
    return merge_heads(o_heads) @ W_O.T


# PAPER: §2.1.1 Eq.1-8 —— 全套标准 MHA 前向的装配壳
class StandardMHA:
    """标准 MHA 的完整前向（Eq.1-8），KV cache = 2*n_h*d_h*l 元素/token 的基线就来自这里。"""

    # PAPER: §2.1.1 Eq.1-3（持有 W^Q/W^K/W^V/W^O）
    def __init__(self, d: int, n_h: int, d_h: int, seed: int = 0):
        self.d, self.n_h, self.d_h = d, n_h, d_h
        self.weights = init_mha_weights(d, n_h, d_h, seed)

    # PAPER: §2.1.1 Eq.1-8 —— 投影→切头→注意力→拼接→输出投影
    def forward(self, h_seq: np.ndarray) -> np.ndarray:
        q_seq, k_seq, v_seq = project_qkv(h_seq, self.weights)
        q_heads = split_heads(q_seq, self.n_h, self.d_h)
        k_heads = split_heads(k_seq, self.n_h, self.d_h)
        v_heads = split_heads(v_seq, self.n_h, self.d_h)
        o_heads = scaled_dot_product_attention(q_heads, k_heads, v_heads)
        return output_projection(o_heads, self.weights.W_O)
