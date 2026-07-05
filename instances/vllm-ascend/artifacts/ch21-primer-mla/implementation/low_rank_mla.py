"""ch31 §2.1.2 —— 低秩 KV 联合压缩 + q 侧低秩 + 权重吸收恒等式。

DeepSeek-V2 论文 arXiv:2405.04434 §2.1.2。核心思想：不再直接产生满维 k_t/v_t（Eq.2-3），
而是先把 h_t 压到一个低维潜向量 c_t^{KV}，K、V 都从这个共享的潜向量上投影出来（Eq.9-11）。
推理只需要缓存 c_t^{KV}（d_c 维），比标准 MHA 的 2*n_h*d_h 维便宜得多。

更关键的是论文原文紧跟着 Eq.11 之后的一句话（本章第二个核心论点）：
"since W^{UK} can be absorbed into W^Q, and W^{UV} can be absorbed into W^O, we even do not
need to compute keys and values out for attention"——即注意力可以完全在潜空间里做，两处
上投影矩阵可以分别吸进 q 侧、o 侧。本文件把这句话变成可验证的恒等式。
"""
from dataclasses import dataclass

import numpy as np

from mha_baseline import split_heads


# PAPER: §2.1.2 Eq.9-11 —— W^{DKV}/W^{UK}/W^{UV} 的容器
@dataclass
class KVCompressionWeights:
    W_DKV: np.ndarray  # (d_c, d)      下投影：h -> c^{KV}
    W_UK: np.ndarray   # (n_h*d_h, d_c) 上投影：c^{KV} -> k^C
    W_UV: np.ndarray   # (n_h*d_h, d_c) 上投影：c^{KV} -> v^C


# PAPER: §2.1.2 Eq.12-13 —— W^{DQ}/W^{UQ} 的容器
@dataclass
class QCompressionWeights:
    W_DQ: np.ndarray  # (d_c_q, d)       下投影：h -> c^Q
    W_UQ: np.ndarray  # (n_h*d_h, d_c_q) 上投影：c^Q -> q^C


# PAPER: §2.1.2 Eq.9-11 —— 随机初始化 W^{DKV}/W^{UK}/W^{UV}
def init_kv_compression_weights(d: int, n_h: int, d_h: int, d_c: int, seed: int = 0) -> KVCompressionWeights:
    rng = np.random.default_rng(seed)
    return KVCompressionWeights(
        W_DKV=rng.normal(scale=1.0 / np.sqrt(d), size=(d_c, d)),
        W_UK=rng.normal(scale=1.0 / np.sqrt(d_c), size=(n_h * d_h, d_c)),
        W_UV=rng.normal(scale=1.0 / np.sqrt(d_c), size=(n_h * d_h, d_c)),
    )


# PAPER: §2.1.2 Eq.12-13 —— 随机初始化 W^{DQ}/W^{UQ}
def init_q_compression_weights(d: int, n_h: int, d_h: int, d_c_q: int, seed: int = 0) -> QCompressionWeights:
    rng = np.random.default_rng(seed)
    return QCompressionWeights(
        W_DQ=rng.normal(scale=1.0 / np.sqrt(d), size=(d_c_q, d)),
        W_UQ=rng.normal(scale=1.0 / np.sqrt(d_c_q), size=(n_h * d_h, d_c_q)),
    )


# PAPER: §2.1.2 Eq.9-11 —— c_t^{KV}=W^{DKV}h_t, k_t^C=W^{UK}c_t^{KV}, v_t^C=W^{UV}c_t^{KV}
def kv_joint_compression(h_seq: np.ndarray, w: KVCompressionWeights):
    c_kv_seq = h_seq @ w.W_DKV.T          # (T, d_c) —— 推理期只需要缓存这个
    k_c_seq = c_kv_seq @ w.W_UK.T         # (T, n_h*d_h)
    v_c_seq = c_kv_seq @ w.W_UV.T         # (T, n_h*d_h)
    return c_kv_seq, k_c_seq, v_c_seq


# PAPER: §2.1.2 Eq.12-13 —— c_t^Q=W^{DQ}h_t, q_t^C=W^{UQ}c_t^Q（只降训练激活显存，不减 KV cache）
def q_joint_compression(h_seq: np.ndarray, w: QCompressionWeights):
    c_q_seq = h_seq @ w.W_DQ.T            # (T, d_c_q)
    q_c_seq = c_q_seq @ w.W_UQ.T          # (T, n_h*d_h)
    return c_q_seq, q_c_seq


# PAPER: §2.1.1 Eq.4-6 的按头切片写法（对权重矩阵而非激活值切）——供吸收恒等式定位 W^{UQ}_i/W^{UK}_i/W^{UV}_i
def _head_slice(W: np.ndarray, head: int, d_h: int) -> np.ndarray:
    return W[head * d_h:(head + 1) * d_h, :]


# PAPER: §2.1.2 文字（Eq.11 之后）—— 权重吸收恒等式，q 侧：W^{UK} 吸进 W^{UQ}
def precompute_absorbed_query_weights(q_w: QCompressionWeights, kv_w: KVCompressionWeights, n_h: int, d_h: int):
    """对每个头返回 W~_i = (W^{UK}_i)^T @ (W^{UQ}_i)，形状 (d_c, d_c_q)。

    推导：score_{t,j,i} = q_{t,i}^{C,T} k_{j,i}^C = (W^{UQ}_i c_t^Q)^T (W^{UK}_i c_j^{KV})
                        = c_t^{Q,T} (W^{UQ}_i)^T (W^{UK}_i) c_j^{KV}
    令 q~_{t,i} = W~_i @ c_t^Q，则 score = q~_{t,i} . c_j^{KV}——注意力可以直接在 c^{KV} 的潜空间里做，
    W~_i 只需在权重加载后算一次（推理期是静态常量，这正是 vllm_ascend 里 process_weights_after_loading
    把 kv_b_proj 拆出 W_UK_T 的落地原因）。
    """
    w_tildes = []
    for i in range(n_h):
        w_uq_i = _head_slice(q_w.W_UQ, i, d_h)   # (d_h, d_c_q)
        w_uk_i = _head_slice(kv_w.W_UK, i, d_h)  # (d_h, d_c)
        w_tildes.append(w_uk_i.T @ w_uq_i)        # (d_c, d_c_q)
    return w_tildes


# PAPER: §2.1.2 Eq.10 隐含的打分 q_{t,i}^{C,T} k_{j,i}^C —— 物化路径（对照组，等价于 prefill）
def score_materialized_nope(q_c_head_t: np.ndarray, k_c_head_j: np.ndarray) -> float:
    """物化路径打分：显式产出 q^C、k^C 后直接内积（对照组，等价于 prefill 路径）。"""
    return float(q_c_head_t @ k_c_head_j)


# PAPER: §2.1.2 文字（Eq.11 之后）—— 吸收路径打分，与 score_materialized_nope 应逐元素相等
def score_absorbed_nope(c_q_t: np.ndarray, c_kv_j: np.ndarray, w_tilde_i: np.ndarray) -> float:
    """吸收路径打分：把 q^C 换成潜空间的 q~，直接和缓存的 c^{KV} 内积——不产出任何满维 key。"""
    q_tilde_t = w_tilde_i @ c_q_t
    return float(q_tilde_t @ c_kv_j)


# PAPER: §2.1.2 文字 —— 权重吸收恒等式，o 侧：W^{UV} 吸进 W^O（体现为"先在潜空间做注意力，再吸收"）
def attention_in_latent_space(scores_row: np.ndarray, c_kv_seq: np.ndarray) -> np.ndarray:
    """用（已做过因果 softmax 的）注意力权重直接对缓存的潜向量 c^{KV} 加权求和——
    而不是对物化的 v^C 加权求和。返回值仍在潜空间（d_c 维），对应 vllm_ascend 里
    _forward_decode 对 c_kv 做 npu_fused_infer_attention_score_v2 之后、_v_up_proj 之前的中间态。
    """
    return scores_row @ c_kv_seq  # (d_c,)


# PAPER: §2.1.2 文字 —— 权重吸收恒等式，o 侧：把潜空间输出乘 W^{UV} 还原到 value 空间
def latent_to_value(latent_vec: np.ndarray, w_uv_head: np.ndarray) -> np.ndarray:
    """把潜空间的注意力输出乘 W^{UV}_i 还原到 value 空间——对应 _v_up_proj。

    恒等式：Σ_j w_j v_{j,i}^C = Σ_j w_j (W^{UV}_i c_j) = W^{UV}_i (Σ_j w_j c_j)
    即"先加权求和 c_j 再乘 W^{UV}"与"先把每个 c_j 乘成 v_j^C 再加权求和"完全等价（矩阵乘对加权和可分配）。
    """
    return w_uv_head @ latent_vec


# PAPER: §2.1.2 文字 —— o 侧吸收所需的按头 W^{UV}_i 切片，供 latent_to_value 逐头调用
def precompute_uv_head_slices(kv_w: KVCompressionWeights, n_h: int, d_h: int):
    return [_head_slice(kv_w.W_UV, i, d_h) for i in range(n_h)]


# PAPER: §2.1.1 Eq.4-6 的切头操作套用到 Eq.10-13 产出的 k^C/v^C/q^C 上
def split_kv_heads(k_c_seq: np.ndarray, v_c_seq: np.ndarray, q_c_seq: np.ndarray, n_h: int, d_h: int):
    return (
        split_heads(k_c_seq, n_h, d_h),
        split_heads(v_c_seq, n_h, d_h),
        split_heads(q_c_seq, n_h, d_h),
    )
