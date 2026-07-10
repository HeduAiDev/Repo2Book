"""arXiv:2606.19348 (DeepSeek-V4) §2.3.1 "Compressed Sparse Attention" —— CSA 把
lightning indexer 从"逐 token 打分"升级成"逐压缩块打分"：先把每 m 个 KV 条目压成 1 个
（Eq.9-12，主 KV 压缩键 C^Comp 与索引器专属压缩键 K^IComp 用同一套压缩操作并行产出），
indexer 在压缩键上打分选 top-k 压缩块（Eq.13-17，与 Eq.1/2 同构，只是 key 换成压缩块），
最后用共享 latent 的 MQA 对选中压缩块做核心注意力（Eq.18-19）。
"""
import numpy as np

from lightning_indexer import index_score, topk_select


# PAPER: §2.3.1 Eq.(11) —— Softmax_row(.) 沿 2m 个条目那一维归一化、c 个通道各自独立。
def _softmax_axis0(x: np.ndarray) -> np.ndarray:
    """沿 axis=0 做 softmax（每一列独立归一化），对应这里的 axis=0。"""
    m = x.max(axis=0, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=0, keepdims=True)


# 四个投影收在一个函数里返回 [n,c] 张量，供 compress() 消费；同一函数既可用来产出主 KV
# 的 C^a/C^b，也可用来产出索引器专属的类比输入（论文原句："CSA performs the same
# compression operation used for C^Comp to get compressed indexer keys K^IComp"）。
# PAPER: §2.3.1 Eq.(9)-(10) —— C^a=H.W^{aKV}, C^b=H.W^{bKV}, Z^a=H.W^{aZ}, Z^b=H.W^{bZ}
def project_kv_and_gates(h: np.ndarray, w_akv, w_bkv, w_az, w_bz):
    c_a = h @ w_akv
    c_b = h @ w_bkv
    z_a = h @ w_az
    z_b = h @ w_bz
    return c_a, c_b, z_a, z_b


# 每 m 个条目压成 1 个：C^Comp_i 由 C^a 的第 i 个窗口 [mi, m(i+1)-1] 与 C^b 的前一个
# 窗口 [m(i-1), mi-1]（共 2m 个候选条目）按加了可学习位置偏置 B^a/B^b 的联合 softmax
# 权重加权求和。i=0 时 Z^b 侧填 -inf、C^b 侧填 0（论文原句对 i=0 的显式约定）。这是
# "同一套压缩操作"——调用方传入哪套 (c_a,c_b,z_a,z_b)，就压出哪套压缩键（主 KV 的
# C^Comp 或 indexer 的 K^IComp），函数本身不区分。
# PAPER: §2.3.1 Eq.(11)-(12)
def compress(
    c_a: np.ndarray,
    c_b: np.ndarray,
    z_a: np.ndarray,
    z_b: np.ndarray,
    b_a: np.ndarray,
    b_b: np.ndarray,
    m: int,
) -> np.ndarray:
    n, c = c_a.shape
    assert n % m == 0, "参考实现只建模 n 能被 m 整除的情形"
    assert b_a.shape == (m, c) and b_b.shape == (m, c)
    n_blocks = n // m
    out = np.zeros((n_blocks, c), dtype=np.result_type(c_a, c_b))
    for i in range(n_blocks):
        za_win = z_a[m * i : m * (i + 1)] + b_a  # [m, c]
        ca_win = c_a[m * i : m * (i + 1)]
        if i == 0:
            zb_win = np.full((m, c), -np.inf, dtype=z_b.dtype)
            cb_win = np.zeros((m, c), dtype=c_b.dtype)
        else:
            zb_win = z_b[m * (i - 1) : m * i] + b_b
            cb_win = c_b[m * (i - 1) : m * i]
        joint_z = np.concatenate([za_win, zb_win], axis=0)  # [2m, c]
        joint_c = np.concatenate([ca_win, cb_win], axis=0)  # [2m, c]
        joint_s = _softmax_axis0(joint_z)  # 沿 2m 那维归一化，逐 c 通道独立
        out[i] = (joint_s * joint_c).sum(axis=0)
    return out


# PAPER: §2.3.1 Eq.(13)-(14) —— c_t^Q = h_t.W^{DQ}（下投影出查询侧压缩 latent），
# q_t^I = c_t^Q.W^{IUQ}（上投影出 n_h^I 个 indexer query 头）。低秩路径与主注意力
# query 共享同一个 c_t^Q（Eq.18 复用此处返回的 c_q）。
def indexer_query_low_rank(
    h: np.ndarray, w_dq: np.ndarray, w_iuq: np.ndarray, n_heads_indexer: int
):
    c_q = h @ w_dq  # [T, d_c]
    q_flat = c_q @ w_iuq  # [T, c^I * n_h^I]
    t = h.shape[0]
    q = q_flat.reshape(t, n_heads_indexer, -1)  # [T, n_h^I, c^I]
    return c_q, q


# PAPER: §2.3.1 Eq.(15) —— w_t^I = h_t.W^w（逐头标量权重，与 Eq.1 的 w_{t,j}^I
# 同构）。
def indexer_head_weights(h: np.ndarray, w_w: np.ndarray) -> np.ndarray:
    return h @ w_w  # [T, n_h^I]


# PAPER: §2.3.1 Eq.(16) —— I_{t,s} = sum_h w_{t,h}^I * ReLU(q^I_{t,h} . K^IComp_s)。
# 与 Eq.(1) 是同一个打分核（复用 lightning_indexer.index_score），差异只在于 s 现在
# 代表一个压缩块（K^IComp 的一行）而不是单个 token 的 key。
def csa_index_score(
    q: np.ndarray, k_comp: np.ndarray, w: np.ndarray
) -> np.ndarray:
    return index_score(q, k_comp, w)


# PAPER: §2.3.1 Eq.(17) —— C_t^{SprsComp} = { C^Comp_s | I_{t,s} in Top-k(I_{t,:}) }。
# 与 Eq.(2) 是同一个选择器（复用 lightning_indexer.topk_select）。
def csa_topk_select(scores: np.ndarray, k: int) -> np.ndarray:
    return topk_select(scores, k)


# PAPER: §2.3.1 Eq.(18) —— q_t = c_t^Q.W^{UQ}（主注意力 query 的上投影，与
# indexer query 共享同一个低秩 c_t^Q，但用另一套独立的上投影矩阵 W^{UQ} != W^{IUQ}）。
def main_query_low_rank(c_q: np.ndarray, w_uq: np.ndarray, n_heads: int) -> np.ndarray:
    q_flat = c_q @ w_uq
    t = c_q.shape[0]
    return q_flat.reshape(t, n_heads, -1)


# PAPER: §2.3.1 Eq.(19) —— CoreAttn(.) 的 softmax 子例程（标准缩放点积注意力）。
def _softmax_rows(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


# PAPER: §2.3.1 Eq.(19) —— "Shared Key-Value MQA"：每个被 Eq.(17) 选中的压缩 KV
# 条目同时充当 key 与 value；对每个 query 头独立做标准缩放点积注意力，只在该 query
# 的选中集合 C_t^{SprsComp} 上进行（而不是全部 n/m 个压缩块）。
def core_attention_sparse(
    q: np.ndarray, c_comp: np.ndarray, topk_idx: np.ndarray
) -> np.ndarray:
    """
    q: [T, n_h, c]    Eq.(18) 的多头 query
    c_comp: [n/m, c]  全部压缩 KV 条目（key=value 共享，即 Eq.(19) 的 CoreAttn 输入）
    topk_idx: [T, k]  Eq.(17) 选中的压缩块索引
    returns o: [T, n_h, c]
    """
    t, n_h, c = q.shape
    scale = 1.0 / np.sqrt(c)
    out = np.zeros_like(q)
    for i in range(t):
        kv = c_comp[topk_idx[i]]  # [k, c]，key=value 共享
        logits = np.einsum("hc,kc->hk", q[i], kv) * scale
        attn = _softmax_rows(logits)  # [n_h, k]
        out[i] = attn @ kv  # [n_h, c]
    return out
