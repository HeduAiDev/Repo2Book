"""arXiv:2205.14135 (Dao, Fu, Ermon, Rudra, Ré — "FlashAttention: Fast and Memory-Efficient
Exact Attention with IO-Awareness") §2.2 Algorithm 0(标准注意力,物化 N×N 的 S/P 到 HBM)、
§3.1 Algorithm 1(分块 tiling + online-softmax 递推,免物化,Theorem 1 保证结果精确等于
标准注意力)、§3.2 Theorem 2(IO 复杂度账:Θ(Nd+N²) vs Θ(N²d²/M))。

因果掩码(causal)不是 Algorithm 0/1 本身的一部分,但两篇论文都在别处讨论它(FA §3.3 的
block-sparse 掩码推广;FA-2 §3.1.1 "Causal masking" 段落),这里按同一约定加上——每个
query 只能看到位置 <= 自己(可选 query_offset 偏移,供 cascade attention 的前缀/后缀切分
建模真实序列位置,见 merge_attn_states.py)。
"""
import math

import numpy as np

from online_softmax import online_softmax_merge  # noqa: F401  (⊕ 的数学同构见 impl-notes.md)


# 处理 exp(-inf - (-inf)) = nan 的情形:整块被 causal 掩码全部遮住时,该块对当前 query
# 行贡献应为 0,而不是 nan——这不是论文之外发明的机制,只是 IEEE 浮点 -inf 运算的安全
# 包装。调用方总是先算好差值(如 S-m、mi-m_new)再传进来,这里只接一个已减好的差值参数。
# PAPER: §3.1 (softmax 的 f(x):=[e^{x_1-m(x)},...,e^{x_B-m(x)}] 定义)
def _safe_exp(diff: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.exp(diff)
    return np.where(np.isnan(out), 0.0, out)


# PAPER: §2.2 Algorithm 0 —— 标准注意力:S=QK^T 物化整张 N×N,softmax(S)=P 再物化一张
# N×N,O=PV。三步各读写一次 N×N,是 §3.2 IO 复杂度 Θ(Nd+N²) 的来源。
def standard_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    causal: bool = False,
    scale: float = None,
    return_weights: bool = False,
    query_offset: int = 0,
):
    d = Q.shape[-1]
    scale = scale if scale is not None else 1.0 / math.sqrt(d)
    S = (Q @ K.T) * scale  # Algorithm 0 line 1
    if causal:
        n_q, n_k = S.shape
        q_idx = np.arange(n_q)[:, None] + query_offset
        k_idx = np.arange(n_k)[None, :]
        S = np.where(k_idx > q_idx, -np.inf, S)
    m = S.max(axis=-1, keepdims=True)  # §3.1 的安全 softmax 定义 m(x):=max_i x_i
    P = _safe_exp(S - m)
    P = P / P.sum(axis=-1, keepdims=True)  # Algorithm 0 line 2: P = softmax(S)
    O = P @ V  # Algorithm 0 line 3
    if return_weights:
        return O, P
    return O


# PAPER: §3.1 Algorithm 1 line 1 —— Bc=ceil(M/4d), Br=min(ceil(M/4d), d)
def fa_block_sizes(sram_size_m: float, head_dim_d: int):
    bc = math.ceil(sram_size_m / (4 * head_dim_d))
    br = min(bc, head_dim_d)
    return bc, br


# PAPER: §3.1 Algorithm 1 —— 分块 tiling:外层遍历 K,V 块 j,内层遍历 Q 块 i;每个 (i,j)
# 块局部算 S_ij(至多 Br x Bc,从不是 N x N),用 online-softmax 递推(与 online_softmax.py
# 的 ⊕ 算子同构)把 running (m_i,l_i,O_i) 更新到新的全局 max,再当场按 diag(l_i^new)^-1
# 归一化写回(Algorithm 1 line 12——FA-2 才把这次归一化推迟到最后,一节带过,不在本参考
# 实现范围内,见 arXiv:2307.08691 §3.1.1)。
# Theorem 1:输出精确等于 softmax(QK^T)V,S/P 全程只在"片上"(此处即 Sij 局部变量),
# 从不落到代表 HBM 的完整数组。
# PAPER: §3.1 Algorithm 1 lines 5-14
def flash_attention_forward(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    block_size_r: int,
    block_size_c: int,
    causal: bool = False,
    scale: float = None,
    return_max_block_shape: bool = False,
):
    n, d = Q.shape
    scale = scale if scale is not None else 1.0 / math.sqrt(d)

    t_r = math.ceil(n / block_size_r)
    t_c = math.ceil(n / block_size_c)
    q_blocks = [(i * block_size_r, min((i + 1) * block_size_r, n)) for i in range(t_r)]
    kv_blocks = [(j * block_size_c, min((j + 1) * block_size_c, n)) for j in range(t_c)]

    # Algorithm 1 line 2: 初始化 O,l,m(概念上在 HBM 里;这里就是普通 numpy 数组)。
    O = np.zeros((n, d))
    l = np.zeros(n)
    m = np.full(n, -np.inf)
    max_block_shape = [0, 0]

    for kc0, kc1 in kv_blocks:  # Algorithm 1 line 5: for 1<=j<=Tc
        Kj, Vj = K[kc0:kc1], V[kc0:kc1]  # line 6: load Kj,Vj
        for qr0, qr1 in q_blocks:  # line 7: for 1<=i<=Tr
            Qi = Q[qr0:qr1]
            Oi, li, mi = O[qr0:qr1], l[qr0:qr1], m[qr0:qr1]  # line 8: load Qi,Oi,li,mi

            s_ij = (Qi @ Kj.T) * scale  # line 9
            max_block_shape[0] = max(max_block_shape[0], s_ij.shape[0])
            max_block_shape[1] = max(max_block_shape[1], s_ij.shape[1])
            if causal:
                q_idx = np.arange(qr0, qr1)[:, None]
                k_idx = np.arange(kc0, kc1)[None, :]
                s_ij = np.where(k_idx > q_idx, -np.inf, s_ij)

            m_tilde_ij = s_ij.max(axis=-1)  # line 10
            # 整块被 causal 掩码全遮住时 m_tilde_ij=-inf,s_ij-m_tilde_ij 是 -inf-(-inf)=nan
            # 的合法 IEEE 结果(不是错误)——用 errstate 静默这个已被 _safe_exp 接住的警告。
            with np.errstate(invalid="ignore"):
                p_tilde_ij = _safe_exp(s_ij - m_tilde_ij[:, None])
            l_tilde_ij = p_tilde_ij.sum(axis=-1)

            m_new = np.maximum(mi, m_tilde_ij)  # line 11
            with np.errstate(invalid="ignore"):
                l_new = (
                    _safe_exp(mi - m_new) * li
                    + _safe_exp(m_tilde_ij - m_new) * l_tilde_ij
                )

            unnormalized = (
                li[:, None] * _safe_exp(mi - m_new)[:, None] * Oi
                + _safe_exp(m_tilde_ij - m_new)[:, None] * (p_tilde_ij @ Vj)
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                o_new = unnormalized / l_new[:, None]  # line 12: diag(l_i^new)^-1 * (...)
            o_new = np.nan_to_num(o_new, nan=0.0)  # l_new==0 只会发生在整行从未见过合法 key

            O[qr0:qr1] = o_new
            l[qr0:qr1] = l_new  # line 13
            m[qr0:qr1] = m_new

    if return_max_block_shape:
        return tuple(max_block_shape)
    return O


# PAPER: §3.2 Algorithm 0 的三步访存账(读 Q,K 写 S / 读 S 写 P / 读 P,V 写 O),
# 对应 Theorem 2 的 Θ(Nd+N²) —— 逐步精确计数(单位:元素个数)而非仅取渐近类。
def hbm_accesses_standard(seq_len_n: int, head_dim_d: int) -> int:
    n, d = seq_len_n, head_dim_d
    step1 = n * d + n * d + n * n  # line 1: read Q,K; write S
    step2 = n * n + n * n  # line 2: read S; write P
    step3 = n * n + n * d + n * d  # line 3: read P,V; write O
    return step1 + step2 + step3


# PAPER: §3.2 Theorem 2 证明梗概(paper.md L180)—— "given SRAM size M, load blocks of
# K,V of size Θ(M);for each block iterate over all blocks of Q(Θ(Nd/M) passes);each
# pass loads Θ(Nd) elements ⇒ Θ(N²d²/M) accesses"。按 Algorithm 1 的每行精确计数:
# 外层每个 j 只搬一次 Kj,Vj(共 Θ(Nd));但内层对每个 j 都要把整个 Q,O,l,m 重新过一遍
# (line 8 读 + line 12-13 写),循环 Tc=ceil(N/Bc) 次——Bc 越大 Tc 越小,访存越少
# (block_size_r 只影响块内并行粒度,不改变"整个 Q 被重扫 Tc 遍"这一项,故此处的渐近
# 复杂度公式只依赖 N,d,Bc;block_size_r 仍作为参数保留,呼应 Algorithm 1 的 (Bc,Br) 配对)。
# PAPER: §3.2 Theorem 2
def hbm_accesses_flash(
    seq_len_n: int, head_dim_d: int, block_size_c: int, block_size_r: int = None
) -> int:
    n, d = seq_len_n, head_dim_d
    t_c = math.ceil(n / block_size_c)
    outer = 2 * n * d  # 每个 j 恰好搬一次 Kj,Vj,累加起来正好是整个 K,V 各一遍
    inner_per_pass = 3 * n * d + 4 * n  # 每次重扫 Q,O,l,m:读(2Nd+2N)+写(Nd+2N)
    return outer + t_c * inner_per_pass
