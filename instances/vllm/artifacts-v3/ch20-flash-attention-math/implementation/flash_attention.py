"""arXiv:2205.14135 (Dao, Fu, Ermon, Rudra, Ré — "FlashAttention: Fast and
Memory-Efficient Exact Attention with IO-Awareness") §2.2 Algorithm 0(标准注意力:
物化 S/P 两张 N×N 到 HBM)、§3.1 Algorithm 1(分块 tiling + online-softmax 递推,
Theorem 1 保证输出精确等于 softmax(QK^T)V)、§3.2 Theorem 2(IO 复杂度账
Θ(Nd+N²) vs Θ(N²d²/M));arXiv:2307.08691 (FlashAttention-2) §3.1.1 Algorithm 1
(循环序对调 + 未归一化 O + logsumexp L)与 Causal masking(块级跳过)。

numpy 数组即扮演 HBM(算法里的 "write to HBM" 就是写回这几个数组;"on chip /
SRAM" 就是局部变量)——物理存储层被抽象掉,数值递推逐字保留。因果掩码(causal)
不是 FA 主文 Algorithm 0/1 的一部分,但 FA-2 §3.1.1 "Causal masking" 一节明确讨
论它(S_ij = -inf for j > i,整块在因果上侧的直接跳过);seqlen_q != seqlen_k 的
右下对齐对齐约定取自 vLLM flash_attn_varlen_func docstring 的两个掩码例子。

「trace」可选参数只逐格记录算法自身循环变量的快照(供示教轨迹用),不是论文之外
的新机制。
"""
import math

import numpy as np


# PAPER: §3.1 (softmax 的 f(x) := [e^{x_1-m(x)}, ..., e^{x_B-m(x)}] 定义) —— 对
# 「已减好 max 的差值」求指数。差值为 -inf(causal 掩码遮住的列)时 e^{-inf}=0;
# 整行被遮时差值是 -inf-(-inf)=nan,这是 IEEE 浮点的合法结果而非错误,归零即该块
# 对本行贡献为 0——数值安全包装,不是新算法机制。
# PAPER: §3.1 (f(x) := e^{x - m(x)}, pointwise)
def _safe_exp(diff: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.exp(diff)
    return np.where(np.isnan(out), 0.0, out)


# PAPER: arXiv:2307.08691 §3.1.1 Causal masking (S_ij = -inf for j > i) —— 因果
# keep 掩码:keep[r, c] = (列 c) <= (行 r + query_offset)。query 行 r 的全序列
# 位置是 r + query_offset;取 offset = n_k - n_q 即把掩码贴到注意力矩阵右下角
# (vllm/vllm_flash_attn/flash_attn_interface.py docstring 的对齐约定:decode 的
# seqlen_q=1 对 seqlen_k=N 天然落在最右列,prefill/decode 才能共用同一 kernel)。
# PAPER: arXiv:2307.08691 §3.1.1 Causal masking (S_ij = -inf for j > i)
def causal_keep_mask(n_q: int, n_k: int, query_offset: int = 0) -> np.ndarray:
    q_idx = np.arange(n_q)[:, None] + query_offset
    k_idx = np.arange(n_k)[None, :]
    return k_idx <= q_idx


# PAPER: §2.2 Algorithm 0 —— 标准注意力三步:S = QK^T(物化整张 N×N)、
# P = softmax(S)(再物化一张 N×N)、O = PV。两张 N×N 中间矩阵真实存在于内存
# (return_weights=True 可拿到 P)——§3.2 IO 复杂度 Θ(Nd+N²) 与 O(N²) 显存的
# 来源。softmax_scale 默认 1/sqrt(d) 按 Appendix B.3 的 τ("typically 1/sqrt(d)",
# 与 vLLM softmax_scale 默认一致);causal 掩码经 _safe_exp 的 -inf 路径施加,
# 全零行按接口语义输出 0。
# PAPER: §2.2 Algorithm 0
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
    S = (Q @ K.T) * scale  # Algorithm 0 line 1: S = QK^T(整张 N x N,物化)
    if causal:
        S = np.where(causal_keep_mask(S.shape[0], S.shape[1], query_offset), S, -np.inf)
    m = S.max(axis=-1, keepdims=True)  # §3.1 的 m(x) := max_i x_i(safe softmax)
    with np.errstate(invalid="ignore"):  # 整行被遮时 -inf - (-inf) = nan,_safe_exp 归零
        P = _safe_exp(S - m)
    l = P.sum(axis=-1, keepdims=True)
    has_key = l > 0  # 整行被掩码遮空:输出为零(而非 0/0 = nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.where(has_key, P / np.where(has_key, l, 1.0), 0.0)  # line 2: P = softmax(S)
    O = P @ V  # line 3: O = PV
    if return_weights:
        return O, P
    return O


# PAPER: §2.2 ("Standard attention implementations materialize the matrices S and P
# to HBM, which takes O(N^2) memory") —— 精确计数被物化的中间矩阵元素:S 与 P 各
# N x N,共 2·N²。8K 上下文一张表即 8192² = 67,108,864 ≈ 6700 万元素,fp16 下
# 134MB —— 内存带宽墙的感受数字。
# PAPER: §2.2 ("materialize the matrices S and P to HBM, which takes O(N^2) memory")
def materialized_intermediate_elements(seq_len_n: int) -> int:
    return 2 * seq_len_n * seq_len_n  # S 一张 + P 一张


# PAPER: §3.1 Algorithm 1 line 1 —— Bc = ceil(M/4d), Br = min(ceil(M/4d), d)
# (M 为 SRAM 大小,d 为 head 维)。
def fa_block_sizes(sram_size_m: float, head_dim_d: int):
    bc = math.ceil(sram_size_m / (4 * head_dim_d))
    br = min(bc, head_dim_d)
    return bc, br


# PAPER: §3.1 Algorithm 1 lines 2-16 —— FlashAttention forward:外层遍历 KV 列块 j
# (line 5)、内层遍历 Q 行块 i(line 7);每个 (i,j) 块在片上算 S_ij(至多
# Br x Bc,从不是 N x N)、局部 softmax (m̃, P̃, ℓ̃)(line 10)、用 online-softmax
# 递推把 running (m_i, ℓ_i) 更新到新全局 max(line 11)、O_i 折算-累加后当场按
# diag(ℓ_new)^{-1} 归一化写回(line 12——FA-2 才把这次除法推迟到最后)。S/P 全程
# 只存在于局部变量,从不落代表 HBM 的完整数组。Theorem 1:输出精确等于
# softmax(QK^T)V,任意合法分块皆然。
# PAPER: §3.1 Algorithm 1 lines 2-16 + Theorem 1
def flash_attention_forward(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    block_size_r: int,
    block_size_c: int,
    causal: bool = False,
    scale: float = None,
    query_offset: int = 0,
    return_max_block_shape: bool = False,
    trace: list = None,
):
    n, d = Q.shape
    scale = scale if scale is not None else 1.0 / math.sqrt(d)

    t_r = math.ceil(n / block_size_r)
    t_c = math.ceil(n / block_size_c)
    q_blocks = [(i * block_size_r, min((i + 1) * block_size_r, n)) for i in range(t_r)]
    kv_blocks = [(j * block_size_c, min((j + 1) * block_size_c, n)) for j in range(t_c)]

    # line 2: 初始化 O, ℓ, m(概念上在 HBM;这里就是普通 numpy 数组)
    O = np.zeros((n, d))
    l = np.zeros(n)
    m = np.full(n, -np.inf)
    max_block_shape = [0, 0]

    for j, (kc0, kc1) in enumerate(kv_blocks):  # line 5: for 1 <= j <= Tc
        Kj, Vj = K[kc0:kc1], V[kc0:kc1]  # line 6: load Kj, Vj
        for i, (qr0, qr1) in enumerate(q_blocks):  # line 7: for 1 <= i <= Tr
            Qi = Q[qr0:qr1]
            Oi, li, mi = O[qr0:qr1], l[qr0:qr1], m[qr0:qr1]  # line 8: load Qi, Oi, ℓi, mi

            s_ij = (Qi @ Kj.T) * scale  # line 9: S_ij = Qi Kj^T(片上,至多 Br x Bc)
            max_block_shape[0] = max(max_block_shape[0], s_ij.shape[0])
            max_block_shape[1] = max(max_block_shape[1], s_ij.shape[1])
            if causal:
                keep = causal_keep_mask(
                    qr1 - qr0, kc1 - kc0, query_offset + qr0 - kc0
                )  # 局部行列 -> 全局位置:col c + kc0 <= row r + qr0 + query_offset
                s_ij = np.where(keep, s_ij, -np.inf)

            # line 10: m̃ij = rowmax(S_ij), P̃ij = exp(S_ij - m̃ij), ℓ̃ij = rowsum(P̃ij)
            # (整行被遮时 -inf - (-inf) = nan,errstate 静默警告、_safe_exp 归零)
            m_tilde_ij = s_ij.max(axis=-1)
            with np.errstate(invalid="ignore"):
                p_tilde_ij = _safe_exp(s_ij - m_tilde_ij[:, None])
            l_tilde_ij = p_tilde_ij.sum(axis=-1)

            # line 11: m_new = max(m_i, m̃ij);ℓ_new = e^{m_i-m_new}·ℓ_i + e^{m̃-m_new}·ℓ̃
            m_new = np.maximum(mi, m_tilde_ij)
            with np.errstate(invalid="ignore"):
                l_new = _safe_exp(mi - m_new) * li + _safe_exp(m_tilde_ij - m_new) * l_tilde_ij

            # line 12: O_i <- diag(ℓ_new)^{-1} (diag(ℓ_i) e^{m_i-m_new} O_i + e^{m̃-m_new} P̃ V_j)
            unnormalized = (
                li[:, None] * _safe_exp(mi - m_new)[:, None] * Oi
                + _safe_exp(m_tilde_ij - m_new)[:, None] * (p_tilde_ij @ Vj)
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                o_new = unnormalized / l_new[:, None]
            o_new = np.nan_to_num(o_new, nan=0.0)  # ℓ_new == 0 只发生在整行从未见过合法 key

            O[qr0:qr1] = o_new
            l[qr0:qr1] = l_new  # line 13: 写回 ℓ_i <- ℓ_new
            m[qr0:qr1] = m_new  # line 13: 写回 m_i <- m_new
            if trace is not None:
                trace.append(
                    {
                        "j": j,
                        "i": i,
                        "q0": qr0,
                        "q1": qr1,
                        "kv_end": kc1,
                        "m": m_new.copy(),
                        "l": l_new.copy(),
                        "O": O[qr0:qr1].copy(),
                    }
                )

    if return_max_block_shape:
        return tuple(max_block_shape)
    return O


# PAPER: arXiv:2307.08691 §3.1.1 Algorithm 1 lines 3-17 —— FlashAttention-2
# forward:外层改遍历 Q 行块 i(§3.2:行块间 embarrassingly parallel,长序列小
# batch 时占满 SM)、内层遍历 KV 列块 j;O 累加器保持未归一化(Tweak 1:中间不
# 除 ℓ,收尾 line 12 才 diag(ℓ)^{-1} 一次);只存 logsumexp L = m + log(ℓ)
# (Tweak 2,line 13;vLLM return_softmax_lse 拿到的正是它);line 17 返回 (O, L)。
# Causal masking(§3.1.1 两条):(1) 整块列索引在行块之上的直接跳过(约省一半
# 计算);(2) 只有跨对角的块需要施加逐元素掩码。
# 注意 line 10 的缩放方向:arXiv 原文印作 diag(e^{m^{(j-1)}-m^{(j)}})^{-1} O^{(j-1)}
# (即乘 e^{m^{(j)}-m^{(j-1)}},把旧账放大),与论文自己的不变式
# O = Σ_j e^{S^{(j)}-m} V^{(j)}(§3.1.1 两块表的最终恒等式)方向相反——按原文
# 逐字转写会算错。旧账必须按 e^{m^{(j-1)}-m^{(j)}} 折算到新 max(与 FA Alg.1
# line 12 的 e^{m_i-m_new} 同向),此处按不变式方向实现,测试对标准注意力逐位验证。
# PAPER: arXiv:2307.08691 §3.1.1 Algorithm 1 lines 3-17 (Tweak 1/2 + Causal masking)
def flash_attention_2_forward(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    block_size_r: int,
    block_size_c: int,
    causal: bool = False,
    scale: float = None,
    query_offset: int = 0,
    trace: list = None,
):
    n, d = Q.shape
    scale = scale if scale is not None else 1.0 / math.sqrt(d)

    t_r = math.ceil(n / block_size_r)
    t_c = math.ceil(n / block_size_c)
    q_blocks = [(i * block_size_r, min((i + 1) * block_size_r, n)) for i in range(t_r)]
    kv_blocks = [(j * block_size_c, min((j + 1) * block_size_c, n)) for j in range(t_c)]

    O = np.zeros((n, d))
    L = np.full(n, -np.inf)

    for i, (qr0, qr1) in enumerate(q_blocks):  # line 3: for 1 <= i <= Tr(外层 Q 行块)
        Qi = Q[qr0:qr1]  # line 4: load Qi(本行块全程驻留片上)
        # line 5: O_i^{(0)} = 0, ℓ_i^{(0)} = 0, m_i^{(0)} = -inf(片上初始化)
        o_acc = np.zeros((qr1 - qr0, d))  # 未归一化累加器(Tweak 1)
        l_i = np.zeros(qr1 - qr0)
        m_i = np.full(qr1 - qr0, -np.inf)
        row_lo = qr0 + query_offset  # 本行块最小/最大 query 全局位置
        row_hi = qr1 - 1 + query_offset

        for j, (kc0, kc1) in enumerate(kv_blocks):  # line 6: for 1 <= j <= Tc
            if causal and kc0 > row_hi:
                # §3.1.1 Causal masking (1): 整块都在因果上侧,跳过(对角线以上约
                # 一半的块),约 1.7-1.8x 计算量红利即来自此。
                continue
            Kj, Vj = K[kc0:kc1], V[kc0:kc1]  # line 7: load Kj, Vj
            s = (Qi @ Kj.T) * scale  # line 8: S_i^{(j)} = Qi Kj^T
            if causal and kc1 - 1 > row_lo:
                # §3.1.1 Causal masking (2): 只有存在被遮列的块(跨对角)才施掩码;
                # 全可见块(kc1-1 <= row_lo)免掩码直通。
                keep = causal_keep_mask(
                    qr1 - qr0, kc1 - kc0, query_offset + qr0 - kc0
                )
                s = np.where(keep, s, -np.inf)

            # line 9: m^{(j)} = max(m^{(j-1)}, rowmax(S));P̃ = exp(S - m^{(j)})
            #         (注意:P̃ 相对新 max);ℓ^{(j)} = e^{m^{(j-1)}-m^{(j)}} ℓ^{(j-1)} + rowsum(P̃)
            m_new = np.maximum(m_i, s.max(axis=-1))
            with np.errstate(invalid="ignore"):
                p_tilde = _safe_exp(s - m_new[:, None])
                l_new = _safe_exp(m_i - m_new) * l_i + p_tilde.sum(axis=-1)

            # line 10(按不变式修正方向,见函数 docstring):
            #         O_acc <- e^{m^{(j-1)}-m^{(j)}} · O_acc + P̃^{(j)} V_j   (未归一化,不除 ℓ)
            with np.errstate(invalid="ignore"):
                o_acc = o_acc * _safe_exp(m_i - m_new)[:, None] + p_tilde @ Vj
            m_i, l_i = m_new, l_new
            if trace is not None:
                trace.append(
                    {
                        "i": i,
                        "j": j,
                        "q0": qr0,
                        "q1": qr1,
                        "m": m_i.copy(),
                        "l": l_i.copy(),
                        "O_unnormalized": o_acc.copy(),
                    }
                )

        has_key = l_i > 0  # 整行从未见过合法 key(如右下对齐下的空行):输出为零
        # line 12: O_i = diag(ℓ^{(Tc)})^{-1} O_acc —— 收尾只除一次(Tweak 1)
        O[qr0:qr1] = np.where(
            has_key[:, None], o_acc / np.where(has_key, l_i, 1.0)[:, None], 0.0
        )
        # line 13: L_i = m^{(Tc)} + log(ℓ^{(Tc)}) —— 只存这一个统计量(Tweak 2)
        L[qr0:qr1] = np.where(has_key, m_i + np.log(np.where(has_key, l_i, 1.0)), -np.inf)

    return O, L  # line 17: Return the output O and the logsumexp L


# PAPER: §3.2 Theorem 2 (标准侧 Θ(Nd + N²)) —— Algorithm 0 三步的元素级访存账:
# line 1 读 Q,K 写 S;line 2 读 S 写 P;line 3 读 P,V 写 O。N≫d 时 N² 项主导。
def hbm_accesses_standard(seq_len_n: int, head_dim_d: int) -> int:
    n, d = seq_len_n, head_dim_d
    step1 = n * d + n * d + n * n  # line 1: read Q,K; write S
    step2 = n * n + n * n  # line 2: read S; write P
    step3 = n * n + n * d + n * d  # line 3: read P,V; write O
    return step1 + step2 + step3


# PAPER: §3.2 Theorem 2 (FlashAttention 侧 Θ(N²d²/M)) 的证明梗概 —— "given SRAM
# size M, load blocks of K,V of size Θ(M); for each block iterate over all blocks of
# Q (Θ(Nd/M) passes); each pass loads Θ(Nd) elements ⇒ Θ(N²d²/M) accesses"。按
# Algorithm 1 的每行精确计数:外层每个 j 只搬一次 Kj,Vj(合计 Θ(Nd));但内层对
# 每个 j 都要把整个 Q,O,ℓ,m 重过一遍(line 8 读 + line 12-13 写),共
# Tc = ceil(N/Bc) 遍 —— Bc 越大遍数越少、访存越少(§3.2:块再大下去收益封顶、
# 且超出 SRAM 容量;渐近项只依赖 N, d, Bc,Br 仅伴同 Alg.1 的 (Bc,Br) 配对保留)。
# PAPER: §3.2 Theorem 2 证明梗概 (FlashAttention 侧 Θ(N²d²/M))
def hbm_accesses_flash(
    seq_len_n: int, head_dim_d: int, block_size_c: int, block_size_r: int = None
) -> int:
    n, d = seq_len_n, head_dim_d
    t_c = math.ceil(n / block_size_c)
    outer = 2 * n * d  # 每个 j 恰好搬一次 Kj,Vj,合计正好整个 K,V 各一遍
    inner_per_pass = 3 * n * d + 4 * n  # 每次重扫 Q,O,ℓ,m:读 (2Nd+2N) + 写 (Nd+2N)
    return outer + t_c * inner_per_pass
