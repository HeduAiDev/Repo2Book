"""AWQ —— arXiv:2306.00978 §3.1(显著通道「看激活不看权重」:按激活幅度选,
混合精度不可行的动机)、§3.2 Eq.1(组量化器 Q(w) = Δ·Round(w/Δ)、
Δ = max|w|/2^{N-1})、Eq.2(等价缩放 Q(w·s)(x/s))、Eq.3(两条误差表达式)、
obs.(1)-(3)(RoundErr ~ 0.25、Δ' ≈ Δ、FP16 中间量无量化误差)与误差比
(Δ'/Δ)·(1/s)、Table 2(1% 显著通道乘 s 的逐组统计协议)、Eq.4-Eq.5
(L(s) 目标与 s = s_X^α、α ∈ [0,1] 网格搜索,附录 grid size 20)、§4.2
(SIMD-aware packing:GPU 上把每 8 个权重按 w_{0,2,4,6,1,3,5,7} 打包——
vLLM awq_pack 的 interleave 与此一字不差)。

布局约定:X (n_tokens, C_i)、W (C_i, C_o)、Y = X @ W(SmoothQuant §2 的
记法);组 = 输入通道维上连续 group_size 个权重(每输出通道一列),
group_size 默认 128(附录:"We used a group size of 128 throughout the
work")。Eq.1 的网格以 ±max 为界(±2^{N-1}·Δ 都可达,round-trip 误差恒
≤ Δ/2)。np.round 为银行家舍入(0.5 取偶),论文未指定、不影响性质。
"""
import numpy as np


# PAPER: arXiv:2306.00978 §3.2 Eq.1 —— 组量化器:Q(w) = Δ·Round(w/Δ),
# Δ = max(|w|)/2^{N-1},组内 absmax 定尺(Eq.1 逐字;clip 只是 ±2^{N-1}
# 码域的浮点护栏,网格本身以 ±max 为界)。
def awq_group_quantize(w_group, num_bits=4):
    qmax = 2 ** (num_bits - 1)
    delta = np.max(np.abs(w_group)) / qmax
    q = np.clip(np.round(w_group / delta), -qmax, qmax).astype(np.int64)
    return q, float(delta)


# PAPER: arXiv:2306.00978 §3.2 Eq.1 —— 反量化 Q(w) 的实值 = q·Δ。
def awq_dequantize(q, delta):
    return q * delta


# PAPER: arXiv:2306.00978 §3.2 Eq.1 —— 整矩阵分组量化:每列沿输入通道维按
# group_size 分组,组内各自 Eq.1;返回反量化后的权重矩阵。
def awq_quantize_matrix(W, num_bits=4, group_size=128):
    W = np.asarray(W, dtype=float)
    d_in, d_out = W.shape
    W_hat = np.empty_like(W)
    for c in range(d_out):
        for g1 in range(0, d_in, group_size):
            g2 = min(g1 + group_size, d_in)
            q, delta = awq_group_quantize(W[g1:g2, c], num_bits)
            W_hat[g1:g2, c] = q * delta
    return W_hat


# PAPER: arXiv:2306.00978 §3.2 obs.(1) —— RoundErr(·) = Round(·) − (·):
# round 把浮点映到整数,误差大致均匀分布于 [0, 0.5],平均 0.25。
def round_err(u):
    return np.round(u) - u


# PAPER: arXiv:2306.00978 §3.2 Eq.3 第一式 —— Err(Q(w)x) = Δ·RoundErr(w/Δ)·x。
def err_Qwx(w, x, delta):
    return delta * round_err(w / delta) * x


# PAPER: arXiv:2306.00978 §3.2 Eq.3 第二式 —— Err(Q(w·s)(x/s)) =
# Δ'·RoundErr(w·s/Δ')·x·(1/s);比值 (Δ'/Δ)·(1/s) < 1 即「戴放大镜」的收益。
def err_Qws_xs(w, x, s, delta_prime):
    return delta_prime * round_err(w * s / delta_prime) * x / s


# PAPER: arXiv:2306.00978 §3.2(s_X 的定义:"the average magnitude of
# activation (per-channel)")—— 逐输入通道的平均激活幅度,Eq.5 的底数。
def channel_mean_activation(X):
    return np.abs(np.asarray(X, dtype=float)).mean(axis=0)


# PAPER: arXiv:2306.00978 §3.1 —— 显著通道按激活幅度(而非权重范数)选:
# "selecting weights based on activation magnitude can significantly improve
# the performance";Table 1 的对照即「按 W 选 ≈ 随机、按 X 选显著有效」。
def salient_channels(X, frac=0.01):
    s_x = channel_mean_activation(X)
    k = max(1, int(round(frac * len(s_x))))
    order = np.argsort(s_x)[::-1]
    mask = np.zeros(len(s_x), dtype=bool)
    mask[order[:k]] = True
    return mask


# PAPER: arXiv:2306.00978 §3.2 Table 2 协议 —— 给 1% 显著通道乘 s>1,逐组
# 统计 Δ 的变化:Δ'≠Δ 的比例、平均 Δ'/Δ、平均 (Δ'/Δ)·(1/s)(表 2 的三行)。
def table2_statistics(W, X, s=2.0, num_bits=4, group_size=128, salient_frac=0.01):
    W = np.asarray(W, dtype=float)
    d_in, d_out = W.shape
    mask = salient_channels(X, salient_frac)
    s_vec = np.ones(d_in)
    s_vec[mask] = s
    W_scaled = W * s_vec[:, None]
    changed = 0
    total = 0
    ratios = []
    for c in range(d_out):
        for g1 in range(0, d_in, group_size):
            g2 = min(g1 + group_size, d_in)
            _, delta = awq_group_quantize(W[g1:g2, c], num_bits)
            _, delta_p = awq_group_quantize(W_scaled[g1:g2, c], num_bits)
            total += 1
            ratio = delta_p / delta if delta > 0 else 1.0
            ratios.append(ratio)
            if delta_p > delta * (1 + 1e-9):
                changed += 1
    ratios = np.array(ratios)
    return {
        "proportion_delta_changed": changed / total,
        "mean_delta_ratio": float(np.mean(ratios)),
        "mean_scaled_error_ratio": float(np.mean(ratios / s)),
    }


# PAPER: arXiv:2306.00978 §3.2 Eq.4 —— 搜索目标 L(s) = ‖Q(W·diag(s))
# (diag(s)^{-1}·X) − WX‖:缩放后的量化输出对原输出的偏差(Q 即上面的分组
# 量化器;W·diag(s) 是行乘 s、diag(s)^{-1}·X 是列除 s)。
def awq_loss(W, X, s, num_bits=4, group_size=128):
    W = np.asarray(W, dtype=float)
    X = np.asarray(X, dtype=float)
    s = np.asarray(s, dtype=float)
    W_hat = awq_quantize_matrix(W * s[:, None], num_bits, group_size)
    return float(np.linalg.norm((X / s[None, :]) @ W_hat - X @ W))


# PAPER: arXiv:2306.00978 §3.2 Eq.5 + 附录(grid size 20)—— 搜索空间
# s = s_X^α,α 在 [0,1] 上均匀网格扫 20 点取 L(s) 最小者;α=0 即 s≡1(RTN),
# α=1 是空间内最激进缩放。单超参 α 平衡显著/非显著通道的保护。
def awq_search_scale(W, X, num_bits=4, group_size=128, grid_size=20):
    s_x = channel_mean_activation(X)
    history = []
    for alpha in np.linspace(0.0, 1.0, grid_size):
        alpha = float(alpha)
        loss = awq_loss(W, X, s_x ** alpha, num_bits, group_size)
        history.append((alpha, loss))
    best_alpha, _ = min(history, key=lambda t: t[1])
    return best_alpha, s_x ** best_alpha, history


# PAPER: arXiv:2306.00978 §4.2 SIMD-aware weight packing —— "On GPUs, we
# found it more efficient to pack each 8 weights into w_{0,2,4,6,1,3,5,7}":
# 每 8 个连续 4-bit 权重按偶下标在前重排后压进一个 32-bit 字(第 i 个 nibble
# 放重排后的第 i 个权重)。vLLM 的 awq_pack 在列维做同一 interleave。
# PAPER: arXiv:2306.00978 §4.2 SIMD-aware packing(w_{0,2,4,6,1,3,5,7})
def awq_pack(q, num_bits=4):
    if num_bits != 4:
        raise ValueError("论文 §4.2 只定义 4-bit 打包(interleave [0,2,4,6,1,3,5,7])")
    q = np.asarray(q).ravel().astype(np.int64)
    if len(q) % 8 != 0:
        raise ValueError("长度须为 8 的倍数")
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    grouped = q.reshape(-1, 8)[:, interleave]
    packed = np.zeros(len(grouped), dtype=np.int64)
    for i in range(8):
        # & 0xF:负码取 4-bit 二补码 nibble(负 int64 直接 OR 会污染高位)
        packed |= (grouped[:, i] & 0xF) << (4 * i)
    return packed.astype(np.int32)


# PAPER: arXiv:2306.00978 §4.2 —— awq_pack 的逆:逐 nibble 取出、符号扩展
# 回 [-8, 7] 码域、再逆 interleave 还原原顺序(打包正确性的对账件)。
def awq_unpack(packed, num_bits=4):
    if num_bits != 4:
        raise ValueError("论文 §4.2 只定义 4-bit 打包")
    p = np.asarray(packed).astype(np.int64).ravel()
    out = np.empty((len(p), 8), dtype=np.int64)
    for i in range(8):
        nib = (p >> (4 * i)) & 0xF
        out[:, i] = np.where(nib >= 8, nib - 16, nib)  # 4-bit 有符号符号扩展
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    inv = np.empty(8, dtype=np.int64)
    inv[interleave] = np.arange(8)
    return out[:, inv].ravel()
