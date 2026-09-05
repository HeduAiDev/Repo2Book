"""均匀量化底座 —— arXiv:2211.10438 (SmoothQuant) §2 Eq.1(整数均匀量化标准式
X_bar = round(X/Δ)、Δ = max|X|/(2^{N-1}-1),对称假设 0 居中)与 §2 Figure 3
(粒度谱:per-tensor / per-token / per-channel);§3 obs.2(有效量化级数
2^N·m_i/m —— RTN 之死的定量版);非对称 min-max 网格 + zero-point 取
arXiv:2210.17323 (GPTQ) §5 Setup("standard uniform per-row asymmetric
quantization on the min-max grid",GPTQ 与 RTN 共用的那副网格)与 SmoothQuant
§2 的 zero-point 提法(after ReLU 等偏置分布加零点)。

这一页是三篇论文共同的地基:GPTQ 的 quant() 在这副网格上找补、AWQ Eq.1 的
Δ 换了分母约定(2^{N-1})、SmoothQuant Eq.3/Eq.4 只是给这副网格换尺子。
对应 vLLM 参考实现 vllm/model_executor/layers/quantization/utils/quant_utils.py
的 quantize_weights(对称/非对称两分支)。

约定:x 为 1-D 向量或 2-D 矩阵;2-D 时行 = token、列 = 输入通道(SmoothQuant
§2 Figure 3 的 X ∈ R^{T×C_i})。反量化误差上界 Δ/2 是 round-to-nearest 的
网格性质。np.round 为银行家舍入(0.5 取偶),论文未指定舍入细节、不影响性质。
"""
import numpy as np


# PAPER: arXiv:2211.10438 §2 Eq.1 —— 对称均匀量化:X_bar = round(X/Δ),
# Δ = max|X|/(2^{N-1}-1);整数码域 [-(2^{N-1}-1), 2^{N-1}-1]。
def quantize_symmetric(x, num_bits=8):
    qmax = 2 ** (num_bits - 1) - 1
    delta = np.max(np.abs(x)) / qmax
    q = np.clip(np.round(x / delta), -qmax, qmax).astype(np.int64)
    return q, float(delta)


# PAPER: arXiv:2211.10438 §2 Eq.1 —— 反量化 x_hat = q·Δ(Eq.1 的逆映射)。
def dequantize_symmetric(q, delta):
    return q * delta


# PAPER: arXiv:2211.10438 §2 Figure 3 —— per-tensor:整张矩阵一把尺子
# (单一 Δ;最粗、实现最省)。
def quantize_per_tensor(x, num_bits=8):
    return quantize_symmetric(x, num_bits)


# PAPER: arXiv:2211.10438 §2 Figure 3 —— per-token:每个 token(行)一把尺子
# (Δ 挂外维 T,GEMM 友好:vLLM QuantFP8 的 dynamic per-token 即此档)。
def quantize_per_token(x, num_bits=8):
    qmax = 2 ** (num_bits - 1) - 1
    deltas = np.max(np.abs(x), axis=1) / qmax  # (T,)
    q = np.clip(
        np.round(x / deltas[:, None]), -qmax, qmax
    ).astype(np.int64)
    return q, deltas


# PAPER: arXiv:2211.10438 §2 Figure 3 —— per-channel:每个输入通道(列)一把
# 尺子;对激活这挂在内维 C_i、INT8 GEMM kernel 不认(§3 Table 1 灰色行),
# 对权重则挂外维 C_o、是标准做法。
def quantize_per_channel(x, num_bits=8):
    qmax = 2 ** (num_bits - 1) - 1
    deltas = np.max(np.abs(x), axis=0) / qmax  # (C,)
    q = np.clip(np.round(x / deltas[None, :]), -qmax, qmax).astype(np.int64)
    return q, deltas


# PAPER: arXiv:2210.17323 §5 Setup + arXiv:2211.10438 §2 —— 非对称 min-max
# 网格:scale = (xmax-xmin)/(qmax-qmin)、zero-point zp = qmin - round(xmin/scale),
# 使 xmin 精确映到 qmin、xmax 精确映到 qmax;0 不必在格点上(ReLU 后分布)。
def quantize_asymmetric(x, num_bits=8):
    qmax = 2 ** (num_bits - 1) - 1
    qmin = -(2 ** (num_bits - 1))
    xmax = np.max(x)
    xmin = np.min(x)
    scale = max((xmax - xmin) / (qmax - qmin), 1e-12)  # 数值护栏:全常数向量
    zp = qmin - round(xmin / scale)
    q = np.clip(np.round(x / scale) + zp, qmin, qmax).astype(np.int64)
    return q, float(scale), int(zp)


# PAPER: arXiv:2210.17323 §5 Setup —— 反量化 x_hat = (q - zp)·scale。
def dequantize_asymmetric(q, scale, zero_point):
    return (q - zero_point) * scale


# PAPER: arXiv:2211.10438 §3 obs.2 —— 通道 i 的有效量化级数 = 2^N·m_i/m
# (m_i = 通道 max,m = 全张量 max):离群通道一进来,普通通道只剩 2-3 级。
def effective_quant_levels(channel_max, tensor_max, num_bits=8):
    return (2.0 ** num_bits) * channel_max / tensor_max
