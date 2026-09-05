"""SmoothQuant —— arXiv:2211.10438 §4 Eq.3(数学等价的逐通道缩放
Y = (X·diag(s)^{-1})·(diag(s)·W) = X̂·Ŵ,s 离线折进前一层、运行期零开销)、
Eq.4(迁移强度 α 配平:s_j = max|X_j|^α/max|W_j|^{1-α};α=0 权重全扛、
α=1 激活全扛,两个极端都崩,§5.5 Figure 10 的甜点在 0.4-0.6);§2 Eq.1 的
对称 INT8 量化器做 W8A8 per-tensor 模拟(§4:"using the same quantizer for
weights and activations (e.g., per-tensor, static quantization)")。

布局约定:X (T, C_i)、W (C_i, C_o)、Y = X @ W(§2 Figure 3 记法);
逐通道 j 指输入通道维(X 的列、W 的行)。granularity 用 uniform_quant 的
per-tensor 档(激活 per-token/per-channel 档在 uniform_quant.py,粒度谱归
m02 讲)。
"""
import numpy as np

from uniform_quant import dequantize_symmetric, quantize_symmetric


# PAPER: arXiv:2211.10438 §4 Eq.4 —— s_j = max(|X_j|)^α / max(|W_j|)^{1-α},
# j = 1..C_i;α 控制「量化难度从激活搬多少进权重」。
def smooth_scale(X, W, alpha):
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)
    act_max = np.abs(X).max(axis=0)  # max|X_j|(逐输入通道,跨 token)
    w_max = np.abs(W).max(axis=1)  # max|W_j|(逐输入通道,跨输出通道)
    return act_max ** alpha / w_max ** (1.0 - alpha)


# PAPER: arXiv:2211.10438 §4 Eq.3 —— X̂ = X·diag(s)^{-1}(列除 s)、
# Ŵ = diag(s)·W(行乘 s);任意 s 下 X̂·Ŵ == X·W(严格等价,浮点可验)。
def apply_smoothing(X, W, s):
    s = np.asarray(s, dtype=float)
    X_hat = np.asarray(X, dtype=float) / s[None, :]
    W_hat = np.asarray(W, dtype=float) * s[:, None]
    return X_hat, W_hat


# PAPER: arXiv:2211.10438 §2 Eq.1 + §4 —— W8A8 per-tensor(静态对称)模拟:
# 激活与权重各用一把整张矩阵的尺子(Δ = max|·|/(2^{N-1}-1)),量化-反量化后
# 相乘;对照 FP16 的 Y = X@W 即 SmoothQuant 的精度记分板。
def w8a8_per_tensor_output(X, W, num_bits=8):
    q_x, d_x = quantize_symmetric(X, num_bits)
    q_w, d_w = quantize_symmetric(W, num_bits)
    return dequantize_symmetric(q_x, d_x) @ dequantize_symmetric(q_w, d_w)


# PAPER: arXiv:2211.10438 §4 Eq.3-Eq.4 —— 先按 α 平滑再做 W8A8 per-tensor
# 量化,输出误差 ‖Y_sim − Y_fp16‖_F。α 的一维扫描即 §5.5 Figure 10 的协议。
def w8a8_output_error(X, W, alpha, num_bits=8):
    s = smooth_scale(X, W, alpha)
    X_hat, W_hat = apply_smoothing(X, W, s)
    Y_sim = w8a8_per_tensor_output(X_hat, W_hat, num_bits)
    return float(np.linalg.norm(Y_sim - np.asarray(X, dtype=float) @ np.asarray(W, dtype=float)))


# PAPER: arXiv:2211.10438 §5.5 Figure 10 —— 迁移强度消融:α 从 0 扫到 1,
# 两端(激活难量化 / 权重难量化)误差都大、甜点在中段(OPT/BLOOM 0.5)。
def migration_ablation(X, W, num_bits=8, alphas=None):
    if alphas is None:
        alphas = np.linspace(0.0, 1.0, 9)
    return [
        (float(a), w8a8_output_error(X, W, a, num_bits)) for a in alphas
    ]
