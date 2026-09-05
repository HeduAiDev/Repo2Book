"""arXiv:1805.02867 (Milakov & Gimelshein, "Online normalizer calculation for softmax")
§2-§3 —— naive softmax(两遍扫描,数值范围有限的真实硬件上会溢出)→ safe softmax
(三遍扫描,减 max 稳定,主流框架所用)→ online softmax(单遍扫描,同样稳定)的
收敛过程;§3.1 Eq.(3)-(4) —— 把 (m, d) 抽象为二元算子 ⊕(论文声明其满足结合律
与交换律,故归一化统计量可任意分块、乱序归并)。这一页数学是 FlashAttention tiling
(arXiv:2205.14135 §3.1 Algorithm 1)与 vLLM merge_attn_states(LSE 合并)共同的
地基。

「trace」可选参数只逐格记录算法自身循环变量的快照(供示教轨迹用),不是论文之外
的新机制。
"""
import numpy as np


# PAPER: §2 Eq.(1), Algorithm 1 —— 朴素 softmax:两遍扫描(先求归一项 d_V,再算
# y_i),无 max 平移。论文原话:"the line 3 of the algorithm 1 can overflow or
# underflow due to the exponent" —— e^1000 在 fp32 上就是 inf。
def naive_softmax(x: np.ndarray) -> np.ndarray:
    d_v = np.exp(x).sum()
    return np.exp(x) / d_v


# PAPER: §2 Eq.(2), Algorithm 2 —— 安全 softmax:三遍扫描(求 max m_V → 求归一项
# d_V → 算 y_i),减去逐元素最大值后不溢出;TensorFlow/PyTorch/MXNet 等主流框架
# 均用此版(论文 §2 列了各家版本号)。
def safe_softmax(x: np.ndarray) -> np.ndarray:
    m_v = x.max()
    d_v = np.exp(x - m_v).sum()
    return np.exp(x - m_v) / d_v


# PAPER: §3 Algorithm 3 lines 1-6 —— 单遍在线归一化:同时维护 running max m_j 与
# rescale 后的 running sum d_j。每来一个新元素 x_j,先把旧账 d_{j-1} 按新旧 max
# 之差 e^{m_{j-1}-m_j} 折算,再累加新项 e^{x_j-m_j}。trace 给出时逐元素追加
# (m_j, d_j) 快照。
# PAPER: §3 Algorithm 3 lines 1-6
def online_softmax_stats(x: np.ndarray, trace: list = None) -> tuple:
    m = -np.inf
    d = 0.0
    for xj in x:
        m_new = max(m, xj)
        d = d * np.exp(m - m_new) + np.exp(xj - m_new)
        m = m_new
        if trace is not None:
            trace.append((m, d))
    return m, d


# PAPER: §3 Algorithm 3 lines 1-9 —— 完整单遍在线 softmax:lines 1-6 单遍算出
# (m_V, d_V),lines 7-9 再算每个 y_i(与 safe softmax 的第三遍数学等价,只是前两
# 遍已融合成一遍:每元素访存 3 次 vs safe 的 4 次)。Theorem 1 保证末值与 safe
# softmax 恒等。
# PAPER: §3 Algorithm 3 lines 1-9 + Theorem 1
def online_softmax(x: np.ndarray) -> np.ndarray:
    m_v, d_v = online_softmax_stats(x)
    return np.exp(x - m_v) / d_v


# PAPER: §3.1 Eq.(4) —— 二元结合算子 ⊕:[m_i; d_i] ⊕ [m_j; d_j] =
# [max(m_i, m_j); d_i·e^{m_i-max} + d_j·e^{m_j-max}]。论文声明 ⊕ 满足结合律
# (可任意括号分块、并行求值)与交换律(可乱序归并)。FlashAttention 的 tiling
# 递推与 vLLM merge_attn_states 的 LSE 合并都是这个算子的具体化。
# PAPER: §3.1 Eq.(4)
def online_softmax_merge(state_i: tuple, state_j: tuple) -> tuple:
    m_i, d_i = state_i
    m_j, d_j = state_j
    m_new = max(m_i, m_j)
    d_new = d_i * np.exp(m_i - m_new) + d_j * np.exp(m_j - m_new)
    return m_new, d_new


# PAPER: §3.1 Eq.(3) —— 把 x 切成若干块,每块先算局部 (m, d)(Algorithm 3 的单块
# 特例),再用 ⊕ 依次归并成全局 (m_V, d_V)。Eq.(3) 从左到右顺序应用 == 逐元素跑
# Algorithm 3 lines 1-6;结合律+交换律则保证任何分块方式、任何归并顺序都得到同一
# 末值 —— tiling 与并行归并的合法性来源。
# PAPER: §3.1 Eq.(3)
def combine_blocks_via_merge(x: np.ndarray, block_size: int) -> tuple:
    blocks = [x[i : i + block_size] for i in range(0, len(x), block_size)]
    state = online_softmax_stats(blocks[0])
    for block in blocks[1:]:
        state = online_softmax_merge(state, online_softmax_stats(block))
    return state
