"""arXiv:1805.02867 (Milakov & Gimelshein, "Online normalizer calculation for softmax")
§2-3 —— naive softmax(两遍,溢出) -> safe softmax(三遍,数值稳定) -> online softmax
(一遍,同样稳定)的收敛过程,以及 §3.1 Eq.3-4 的结合律 ⊕ 合并算子(FlashAttention 分块
tiling 与 vLLM merge_attn_states 的数学地基)。
"""
import numpy as np


# PAPER: §2 Eq.(1), Algorithm 1 —— 朴素 softmax:两遍扫描(求归一项 d_V,再算 y_i),
# 无 max 平移,在有限数值范围的真实硬件上会上溢/下溢(论文原句:"the line 3 of the
# algorithm 1 can overflow or underflow due to the exponent")
def naive_softmax(x: np.ndarray) -> np.ndarray:
    d_v = np.exp(x).sum()
    return np.exp(x) / d_v


# PAPER: §2 Eq.(2), Algorithm 2 —— 安全 softmax:三遍扫描(求 max m_V、求归一项 d_V、
# 算 y_i),减去逐元素最大值防止上溢/下溢,是当前主流框架(TensorFlow/PyTorch/…)采用的版本
def safe_softmax(x: np.ndarray) -> np.ndarray:
    m_v = x.max()
    d_v = np.exp(x - m_v).sum()
    return np.exp(x - m_v) / d_v


# PAPER: §3 Algorithm 3 lines 1-6 —— 单遍在线归一化:同时维护 running max m_j 与
# rescale 后的 running sum d_j。每来一个新元素 x_j,先把旧的 d_{j-1} 按新旧 max 之差
# exp(m_{j-1}-m_j) 缩放,再加上新项 exp(x_j-m_j)——这是分块 softmax 的数学地基。
def online_softmax_stats(x: np.ndarray) -> tuple:
    m = -np.inf
    d = 0.0
    for xj in x:
        m_new = max(m, xj)
        d = d * np.exp(m - m_new) + np.exp(xj - m_new)
        m = m_new
    return m, d


# PAPER: §3 Algorithm 3 lines 1-9 —— 完整单遍在线 softmax:先用 online_softmax_stats
# 单遍算出 (m_V,d_V),再用它们算每个元素的输出值 y_i(第二个 for 循环,与 safe softmax
# 的第三遍数学等价,只是第一、二遍已经融合成了一遍)
def online_softmax(x: np.ndarray) -> np.ndarray:
    m_v, d_v = online_softmax_stats(x)
    return np.exp(x - m_v) / d_v


# PAPER: §3.1 Eq.(4) —— 二元结合算子 ⊕:[m_i;d_i] ⊕ [m_j;d_j] = [max(m_i,m_j);
# d_i*exp(m_i-max)+d_j*exp(m_j-max)]。论文证明它满足结合律与交换律,因此 softmax
# 的归一化统计量可以任意分块/并行/乱序归并——FlashAttention 的 tiling 递推与 vLLM
# merge_attn_states 的 LSE 合并都是这个算子的具体化。
# PAPER: §3.1 Eq.(4)
def online_softmax_merge(state_i: tuple, state_j: tuple) -> tuple:
    m_i, d_i = state_i
    m_j, d_j = state_j
    m_new = max(m_i, m_j)
    d_new = d_i * np.exp(m_i - m_new) + d_j * np.exp(m_j - m_new)
    return m_new, d_new


# PAPER: §3.1 Eq.(3) —— 把 x 切成若干块,每块先算局部 (m,d)(Algorithm 3 的单块特例),
# 再用 ⊕ 依次归并成全局 (m_V,d_V)。用来在 worked example 里验证:分块合并 == 一次性
# 单遍遍历 == 三遍 safe softmax,三者数值恒等(结合律保证与合并顺序无关)。
def combine_blocks_via_merge(x: np.ndarray, block_size: int) -> tuple:
    blocks = [x[i : i + block_size] for i in range(0, len(x), block_size)]
    state = online_softmax_stats(blocks[0])
    for block in blocks[1:]:
        state = online_softmax_merge(state, online_softmax_stats(block))
    return state
