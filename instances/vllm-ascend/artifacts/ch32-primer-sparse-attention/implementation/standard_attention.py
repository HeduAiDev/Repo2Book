"""ch32 §3.1 Eq.(1)-(2) —— 标准因果自注意力,O(L^2) 注意力税的算式来源。

NSA 论文(arXiv:2502.11089)§3.1 Background:每个 query token q_t 对全部前驱 key k_{:t} 打分、
softmax 归一、对 value 加权求和(Eq.2)。对整条长度 L 的序列,query t 要处理 t 个前驱,
Sum_{t=1}^{L} t = L(L+1)/2 = Theta(L^2) 次 q.k 点积——这就是"注意力税"的定量来源,
论文估计 64K 解码时注意力占总延迟 70-80%(§1 Introduction)。
"""
import numpy as np


# PAPER: §3.1 Eq.(2) —— alpha_{t,i}=exp(q_t^T k_i / sqrt(d_k)),标准 softmax 注意力权重
def causal_attention_scores(q_t: np.ndarray, k_seq: np.ndarray, d_k: int) -> np.ndarray:
    """对单个 query 向量 q_t(d_k,)与前驱 keys k_seq=k_{:t}(t,d_k)算 softmax 权重 alpha_{t,:}。

    减去 max 只是数值稳定手段,不改变论文公式的数学结果(softmax 平移不变性)。
    """
    logits = (k_seq @ q_t) / np.sqrt(d_k)  # (t,)
    logits = logits - logits.max()
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum()


# PAPER: §3.1 Eq.(1)-(2) —— o_t = Attn(q_t,k_{:t},v_{:t}) = Sum_i alpha_{t,i} v_i
def causal_attention_output(q_t: np.ndarray, k_seq: np.ndarray, v_seq: np.ndarray, d_k: int) -> np.ndarray:
    alpha = causal_attention_scores(q_t, k_seq, d_k)
    return alpha @ v_seq


# PAPER: §3.1 文字("As sequence length increases, attention computation becomes increasingly
# dominant in the overall computational cost") —— 整条长度 L 序列因果注意力的 q.k 点积总次数,
# 精确值(非近似):Sum_{t=1}^{L} t = L(L+1)/2
def quadratic_dot_product_count(seq_len: int) -> int:
    return seq_len * (seq_len + 1) // 2


# PAPER: §3.1 文字 —— 注意力税的 FLOPs 账本:每次 q.k 点积是 d_k 维乘加,
# 整条序列总代价 ~= L(L+1)/2 * d_k(只计点积主项,不计 softmax/加权求和的低阶项)
def quadratic_attention_flops(seq_len: int, d_k: int) -> int:
    return quadratic_dot_product_count(seq_len) * d_k
