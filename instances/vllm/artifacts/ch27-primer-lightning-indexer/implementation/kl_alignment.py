"""arXiv:2512.02556 §2.1.1 "Continued Pre-Training" —— 让 lightning indexer 的打分
对齐主注意力质量分布的两阶段训练目标：Dense Warm-up（Eq.3，全序列）与
Sparse Training（Eq.4，只在选中集 S_t 上）。这两条损失解释了"为何独立小头可信"这一
设计选择在训练侧的根据——indexer 只受 L^I 监督，不接受主模型语言建模损失的梯度。
"""
import numpy as np


# "we first aggregate the main attention scores by summing across all attention
# heads. This sum is then L1-normalized ... to produce a target distribution
# p_{t,:}" —— 构造监督信号的确定性步骤，不是损失本身。
# PAPER: §2.1.1 "Dense Warm-up Stage"
def main_attention_target_distribution(attn_scores: np.ndarray) -> np.ndarray:
    """
    attn_scores: [T, H, S] 主注意力对每个 query token、各头、在 S 个历史 token 上
                 的（已过 softmax 的）注意力权重
    returns p: [T, S] 各头求和后沿序列维 L1 归一化的目标分布
    """
    assert attn_scores.ndim == 3
    summed = attn_scores.sum(axis=1)  # 各头求和 -> [T, S]
    return summed / summed.sum(axis=-1, keepdims=True)  # 沿序列维 L1 归一化


# PAPER: §2.1.1 Eq.(3)/(4) —— Softmax(I_{t,:}) / Softmax(I_{t,S_t}) 的共享子例程。
def _softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


# PAPER: §2.1.1 Eq.(3)/(4) —— D_KL(p||Softmax(I)) 的逐行求和实现。
def _kl_row(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """D_KL(p || q) 逐元素求和；p、q 按公式原样传入，不额外重新归一化——
    Eq.(4) 里 p_{t,S_t} 是 p_{t,:} 限制到选中集后的子向量，未必求和为 1
    （限制一个真分布到子集，天然会丢掉部分概率质量），这是照字面公式实现的直接
    后果，不是本参考实现新引入的近似。"""
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return float(np.sum(p * np.log(p / q)))


# PAPER: §2.1.1 Eq.(3) —— Dense Warm-up:
#   L^I = sum_t D_KL(p_{t,:} || Softmax(I_{t,:}))
# 全序列上的 KL 对齐；此阶段冻结除 indexer 外的全部参数、仍用稠密注意力。
def dense_warmup_loss(index_scores: np.ndarray, p_target: np.ndarray) -> float:
    """
    index_scores: [T, S] 全序列的 I_{t,:}
    p_target:     [T, S] main_attention_target_distribution 的输出
    """
    assert index_scores.shape == p_target.shape
    q = _softmax(index_scores)
    total = 0.0
    for t in range(index_scores.shape[0]):
        total += _kl_row(p_target[t], q[t])
    return total


# PAPER: §2.1.1 Eq.(4) —— Sparse Training:
#   L^I = sum_t D_KL(p_{t,S_t} || Softmax(I_{t,S_t})), S_t = Top-k(I_{t,:})
# 只在选中集 S_t 上对齐——训练目标与推理时实际被下游消费的子集一致。
def sparse_training_loss(
    index_scores: np.ndarray, p_target: np.ndarray, topk_idx: np.ndarray
) -> float:
    """
    topk_idx: [T, k] 每行选中集 S_t 的索引（与 lightning_indexer.topk_select
              输出同构）；Softmax(I_{t,S_t}) 是在选中子集上重新做的 softmax
              （只在 k 个分数间归一化，而非对整行 S 个分数）。
    """
    assert index_scores.shape[0] == p_target.shape[0] == topk_idx.shape[0]
    total = 0.0
    for t in range(topk_idx.shape[0]):
        idx = topk_idx[t]
        p_sub = p_target[t, idx]  # p_{t,S_t}：限制到选中集，不重新归一化
        q_sub = _softmax(index_scores[t, idx])  # Softmax(I_{t,S_t})：在子集上重新 softmax
        total += _kl_row(p_sub, q_sub)
    return total


# "we detach the indexer input ... for separate optimization. The training
# signal of the indexer is from only L^I" —— 本参考实现无 autograd（纯前向数值），
# 用恒等拷贝把这句训练协议语义化为可读代码契约（不是真正的梯度阻断机制）。
# PAPER: §2.1.1 "Continued Pre-Training" (detach for separate optimization)
def detach(x: np.ndarray) -> np.ndarray:
    return np.array(x, copy=True)
