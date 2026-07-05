"""ch32 §2.1.1 Eq.(3)-(4)(paper-dsa.md) —— 训练协同适配:dense warm-up + sparse stage 的
KL 对齐。这是"top-k 稀疏为何不掉点"的答案——不是因为打分函数天生准,而是训练把它
对齐到了真实注意力分布上。

dense warm-up(Eq.3):冻结主模型,只训 indexer。对每个 query token,把主注意力所有头的
分数求和后 L1 归一化,得到目标分布 p_{t,:};用 KL 散度把 Softmax(I_{t,:}) 对齐到 p_{t,:}。
sparse stage(Eq.4):引入 top-k 后放开全参微调,KL 只在选中集 S_t = {s | I_{t,s} in
Top-k(I_{t,:})} 上算,indexer 输入 detach、只由 L^I 训练,主模型只由 LM loss 训练。

论文没有给出可复现的梯度下降训练代码(那超出 primer 与 arXiv 论文本身的范围)——本文件
把 Eq.(3)/(4) 的损失函数本身做成可调用的度量,并用一个"indexer 对齐程度"旋钮做数值实验,
演示低 KL(对齐好)如何直接换来高 top-k 注意力质量召回(mass recall)——这就是 NSA §2.2
引 Chen et al. 2024b 指出的"post-hoc 剪枝掉点"问题(top 20% attention 只覆盖 70% 总分数)
在 DSA 训练协同适配下如何被解决的定量说明。
"""
import numpy as np


# PAPER: §2.1.1 文字("we first aggregate the main attention scores by summing across all
# attention heads...L1-normalized...to produce a target distribution p_{t,:}")
# —— p_{t,:} 的构造:多头主注意力分数求和后 L1 归一化
def aggregate_main_attention(head_scores: np.ndarray) -> np.ndarray:
    """head_scores:(H, t),每个主注意力头对前驱 token 的 softmax 分数。
    返回 p_{t,:} 形状 (t,),各元素求和为 1(L1 归一化)。"""
    summed = head_scores.sum(axis=0)
    total = summed.sum()
    if total == 0:
        return np.zeros_like(summed)
    return summed / total


# PAPER: §2.1.1 Eq.(3) —— L^I = Sum_t D_KL(p_{t,:} || Softmax(I_{t,:})),dense warm-up 损失,
# 对齐 indexer 输出与主注意力聚合分布
def dense_warmup_kl(p: np.ndarray, indexer_scores: np.ndarray, eps: float = 1e-12) -> float:
    q = _softmax(indexer_scores)
    return _kl_divergence(p, q, eps)


# PAPER: §2.1.1 Eq.(4) —— L^I = Sum_t D_KL(p_{t,S_t} || Softmax(I_{t,S_t})),sparse stage
# 损失,S_t = {s | I_{t,s} in Top-k(I_{t,:})},只在选中集上重新归一化后算 KL
def sparse_stage_kl(
    p: np.ndarray, indexer_scores: np.ndarray, topk_indices: np.ndarray, eps: float = 1e-12
) -> float:
    p_restricted = p[topk_indices]
    denom = p_restricted.sum()
    p_restricted = p_restricted / denom if denom > eps else p_restricted
    q_restricted = _softmax(indexer_scores[topk_indices])
    return _kl_divergence(p_restricted, q_restricted, eps)


# PAPER: §2.2 文字(引 Chen et al. 2024b,"top 20% attention can only cover 70% of the total
# attention scores") —— 度量:top-k(indexer 打分) 命中的 KV 条目占真实注意力质量 p 的比例。
# 这是"top-k 是否掉点"的定量指标:trained indexer 应让这个值接近 1,乱打分的 indexer 会很低。
def topk_mass_recall(p: np.ndarray, indexer_scores: np.ndarray, k: int) -> float:
    k = min(k, len(indexer_scores))
    topk_indices = np.argsort(-indexer_scores, kind="stable")[:k]
    return float(p[topk_indices].sum())


# 用"对齐程度"旋钮 alpha 插值出 indexer 打分:alpha=0 时 indexer 完全对齐真实分布 p
# (对应 dense warm-up + sparse stage 训练收敛后的理想 indexer),alpha=1 时 indexer 完全
# 随机(对应未训练 / post-hoc 启发式打分)。用于演示 KL 损失与 top-k 质量召回之间的定量关系
# PAPER: §2.1.1 文字 —— 这正是 Eq.(3)/(4) 存在的意义:压低 KL 就是在提升 top-k 召回
def simulate_indexer_logits(p: np.ndarray, alpha: float, rng: np.random.Generator) -> np.ndarray:
    aligned_logits = np.log(p + 1e-12)
    random_logits = rng.normal(size=p.shape)
    # 标准化到相近尺度,避免其中一路数值量级压制另一路,只让 alpha 控制"对齐 vs 随机"的混合比
    aligned_logits = (aligned_logits - aligned_logits.mean()) / (aligned_logits.std() + 1e-12)
    random_logits = (random_logits - random_logits.mean()) / (random_logits.std() + 1e-12)
    return (1 - alpha) * aligned_logits + alpha * random_logits


# PAPER: §2.1.1 Eq.(3)/(4) 里的 Softmax(I_{t,:}) —— indexer 打分转成分布,供 KL 度量用
def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max()
    e = np.exp(shifted)
    return e / e.sum()


# PAPER: §2.1.1 Eq.(3)/(4) 里的 D_KL(.||.) 算子本身
def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log((p[mask] + eps) / (q[mask] + eps))))
