"""ch32 §3.3.2 Eq.(8)-(12) —— NSA token 选择支路:复用压缩注意力分数诱导块重要性,top-n 选块。

核心洞见(Eq.8):块重要性打分不必额外算——直接复用"压缩注意力"分支已经算过的中间
softmax 分数 p_t^cmp。Eq.9 做块粒度对齐(压缩块大小 l/滑动步长 d 与选择块大小 l' 不同时的
映射);Eq.10 在 GQA/MQA 组内跨头求和,保证同组内所有 query 头共享同一份块选择(省 KV
cache 加载);Eq.11-12 按重要性排名截断,只对 top-n 个块内的 token 做真实注意力。

这是"打分函数为何能代理相关性"这个推导论点在 NSA 里的第一次出现——DSA 的 lightning
indexer(见 lightning_indexer.py)是它的简化后裔:用一个独立小打分器代替"复用压缩注意力
分数"这一招,但"排序打分 -> 截断 -> 只算选中项"的逻辑完全一致。
"""
import numpy as np


# PAPER: §3.3.2 Eq.(8) —— p_t^cmp = Softmax(q_t^T K~_t^cmp),压缩注意力的中间打分,
# 被直接复用为块重要性的来源(不需要额外计算)
def compression_attention_scores(q_t: np.ndarray, k_cmp: np.ndarray) -> np.ndarray:
    logits = k_cmp @ q_t
    logits = logits - logits.max()
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum()


# PAPER: §3.3.2 Eq.(9) —— l<=l', d|l, d|l' 时的块粒度映射 p_t^slc[j] =
# Sum_{m=0}^{l'/d-1} Sum_{n=0}^{l/d-1} p_t^cmp[l'/d*j - m - n];
# l'=l=d 特例(论文明说"直接得到 p_slc=p_cmp")时跳过映射直接复用
def block_importance_from_compression(p_cmp: np.ndarray, l: int, d: int, l_prime: int) -> np.ndarray:
    if l_prime == l == d:
        return p_cmp.copy()
    n_blocks = len(p_cmp) * d // l_prime
    p_slc = np.zeros(max(n_blocks, 0))
    for j in range(len(p_slc)):
        acc = 0.0
        for m in range(l_prime // d):
            for n in range(l // d):
                idx = (l_prime // d) * j - m - n
                if 0 <= idx < len(p_cmp):
                    acc += p_cmp[idx]
        p_slc[j] = acc
    return p_slc


# PAPER: §3.3.2 Eq.(10) —— p_t^{slc'} = Sum_{h=1}^{H} p_t^{slc,(h)},GQA/MQA 组内跨头求和,
# 保证同组内所有 query 头共享同一份块选择(减少 KV cache 加载)
def gqa_group_importance(p_slc_per_head: np.ndarray) -> np.ndarray:
    """p_slc_per_head:(H, n_blocks)。返回 (n_blocks,) 组内求和后的 p_t^{slc'}。"""
    return p_slc_per_head.sum(axis=0)


# PAPER: §3.3.2 Eq.(11) —— I_t = {i | rank(p_t^{slc'}[i]) <= n},top-n 块下标
# (rank=1 对应最高分,降序排名)
def topn_block_selection(p_slc_prime: np.ndarray, n: int) -> np.ndarray:
    n = min(n, len(p_slc_prime))
    return np.argsort(-p_slc_prime, kind="stable")[:n]


# PAPER: §3.3.2 Eq.(12) —— K~_t^slc = Cat[{k_{il'+1:(i+1)l'} | i in I_t}],
# 拼出被选中块内的真实 key(value 同理)
def gather_selected_blocks(k_seq: np.ndarray, selected_blocks: np.ndarray, l_prime: int) -> np.ndarray:
    chunks = [k_seq[i * l_prime:(i + 1) * l_prime] for i in sorted(selected_blocks.tolist())]
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, k_seq.shape[-1]))
