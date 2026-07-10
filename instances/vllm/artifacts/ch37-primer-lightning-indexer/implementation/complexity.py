"""arXiv:2512.02556 §2.3 "Inference Costs" —— DSA 把主注意力的复杂度从 O(L^2) 降到
O(Lk)，但 lightning indexer 自身打分仍是 O(L^2)（只是常数远小于原本的 MLA 全头计算）。
这里不满足于渐近符号，而是用"每 query-key 对一次核算"的逐元素计数，把这句"诚实账"
落成可代入具体 L、k 的数字。
"""


# 稠密（因果）注意力对第 t 个 query 要核算 t+1 个历史 token；求和得到 O(L^2)/2 量级。
# 稀疏时每个 query 固定只核算 min(t+1, k) 个 token（因果掩码下 t 前历史 token 数不足
# k 时退化为稠密，这是因果 top-k 选择的自然边界，不是额外发明的特殊情况）。
# PAPER: §2.3 —— "DSA reduces the core attention complexity ... from O(L^2) to O(Lk)"
def main_attention_ops(seq_len: int, topk: int | None = None) -> int:
    if topk is None:
        return sum(t + 1 for t in range(seq_len))  # O(L^2)
    assert topk > 0
    return sum(min(t + 1, topk) for t in range(seq_len))  # O(Lk)


# indexer 打分对每个 query 都要扫过它之前的全部历史 token——与稠密注意力同构的
# O(L^2)，只是每次核算的常数（indexer 头数少、可 FP8/FP4）远小于主 MLA 的常数。
# cost_ratio 把"常数远小"这句定性描述换成可乘的比例，供 worked example 代入。
# PAPER: §2.3 —— "the lightning indexer still has a complexity of O(L^2), ... much
# less computation compared with MLA"
def indexer_ops(seq_len: int, cost_ratio: float = 1.0) -> float:
    assert 0.0 <= cost_ratio
    return main_attention_ops(seq_len, topk=None) * cost_ratio


# PAPER: §2.3 —— 主注意力核算量的稠密/稀疏比值；当 L >> k 时趋近 L/(2k)
# （因果掩码下稠密总量约 L^2/2，稀疏总量约 L*k，两者之比约 L/(2k)）。
def speedup_ratio(seq_len: int, topk: int) -> float:
    dense = main_attention_ops(seq_len, topk=None)
    sparse = main_attention_ops(seq_len, topk=topk)
    return dense / sparse
