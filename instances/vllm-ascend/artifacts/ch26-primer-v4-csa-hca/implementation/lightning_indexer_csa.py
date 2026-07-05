"""ch36 §2.3.1 Eq.(13)-(17) (paper.md, arXiv:2606.19348) -- CSA 的 lightning indexer:
对已压缩的 KV 条目再做一轮 top-k 稀疏选择(DSA 的直接复用,ch32 primer 已讲过 DSA 在
raw token 上怎么打分;这里把同一套打分公式套用到"压缩块"这个粒度上)。

流程(对应 Eq.13-17):
  1. 查询侧低秩 latent c_t^Q = h_t.W^{DQ}(Eq.13,与主注意力共享,见 shared_mqa_grouped_output.py)。
  2. indexer query q_t^I = c_t^Q.W^{IUQ},reshape 成 n_h^I 个头(Eq.14)。
  3. indexer 权重 w_t^I = h_t.W^w(Eq.15)。
  4. indexer key K^IComp:用与 C^Comp *同一个* 压缩算子(csa_compress_sequence,Eq.9-12)
     套一份 indexer 自己的权重得到——论文原文"CSA performs the same compression operation
     used for C^Comp to get compressed indexer keys"。落地:Indexer.__init__ 里
     self.compressor = Compressor(...) 是一个独立的 Compressor 实例,专给 indexer 用
     (vllm_ascend/models/deepseek_v4.py:L590-592)。
  5. 打分 I_{t,s} = sum_h w_{t,h}^I . ReLU(q_{t,h}^I . K_s^IComp)(Eq.16,ReLU 而非 softmax)。
  6. top-k 选出 C_t^SprsComp(Eq.17),因果约束 s < floor(t/m)。
"""
import numpy as np

from csa_compression import csa_compress_sequence


# PAPER: §2.3.1 Eq.(13) —— c_t^Q = h_t.W^{DQ}(与 shared_mqa_grouped_output 的注意力
# query 共享同一份低秩 latent,这是本章"共享 KV 的 MQA"设计的前提)
def low_rank_query_latent(h_t: np.ndarray, W_DQ: np.ndarray) -> np.ndarray:
    return h_t @ W_DQ


# PAPER: §2.3.1 Eq.(14) —— q_t^I = c_t^Q.W^{IUQ},reshape 成 n_h^I 个 indexer query 头
def indexer_queries(c_q_t: np.ndarray, W_IUQ: np.ndarray, n_heads_idx: int, head_dim_idx: int) -> np.ndarray:
    flat = c_q_t @ W_IUQ
    return flat.reshape(n_heads_idx, head_dim_idx)


# PAPER: §2.3.1 Eq.(15) —— w_t^I = h_t.W^w,每个 indexer 头一个标量权重
def indexer_head_weights(h_t: np.ndarray, W_w: np.ndarray) -> np.ndarray:
    return h_t @ W_w


# PAPER: §2.3.1 文字"CSA performs the same compression operation used for C^Comp
# to get compressed indexer keys K^IComp" —— 复用 csa_compress_sequence,套 indexer
# 自己的一份权重(维度 c^I 而非核注意力的 c)
def indexer_compressed_keys(H: np.ndarray, W_a_kv, W_b_kv, W_a_z, W_b_z, B_a, B_b, m: int) -> np.ndarray:
    """返回 K^IComp:(n/m, c^I)。"""
    return csa_compress_sequence(H, W_a_kv, W_b_kv, W_a_z, W_b_z, B_a, B_b, m)


# PAPER: §2.3.1 Eq.(16) —— I_{t,s} = sum_h w_{t,h}^I . ReLU(q_{t,h}^I . K_s^IComp),
# 单个候选压缩块 s
def index_score(q_idx_heads: np.ndarray, w_idx_heads: np.ndarray, k_comp_s: np.ndarray) -> float:
    dots = q_idx_heads @ k_comp_s          # (n_heads_idx,)
    relu = np.maximum(dots, 0.0)
    return float(w_idx_heads @ relu)


# PAPER: §2.3.1 Eq.(16) —— 对全部候选压缩块批量打分 I_{t,:},与逐个调用 index_score 等价
def index_scores_for_query(q_idx_heads: np.ndarray, w_idx_heads: np.ndarray, K_IComp: np.ndarray) -> np.ndarray:
    """q_idx_heads:(n_heads_idx,c^I);w_idx_heads:(n_heads_idx,);K_IComp:(S,c^I)。返回 (S,)。"""
    dots = q_idx_heads @ K_IComp.T          # (n_heads_idx, S)
    relu = np.maximum(dots, 0.0)
    return w_idx_heads @ relu                # (S,)


# PAPER: §2.3.1 Eq.(17) —— C_t^SprsComp = {C_s^Comp | I_{t,s} in Top-k(I_{t,:})},
# 因果约束:s < floor(t/m)(query 只能看到严格早于自己所在块的压缩块)
def topk_sparse_selection(scores: np.ndarray, C_comp: np.ndarray, k: int, causal_limit: int | None = None):
    """scores:(S,) 全部候选块的 I_{t,:};C_comp:(S,c) 全部压缩 KV。causal_limit 传 floor(t/m),
    只在 s < causal_limit 的候选里选 top-k(为 None 时不做因果裁剪,供单元测试直接验证选择逻辑)。

    返回 (selected_indices 升序排列的 np.ndarray, C_t^SprsComp 按同顺序堆叠的 (k',c) 数组)。
    """
    if causal_limit is not None:
        candidates = np.arange(min(causal_limit, len(scores)))
    else:
        candidates = np.arange(len(scores))
    if len(candidates) == 0:
        return np.array([], dtype=int), np.zeros((0, C_comp.shape[-1]))
    k_eff = min(k, len(candidates))
    cand_scores = scores[candidates]
    # argpartition 找出前 k_eff 大的局部下标,再映射回原始候选下标并排序(保持相对顺序好定位)
    top_local = np.argpartition(-cand_scores, k_eff - 1)[:k_eff]
    selected = np.sort(candidates[top_local])
    return selected, C_comp[selected]
