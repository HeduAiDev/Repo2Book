"""ch36 §2.3.1 Eq.(18)-(19) / §2.3.2 Eq.(24)-(26) (paper.md, arXiv:2606.19348) --
CSA 与 HCA 共用的"共享 KV 的 MQA 核注意力 + 分组输出投影"。两节公式形式完全同构
(HCA 只是把 CSA 的稀疏候选集 C_t^SprsComp 换成全部压缩块 C^Comp),故合成一份实现。

流程:
  1. 从共享的低秩 latent c_t^Q(lightning_indexer_csa.low_rank_query_latent 产出的同一份)
     升出 n_h 个注意力 query 头(Eq.18/25)。
  2. 用 MQA 方式做核注意力:每个头各自的 query 都去 attend 同一份压缩 KV 集合
     (该 KV 集合同时充当 key 与 value)——Eq.19/26。
  3. 分组输出投影(Grouped Output Projection):把 n_h 个头的输出拼接后先按组降维再拼回 d 维,
     避免 c*n_h 直接投回 d 维的算力开销。

落地:vllm_ascend/models/deepseek_v4.py:L774-789(wo_a/wo_b、n_groups、o_lora_rank)。
"""
import numpy as np


# PAPER: §2.3.1 Eq.(18) / §2.3.2 Eq.(25) —— q_t = c_t^Q.W^{UQ},reshape 成 n_h 个头
def attention_queries(c_q_t: np.ndarray, W_UQ: np.ndarray, n_heads: int, head_dim: int) -> np.ndarray:
    flat = c_q_t @ W_UQ
    return flat.reshape(n_heads, head_dim)


# PAPER: §2.3.1 Eq.(19) / §2.3.2 Eq.(26) —— o_{t,i} = CoreAttn(query=q_{t,i}, key=kv, value=kv),
# 共享 KV 的 MQA:全部 n_h 个头用同一份 kv_entries 当 key 与 value
def mqa_core_attention(q_heads: np.ndarray, kv_entries: np.ndarray, scale: float | None = None) -> np.ndarray:
    """q_heads:(n_h,c);kv_entries:(S,c)(压缩块,CSA 传稀疏选中的子集,HCA 传全部)。返回 (n_h,c)。"""
    n_h, c = q_heads.shape
    if kv_entries.shape[0] == 0:
        return np.zeros((n_h, c))
    if scale is None:
        scale = 1.0 / np.sqrt(c)
    logits = (q_heads @ kv_entries.T) * scale        # (n_h, S)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    return weights @ kv_entries                       # (n_h, c)


# PAPER: §2.3.1 "Grouped Output Projection" 段落 —— 把 n_h 个头的输出按 g 组切分,
# 每组先降到 d_g 维,再把 g 段拼接投回 d 维;避免 c*n_h 直接投回 d 的算力开销
def grouped_output_projection(o_heads: np.ndarray, group_weights: list[np.ndarray], W_out: np.ndarray) -> np.ndarray:
    """o_heads:(n_h,c);group_weights:长度 g 的列表,每个 (n_h//g * c, d_g);W_out:(g*d_g, d)。
    返回最终注意力输出 (d,)。要求 n_h 能被 g=len(group_weights) 整除。"""
    n_h, c = o_heads.shape
    g = len(group_weights)
    if n_h % g != 0:
        raise ValueError(f"n_h={n_h} 必须能被组数 g={g} 整除")
    heads_per_group = n_h // g
    flat_groups = o_heads.reshape(g, heads_per_group * c)   # (g, group_size)
    intermediate = np.stack([flat_groups[i] @ group_weights[i] for i in range(g)])  # (g, d_g)
    concatenated = intermediate.reshape(-1)                  # (g*d_g,)
    return concatenated @ W_out                              # (d,)


# PAPER: §2.3.1 Eq.(18)-(19) / §2.3.2 Eq.(24)-(26) 打包 —— 便捷组装:CSA/HCA 一层的
# 核注意力 + 输出投影全流程(供 hybrid_layer.py 与测试复用)
def attention_output_pipeline(c_q_t: np.ndarray, W_UQ: np.ndarray, kv_entries: np.ndarray,
                               n_heads: int, head_dim: int, group_weights: list[np.ndarray],
                               W_out: np.ndarray) -> np.ndarray:
    q_heads = attention_queries(c_q_t, W_UQ, n_heads, head_dim)
    o_heads = mqa_core_attention(q_heads, kv_entries)
    return grouped_output_projection(o_heads, group_weights, W_out)
