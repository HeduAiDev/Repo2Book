"""ch36 §2.3.3 Other Details / Eq.(27) (paper.md, arXiv:2606.19348) -- CSA/HCA 共用的
"其他细节":Query/KV RMSNorm、部分 RoPE、滑窗支线、注意力 sink。

论文原话给出的动机链条(§2.3.3):
  - 因果约束下 query 只能看到严格早于自己所在压缩块的压缩块 => 看不到本块内、也看不到
    最近若干 token 的细粒度信息 => 补一条滑窗支线(未压缩的最近 n_win 个 token)。
  - 压缩 KV 条目同时充当 core attention 的 key 与 value,若直接对它加 RoPE,核注意力输出
    会带上"绝对位置"信息(因为是多个 token 加权求和后的产物,权重与位置相关)=> 只对最后
    64 维做 RoPE(partial RoPE),并对输出再做一次"负位置"RoPE 抵消掉这一效应,使输出携带
    的是相对位置信息而非绝对位置信息。
  - 注意力 sink:给每个头一个可学习的 sink logit,加进 softmax 分母,允许某些 query
    把注意力质量的总和调到接近 0(吸收无信息 token)。

落地:vllm_ascend/models/deepseek_v4.py:L737-744(window_size/attn_sink),
L685-708 / L792-810(部分 RoPE 的 ComplexExpRotaryEmbedding 装配)。
"""
import numpy as np


# ---------------------------------------------------------------------------
# Query/KV RMSNorm(§2.3.3 首段:核注意力前对每头 query 与压缩 KV 做 RMSNorm)
# ---------------------------------------------------------------------------


# PAPER: §2.3.3 "Query and Key-Value Entry Normalization" —— 核注意力前对每头 query
# 与压缩 KV 做 RMSNorm,避免注意力 logits 爆炸
def rms_norm(x: np.ndarray, weight: np.ndarray | None = None, eps: float = 1e-6) -> np.ndarray:
    scale = 1.0 / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    out = x * scale
    if weight is not None:
        out = out * weight
    return out


# ---------------------------------------------------------------------------
# 部分 RoPE(Partial Rotary Positional Embedding)
# ---------------------------------------------------------------------------


# PAPER: §2.3.3 "Partial Rotary Positional Embedding" —— 标准 RoPE 旋转矩阵(仅用于最后
# rope_dims 维),显式构造成矩阵便于数值验证
def _rope_rotation_matrix(pos: float, dim: int, base: float = 10000.0) -> np.ndarray:
    assert dim % 2 == 0, "RoPE 维度必须是偶数(成对旋转)"
    half = dim // 2
    inv_freq = base ** (-(2.0 * np.arange(half)) / dim)
    angles = pos * inv_freq
    R = np.zeros((dim, dim))
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    for i in range(half):
        R[2 * i, 2 * i] = cos_a[i]
        R[2 * i, 2 * i + 1] = -sin_a[i]
        R[2 * i + 1, 2 * i] = sin_a[i]
        R[2 * i + 1, 2 * i + 1] = cos_a[i]
    return R


# PAPER: §2.3.3 "for each query vector and KV entry vector ... we apply RoPE to its
# last 64 dimensions" —— 只旋转向量的最后 rope_dims 维,其余维不变
def apply_partial_rope(vec: np.ndarray, position: float, rope_dims: int = 64, base: float = 10000.0) -> np.ndarray:
    d = vec.shape[-1]
    if rope_dims > d:
        raise ValueError(f"rope_dims={rope_dims} 不能超过向量维度 {d}")
    head, tail = vec[..., : d - rope_dims], vec[..., d - rope_dims:]
    R = _rope_rotation_matrix(position, rope_dims, base)
    tail_rot = tail @ R.T
    return np.concatenate([head, tail_rot], axis=-1)


# PAPER: §2.3.3 "we also apply RoPE with position -i on the last 64 dimensions of
# each o_{t,i}" —— 核注意力输出的反制项:压缩 KV 条目本身携带绝对位置信息(因为是多个
# token 的加权求和),若不处理,输出会"沾"上绝对位置;对输出再做一次"负的 query 位置"
# 旋转,使其携带的是"query 与被聚合 KV 之间"的相对位置信息。
#
# 注:论文原文这里复用符号 "i" 既指"第 i 个注意力头"(与 Eq.18 的 o_{t,i} 记号一致)又写
# "position -i",两者若字面理解会冲突(头索引不该是一个"位置")——本实现取更貌似合理的
# 读法,把 -i 理解成 "-t"(query 自身的 token 位置取负),这与紧邻的说明"output ... carries
# relative position embeddings ... related to the distance between the query and the KV
# entry"在数学上自洽(标准 RoPE 反向旋转技巧:对输出施加 R(-t) 抵消聚合过程带入的 R(+t)
# 分量)。若原论文本意确是别的下标,读者应以 open-source 实现为准——这里的参数名
# `query_position` 已明确标出这一选择,供 writer/reviewer 核对时不会被表面记号误导。
# PAPER: §2.3.3 "we also apply RoPE with position -i on the last 64 dimensions" (见上)
def apply_output_relative_rope(o_head_vec: np.ndarray, query_position: float,
                                rope_dims: int = 64, base: float = 10000.0) -> np.ndarray:
    return apply_partial_rope(o_head_vec, -query_position, rope_dims=rope_dims, base=base)


# ---------------------------------------------------------------------------
# 滑窗支线(Additional Branch of Sliding Window Attention)
# ---------------------------------------------------------------------------


# PAPER: §2.3.3 "we additionally produce n_win uncompressed KV entries corresponding
# to the recent n_win tokens" —— 供因果 query 在 query_pos 处取最近 n_win 个未压缩 token
def sliding_window_recent_kv(token_kv_seq: np.ndarray, query_pos: int, n_win: int) -> np.ndarray:
    start = max(0, query_pos - n_win + 1)
    return token_kv_seq[start: query_pos + 1]


# ---------------------------------------------------------------------------
# 注意力 sink(Eq.27)
# ---------------------------------------------------------------------------


# PAPER: §2.3.3 Eq.(27) —— s_{h,i,j} = Exp(z_{h,i,j}) / (sum_k Exp(z_{h,i,k}) + Exp(z'_h))
def attention_sink_scores(logits: np.ndarray, sink_logit: float) -> np.ndarray:
    """logits:(J,) 某一头对 J 个候选(压缩块+滑窗 raw token)的打分 z_{h,i,:};
    sink_logit:该头的可学习 sink logit z'_h(标量)。返回归一化分数,和可以 < 1
    ("允许每个 query 头把注意力总和调到不等于 1、甚至接近 0")。"""
    m = max(np.max(logits), sink_logit)
    exp_logits = np.exp(logits - m)
    exp_sink = np.exp(sink_logit - m)
    denom = np.sum(exp_logits) + exp_sink
    return exp_logits / denom


# PAPER: §2.3.3 Eq.(27) 推论 —— sink 吸收掉的注意力质量(1 - 分数之和),数值推演时用来
# 演示"总和 < 1"这件事(便捷函数,非独立公式)
def sink_absorbed_mass(scores: np.ndarray) -> float:
    return float(1.0 - np.sum(scores))
