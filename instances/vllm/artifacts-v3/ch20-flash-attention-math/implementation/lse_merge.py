"""LSE 合并——⊕ 算子(arXiv:1805.02867 §3.1 Eq.(4))在 (lse, output) 表示上的作用,
vLLM merge_attn_states 的数学原型(vllm/v1/attention/ops/triton_merge_attn_states.py:
L259-L322;其分派器 docstring 自引 arXiv:2501.01005 §2.2 的 split-KV 合并——该文不在
论文包内,但数学与 Eq.(4) 同源,这里的推导只从论文包两文出发):

  arXiv:2307.08691 §3.1.1 Tweak 2 —— 每段注意力只需存一个标量 logsumexp
  L = m + log(ℓ)(不必同时存 m 与 ℓ);vLLM flash_attn_varlen_func 的
  return_softmax_lse 返回的正是它。
  arXiv:2307.08691 §2.3.1 两块表 —— 把论文自己的两块恒等式改写到 lse 域:
  e^{S^(a)-m}·V^(a) = e^{lse_a-m}·ℓ^(a)·O^(a)(O^(a) 是该段已归一化的局部输出),
  代入 O = diag(ℓ)^{-1}·[e^{S^(1)-m}V^(1) + e^{S^(2)-m}V^(2)] 即得
  O = (e^{lse_a-m}·O^(a) + e^{lse_b-m}·O^(b)) / (e^{lse_a-m} + e^{lse_b-m})。

两段各调用一次 attention_with_lse 得 (O, lse),再 merge_lse_states 合并——worked
example(cascade:共享前缀 causal=False + 私有后缀 causal=True 两段)合并后与「对拼接
KV 一次性做因果注意力」逐位相等,这正是 vLLM cascade_attention 前缀只算一遍被全批
复用的正确性根基(⊕ 的结合律)。

空段约定:lse = -inf、输出为零(该行在这段一个 key 都没看见),其合并权重自然为 0;
双空(权重 0/0 = NaN)按 vLLM Triton kernel 同款护栏(vllm/v1/attention/ops/
triton_merge_attn_states.py:L319-L322)输出零、合并 lse 保持 -inf。FA2 对空序列返回
inf、FA3 返回 -inf,vLLM kernel 里先做 inf→-inf 归一(L270-L276)——本实现直接
采用 -inf 约定。

「trace」可选参数只记录合并记账的快照(供示教轨迹用),不是论文之外的新机制。
"""
import math

import numpy as np

from flash_attention import _safe_exp, causal_keep_mask


# 单遍算完一段注意力(非分块;分块版本见 flash_attention.flash_attention_2_forward
# 的同名统计量):行最大 m、归一项 ℓ = Σ_j e^{S_j - m}(FA-2 §2.3.1 两块表对「块 1」
# 的前三行:m^(1)/ℓ^(1)/O^(1)),输出 O = diag(ℓ)^{-1} e^{S-m} V(表里的 O^(1)),
# 并按 Tweak 2 把 lse = m + log(ℓ) 一起返回。cascade 的前缀段/后缀段各调用一次,
# 得到的 (O, lse) 即 merge_lse_states 的输入。softmax_scale 默认 1/sqrt(d) 与 vLLM
# flash_attn_varlen_func 一致;causal 右下对齐语义同 flash_attention.standard_attention
# (整行被遮 → 输出 0、lse = -inf,即上面说的空段约定)。
# PAPER: arXiv:2307.08691 §3.1.1 Tweak 2 (L = m + log(ℓ))
def attention_with_lse(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    causal: bool = False,
    scale: float = None,
    query_offset: int = 0,
):
    d = Q.shape[-1]
    scale = scale if scale is not None else 1.0 / math.sqrt(d)
    S = (Q @ K.T) * scale
    if causal:
        keep = causal_keep_mask(S.shape[0], S.shape[1], query_offset)
        S = np.where(keep, S, -np.inf)

    m = S.max(axis=-1)  # rowmax(FA-2 §2.3.1: m^(a) = rowmax(S^(a)))
    with np.errstate(invalid="ignore"):
        p = _safe_exp(S - m[:, None])  # e^{S - m},逐元素(整行被遮时 -inf-(-inf)=nan,归零)
    l = p.sum(axis=-1)  # ℓ^(a) = rowsum(e^{S^(a) - m^(a)})

    has_key = l > 0  # 空段:这一行在本段没看见任何合法 key(全被 causal 遮住)
    with np.errstate(invalid="ignore", divide="ignore"):
        # O^(a) = diag(ℓ^(a))^{-1} e^{S^(a)-m^(a)} V^(a)(§2.3.1 表的 O^(1),已归一化)
        O = np.where(
            has_key[:, None],
            (p @ V) / np.where(has_key, l, 1.0)[:, None],
            0.0,
        )
        # lse^(a) = m^(a) + log(ℓ^(a))(Tweak 2);空段取 -inf(FA2/FA3 inf→-inf 归一后的约定)
        lse = np.where(has_key, m + np.log(np.where(has_key, l, 1.0)), -np.inf)
    return O, lse


# ⊕ 作用在 (lse, output) 上——与 online_softmax.online_softmax_merge 作用在 (m, d)
# 上是同一个算子换了表示(lse = log d):max_lse = max(lse_a, lse_b) 稳定化(两个
# e^{lse-max} 均 ≤ 1,不溢出——与 safe-softmax 减 max 同理);p_se = e^{lse_a-max}、
# s_se = e^{lse_b-max} 即两段归一化质量 e^{lse}=Σe^S 折算到新 max 的值,⊕ 的
# d_new = d_a·e^{m_a-max} + d_b·e^{m_b-max} 两侧取 log、代入 lse=log d 即
# out_lse = log(p_se+s_se) + max_lse;权重 = 各自占比 p_se/out_se,输出 = 加权合并。
# 落地:vLLM Triton merge kernel(vllm/v1/attention/ops/triton_merge_attn_states.py:
# L259-L322)逐 (token,head) 即此六步(max_lse 稳定化 → e^{lse-max} → out_se=Σ →
# 权重=占比 → 加权合并 → lse=log(out_se)+max_lse);NOTE(woosuk) 的数值纪律
# 「先算 scale 再乘 output,不要拿 e^{lse-max} 直接乘 output」在此遵守。
# ⊕ 交换律保证两段地位对称,prefix/suffix 命名只是沿用 vLLM cascade 调用现场
# (前缀段 causal=False + block_table[:1],后缀段 causal=True + 私有 KV)的习惯。
# PAPER: arXiv:1805.02867 §3.1 Eq.(4) + arXiv:2307.08691 §2.3.1
def merge_lse_states(
    prefix_output: np.ndarray,
    prefix_lse: np.ndarray,
    suffix_output: np.ndarray,
    suffix_lse: np.ndarray,
    trace: list = None,
):
    max_lse = np.maximum(prefix_lse, suffix_lse)  # 稳定化:新「max」
    with np.errstate(invalid="ignore"):
        p_se = _safe_exp(prefix_lse - max_lse)  # 双空时 -inf-(-inf)=nan,_safe_exp 归 0
        s_se = _safe_exp(suffix_lse - max_lse)
    out_se = p_se + s_se  # ⊕ 的新 d(在 e^{lse}=Σe^S 的量纲上)

    both_empty = max_lse == -np.inf  # 双空护栏(vLLM kernel L319-L322 同款):0/0=NaN 须置 0
    with np.errstate(invalid="ignore", divide="ignore"):
        # 权重 = 归一化质量占比;双空时权重取 0(输出零而非 NaN)
        p_scale = np.where(both_empty, 0.0, p_se / np.where(both_empty, 1.0, out_se))
        s_scale = np.where(both_empty, 0.0, s_se / np.where(both_empty, 1.0, out_se))
        # 合并 lse = log(out_se) + max_lse;双空保持 -inf,让下游继续当空段处理
        out_lse = np.where(
            both_empty, -np.inf, np.log(np.where(both_empty, 1.0, out_se)) + max_lse
        )

    # 先算 scale 再乘 output(NOTE(woosuk) 的数值稳定纪律),不要拿 e^{lse-max} 直接乘
    out = prefix_output * p_scale[:, None] + suffix_output * s_scale[:, None]
    if trace is not None:
        trace.append(
            {
                "max_lse": max_lse.copy(),
                "p_se": p_se.copy(),
                "s_se": s_se.copy(),
                "out_se": out_se.copy(),
                "p_scale": p_scale.copy(),
                "s_scale": s_scale.copy(),
                "out_lse": out_lse.copy(),
            }
        )
    return out, out_lse
