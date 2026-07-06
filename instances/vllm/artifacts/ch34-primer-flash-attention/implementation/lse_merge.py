"""LSE 合并——vLLM merge_attn_states(cascade attention / split-KV 的地基)在数学层面的
原型。锚点两处:

  arXiv:1805.02867 §3.1 Eq.(4)  —— [m_i;d_i] ⊕ [m_j;d_j] 结合律合并算子(在 online_softmax.py
  里作用在 (m,d) 这对状态上)。
  arXiv:2307.08691 §3.1.1       —— logsumexp L^(j) = m^(j) + log(l^(j)):FlashAttention-2
  只存这一个标量而非 (m,l) 两个,backward / 合并都只需要它。

vLLM 的 merge_attn_states 把 ⊕ 算子实现在 (lse, output) 这对状态上而不是 (m, d) 上——
lse := m + log(d) 恰是 FA-2 存的量,output 是已经除过 d 归一化的部分注意力结果(FA 论文
里的 O,而不是未归一化的 O~)。这里先给出"一次性算好 (O, lse)"的 attention_with_lse
(供两段各自调用一次),再给出把两段 (O, lse) 合并成精确整体结果的 merge_lse_states——
worked example:前缀段(causal=False,共享)+ 后缀段(causal=True,私有)分别算,合并后
应与对合并 KV 一次性做完整 causal attention 逐位相等,这正是 vLLM cascade_attention 依赖
的正确性保证。
"""
import math

import numpy as np

from flash_attention import _safe_exp


# PAPER: arXiv:2307.08691 §3.1.1 (L^{(j)} = m^{(j)} + log(l^{(j)})) —— 一次性(非分块)
# 跑完一段注意力,除了输出 O 之外顺带算出它的 logsumexp lse = m + log(l)(m 是行最大值,
# l 是 safe-softmax 的归一化项)。两段注意力(如 cascade 的前缀/后缀)各调用一次,得到
# 的 (O, lse) 就是 merge_lse_states 的输入。
# PAPER: arXiv:2307.08691 §3.1.1
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
        n_q, n_k = S.shape
        q_idx = np.arange(n_q)[:, None] + query_offset
        k_idx = np.arange(n_k)[None, :]
        S = np.where(k_idx > q_idx, -np.inf, S)

    m = S.max(axis=-1)  # rowmax
    with np.errstate(invalid="ignore"):
        p = _safe_exp(S - m[:, None])
    l = p.sum(axis=-1)  # rowsum(exp(S-m))

    has_any_key = l > 0  # 一整行从未看到合法 key(如 causal 下第一行且 kv 段为空)
    with np.errstate(invalid="ignore", divide="ignore"):
        O = np.where(has_any_key[:, None], (p @ V) / np.where(l[:, None] == 0, 1.0, l[:, None]), 0.0)
        lse = np.where(has_any_key, m + np.log(np.where(l == 0, 1.0, l)), -np.inf)
    return O, lse


# PAPER: arXiv:1805.02867 §3.1 Eq.(4) —— ⊕ 算子作用在 (lse, output) 而非 (m, d) 上:
# 先以 max_lse 稳定化(两个 exp(lse-max) 都 <= 1,不溢出——与 safe-softmax 减 max 同理),
# 按 e^{lse-max} 求两段各自的权重 a_scale/b_scale,加权合并 output;合并 lse =
# log(a_se+b_se) + max_lse,与 online_softmax.online_softmax_merge 的 d_new 是同一个量
# 换了对数底(lse = log d)。vLLM 的 merge_attn_states_kernel 就是这段数学的逐 token
# 逐 head 落地(见 vllm/v1/attention/ops/triton_merge_attn_states.py:L118-L161)。
# PAPER: arXiv:1805.02867 §3.1 Eq.(4)
def merge_lse_states(
    o_a: np.ndarray, lse_a: np.ndarray, o_b: np.ndarray, lse_b: np.ndarray
):
    max_lse = np.maximum(lse_a, lse_b)
    with np.errstate(invalid="ignore"):
        a_se = _safe_exp(lse_a - max_lse)
        b_se = _safe_exp(lse_b - max_lse)
    out_se = a_se + b_se
    with np.errstate(invalid="ignore", divide="ignore"):
        out_lse = np.where(out_se > 0, np.log(np.where(out_se == 0, 1.0, out_se)) + max_lse, -np.inf)
        a_scale = np.where(out_se > 0, a_se / np.where(out_se == 0, 1.0, out_se), 0.0)
        b_scale = np.where(out_se > 0, b_se / np.where(out_se == 0, 1.0, out_se), 0.0)
    out = o_a * a_scale[:, None] + o_b * b_scale[:, None]
    return out, out_lse
