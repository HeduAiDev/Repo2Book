"""arXiv:1805.02867 §3.1 Eq.(4)(⊕ 算子)+ arXiv:2307.08691 §3.1.1(logsumexp 定义)——
LSE 合并:两段各自算出 (output, lse) 的部分注意力,合并后应与"一次性对合并 KV 做
softmax"逐位相等。附 cascade attention(共享前缀 + 私有后缀两段合并)worked example,
对应 vLLM merge_attn_states 在推理期的真实调用现场。
"""
import numpy as np

from flash_attention import standard_attention
from lse_merge import attention_with_lse, merge_lse_states


def _random_qkv(n_q, n_k, d=8, seed=0):
    rng = np.random.default_rng(seed)
    Q = rng.normal(size=(n_q, d))
    K = rng.normal(size=(n_k, d))
    V = rng.normal(size=(n_k, d))
    return Q, K, V


def test_attention_with_lse_output_matches_standard_attention():
    Q, K, V = _random_qkv(n_q=10, n_k=12, seed=0)
    O_std = standard_attention(Q, K, V)
    O_lse, lse = attention_with_lse(Q, K, V)
    np.testing.assert_allclose(O_lse, O_std, rtol=1e-6, atol=1e-8)


def test_attention_with_lse_matches_naive_logsumexp_definition():
    # lse := m + log(l) ,即 FA-2 §3.1.1 存的那个标量(arXiv:2307.08691 L146-L147)。
    Q, K, V = _random_qkv(n_q=6, n_k=9, seed=1)
    _, lse = attention_with_lse(Q, K, V)
    S = (Q @ K.T) / np.sqrt(Q.shape[-1])
    m_ref = S.max(axis=-1)
    l_ref = np.exp(S - m_ref[:, None]).sum(axis=-1)
    lse_ref = m_ref + np.log(l_ref)
    np.testing.assert_allclose(lse, lse_ref, rtol=1e-6)


def test_merge_lse_states_two_key_blocks_equals_one_shot_attention():
    # worked example:把 K,V 切成两个列块(与 tiling 完全一致的切法),各自算 (O,lse),
    # 合并后应与一次性对整段 K,V 做 attention 逐位相等——这就是 ⊕ 算子的合法性在
    # (lse,output) 表示下的验证。
    Q, K, V = _random_qkv(n_q=8, n_k=14, seed=2)
    K1, V1 = K[:6], V[:6]
    K2, V2 = K[6:], V[6:]

    O1, lse1 = attention_with_lse(Q, K1, V1)
    O2, lse2 = attention_with_lse(Q, K2, V2)
    O_merged, lse_merged = merge_lse_states(O1, lse1, O2, lse2)

    O_full, lse_full = attention_with_lse(Q, K, V)
    np.testing.assert_allclose(O_merged, O_full, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(lse_merged, lse_full, rtol=1e-6)


def test_merge_lse_states_three_way_associative_like_online_softmax_merge():
    # 与 online_softmax_merge 的结合律测试对应:合并顺序不影响结果。
    Q, K, V = _random_qkv(n_q=5, n_k=15, seed=3)
    K1, V1 = K[:5], V[:5]
    K2, V2 = K[5:10], V[5:10]
    K3, V3 = K[10:], V[10:]
    O1, l1 = attention_with_lse(Q, K1, V1)
    O2, l2 = attention_with_lse(Q, K2, V2)
    O3, l3 = attention_with_lse(Q, K3, V3)

    left_O, left_lse = merge_lse_states(*merge_lse_states(O1, l1, O2, l2), O3, l3)
    right_O, right_lse = merge_lse_states(O1, l1, *merge_lse_states(O2, l2, O3, l3))

    np.testing.assert_allclose(left_O, right_O, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(left_lse, right_lse, rtol=1e-6)


def test_cascade_shared_prefix_merge_equals_full_causal_attention():
    # cascade attention 的真实现场(vllm/v1/attention/backends/flash_attn.py:L1145-L1236):
    # 前缀段(causal=False,所有 query 共享同一段前缀,全可见)+ 后缀段(causal=True,
    # 各 query 只看自己私有后缀里 <= 自己位置的部分),两段各 return_softmax_lse,
    # merge_attn_states 合并。这里验证合并结果与"一次性对 [前缀;后缀] 整体做因果注意力"
    # (每个 query 看得到全部前缀 + 自己私有后缀里 <= 自己位置的部分)逐位相等。
    n_prefix, n_q, d = 9, 6, 8
    Qp, Kp, Vp = _random_qkv(n_q, n_prefix, d, seed=4)
    Q = Qp
    _, Ks, Vs = _random_qkv(n_q, n_q, d, seed=5)

    prefix_out, prefix_lse = attention_with_lse(Q, Kp, Vp, causal=False)
    suffix_out, suffix_lse = attention_with_lse(Q, Ks, Vs, causal=True, query_offset=0)
    merged_out, _ = merge_lse_states(prefix_out, prefix_lse, suffix_out, suffix_lse)

    # 参照:把前缀/后缀拼成一整段 KV,对 query i 用 query_offset=n_prefix 做因果掩码——
    # 这恰好复现"全部前缀可见 + 后缀只看 <= 自己位置"的两段掩码语义。
    K_full = np.concatenate([Kp, Ks], axis=0)
    V_full = np.concatenate([Vp, Vs], axis=0)
    full_out = standard_attention(Q, K_full, V_full, causal=True, query_offset=n_prefix)

    np.testing.assert_allclose(merged_out, full_out, rtol=1e-6, atol=1e-8)
