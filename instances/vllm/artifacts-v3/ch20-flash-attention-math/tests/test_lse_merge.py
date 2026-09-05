"""arXiv:1805.02867 §3.1 Eq.(4)(⊕ 算子)+ arXiv:2307.08691 §3.1.1(logsumexp 定义)
—— LSE 合并:两段各自算出 (output, lse) 的部分注意力,合并后应与「一次性对合并 KV
做 softmax」逐位相等。附 cascade attention(共享前缀 + 私有后缀两段合并)worked
example,对应 vLLM merge_attn_states / cascade_attention 的真实调用现场。"""
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
    # lse := m + log(ℓ) —— FA-2 §3.1.1 Tweak 2 存的那个标量。
    Q, K, V = _random_qkv(n_q=6, n_k=9, seed=1)
    _, lse = attention_with_lse(Q, K, V)
    S = (Q @ K.T) / np.sqrt(Q.shape[-1])
    m_ref = S.max(axis=-1)
    l_ref = np.exp(S - m_ref[:, None]).sum(axis=-1)
    np.testing.assert_allclose(lse, m_ref + np.log(l_ref), rtol=1e-6)


def test_merge_lse_states_two_key_blocks_equals_one_shot_attention():
    # worked example:把 K,V 切成两个列块(与 tiling 一致的切法),各自算 (O, lse),
    # 合并后与一次性对整段 K,V 做 attention 逐位相等 —— ⊕ 算子在 (lse, output)
    # 表示下的合法性验证(对应 FA-2 §2.3.1 的两块推导表)。
    Q, K, V = _random_qkv(n_q=8, n_k=14, seed=2)
    K1, V1 = K[:6], V[:6]
    K2, V2 = K[6:], V[6:]

    O1, lse1 = attention_with_lse(Q, K1, V1)
    O2, lse2 = attention_with_lse(Q, K2, V2)
    trace = []
    O_merged, lse_merged = merge_lse_states(O1, lse1, O2, lse2, trace=trace)

    # 合并记账快照(示教素材):权重是归一化占比 —— p_scale + s_scale == 1、
    # out_se = p_se + s_se,且 out 正是两段按占比的加权合并。
    assert len(trace) == 1
    t = trace[0]
    np.testing.assert_allclose(t["p_se"] + t["s_se"], t["out_se"], rtol=1e-12)
    np.testing.assert_allclose(t["p_scale"] + t["s_scale"], 1.0, rtol=1e-12)
    np.testing.assert_allclose(
        O_merged, O1 * t["p_scale"][:, None] + O2 * t["s_scale"][:, None], rtol=1e-12
    )

    O_full, lse_full = attention_with_lse(Q, K, V)
    np.testing.assert_allclose(O_merged, O_full, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(lse_merged, lse_full, rtol=1e-6)


def test_merge_lse_states_three_way_associative_like_online_softmax_merge():
    # 与 ⊕ 算子的结合律对应:合并顺序不影响结果。
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


def test_merge_lse_states_with_one_empty_segment_returns_other_unchanged():
    # 空段(lse = -inf,该 token 在这段一个 key 都没看见)权重为 0:
    # 合并结果 == 另一段原样 —— cascade 里前缀/后缀某段为空时的正确性。
    Q, K, V = _random_qkv(n_q=4, n_k=6, seed=5)
    O_b, lse_b = attention_with_lse(Q, K, V)
    lse_a = np.full(len(Q), -np.inf)
    O_a = np.zeros_like(O_b)

    O_merged, lse_merged = merge_lse_states(O_a, lse_a, O_b, lse_b)
    np.testing.assert_allclose(O_merged, O_b, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(lse_merged, lse_b, rtol=1e-6)


def test_merge_lse_states_both_empty_guards_zero_output_and_neg_inf_lse():
    # 双空护栏(vLLM Triton merge kernel L319-L322 的同款边界):两侧都空时
    # 权重是 0/0,须输出零而不是 NaN,合并 lse 保持 -inf 让下游继续当空处理。
    n, d = 4, 8
    O_a = O_b = np.zeros((n, d))
    lse_a = lse_b = np.full(n, -np.inf)

    O_merged, lse_merged = merge_lse_states(O_a, lse_a, O_b, lse_b)
    np.testing.assert_allclose(O_merged, np.zeros((n, d)), atol=0.0)
    assert np.all(lse_merged == -np.inf)
    assert np.all(np.isfinite(O_merged))


def test_cascade_shared_prefix_merge_equals_full_causal_attention():
    # cascade attention 的调用现场(vllm/v1/attention/backends/flash_attn.py
    # cascade_attention:前缀段 causal=False + block_table[:1],后缀段 causal=True +
    # 私有 KV,两段各 return_softmax_lse 再 merge_attn_states):前缀段(所有 query
    # 共享同一段前缀,全可见)+ 后缀段(各 query 只看自己私有后缀里 <= 自己位置的
    # key),合并后应与「对 [前缀;后缀] 拼接 KV 一次性做因果注意力(每个 query 看
    # 得到全部前缀 + 私有后缀里 <= 自己位置的部分)」逐位相等 —— 前缀只算一遍被
    # 全批复用的正确性根基就是 ⊕ 的结合律。
    n_prefix, n_q, d = 9, 6, 8
    Q, Kp, Vp = _random_qkv(n_q, n_prefix, d, seed=4)
    _, Ks, Vs = _random_qkv(n_q, n_q, d, seed=5)

    prefix_out, prefix_lse = attention_with_lse(Q, Kp, Vp, causal=False)
    suffix_out, suffix_lse = attention_with_lse(Q, Ks, Vs, causal=True, query_offset=0)
    merged_out, _ = merge_lse_states(prefix_out, prefix_lse, suffix_out, suffix_lse)

    # 参照:拼接 KV + query_offset=n_prefix 的因果掩码,恰好复现两段掩码语义。
    K_full = np.concatenate([Kp, Ks], axis=0)
    V_full = np.concatenate([Vp, Vs], axis=0)
    full_out = standard_attention(Q, K_full, V_full, causal=True, query_offset=n_prefix)

    np.testing.assert_allclose(merged_out, full_out, rtol=1e-6, atol=1e-8)
