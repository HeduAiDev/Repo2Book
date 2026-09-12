"""arXiv:2405.04434 §2.1.1 Eq.(1)-(8)(标准 MHA 基线)+ arXiv:2305.13245 §2.2
(GQA-g 分组插值:GQA-1=MQA、GQA-H=MHA)—— MHA/GQA/MQA 是同一前向在
num_kv_heads 一根参数轴上的三个点;§2.1/§2.2 mean-pool 转换(uptraining 第一步)。
测试先于实现书写(TDD),每条断言对应论文的一句可复现声明:向量化前向 == 逐 token/
逐头/逐位置三重循环的论文原文参考实现。"""
import numpy as np

from mha_gqa import (
    attention_forward,
    kv_head_of_query_head,
    mha_to_gqa_mean_pool,
    softmax_lastdim,
)


def _make_mha_weights(d, n_h, d_h, G, seed=0):
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((n_h * d_h, d)),   # W_Q  (out=n_h*d_h, in=d) 论文方向
        rng.standard_normal((G * d_h, d)),     # W_K
        rng.standard_normal((G * d_h, d)),     # W_V
        rng.standard_normal((d, n_h * d_h)),   # W_O
    )


def _reference_by_paper_loops(h, W_Q, W_K, W_V, W_O, num_heads, num_kv_heads):
    """Eq.(1)-(8) 的逐 token/逐头/逐位置三重循环直译(含 §2.2 组内共享)。

    q_t=W^Q h_t (1)、k_t/v_t 同 (2)(3) → 切 n_h 个 query 头 (4)、G 个 KV 头
    (§2.2:query 头分 G 组、每组共享单个 K/V 头) → 逐头
    o_{t,i}=Σ_{j<=t} Softmax_j(q_{t,i}ᵀk_{j,i}/√d_h) v_{j,i} (7) →
    u_t=W^O[o_{t,1};…;o_{t,n_h}] (8)。
    """
    T, d = h.shape
    d_h = W_Q.shape[0] // num_heads
    G = num_kv_heads
    q = h @ W_Q.T          # Eq.(1) 全 token 一次算
    k = h @ W_K.T          # Eq.(2)
    v = h @ W_V.T          # Eq.(3)
    kv_map = kv_head_of_query_head(num_heads, G)
    u = np.zeros((T, d))
    for t in range(T):
        o_heads = []
        for i in range(num_heads):
            g = int(kv_map[i])          # §2.2:第 i 个 query 头所在组的共享 KV 头
            q_i = q[t, i * d_h:(i + 1) * d_h]
            scores = np.array([
                q_i @ k[j, g * d_h:(g + 1) * d_h] for j in range(t + 1)
            ]) / np.sqrt(d_h)           # Eq.(7) 分母 √d_h
            p = np.exp(scores - scores.max())
            p /= p.sum()                # Softmax_j
            o_i = sum(
                p[j] * v[j, g * d_h:(g + 1) * d_h] for j in range(t + 1)
            )
            o_heads.append(o_i)
        u[t] = W_O @ np.concatenate(o_heads)   # Eq.(8)
    return u


def test_softmax_lastdim_rows_sum_to_one_and_match_manual():
    x = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 0.5]])
    out = softmax_lastdim(x)
    manual = np.exp(x - x.max(axis=-1, keepdims=True))
    manual /= manual.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(out, manual, rtol=1e-12)
    np.testing.assert_allclose(out.sum(axis=-1), 1.0, rtol=1e-12)


def test_mha_forward_matches_paper_loop_reference():
    # §2.1.1 Eq.(1)-(8):num_kv_heads=num_heads 即 MHA。向量化前向须与
    # 三重循环直译逐位一致。
    T, d, n_h, d_h = 4, 5, 2, 3
    rng = np.random.default_rng(1)
    h = rng.standard_normal((T, d))
    W_Q, W_K, W_V, W_O = _make_mha_weights(d, n_h, d_h, n_h, seed=2)
    u, _ = attention_forward(h, W_Q, W_K, W_V, W_O, num_heads=n_h)
    ref = _reference_by_paper_loops(h, W_Q, W_K, W_V, W_O, n_h, n_h)
    np.testing.assert_allclose(u, ref, rtol=1e-12, atol=1e-14)


def test_num_kv_heads_none_defaults_to_num_heads_mha():
    # num_kv_heads=None == num_kv_heads=num_heads(vLLM QKVParallelLinear 的
    # total_num_kv_heads=None 默认语义;论文侧即 MHA 定义——K/V 头数=query 头数)。
    T, d, n_h, d_h = 3, 6, 4, 2
    rng = np.random.default_rng(3)
    h = rng.standard_normal((T, d))
    W_Q, W_K, W_V, W_O = _make_mha_weights(d, n_h, d_h, n_h, seed=4)
    u_none, _ = attention_forward(h, W_Q, W_K, W_V, W_O, num_heads=n_h)
    u_explicit, _ = attention_forward(
        h, W_Q, W_K, W_V, W_O, num_heads=n_h, num_kv_heads=n_h
    )
    np.testing.assert_array_equal(u_none, u_explicit)


def test_gqa_equals_mha_at_H_groups():
    # §2.2:"GQA-h, with groups equal to number of heads, is equivalent to MHA"
    # —— G=H 时组退化成每头一份,同一份 W_K/W_V 下输出与 MHA 完全一致。
    T, d, n_h, d_h = 4, 7, 4, 3
    rng = np.random.default_rng(5)
    h = rng.standard_normal((T, d))
    W_Q, W_K, W_V, W_O = _make_mha_weights(d, n_h, d_h, n_h, seed=6)
    u_mha, _ = attention_forward(h, W_Q, W_K, W_V, W_O, num_heads=n_h)
    u_gqa_h, _ = attention_forward(
        h, W_Q, W_K, W_V, W_O, num_heads=n_h, num_kv_heads=n_h
    )
    np.testing.assert_array_equal(u_mha, u_gqa_h)


def test_gqa_1_equals_mqa_all_heads_share_single_kv():
    # §2.2:"GQA-1, with a single group and therefore single key and value head,
    # is equivalent to MQA"—— G=1 时所有 query 头吃同一份 K/V。
    T, d, n_h, d_h = 4, 5, 3, 2
    rng = np.random.default_rng(7)
    h = rng.standard_normal((T, d))
    W_Q, W_K, W_V, W_O = _make_mha_weights(d, n_h, d_h, 1, seed=8)
    u, (k_heads, v_heads) = attention_forward(
        h, W_Q, W_K, W_V, W_O, num_heads=n_h, num_kv_heads=1
    )
    assert k_heads.shape == (T, 1, d_h)   # 单 KV 头
    ref = _reference_by_paper_loops(h, W_Q, W_K, W_V, W_O, n_h, 1)
    np.testing.assert_allclose(u, ref, rtol=1e-12, atol=1e-14)


def test_gqa_intermediate_group_matches_reference():
    # §2.2 中间组数:G 介于 1 与 H 之间(如 H=8、G=2——每组 4 个 query 头)。
    T, d, n_h, d_h, G = 5, 6, 8, 2, 2
    rng = np.random.default_rng(9)
    h = rng.standard_normal((T, d))
    W_Q, W_K, W_V, W_O = _make_mha_weights(d, n_h, d_h, G, seed=10)
    u, _ = attention_forward(h, W_Q, W_K, W_V, W_O, num_heads=n_h, num_kv_heads=G)
    ref = _reference_by_paper_loops(h, W_Q, W_K, W_V, W_O, n_h, G)
    np.testing.assert_allclose(u, ref, rtol=1e-12, atol=1e-14)


def test_kv_head_mapping_is_contiguous_groups():
    # §2.2 "divides query heads into G groups":连续分组(Fig.2 画法)——
    # 头 idx // (H/G) 即所在组。端点:G=H → 恒等映射(MHA)、G=1 → 全 0(MQA)。
    np.testing.assert_array_equal(
        kv_head_of_query_head(8, 2), np.array([0, 0, 0, 0, 1, 1, 1, 1])
    )
    np.testing.assert_array_equal(
        kv_head_of_query_head(6, 3), np.array([0, 0, 1, 1, 2, 2])
    )
    np.testing.assert_array_equal(kv_head_of_query_head(4, 4), np.arange(4))
    np.testing.assert_array_equal(
        kv_head_of_query_head(4, 1), np.zeros(4, dtype=int)
    )


def test_mha_to_gqa_mean_pool_pools_heads_within_each_group():
    # §2.1/§2.2:"The projection matrices for key and value heads are mean
    # pooled into single projection matrices"/"construct each group key and
    # value head by mean-pooling all the original heads within that group"
    # —— 组头 = 组内原头投影矩阵的算术平均(线性,故等价于先投再平均)。
    n_h, d_h, d, G = 4, 2, 3, 2
    rng = np.random.default_rng(11)
    W_K = rng.standard_normal((n_h * d_h, d))
    W_V = rng.standard_normal((n_h * d_h, d))
    W_K_g, W_V_g = mha_to_gqa_mean_pool(W_K, W_V, n_h, d_h, G)
    assert W_K_g.shape == (G * d_h, d)
    heads = W_K.reshape(n_h, d_h, d)
    np.testing.assert_allclose(
        W_K_g.reshape(G, d_h, d)[0], heads[0:2].mean(axis=0), rtol=1e-15
    )
    np.testing.assert_allclose(
        W_K_g.reshape(G, d_h, d)[1], heads[2:4].mean(axis=0), rtol=1e-15
    )
    np.testing.assert_allclose(
        W_V_g.reshape(G, d_h, d)[1], W_V.reshape(n_h, d_h, d)[2:4].mean(axis=0),
        rtol=1e-15,
    )
    # mean-pool 的线性等价:组共享 K = 组内原头 K 的平均(先投再平均 == 先平均再投)
    h = rng.standard_normal((1, d))
    k_pooled = h @ W_K_g.T
    k_avg_of_heads = np.stack([h @ W_K[i * d_h:(i + 1) * d_h].T for i in range(4)])
    np.testing.assert_allclose(
        k_pooled[0, 0:d_h], k_avg_of_heads[0:2].mean(axis=0)[0], rtol=1e-12
    )


def test_uptrained_style_gqa_forward_runs_with_pooled_weights():
    # §2.1 uptraining 第一步的完整链路:MHA 权重 → mean-pool 成 GQA → 前向可跑,
    # 且每个 query 头实际吃到的 K/V 是组内成员头投影的平均(与循环参考一致)。
    T, d, n_h, d_h, G = 4, 5, 4, 2, 2
    rng = np.random.default_rng(12)
    h = rng.standard_normal((T, d))
    W_Q, W_K, W_V, W_O = _make_mha_weights(d, n_h, d_h, n_h, seed=13)
    W_K_g, W_V_g = mha_to_gqa_mean_pool(W_K, W_V, n_h, d_h, G)
    u, _ = attention_forward(
        h, W_Q, W_K_g, W_V_g, W_O, num_heads=n_h, num_kv_heads=G
    )
    ref = _reference_by_paper_loops(h, W_Q, W_K_g, W_V_g, W_O, n_h, G)
    np.testing.assert_allclose(u, ref, rtol=1e-12, atol=1e-14)


def test_trace_records_worked_example_shapes():
    # dossier m02 worked example:T=3、H=2、d_h=4、d=6 的形状走一遍——
    # 投影 (3,8) → view (3,2,4) → 逐头分数 (2,3,3) → o (3,2,4)。
    T, d, n_h, d_h = 3, 6, 2, 4
    rng = np.random.default_rng(14)
    h = rng.standard_normal((T, d))
    W_Q, W_K, W_V, W_O = _make_mha_weights(d, n_h, d_h, n_h, seed=15)
    trace = {}
    u, (k_heads, v_heads) = attention_forward(
        h, W_Q, W_K, W_V, W_O, num_heads=n_h, trace=trace
    )
    assert trace["q"].shape == (T, n_h * d_h)      # (3,8) 投影
    assert trace["q_heads"].shape == (T, n_h, d_h)  # (3,2,4) view
    assert trace["scores"].shape == (n_h, T, T)     # (2,3,3) 逐头分数
    assert trace["probs"].shape == (n_h, T, T)
    assert trace["o_heads"].shape == (T, n_h, d_h)
    assert trace["scale"] == d_h ** -0.5            # Eq.(7) 的 √d_h
    assert k_heads.shape == (T, n_h, d_h) and v_heads.shape == (T, n_h, d_h)


def test_kv_cache_elements_from_forward_shapes():
    # §2.1.1 尾句 + §2.1.4 Table 1:MHA 每 token 缓存 2·n_h·d_h 个元素
    # (K、V 各 n_h 份);GQA-g 为 2·n_g·d_h;MQA 为 2·d_h——前向返回的
    # (k_heads, v_heads) 就是缓存内容本身。
    T, d, n_h, d_h = 2, 6, 4, 2
    rng = np.random.default_rng(16)
    h = rng.standard_normal((T, d))
    for G in (4, 2, 1):
        W_Q, W_K, W_V, W_O = _make_mha_weights(d, n_h, d_h, G, seed=17)
        _, (k_heads, v_heads) = attention_forward(
            h, W_Q, W_K, W_V, W_O, num_heads=n_h, num_kv_heads=G
        )
        per_token = (k_heads[0].size + v_heads[0].size)
        assert per_token == 2 * G * d_h


def test_causal_mask_no_future_leakage():
    # Eq.(7) 求和上限 j<=t:改动未来 token 的 hidden state 不影响历史输出。
    T, d, n_h, d_h = 5, 4, 2, 2
    rng = np.random.default_rng(18)
    h = rng.standard_normal((T, d))
    W_Q, W_K, W_V, W_O = _make_mha_weights(d, n_h, d_h, n_h, seed=19)
    u1, _ = attention_forward(h, W_Q, W_K, W_V, W_O, num_heads=n_h)
    h2 = h.copy()
    h2[3:] += 10.0        # 只改 t=3 及以后
    u2, _ = attention_forward(h2, W_Q, W_K, W_V, W_O, num_heads=n_h)
    np.testing.assert_array_equal(u1[:3], u2[:3])
    assert not np.allclose(u1[3:], u2[3:])
