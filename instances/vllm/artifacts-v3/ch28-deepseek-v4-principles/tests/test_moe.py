"""MoE 与 hash 路由测试 —— 论文锚：
arXiv:2401.06066 Eq.(3)-(11)（标准 MoE → 细粒度切分 → 共享专家隔离）与 Eq.(12)-(17)（均衡损失）、
arXiv:2106.04426（hash 路由：按原始输入 token 查表）、
arXiv:2606.19348 §2.1（Sqrt(Softplus) 亲和分、auxiliary-loss-free、hash 前 3 层）。
"""
import numpy as np

from moe import (
    affinity_scores_v3,
    affinity_scores_v4,
    deepseekmoe_forward,
    device_balance_loss,
    expert_balance_loss,
    fine_grained_counts,
    hash_route,
    moe_gate_scores,
    segment_counts,
    shared_expert_output,
    standard_moe_forward,
    topk_gate,
    topk_with_correction_bias,
)

RNG = np.random.default_rng(106066)


# ── Eq.(3)(4)(5)：标准 MoE ─────────────────────────────────────────────
def test_moe_gate_scores_are_softmax_over_expert_embeddings():
    u = np.array([1.0, 0.0, -1.0])
    E = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    s = moe_gate_scores(u, E)
    assert s.shape == (4,)
    np.testing.assert_allclose(s.sum(), 1.0)
    logits = E @ u
    np.testing.assert_allclose(s, np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum())


def test_topk_gate_keeps_k_and_zeroes_rest():
    s = np.array([0.1, 0.5, 0.3, 0.05, 0.05])
    g = topk_gate(s, k=2)
    assert np.count_nonzero(g) == 2
    assert np.argmax(g) == 1 and g[2] == 0.3
    assert np.isclose(g.sum(), 0.8)  # Eq.(4)：只留 top-k 的分值，不重归一


def test_standard_moe_forward_is_weighted_sum_plus_residual():
    u = np.array([1.0, 2.0])
    expert_outs = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])  # 3 个专家的 FFN_i(u)
    g = np.array([0.4, 0.0, 0.6])
    h = standard_moe_forward(u, expert_outs, g)
    np.testing.assert_allclose(h, 0.4 * expert_outs[0] + 0.6 * expert_outs[2] + u)  # Eq.(3)


# ── Eq.(6)(7)(8)：板斧一，细粒度切分 ───────────────────────────────────
def test_fine_grained_counts_keep_compute_and_grow_combinations():
    assert fine_grained_counts(N=4, K=2, m=2) == (8, 4)  # mN 个专家、激活 mK 个
    assert segment_counts(8, 4) > segment_counts(4, 2)  # 组合空间变大
    assert segment_counts(4, 2) == 6 and segment_counts(8, 4) == 70


# ── Eq.(9)(10)(11)：板斧二，共享专家无条件计算 ─────────────────────────
def test_shared_experts_are_unconditional():
    u = np.array([1.0, 1.0])
    shared_outs = [np.array([3.0, 0.0]), np.array([0.0, 5.0])]  # K_s=2 个共享专家
    routed_outs = [np.array([1.0, 1.0]), np.array([2.0, 2.0])]
    g1 = np.array([0.5, 0.25])

    np.testing.assert_allclose(shared_expert_output(shared_outs), np.array([3.0, 5.0]))
    # Eq.(9)：Σ_{i≤Ks} FFN_i(u) + Σ_{i>Ks} g_i FFN_i(u) + u
    out = deepseekmoe_forward(u, shared_outs, routed_outs, g1)
    np.testing.assert_allclose(
        out, np.array([3.0, 5.0]) + 0.5 * routed_outs[0] + 0.25 * routed_outs[1] + u
    )
    # 换一组路由门控：只有路由项变，共享项一字不动（Eq.9 第一项没有 gate）
    g2 = np.array([0.0, 1.0])
    out2 = deepseekmoe_forward(u, shared_outs, routed_outs, g2)
    np.testing.assert_allclose(
        out2 - (g2[0] * routed_outs[0] + g2[1] * routed_outs[1]), np.array([3.0, 5.0]) + u
    )


def test_shared_expert_routing_pool_excludes_shared_experts():
    # Eq.(10)：路由只在 K_s 之后的 mN−K_s 个专家里挑 mK−K_s 个
    s = np.array([0.9, 0.8, 0.3, 0.2, 0.1])  # 前 2 个是共享专家
    g = topk_gate(s, k=2, start=2)
    assert g[0] == 0 and g[1] == 0
    assert np.count_nonzero(g) == 2
    assert set(np.nonzero(g)[0].tolist()).issubset({2, 3, 4})


# ── V4 §2.1：Sqrt(Softplus) 亲和分 ─────────────────────────────────────
def test_affinity_scores_v4_is_sqrt_softplus_and_keeps_growing():
    z = np.array([-5.0, 0.0, 3.0, 10.0])
    s = affinity_scores_v4(z)
    np.testing.assert_allclose(s, np.sqrt(np.log1p(np.exp(z))), atol=1e-12)
    assert np.all(np.diff(s) > 0)
    # 对照 V3 的 Sigmoid：大 logits 处饱和（把分布压平）
    assert affinity_scores_v3(np.array([10.0]))[0] < 1.0
    assert s[-1] > affinity_scores_v3(np.array([10.0]))[0]


# ── arXiv:2606.19348 §2.1：bias 只进选择、不进权重 ─────────────────────
def test_topk_with_correction_bias_selection_only():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    bias = np.array([0.0, 0.0, 1.0, 0.0])
    idx, w = topk_with_correction_bias(scores, bias, k=2)
    # 带偏分数 s+b = [0.9, 0.8, 1.2, 0.1] ⇒ 专家 3（下标 2）被推进 top-2、挤掉下标 1
    assert set(idx.tolist()) == {0, 2}
    order = np.argsort(idx)  # 归一化到下标升序再比对（返回顺序按分值降序）
    np.testing.assert_allclose(w[order], np.array([0.9, 0.2]) / 1.1, atol=1e-12)  # 权重仍来自无偏分数

    _, w_biased = topk_with_correction_bias(scores, bias, k=2, use_biased_weights=True)
    np.testing.assert_allclose(w_biased[order], np.array([0.9, 1.2]) / 2.1, atol=1e-12)
    # 注释点破的现象：用带偏分数当权重会把分布压平（熵更大）
    ent = lambda p: -(p * np.log(p)).sum()
    assert ent(w_biased) > ent(w)


def test_hash_route_lookup_only_uses_token_ids():
    vocab, topk = 6, 2
    table = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]])
    tokens = np.array([4, 1, 0])
    # scores = 已过亲和激活的分数（V4 口径 sqrt(softplus(·))），不是原始 logits
    scores = affinity_scores_v4(RNG.normal(size=(3, 6)).round(2))
    idx, w = hash_route(table, tokens, scores)

    np.testing.assert_array_equal(idx, table[tokens])
    rows = np.arange(3)[:, None]
    want = scores[rows, table[tokens]]
    np.testing.assert_allclose(w, want / want.sum(axis=1, keepdims=True))
    assert np.all(w > 0)  # 亲和分非负 ⇒ 权重非负
    # 换掉 hidden（连带 scores 全变）也不改变"选谁"——选谁由 token id 钉死
    idx2, _ = hash_route(table, tokens, affinity_scores_v4(RNG.normal(size=(3, 6))))
    np.testing.assert_array_equal(idx, idx2)


# ── Eq.(12)-(17)：均衡损失（谱系背景，V4 不再用它作主力） ───────────────
def test_expert_balance_loss_hand_computed():
    # N'=4 个路由专家、K'=2、T=2 个 token；alpha=1
    selections = [np.array([0, 1]), np.array([1, 2])]
    s = np.array([[0.5, 0.5, 0.0, 0.0], [0.1, 0.6, 0.3, 0.0]])
    loss = expert_balance_loss(selections, s, alpha=1.0)
    f = np.array([1.0, 2.0, 1.0, 0.0])  # f_i = N'/(K'T)·Σ_t 1(t 选 i)
    P = s.mean(axis=0)
    np.testing.assert_allclose(loss, (f * P).sum(), atol=1e-12)
    assert np.isclose(f.sum(), 4.0)  # 归一化约定：Σf_i = N'


def test_device_balance_loss_groups_experts():
    f = np.array([1.0, 2.0, 1.0, 0.0])
    P = np.array([0.2, 0.3, 0.1, 0.4])
    loss = device_balance_loss(f, P, groups=[[0, 1], [2, 3]], alpha=0.5)
    f_groups = np.array([1.5, 0.5])
    P_groups = np.array([0.5, 0.5])
    np.testing.assert_allclose(loss, 0.5 * (f_groups * P_groups).sum(), atol=1e-12)
