# -*- coding: utf-8 -*-
"""DSpark 半自回归 drafter —— §2.2 Eq.(2)(3)(DFlash 骨干 KV 注入)+ §3.1 Eq.(4)(5)
(序列阶段/Markov 头/anchor-as-first)+ §3.2.1 Eq.(7)(8)(置信度头)。

手算基准:
- 'of course'/'no problem' 多模态碰撞 → 'of problem'(§3.1 开篇例子);
- 低秩账:V=129280、r=256 → W_1/W_2 各 ≈33.1M、全表 V×V ≈1.67e10(§3.1/§4.3.2,≈253x 省);
- Eq.(7) c_k=σ(w·[h_k;W_1[x_{k−1}]]);Eq.(8) c*=1−½‖p_d−p_t‖₁=0.8((0.5,0.5)vs(0.7,0.3))。
"""
import numpy as np
import pytest

from dspark import (
    MASK,
    rms_norm,
    project_context,
    bidirectional_attention_with_context,
    ParallelBackbone,
    MarkovHead,
    ConfidenceHead,
    markov_head_param_account,
    analytical_confidence_label,
    anchor_as_first_inputs,
    fill_in_inputs,
    query_layout,
    num_lookahead_slots,
    sample_sequential,
    dspark_draft,
    marginal_over_predecessors,
    multimodal_collision_demo,
    make_toy_backbone,
)
from spec_decode import acceptance_rate_sum_min, softmax_lastdim


# ── Eq.(2) 上下文投影 + RMSNorm 算子 ──

def test_rms_norm_hand():
    x = np.array([[3.0, 4.0]])
    got = rms_norm(x, eps=0.0)
    assert np.allclose(got, x / np.sqrt(12.5))
    # 行 RMS ≈ 1
    h = np.random.default_rng(0).normal(size=(4, 6))
    assert np.allclose(np.sqrt(np.mean(rms_norm(h) ** 2, axis=-1)), 1.0, atol=1e-5)


def test_project_context_eq2():
    # H_ctx = RMSNorm(W_c[H^{(l1)};…;H^{(lm)}]):拼接沿特征维(m·d → d),再逐行 RMSNorm
    rng = np.random.default_rng(1)
    concat = rng.normal(size=(3, 8))          # T=3,m·d=8(两层 target 隐状态拼接)
    W_c = rng.normal(size=(4, 8))             # d=4
    got = project_context(concat, W_c)
    manual = rms_norm(concat @ W_c.T)
    assert np.allclose(got, manual)
    assert got.shape == (3, 4)


# ── Eq.(3) 注入式双向注意 ──

def test_bidirectional_attention_eq3_manual():
    rng = np.random.default_rng(2)
    d = 3
    ctx = rng.normal(size=(2, d))
    h_d = rng.normal(size=(2, d))
    Wq, Wk, Wv, Wo = (rng.normal(size=(d, d)) for _ in range(4))
    K_ctx, V_ctx = ctx @ Wk.T, ctx @ Wv.T
    got = bidirectional_attention_with_context(h_d, K_ctx, V_ctx, Wq, Wk, Wv, Wo)
    # 手工:K=[W_K H_ctx; W_K H_d] 序列维拼接、块查询双向注意全部键
    K = np.vstack([K_ctx, h_d @ Wk.T])
    V = np.vstack([V_ctx, h_d @ Wv.T])
    scores = (h_d @ Wq.T) @ K.T / np.sqrt(d)
    w = softmax_lastdim(scores)
    assert np.allclose(got, h_d + (w @ V) @ Wo.T)


def test_bidirectional_attention_block_depends_on_later_positions():
    # 双向:块位置 0 的输出依赖块位置 1(因果注意则不会)
    rng = np.random.default_rng(3)
    d = 3
    ctx = rng.normal(size=(2, d))
    Wq, Wk, Wv, Wo = (rng.normal(size=(d, d)) for _ in range(4))
    K_ctx, V_ctx = ctx @ Wk.T, ctx @ Wv.T
    h_a = rng.normal(size=(2, d))
    h_b = h_a.copy()
    h_b[1] += 5.0  # 只改块位置 1
    out_a = bidirectional_attention_with_context(h_a, K_ctx, V_ctx, Wq, Wk, Wv, Wo)
    out_b = bidirectional_attention_with_context(h_b, K_ctx, V_ctx, Wq, Wk, Wv, Wo)
    assert not np.allclose(out_a[0], out_b[0])


# ── 并行骨干:预计算 + 单次前向 ──

def test_backbone_shapes_single_pass_any_gamma():
    # 单次前向:γ 个输入 → h [γ,d] 与 U [γ,V](T_draft 与 γ 无关的结构性来源)
    bb = make_toy_backbone(vocab_size=6, d=4, num_layers=2, num_target_layers=2, seed=0)
    rng = np.random.default_rng(4)
    ctx = bb.precompute_context(rng.normal(size=(5, 8)))
    for gamma in (1, 2, 3, 5):
        inputs = anchor_as_first_inputs(anchor_id=2, gamma=gamma)
        h, U = bb.forward(inputs, ctx)
        assert h.shape == (gamma, 4) and U.shape == (gamma, 6)


def test_backbone_context_injection_changes_predictions():
    # Eq.(3) 注入:不同上下文 → 不同 logits(KV cache 寻址=序列维拼接)
    bb = make_toy_backbone(vocab_size=6, d=4, num_layers=1, num_target_layers=2, seed=1)
    rng = np.random.default_rng(5)
    ctx_a = bb.precompute_context(rng.normal(size=(4, 8)))
    ctx_b = bb.precompute_context(rng.normal(size=(4, 8)))
    inputs = anchor_as_first_inputs(anchor_id=0, gamma=3)
    _, U_a = bb.forward(inputs, ctx_a)
    _, U_b = bb.forward(inputs, ctx_b)
    assert not np.allclose(U_a, U_b)


def test_backbone_shares_target_embed_and_lm_head():
    # §2.2 末句:draft 共享 target 的 embedding 与 lm_head(冻结)——引用同一对象
    rng = np.random.default_rng(6)
    embed = rng.normal(size=(6, 4))
    lm_head = rng.normal(size=(6, 4))
    bb = ParallelBackbone(
        embed_matrix=embed,
        lm_head=lm_head,
        mask_embed=rng.normal(size=4),
        W_c=rng.normal(size=(4, 8)),
        layers=[tuple(rng.normal(size=(4, 4)) for _ in range(4))],
    )
    assert bb.embed_matrix is embed
    assert bb.lm_head is lm_head


def test_backbone_mask_slot_uses_mask_embedding():
    # mask 槽不走词表嵌入,走 mask_embed(MASK=-1)
    rng = np.random.default_rng(7)
    bb = make_toy_backbone(vocab_size=6, d=4, num_layers=1, num_target_layers=1, seed=2)
    assert bb.forward([0], bb.precompute_context(rng.normal(size=(1, 4))))[0].shape == (1, 4)
    assert bb.forward([MASK, MASK], bb.precompute_context(rng.normal(size=(1, 4))))[0].shape == (2, 4)


# ── Eq.(5) Markov 头 + 低秩账 ──

def test_markov_head_eq5():
    rng = np.random.default_rng(8)
    w1, w2 = rng.normal(size=(5, 3)), rng.normal(size=(3, 5))
    head = MarkovHead(w1, w2)
    for x in range(5):
        assert np.allclose(head.bias(head.embed([x]))[0], w1[x] @ w2)  # B(x,·)=W_1[x]W_2


def test_markov_param_account():
    acc = markov_head_param_account(vocab_size=129280, rank=256)
    assert acc["low_rank"] == 2 * 129280 * 256          # ≈6.6e7(W_1/W_2 各 ≈33.1M)
    assert acc["full"] == 129280 * 129280               # ≈1.67e10(V×V 全表)
    assert acc["saving_factor"] == pytest.approx(252.5, abs=0.1)  # V/(2r);论文口径 ≈253x


# ── Eq.(7)(8) 置信度头 ──

def test_confidence_head_eq7():
    w = np.array([1.0, -1.0, 0.5, 0.5])   # d=2, r=2
    head = ConfidenceHead(w)
    h_k = np.array([0.3, 0.2])
    emb_prev = np.array([0.1, -0.1])
    z = 0.3 - 0.2 + 0.05 - 0.05
    assert head.confidence(h_k, emb_prev) == pytest.approx(1.0 / (1.0 + np.exp(-z)))


def test_analytical_confidence_label_eq8():
    # c* = 1 − ½‖p_d − p_t‖₁;(0.5,0.5) vs (0.7,0.3) → 0.8(与 §2.1 Σmin=0.8 同一条 TV 恒等式)
    assert analytical_confidence_label(np.array([0.5, 0.5]), np.array([0.7, 0.3])) == pytest.approx(0.8)
    assert analytical_confidence_label(np.array([0.5, 0.5]), np.array([0.7, 0.3])) == pytest.approx(
        acceptance_rate_sum_min(np.array([0.7, 0.3]), np.array([0.5, 0.5]))
    )


# ── anchor-as-first 布局(§3.1 "minor modification") ──

def test_input_layouts_dspark_vs_dflash():
    # DSpark:γ 输入(anchor + γ−1 mask)出 γ logits;DFlash:1+γ 输入(anchor + γ mask)
    assert anchor_as_first_inputs(anchor_id=9, gamma=4) == [9, MASK, MASK, MASK]
    assert fill_in_inputs(anchor_id=9, gamma=4) == [9, MASK, MASK, MASK, MASK]


def test_query_layout_table_n5():
    # m09 worked example:N=5 小表——DSpark 恰 5 查询/全位采样/预测位=查询位+1;
    # DFlash 6 查询/anchor 位不采样/偏移 1 起
    ds = query_layout("dspark", 5)
    assert ds["num_queries"] == 5
    assert [(r["query_off"], r["input"], r["sampled"], r["target_off"]) for r in ds["records"]] == [
        (0, "anchor", True, 1),
        (1, "mask", True, 2),
        (2, "mask", True, 3),
        (3, "mask", True, 4),
        (4, "mask", True, 5),
    ]
    df = query_layout("dflash", 5)
    assert df["num_queries"] == 6
    assert [(r["query_off"], r["input"], r["sampled"], r["target_off"]) for r in df["records"]] == [
        (0, "anchor", False, None),
        (1, "mask", True, 1),
        (2, "mask", True, 2),
        (3, "mask", True, 3),
        (4, "mask", True, 4),
        (5, "mask", True, 5),
    ]


def test_num_lookahead_slots():
    # 调度器侧:DSpark 恰 N(anchor 是预测位)、DFlash N+1(anchor 是 bonus 槽)
    assert num_lookahead_slots("dspark", 5) == 5
    assert num_lookahead_slots("dflash", 5) == 6


# ── Eq.(4) 序列阶段:左到右 prev 链 ──

def test_sample_sequential_greedy_prev_chain():
    # γ=3 手推 prev 链(x_0=anchor):logits_k = U_k + B(x_{k−1}),greedy argmax
    rng = np.random.default_rng(9)
    w1, w2 = rng.normal(size=(4, 2)), rng.normal(size=(2, 4))
    head = MarkovHead(w1, w2)
    U = rng.normal(size=(3, 4))
    tokens, dists, steps = sample_sequential(U, anchor_token=2, markov_head=head, greedy=True)
    # 手工重放左到右规则
    prev, expect_tokens, expect_logits = 2, [], []
    for k in range(3):
        logits_k = U[k] + head.bias(head.embed([prev]))[0]
        x = int(np.argmax(logits_k))
        expect_tokens.append(x)
        expect_logits.append(logits_k)
        prev = x
    assert tokens == expect_tokens
    for k in range(3):
        assert np.allclose(steps[k]["logits"], expect_logits[k])
        assert np.allclose(dists[k], softmax_lastdim(expect_logits[k]))
        assert steps[k]["prev"] == (2 if k == 0 else expect_tokens[k - 1])
    # 概率分布行和为 1(逐位条件分布 = 验证时的 q(x))
    assert np.allclose(dists.sum(axis=-1), 1.0)


def test_sample_sequential_sampling_statistical():
    # 强偏置下采样频率跟随 p_k(序列阶段对条件分布采样)
    rng = np.random.default_rng(10)
    head = MarkovHead(np.ones((4, 1)), np.array([[0.0, 0.0, 6.0, 0.0]]))  # B(·,C)=+6 恒定(常数嵌入)
    U = np.zeros((1, 4))
    counts = np.zeros(4)
    for _ in range(1000):
        toks, dists, _ = sample_sequential(U, anchor_token=0, markov_head=head, rng=rng)
        counts[toks[0]] += 1
    assert counts[2] / 1000 > 0.95  # softmax(+6) 主导 token C


# ── 端到端 draft(并行阶段 → 序列阶段 → 置信度) ──

def test_dspark_draft_end_to_end():
    bb = make_toy_backbone(vocab_size=6, d=4, num_layers=2, num_target_layers=2, seed=3)
    rng = np.random.default_rng(11)
    ctx = bb.precompute_context(rng.normal(size=(5, 8)))
    markov = MarkovHead(rng.normal(size=(6, 3)), rng.normal(size=(3, 6)))
    conf = ConfidenceHead(rng.normal(size=4 + 3))
    r1 = dspark_draft(bb, ctx, anchor_token=2, gamma=3, markov_head=markov,
                      conf_head=conf, greedy=True)
    r2 = dspark_draft(bb, ctx, anchor_token=2, gamma=3, markov_head=markov,
                      conf_head=conf, greedy=True)
    assert r1["draft_tokens"] == r2["draft_tokens"]  # greedy 确定性
    assert len(r1["draft_tokens"]) == 3
    assert r1["draft_probs"].shape == (3, 6)
    assert np.allclose(r1["draft_probs"].sum(axis=-1), 1.0)
    # 逐位条件分布 = softmax(U_k + B(prev)):由公开接口交叉复算
    prev = 2
    for k in range(3):
        logits_k = r1["base_logits"][k] + markov.bias(markov.embed([prev]))[0]
        assert np.allclose(r1["draft_probs"][k], softmax_lastdim(logits_k))
        assert np.allclose(
            r1["confidences"][k], conf.confidence(r1["hiddens"][k], markov.embed([prev])[0])
        )
        prev = r1["draft_tokens"][k]
    assert all(0.0 < c < 1.0 for c in r1["confidences"])
    # 概率模式:输出随 rng 变化(采样而非 argmax)
    r3 = dspark_draft(bb, ctx, anchor_token=2, gamma=3, markov_head=markov,
                      conf_head=conf, rng=np.random.default_rng(42))
    assert r3["draft_probs"].shape == (3, 6)


# ── §3.1 多模态碰撞 ──

def test_marginal_over_predecessors():
    # p(x_2) = Σ_{x_1} p(x_1) p(x_2|x_1):对前驱边缘化的形式化
    p_prev = np.array([0.6, 0.4])
    cond = np.array([[0.1, 0.9], [0.5, 0.5]])
    assert np.allclose(marginal_over_predecessors(p_prev, cond), [0.26, 0.74])


def test_multimodal_collision_demo():
    demo = multimodal_collision_demo(greedy=True)
    vocab = demo["vocab"]  # ['<anchor>', 'of', 'no', 'course', 'problem']
    OF, NO, COURSE, PROBLEM = vocab.index("of"), vocab.index("no"), vocab.index("course"), vocab.index("problem")
    # 并行 drafter:逐位独立 argmax → 'of problem'(跨 mode 拼接)
    assert demo["parallel_block"] == [OF, PROBLEM]
    # 位 2 边缘:'problem' 主导(两 mode 混合、'no problem' 更尖)
    assert demo["marginal_x2"][PROBLEM] > demo["marginal_x2"][COURSE]
    # DSpark:位置 1 采出 'of' 后,Markov 头抬 'course' 压 'problem' → 'of course'
    assert demo["dspark_block"] == [OF, COURSE]
    # 并行独立采样的连贯率(解析)远低于 DSpark 采样连贯率(经验)
    assert demo["parallel_coherent_prob"] == pytest.approx(0.445, abs=0.01)
    rng = np.random.default_rng(12)
    coherent = 0
    n = 500
    for _ in range(n):
        toks, _, _ = sample_sequential(
            demo["base_logits"], anchor_token=0, markov_head=MarkovHead(demo["markov_w1"], demo["markov_w2"]),
            rng=rng,
        )
        coherent += toks in ([OF, COURSE], [NO, PROBLEM])
    assert coherent / n > 0.95
