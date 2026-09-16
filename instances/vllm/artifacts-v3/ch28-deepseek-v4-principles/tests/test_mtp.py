"""MTP 测试 —— 论文锚：
arXiv:2412.19437 §2.2 Eq.(21)(22)(23)(24)(25)（模块结构、逐深度损失、总损失、D=1、λ=0.3）；
V4 沿用声明见 arXiv:2606.19348 §2.1（without modification）。

对齐口径（T=5、D=1）：h'^1_i = M_1[RMSNorm(h^0_i); RMSNorm(Emb(t_{i+1}))] → TRM_1 → P^1_{i+2} = OutHead(h^1_i)；
损失对齐 t_{3:T+1}（= 0 基下标 2,3,4）。
"""
import numpy as np

from mtp import (
    mtp_alignment,
    mtp_depth_loss,
    mtp_input,
    mtp_predict,
    mtp_total_loss,
    out_head,
    rms_norm,
    trm_block,
)

RNG = np.random.default_rng(241219437)


def test_rms_norm_returns_unit_rms_rows():
    x = RNG.normal(size=(3, 5)).round(2)
    y = rms_norm(x, eps=1e-6)
    rms = np.sqrt((y**2).mean(axis=-1))
    np.testing.assert_allclose(rms, np.ones(3), atol=1e-5)
    # 带增益时按维缩放（Eq.21 的两个 RMSNorm 都是这种带权形式）
    w = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(rms_norm(x, w), y * w, atol=1e-12)


def test_mtp_alignment_T5_D1():
    a = mtp_alignment(T=5, k=1)
    assert a["hidden_indices"] == [0, 1, 2, 3]  # i+k ≤ T-1 ⇒ i ≤ 3
    assert a["targets"] == [(0, 2), (1, 3), (2, 4)]  # 预测 t_{i+k+1}
    # 论文的损失下标区间 t_{2+k:T+1}（1 基）= 0 基的 [2, 3, 4]
    assert [t for _, t in a["targets"]] == [2, 3, 4]
    assert a["n_loss_positions"] == 3


def test_mtp_alignment_depth_two():
    a = mtp_alignment(T=5, k=2)
    assert a["hidden_indices"] == [0, 1, 2]  # i+2 ≤ 4
    assert a["targets"] == [(0, 3), (1, 4)]
    assert a["n_loss_positions"] == 2


def test_mtp_input_concatenates_two_rmsnorms():
    T, d = 5, 3
    h_prev = RNG.normal(size=(T, d)).round(2)
    emb = RNG.normal(size=(T, d)).round(2)
    hnorm = np.ones(d)
    enorm = np.ones(d) * 2.0
    W_mix = RNG.normal(size=(2 * d, d)).round(2)  # M_1：2d → d

    h_prime, concat = mtp_input(h_prev, emb, W_mix, hnorm_w=hnorm, enorm_w=enorm, eps=1e-6)
    # Eq.(21)：先各自 RMSNorm 再拼接，再过 M_k
    want_concat = np.hstack([rms_norm(h_prev, hnorm, 1e-6), rms_norm(emb, enorm, 1e-6)])
    np.testing.assert_allclose(concat, want_concat, atol=1e-6)
    np.testing.assert_allclose(h_prime, want_concat @ W_mix, atol=1e-6)
    assert h_prime.shape == (T, d)


def test_trm_block_keeps_causal_chain():
    T, d = 5, 4
    h = RNG.normal(size=(T, d)).round(2)
    W_q = RNG.normal(size=(d, d)).round(2) * 0.3
    W_k = RNG.normal(size=(d, d)).round(2) * 0.3
    W_v = RNG.normal(size=(d, d)).round(2) * 0.3
    W_o = RNG.normal(size=(d, d)).round(2) * 0.3
    out = trm_block(h, W_q=W_q, W_k=W_k, W_v=W_v, W_o=W_o, eps=1e-6)

    h2 = h.copy()
    h2[-1] += 10.0  # 只改最后一个位置的输入
    out2 = trm_block(h2, W_q=W_q, W_k=W_k, W_v=W_v, W_o=W_o, eps=1e-6)
    # 因果链完整：后面的 token 变不了前面的输出（否则 MTP 的逐深度监督就漏看了未来）
    np.testing.assert_allclose(out[:-1], out2[:-1], atol=1e-10)
    assert not np.allclose(out[-1], out2[-1])


def test_mtp_predict_uses_shared_output_head():
    T, d, vocab = 5, 3, 7
    h = RNG.normal(size=(T, d)).round(2)
    W_out = RNG.normal(size=(vocab, d)).round(2)
    P = mtp_predict(h, W_out)
    assert P.shape == (T, vocab)
    np.testing.assert_allclose(P, h @ W_out.T, atol=1e-12)  # Eq.(23)：OutHead(h^k_i)
    np.testing.assert_allclose(out_head(h, W_out), P)


def test_mtp_depth_loss_matches_cross_entropy_over_valid_positions():
    logits = np.array([[2.0, 0.0, 1.0], [0.0, 3.0, 0.0], [1.0, 1.0, 1.0]])
    targets = np.array([0, 1, 2])
    loss = mtp_depth_loss(logits, targets)

    def ce(row, t):
        z = row - row.max()
        return -(z[t] - np.log(np.exp(z).sum()))

    np.testing.assert_allclose(loss, np.mean([ce(logits[i], targets[i]) for i in range(3)]), atol=1e-12)
    # Eq.(24) 的均值口径：每档一次交叉熵平均
    assert loss > 0


def test_mtp_total_loss_weighted_by_lambda_over_D():
    losses = [1.0]
    np.testing.assert_allclose(mtp_total_loss(losses, lam=0.3, D=1), 0.3, atol=1e-12)
    np.testing.assert_allclose(mtp_total_loss([1.0, 3.0], lam=0.3, D=2), 0.3 * (1.0 + 3.0) / 2, atol=1e-12)
    # 论文口径：D=1、λ=0.3 是默认值（函数默认值即论文超参）
    np.testing.assert_allclose(mtp_total_loss([2.0]), 0.6, atol=1e-12)
