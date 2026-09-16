"""mHC 测试 —— 论文锚：
arXiv:2606.19348 §2.2 Eq.(1)-(8)（V4 的 A/B/C 口径）、
arXiv:2512.24880 Eq.(3)(5)(6)(7)(8)(9) + §4.1（三映射、Birkhoff 多面体、Sinkhorn、t_max=20）、
arXiv:2409.19606（前身 HC：多流与病根）。
"""
import numpy as np

from mhc import (
    doubly_stochastic_residual,
    hc_head,
    is_doubly_stochastic,
    mhc_gates,
    mhc_raw_mappings,
    mhc_layer_input,
    mhc_update,
    rms_norm_flat,
    sinkhorn_knopp,
    sinkhorn_trace,
    spectral_norm,
)

RNG = np.random.default_rng(24880)

DOUBLY = np.array([[0.7, 0.3], [0.3, 0.7]])
NOT_DOUBLY = np.array([[0.9, 0.1], [0.2, 0.8]])


# ── §4.1 / Eq.(6)：双随机集合（Birkhoff 多面体）的三条性质 ─────────────
def test_is_doubly_stochastic_accepts_and_rejects():
    assert is_doubly_stochastic(DOUBLY)
    assert not is_doubly_stochastic(NOT_DOUBLY)  # 行和=1 但列和 = 1.1 / 0.9
    assert is_doubly_stochastic(np.array([[1.0, 0.0], [0.0, 1.0]]))  # 顶点=置换矩阵


def test_doubly_stochastic_spectral_norm_bound():
    # 论文原句：the spectral norm of the mapping matrix ‖B_l‖₂ is bounded by 1
    assert spectral_norm(DOUBLY) <= 1.0 + 1e-12
    # 反例：行和为 1 但列和不为 1 ⇒ 范数 > 1（深堆叠会放大信号）
    assert spectral_norm(NOT_DOUBLY) > 1.0
    np.testing.assert_allclose(spectral_norm(NOT_DOUBLY), 1.009583, atol=1e-5)


def test_birkhoff_set_is_closed_under_multiplication():
    # §4.1：该集合对乘法封闭 ⇒ 多层复合仍是双随机、范数仍 ≤ 1
    product = DOUBLY @ DOUBLY
    assert is_doubly_stochastic(product)
    np.testing.assert_allclose(product, np.array([[0.58, 0.42], [0.42, 0.58]]), atol=1e-12)
    assert spectral_norm(DOUBLY @ DOUBLY @ DOUBLY) <= 1.0 + 1e-12


# ── Eq.(8)(9)：Sinkhorn-Knopp 交替归一 ─────────────────────────────────
def test_sinkhorn_two_by_two_hand_computed():
    # dossier m14 的 worked example：[[4,1],[1,3]] 手做「行归一 → 列归一」
    M, trace = sinkhorn_knopp(np.array([[4.0, 1.0], [1.0, 3.0]]), iters=1, start="raw", trace=True)
    np.testing.assert_allclose(M, [[0.761905, 0.210526], [0.238095, 0.789474]], atol=1e-5)
    # 第一轮里：行归一后行和 = 1；列归一后列和 = 1，行和被打散
    np.testing.assert_allclose(trace[0]["row_sums_after_row"], [1.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(trace[0]["col_sums_after_row"], [1.05, 0.95], atol=1e-5)


def test_sinkhorn_converges_toward_doubly_stochastic():
    raw = np.array([[4.0, 1.0], [1.0, 3.0]])
    devs = []
    for iters in (1, 2, 5, 20):
        M = sinkhorn_knopp(raw, iters=iters, start="raw")
        devs.append(max(np.abs(M.sum(axis=1) - 1).max(), np.abs(M.sum(axis=0) - 1).max()))
    assert devs[0] > devs[1] > devs[2] > devs[3]  # 每多迭代更接近双随机
    assert devs[3] < 2e-6  # eps=1e-6 给偏差设了地板（分母加 eps）
    M20 = sinkhorn_knopp(raw, iters=20, start="raw")
    assert is_doubly_stochastic(M20, atol=1e-6)
    # t_max=20 是实用折中、不是精确投影：第 1 轮还远不是双随机
    assert devs[0] > 1e-2


def _reference_sinkhorn_literal(logits, iters=20, eps=1e-6):
    """两侧实现的**字面转写**（官方 DeepSeek-V4-Pro inference/kernel.py:L401-L423 的
    `hc_split_sinkhorn`；pin 的 torch 回退 vllm/model_executor/kernels/mhc/torch.py:L75-L91
    同序），用来给本实现做独立对账：
    `softmax(dim=-1) + eps` → 列归一 → 再 (iters−1) 轮 (行, 列) 归一。"""
    z = logits - logits.max(axis=-1, keepdims=True)
    M = np.exp(z) / np.exp(z).sum(axis=-1, keepdims=True)
    M = M + eps
    M = M / (M.sum(axis=-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        M = M / (M.sum(axis=-1, keepdims=True) + eps)
        M = M / (M.sum(axis=-2, keepdims=True) + eps)
    return M


def test_sinkhorn_matches_reference_implementation_literal():
    raw = np.array([[1.0, 0.5], [0.2, 1.2]])
    # 论文：M^(0) = exp(H̃^res)；两侧实现：softmax(dim=-1)（= exp + 行归一）起步。
    # 两者只差 eps 的落点 ⇒ 数值上差 ~4e-7（实测），远小于 20 轮的实用精度。
    m_exp = sinkhorn_knopp(raw, iters=20, start="exp", order="row-first")
    m_ref = sinkhorn_knopp(raw, iters=20, start="softmax", order="row-first")
    literal = _reference_sinkhorn_literal(raw)
    assert np.abs(m_exp - m_ref).max() < 1e-6
    assert np.abs(m_ref - literal).max() < 1e-6


def test_sinkhorn_iteration_order_agrees_at_the_limit():
    raw = np.array([[1.0, 0.5], [0.2, 1.2]])
    # 论文书面顺序 T_r(T_c(·))（列先）与两侧实现的顺序（行先）：极限相同
    m_row = sinkhorn_knopp(raw, iters=20, order="row-first")
    m_col = sinkhorn_knopp(raw, iters=20, order="col-first")
    assert is_doubly_stochastic(m_col, atol=1e-6)
    assert np.abs(m_row - m_col).max() < 1e-9
    # 但低轮数下定点确实不同（同一张矩阵、只改轮内顺序）
    b = np.random.default_rng(7).normal(size=(3, 3)) * 2
    d1 = np.abs(sinkhorn_knopp(b, iters=1, order="row-first") - sinkhorn_knopp(b, iters=1, order="col-first")).max()
    assert d1 > 1e-2


def test_sinkhorn_twenty_is_practical_not_exact():
    # 论文原话：We choose t_max=20 as a practical value —— 精度与开销的折中
    dev = lambda M: max(np.abs(M.sum(1) - 1).max(), np.abs(M.sum(0) - 1).max())
    mats = np.random.default_rng(24880).normal(size=(8, 3, 3)) * 2  # 全正、条件数差别很大
    d20 = [dev(sinkhorn_knopp(b, iters=20)) for b in mats]
    d200 = [dev(sinkhorn_knopp(b, iters=200)) for b in mats]
    assert max(d20) > 1e-5  # 有的矩阵 20 轮还没到位（收敛快慢取决于矩阵本身）
    assert min(d20) < 2e-6  # 多数矩阵 20 轮就够了 ⇒ 20 是"实用值"不是"精确值"
    assert max(d200) < 2e-6 and max(d200) <= max(d20)  # 继续迭代能到位，且不会更差


def test_sinkhorn_trace_reports_every_round():
    rows = sinkhorn_trace(np.array([[4.0, 1.0], [1.0, 3.0]]), iters=3, start="raw")
    assert len(rows) == 3
    assert set(rows[0]) >= {"row_sums_after_row", "col_sums_after_row", "row_dev", "col_dev"}
    assert rows[-1]["row_dev"] < rows[0]["row_dev"]


# ── Eq.(3)(4)(5)：一次 GEMM 出 A/B/C 的 raw 参数 ────────────────────────
def test_flat_rms_norm_scale_invariance():
    x = RNG.normal(size=(2, 3, 4)).round(2)
    n1 = rms_norm_flat(x, eps=1e-6)
    n2 = rms_norm_flat(2.0 * x, eps=1e-6)
    # 无权 RMSNorm 对缩放不变；eps=1e-6 带来 ~1e-6 量级的相对偏差（不是位级相等）
    np.testing.assert_allclose(n1, n2, atol=1e-5)


def test_mhc_raw_mappings_split_and_formula():
    hc, d = 2, 3
    mix = (2 + hc) * hc
    X = RNG.normal(size=(4, hc, d)).round(2)  # (T, n_hc, d)
    fn = RNG.normal(size=(mix, hc * d)).round(2) * 0.1
    base = RNG.normal(size=(mix,)).round(2) * 0.1
    scale = np.array([0.3, 0.4, 0.5])

    A_tilde, B_tilde, C_tilde = mhc_raw_mappings(X, fn, base, scale, eps=1e-6)
    assert A_tilde.shape == (4, hc)
    assert B_tilde.shape == (4, hc, hc)
    assert C_tilde.shape == (4, hc, 1)

    flat = rms_norm_flat(X.reshape(4, hc * d), eps=1e-6)
    mixes = flat @ fn.T
    # 两侧实现的打包顺序是 (pre, post, res)（官方 kernel.py:L391-L396）—— 与论文 Eq.(3)(4)(5) 的列举顺序不同
    np.testing.assert_allclose(A_tilde, mixes[:, :hc] * scale[0] + base[:hc], atol=1e-6)
    np.testing.assert_allclose(C_tilde[:, :, 0], mixes[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc], atol=1e-6)
    np.testing.assert_allclose(
        B_tilde, (mixes[:, 2 * hc :] * scale[2] + base[2 * hc :]).reshape(4, hc, hc), atol=1e-6
    )


def test_mhc_gates_ranges_match_paper():
    rng = np.random.default_rng(20260916)  # 本地 RNG：不让同文件其它测试的抽样顺序影响本例
    A_tilde = rng.normal(size=(5, 3)) * 2
    B_tilde = rng.normal(size=(5, 3, 3)) * 2
    C_tilde = rng.normal(size=(5, 3, 1)) * 2
    A, B, C = mhc_gates(A_tilde, B_tilde, C_tilde, sinkhorn_iters=20, eps=1e-6)

    assert np.all((A > 0) & (A < 1))  # Eq.(6)：A = σ(Ã)
    assert np.all((C > 0) & (C < 2))  # Eq.(7)：C = 2σ(C̃)
    for b in B:
        # Eq.(8)：B 落在 Birkhoff 多面体。20 轮是实用折中 ⇒ 判据取 1e-3（条件数差的矩阵
        # 20 轮内到不了 1e-6，见 test_sinkhorn_twenty_is_practical_not_exact）。
        assert is_doubly_stochastic(b, atol=1e-3)
    # 两侧实现的 σ 后加 eps（数值保护；官方 kernel.py:L391-L392）⇒ A 严格大于 0
    assert np.all(A >= 1e-6)


# ── Eq.(1)：更新式与恒等映射性质 ────────────────────────────────────────
def test_mhc_update_is_identity_mapping_when_b_and_c_trivial():
    X = RNG.normal(size=(3, 2, 4)).round(2)
    A = np.array([0.5, 0.5])
    B = np.eye(2)
    C = np.ones((2, 1))
    sublayer_input = mhc_layer_input(A, X)
    sublayer_out = RNG.normal(size=(3, 4)).round(2)  # 假设子层把输入映射成这个（F(A X)）
    X_next = mhc_update(X, B, C, sublayer_out)
    # B=I、C=1 ⇒ 每条流 = 原流 + 子层输出（恒等映射那条直通线还在）
    np.testing.assert_allclose(X_next, X + sublayer_out[:, None, :], atol=1e-6)
    assert sublayer_input.shape == (3, 4)


def test_mhc_layer_input_is_weighted_sum_of_streams():
    X = RNG.normal(size=(2, 3, 4)).round(2)
    A = np.array([0.2, 0.3, 0.5])
    collapsed = mhc_layer_input(A, X)
    np.testing.assert_allclose(collapsed, (A[:, None] * X).sum(axis=1), atol=1e-12)
    assert collapsed.shape == (2, 4)


def test_doubly_stochastic_mixing_does_not_amplify_signal():
    # 约束的意义：任意层数复合后 ‖B‖₂ ≤ 1（信号不放大）。
    # 把信号对准最大奇异方向来看这条性质（否则随机方向可能恰好躲开放大方向）。
    v = np.linalg.svd(NOT_DOUBLY)[2][0]
    X = np.zeros((1, 2, 1))
    X[0, :, 0] = v

    out = doubly_stochastic_residual(DOUBLY, X)
    assert np.linalg.norm(out) <= np.linalg.norm(X) + 1e-9  # 双随机：绝不放大
    bad = doubly_stochastic_residual(NOT_DOUBLY, X)
    assert np.linalg.norm(bad) > np.linalg.norm(X)  # 行和为 1 但列和不为 1 ⇒ 放大


# ── V4 自造件：hc_head（把 n_hc 条流压回单流） ─────────────────────────
def test_hc_head_is_sigmoid_gated_weighted_sum():
    hc, d = 2, 3
    x = RNG.normal(size=(4, hc, d)).round(2)
    hc_fn = RNG.normal(size=(hc, hc * d)).round(2) * 0.1
    hc_base = np.array([0.1, -0.2])
    hc_scale = np.array([0.7])
    out = hc_head(x, hc_fn, hc_base, hc_scale, eps=1e-6)
    assert out.shape == (4, d)

    mixes = rms_norm_flat(x.reshape(4, hc * d), eps=1e-6) @ hc_fn.T
    pre = 1.0 / (1.0 + np.exp(-(mixes * hc_scale[0] + hc_base))) + 1e-6
    np.testing.assert_allclose(out, (pre[:, :, None] * x).sum(axis=1), atol=1e-6)
    # 门控在 (0,1)：加权求和是凸组合 ⇒ 输出范数不超过最大流
    assert np.all(pre > 0) and np.all(pre < 1.0 + 1e-6)
