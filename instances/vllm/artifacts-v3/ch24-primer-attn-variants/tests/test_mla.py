"""arXiv:2405.04434 §2.1.2-§2.1.3 Eq.(9)-(19)+ Appendix C Eq.(37)-(47)+ §3.1.2
(RMSNorm/scale 注)+ App B.1(V2-Lite 不压 query)—— MLA 参考实现的性质测试:
naive 展开形态 == 附录 C 计算序循环参考;吸收形态(App C 结合律:W^UK→W^UQ、
W^UV→W^O,离线一次)== naive 逐位恒等;RoPE 只旋 R 段、分数分母 √(d_h+d_h^R);
KV cache 只含蓝框两向量 (c^KV, k^R)。测试先于实现书写(TDD)。"""
import numpy as np

from mla import (
    absorb_weights,
    absorption_score_three_orders,
    make_mla_weights,
    mla_absorbed_forward,
    mla_naive_forward,
    noncommutativity_and_rope_coupling,
    rms_norm,
    rope_rotate,
)


def _toy_dims():
    # 缩小教具维度(保持结构:n_h 个头、d_h 维、d_c 潜、d_h^R 解耦 RoPE)
    return dict(d=8, n_h=2, d_h=4, d_c=6, d_h_R=2)


def _toy_inputs(T=4, seed=20):
    rng = np.random.default_rng(seed)
    dims = _toy_dims()
    h = rng.standard_normal((T, dims["d"]))
    positions = np.arange(T)
    return h, positions


def _reference_mla_naive_by_appc_loops(h, positions, w):
    """Appendix C Eq.(37)-(47) 逐式循环直译(含 §3.1.2 潜向量 RMSNorm)。

    (37) c^Q=W^DQ h →(42) q^C=W^UQ c^Q;(38) c^KV=W^DKV h →(43)(44)
    k^C/v^C=W^UK/W^UV c^KV;(39) q^R=RoPE(W^QR c^Q) 逐头、(40) k^R=RoPE(W^KR h)
    共享单份;(45)(46) q/k 拼接 [C;R];(47) 逐头 Softmax(qᵀk/√(d_h+d_h^R)) v^C
    → W^O 拼回。
    """
    T = h.shape[0]
    n_h, d_h, d_h_R = w.n_h, w.d_h, w.d_h_R
    if w.q_compressed:
        c_Q = rms_norm(h @ w.W_DQ.T)          # (37) + §3.1.2 RMSNorm
        q_in = c_Q
    else:                                       # App B.1:V2-Lite 不压 query
        c_Q = h
        q_in = h
    c_KV = rms_norm(h @ w.W_DKV.T)            # (38) + §3.1.2 RMSNorm
    q_R = rope_rotate((q_in @ w.W_QR.T).reshape(T, n_h, d_h_R), positions)  # (39)
    k_R = rope_rotate(h @ w.W_KR.T, positions)                                 # (40)
    q_C = (q_in @ w.W_UQ.T if w.q_compressed else h @ w.W_QC.T).reshape(T, n_h, d_h)  # (42)
    k_C = (c_KV @ w.W_UK.T).reshape(T, n_h, d_h)   # (43)
    v_C = (c_KV @ w.W_UV.T).reshape(T, n_h, d_h)   # (44)
    scale = 1.0 / np.sqrt(d_h + d_h_R)             # (47) 分母 √(d_h+d_h^R)
    u = np.zeros((T, w.d))
    for t in range(T):
        o_heads = []
        for i in range(n_h):
            q_ti = np.concatenate([q_C[t, i], q_R[t, i]])       # (45)
            scores = []
            for j in range(t + 1):
                k_ji = np.concatenate([k_C[j, i], k_R[j]])       # (46) k^R 共享
                scores.append(q_ti @ k_ji * scale)
            scores = np.array(scores)
            p = np.exp(scores - scores.max())
            p /= p.sum()
            o_heads.append(sum(p[j] * v_C[j, i] for j in range(t + 1)))
        u[t] = w.W_O @ np.concatenate(o_heads)
    return u, (c_KV, k_R)


# ── §3.1.2 RMSNorm ──


def test_rms_norm_gives_unit_rms_per_row():
    x = np.array([[3.0, -4.0], [1.0, 1.0], [0.0, 0.0]])
    out = rms_norm(x, eps=1e-6)
    rms = np.sqrt((out ** 2).mean(axis=-1))
    np.testing.assert_allclose(rms[:2], 1.0, atol=1e-5)   # 单位 RMS
    np.testing.assert_allclose(out[2], 0.0, atol=1e-3)    # 全零行 → ~0(eps 护栏)


# ── §2.1.3 RoPE 算子 ──


def test_rope_identity_at_zero_position_and_norm_preserved():
    rng = np.random.default_rng(21)
    x = rng.standard_normal((3, 4))
    np.testing.assert_allclose(rope_rotate(x, np.zeros(3, dtype=int)), x, atol=1e-15)
    p = np.array([0, 7, 1000])
    xr = rope_rotate(x, p)
    np.testing.assert_allclose(
        np.linalg.norm(xr, axis=-1), np.linalg.norm(x, axis=-1), rtol=1e-12
    )


def test_rope_relative_position_invariance():
    # RoPE 的本命性质(论文 §2.1.3 "RoPE is position-sensitive" 的另一面):
    # 同一位置旋转后的内积 == 原内积(旋转正交)——q/k 同旋不改变分数绝对部分,
    # 只在相对位置差上编码信息。
    rng = np.random.default_rng(22)
    q = rng.standard_normal(6)
    k = rng.standard_normal(6)
    p = 123
    qr = rope_rotate(q[None, :], np.array([p]))[0]
    kr = rope_rotate(k[None, :], np.array([p]))[0]
    np.testing.assert_allclose(qr @ kr, q @ k, rtol=1e-12)


def test_rope_norm_and_inner_product_statistical_sweep():
    # 分布保持类统计检验(固定种子、宽松阈值防 flaky):Eq.(14)-(15) 的
    # RoPE(·) 是正交旋转——① 范数保持:任意位置(含大位置)旋转不改变向量
    # 长度;② 同位内积不变:q/k 同位同旋后点积与旋转前逐对相等。float64 下
    # 实际偏差 ~1e-15,阈值放宽到 1e-9 只挡真错。
    rng = np.random.default_rng(42)
    X = rng.standard_normal((512, 8))
    pos = rng.integers(0, 8192, size=512)
    Xr = rope_rotate(X, pos)
    np.testing.assert_allclose(
        np.linalg.norm(Xr, axis=-1), np.linalg.norm(X, axis=-1), rtol=1e-9
    )
    q = rng.standard_normal((256, 8))
    k = rng.standard_normal((256, 8))
    p = rng.integers(1, 4096, size=256)
    qr = rope_rotate(q, p)
    kr = rope_rotate(k, p)
    np.testing.assert_allclose(
        (qr * kr).sum(axis=-1), (q * k).sum(axis=-1), rtol=1e-9, atol=1e-9
    )


def test_noncommutativity_and_rope_coupling_breaks_absorption():
    # §2.1.3 原论证的 2×2 手算(dossier m08 worked example):
    # ① AB≠BA——"matrix multiplication does not obey a commutative law" 直感;
    # ② 灾难版:RoPE 若作用于 k^C,折叠矩阵从固定 B=(W^UK)^T W^UQ 变成
    # F(m,t)=(W^UK)^T R(m)^T R(t) W^UQ——位置对一变就变,无单一离线形态
    # (吸收失效);R(0)=I 时才退化回固定 B。故位置信息须解耦成旁路 k^R。
    W_UQ = np.array([[1.0, 2.0], [3.0, 4.0]])
    W_UK = np.array([[5.0, 6.0], [7.0, 8.0]])
    out = noncommutativity_and_rope_coupling(W_UQ, W_UK)
    np.testing.assert_allclose(
        out["AB"], np.array([[19.0, 22.0], [43.0, 50.0]])
    )                                    # AB=[1·5+2·7, 1·6+2·8; 3·5+4·7, 3·6+4·8]
    np.testing.assert_allclose(
        out["BA"], np.array([[23.0, 34.0], [31.0, 46.0]])
    )                                    # BA≠AB(逐位都不同)
    assert out["not_commutative"]
    np.testing.assert_allclose(out["F_00"], out["B_fixed"])   # R(0)=I 退化
    assert out["fold_is_position_dependent"]                  # 位置一变 → 矩阵变


# ── App C naive 展开形态 ──


def test_mla_naive_forward_matches_appc_loop_reference():
    dims = _toy_dims()
    w = make_mla_weights(**dims, q_lora_rank=5, seed=23)
    h, positions = _toy_inputs(seed=24)
    u, (c_KV, k_R) = mla_naive_forward(h, positions, w)
    ref_u, (ref_c, ref_kr) = _reference_mla_naive_by_appc_loops(h, positions, w)
    np.testing.assert_allclose(u, ref_u, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(c_KV, ref_c, rtol=1e-12)
    np.testing.assert_allclose(k_R, ref_kr, rtol=1e-12)


def test_mla_scale_denominator_is_sqrt_dh_plus_dhR():
    # Eq.(18):分母是 √(d_h+d_h^R),不是 MHA 的 √d_h(拼接后 q/k 维度变了)。
    dims = _toy_dims()
    w = make_mla_weights(**dims, q_lora_rank=5, seed=25)
    h, positions = _toy_inputs(seed=26)
    trace = {}
    mla_naive_forward(h, positions, w, trace=trace)
    assert trace["scale"] == 1.0 / np.sqrt(dims["d_h"] + dims["d_h_R"])
    assert trace["q"].shape[-1] == dims["d_h"] + dims["d_h_R"]   # (45) 拼接宽


def test_cache_is_exactly_the_two_blue_boxed_vectors():
    # App C Eq.(41)/V3-1/V3-3 蓝框:需缓存的只有 c^KV 与 k^R 两个向量;
    # 每 token 宽 d_c+d_h^R(§2.1.3 尾句 "(d_c+d_h^R)l elements")。
    dims = _toy_dims()
    w = make_mla_weights(**dims, q_lora_rank=5, seed=27)
    h, positions = _toy_inputs(seed=28)
    _, (c_KV, k_R) = mla_naive_forward(h, positions, w)
    assert c_KV.shape == (h.shape[0], dims["d_c"])
    assert k_R.shape == (h.shape[0], dims["d_h_R"])   # 共享单份,不分头
    per_token = c_KV[0].size + k_R[0].size
    assert per_token == dims["d_c"] + dims["d_h_R"]


def test_decode_from_cached_latents_equals_full_forward():
    # §2.1.2「During inference, MLA only needs to cache c^KV」+ §2.1.3「the
    # decoupled key should also be cached」+ App C Eq.(41)/V3-1/V3-3 蓝框——
    # 行为级验证(蓝框充分性):逐 token 解码时,新 token 的输出只需
    # (a) 它自己的 h_t、(b) 前缀缓存 (c^KV, k^R) 即可精确重算;(43)/(44)
    # 从缓存逐头恢复 k^C/v^C、k^R 缓存原样进拼接——前缀的 h/W^DKV/W^KR
    # 全程不需要。解码步 j<=t 天然成立,无因果掩码。
    dims = _toy_dims()
    w = make_mla_weights(**dims, q_lora_rank=5, seed=40)
    T = 6
    rng = np.random.default_rng(41)
    h = rng.standard_normal((T, dims["d"]))
    positions = np.arange(T)
    u_full, (c_KV, k_R) = mla_naive_forward(h, positions, w)

    def _softmax(v):
        e = np.exp(v - v.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    for t in range(T):
        # 新 token 自己的两个缓存向量只由 h_t 决定(单 token 重算 == 整段)
        c_new = rms_norm(h[t : t + 1] @ w.W_DKV.T)                          # (38)
        kR_new = rope_rotate(h[t : t + 1] @ w.W_KR.T, positions[t : t + 1])  # (40)
        np.testing.assert_allclose(c_new[0], c_KV[t], rtol=1e-12)
        np.testing.assert_allclose(kR_new[0], k_R[t], rtol=1e-12)

        # q 侧只由 h_t 产生:(37)→(42)/(39)→(45)
        c_Q = rms_norm(h[t : t + 1] @ w.W_DQ.T)
        q_C = (c_Q @ w.W_UQ.T).reshape(1, dims["n_h"], dims["d_h"])
        q_R = rope_rotate(
            (c_Q @ w.W_QR.T).reshape(1, dims["n_h"], dims["d_h_R"]),
            positions[t : t + 1],
        )
        q_t = np.concatenate([q_C, q_R], axis=-1)                           # (45)

        # k/v 侧:前缀全部从缓存恢复——(43)/(44) 的上投影吃缓存 c^KV
        n = t + 1
        prefix_c = np.concatenate([c_KV[:t], c_new], axis=0)
        prefix_kR = np.concatenate([k_R[:t], kR_new], axis=0)
        k_C = (prefix_c @ w.W_UK.T).reshape(n, dims["n_h"], dims["d_h"])     # (43)
        v_C = (prefix_c @ w.W_UV.T).reshape(n, dims["n_h"], dims["d_h"])     # (44)
        k_t = np.concatenate(
            [
                k_C,
                np.broadcast_to(
                    prefix_kR[:, None, :], (n, dims["n_h"], dims["d_h_R"])
                ),
            ],
            axis=-1,
        )                                                                    # (46)
        scale = 1.0 / np.sqrt(dims["d_h"] + dims["d_h_R"])
        scores = np.einsum("tid,sid->its", q_t, k_t) * scale
        probs = _softmax(scores)
        o = np.einsum("its,sid->tid", probs, v_C)                            # (47)
        u_t = o.reshape(1, dims["n_h"] * dims["d_h"]) @ w.W_O.T
        np.testing.assert_allclose(u_t[0], u_full[t], rtol=1e-10, atol=1e-12)


def test_rope_decoupling_position_shift_only_moves_rope_segment():
    # §2.1.3 解耦结构:位置信息只走 R 段——整体平移 positions 后 C 段
    # (q^C/k^C/c^KV,可吸收侧)逐位不变,R 段(q^R/k^R)被旋转。
    dims = _toy_dims()
    w = make_mla_weights(**dims, q_lora_rank=5, seed=29)
    h, positions = _toy_inputs(seed=30)
    tr1, tr2 = {}, {}
    mla_naive_forward(h, positions, w, trace=tr1)
    mla_naive_forward(h, positions + 100, w, trace=tr2)
    np.testing.assert_array_equal(tr1["q_C"], tr2["q_C"])   # (42) 不旋
    np.testing.assert_array_equal(tr1["k_C"], tr2["k_C"])   # (43) 不旋
    np.testing.assert_array_equal(tr1["c_KV"], tr2["c_KV"])  # (38) 缓存侧不旋
    assert not np.allclose(tr1["q_R"], tr2["q_R"])          # (39) 才旋
    assert not np.allclose(tr1["k_R"], tr2["k_R"])          # (40) 才旋


# ── App C 吸收形态(结合律,离线一次) ──


def test_absorption_score_three_orders_hand_example():
    # dossier m07 worked example:2×2 手算——三种结合次序逐位相等。
    # q=W^UQ c^Q=[-1.5,-2.5],k=W^UK c^KV=[28,38] → s1=q·k=-137;
    # 吸收:q̂=(W^UK)^T q=[-25,-29] → s2=q̂·c^KV=-137;
    # 折叠:B=(W^UK)^T W^UQ 离线一次 → s3=(B c^Q)·c^KV=-137。
    W_UQ = np.array([[1.0, 2.0], [3.0, 4.0]])
    W_UK = np.array([[5.0, 6.0], [7.0, 8.0]])
    c_Q = np.array([0.5, -1.0])
    c_KV = np.array([2.0, 3.0])
    s1, s2, s3 = absorption_score_three_orders(W_UQ, W_UK, c_Q, c_KV)
    assert s1 == -137.0 and s2 == -137.0 and s3 == -137.0


def test_absorb_weights_is_offline_and_shapes_follow_paper():
    # App C:"related to only model parameters, it can be completed offline at
    # once"——吸收只动参数:逐头 B_i=(W^UK_i)^T W^UQ_i ∈ R^{d_c×d_c'};
    # W^UV 吸进 W^O:W_O_absorbed ∈ R^{d×(n_h·d_c)}。
    dims = _toy_dims()
    w = make_mla_weights(**dims, q_lora_rank=5, seed=31)
    wa = absorb_weights(w)
    assert wa.B.shape == (dims["n_h"], dims["d_c"], 5)
    assert wa.W_O_absorbed.shape == (dims["d"], dims["n_h"] * dims["d_c"])


def test_absorbed_forward_equals_naive_exactly():
    # App C 吸收论断:naive(从 c^KV 恢复 k^C/v^C)与吸收(c^KV 直接当单头
    # K/V 参与 attention)由结合律数学等价——两条完整前向逐位恒等。
    dims = _toy_dims()
    w = make_mla_weights(**dims, q_lora_rank=5, seed=32)
    wa = absorb_weights(w)
    h, positions = _toy_inputs(T=6, seed=33)
    u_naive, (c1, kr1) = mla_naive_forward(h, positions, w)
    u_abs, (c2, kr2) = mla_absorbed_forward(h, positions, wa)
    np.testing.assert_allclose(u_abs, u_naive, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(c1, c2)      # 缓存内容两条路完全一致
    np.testing.assert_array_equal(kr1, kr2)


def test_absorbed_forward_keeps_sqrt_dh_plus_dhR_scale():
    # dossier m07 注:吸收不改变缩放——分数对 √(d_h+d_h^R) 的标量乘与
    # 结合次序交换,吸收后分母仍是 √(d_h+d_h^R) 而非 √(d_c+d_h^R)。
    dims = _toy_dims()
    w = make_mla_weights(**dims, q_lora_rank=5, seed=34)
    wa = absorb_weights(w)
    h, positions = _toy_inputs(seed=35)
    trace = {}
    mla_absorbed_forward(h, positions, wa, trace=trace)
    assert trace["scale"] == 1.0 / np.sqrt(dims["d_h"] + dims["d_h_R"])
    # 潜空间 query 是 d_c 维、与缓存 c^KV 直接点积(MQA 形状、潜向量内容)
    assert trace["q_latent"].shape == (h.shape[0], dims["n_h"], dims["d_c"])
    assert trace["scores"].shape == (dims["n_h"], h.shape[0], h.shape[0])


# ── App B.1:DeepSeek-V2-Lite 不压 query 的退化分支 ──


def test_v2_lite_no_query_compression_branch_runs_and_absorbs():
    # App B.1:"it does not compress the queries"——q 侧无 c^Q 低秩,
    # q^C/q^R 直接从 h 投;query 压缩是独立旋钮,吸收恒等式照样成立。
    dims = _toy_dims()
    w = make_mla_weights(**dims, q_lora_rank=None, seed=36)
    assert not w.q_compressed
    h, positions = _toy_inputs(seed=37)
    u_naive, _ = mla_naive_forward(h, positions, w)
    ref_u, _ = _reference_mla_naive_by_appc_loops(h, positions, w)
    np.testing.assert_allclose(u_naive, ref_u, rtol=1e-12, atol=1e-14)
    wa = absorb_weights(w)
    u_abs, _ = mla_absorbed_forward(h, positions, wa)
    np.testing.assert_allclose(u_abs, u_naive, rtol=1e-10, atol=1e-12)


def test_v2_lite_branch_shapes_match_paper():
    # App B.1 退化分支的形状账:W_DQ/W_UQ 缺席,W^QR/W^QC 直接吃 h。
    dims = _toy_dims()
    w = make_mla_weights(**dims, q_lora_rank=None, seed=38)
    assert w.W_DQ is None and w.W_UQ is None
    assert w.W_QC.shape == (dims["n_h"] * dims["d_h"], dims["d"])
    assert w.W_QR.shape == (dims["n_h"] * dims["d_h_R"], dims["d"])
    # 压缩分支则相反(§2.1.2 Eq.(12)-(13) 的 d_c' 记号)
    w_full = make_mla_weights(**dims, q_lora_rank=5, seed=39)
    assert w_full.W_DQ.shape == (5, dims["d"])
    assert w_full.W_UQ.shape == (dims["n_h"] * dims["d_h"], 5)
    assert w_full.W_QR.shape == (dims["n_h"] * dims["d_h_R"], 5)
