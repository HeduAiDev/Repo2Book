"""论文**性质**测试（tester 独立设计）——测的是论文断言的可观察后果，不是实现自洽。

与 `test_*.py`（implementer 的 TDD 断言：逐式数值、逐函数形状）分工不同：本文件按**性质类**
组织，每类挑一条论文原话落成可判定的判据，覆盖 dossier mechanisms m01-m20 里能写成性质的条目。

四类（每类开头的注释标出判据来自论文哪一句）：

- **A 分布/负载类**（统计检验：固定种子 + 宽松阈值 ⇒ 无 flaky）
  - arXiv:2106.04426 摘要：hash 路由 `requiring no routing parameters or extra terms in the
    objective function such as a load balancing loss` ⇒ 没有均衡损失也得给出**均衡的负载分布**。
  - arXiv:2606.19348 §2.1：auxiliary-loss-free 的作用是均衡 ⇒ 偏置更新必须让**负载不均这个
    目标量真的改善**。
- **B 恒等/极限类**（数值对照：论文式子在退化/闭式情形下必须给出可手算的值）
  - arXiv:2606.19348 Eq.(11)(12)：软池化是**权重和为 1 的加权和** ⇒ 每个坐标落在窗内输入的
    [min, max] 内（凸组合 ⇒ 是池化、不是生成）；Z=B=0 ⇒ 退化成算术平均。
  - arXiv:2606.19348 §2.3.3：RoPE 是旋转 ⇒ 保范数；输出侧按 −i 反旋 ⇒ 正反成对为恒等。
  - arXiv:2512.24880 §4.1 / arXiv:2606.19348 Eq.(1)(2)：Birkhoff 多面体的三条性质
    （恒等映射顶点 / 谱范数 ≤ 1 / 对乘法封闭）。
- **C 因果/可观察后果类**（把论文原话直接算成 token 集合的包含关系）
  - arXiv:2606.19348 §2.3.3：`each query attends to only preceding compressed KV blocks`、
    `a query cannot access information from other tokens within its own compressed block`。
  - arXiv:2606.19348 Eq.(17)：`Top-k(I_{t,:})` 的字面含义 = 分数最大的 k 个。
- **D 优化目标类**（目标量朝论文说的方向改善）
  - arXiv:2401.06066 Eq.(12)-(14)：`L_ExpBal = α₁ Σ f_i P_i` 在均衡路由下必须**更小**。
  - arXiv:2412.19437 Eq.(24)：`L^k_MTP = CrossEntropy(P^k, t)` 朝目标走必须**单调下降**。

阈值口径：所有统计量都固定 `np.random.default_rng(seed)`（可复现），阈值取得比实测值宽裕
（实测值写在断言旁），目的是抓"机制被改坏"而不是抓抽样噪声。
"""
import numpy as np
import pytest

from compressor import (
    csa_compress,
    csa_entry_input_index_sets,
    csa_project,
)
from indexer import select_topk_compressed
from ledger import V4_FLASH_COMPRESS_RATIOS, kv_byte_account
from mhc import (
    doubly_stochastic_residual,
    is_doubly_stochastic,
    mhc_gates,
    mhc_layer_input,
    mhc_update,
    sinkhorn_knopp,
    spectral_norm,
)
from moe import (
    affinity_scores_v3,
    affinity_scores_v4,
    expert_balance_loss,
    hash_route,
    topk_with_correction_bias,
)
from mtp import mtp_depth_loss
from windowed_attention import (
    compose_attention_inputs,
    compressed_entry_count,
    rope_forward,
    rope_inverse,
    sliding_window_slice,
)

# ══════════════════════════════════════════════════════════════════════════
# A 分布/负载类
# ══════════════════════════════════════════════════════════════════════════


# PAPER: arXiv:2106.04426 摘要 —— no routing parameters / no load balancing loss
def test_hash_routing_load_is_near_uniform_without_any_balance_loss():
    """Hash Layers 的卖点：**没有均衡损失项也拿得到均衡的负载**。

    判据（统计）：均匀采样的 token id 经一张随机 hash 表派单后，各专家的负载应接近
    均匀——最大负载 / 平均负载 ≈ 1（实测 1.061），远小于学出来的路由在无约束时的塌缩
    （见 `test_auxiliary_loss_free_bias_improves_load_balance` 的对照：3.47）。
    阈值 1.25 是宽松值：只抓"负载塌到少数专家"这种机制级故障。

    走的是实现的派单函数 `hash_route`（选择 = 一次 gather，scores 只用来定权重）。
    """
    rng = np.random.default_rng(210604426)  # 固定种子：本测试完全可复现
    vocab, n_experts, topk, n_tokens = 4096, 8, 2, 20000
    table = rng.integers(0, n_experts, size=(vocab, topk))
    tokens = rng.integers(0, vocab, size=n_tokens)
    scores = affinity_scores_v4(rng.normal(size=(n_tokens, n_experts)))

    expert_ids, weights = hash_route(table, tokens, scores)  # 派单：一次 gather，无路由参数
    np.testing.assert_array_equal(expert_ids, table[tokens])
    # 权重从**无偏的 gate 亲和分**现算（hash 只管"选谁"）：逐位对照 gather+归一
    rows = np.arange(n_tokens)[:, None]
    gathered = scores[rows, expert_ids]
    np.testing.assert_allclose(weights, gathered / gathered.sum(axis=1, keepdims=True), atol=1e-12)

    loads = np.bincount(expert_ids.ravel(), minlength=n_experts).astype(float)
    assert loads.sum() == n_tokens * topk
    assert loads.max() / loads.mean() < 1.25  # 实测 1.061
    assert loads.min() / loads.mean() > 0.75  # 没有专家饿死（实测 0.929）


# PAPER: arXiv:2106.04426 摘要 —— balanced and random hashes work best（random hash 的负载界）
def test_hash_routing_load_stays_bounded_under_skewed_token_frequencies():
    """真实语料的 token 频率是重尾的（Zipf）。hash 把「token id」映射到专家，只要表不是与
    频率耦合的，负载不均就有界——这正是"不需要均衡损失"的统计依据。

    判据（统计，宽松）：Zipf 采样下 max/mean < 2.0（实测 1.25）。
    """
    rng = np.random.default_rng(210604427)
    vocab, n_experts, topk, n_tokens = 4096, 8, 2, 20000
    table = rng.integers(0, n_experts, size=(vocab, topk))
    zipf = 1.0 / np.arange(1, vocab + 1)
    zipf = zipf / zipf.sum()
    tokens = rng.choice(vocab, size=n_tokens, p=zipf)
    scores = affinity_scores_v4(rng.normal(size=(n_tokens, n_experts)))

    expert_ids, _ = hash_route(table, tokens, scores)
    loads = np.bincount(expert_ids.ravel(), minlength=n_experts).astype(float)
    assert loads.max() / loads.mean() < 2.0  # 实测 1.25：重尾频率也没有让某个专家塌掉


# PAPER: arXiv:2606.19348 §2.1 —— auxiliary-loss-free strategy（augmented by a slight sequence-wise balance loss）
def test_auxiliary_loss_free_bias_improves_load_balance():
    """优化类：auxiliary-loss-free 的**目标量**是负载均衡，偏置更新必须让它变好。

    构造：8 个专家、挑 2 个，专家 0 的 gate logits 系统性 +2（模拟"某专家被偏爱"的塌缩）。
    无偏时 max/mean = 3.47（实测）——因为 `topk_with_correction_bias` 的选择用
    `scores + bias`，给缺载专家调大 bias 就等价于"把热度从过载专家搬走"。
    做 20 轮「缺载 ⇒ 调大、过载 ⇒ 调小」的更新后，max/mean = 1.00（实测）。
    """
    rng = np.random.default_rng(20260916)
    n_tokens, n_experts, k = 3000, 8, 2
    logits = rng.normal(size=(n_tokens, n_experts))
    logits[:, 0] += 2.0
    scores = affinity_scores_v4(logits)

    def loads(bias):
        out = np.zeros(n_experts)
        for t in range(n_tokens):
            idx, _ = topk_with_correction_bias(scores[t], bias, k, renormalize=False)
            out[idx] += 1
        return out

    def imbalance(l):
        return l.max() / l.mean()

    before = imbalance(loads(np.zeros(n_experts)))
    assert before > 3.0  # 触发前置条件：确实塌了（实测 3.47）

    bias = np.zeros(n_experts)
    for _ in range(20):
        l = loads(bias)
        bias = np.clip(bias + 0.1 * (l.mean() - l) / l.mean(), -2.0, 2.0)  # 缺载 ↑ / 过载 ↓
        bias -= bias.mean()  # 整体平移不改变排序，只为可读

    after = imbalance(loads(bias))
    assert after < 1.2  # 实测 1.00
    assert after < 0.5 * before  # 目标量确有改善（不是随机波动）


# PAPER: arXiv:2606.19348 §2.1 + 本章 pin 注释 —— bias 只进 scores_for_choice、不进权重
def test_correction_bias_never_leaks_into_the_gathered_weights():
    """bias 的**唯一**职责是改选择，权重必须仍从无偏亲和分 gather（pin 注释点破：bias≈8.08
    近均匀时用带偏分数当权重会把分布压平）。

    判据（数值对照）：同一批分数，加/不加 bias 后选中的专家可能不同，但每个被选中专家的权重
    都等于「无偏分数 ÷ 选中集合的无偏分数之和」。
    """
    rng = np.random.default_rng(2606193)
    scores = affinity_scores_v4(rng.normal(size=(64, 8)))
    bias = np.array([-0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 0.9, 0.0])

    for row in scores:
        idx, w = topk_with_correction_bias(row, bias, k=3)
        want = row[idx] / row[idx].sum()  # 无偏分数 gather 后再归一
        np.testing.assert_allclose(w, want, atol=1e-12)
        # 选择确实被 bias 影响过（否则这个断言没有区分力）——至少有一行会不同
    assert any(
        set(topk_with_correction_bias(r, bias, k=3)[0].tolist())
        != set(topk_with_correction_bias(r, np.zeros(8), k=3)[0].tolist())
        for r in scores[:8]
    )


# ══════════════════════════════════════════════════════════════════════════
# B 恒等/极限类
# ══════════════════════════════════════════════════════════════════════════


def _toy_csa(seed=11, m=4, c=3, d=5, n_windows=3, zero_gates=False, zero_bias=False):
    rng = np.random.default_rng(seed)
    H = rng.normal(size=(n_windows * m, d))
    W_kv_a = rng.normal(size=(d, c))
    W_kv_b = rng.normal(size=(d, c))
    W_z_a = np.zeros((d, c)) if zero_gates else rng.normal(size=(d, c))
    W_z_b = np.zeros((d, c)) if zero_gates else rng.normal(size=(d, c))
    B_a = np.zeros((m, c)) if zero_bias else rng.normal(size=(m, c)) * 0.5
    B_b = np.zeros((m, c)) if zero_bias else rng.normal(size=(m, c)) * 0.5
    return H, W_kv_a, W_kv_b, W_z_a, W_z_b, B_a, B_b


# PAPER: arXiv:2606.19348 Eq.(11)(12) —— Softmax_row 归一 ⇒ 压缩条目是 2m 个输入的凸组合
def test_csa_compressed_entry_lies_inside_the_window_convex_hull():
    """「软池化」= 权重非负且每列和为 1 的加权和 ⇒ 每个坐标被夹在窗内 2m 个输入的同坐标
    [min, max] 之间。这条性质把"压缩"钉死为**池化**（不生成新值），是 Eq.(11)(12) 最直接的
    可观察后果——任何越界都说明 softmax 或求和写错了。"""
    H, W_kv_a, W_kv_b, W_z_a, W_z_b, B_a, B_b = _toy_csa()
    C_comp, S = csa_compress(H, W_kv_a, W_kv_b, W_z_a, W_z_b, B_a, B_b, m=4)
    C_a, C_b, Z_a, Z_b = csa_project(H, W_kv_a, W_kv_b, W_z_a, W_z_b)

    assert np.all(S >= 0.0)
    assert np.allclose(S.sum(axis=1), 1.0)  # 每条条目、每个坐标：2m 个权重和 = 1
    for i in range(C_comp.shape[0]):
        # 窗内输入 = [当前窗的 C^a ; 前一个窗的 C^b]；i=0 的 b 半区按论文填 0
        idx_a = slice(4 * i, 4 * (i + 1))
        stack = [C_a[idx_a]]
        if i > 0:
            stack.append(C_b[slice(4 * (i - 1), 4 * i)])
        else:
            stack.append(np.zeros_like(C_a[idx_a]))
        C_stack = np.vstack(stack)
        lo, hi = C_stack.min(axis=0), C_stack.max(axis=0)
        assert np.all(C_comp[i] >= lo - 1e-12) and np.all(C_comp[i] <= hi + 1e-12)
        # 且它是**逐元素**加权和（不是"整条挑一个"的硬选择）
        np.testing.assert_allclose(C_comp[i], np.sum(S[i] * C_stack, axis=0), atol=1e-12)


# PAPER: arXiv:2606.19348 Eq.(11)(12) —— softmax 在等值输入上退化为均匀权重
def test_csa_with_zero_gates_and_bias_reduces_to_the_window_average():
    """极限恒等：`Z ≡ 0` 且 `B ≡ 0` ⇒ 2m 个元素全相等 ⇒ softmax 输出均匀 `1/(2m)`
    ⇒ 压缩条目 = 窗内 2m 个 C 的**算术平均**（可手算的闭式）。

    i=0 是论文唯一的特例（b 半区 −∞/0）⇒ 它的权重是 a 半区均匀、b 半区为 0，条目 = 首个窗
    自己的平均——即"第一个窗只看自己"。
    """
    H, W_kv_a, W_kv_b, W_z_a, W_z_b, B_a, B_b = _toy_csa(zero_gates=True, zero_bias=True)
    C_comp, S = csa_compress(H, W_kv_a, W_kv_b, W_z_a, W_z_b, B_a, B_b, m=4)
    C_a, C_b, Z_a, Z_b = csa_project(H, W_kv_a, W_kv_b, W_z_a, W_z_b)

    np.testing.assert_allclose(S[1:], 1.0 / 8, atol=1e-12)  # 后续条目：2m=8 个位置均权
    for i in (1, 2):
        stack = np.vstack([C_a[4 * i : 4 * (i + 1)], C_b[4 * (i - 1) : 4 * i]])
        np.testing.assert_allclose(C_comp[i], stack.mean(axis=0), atol=1e-12)

    np.testing.assert_allclose(S[0, :4], 1.0 / 4, atol=1e-12)  # i=0：只看自己的 4 个
    np.testing.assert_allclose(S[0, 4:], 0.0, atol=1e-12)
    np.testing.assert_allclose(C_comp[0], C_a[0:4].mean(axis=0), atol=1e-12)


# PAPER: arXiv:2606.19348 §2.3.3 —— we also apply RoPE with position −i on the last 64 dimensions
def test_rope_pair_is_a_norm_preserving_rotation():
    """RoPE 是（逐对）旋转：`cos²+sin²=1` ⇒ 每对的 2-范数与整行范数都不变；输出侧的
    position −i 反旋是同一件事的逆 ⇒ 正反两次 = 恒等，而单次反旋**仍然是旋转**（保范数），
    只是把角度取反——这两条一起说明"转回来"不会引入幅度变化。"""
    rng = np.random.default_rng(2812)
    for rope_dim in (2, 6, 64):
        d = rope_dim + 4  # 前 4 维是 NoPE 段（论文：RoPE 只作用最后 64 维）
        x = rng.normal(size=(6, d))
        pos = np.array([0, 1, 3, 7, 33, 512])
        fwd = rope_forward(x, pos, rope_dim=rope_dim, theta=10000.0)
        inv = rope_inverse(x, pos, rope_dim=rope_dim, theta=10000.0)

        np.testing.assert_allclose(np.linalg.norm(fwd, axis=1), np.linalg.norm(x, axis=1), atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(inv, axis=1), np.linalg.norm(x, axis=1), atol=1e-12)
        # 逐对保范数（配对口径 = pin 核的 offsets^1：相邻成对）
        nope = d - rope_dim
        pairs = fwd[:, nope:].reshape(6, rope_dim // 2, 2)
        xp = x[:, nope:].reshape(6, rope_dim // 2, 2)
        np.testing.assert_allclose(
            np.linalg.norm(pairs, axis=-1), np.linalg.norm(xp, axis=-1), atol=1e-12
        )
        # 正反成对 = 恒等；单次反旋仍改变数值（不是 no-op）
        np.testing.assert_allclose(
            rope_inverse(fwd, pos, rope_dim=rope_dim, theta=10000.0), x, atol=1e-10
        )
        assert not np.allclose(inv, x)


# PAPER: arXiv:2606.19348 Eq.(1)(2) + arXiv:2512.24880 §4.1 —— 恒等映射是 Birkhoff 多面体的顶点
def test_mhc_identity_vertex_restores_the_plain_residual_stream():
    """mHC 摘要：HC 的多流破坏了恒等映射性质；mHC 的药方是**约束**——而恒等矩阵本身就在
    Birkhoff 多面体里（置换矩阵是该多面体的顶点）。取 `B = I`、`C = 1`，Eq.(1) 必须逐字退化
    成普通残差 `X + F(A X)`：多流框架没有替掉恒等映射，只是把它变成了集合里的一个点。"""
    rng = np.random.default_rng(2606193)
    T, hc, d = 4, 3, 5
    X = rng.normal(size=(T, hc, d))
    A = np.array([0.2, 0.3, 0.5])
    B = np.eye(hc)
    C = np.ones((hc, 1))
    assert is_doubly_stochastic(B)  # 顶点 ∈ Birkhoff 多面体

    sub_in = mhc_layer_input(A, X)  # A_l X_l
    sub_out = rng.normal(size=(T, d))  # 假想子层输出 F(A_l X_l)
    X_next = mhc_update(X, B, C, sub_out)

    for j in range(hc):  # 每条流都拿到同一个子层输出（C=1 ⇒ 写回不打折）
        np.testing.assert_allclose(X_next[:, j, :], X[:, j, :] + sub_out, atol=1e-12)
    # 且子层读入确实是流间聚合（A 的加权和）
    np.testing.assert_allclose(sub_in, (A[None, :, None] * X).sum(axis=1), atol=1e-12)

    # F = 0 时残差流原样穿过（多流版恒等映射）
    np.testing.assert_allclose(mhc_update(X, B, C, np.zeros((T, d))), X, atol=1e-12)


# PAPER: arXiv:2606.19348 Eq.(1)(2) —— X_{l+1} = B_l X_l + C_l F_l(A_l X_l) 的**第一项**就是流间混合
def test_mhc_update_mixes_streams_and_counts_the_per_stream_write_back():
    """Eq.(1) 的两项各有各的可判定后果，且都只有在 B ≠ I / C 逐流不等时才看得出来：

    - **B X 是流间混合**：取 B 为置换矩阵（Birkhoff 多面体的顶点，双随机的最简非恒等形态）
      ⇒ 输出第 j 条流必须等于输入第 σ(j) 条流 + 写回项（"哪条流从哪来"被整条换掉）；
    - **C 是逐流的写回系数**：`C_l` 形状 (n_hc, 1)，第 j 条流配第 j 个系数——取
      `C = [1.5, 0.5, 0.75]`，任何"系数串味/整体共用"的写法都会错位；
    - **一般双随机 B**：逐流对照 `Σ_i B[j,i]·X[:,i] + C_j·f`；且 B ≠ I 时结果与"不混合"必须
      不同（这条把"多流残差"与"n_hc 条互不相干的普通残差"区分开）；
    - **非扩张**：纯混合那一步 `‖B X‖ ≤ ‖X‖`（§4.1 的谱范数 ≤ 1），混合不放大信号。
    """
    rng = np.random.default_rng(2606_1)
    hc, d, T = 3, 4, 5
    X = rng.normal(size=(T, hc, d))
    f = rng.normal(size=(T, d))
    C = np.array([[1.5], [0.5], [0.75]])  # 逐流不同的写回系数

    # (a) B = 置换：输出第 j 条流 = 输入第 σ(j) 条 + C_j·f
    perm = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    assert is_doubly_stochastic(perm)  # 置换矩阵 ∈ Birkhoff 多面体（顶点）
    X_perm = mhc_update(X, perm, C, f)
    for j in range(hc):
        src = int(np.argmax(perm[j]))
        np.testing.assert_allclose(X_perm[:, j, :], X[:, src, :] + C[j, 0] * f, atol=1e-12)

    # (b) 一般双随机 B（2 条流）：逐流对照 Σ_i B[j,i]·X[:,i] + C_j·f
    B2 = np.array([[0.7, 0.3], [0.3, 0.7]])
    X2 = rng.normal(size=(T, 2, d))
    C2 = np.array([[1.25], [0.4]])
    X2_next = mhc_update(X2, B2, C2, f)
    for j in range(2):
        np.testing.assert_allclose(
            X2_next[:, j, :], B2[j, 0] * X2[:, 0, :] + B2[j, 1] * X2[:, 1, :] + C2[j, 0] * f,
            atol=1e-12,
        )
    # B ≠ I ⇒ 与"不做流间混合"的结果必须不同（否则多流残差与 n_hc 条独立残差没有区别）
    assert not np.allclose(X2_next, X2 + C2[:, 0][None, :, None] * f[:, None, :])
    # 流确实被耦合：动第 0 条流 ⇒ 第 1 条流的下一步也动
    X2b = X2.copy()
    X2b[:, 0, :] += 1.0
    assert not np.allclose(mhc_update(X2b, B2, C2, f)[:, 1, :], X2_next[:, 1, :])

    # (c) 纯混合那一步非扩张（§4.1 谱范数 ≤ 1 在 Eq.(1) 第一项上的形态）
    v = np.linalg.svd(B2)[2][0]  # 对准最大奇异方向，别让随机方向躲开放大
    Xv = np.zeros((1, 2, 1))
    Xv[0, :, 0] = v
    mixed = doubly_stochastic_residual(B2, Xv)
    assert np.linalg.norm(mixed) <= np.linalg.norm(Xv) + 1e-12


# PAPER: arXiv:2606.19348 Eq.(6)(7) —— A_l = σ(Ã_l)、C_l = 2σ(C̃_l) 的**系数本身**
def test_mhc_gate_coefficients_are_exactly_sigma_and_two_sigma():
    """Eq.(6)(7) 的系数是式子的一部分，不是"落在某个区间"就够了：

    - `C_l = 2σ(C̃_l)`：写成 `σ(C̃_l)`（漏掉因子 2）值域仍在 (0,2) 内，只有**逐元素对照
      2σ(C̃)** 才抓得住——所以这里既做精确对照，也断言"正的 C̃ 能推出 C > 1"（σ 永远给不出 >1）；
    - `A_l = σ(Ã_l)`：本实现按两侧代码的数值保护在 σ 之后加 `eps`（pin 的 `hc_pre_eps`、
      官方 kernel 的 `+ eps`，见实现 docstring）——故精确式是 `σ(Ã)+eps`；**post 侧不加 eps**
      （官方 kernel.py:L393-L394 是裸 2σ）。
    """
    rng = np.random.default_rng(2606_7)
    A_t = rng.normal(size=(6, 3)) * 2
    B_t = rng.normal(size=(6, 3, 3)) * 2
    C_t = rng.normal(size=(6, 3, 1)) * 2
    A, _, C = mhc_gates(A_t, B_t, C_t, sinkhorn_iters=20, eps=1e-6)

    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    np.testing.assert_allclose(C[:, :, 0], 2.0 * sig(C_t[:, :, 0]), atol=1e-12)
    np.testing.assert_allclose(A, sig(A_t) + 1e-6, atol=1e-12)

    # 因子 2 的可观察后果：正的 C̃ 能给出 C > 1，σ 永远给不出
    C_pos = np.full((1, 1, 1), 2.0)
    _, _, C_big = mhc_gates(np.zeros((1, 1)), np.zeros((1, 1, 1)), C_pos, sinkhorn_iters=1)
    assert float(C_big.max()) > 1.0  # 2σ(2.0) = 1.76；σ(2.0) = 0.88
    assert np.all(C > 0) and np.all(C < 2)


# PAPER: arXiv:2512.24880 §4.1 + Eq.(6)(9) —— 范数 ≤ 1、恢复恒等映射、集合对乘法封闭
def test_birkhoff_properties_survive_deep_stacking():
    """§4.1 三条性质在**深堆叠**下的形态：

    (a) 真双随机矩阵（收敛后的 Sinkhorn 投影）连乘仍是双随机、谱范数 ≤ 1（严格判据）；
    (b) mHC 在生产路径上用 t_max=20 的**近似**投影——范数仍 ≤ 1（实测 0.999999），
        但"封闭"只能到 20 轮的精度（实测 50 次连乘偏差 5e-5）：这正是论文
        `as a practical value` 的含义，不能当成精确投影。
    (c) 非双随机对照（行和为 1、列和不为 1）在深堆叠里放大信号——约束不是装饰。
    """
    rng = np.random.default_rng(2488_1)

    # (a) 收敛后的投影：严格闭包 + 范数界
    E1 = sinkhorn_knopp(rng.normal(size=(4, 4)) * 2, iters=3000, start="exp", eps=1e-9)
    E2 = sinkhorn_knopp(rng.normal(size=(4, 4)) * 2, iters=3000, start="exp", eps=1e-9)
    assert is_doubly_stochastic(E1, atol=1e-9)
    product = E1 @ E2
    assert is_doubly_stochastic(product, atol=1e-9)  # 乘法封闭
    assert spectral_norm(product) <= 1.0 + 1e-9

    # (b) mHC 生产路径（t_max=20）的近似投影
    A_t = rng.normal(size=(2, 4)) * 0.5
    B_t = rng.normal(size=(2, 4, 4)) * 0.5
    C_t = rng.normal(size=(2, 4, 1)) * 0.5
    _, B20, _ = mhc_gates(A_t, B_t, C_t, sinkhorn_iters=20)
    assert is_doubly_stochastic(B20[0], atol=1e-3)  # 20 轮不是精确投影 ⇒ 松判据
    assert spectral_norm(B20[0]) <= 1.0 + 1e-6  # 实测 0.999999：不放大
    P50 = np.linalg.matrix_power(B20[0], 50)
    assert np.abs(P50.sum(axis=1) - 1).max() < 1e-3  # 实测 5e-5：近似封闭
    assert spectral_norm(P50) <= 1.0

    # 50 层"只混合、无子层"的残差：范数不放大（信号不被流间混合放大）
    X = rng.normal(size=(2, 4, 8))
    Y = X.copy()
    for _ in range(50):
        Y = doubly_stochastic_residual(B20[0], Y)
    assert np.linalg.norm(Y) <= np.linalg.norm(X) * (1.0 + 1e-6)  # 实测 0.53（收缩）

    # (c) 反例对照：列和 ≠ 1 ⇒ 谱范数 > 1 ⇒ 深堆叠放大
    bad = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert not is_doubly_stochastic(bad)
    assert spectral_norm(np.linalg.matrix_power(bad, 50)) > 1.0


# PAPER: arXiv:2606.19348 §2.1 —— from Sigmoid(·) into Sqrt(Softplus(·))
def test_sqrt_softplus_keeps_expert_separation_where_sigmoid_saturates():
    """换激活的可观察后果：`Sigmoid` 在 |logit| 大时饱和 ⇒ 专家之间**分不开**（分数差 →
    0，路由对 logits 不再敏感）；`Sqrt(Softplus)` 不饱和 ⇒ 同样两个 logits 仍保持可观的分差。

    判据：logits = [8, 10] 处，sigmoid 分差 < 1e-3（实测 2.9e-4），sqrt(softplus) 分差 >
    0.1（实测 0.334）。这条也解释了为什么 V4 要在 auxiliary-loss-free 之外还换掉激活。
    """
    logits = np.array([8.0, 10.0])
    sig = affinity_scores_v3(logits)
    spl = affinity_scores_v4(logits)
    assert (sig[1] - sig[0]) < 1e-3  # sigmoid 饱和：几乎并排
    assert (spl[1] - spl[0]) > 0.1  # sqrt(softplus) 仍能区分
    # 严格单调（分差不为 0 的前提）
    grid = np.linspace(-6.0, 12.0, 37)
    assert np.all(np.diff(affinity_scores_v4(grid)) > 0)


# ══════════════════════════════════════════════════════════════════════════
# C 因果 / 可观察后果类
# ══════════════════════════════════════════════════════════════════════════


def _compressed_source_tokens(t, m, rule):
    """query 在位置 `t` 能通过**压缩路径**读到的原始 token 下标集合。

    候选 = `compressed_entry_count(t, m, rule)` 条已完成块；第 i 条条目按 Eq.(11)(12) 由
    a 半区（当前窗 = 第 i 段）与 b 半区（前一个窗 = 第 i−1 段）的 token 组成。
    """
    n_entries = compressed_entry_count(t, m, rule=rule)
    srcs = set()
    for a, b in csa_entry_input_index_sets(m, n_entries):
        srcs |= set(a) | set(b)
    return srcs


# PAPER: arXiv:2606.19348 §2.3.3 —— each query attends to only preceding compressed KV blocks
def test_no_query_ever_reads_a_future_token_through_the_compressed_path():
    """严格因果的可判定形态：把 query 能看到的压缩条目展开成**原始 token 集合**，
    最大下标必须 ≤ t（任何一条路径都不许看见未来）。两套因果口径（论文 `s<Floor(t/m)`、
    实现 `(t+1)//m`）都要过这一关；滑窗那一路的可见范围恰是 [max(t−n_win+1,0), t]。"""
    m, n_win, n_tokens = 4, 3, 32
    for t in range(n_tokens):
        start, count = sliding_window_slice(t, n_win)
        assert (start, start + count - 1) == (max(t - n_win + 1, 0), t)  # 含 query 自己

        for rule in ("paper", "impl"):
            srcs = _compressed_source_tokens(t, m, rule)
            if srcs:
                assert max(srcs) <= t, (t, rule, max(srcs))


# PAPER: arXiv:2606.19348 §2.3.3 —— a query cannot access information from other tokens within its own compressed block
def test_block_blindness_of_the_compressed_path_and_the_documented_one_cell_gap():
    """块内失明的可判定形态：query 自己的块是 [m·⌊t/m⌋, m·⌊t/m⌋+m−1]。

    - 论文口径（`s < Floor(t/m)`）：压缩路径的 token 全部 **< 自己块的起点** ⇒ 同块内其它
      token 一条都读不到（原话的字面后果，块内可见性只由滑窗提供）。
    - 实现口径（`(t+1)//m`，pin 与官方参考实现同款）：整体仍严格因果，只在**块尾 token** 上多
      一格——它能读到自己所在的那条压缩块（块尾正是触发压缩的那个 token，块内 ≤ t 的部分）。
      这一格是 dossier m06 记录的两口径之差，本测试把它钉住（正文按论文写公式、按代码讲口径）。
    """
    m, n_tokens = 4, 32
    for t in range(n_tokens):
        block_start = m * (t // m)

        paper_srcs = _compressed_source_tokens(t, m, "paper")
        if paper_srcs:
            assert max(paper_srcs) < block_start  # 绝对看不见自己块内任何 token

        impl_srcs = _compressed_source_tokens(t, m, "impl")
        if impl_srcs:
            assert max(impl_srcs) <= t  # 仍严格因果
            if (t + 1) % m == 0:
                assert max(impl_srcs) >= block_start  # 块尾：多这一格
            else:
                assert max(impl_srcs) < block_start  # 非块尾：与论文口径一致

    # 两口径的差恰好只出现在块尾（逐 t 核对）
    gaps = [
        t
        for t in range(n_tokens)
        if compressed_entry_count(t, m, "impl") != compressed_entry_count(t, m, "paper")
    ]
    assert gaps == [m - 1 + k * m for k in range(n_tokens // m)][: len(gaps)]
    assert all((t + 1) % m == 0 for t in gaps)


# PAPER: arXiv:2606.19348 Eq.(17) —— C^SprsComp_t = {C^Comp_s | I_{t,s} ∈ Top-k(I_{t,:})}
def test_indexer_topk_is_literally_the_k_largest_scores():
    """Eq.(17) 的定义式：选中的就是分数最大的 k 个（有序判据：任何被选中的块的分数都 ≥
    任何落选的块），且候选不足 topk 时余位填 −1 哨兵。随机分数 + 固定种子。"""
    rng = np.random.default_rng(2606_17)
    I = rng.normal(size=(1, 16))
    idx, valid = select_topk_compressed(I, topk=5)

    chosen = sorted(int(x) for x in idx[0] if x >= 0)
    assert chosen == sorted(np.argsort(-I[0])[:5].tolist())
    assert valid[0].all()
    assert min(I[0][chosen]) >= max(np.delete(I[0], chosen))  # 阈值判据

    idx_all, valid_all = select_topk_compressed(I, topk=16)  # 候选恰好够 k
    assert sorted(int(x) for x in idx_all[0]) == list(range(16))
    assert valid_all[0].all()


# PAPER: arXiv:2606.19348 Eq.(17) —— 没有合法候选的槽位以 −1 哨兵表示（不冒充"选中了块 0"）
def test_topk_slots_without_a_valid_candidate_carry_the_minus_one_sentinel():
    """Eq.(17) 的选择集只由 `Top-k(I_{t,:})` 定义；因果把候选压到 topk 以下时，**空出来的
    槽位必须能被认出来**（−1 哨兵：pin 与官方参考实现同款，见 indexer.select_topk_compressed
    的 docstring）。哨兵写成 0 会让空位冒充"选中了块 0"，下游按槽位取条目就取错块。
    """
    I = np.array([[1.0, 2.0, 3.0]])
    idx, valid = select_topk_compressed(I, topk=3, causal_threshold=1)  # 因果只放行块 0
    assert idx[0].tolist() == [0, -1, -1]
    assert valid[0].tolist() == [True, False, False]
    # 对照：候选够时一个哨兵都不出现（哨兵不是"总是填"的装饰；返回顺序按分值降序）
    idx_ok, valid_ok = select_topk_compressed(I, topk=3, causal_threshold=3)
    assert sorted(idx_ok[0].tolist()) == [0, 1, 2] and valid_ok[0].all()
    assert idx_ok[0].tolist() == [2, 1, 0]  # 分值是 [1,2,3] ⇒ 降序返回


# PAPER: arXiv:2606.19348 Eq.(17)（Top-k(I_{t,:})）+ §2.3.2（HCA "does not employ sparse attention"）
def test_sparse_cap_is_exactly_topk_while_the_non_sparse_layer_sees_every_block():
    """两种层的 KV 轴构成，正是论文给的两个极端：

    - **CSA 层**：候选 = 已完成块数，实看 = `min(k, 候选)`——Eq.(17) 的 `Top-k` 就是「至多 k
      条」；长上下文处 k 是真正的封顶（1M 场景候选 262144 ⇒ 实看 512），短上下文处等于候选数
      （论文侧没有这条，是"候选不足必然全选"的形态，pin 与官方参考实现都显式短路）；
    - **HCA 层**：`topk=None` ⇒ 整份 `C^Comp` 进注意力（Eq.(26) 的 key/value 是全份，不是子集）
      ——"压完不挑"。
    """
    for t in range(0, 64, 7):
        info = compose_attention_inputs(pos=t, m=4, n_win=3, rule="impl", topk=5)
        assert info["candidates"] == (t + 1) // 4
        assert info["n_compressed"] == min(5, info["candidates"])
        assert info["kv_len"] == info["n_swa"] + info["n_compressed"]

    long_ctx = compose_attention_inputs(pos=1_048_575, m=4, n_win=128, rule="impl", topk=512)
    assert long_ctx["candidates"] == 262_144 and long_ctx["n_compressed"] == 512  # k 封顶

    hca = compose_attention_inputs(pos=1_048_575, m=128, n_win=128, rule="impl", topk=None)
    assert hca["candidates"] == 8_192 and hca["n_compressed"] == hca["candidates"]  # 不挑


# ══════════════════════════════════════════════════════════════════════════
# D 优化目标类
# ══════════════════════════════════════════════════════════════════════════


# PAPER: arXiv:2401.06066 Eq.(12)(13)(14) —— L_ExpBal = α₁ Σ_i f_i P_i
def test_expert_balance_loss_is_lower_for_balanced_routing():
    """优化类：均衡损失是 DeepSeekMoE 训练时要最小化的目标量，它必须在**均衡**路由上给出
    更小的值——否则这个损失的方向就是错的。

    手算对照（N′=8 专家、K′=2、T 个 token、α=1）：
    - 均衡：每个专家被选的次数 = T·K′/N′ ⇒ f_i = N′/(K′T)·(T K′/N′) = 1 ∀i；P_i = 1/8
      ⇒ L = Σ 1·(1/8)·8 = 1.0；
    - 塌缩：全部 token 都选专家 {0,1}，分数也集中在它们 ⇒ f_0=f_1=N′/K′=4、其余 0、
      P_0=P_1=0.5 ⇒ L = 4·0.5 + 4·0.5 = 4.0。
    """
    n_experts, k, n_tokens = 8, 2, 96
    balanced_probs = np.full((n_tokens, n_experts), 1.0 / n_experts)
    balanced_sel = [np.array([(2 * t) % n_experts, (2 * t + 1) % n_experts]) for t in range(n_tokens)]

    skewed_probs = np.zeros((n_tokens, n_experts))
    skewed_probs[:, 0] = skewed_probs[:, 1] = 0.5
    skewed_sel = [np.array([0, 1]) for _ in range(n_tokens)]

    l_balanced = expert_balance_loss(balanced_sel, balanced_probs, alpha=1.0)
    l_skewed = expert_balance_loss(skewed_sel, skewed_probs, alpha=1.0)

    np.testing.assert_allclose(l_balanced, 1.0, atol=1e-12)  # 手算值
    np.testing.assert_allclose(l_skewed, 4.0, atol=1e-12)  # 手算值
    assert l_balanced < 0.5 * l_skewed  # 目标量朝均衡方向确实下降

    # 单调性：把塌缩程度从"全选 {0,1}"连续调到"均匀选 8 个"，损失应单调不增
    losses = []
    for frac in np.linspace(0.0, 1.0, 11):
        sel = []
        probs = np.zeros((n_tokens, n_experts))
        for t in range(n_tokens):
            if t < (1 - frac) * n_tokens:
                sel.append(np.array([0, 1]))
                probs[t, 0] = probs[t, 1] = 0.5
            else:
                i = (2 * t) % n_experts
                sel.append(np.array([i, (i + 1) % n_experts]))
                probs[t, i] = probs[t, (i + 1) % n_experts] = 0.5
        losses.append(expert_balance_loss(sel, probs, alpha=1.0))
    assert all(b <= a + 1e-12 for a, b in zip(losses, losses[1:]))  # 不增


# PAPER: arXiv:2412.19437 Eq.(24) —— L^k_MTP = CrossEntropy(P^k_{2+k:T+1}, t_{2+k:T+1})
def test_mtp_depth_loss_improves_monotonically_toward_the_target():
    """优化类：MTP 的逐深度交叉熵是训练目标，它必须**朝目标单调下降**、并在预测与目标一致
    时趋于 0（否则这个损失不可优化）。

    做法：从一个随机 logits 出发，把目标位按 a∈[0,1] 逐步加 +6（连续把预测推向正确 token），
    损失必须每步严格下降；抬到 +1e3 ⇒ 损失 ≈ 0；同样幅度抬一个**非目标**位 ⇒ 损失不降。
    （最后一条用同一 base、同一抬升幅度做对照——"打乱 target 后损失更大"那种写法不成立：
    随机打分下被打乱的标签可能恰好更容易，那不是损失的错。）
    """
    rng = np.random.default_rng(2412_19437)
    targets = np.array([2, 3, 4])
    base = rng.normal(size=(3, 7))

    losses = []
    for a in np.linspace(0.0, 1.0, 6):
        z = base.copy()
        z[np.arange(3), targets] += 6.0 * a
        losses.append(mtp_depth_loss(z, targets))
    assert all(b < a for a, b in zip(losses, losses[1:])), losses  # 严格单调下降
    assert losses[0] > losses[-1]

    exact = np.zeros((3, 7))
    exact[np.arange(3), targets] = 1e3
    assert mtp_depth_loss(exact, targets) < 1e-3  # 目标达成 ⇒ 损失 ~0

    boosted_wrong = base.copy()
    boosted_wrong[np.arange(3), np.array([0, 1, 5])] += 6.0  # 抬非目标位（都不是对应行的 target）
    assert mtp_depth_loss(boosted_wrong, targets) > mtp_depth_loss(base, targets)
    boosted_right = base.copy()
    boosted_right[np.arange(3), targets] += 6.0
    assert mtp_depth_loss(boosted_right, targets) < mtp_depth_loss(base, targets)


# PAPER: arXiv:2606.19348 摘要 + §2.3.4 —— 27%/10% 对 V3.2、~2% 对 BF16 GQA8 基线（三个数字三个分母）
def test_kv_account_exposes_both_denominators_and_pins_the_self_computed_envelope():
    """给 writer 的**护栏测试**（不是论文断言的复现）：

    - 论文断言"约 2%"的那一半在本账的四种口径下都成立（实测 1.79%–2.26%，与 `approximately
      2%` 一致）；
    - 论文"10%（对 V3.2）"的那一半，本账在**任何**口径组合下都到不了（实测 11.19%–14.13%）——
      账面口径（计不计 IndexCache / 滑窗、逐层算术平均）解释不了这一格，正文**不得**把 10%
      写成"按 config 自算得到"，只能引用论文原话并带分母。
    本条测试把这两个事实钉住：改坏账本或悄悄换分母都会红。
    """
    ratios = V4_FLASH_COMPRESS_RATIOS[:43]  # 43 层主干（第 44 项是 MTP 槽）
    gqa8, v32 = [], []
    for count_indexer in (True, False):
        for count_swa in (False, True):
            acct = kv_byte_account(
                ratios, count_indexer=count_indexer, count_swa_window=count_swa, context_len=1_000_000
            )
            assert 0.0 < acct["ratio_vs_gqa8"] < acct["ratio_vs_v32"]  # 两个分母都在账上
            gqa8.append(acct["ratio_vs_gqa8"])
            v32.append(acct["ratio_vs_v32"])

    assert max(gqa8) < 0.03 and min(gqa8) > 0.01  # 实测 1.79%–2.26% ⇒ 复现论文的"约 2%"
    assert min(v32) > 0.10  # 实测 11.19%：四种口径都够不到论文的 10%
    assert max(v32) < 0.15


if __name__ == "__main__":  # pragma: no cover - 便于单独跑
    raise SystemExit(pytest.main([__file__, "-q"]))
