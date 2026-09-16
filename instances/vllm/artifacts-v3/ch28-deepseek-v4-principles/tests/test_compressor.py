"""CSA/HCA 压缩器测试 —— 论文锚：
arXiv:2606.19348 Eq.(9)(10)(11)(12)（CSA 软池化压缩）、Eq.(20)-(23)（HCA）、§2.3.1 重叠窗 ⇒ 1/m。

TDD 说明：这些断言测的是**论文断言的可观察后果**（权重列和=1、条目数=n/m、相邻条目共享
m 个输入、i=0 的 −inf/0 边界、官方参考实现标签互换后的数值等价），不是精简版自洽。
"""
import numpy as np
import pytest

from compressor import (
    WindowAccumulator,
    csa_compress,
    csa_compress_reference_layout,
    csa_entry_input_index_sets,
    csa_entry_inputs,
    csa_project,
    hca_compress,
    n_entries,
    softpool_single_series,
    softmax_row,
)

RNG = np.random.default_rng(20260916)


def _toy_csa(m=4, c=2, scale=0.5):
    """m=4、c=2 的玩具件（dossier m03 的 worked example 口径）。"""
    H = RNG.normal(size=(3 * m, 3)).round(2)
    W_kv_a = RNG.normal(size=(3, c)).round(2)
    W_kv_b = RNG.normal(size=(3, c)).round(2)
    W_z_a = RNG.normal(size=(3, c)).round(2)
    W_z_b = RNG.normal(size=(3, c)).round(2)
    B_a = (RNG.normal(size=(m, c)) * scale).round(2)
    B_b = (RNG.normal(size=(m, c)) * scale).round(2)
    return H, W_kv_a, W_kv_b, W_z_a, W_z_b, B_a, B_b


# ── Eq.(9)(10)：四组投影 ────────────────────────────────────────────────
def test_csa_project_shapes_and_values():
    H, W_kv_a, W_kv_b, W_z_a, W_z_b, _, _ = _toy_csa()
    C_a, C_b, Z_a, Z_b = csa_project(H, W_kv_a, W_kv_b, W_z_a, W_z_b)
    assert C_a.shape == (H.shape[0], W_kv_a.shape[1])
    assert C_b.shape == (H.shape[0], W_kv_b.shape[1])
    assert Z_a.shape == (H.shape[0], W_z_a.shape[1])
    assert Z_b.shape == (H.shape[0], W_z_b.shape[1])
    # 逐字：C^a = H·W^{aKV}、Z^b = H·W^{bZ}（Eq.9/10）
    assert np.allclose(C_a, H @ W_kv_a)
    assert np.allclose(C_b, H @ W_kv_b)
    assert np.allclose(Z_a, H @ W_z_a)
    assert np.allclose(Z_b, H @ W_z_b)


# ── Eq.(11)：softmax 跨 2m 个元素（逐列归一） ───────────────────────────
def test_softmax_row_normalizes_across_stacked_axis():
    stack = RNG.normal(size=(8, 3)).round(2)
    S = softmax_row(stack, axis=0)
    assert S.shape == stack.shape
    # 「normalization across the total of 2m elements」⇒ 每一列的 2m 个权重和 = 1
    assert np.allclose(S.sum(axis=0), 1.0)
    assert np.all(S >= 0.0)


def test_csa_entry_inputs_layout_and_i0_boundary():
    H, W_kv_a, W_kv_b, W_z_a, W_z_b, B_a, B_b = _toy_csa(m=4, c=2)
    C_a, C_b, Z_a, Z_b = csa_project(H, W_kv_a, W_kv_b, W_z_a, W_z_b)

    # i=1：a 半 = 当前窗（token 4..7）、b 半 = 前一个窗（token 0..3）
    Z_stack, C_stack, info = csa_entry_inputs(C_a, C_b, Z_a, Z_b, B_a, B_b, m=4, i=1)
    assert Z_stack.shape == (8, 2)
    assert np.allclose(Z_stack[:4], Z_a[4:8] + B_a)
    assert np.allclose(Z_stack[4:], Z_b[0:4] + B_b)
    assert np.allclose(C_stack[:4], C_a[4:8])
    assert np.allclose(C_stack[4:], C_b[0:4])
    assert info["a_tokens"] == [4, 5, 6, 7] and info["b_tokens"] == [0, 1, 2, 3]

    # i=0：Z^b 填 −inf、C^b 填 0（论文唯一的特例）
    Z_stack0, C_stack0, info0 = csa_entry_inputs(C_a, C_b, Z_a, Z_b, B_a, B_b, m=4, i=0)
    assert np.all(np.isneginf(Z_stack0[4:]))
    assert np.allclose(C_stack0[4:], 0.0)
    assert info0["b_tokens"] == []


# ── Eq.(12) + §2.3.1：条目数与重叠 ───────────────────────────────────────
def test_csa_entry_count_is_sequence_over_m():
    for n, m, want in [(12, 4, 3), (1024, 128, 8), (1023, 128, 7), (8, 4, 2), (3, 4, 0)]:
        assert n_entries(n, m) == want


def test_csa_compress_produces_n_over_m_entries_with_normalized_weights():
    H, W_kv_a, W_kv_b, W_z_a, W_z_b, B_a, B_b = _toy_csa()
    C_comp, S = csa_compress(H, W_kv_a, W_kv_b, W_z_a, W_z_b, B_a, B_b, m=4)

    assert C_comp.shape == (3, 2)  # 12 个 token ⇒ 3 条（恰好 1/m）
    assert S.shape == (3, 8, 2)  # 每条条目跨 2m=8 个元素
    # 「Softmax_row ... across the total of 2m elements」：逐列和 = 1
    assert np.allclose(S.sum(axis=1), 1.0)
    # i=0 时 b 半区的权重恰为 0（−inf 填充的后果）
    assert np.allclose(S[0, 4:], 0.0)
    assert np.all(S[1:, 4:].sum(axis=1) > 0.0)  # 后续条目的 b 半区确实有权重


def test_csa_adjacent_entries_share_half_their_inputs():
    # 论文原话：the indexes of C^b used for C^Comp_i and the indexes of C^a used for
    # C^Comp_{i-1} are overlapped ⇒ 条目数 = n/m 而不是 n/(2m)
    sets = csa_entry_input_index_sets(m=4, count=3)
    assert sets[0] == [[0, 1, 2, 3], []]
    assert sets[1] == [[4, 5, 6, 7], [0, 1, 2, 3]]
    assert sets[2] == [[8, 9, 10, 11], [4, 5, 6, 7]]
    assert set(sets[1][0]) & set(sets[1][1]) == set()  # 同一条目的两半不重叠
    # 相邻条目共享：entry 1 的 b 半 = entry 0 的 a 半；entry 2 的 b 半 = entry 1 的 a 半
    assert set(sets[1][1]) == set(sets[0][0]) == {0, 1, 2, 3}
    assert set(sets[2][1]) == set(sets[1][0]) == {4, 5, 6, 7}


def test_csa_entry0_equals_single_series_pool_of_first_window():
    # i=0 的边界等价于「第一个窗只看自己」：与单序列软池化（HCA 式，m'=m）逐位相同
    H, W_kv_a, W_kv_b, W_z_a, W_z_b, B_a, B_b = _toy_csa()
    C_comp, _ = csa_compress(H, W_kv_a, W_kv_b, W_z_a, W_z_b, B_a, B_b, m=4)
    C_a, _, Z_a, _ = csa_project(H, W_kv_a, W_kv_b, W_z_a, W_z_b)
    single = softpool_single_series(C_a[0:4], Z_a[0:4], B_a, m_prime=4)
    assert np.allclose(C_comp[0], single)


def test_csa_weights_are_learned_not_uniform_average():
    # 「软池化」不是平均：权重来自可学习投影 + 位置偏置 ⇒ 与直接平均的结果不同
    H, W_kv_a, W_kv_b, W_z_a, W_z_b, B_a, B_b = _toy_csa()
    C_comp, S = csa_compress(H, W_kv_a, W_kv_b, W_z_a, W_z_b, B_a, B_b, m=4)
    C_a, C_b, _, _ = csa_project(H, W_kv_a, W_kv_b, W_z_a, W_z_b)
    # 第 2 条条目（i=1）的输入是 a 半 token 4..7 与 b 半 token 0..3；与均匀平均对照
    stacked_C = np.vstack([C_a[4:8], C_b[0:4]])
    uniform = stacked_C.mean(axis=0)
    assert not np.allclose(C_comp[1], uniform)
    assert not np.allclose(S[1], np.full_like(S[1], 1.0 / 8))


def test_csa_streaming_prior_carries_the_previous_chunk_window():
    # 跨 forward 续窗：第二个 chunk 的第一条条目，其 b 半区由上一个 chunk 的末窗补上
    # （官方 Compressor 的 kv_state/score_state 前 ratio 行 / pin 的 state_cache 就是这件事）。
    m, c = 4, 2
    H = RNG.normal(size=(2 * m, 3)).round(2)
    W_kv_a = RNG.normal(size=(3, c)).round(2)
    W_kv_b = RNG.normal(size=(3, c)).round(2)
    W_z_a = RNG.normal(size=(3, c)).round(2)
    W_z_b = RNG.normal(size=(3, c)).round(2)
    B_a = RNG.normal(size=(m, c)).round(2) * 0.5
    B_b = RNG.normal(size=(m, c)).round(2) * 0.5

    full, _ = csa_compress(H, W_kv_a, W_kv_b, W_z_a, W_z_b, B_a, B_b, m=m)
    assert full.shape[0] == 2

    first, _ = csa_compress(H[:m], W_kv_a, W_kv_b, W_z_a, W_z_b, B_a, B_b, m=m)
    _, C_b, _, Z_b = csa_project(H, W_kv_a, W_kv_b, W_z_a, W_z_b)
    second, S2 = csa_compress(
        H[m:], W_kv_a, W_kv_b, W_z_a, W_z_b, B_a, B_b, m=m,
        prior_a=(C_b[:m], Z_b[:m]),  # 上一个 chunk 末窗「贡献给下一个窗」的那半区
    )
    # 分段跑与整段跑逐位相同（第一段就是那个带 −inf 边界的首窗口）
    np.testing.assert_allclose(first[0], full[0], atol=1e-12)
    np.testing.assert_allclose(second, full[1:], atol=1e-12)
    # 有 prior 时 b 半区不再被 −inf 填充 ⇒ 8 个位置都有权重
    assert np.all(S2[0, m:].sum(axis=0) > 0.0)


def test_hca_compress_non_overlapping_windows():
    n, m_prime, c = 1024, 128, 3
    H = RNG.normal(size=(n, 4)).round(3)
    W_kv = RNG.normal(size=(4, c)).round(3)
    W_z = RNG.normal(size=(4, c)).round(3)
    B = RNG.normal(size=(m_prime, c)).round(3)

    C_comp, S = hca_compress(H, W_kv, W_z, B, m_prime=m_prime)
    assert C_comp.shape == (8, c)  # 1024 / 128 = 8 条
    assert S.shape == (8, m_prime, c)  # 每条只跨自己窗内的 m' 个元素（无重叠）
    assert np.allclose(S.sum(axis=1), 1.0)

    # 不整除时：只产完整窗的条目，余数被留下（官方 prefill 分支的 cutoff/remainder 切法）
    C_comp2, _ = hca_compress(H[:1000], W_kv, W_z, B, m_prime=m_prime)
    assert C_comp2.shape == (7, c)


def test_hca_window_width_differs_by_compress_rate():
    # 同一套机制的两档：CSA 窗宽 2m（Eq.11）、HCA 窗宽 m'（Eq.22）
    H = RNG.normal(size=(16, 3)).round(2)
    W_kv = RNG.normal(size=(3, 2)).round(2)
    W_z = RNG.normal(size=(3, 2)).round(2)
    _, S4 = hca_compress(H, W_kv, W_z, np.zeros((4, 2)), m_prime=4)
    _, S8 = hca_compress(H, W_kv, W_z, np.zeros((8, 2)), m_prime=8)
    assert S4.shape == (4, 4, 2)
    assert S8.shape == (2, 8, 2)


# ── 标签陷阱：论文 a/b 与官方参考实现两半区相反 ──────────────────────────────
def test_reference_two_half_layout_matches_paper_with_swapped_labels():
    m, c = 4, 2
    H = RNG.normal(size=(3 * m, 3)).round(2)
    kv_proj_w = RNG.normal(size=(3, 2 * c)).round(2)  # [..., :c] = 前窗、[..., c:] = 当前窗
    gate_proj_w = RNG.normal(size=(3, 2 * c)).round(2)
    position_bias = RNG.normal(size=(m, 2 * c)).round(2)

    ref, _ = csa_compress_reference_layout(H, kv_proj_w, gate_proj_w, position_bias, m=m)
    paper, _ = csa_compress(
        H,
        W_kv_a=kv_proj_w[:, c:],  # 论文的 a 半 = 当前窗 = 官方参考实现的后半区
        W_kv_b=kv_proj_w[:, :c],  # 论文的 b 半 = 前一个窗 = 官方参考实现的前半区
        W_z_a=gate_proj_w[:, c:],
        W_z_b=gate_proj_w[:, :c],
        B_a=position_bias[:, c:],
        B_b=position_bias[:, :c],
        m=m,
    )
    assert np.allclose(ref, paper)


def _official_compressor_prefill(H, kv_proj_w, gate_proj_w, ape, ratio, overlap):
    """**官方参考实现 `Compressor.forward` 的 prefill 分支字面转写**（NumPy 直译
    DeepSeek-V4-Pro inference/model.py:L325-L342，start_pos==0 且 seqlen 整除 ratio 因此
    remainder=0）：

        kv = wkv(x); score = wgate(x)
        kv    = kv.unflatten(0, (-1, ratio))
        score = score.unflatten(0, (-1, ratio)) + ape
        [overlap] kv = overlap_transform(kv, 0); score = overlap_transform(score, -inf)
        kv = (kv * score.softmax(dim=1)).sum(dim=1)

    与 `csa_compress` / `hca_compress` 是**两条独立写出来的路径**（本函数保留官方的
    unflatten + overlap_transform 算子序列与 0/−inf 填充值），用来把「本实现的逐窗循环」
    对账到「官方的一枪张量操作」上——不是把本实现的函数换个写法再比一遍。
    """
    n = H.shape[0]
    assert n % ratio == 0
    kv = H @ kv_proj_w                                     # (n, coff*c)
    score = H @ gate_proj_w                                # (n, coff*c)
    kv = kv.reshape(n // ratio, ratio, -1)
    score = score.reshape(n // ratio, ratio, -1) + ape
    if overlap:
        s, d = kv.shape[0], ape.shape[1] // 2
        new_kv = np.zeros((s, 2 * ratio, d))
        new_sc = np.full((s, 2 * ratio, d), -np.inf)
        new_kv[:, ratio:] = kv[:, :, d:]
        new_sc[:, ratio:] = score[:, :, d:]
        new_kv[1:, :ratio] = kv[:-1, :, :d]
        new_sc[1:, :ratio] = score[:-1, :, :d]
        kv, score = new_kv, new_sc
    score = score - score.max(axis=1, keepdims=True)       # softmax(dim=1)，-inf → 权重 0
    weights = np.exp(score)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return (kv * weights).sum(axis=1)


# ── 官方算子序列（一枪张量操作）与论文逐式，两条独立路径必须逐位一致 ────
def test_official_operator_sequence_and_paper_form_agree_bitwise():
    """独立对账：官方 `Compressor.forward` 的 prefill 算子序列（unflatten + overlap_transform
    一枪出）与论文 Eq.(9)-(12) 的逐窗式子（本实现的逐条循环）**逐位相同**。

    这一条覆盖了标签陷阱之外的另一层风险：**窗口与条目的对齐**。官方的 `overlap_transform`
    把"前一个窗"写进前半区时用的是**条目轴平移**（`new[1:, :ratio] = old[:-1, :, :d]`，
    对条目平移、不对窗内行平移）——如果谁把它读成窗内行平移，这里就会红。
    HCA 侧同法核一遍（ratio=128 档：overlap=False ⇒ coff=1 ⇒ 只有一个半区、无重叠）。
    """
    for ratio, overlap in ((4, True), (128, False)):
        c = 3
        H = RNG.normal(size=(3 * ratio, 4)).round(3)
        d = ratio * c
        kv_proj_w = RNG.normal(size=(4, (1 + overlap) * d)).round(3)
        gate_proj_w = RNG.normal(size=(4, (1 + overlap) * d)).round(3)
        ape = RNG.normal(size=(ratio, (1 + overlap) * d)).round(3)

        official = _official_compressor_prefill(H, kv_proj_w, gate_proj_w, ape, ratio, overlap)
        if overlap:
            paper, _ = csa_compress(
                H,
                W_kv_a=kv_proj_w[:, d:], W_kv_b=kv_proj_w[:, :d],
                W_z_a=gate_proj_w[:, d:], W_z_b=gate_proj_w[:, :d],
                B_a=ape[:, d:], B_b=ape[:, :d], m=ratio,
            )
        else:
            paper = hca_compress(H, kv_proj_w, gate_proj_w, ape, m_prime=ratio, return_weights=False)
        assert official.shape == paper.shape == (3, d)
        np.testing.assert_allclose(paper, official, atol=1e-10)

        # 同一条官方序列也要对上"官方口径"的那个对账函数
        ref, _ = csa_compress_reference_layout(H, kv_proj_w, gate_proj_w, ape, m=ratio) if overlap else (official, None)
        np.testing.assert_allclose(ref, official, atol=1e-10)


# ── 攒批状态：块尾 token 才产出条目 ─────────────────────────────────────
def test_window_accumulator_emits_one_entry_per_full_window():
    acc = WindowAccumulator(m=4, width=2)
    emitted = []
    for pos in range(6):
        kv = np.full((1, 2), float(pos))
        gate = np.zeros((1, 2))
        chunk_kv, chunk_gate, first_pos, complete = acc.push(kv, gate, pos)
        if complete:
            emitted.append((first_pos, float(chunk_kv[0, 0])))
    # 只有 (pos+1)%m==0 的 token 触发压缩：pos=3 产出 1 条，pos=4/5 还在攒
    assert emitted == [(0, 0.0)]
    assert acc.entry_count == 1
    assert acc.buffered == 2

    # 再推 6 个 → 补满两个完整的窗（pos 6,7 与 8,9）⇒ 一次调用产出 2 条
    for pos in range(6, 12):
        kv = np.full((1, 2), float(pos))
        gate = np.zeros((1, 2))
        chunk_kv, _, first_pos, complete = acc.push(kv, gate, pos)
        if complete:
            emitted.append((first_pos, float(chunk_kv[0, 0])))
    assert emitted[1][0] == 4
    assert acc.entry_count == 3
    # 条目序号 = pos_after_compress = pos // m（compressor_utils 的坐标系）
    assert acc.entry_slot(pos=11) == 2
