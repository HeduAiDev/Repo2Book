"""arXiv:2205.14135 (FlashAttention) §2.2 Algorithm 0(标准注意力,物化两张 N×N)、
§3.1 Algorithm 1(tiling + online-softmax 递推,Theorem 1 精确性)、§3.2 Theorem 2
(IO 复杂度账);arXiv:2307.08691 (FlashAttention-2) §3.1.1 Algorithm 1(循环序对调 +
未归一化 O + logsumexp L)。"""
import math

import numpy as np
import pytest

from flash_attention import (
    causal_keep_mask,
    fa_block_sizes,
    flash_attention_2_forward,
    flash_attention_forward,
    hbm_accesses_flash,
    hbm_accesses_standard,
    materialized_intermediate_elements,
    standard_attention,
)


def _random_qkv(seq_len=16, d=8, seed=0):
    rng = np.random.default_rng(seed)
    Q = rng.normal(size=(seq_len, d))
    K = rng.normal(size=(seq_len, d))
    V = rng.normal(size=(seq_len, d))
    return Q, K, V


def test_standard_attention_rows_are_probability_distributions():
    Q, K, V = _random_qkv()
    O, P = standard_attention(Q, K, V, return_weights=True)
    np.testing.assert_allclose(P.sum(axis=-1), np.ones(len(Q)), rtol=1e-6)
    assert O.shape == (len(Q), Q.shape[1])


def test_standard_attention_materializes_two_n_by_n_matrices():
    # §2.2:"Standard attention implementations materialize the matrices S and P to
    # HBM, which takes O(N^2) memory" —— 8K 上下文的感受数字:一张 S 表
    # 8192^2 = 67,108,864 ≈ 6700 万元素,fp16(2 字节)即 134,217,728 字节 ≈ 134MB;
    # S 与 P 各一张,共 2·N^2。
    n = 8192
    per_table = n * n
    total = materialized_intermediate_elements(n)
    assert per_table == 67_108_864
    assert per_table * 2 == 134_217_728
    assert total == 2 * per_table


def test_causal_mask_matches_bottom_right_alignment_examples():
    # 右下对齐语义,例取自 vllm/vllm_flash_attn/flash_attn_interface.py docstring
    # (L224-L234)的两个原例。FA-2 §3.1.1 只定义方阵 j>i 置 -inf;右下对齐是
    # seqlen_q != seqlen_k 时的对齐约定:query 行 r 的全序列位置 = r + query_offset,
    # 偏移 n_k - n_q 恰好把掩码贴到右下角。
    # seqlen_q=2, seqlen_k=5(例:1 1 1 1 0 / 1 1 1 1 1)
    np.testing.assert_array_equal(
        causal_keep_mask(2, 5, query_offset=3),
        np.array([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=bool),
    )
    # seqlen_q=5, seqlen_k=2(例:0 0 / 0 0 / 0 0 / 1 0 / 1 1)
    np.testing.assert_array_equal(
        causal_keep_mask(5, 2, query_offset=-3),
        np.array([[0, 0], [0, 0], [0, 0], [1, 0], [1, 1]], dtype=bool),
    )
    # decode:seqlen_q=1 对 seqlen_k=N 天然落在最右列,全部可见
    np.testing.assert_array_equal(
        causal_keep_mask(1, 5, query_offset=4), np.ones((1, 5), dtype=bool)
    )


def test_standard_attention_all_zero_mask_rows_output_zero():
    # 接口语义(docstring 原话):"If the row of the mask is all zero, the output
    # will be zero." —— seqlen_q=5, seqlen_k=2 右下对齐时前三行全零。
    rng = np.random.default_rng(7)
    Q = rng.normal(size=(5, 4))
    K = rng.normal(size=(2, 4))
    V = rng.normal(size=(2, 4))
    O = standard_attention(Q, K, V, causal=True, query_offset=-3)
    np.testing.assert_allclose(O[:3], np.zeros((3, 4)), atol=1e-12)
    # 后两行是正常因果注意力(row 3 只看 key 0,row 4 看 key 0/1)
    assert np.all(np.isfinite(O[3:]))


@pytest.mark.parametrize("block_r,block_c", [(4, 4), (3, 5), (1, 16), (16, 1), (7, 6)])
def test_flash_attention_matches_standard_attention_theorem1(block_r, block_c):
    # Theorem 1:Algorithm 1 returns O = softmax(QK^T)V —— 任意合法分块下逐位相等。
    Q, K, V = _random_qkv(seq_len=16, d=8, seed=1)
    O_std = standard_attention(Q, K, V)
    O_flash = flash_attention_forward(Q, K, V, block_size_r=block_r, block_size_c=block_c)
    np.testing.assert_allclose(O_flash, O_std, rtol=1e-6, atol=1e-8)


def test_flash_attention_never_materializes_full_score_matrix():
    # 白盒断言:tiling 过程中任意时刻在算的局部块 S_ij 至多 Br x Bc,从不是 N x N
    # ——「免物化」是结构性质,不只是结果相等。
    Q, K, V = _random_qkv(seq_len=32, d=8, seed=2)
    max_block_shape = flash_attention_forward(
        Q, K, V, block_size_r=8, block_size_c=8, return_max_block_shape=True
    )
    assert max_block_shape[0] <= 8
    assert max_block_shape[1] <= 8
    assert max_block_shape[0] * max_block_shape[1] < len(Q) * len(Q)


def test_flash_attention_causal_matches_standard_causal():
    # 因果掩码下 Theorem 1 同样成立;block 6/5 对 seq_len=20 会产生整块被遮住的
    # (i,j) 组合,顺带覆盖 -inf - (-inf) 的数值边界。
    Q, K, V = _random_qkv(seq_len=20, d=8, seed=3)
    O_std = standard_attention(Q, K, V, causal=True)
    O_flash = flash_attention_forward(Q, K, V, block_size_r=6, block_size_c=5, causal=True)
    np.testing.assert_allclose(O_flash, O_std, rtol=1e-6, atol=1e-8)


def test_flash_attention_running_output_equals_prefix_attention_after_each_kv_block():
    # Alg.1 line 12 每步写回的都是「归一化到当前为止」的 O_i:外层每处理完一个 KV 块 j,
    # 已写回的 O 应精确等于对 K[:kv_end], V[:kv_end] 一次性做标准注意力 —— 这是
    # worked example 的核心可示教性质(O_i 每步都是「至今为止的正确答案」)。
    Q, K, V = _random_qkv(seq_len=8, d=4, seed=11)
    trace = []
    flash_attention_forward(Q, K, V, block_size_r=3, block_size_c=3, trace=trace)
    kv_ends = sorted({e["kv_end"] for e in trace})
    assert kv_ends == [3, 6, 8]
    for kv_end in kv_ends:
        entries = sorted((e for e in trace if e["kv_end"] == kv_end), key=lambda e: e["q0"])
        assert entries[0]["q0"] == 0  # 覆盖全部 Q 行块
        O_now = np.concatenate([e["O"] for e in entries], axis=0)
        O_prefix = standard_attention(Q, K[:kv_end], V[:kv_end])
        np.testing.assert_allclose(O_now, O_prefix, rtol=1e-6, atol=1e-8)


def test_fa_block_sizes_matches_algorithm1_line1_formula():
    # Algorithm 1 line 1:Bc = ceil(M/4d), Br = min(ceil(M/4d), d)
    M, d = 100_000, 64
    Bc, Br = fa_block_sizes(M, d)
    assert Bc == math.ceil(M / (4 * d))
    assert Br == min(math.ceil(M / (4 * d)), d)


def test_flash_attention_2_matches_standard_attention():
    # FA-2 §3.1.1 Correctness:"Algorithm 1 returns the correct output O =
    # softmax(QK^T)V (with no approximation)" —— 循环序对调后精确性不变。
    Q, K, V = _random_qkv(seq_len=16, d=8, seed=12)
    O2, _ = flash_attention_2_forward(Q, K, V, block_size_r=4, block_size_c=5)
    np.testing.assert_allclose(O2, standard_attention(Q, K, V), rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("block_r,block_c", [(4, 4), (3, 5), (1, 16), (16, 1), (7, 6)])
def test_flash_attention_1_and_2_agree_for_all_block_sizes(block_r, block_c):
    # 外层 KV(FA Alg.1)与外层 Q(FA-2 Alg.1)两种循环序,输出逐位相等 ——
    # 循环序对调(§3.2 Parallelism)是调度改进,不改变数学。
    Q, K, V = _random_qkv(seq_len=16, d=8, seed=13)
    O1 = flash_attention_forward(Q, K, V, block_size_r=block_r, block_size_c=block_c)
    O2, _ = flash_attention_2_forward(Q, K, V, block_size_r=block_r, block_size_c=block_c)
    np.testing.assert_allclose(O2, O1, rtol=1e-6, atol=1e-8)


def test_flash_attention_2_logsumexp_matches_closed_form():
    # Tweak 2:"We only need to store the logsumexp L^(j) = m^(j) + log(l^(j))" ——
    # 即每行 log Σ_j e^{s_j}(s = scale·q·k);vLLM flash_attn_varlen_func 的
    # return_softmax_lse 拿到的正是这个 L。
    Q, K, V = _random_qkv(seq_len=10, d=8, seed=14)
    _, L = flash_attention_2_forward(Q, K, V, block_size_r=4, block_size_c=4)
    S = (Q @ K.T) / math.sqrt(Q.shape[-1])
    m_ref = S.max(axis=-1)
    L_ref = m_ref + np.log(np.exp(S - m_ref[:, None]).sum(axis=-1))
    np.testing.assert_allclose(L, L_ref, rtol=1e-6)


def test_flash_attention_2_causal_matches_standard_causal():
    Q, K, V = _random_qkv(seq_len=20, d=8, seed=15)
    O2, L2 = flash_attention_2_forward(
        Q, K, V, block_size_r=6, block_size_c=5, causal=True
    )
    np.testing.assert_allclose(O2, standard_attention(Q, K, V, causal=True), rtol=1e-6, atol=1e-8)
    # causal 下 L 是「行内可见键」的 logsumexp(前缀行可见 key 少)
    S = (Q @ K.T) / math.sqrt(Q.shape[-1])
    keep = np.tril(np.ones(S.shape, dtype=bool))
    S_masked = np.where(keep, S, -np.inf)
    m_ref = S_masked.max(axis=-1)
    l_ref = np.exp(S_masked - m_ref[:, None]).sum(axis=-1)
    np.testing.assert_allclose(L2, m_ref + np.log(l_ref), rtol=1e-6)


def test_flash_attention_2_causal_skips_blocks_above_diagonal():
    # FA-2 §3.1.1 Causal masking (1) 原话:"for any blocks where all the column indices
    # are more than the row indices (approximately half of the blocks for large sequence
    # length), we can skip the computation of that block. This leads to around 1.7-1.8x
    # speedup compared to attention without the causal mask." —— 条件触发型断言:
    # 前置条件(因果上侧整块确实存在)成立 + 目标效应(这些块确实没被计算)发生。
    # 期望集从地面真值掩码独立推导(块内含任一可见格 = 该块需要计算),不复制
    # 实现的 kc0 > row_hi 判跳式。
    n, d, block = 64, 8, 8
    rng = np.random.default_rng(21)
    Q, K, V = rng.normal(size=(n, d)), rng.normal(size=(n, d)), rng.normal(size=(n, d))
    trace = []
    O2, _ = flash_attention_2_forward(Q, K, V, block, block, causal=True, trace=trace)
    t_r = math.ceil(n / block)
    keep = causal_keep_mask(n, n, 0)
    needed = {
        (i, j)
        for i in range(t_r)
        for j in range(t_r)
        if keep[i * block : (i + 1) * block, j * block : (j + 1) * block].any()
    }
    computed = {(t["i"], t["j"]) for t in trace}
    assert len(computed) < t_r * t_r  # 前置条件:确实存在可跳过的因果上侧整块
    assert computed == needed  # 效应:恰好跳过「全被遮」的块,且不误跳任何需要的块
    # 论文口径:大方阵约可跳一半块 -> 计算量红利 total/needed ≈ 1.7-1.8x
    speedup = (t_r * t_r) / len(needed)
    assert 1.7 <= speedup <= 1.8
    # 跳过之后输出仍精确(Theorem 1 在 causal 下不变)
    np.testing.assert_allclose(O2, standard_attention(Q, K, V, causal=True), rtol=1e-6, atol=1e-8)


def test_flash_attention_2_trace_shows_unnormalized_accumulator():
    # Tweak 1:O 中间保持未归一化、收尾 diag(ℓ)^{-1} 只除一次 —— 轨迹逐 (i,j) 记录
    # (m, ℓ, O_unnormalized),每个 Q 行块最后一条轨迹的 O_unnormalized / ℓ 即最终 O。
    Q, K, V = _random_qkv(seq_len=8, d=4, seed=16)
    trace = []
    O, _ = flash_attention_2_forward(Q, K, V, block_size_r=3, block_size_c=3, trace=trace)
    assert len(trace) == 9  # ceil(8/3) x ceil(8/3) = 3 x 3 个 (i,j) 块
    for i in range(3):
        e = [t for t in trace if t["i"] == i][-1]
        O_final_block = O[e["q0"] : e["q1"]]
        np.testing.assert_allclose(
            e["O_unnormalized"] / e["l"][:, None], O_final_block, rtol=1e-6, atol=1e-8
        )


def test_io_complexity_theorem2_flash_needs_far_fewer_accesses_for_typical_sizes():
    # Theorem 2 + §3.2 原话:"For typical values of d (64-128) and M (around 100KB),
    # d^2 is many times smaller than M, and thus FlashAttention requires many times
    # fewer HBM accesses than standard implementation" —— 论文 Fig.2 实测最多 9x,
    # 这里断言方向与量级下界。
    N, d, M = 1024, 64, 100_000
    Bc, Br = fa_block_sizes(M, d)
    standard = hbm_accesses_standard(N, d)
    flash = hbm_accesses_flash(N, d, Bc, Br)
    assert flash < standard
    assert standard / flash > 2


def test_io_complexity_flash_accesses_decrease_as_block_size_grows():
    # §3.2:"As block size increases, the number of HBM accesses decreases (as we
    # make fewer passes over the input)"
    N, d = 1024, 64
    accesses = []
    for block in (16, 32, 64, 128, 256):
        accesses.append(hbm_accesses_flash(N, d, block, block))
    assert all(a2 <= a1 for a1, a2 in zip(accesses, accesses[1:]))


def test_io_complexity_standard_is_quadratic_in_seq_len():
    # Theorem 2:标准注意力 Θ(Nd + N^2),N≫d 时 N^2 主导 —— 访存数/N^2 趋于常数。
    d = 64
    ratios = []
    for N in (256, 512, 1024, 2048):
        ratios.append(hbm_accesses_standard(N, d) / (N * N))
    assert ratios[-1] / ratios[0] < 1.5
