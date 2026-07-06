"""arXiv:2205.14135 (Dao et al., FlashAttention) §2.2 Algorithm 0 (standard attention),
§3.1 Algorithm 1 (tiling + online-softmax recurrence, Theorem 1 correctness),
§3.2 Theorem 2 (IO complexity)。
"""
import numpy as np
import pytest

from flash_attention import (
    standard_attention,
    flash_attention_forward,
    fa_block_sizes,
    hbm_accesses_standard,
    hbm_accesses_flash,
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


@pytest.mark.parametrize("block_r,block_c", [(4, 4), (3, 5), (1, 16), (16, 1), (7, 6)])
def test_flash_attention_matches_standard_attention_theorem1(block_r, block_c):
    # Theorem 1: Algorithm 1 returns O = softmax(QK^T)V exactly, for any valid block size.
    Q, K, V = _random_qkv(seq_len=16, d=8, seed=1)
    O_std = standard_attention(Q, K, V)
    O_flash = flash_attention_forward(Q, K, V, block_size_r=block_r, block_size_c=block_c)
    np.testing.assert_allclose(O_flash, O_std, rtol=1e-6, atol=1e-8)


def test_flash_attention_never_materializes_full_score_matrix():
    # 白盒断言:tiling 过程中任意时刻在算的局部块 S_ij 至多 Br x Bc,不是完整 N x N。
    Q, K, V = _random_qkv(seq_len=32, d=8, seed=2)
    max_block_shape = flash_attention_forward(
        Q, K, V, block_size_r=8, block_size_c=8, return_max_block_shape=True
    )
    assert max_block_shape[0] <= 8
    assert max_block_shape[1] <= 8
    assert max_block_shape[0] * max_block_shape[1] < len(Q) * len(Q)


def test_flash_attention_causal_matches_standard_causal():
    Q, K, V = _random_qkv(seq_len=20, d=8, seed=3)
    O_std = standard_attention(Q, K, V, causal=True)
    O_flash = flash_attention_forward(Q, K, V, block_size_r=6, block_size_c=5, causal=True)
    np.testing.assert_allclose(O_flash, O_std, rtol=1e-6, atol=1e-8)


def test_fa_block_sizes_matches_algorithm1_line1_formula():
    # Algorithm 1 line 1: Bc=ceil(M/4d), Br=min(ceil(M/4d), d)
    M, d = 100_000, 64
    Bc, Br = fa_block_sizes(M, d)
    assert Bc == int(np.ceil(M / (4 * d)))
    assert Br == min(int(np.ceil(M / (4 * d))), d)


def test_io_complexity_theorem2_flash_needs_far_fewer_accesses_for_typical_sizes():
    # Theorem 2 + 论文原句(§3.2):"For typical values of d (64-128) and M (around 100KB),
    # d^2 is many times smaller than M, and thus FlashAttention requires many times fewer
    # HBM accesses than standard implementation."
    N, d, M = 1024, 64, 100_000
    Bc, Br = fa_block_sizes(M, d)
    standard = hbm_accesses_standard(N, d)
    flash = hbm_accesses_flash(N, d, Bc, Br)
    assert flash < standard
    assert standard / flash > 2  # 论文图 2 报告的量级(最多 9x),这里只断言方向与量级


def test_io_complexity_flash_accesses_decrease_as_block_size_grows():
    # 论文 §3.2:"As block size increases,...the number of HBM accesses decreases"
    N, d = 1024, 64
    accesses = []
    for block in (16, 32, 64, 128, 256):
        accesses.append(hbm_accesses_flash(N, d, block, block))
    assert all(a2 <= a1 for a1, a2 in zip(accesses, accesses[1:]))


def test_io_complexity_standard_is_quadratic_in_seq_len():
    d = 64
    ratios = []
    for N in (256, 512, 1024, 2048):
        ratios.append(hbm_accesses_standard(N, d) / (N * N))
    # Θ(N^2) 主导项:accesses/N^2 应随 N 增大而趋于常数(而不是继续显著增长)
    assert ratios[-1] / ratios[0] < 1.5
