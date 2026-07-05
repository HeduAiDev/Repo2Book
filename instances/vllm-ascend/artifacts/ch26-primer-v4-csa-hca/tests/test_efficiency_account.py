import pytest

from efficiency_account import (
    csa_layer_cost,
    dense_baseline_layer_cost,
    hca_layer_cost,
    hybrid_stack_average_cost,
    mixed_precision_kv_bytes,
    pure_bf16_kv_bytes,
    relative_efficiency,
    worked_example_efficiency,
)


def test_csa_layer_cost_kv_scales_with_1_over_m():
    seq_len = 100_000
    c1 = csa_layer_cost(seq_len, m=1, k=512, n_win=256, head_dim=128, indexer_heads=4, indexer_dim=64)
    c4 = csa_layer_cost(seq_len, m=4, k=512, n_win=256, head_dim=128, indexer_heads=4, indexer_dim=64)
    assert c4.kv_entries_stored == pytest.approx(c1.kv_entries_stored / 4)


def test_csa_layer_cost_flops_independent_of_seq_len_for_core_attn_but_indexer_scales():
    seq_len_small, seq_len_large = 10_000, 1_000_000
    small = csa_layer_cost(seq_len_small, m=4, k=512, n_win=0, head_dim=128, indexer_heads=1, indexer_dim=128)
    large = csa_layer_cost(seq_len_large, m=4, k=512, n_win=0, head_dim=128, indexer_heads=1, indexer_dim=128)
    # indexer 开销随 seq_len/m 线性增长,core-attn 部分(k*head_dim)与 seq_len 无关
    core_only = 512 * 128
    assert (small.flops_per_token - core_only) * (seq_len_large / seq_len_small) == pytest.approx(
        large.flops_per_token - core_only
    )


def test_hca_layer_cost_kv_scales_with_1_over_m_prime():
    seq_len = 100_000
    c = hca_layer_cost(seq_len, m_prime=128, n_win=0, head_dim=64)
    assert c.kv_entries_stored == pytest.approx(seq_len / 128)


def test_dense_baseline_layer_cost_stores_full_sequence():
    c = dense_baseline_layer_cost(seq_len=50_000, head_dim=128)
    assert c.kv_entries_stored == 50_000
    assert c.flops_per_token == 50_000 * 128


def test_csa_with_m_equals_1_matches_dsa_only_semantics():
    """m=1 时 CSA 退化为"不压缩、只做 top-k 稀疏"——即论文对照基线 DSA。"""
    seq_len = 200_000
    c = csa_layer_cost(seq_len, m=1, k=1024, n_win=128, head_dim=128, indexer_heads=2, indexer_dim=64)
    assert c.kv_entries_stored == pytest.approx(seq_len)   # 不压缩,KV 存满
    expected_flops = (1024 + 128) * 128 + seq_len * 2 * 64
    assert c.flops_per_token == pytest.approx(expected_flops)


def test_hybrid_stack_average_cost_mixes_csa_hca():
    seq_len = 100_000
    ratios = [4, 128]
    csa = csa_layer_cost(seq_len, m=4, k=512, n_win=256, head_dim=128, indexer_heads=4, indexer_dim=64)
    hca = hca_layer_cost(seq_len, m_prime=128, n_win=256, head_dim=128)
    avg = hybrid_stack_average_cost(ratios, seq_len, k=512, n_win=256, head_dim=128,
                                     indexer_heads=4, indexer_dim=64)
    assert avg.kv_entries_stored == pytest.approx((csa.kv_entries_stored + hca.kv_entries_stored) / 2)
    assert avg.flops_per_token == pytest.approx((csa.flops_per_token + hca.flops_per_token) / 2)


def test_hybrid_stack_average_cost_rejects_unsupported_ratio():
    with pytest.raises(ValueError):
        hybrid_stack_average_cost([16], 1000, k=1, n_win=1, head_dim=8)


def test_hybrid_stack_average_cost_treats_le_1_as_dense():
    seq_len = 1000
    avg = hybrid_stack_average_cost([0], seq_len, k=1, n_win=1, head_dim=8)
    dense = dense_baseline_layer_cost(seq_len, 8)
    assert avg.kv_entries_stored == pytest.approx(dense.kv_entries_stored)
    assert avg.flops_per_token == pytest.approx(dense.flops_per_token)


def test_relative_efficiency_ratio_less_than_one_when_hybrid_cheaper():
    seq_len = 1_000_000
    hybrid = hybrid_stack_average_cost([4, 4, 4, 128], seq_len, k=2048, n_win=1024, head_dim=128,
                                        indexer_heads=4, indexer_dim=64)
    baseline = dense_baseline_layer_cost(seq_len, 128)
    flops_ratio, kv_ratio = relative_efficiency(hybrid, baseline)
    assert 0 < flops_ratio < 1
    assert 0 < kv_ratio < 1


def test_mixed_precision_kv_bytes_less_than_pure_bf16():
    kv_entries = 1000.0
    mixed = mixed_precision_kv_bytes(kv_entries, rope_dims=64, other_dims=64)
    pure = pure_bf16_kv_bytes(kv_entries, total_dims=128)
    assert mixed < pure


def test_worked_example_efficiency_produces_sub_one_ratios_and_is_honest_about_scope():
    """跑一遍示意性数值推演:验证账本模型本身自洽(hybrid 相对两个基线都更省),
    但明确不断言这就是论文的 27%/10%(那需要 DeepSeek 未公开的完整配置)。"""
    result = worked_example_efficiency()
    assert 0 < result.flops_ratio_vs_dense < 1
    assert 0 < result.kv_ratio_vs_dense < 1
    assert 0 < result.flops_ratio_vs_dsa < 1
    assert 0 < result.kv_ratio_vs_dsa < 1
    # 混合精度存储应比纯 BF16 更省
    assert result.kv_bytes_mixed_precision < result.kv_bytes_pure_bf16
