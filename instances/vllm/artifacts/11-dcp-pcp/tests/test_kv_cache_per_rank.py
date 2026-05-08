"""Tests for kv_cache_per_rank.py — max_memory_usage_bytes and HBM sweep.

Source: vllm/v1/kv_cache_interface.py:L195-L205
"""

from __future__ import annotations

import pytest

from implementation.kv_cache_per_rank import (
    LLAMA_70B_KV_SPEC,
    KVCacheSpec,
    cdiv,
    fmt_gb,
    hbm_naive_total,
    hbm_per_rank,
)


# --------------------------------------------------------------------------
# cdiv
# --------------------------------------------------------------------------


def test_cdiv_exact_division():
    assert cdiv(100, 4) == 25


def test_cdiv_with_remainder_rounds_up():
    assert cdiv(100, 3) == 34
    assert cdiv(7, 4) == 2


def test_cdiv_zero_dividend():
    assert cdiv(0, 4) == 0


def test_cdiv_dividend_less_than_divisor():
    """cdiv(1, 8) = 1."""
    assert cdiv(1, 8) == 1


# --------------------------------------------------------------------------
# page_size_bytes = 2 * block_size * num_kv_heads * head_size * dtype_bytes
# --------------------------------------------------------------------------


def test_page_size_bytes_llama_70b():
    """Llama-70B: 2 * 16 * 8 * 128 * 2 = 65536 bytes per page."""
    assert LLAMA_70B_KV_SPEC.page_size_bytes == 65536


def test_page_size_bytes_factor_2_for_kv():
    """The factor of 2 is for K + V."""
    spec_2 = KVCacheSpec(num_layers=1, num_kv_heads=4, head_size=64, block_size=16, dtype_bytes=2)
    spec_1_kv = KVCacheSpec(num_layers=1, num_kv_heads=2, head_size=64, block_size=16, dtype_bytes=2)
    assert spec_2.page_size_bytes == 2 * spec_1_kv.page_size_bytes


# --------------------------------------------------------------------------
# Demo §1 — HBM per-rank sweep at Llama-70B / 128K
# --------------------------------------------------------------------------


def test_demo_naive_total_kv_bytes():
    """Demo §1: 42,949,672,960 = 40.0 GB at 128K, 80 layers, 8 KV heads, 128 head, bf16."""
    naive = hbm_naive_total(128 * 1024, LLAMA_70B_KV_SPEC)
    assert naive == 42_949_672_960


def test_demo_naive_total_is_40_gb():
    """fmt_gb(42949672960) = '40.0 GB'."""
    naive = hbm_naive_total(128 * 1024, LLAMA_70B_KV_SPEC)
    assert fmt_gb(naive) == "40.0 GB"


def test_demo_hbm_per_rank_no_cp():
    """Demo §1 cell (1,1): 42_949_672_960 bytes = 40.0 GB."""
    b = hbm_per_rank(128 * 1024, LLAMA_70B_KV_SPEC, dcp=1, pcp=1)
    assert b == 42_949_672_960
    assert fmt_gb(b) == "40.0 GB"


def test_demo_hbm_per_rank_dcp_2_pcp_2():
    """Demo §1 cell (2,2): 10_737_418_240 bytes = 10.0 GB."""
    b = hbm_per_rank(128 * 1024, LLAMA_70B_KV_SPEC, dcp=2, pcp=2)
    assert b == 10_737_418_240


def test_demo_hbm_per_rank_dcp_4_pcp_4():
    """Demo §1 cell (4,4): 2_684_354_560 bytes = 2.5 GB. The headline win."""
    b = hbm_per_rank(128 * 1024, LLAMA_70B_KV_SPEC, dcp=4, pcp=4)
    assert b == 2_684_354_560
    assert fmt_gb(b) == "2.5 GB"


def test_demo_hbm_per_rank_dcp_2_pcp_4():
    """Demo §1 cell (2,4): 5_368_709_120 bytes = 5.0 GB."""
    b = hbm_per_rank(128 * 1024, LLAMA_70B_KV_SPEC, dcp=2, pcp=4)
    assert b == 5_368_709_120


@pytest.mark.parametrize("dcp,pcp,expected_bytes", [
    (1, 1, 42_949_672_960),
    (1, 2, 21_474_836_480),
    (2, 1, 21_474_836_480),
    (2, 2, 10_737_418_240),
    (1, 4, 10_737_418_240),
    (4, 1, 10_737_418_240),
    (2, 4, 5_368_709_120),
    (4, 4, 2_684_354_560),
])
def test_demo_full_sweep_grid_verbatim(dcp, pcp, expected_bytes):
    """Demo §1 verbatim: all 8 (dcp, pcp) cells."""
    b = hbm_per_rank(128 * 1024, LLAMA_70B_KV_SPEC, dcp=dcp, pcp=pcp)
    assert b == expected_bytes


def test_demo_4_4_is_16x_smaller_than_no_cp():
    """Demo §1 headline: 40 GB → 2.5 GB = 16x reduction."""
    full = hbm_per_rank(128 * 1024, LLAMA_70B_KV_SPEC, dcp=1, pcp=1)
    sharded = hbm_per_rank(128 * 1024, LLAMA_70B_KV_SPEC, dcp=4, pcp=4)
    assert full // sharded == 16


# --------------------------------------------------------------------------
# max_memory_usage_bytes formula: cdiv(max_model_len, dcp*pcp) * cdiv(.., block_size) * page_size
# REFERENCE: vllm/v1/kv_cache_interface.py:L196-L204
# --------------------------------------------------------------------------


def test_max_memory_usage_formula_no_cp():
    """No CP: cdiv(L, block) * page_size, no per-rank shard."""
    spec = KVCacheSpec(num_layers=1, num_kv_heads=8, head_size=128, block_size=16, dtype_bytes=2)
    L = 128 * 1024
    expected = cdiv(L, 16) * spec.page_size_bytes
    assert spec.max_memory_usage_bytes(L) == expected


def test_max_memory_usage_formula_with_cp():
    """With CP: cdiv(cdiv(L, dcp*pcp), block) * page_size."""
    spec = KVCacheSpec(num_layers=1, num_kv_heads=8, head_size=128, block_size=16, dtype_bytes=2)
    L = 128 * 1024
    expected = cdiv(cdiv(L, 8), 16) * spec.page_size_bytes
    assert spec.max_memory_usage_bytes(L, dcp_world_size=4, pcp_world_size=2) == expected


def test_max_memory_usage_dcp_pcp_1_falls_to_no_cp_path():
    """When dcp*pcp == 1, source skips the cdiv reduction."""
    spec = KVCacheSpec(num_layers=1, num_kv_heads=8, head_size=128, block_size=16, dtype_bytes=2)
    L = 128 * 1024
    a = spec.max_memory_usage_bytes(L, dcp_world_size=1, pcp_world_size=1)
    b = spec.max_memory_usage_bytes(L)
    assert a == b


def test_max_memory_usage_decreases_with_total_cp():
    """Per-rank HBM shrinks monotonically as total_cp grows."""
    spec = KVCacheSpec(num_layers=1, num_kv_heads=8, head_size=128, block_size=16, dtype_bytes=2)
    L = 128 * 1024
    bytes_history = []
    for total in [(1, 1), (2, 1), (2, 2), (4, 2), (4, 4), (8, 4), (8, 8)]:
        b = spec.max_memory_usage_bytes(L, dcp_world_size=total[0], pcp_world_size=total[1])
        bytes_history.append(b)
    assert all(bytes_history[i] >= bytes_history[i + 1] for i in range(len(bytes_history) - 1))


# --------------------------------------------------------------------------
# Block-size padding behaviour (cdiv ceiling)
# --------------------------------------------------------------------------


def test_unaligned_seq_len_rounds_up_to_block():
    """cdiv: a seq_len of 17 with block_size=16 needs 2 blocks."""
    spec = KVCacheSpec(num_layers=1, num_kv_heads=8, head_size=128, block_size=16, dtype_bytes=2)
    # 17 tokens at block=16 → 2 blocks → 2 * page_size_bytes.
    assert spec.max_memory_usage_bytes(17) == 2 * spec.page_size_bytes


def test_seq_len_equal_to_block_size_one_block():
    spec = KVCacheSpec(num_layers=1, num_kv_heads=8, head_size=128, block_size=16, dtype_bytes=2)
    assert spec.max_memory_usage_bytes(16) == spec.page_size_bytes


def test_seq_len_one_token_one_block():
    """Even 1 token requires 1 full block (cdiv ceiling)."""
    spec = KVCacheSpec(num_layers=1, num_kv_heads=8, head_size=128, block_size=16, dtype_bytes=2)
    assert spec.max_memory_usage_bytes(1) == spec.page_size_bytes


# --------------------------------------------------------------------------
# Total-CP invariants
# --------------------------------------------------------------------------


def test_dcp_4_pcp_2_equals_dcp_2_pcp_4():
    """Per-rank HBM depends on the PRODUCT dcp*pcp, not on which axis."""
    a = hbm_per_rank(128 * 1024, LLAMA_70B_KV_SPEC, dcp=4, pcp=2)
    b = hbm_per_rank(128 * 1024, LLAMA_70B_KV_SPEC, dcp=2, pcp=4)
    assert a == b


def test_dcp_8_pcp_1_equals_dcp_1_pcp_8():
    a = hbm_per_rank(128 * 1024, LLAMA_70B_KV_SPEC, dcp=8, pcp=1)
    b = hbm_per_rank(128 * 1024, LLAMA_70B_KV_SPEC, dcp=1, pcp=8)
    assert a == b


# --------------------------------------------------------------------------
# Scaling laws
# --------------------------------------------------------------------------


def test_doubling_total_cp_halves_hbm():
    """HBM scales as 1/total_cp at large seq_len (cdiv noise negligible)."""
    spec = KVCacheSpec(num_layers=80, num_kv_heads=8, head_size=128, block_size=16, dtype_bytes=2)
    seq = 128 * 1024  # large enough for cdiv to be exact
    h1 = hbm_per_rank(seq, spec, dcp=1, pcp=1)
    h2 = hbm_per_rank(seq, spec, dcp=2, pcp=1)
    h4 = hbm_per_rank(seq, spec, dcp=4, pcp=1)
    h8 = hbm_per_rank(seq, spec, dcp=8, pcp=1)
    assert h1 // h2 == 2
    assert h2 // h4 == 2
    assert h4 // h8 == 2


def test_doubling_seq_len_doubles_hbm():
    """At fixed CP, HBM scales linearly with seq_len."""
    h_short = hbm_per_rank(64 * 1024, LLAMA_70B_KV_SPEC, dcp=1, pcp=1)
    h_long = hbm_per_rank(128 * 1024, LLAMA_70B_KV_SPEC, dcp=1, pcp=1)
    assert h_long == 2 * h_short


# --------------------------------------------------------------------------
# fmt_gb formatting
# --------------------------------------------------------------------------


def test_fmt_gb_one_decimal():
    assert fmt_gb(1024 ** 3) == "1.0 GB"


def test_fmt_gb_zero_bytes():
    assert fmt_gb(0) == "0.0 GB"


def test_fmt_gb_2_5_gb():
    """Demo §1 (4,4) cell formats as '2.5 GB'."""
    assert fmt_gb(2_684_354_560) == "2.5 GB"


def test_fmt_gb_40_0_gb():
    assert fmt_gb(42_949_672_960) == "40.0 GB"
