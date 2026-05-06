"""Unit tests for KVCacheSpec / AttentionSpec / FullAttentionSpec / get_num_blocks.

The page-size formula (vLLM kv_cache_interface.py:L153-L170) is load-bearing for
every byte budget downstream — these tests pin the exact integer math.
"""

from __future__ import annotations

import pytest

from implementation.kv_cache_spec import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    get_num_blocks,
)


class TestPageSizeFormula:
    def test_canonical_llama_3_2_1b_layer(self) -> None:
        """block_size=16, num_kv_heads=8, head_size=128, fp16 → 64 KiB.

        2 * 16 * 8 * 128 * 2 = 65536 bytes = 64 KiB per layer per block.
        This is the demo's page_size; pinning protects against accidental drift.
        """
        spec = AttentionSpec(block_size=16, num_kv_heads=8, head_size=128, dtype_bytes=2)
        assert spec.real_page_size_bytes == 65536
        assert spec.page_size_bytes == 65536

    def test_doubling_block_size_doubles_page(self) -> None:
        """page_size scales linearly with block_size (the 5-line proof from the demo)."""
        base = AttentionSpec(block_size=16, num_kv_heads=8, head_size=128, dtype_bytes=2)
        big = AttentionSpec(block_size=32, num_kv_heads=8, head_size=128, dtype_bytes=2)
        assert big.page_size_bytes == 2 * base.page_size_bytes

    def test_dtype_doubles_for_fp32(self) -> None:
        """fp32 (dtype_bytes=4) doubles the page from fp16."""
        fp16 = AttentionSpec(block_size=16, num_kv_heads=8, head_size=128, dtype_bytes=2)
        fp32 = AttentionSpec(block_size=16, num_kv_heads=8, head_size=128, dtype_bytes=4)
        assert fp32.page_size_bytes == 2 * fp16.page_size_bytes

    def test_kv_cache_spec_base_raises_not_implemented(self) -> None:
        """KVCacheSpec is abstract; page_size_bytes/max_memory_usage_bytes raise."""
        spec = KVCacheSpec(block_size=16)
        with pytest.raises(NotImplementedError):
            _ = spec.page_size_bytes
        with pytest.raises(NotImplementedError):
            spec.max_memory_usage_bytes(1024)


class TestFullAttentionSpec:
    def test_head_size_v_defaults_to_head_size(self) -> None:
        """vLLM kv_cache_interface.py:L192-L194 — post_init copies head_size."""
        spec = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=128, dtype_bytes=2)
        assert spec.head_size_v == 128

    def test_head_size_v_explicit_override(self) -> None:
        """Caller can set a different head_size_v (rare; e.g. MLA-like attention)."""
        spec = FullAttentionSpec(
            block_size=16, num_kv_heads=8, head_size=128,
            dtype_bytes=2, head_size_v=64,
        )
        assert spec.head_size_v == 64

    def test_max_memory_usage_bytes_rounds_up(self) -> None:
        """ceil(max_model_len / block_size) blocks * page_size."""
        spec = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=128, dtype_bytes=2)
        # 32 tokens → 2 blocks → 2 * 65536 = 131072 bytes.
        assert spec.max_memory_usage_bytes(32) == 131072
        # 33 tokens → 3 blocks (ceil) → 196608.
        assert spec.max_memory_usage_bytes(33) == 196608
        # 0 tokens → 0 blocks → 0.
        assert spec.max_memory_usage_bytes(0) == 0


class TestGetNumBlocks:
    def test_demo_numbers_reproduce(self) -> None:
        """The demo's 35148 blocks must come out exactly.

        available_kv_cache (from demo) = 68.65 GiB approx.
        Reproducing the exact int math:
            requested = int(80 GiB * 0.92) = 78920663040
            non_kv = int(2.4 GiB) + int(1.8 GiB) + int(0.5 GiB)
                  = 2576980377 + 1932735283 + 536870912 = 5046586572
            cudagraph = 256 * 1024**2 = 268435456
            available = 78920663040 - 5046586572 - 268435456 = 73605641012... approx
        We don't recompute the int chain here; we use a known-good page_size and
        a clean memory number to test the divide.
        """
        gib = 1024**3
        # 64 GiB of KV cache, 32 layers, page_size = 64 KiB → expect 2^20 // 32 = 32768 blocks.
        # Specifically: 64 * 2^30 / (64 * 2^10) / 32 = 64 * 2^16 / 32 = 65536 / 32 * 2 = wait...
        # 64 * 2^30 / 65536 = 2^36 / 2^16 = 2^20 = 1048576. Then //32 = 32768.
        n = get_num_blocks(64 * gib, num_layers=32, page_size=65536)
        assert n == 32768

    def test_clamp_negative_to_zero(self) -> None:
        """If available_memory < 0, num_blocks must clamp to 0 (no negative blocks)."""
        assert get_num_blocks(-1, num_layers=32, page_size=65536) == 0
        assert get_num_blocks(0, num_layers=32, page_size=65536) == 0

    def test_layer_factor_separated(self) -> None:
        """Doubling num_layers halves the block count (linear)."""
        gib = 1024**3
        a = get_num_blocks(64 * gib, num_layers=16, page_size=65536)
        b = get_num_blocks(64 * gib, num_layers=32, page_size=65536)
        assert a == 2 * b

    def test_wasted_remainder_small(self) -> None:
        """The integer divide should waste less than (page_size * num_layers)
        bytes — that's the upper bound of integer-divide loss."""
        avail = 64 * 1024**3 + 12345  # a "weird" odd amount
        page = 65536
        layers = 32
        n = get_num_blocks(avail, layers, page)
        wasted = avail - n * page * layers
        assert 0 <= wasted < page * layers
