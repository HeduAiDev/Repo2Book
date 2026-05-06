"""Unit + integration tests for paged_integration (composition with PagedAttention).

Coverage:
- get_computed_blocks: max_cache_hit_length = num_tokens - 1 (K04 / Trap 3)
- allocate_with_prefix_cache: end-to-end match → touch → fresh → cache pipeline
- AllocateResult.num_fresh_blocks computation
- verify_invariants: I1, I2, I3 all return PASS on demo-shaped workloads
"""

from __future__ import annotations

from implementation.paged_integration import (
    AllocateResult,
    allocate_with_prefix_cache,
    get_computed_blocks,
    verify_invariants,
)
from implementation.prefix_cache_manager import PrefixCacheManager


class TestGetComputedBlocks:
    def test_first_request_returns_empty_result(self) -> None:
        """No prior cache → no hits."""
        mgr = PrefixCacheManager(block_size=4)
        tokens = list(range(1, 17))  # 16 tokens
        result = get_computed_blocks(mgr, tokens)
        assert isinstance(result, AllocateResult)
        assert result.num_cache_hit_blocks == 0
        assert result.num_cache_hit_tokens == 0
        assert result.block_ids == []

    def test_second_request_returns_prefix_blocks(self) -> None:
        """A allocates first; B's get_computed_blocks finds A's prefix."""
        mgr = PrefixCacheManager(block_size=4)
        # A: 16 tokens, 4 blocks.
        a_tokens = list(range(1, 17))
        allocate_with_prefix_cache(mgr, "A", a_tokens)
        # B: shares 8 tokens with A.
        b_tokens = list(range(1, 9)) + [101, 102, 103, 104, 105, 106, 107, 108]
        result = get_computed_blocks(mgr, b_tokens)
        assert result.num_cache_hit_blocks == 2
        assert result.num_cache_hit_tokens == 8

    def test_max_cache_hit_length_is_n_minus_one(self) -> None:
        """K04 / vLLM kv_cache_manager.py:L208: max_hit = num_tokens - 1.

        With block_size=4 and num_tokens=8, max_hit=7. Floor(7/4)=1 block max.
        Even if 2 full blocks would otherwise hit, only 1 is reported."""
        mgr = PrefixCacheManager(block_size=4)
        # Cache 2 blocks for A (16 tokens of identical content).
        a_tokens = [1, 2, 3, 4, 5, 6, 7, 8] * 2  # 16 tokens
        allocate_with_prefix_cache(mgr, "A", a_tokens)

        # B has 8 tokens — same as A's first block + second block.
        b_tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        result = get_computed_blocks(mgr, b_tokens)
        # max_hit = 8 - 1 = 7; 7 // 4 = 1 block.
        # Even though 2 blocks could match, only 1 is reported (logits gap).
        assert result.num_cache_hit_blocks == 1


class TestAllocateWithPrefixCache:
    def test_allocate_result_shape(self) -> None:
        mgr = PrefixCacheManager(block_size=4)
        tokens = list(range(1, 9))  # 8 tokens = 2 blocks
        result = allocate_with_prefix_cache(mgr, "A", tokens)
        assert result.num_cache_hit_blocks == 0
        assert len(result.block_ids) == 2
        assert result.num_fresh_blocks == 2

    def test_demo_section_3_chain_break(self) -> None:
        """Reproduces demo §3:
        - A allocates 4 blocks (hit=0, fresh=4)
        - B reuses 2 of A's blocks (hit=2, fresh=2)
        - C unrelated (hit=0, fresh=2)
        - Evict A's block 0 → D's chain breaks immediately (hit=0)"""
        mgr = PrefixCacheManager(block_size=4)
        sys_prompt = [101, 102, 103, 104, 105, 106, 107, 108]

        req_a = sys_prompt + [201, 202, 203, 204, 205, 206, 207, 208]
        res_a = allocate_with_prefix_cache(mgr, "A", req_a)
        assert res_a.num_cache_hit_blocks == 0
        assert res_a.num_fresh_blocks == 4

        req_b = sys_prompt + [301, 302, 303, 304, 305, 306, 307, 308]
        res_b = allocate_with_prefix_cache(mgr, "B", req_b)
        assert res_b.num_cache_hit_blocks == 2
        assert res_b.num_fresh_blocks == 2

        req_c = [501, 502, 503, 504, 505, 506, 507, 508]
        res_c = allocate_with_prefix_cache(mgr, "C", req_c)
        assert res_c.num_cache_hit_blocks == 0
        assert res_c.num_fresh_blocks == 2

        # Free A then evict its first block.
        from implementation.block_hash import chain_block_hashes
        mgr.free_request("A")
        chain_a = chain_block_hashes(req_a, mgr.block_size)
        evicted = mgr.evict_block(res_a.block_ids[0], chain_a[0])
        assert evicted is True

        req_d = sys_prompt + [401, 402, 403, 404, 405, 406, 407, 408]
        res_d = allocate_with_prefix_cache(mgr, "D", req_d)
        # Chain breaks at evicted block 0 → 0 hits, even though block 1 lives.
        assert res_d.num_cache_hit_blocks == 0


class TestKVSavingsClaim:
    """Re-derive the 78% KV savings from demo §4."""

    def test_demo_section_4_arithmetic(self) -> None:
        """50 reqs × (sys_prompt 32 blocks + user 8 blocks) → 2000 naive blocks.
        With prefix cache: first req allocates 40, subsequent 49 share 32 sys blocks
        and add 8 fresh each → 40 + 49*8 = 432 prefix-aware. Saved 1568 of 2000 = 78.4%."""
        block_size = 16
        num_requests = 50
        sys_prompt_tokens = 512   # 32 blocks
        user_tokens = 128         # 8 blocks
        sys_prompt = list(range(1, sys_prompt_tokens + 1))

        mgr = PrefixCacheManager(block_size=block_size)
        pa_blocks = 0
        for r in range(num_requests):
            tokens = sys_prompt + list(
                range(10000 + r * 1000, 10000 + r * 1000 + user_tokens)
            )
            res = allocate_with_prefix_cache(mgr, f"r{r}", tokens)
            pa_blocks += res.num_fresh_blocks

        naive_blocks = num_requests * (sys_prompt_tokens + user_tokens) // block_size
        saved = naive_blocks - pa_blocks

        assert naive_blocks == 2000
        assert pa_blocks == 432
        assert saved == 1568
        # Saving fraction: 1568 / 2000 = 0.784 → demo prints "78%".
        assert abs(saved / naive_blocks - 0.784) < 0.001


class TestNumBlocksSavedFormula:
    """Critical fidelity check: num_blocks_saved ≈ (num_reqs - 1) * sys_prompt_blocks."""

    def test_savings_scale_with_shared_prefix(self) -> None:
        """If N requests share a K-block prefix, savings ≈ (N-1) * K."""
        for n_reqs, sys_blocks in [(10, 4), (20, 8), (50, 32)]:
            block_size = 16
            sys_prompt = list(range(1, sys_blocks * block_size + 1))
            mgr = PrefixCacheManager(block_size=block_size)
            pa_blocks = 0
            for r in range(n_reqs):
                # Each request: sys_prompt + tiny unique tail (1 fresh block).
                tokens = sys_prompt + list(range(10000 + r * 100,
                                                 10000 + r * 100 + block_size))
                res = allocate_with_prefix_cache(mgr, f"r{r}", tokens)
                pa_blocks += res.num_fresh_blocks
            naive_blocks = n_reqs * (sys_blocks + 1)
            saved = naive_blocks - pa_blocks
            # First request pays full sys_blocks; remaining N-1 share.
            expected_saved = (n_reqs - 1) * sys_blocks
            assert saved == expected_saved


class TestVerifyInvariants:
    def test_all_three_invariants_pass_on_demo_workload(self) -> None:
        """Demo §5: I1, I2, I3 PASS on a 10-request workload."""
        mgr = PrefixCacheManager(block_size=16)
        sys_prompt = list(range(1, 513))
        for r in range(10):
            tokens = sys_prompt + list(range(10000 + r, 10000 + r + 128))
            allocate_with_prefix_cache(mgr, f"r{r}", tokens)

        inv = verify_invariants(mgr)
        assert inv["I1_append_only_ids"] is True
        assert inv["I2_ref_cnt_consistent"] is True
        assert inv["I3_chain_monotone"] is True

    def test_invariants_pass_on_empty_manager(self) -> None:
        """Empty manager trivially satisfies the invariants."""
        mgr = PrefixCacheManager(block_size=16)
        inv = verify_invariants(mgr)
        assert all(inv.values())
