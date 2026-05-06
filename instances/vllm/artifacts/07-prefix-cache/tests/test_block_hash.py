"""Unit tests for block_hash primitives.

Critical invariants (vLLM kv_cache_utils.py:L40-L78, L539-L566):
- BlockHashWithGroupId packs hash + 4-byte big-endian group_id (K08).
- get_block_hash / get_group_id round-trip.
- hash_block_tokens chain: H_k depends on parent_hash transitively (K03).
- Same body, different parent → different chained hash (collision-resistance).
- Same body, same parent → same hash (determinism).
- Trailing partial block is NOT hashed (only full blocks; impl L131-L139).
- common_prefix_length ends at first divergence.
"""

from __future__ import annotations

from implementation.block_hash import (
    NONE_HASH,
    chain_block_hashes,
    common_prefix_length,
    get_block_hash,
    get_group_id,
    hash_block_tokens,
    make_block_hash_with_group_id,
)


class TestGroupIdPacking:
    """K08: BlockHashWithGroupId = block_hash + 4-byte big-endian group_id."""

    def test_pack_appends_4_bytes(self) -> None:
        """Packed key is exactly len(hash) + 4 bytes."""
        h = b"\x00" * 32
        packed = make_block_hash_with_group_id(h, 0)
        assert len(packed) == 32 + 4

    def test_round_trip_unpack(self) -> None:
        """get_block_hash + get_group_id round-trip the original inputs."""
        h = b"\x12" * 32
        packed = make_block_hash_with_group_id(h, 7)
        assert get_block_hash(packed) == h
        assert get_group_id(packed) == 7

    def test_different_group_ids_distinct(self) -> None:
        """Same hash + different group_id → distinct packed keys (K08).
        Critical for hybrid attention models: full-attn vs sliding-window
        layers must NOT collide in the prefix-cache index."""
        h = b"\x42" * 32
        a = make_block_hash_with_group_id(h, 0)
        b = make_block_hash_with_group_id(h, 1)
        assert a != b

    def test_endianness_is_big(self) -> None:
        """K08: group_id is big-endian. Verify by hand."""
        h = b""
        packed = make_block_hash_with_group_id(h, 0x01020304)
        # Last 4 bytes should be 0x01, 0x02, 0x03, 0x04 — big-endian.
        assert bytes(packed[-4:]) == b"\x01\x02\x03\x04"


class TestHashBlockTokens:
    """K03: H_k = hash(parent_hash, tokens, extra_keys); chained property."""

    def test_determinism_same_inputs_same_hash(self) -> None:
        """Same parent + same tokens + same extras → same hash."""
        a = hash_block_tokens(NONE_HASH, [1, 2, 3, 4])
        b = hash_block_tokens(NONE_HASH, [1, 2, 3, 4])
        assert a == b

    def test_different_tokens_different_hash(self) -> None:
        """Different body → different hash (basic collision-resistance)."""
        a = hash_block_tokens(NONE_HASH, [1, 2, 3, 4])
        b = hash_block_tokens(NONE_HASH, [1, 2, 3, 5])
        assert a != b

    def test_different_parent_different_hash(self) -> None:
        """K03 invariant: same body but different parent → different hash.
        This is what makes the chain provide the 'tree property'."""
        body = [10, 20, 30, 40]
        h1 = hash_block_tokens(b"\x00" * 32, body)
        h2 = hash_block_tokens(b"\xff" * 32, body)
        assert h1 != h2

    def test_extra_keys_change_hash(self) -> None:
        """K09: extra_keys (lora/mm/cache_salt/prompt_embeds) affect the hash.
        Same tokens with different extras must NOT collide."""
        body = [1, 2, 3, 4]
        h1 = hash_block_tokens(NONE_HASH, body, extra_keys=())
        h2 = hash_block_tokens(NONE_HASH, body, extra_keys=("lora-A",))
        h3 = hash_block_tokens(NONE_HASH, body, extra_keys=("lora-B",))
        assert h1 != h2
        assert h2 != h3
        assert h1 != h3

    def test_none_parent_treated_as_NONE_HASH(self) -> None:
        """impl L98-L100: parent=None or empty → use NONE_HASH sentinel."""
        body = [1, 2, 3, 4]
        h_none = hash_block_tokens(None, body)
        h_sentinel = hash_block_tokens(NONE_HASH, body)
        assert h_none == h_sentinel


class TestChainBlockHashes:
    """K03 + impl L131-L139: only FULL blocks are hashed; trailing partial dropped."""

    def test_full_chain_for_aligned_input(self) -> None:
        """16 tokens / block_size=4 → 4 hashes."""
        chain = chain_block_hashes(list(range(1, 17)), block_size=4)
        assert len(chain) == 4

    def test_partial_block_not_hashed(self) -> None:
        """17 tokens / block_size=4 → 4 hashes (last partial block dropped)."""
        chain = chain_block_hashes(list(range(1, 18)), block_size=4)
        assert len(chain) == 4

    def test_single_partial_block_no_hash(self) -> None:
        """3 tokens / block_size=4 → 0 hashes (entire input < one block)."""
        chain = chain_block_hashes([1, 2, 3], block_size=4)
        assert chain == []

    def test_chain_property_each_depends_on_previous(self) -> None:
        """K03: H_k depends on H_{k-1}. Tweaking block 0 changes ALL subsequent."""
        a = chain_block_hashes([1, 2, 3, 4, 5, 6, 7, 8], block_size=4)
        # Change one token in block 0; expect both H_0 and H_1 to differ.
        b = chain_block_hashes([1, 2, 3, 99, 5, 6, 7, 8], block_size=4)
        assert a[0] != b[0]
        assert a[1] != b[1]

    def test_shared_prefix_shared_chain_prefix(self) -> None:
        """Two sequences with the same first 8 tokens have H_0 == H_0', H_1 == H_1'."""
        a = chain_block_hashes([1, 2, 3, 4, 5, 6, 7, 8] + [99, 99, 99, 99],
                               block_size=4)
        b = chain_block_hashes([1, 2, 3, 4, 5, 6, 7, 8] + [42, 42, 42, 42],
                               block_size=4)
        assert a[0] == b[0]
        assert a[1] == b[1]
        # The third block differs because tokens 9-12 differ.
        assert a[2] != b[2]

    def test_extra_keys_shifts_all_hashes(self) -> None:
        """K09: extra_keys are folded in at every block, so changing them
        shifts every hash in the chain."""
        a = chain_block_hashes([1, 2, 3, 4, 5, 6, 7, 8], block_size=4, extra_keys=())
        b = chain_block_hashes([1, 2, 3, 4, 5, 6, 7, 8], block_size=4,
                               extra_keys=("lora-X",))
        assert a[0] != b[0]
        assert a[1] != b[1]


class TestCommonPrefixLength:
    def test_full_match(self) -> None:
        a = chain_block_hashes(list(range(16)), block_size=4)
        b = chain_block_hashes(list(range(16)), block_size=4)
        assert common_prefix_length(a, b) == 4

    def test_no_match(self) -> None:
        a = chain_block_hashes(list(range(0, 16)), block_size=4)
        b = chain_block_hashes(list(range(100, 116)), block_size=4)
        # Different starting tokens → H_0 differs immediately.
        assert common_prefix_length(a, b) == 0

    def test_partial_match(self) -> None:
        """Common 8-token prefix (= 2 blocks at block_size=4)."""
        a = chain_block_hashes([1, 2, 3, 4, 5, 6, 7, 8] + [10, 11, 12, 13],
                               block_size=4)
        b = chain_block_hashes([1, 2, 3, 4, 5, 6, 7, 8] + [99, 99, 99, 99],
                               block_size=4)
        assert common_prefix_length(a, b) == 2

    def test_empty_inputs(self) -> None:
        assert common_prefix_length([], []) == 0
        assert common_prefix_length([], [b"\x00" * 32]) == 0
