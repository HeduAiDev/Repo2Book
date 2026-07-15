"""ch02 -- coalesced vs. scattered global-memory access.

vector-add's own access pattern (``offsets = block_start + arange(0,
BLOCK_SIZE)``) is contiguous, so a warp's 32 lanes at 4-byte (fp32) elements
span exactly one 128-byte transaction. The strided/gather case is the
counter-example that makes the cost of *not* coalescing measurable.
"""
from coalescing import count_transactions, warp_offsets_bytes, strided_offsets_bytes, WARP_SIZE


def test_contiguous_fp32_warp_is_one_transaction():
    # block_start_offset chosen warp-aligned (multiple of WARP_SIZE elements),
    # matching how tl.arange(0, BLOCK_SIZE) tiles align within a program.
    addrs = warp_offsets_bytes(block_start_offset=1024, warp_lane_ids=range(WARP_SIZE), element_bytes=4)
    assert count_transactions(addrs) == 1


def test_strided_gather_costs_32_transactions():
    addrs = strided_offsets_bytes(base_offset=0, warp_lane_ids=range(WARP_SIZE), stride_elements=32, element_bytes=4)
    assert count_transactions(addrs) == WARP_SIZE


def test_coalesced_bandwidth_advantage_matches_transaction_ratio():
    contiguous = warp_offsets_bytes(0, range(WARP_SIZE), element_bytes=4)
    strided = strided_offsets_bytes(0, range(WARP_SIZE), stride_elements=32, element_bytes=4)
    # Effective bandwidth divides by (roughly) the transaction count ratio.
    assert count_transactions(strided) / count_transactions(contiguous) == WARP_SIZE


def test_misaligned_but_still_contiguous_may_straddle_two_segments():
    # Off-by-a-few-bytes contiguous access can straddle a 128-byte boundary --
    # still far cheaper than full scatter, but not always exactly 1.
    addrs = warp_offsets_bytes(block_start_offset=1023, warp_lane_ids=range(WARP_SIZE), element_bytes=4)
    assert count_transactions(addrs) in (1, 2)
    assert count_transactions(addrs) < WARP_SIZE
