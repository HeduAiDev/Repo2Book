"""ch02 -- SIMT hierarchy: block -> warp(32 lane) partitioning."""
from simt_hierarchy import WARP_SIZE, num_warps_per_block, partition_into_warps


def test_block_size_1024_gives_32_warps():
    # vector-add's BLOCK_SIZE=1024 (python/tutorials/01-vector-add.py:L73).
    assert num_warps_per_block(1024) == 32


def test_partition_into_warps_groups_of_32():
    warps = partition_into_warps(1024)
    assert len(warps) == 32
    assert all(len(w) == WARP_SIZE for w in warps)
    assert warps[0] == list(range(0, 32))
    assert warps[-1] == list(range(992, 1024))


def test_ragged_block_gives_partial_last_warp():
    warps = partition_into_warps(100)
    assert num_warps_per_block(100) == 4
    assert [len(w) for w in warps] == [32, 32, 32, 4]


def test_warps_partition_lanes_without_overlap():
    warps = partition_into_warps(100)
    flat = [lane for warp in warps for lane in warp]
    assert flat == list(range(100))
