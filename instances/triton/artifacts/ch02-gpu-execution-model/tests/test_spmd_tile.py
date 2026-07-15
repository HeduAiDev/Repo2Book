"""ch02 -- SPMD tile launch: grid sizing, tile offsets, boundary mask.

Ground truth: python/tutorials/01-vector-add.py's own worked comment
(BLOCK_SIZE tiles are contiguous slices of the input) and dossier mechanism
m08 (the ragged-tail mask example: N=98432, BLOCK_SIZE=1024, tail program
pid=96 masks out 896 of its 1024 lanes).
"""
from spmd_tile import cdiv, tile_offsets, boundary_mask, spmd_grid


def test_cdiv_exact_division():
    assert cdiv(256, 64) == 4


def test_cdiv_ragged_tail_matches_vector_add_worked_example():
    # dossier m07/m08: N=98432, BLOCK_SIZE=1024 -> 97 programs (96 full tiles
    # + 1 ragged tail of 128 valid elements).
    assert cdiv(98432, 1024) == 97


def test_tile_offsets_are_contiguous():
    # python/tutorials/01-vector-add.py:L37-L40's own comment: a 256-length
    # vector with block_size 64 gives tiles [0:64, 64:128, 128:192, 192:256].
    offsets = tile_offsets(program_id=2, block_size=64)
    assert offsets == list(range(128, 192))
    assert all(b - a == 1 for a, b in zip(offsets, offsets[1:]))


def test_tile_offsets_match_vector_add_comment_example():
    expected = [list(range(0, 64)), list(range(64, 128)), list(range(128, 192)), list(range(192, 256))]
    actual = [tile_offsets(pid, block_size=64) for pid in range(4)]
    assert actual == expected


def test_boundary_mask_guards_ragged_tail():
    # dossier m08's exact numbers: tail program pid=96, block_start=98304,
    # offsets 98304..99327; 128 lanes in-bounds, 896 masked out.
    n_elements = 98432
    offsets = tile_offsets(program_id=96, block_size=1024)
    mask = boundary_mask(offsets, n_elements)
    assert sum(mask) == 128
    assert sum(not m for m in mask) == 896


def test_boundary_mask_all_true_for_full_tile():
    offsets = tile_offsets(program_id=0, block_size=1024)
    mask = boundary_mask(offsets, n_elements=98432)
    assert all(mask)


def test_spmd_grid_size_matches_cdiv():
    n_elements, block_size = 256, 64
    grid = spmd_grid(n_elements, block_size)
    assert len(grid) == cdiv(n_elements, block_size) == 4
    assert grid == [0, 1, 2, 3]


def test_tiles_partition_the_full_range_without_overlap():
    n_elements, block_size = 256, 64
    covered = set()
    for pid in spmd_grid(n_elements, block_size):
        offsets = tile_offsets(pid, block_size)
        mask = boundary_mask(offsets, n_elements)
        for off, m in zip(offsets, mask):
            if m:
                assert off not in covered
                covered.add(off)
    assert covered == set(range(n_elements))
