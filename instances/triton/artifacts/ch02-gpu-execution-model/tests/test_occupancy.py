"""ch02 -- occupancy: reproduces the chapter's Ampere-class worked example
(64K registers/SM, 2048 max resident threads/SM = 64 warps):
32 regs/thread -> 100%, 64 regs/thread -> 50%, 128 regs/thread -> 25%.
"""
import pytest

from occupancy import occupancy, occupancy_by_registers


@pytest.mark.parametrize(
    "regs_per_thread,expected_occupancy",
    [
        (32, 1.0),
        (64, 0.5),
        (128, 0.25),
    ],
)
def test_register_limited_occupancy_matches_book_worked_example(regs_per_thread, expected_occupancy):
    assert occupancy_by_registers(regs_per_thread) == pytest.approx(expected_occupancy)


def test_doubling_registers_per_thread_halves_resident_warps():
    occ_32 = occupancy_by_registers(32)
    occ_64 = occupancy_by_registers(64)
    assert occ_64 == pytest.approx(occ_32 / 2)


def test_combined_occupancy_with_non_limiting_shared_memory():
    # vector-add's BLOCK_SIZE=1024, no shared memory used.
    result = occupancy(registers_per_thread=32, threads_per_block=1024, shared_mem_per_block_bytes=0)
    assert result["occupancy"] == pytest.approx(1.0)
    assert result["limiting_factor"] == "tie"


def test_shared_memory_can_become_the_binding_gate():
    # threads_per_block=1024 lets 2 blocks/SM fit under the thread cap; but
    # if shared memory per block leaves room for only 1 block, that gate
    # binds first even though registers alone would allow full occupancy.
    result = occupancy(
        registers_per_thread=32,
        threads_per_block=1024,
        shared_mem_per_block_bytes=100 * 1024,
    )
    assert result["limiting_factor"] == "shared_memory"
    assert result["occupancy"] == pytest.approx(0.5)
    assert result["occupancy_by_registers"] == pytest.approx(1.0)


def test_occupancy_by_registers_rejects_nonpositive():
    with pytest.raises(ValueError):
        occupancy_by_registers(0)
