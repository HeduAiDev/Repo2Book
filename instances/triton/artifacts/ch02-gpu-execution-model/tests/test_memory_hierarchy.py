"""ch02 -- memory-hierarchy latency ladder (real order-of-magnitude numbers)."""
import pytest

from memory_hierarchy import latency_cycles, slower_than, LEVEL_ORDER


def test_ladder_is_monotonically_increasing_in_the_low_bound():
    lows = [latency_cycles(level)[0] for level in LEVEL_ORDER]
    assert lows == sorted(lows)
    assert lows[0] < lows[-1]


def test_global_is_two_orders_of_magnitude_slower_than_register():
    reg_lo, _ = latency_cycles("register")
    glb_lo, _glb_hi = latency_cycles("global")
    assert glb_lo / reg_lo >= 100


def test_shared_memory_sits_strictly_between_register_and_l2():
    reg_lo, _ = latency_cycles("register")
    smem_lo, smem_hi = latency_cycles("shared")
    l2_lo, _ = latency_cycles("l2")
    assert reg_lo < smem_lo
    assert smem_hi < l2_lo


def test_slower_than_orders_the_hierarchy():
    assert slower_than("global", "register")
    assert slower_than("l2", "shared")
    assert not slower_than("register", "global")
    assert not slower_than("register", "register")


def test_unknown_level_raises():
    with pytest.raises(ValueError):
        latency_cycles("nonexistent")
