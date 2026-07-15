"""ch02 -- register spill: over-budget registers fall back to local memory,
which collapses latency from the register tier to the global-memory tier.
"""
from register_spill import spilled_registers, effective_access_cycles


def test_no_spill_when_within_budget():
    assert spilled_registers(registers_needed_per_thread=32, register_budget_per_thread=64) == 0


def test_spill_count_when_over_budget():
    assert spilled_registers(registers_needed_per_thread=80, register_budget_per_thread=64) == 16


def test_spilled_access_falls_back_to_global_memory_latency_tier():
    result = effective_access_cycles(registers_needed_per_thread=80, register_budget_per_thread=64)
    assert result["spilled_registers"] == 16
    assert result["resident_registers"] == 64
    assert result["spilled_access_cycles"] == (400, 800)
    assert result["resident_access_cycles"] == (1, 1)
    # Spilling collapses latency by ~2 orders of magnitude relative to a real
    # register access.
    assert result["spill_latency_multiplier"] >= 100


def test_no_spill_reports_zero_spill_cycles():
    result = effective_access_cycles(registers_needed_per_thread=32, register_budget_per_thread=64)
    assert result["spilled_registers"] == 0
    assert result["spilled_access_cycles"] == (0, 0)
    assert result["spill_latency_multiplier"] == 1.0
