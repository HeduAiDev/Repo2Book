"""Tests for memory_profiling.py — GPU memory profiling and KV cache config."""

import math
import pytest
from implementation.memory_profiling import (
    MemoryBudget,
    KVCacheConfig,
    profile_gpu_memory,
    compute_kv_cache_config,
    format_bytes,
    cdiv,
)


# ─── cdiv ────────────────────────────────────────────────────────────────────

def test_cdiv_exact_division():
    """Ceiling division with exact quotient returns that quotient."""
    assert cdiv(10, 2) == 5
    assert cdiv(100, 10) == 10
    assert cdiv(0, 5) == 0


def test_cdiv_non_exact_division():
    """Ceiling division rounds up to the next integer."""
    assert cdiv(10, 3) == 4
    assert cdiv(7, 2) == 4
    assert cdiv(1, 2) == 1


# ─── format_bytes ────────────────────────────────────────────────────────────

def test_format_bytes_gib():
    """Bytes >= 1 GiB format with GiB suffix."""
    result = format_bytes(3 * 1024**3)
    assert "GiB" in result
    assert result.startswith("3.00")


def test_format_bytes_mib():
    """Bytes in MiB range format with MiB suffix."""
    result = format_bytes(5 * 1024**2)
    assert "MiB" in result


def test_format_bytes_kib():
    """Bytes in KiB range format with KiB suffix."""
    result = format_bytes(512 * 1024)
    assert "KiB" in result


def test_format_bytes_raw():
    """Bytes < 1024 format as plain bytes."""
    result = format_bytes(512)
    assert "B" in result


# ─── profile_gpu_memory ──────────────────────────────────────────────────────

def test_profile_gpu_memory_basic():
    """Basic profiling computes weights_memory = params * dtype_bytes."""
    budget = profile_gpu_memory(
        total_gpu_memory=16 * 1024**3,
        gpu_memory_utilization=0.92,
        model_params_count=1_000_000_000,
        dtype_bytes=2,
    )
    assert budget.weights_memory == 2_000_000_000
    assert budget.total_gpu_memory == 16 * 1024**3
    assert budget.gpu_memory_utilization == 0.92


def test_profile_gpu_memory_requested():
    """Requested memory = total * utilization (integer truncated)."""
    total = 15 * 1024**3
    for util in [0.5, 0.92, 1.0]:
        budget = profile_gpu_memory(
            total_gpu_memory=total,
            gpu_memory_utilization=util,
            model_params_count=1_000_000,
        )
        assert budget.available_kv_cache_memory <= int(total * util)
        assert budget.available_kv_cache_memory >= 0


def test_profile_gpu_memory_peak_activation():
    """Peak activation = layers * max_batched * 12 * hidden * dtype."""
    budget = profile_gpu_memory(
        total_gpu_memory=16 * 1024**3,
        gpu_memory_utilization=0.92,
        model_params_count=1_000_000_000,
        dtype_bytes=2,
        max_num_batched_tokens=8192,
        num_layers=32,
        hidden_size=4096,
    )
    expected_peak = 32 * 8192 * 12 * 4096 * 2
    assert budget.peak_activation_memory == expected_peak


def test_profile_gpu_memory_zero_params():
    """Zero parameters → zero weights, but peak_activation still applies."""
    budget = profile_gpu_memory(
        total_gpu_memory=16 * 1024**3,
        gpu_memory_utilization=0.92,
        model_params_count=0,
    )
    assert budget.weights_memory == 0


def test_profile_gpu_memory_available_zero_when_overcommitted():
    """available_kv_cache_memory is clamped to 0 when non-KV exceeds requested."""
    budget = profile_gpu_memory(
        total_gpu_memory=1 * 1024**3,
        gpu_memory_utilization=0.5,
        model_params_count=2_000_000_000,    # 4 GiB weights alone
        dtype_bytes=2,
    )
    assert budget.available_kv_cache_memory == 0


def test_profile_gpu_memory_cudagraph_and_non_torch():
    """cudagraph_memory and non_torch_memory subtract from available KV cache.

    Use a large total GPU memory to ensure available_kv_cache stays positive
    (so the clamping does not hide the subtraction).
    """
    # Large GPU so available_kv_cache > 0 even with all overheads included
    total = 40 * 1024**3  # 40 GiB GPU
    budget_no_extras = profile_gpu_memory(
        total_gpu_memory=total,
        gpu_memory_utilization=0.92,
        model_params_count=1_000_000_000,
        cudagraph_memory=0,
        non_torch_memory=0,
    )
    budget_with_extras = profile_gpu_memory(
        total_gpu_memory=total,
        gpu_memory_utilization=0.92,
        model_params_count=1_000_000_000,
        cudagraph_memory=500 * 1024**2,
        non_torch_memory=300 * 1024**2,
    )
    diff = budget_no_extras.available_kv_cache_memory - budget_with_extras.available_kv_cache_memory
    expected_diff = (500 + 300) * 1024**2
    assert diff == expected_diff


def test_profile_gpu_memory_repr():
    """MemoryBudget.__repr__ includes all fields."""
    budget = profile_gpu_memory(
        total_gpu_memory=16 * 1024**3,
        gpu_memory_utilization=0.92,
        model_params_count=1_000_000_000,
    )
    s = repr(budget)
    assert "total_gpu_memory" in s
    assert "weights_memory" in s
    assert "available_kv_cache_memory" in s


# ─── compute_kv_cache_config ─────────────────────────────────────────────────

def test_compute_kv_cache_config_page_size():
    """page_size_bytes = 2 * block_size * num_kv_heads * head_size * dtype_bytes."""
    budget = MemoryBudget(
        total_gpu_memory=16 * 1024**3,
        gpu_memory_utilization=0.92,
        weights_memory=2_000_000_000,
        peak_activation_memory=4_000_000_000,
        cudagraph_memory=100_000_000,
        non_torch_memory=200_000_000,
        available_kv_cache_memory=5 * 1024**3,
    )
    config = compute_kv_cache_config(
        budget,
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        dtype_bytes=2,
    )
    expected_page_size = 2 * 16 * 8 * 128 * 2
    assert config.page_size_bytes == expected_page_size


def test_compute_kv_cache_config_num_blocks():
    """num_blocks = available_kv_cache // page_size // num_layers."""
    # Configure so that available_kv_cache = page_size * num_layers * 10
    page_size = 2 * 16 * 8 * 128 * 2  # = 65536
    num_layers = 32
    available_kv_cache = page_size * num_layers * 10  # Exactly 10 blocks

    budget = MemoryBudget(
        total_gpu_memory=16 * 1024**3,
        gpu_memory_utilization=0.92,
        weights_memory=2_000_000_000,
        peak_activation_memory=4_000_000_000,
        cudagraph_memory=0,
        non_torch_memory=0,
        available_kv_cache_memory=available_kv_cache,
    )
    config = compute_kv_cache_config(
        budget, block_size=16, num_layers=num_layers,
        num_kv_heads=8, head_size=128, dtype_bytes=2,
    )
    assert config.num_blocks == 10


def test_compute_kv_cache_config_zero_available():
    """Zero available KV cache → zero blocks."""
    budget = MemoryBudget(
        total_gpu_memory=16 * 1024**3,
        gpu_memory_utilization=0.92,
        weights_memory=16_000_000_000,
        peak_activation_memory=0,
        cudagraph_memory=0,
        non_torch_memory=0,
        available_kv_cache_memory=0,
    )
    config = compute_kv_cache_config(budget)
    assert config.num_blocks == 0


def test_compute_kv_cache_config_fields_preserved():
    """All config fields are carried through correctly."""
    budget = MemoryBudget(
        total_gpu_memory=16 * 1024**3,
        gpu_memory_utilization=0.92,
        weights_memory=2_000_000_000,
        peak_activation_memory=4_000_000_000,
        cudagraph_memory=0,
        non_torch_memory=0,
        available_kv_cache_memory=5 * 1024**3,
    )
    config = compute_kv_cache_config(
        budget, block_size=32, num_layers=24,
        num_kv_heads=4, head_size=64, dtype_bytes=1,
    )
    assert config.block_size == 32
    assert config.num_layers == 24
    assert config.num_kv_heads == 4
    assert config.head_size == 64
    assert config.dtype_bytes == 1


def test_compute_kv_cache_config_zero_blocks_from_insufficient_memory():
    """When available_memory < one page_size * num_layers, num_blocks is 0."""
    page_size = 2 * 16 * 8 * 128 * 2
    num_layers = 32
    min_memory = page_size * num_layers
    budget = MemoryBudget(
        total_gpu_memory=16 * 1024**3,
        gpu_memory_utilization=0.92,
        weights_memory=2_000_000_000,
        peak_activation_memory=4_000_000_000,
        cudagraph_memory=0,
        non_torch_memory=0,
        available_kv_cache_memory=min_memory - 1,
    )
    config = compute_kv_cache_config(
        budget, block_size=16, num_layers=num_layers,
        num_kv_heads=8, head_size=128, dtype_bytes=2,
    )
    assert config.num_blocks == 0
