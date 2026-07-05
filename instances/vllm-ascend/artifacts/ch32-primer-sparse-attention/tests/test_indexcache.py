import numpy as np
import pytest

from indexcache import get_cached_topk_indices, layer_topk_indices


def test_get_cached_topk_indices_returns_buffer_when_present():
    buf = np.array([1, 4, 7])
    assert np.array_equal(get_cached_topk_indices(buf), buf)


def test_get_cached_topk_indices_raises_when_buffer_missing():
    with pytest.raises(RuntimeError, match="IndexCache requires topk_indices_buffer"):
        get_cached_topk_indices(None)


def test_layer_topk_indices_reuses_cache_without_calling_compute_fn():
    buf = np.array([2, 3])
    calls = []

    def compute_fn():
        calls.append(1)
        return np.array([0, 1])

    result = layer_topk_indices(skip_topk=True, topk_indices_buffer=buf, compute_topk_fn=compute_fn)
    assert np.array_equal(result, buf)
    assert calls == []  # the real indexer/top-k path must not run when reusing the cache


def test_layer_topk_indices_computes_when_not_skipping():
    calls = []

    def compute_fn():
        calls.append(1)
        return np.array([5, 6])

    result = layer_topk_indices(skip_topk=False, topk_indices_buffer=None, compute_topk_fn=compute_fn)
    assert np.array_equal(result, np.array([5, 6]))
    assert calls == [1]
