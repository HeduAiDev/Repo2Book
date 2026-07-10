import numpy as np

from index_cache import IndexCache


def test_write_then_gather_roundtrips_within_quantization_error():
    # PAPER: arXiv:2606.19348 §2.3.1 "IndexCache 要点" —— indexer 专属缓存写入
    # 后可原样收集回来（有界量化误差，由 quant_block_size 控制粒度）。
    cache = IndexCache(head_dim=4, quant_block_size=2)
    key0 = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    key1 = np.array([0.5, 0.5, -0.5, -0.5], dtype=np.float32)
    cache.write(key0)
    cache.write(key1)

    assert cache.num_entries == 2
    gathered = cache.gather(0, 2)
    assert gathered.shape == (2, 4)
    # 量化引入的误差应有界（block 内 scale = max(|x|)/127，远小于 1e-2）
    np.testing.assert_allclose(gathered[0], key0, atol=1e-1)
    np.testing.assert_allclose(gathered[1], key1, atol=1e-1)


def test_gather_partial_range_only_returns_requested_rows():
    cache = IndexCache(head_dim=2, quant_block_size=2)
    for v in [1.0, 2.0, 3.0]:
        cache.write(np.array([v, v], dtype=np.float32))

    gathered = cache.gather(1, 3)
    assert gathered.shape == (2, 2)


def test_gather_empty_range_returns_empty_array():
    cache = IndexCache(head_dim=2, quant_block_size=2)
    out = cache.gather(0, 0)
    assert out.shape == (0, 2)


def test_write_rejects_wrong_shape():
    cache = IndexCache(head_dim=4, quant_block_size=2)
    try:
        cache.write(np.zeros(3, dtype=np.float32))
        assert False, "should have raised"
    except AssertionError:
        pass


def test_cache_is_independent_of_any_main_kv_object():
    # PAPER: §2.3.1 —— IndexCache 不持有、不引用任何"主 KV cache"对象；
    # 构造与操作全部只涉及自己的存储。
    cache = IndexCache(head_dim=2, quant_block_size=2)
    assert not hasattr(cache, "main_kv_cache")
    assert set(vars(cache).keys()) == {
        "head_dim",
        "quant_block_size",
        "_entries",
        "_scales",
    }
