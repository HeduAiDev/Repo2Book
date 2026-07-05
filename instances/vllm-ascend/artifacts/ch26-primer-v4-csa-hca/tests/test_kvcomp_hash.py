import numpy as np
import pytest

from kvcomp_hash import (
    HashEncoder,
    KVCompConfig,
    KVCompMetaData,
    chunk_representative,
    hamming_distance_packed,
    must_select_indices_for,
    qr_orthogonal_random_weights,
    select_topk_blocks_by_hamming,
    unpack_hash,
)


def test_kvcomp_config_defaults():
    cfg = KVCompConfig()
    assert cfg.chunk_size == 128
    assert cfg.chunk_repre_method == "max"
    assert cfg.must_select_blocks == [0, -2, -1]


def test_chunk_representative_max_min_sum():
    chunk = np.array([[1.0, 5.0], [3.0, 2.0], [0.0, 4.0]])
    np.testing.assert_allclose(chunk_representative(chunk, "max"), [3.0, 5.0])
    np.testing.assert_allclose(chunk_representative(chunk, "min"), [0.0, 2.0])
    np.testing.assert_allclose(chunk_representative(chunk, "sum"), [4.0, 11.0])


def test_chunk_representative_rejects_unknown_method():
    chunk = np.zeros((2, 2))
    with pytest.raises(ValueError):
        chunk_representative(chunk, "average")


def test_qr_orthogonal_random_weights_are_orthonormal_columns():
    W = qr_orthogonal_random_weights(dim_in=16, hash_bits=8, seed=0)
    assert W.shape == (16, 8)
    gram = W.T @ W
    np.testing.assert_allclose(gram, np.eye(8), atol=1e-8)


def test_qr_orthogonal_random_weights_rejects_dim_in_smaller_than_hash_bits():
    with pytest.raises(ValueError):
        qr_orthogonal_random_weights(dim_in=4, hash_bits=8, seed=0)


def test_hash_encoder_output_shape_and_dtype():
    W = qr_orthogonal_random_weights(dim_in=16, hash_bits=16, seed=1)
    enc = HashEncoder(hash_weights=W)
    x = np.random.default_rng(2).normal(size=(5, 16))
    codes = enc.compute_hash(x)
    assert codes.shape == (5, 2)   # hash_bits=16 -> 2 bytes
    assert codes.dtype == np.uint8


def test_hash_encoder_rejects_hash_bits_not_multiple_of_8():
    W = np.random.default_rng(0).normal(size=(10, 5))
    with pytest.raises(ValueError):
        HashEncoder(hash_weights=W)


def test_hash_encoder_identical_inputs_produce_identical_codes():
    W = qr_orthogonal_random_weights(dim_in=8, hash_bits=8, seed=3)
    enc = HashEncoder(hash_weights=W)
    x = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0])
    c1 = enc.compute_hash(x)
    c2 = enc.compute_hash(x)
    np.testing.assert_array_equal(c1, c2)


def test_unpack_hash_recovers_sign_pattern():
    W = np.eye(8)   # 8 维恒等投影,方便直接从符号预测哈希码
    enc = HashEncoder(hash_weights=W)
    x = np.array([1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0])
    codes = enc.compute_hash(x)
    signs = unpack_hash(codes, hash_bits=8)
    expected = np.where(x > 0, 1, -1)
    np.testing.assert_array_equal(signs, expected)


def test_hamming_distance_packed_zero_for_identical_codes():
    a = np.array([0b10110010], dtype=np.uint8)
    assert hamming_distance_packed(a, a) == 0


def test_hamming_distance_packed_counts_differing_bits():
    a = np.array([0b00000000], dtype=np.uint8)
    b = np.array([0b00000111], dtype=np.uint8)
    assert hamming_distance_packed(a, b) == 3


def test_must_select_indices_for_sink_and_recent():
    idx = must_select_indices_for(sink=1, recent=4, num_blocks=10)
    assert idx == [0, 6, 7, 8, 9]


def test_must_select_indices_for_handles_small_num_blocks():
    idx = must_select_indices_for(sink=1, recent=4, num_blocks=3)
    assert idx == [0, 1, 2]   # sink+recent 覆盖全部块,去重后就是全部下标


def test_select_topk_blocks_by_hamming_forces_must_select():
    rng = np.random.default_rng(0)
    num_blocks, n_bytes = 8, 2
    key_hashes = rng.integers(0, 256, size=(num_blocks, n_bytes), dtype=np.uint8)
    query_hash = key_hashes[3].copy()   # 让块 3 是最相似的(距离 0)
    selected = select_topk_blocks_by_hamming(query_hash, key_hashes, top_k=1,
                                              must_select_blocks=[0, -1])
    # top_k=1 应该选中块 3(距离最小);must_select_blocks=[0,-1] 强制并入块 0 与块 7
    assert set(selected.tolist()) == {0, 3, 7}


def test_select_topk_blocks_by_hamming_no_must_select():
    key_hashes = np.array([[0b11111111], [0b00000000], [0b11111110]], dtype=np.uint8)
    query_hash = np.array([0b00000000], dtype=np.uint8)
    selected = select_topk_blocks_by_hamming(query_hash, key_hashes, top_k=1)
    assert list(selected) == [1]   # 块1(全0)与 query 距离为0,最相似


def test_kvcomp_metadata_defaults():
    cfg = KVCompConfig()
    meta = KVCompMetaData(kvcomp_config=cfg)
    assert meta.sink == 1
    assert meta.recent == 4
