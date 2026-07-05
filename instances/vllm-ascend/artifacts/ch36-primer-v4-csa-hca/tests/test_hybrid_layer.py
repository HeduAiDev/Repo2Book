import pytest

from hybrid_layer import (
    LayerSpec,
    build_layer_spec,
    build_model_layers,
    get_dsv4_compress_ratio,
    kv_cache_spec_for_layer,
)


def test_get_dsv4_compress_ratio_reads_array():
    ratios = [4, 4, 4, 128]
    assert get_dsv4_compress_ratio(ratios, 0) == 4
    assert get_dsv4_compress_ratio(ratios, 3) == 128


def test_get_dsv4_compress_ratio_none_config_is_dense():
    assert get_dsv4_compress_ratio(None, 0) == 0


def test_get_dsv4_compress_ratio_out_of_range_is_dense():
    ratios = [4, 4]
    assert get_dsv4_compress_ratio(ratios, 5) == 0


def test_build_layer_spec_csa_has_compressor_and_indexer():
    spec = build_layer_spec(0, 4)
    assert spec.kind == "CSA"
    assert spec.has_compressor is True
    assert spec.has_indexer is True


def test_build_layer_spec_hca_has_compressor_but_not_indexer():
    spec = build_layer_spec(1, 128)
    assert spec.kind == "HCA"
    assert spec.has_compressor is True
    assert spec.has_indexer is False


def test_build_layer_spec_dense_has_neither():
    spec = build_layer_spec(2, 0)
    assert spec.kind == "dense"
    assert spec.has_compressor is False
    assert spec.has_indexer is False


def test_build_layer_spec_rejects_unsupported_ratio():
    with pytest.raises(ValueError):
        build_layer_spec(0, 16)


def test_build_model_layers_interleave():
    ratios = [4, 4, 4, 128, 4, 4, 4, 128]
    layers = build_model_layers(ratios)
    assert len(layers) == 8
    kinds = [l.kind for l in layers]
    assert kinds == ["CSA", "CSA", "CSA", "HCA", "CSA", "CSA", "CSA", "HCA"]
    assert all(isinstance(l, LayerSpec) for l in layers)


def test_kv_cache_spec_for_layer_dense_uses_swa():
    spec = build_layer_spec(0, 0)
    assert kv_cache_spec_for_layer(spec) == "SWA"


def test_kv_cache_spec_for_layer_csa_hca_use_mla_with_ratio():
    csa_spec = build_layer_spec(0, 4)
    hca_spec = build_layer_spec(1, 128)
    assert kv_cache_spec_for_layer(csa_spec) == "MLA(compress_ratio=4)"
    assert kv_cache_spec_for_layer(hca_spec) == "MLA(compress_ratio=128)"
