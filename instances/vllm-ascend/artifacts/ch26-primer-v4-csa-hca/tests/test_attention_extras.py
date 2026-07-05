import numpy as np
import pytest

from attention_extras import (
    apply_output_relative_rope,
    apply_partial_rope,
    attention_sink_scores,
    rms_norm,
    sink_absorbed_mass,
    sliding_window_recent_kv,
)


def test_rms_norm_unit_rms():
    x = np.array([3.0, 4.0])
    out = rms_norm(x, eps=0.0)
    # RMS(out) 应该是 1(在 eps=0 时严格成立)
    rms = np.sqrt(np.mean(out ** 2))
    assert rms == pytest.approx(1.0, rel=1e-6)


def test_rms_norm_applies_weight():
    x = np.array([1.0, 1.0])
    weight = np.array([2.0, 3.0])
    out = rms_norm(x, weight=weight, eps=0.0)
    unweighted = rms_norm(x, eps=0.0)
    np.testing.assert_allclose(out, unweighted * weight)


def test_apply_partial_rope_leaves_head_untouched():
    vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = apply_partial_rope(vec, position=5.0, rope_dims=4)
    np.testing.assert_allclose(out[:2], vec[:2])   # 前 d-rope_dims 维不变
    assert out.shape == vec.shape


def test_apply_partial_rope_preserves_norm_of_rotated_part():
    vec = np.array([1.0, 2.0, 3.0, 4.0])
    out = apply_partial_rope(vec, position=3.0, rope_dims=4)
    # RoPE 是正交旋转,范数不变
    assert np.linalg.norm(out) == pytest.approx(np.linalg.norm(vec), rel=1e-6)


def test_apply_partial_rope_rejects_rope_dims_too_large():
    vec = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        apply_partial_rope(vec, position=1.0, rope_dims=4)


def test_apply_partial_rope_zero_position_is_identity():
    vec = np.array([1.0, 2.0, 3.0, 4.0])
    out = apply_partial_rope(vec, position=0.0, rope_dims=4)
    np.testing.assert_allclose(out, vec, atol=1e-10)


def test_apply_output_relative_rope_uses_negative_position():
    vec = np.array([1.0, 2.0, 3.0, 4.0])
    out_neg = apply_output_relative_rope(vec, query_position=7.0, rope_dims=4)
    out_direct = apply_partial_rope(vec, position=-7.0, rope_dims=4)
    np.testing.assert_allclose(out_neg, out_direct)


def test_sliding_window_recent_kv_basic():
    seq = np.arange(10).reshape(10, 1)
    window = sliding_window_recent_kv(seq, query_pos=5, n_win=3)
    np.testing.assert_array_equal(window.flatten(), [3, 4, 5])


def test_sliding_window_recent_kv_clamped_at_sequence_start():
    seq = np.arange(10).reshape(10, 1)
    window = sliding_window_recent_kv(seq, query_pos=1, n_win=5)
    # 只能取到 token 0,1(不能越界到负下标)
    np.testing.assert_array_equal(window.flatten(), [0, 1])


def test_attention_sink_scores_sum_less_than_one_when_sink_dominates():
    logits = np.array([1.0, 1.0, 1.0])
    scores = attention_sink_scores(logits, sink_logit=10.0)   # sink logit 远大于其余
    assert np.sum(scores) < 0.1   # 绝大部分注意力质量被 sink 吸收
    assert sink_absorbed_mass(scores) > 0.9


def test_attention_sink_scores_reduces_to_plain_softmax_when_sink_negligible():
    logits = np.array([1.0, 2.0, 3.0])
    scores = attention_sink_scores(logits, sink_logit=-1000.0)
    plain = np.exp(logits - np.max(logits))
    plain = plain / np.sum(plain)
    np.testing.assert_allclose(scores, plain, atol=1e-6)


def test_sink_absorbed_mass_is_complement_of_score_sum():
    scores = np.array([0.2, 0.3])
    assert sink_absorbed_mass(scores) == pytest.approx(0.5)
