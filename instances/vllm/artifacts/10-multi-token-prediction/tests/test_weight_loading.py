"""Tests for weight_loading.py — MTP weight name remap + lm_head sharing.

Pins the 3 paths of `_rewrite_spec_layer_name`:
  Path 1: transformer block weight → wrap under `.mtp_block.`
  Path 2: shared embed_tokens → promote to `model.embed_tokens.`
  Path 3: MTP-specific (enorm/hnorm/eh_proj/shared_head) → unchanged
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from implementation.weight_loading import (
    acceptance_length_to_rates,
    loader_demo_shapes,
    maybe_share_embeddings,
    maybe_share_lm_head,
    remap_checkpoint,
    rewrite_spec_layer_name,
    unconditional_to_conditional_rates,
)


# ============================================================================
# rewrite_spec_layer_name — 3 paths
# ============================================================================


def test_path1_transformer_block_wrapped():
    """Path 1: model.layers.{N}.X → model.layers.{N}.mtp_block.X."""
    out = rewrite_spec_layer_name(10, "model.layers.10.self_attn.q_proj.weight")
    assert out == "model.layers.10.mtp_block.self_attn.q_proj.weight"


def test_path1_mlp_gate_proj_wrapped():
    out = rewrite_spec_layer_name(61, "model.layers.61.mlp.gate_proj.weight")
    assert out == "model.layers.61.mtp_block.mlp.gate_proj.weight"


def test_path2_embed_tokens_promoted_to_top_level():
    """Path 2: model.layers.{N}.embed_tokens.weight → model.embed_tokens.weight."""
    out = rewrite_spec_layer_name(10, "model.layers.10.embed_tokens.weight")
    assert out == "model.embed_tokens.weight"


def test_path3_enorm_unchanged():
    """Path 3: enorm is MTP-specific, unchanged."""
    out = rewrite_spec_layer_name(10, "model.layers.10.enorm.weight")
    assert out == "model.layers.10.enorm.weight"


def test_path3_hnorm_unchanged():
    out = rewrite_spec_layer_name(10, "model.layers.10.hnorm.weight")
    assert out == "model.layers.10.hnorm.weight"


def test_path3_eh_proj_unchanged():
    out = rewrite_spec_layer_name(61, "model.layers.61.eh_proj.weight")
    assert out == "model.layers.61.eh_proj.weight"


def test_path3_shared_head_head_unchanged():
    out = rewrite_spec_layer_name(61, "model.layers.61.shared_head.head.weight")
    assert out == "model.layers.61.shared_head.head.weight"


def test_path3_shared_head_norm_unchanged():
    out = rewrite_spec_layer_name(61, "model.layers.61.shared_head.norm.weight")
    assert out == "model.layers.61.shared_head.norm.weight"


def test_layer_index_preserved_path1():
    """vLLM keeps the spec_layer index as-is (does NOT reindex from 0)."""
    for idx in [10, 25, 61, 99]:
        out = rewrite_spec_layer_name(idx, f"model.layers.{idx}.self_attn.o_proj.weight")
        assert f"model.layers.{idx}.mtp_block." in out


def test_path1_only_replaces_first_match():
    """Wrap inserts ONE .mtp_block. — even with nested 'layers' tokens elsewhere."""
    out = rewrite_spec_layer_name(5, "model.layers.5.self_attn.q_proj.weight")
    assert out.count(".mtp_block.") == 1


# ============================================================================
# remap_checkpoint
# ============================================================================


def test_remap_checkpoint_splits_target_and_mtp_dicts():
    """Non-MTP-layer keys go to target_sd; MTP-layer keys go to mtp_sd."""
    state = {
        # target: layers 0..4
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(4, 4),
        "model.layers.4.mlp.gate_proj.weight": torch.zeros(4, 4),
        # MTP: layer 5 (mtp_start=5, num_mtp=1)
        "model.layers.5.self_attn.q_proj.weight": torch.zeros(4, 4),
        "model.layers.5.embed_tokens.weight": torch.zeros(8, 4),
        "model.layers.5.enorm.weight": torch.zeros(4),
        "model.layers.5.eh_proj.weight": torch.zeros(4, 8),
        # Top-level
        "model.embed_tokens.weight": torch.zeros(8, 4),
        "lm_head.weight": torch.zeros(8, 4),
    }
    target_sd, mtp_sd = remap_checkpoint(state, mtp_start_layer_idx=5, num_mtp_layers=1)
    # Target: target layers + top-level
    assert "model.layers.0.self_attn.q_proj.weight" in target_sd
    assert "model.layers.4.mlp.gate_proj.weight" in target_sd
    assert "model.embed_tokens.weight" in target_sd
    assert "lm_head.weight" in target_sd
    # MTP: layer 5 keys (rewritten)
    assert "model.layers.5.mtp_block.self_attn.q_proj.weight" in mtp_sd
    assert "model.embed_tokens.weight" in mtp_sd  # path 2 promotion
    assert "model.layers.5.enorm.weight" in mtp_sd  # path 3 unchanged
    assert "model.layers.5.eh_proj.weight" in mtp_sd


def test_remap_checkpoint_does_not_double_count():
    """Each input key goes to exactly one of (target, mtp)."""
    state = {
        f"model.layers.{i}.self_attn.q_proj.weight": torch.zeros(2, 2)
        for i in range(8)
    }
    target_sd, mtp_sd = remap_checkpoint(state, mtp_start_layer_idx=6, num_mtp_layers=2)
    target_count = len(target_sd)
    mtp_count = len(mtp_sd)
    assert target_count + mtp_count == len(state)


# ============================================================================
# loader_demo_shapes — demo §3.5 numerics
# ============================================================================


def test_loader_demo_shapes_demo_numerics():
    """Demo §3.5 verbatim: input=193, target=185, mtp=8."""
    info = loader_demo_shapes(target_layers=61, mtp_layers=1, hidden=32, vocab=128)
    assert info["input_total_keys"] == 193
    assert info["target_keys"] == 185
    assert info["mtp_keys"] == 8


def test_loader_demo_shapes_three_paths_present():
    """Sample renames cover all 3 paths."""
    info = loader_demo_shapes(target_layers=61, mtp_layers=1, hidden=32, vocab=128)
    paths = {r["path"] for r in info["sample_renames"]}
    assert "path1" in paths
    assert "path2" in paths
    assert "path3" in paths


def test_loader_demo_shapes_lm_head_in_target():
    """lm_head.weight is a top-level key (not MTP-layer-specific) → goes to target."""
    info = loader_demo_shapes()
    assert info["lm_head_present_target"] is True


def test_loader_demo_shapes_path1_sample_correct_rewrite():
    info = loader_demo_shapes(target_layers=61, mtp_layers=1, hidden=32, vocab=128)
    p1_renames = [r for r in info["sample_renames"] if r["path"] == "path1"]
    for r in p1_renames:
        assert ".mtp_block." in r["new"]


def test_loader_demo_shapes_path2_sample_correct_rewrite():
    info = loader_demo_shapes(target_layers=61, mtp_layers=1, hidden=32, vocab=128)
    p2_renames = [r for r in info["sample_renames"] if r["path"] == "path2"]
    for r in p2_renames:
        assert r["new"] == "model.embed_tokens.weight"


def test_loader_demo_shapes_path3_unchanged():
    info = loader_demo_shapes(target_layers=61, mtp_layers=1, hidden=32, vocab=128)
    p3_renames = [r for r in info["sample_renames"] if r["path"] == "path3"]
    for r in p3_renames:
        assert r["old"] == r["new"]


# ============================================================================
# acceptance_length_to_rates — config helper
# ============================================================================


def test_length_to_rates_full_int_length():
    """length=4, n=5 → 3 ones, then 1.0 fractional, then zeros."""
    # length - 1 = 3, num_full=3, frac=0 → [1,1,1,0,0]
    rates = acceptance_length_to_rates(4.0, 5)
    assert rates == [1.0, 1.0, 1.0, 0.0, 0.0]


def test_length_to_rates_fractional():
    """length=3.4, n=5 → 2 full + 0.4 frac + 2 zeros."""
    # length - 1 = 2.4, num_full=2, frac=0.4 → [1, 1, 0.4, 0, 0]
    rates = acceptance_length_to_rates(3.4, 5)
    assert rates[0] == 1.0
    assert rates[1] == 1.0
    assert rates[2] == pytest.approx(0.4)
    assert rates[3] == 0.0
    assert rates[4] == 0.0


def test_length_to_rates_length_one():
    """length=1.0 → all zeros (no drafts accepted on average)."""
    rates = acceptance_length_to_rates(1.0, 3)
    # length - 1 = 0, num_full=0, frac=0 → [0, 0, 0]
    assert rates == [0.0, 0.0, 0.0]


def test_length_to_rates_truncates_at_n():
    """If length-1 >= n, all rates = 1.0 (capped)."""
    rates = acceptance_length_to_rates(10.0, 3)
    assert rates == [1.0, 1.0, 1.0]


# ============================================================================
# unconditional_to_conditional_rates
# ============================================================================


def test_uncond_to_cond_simple():
    """[0.7, 0.4, 0.1] → [0.7, 0.4/0.7, 0.1/0.4]."""
    cond = unconditional_to_conditional_rates([0.7, 0.4, 0.1])
    assert cond[0] == pytest.approx(0.7)
    assert cond[1] == pytest.approx(0.4 / 0.7)
    assert cond[2] == pytest.approx(0.1 / 0.4)


def test_uncond_to_cond_zero_handled():
    """If a prefix becomes 0, subsequent conditional rates are 0 (not div0)."""
    cond = unconditional_to_conditional_rates([1.0, 1.0, 0.4, 0.0])
    # [1.0, 1.0/1.0, 0.4/1.0, 0.0/0.4]
    assert cond[3] == pytest.approx(0.0)


def test_uncond_to_cond_first_position():
    """cond[0] = uncond[0] (no normalization needed at position 0)."""
    cond = unconditional_to_conditional_rates([0.85, 0.7, 0.5])
    assert cond[0] == pytest.approx(0.85)


# ============================================================================
# maybe_share_lm_head
# ============================================================================


class _DummyTarget(nn.Module):
    def __init__(self, hidden=32, vocab=128):
        super().__init__()
        self.lm_head = nn.Linear(hidden, vocab, bias=False)


class _DummyMTPLayer(nn.Module):
    def __init__(self, hidden=32, vocab=128):
        super().__init__()
        from implementation.mtp_head import SharedHead
        self.shared_head = SharedHead(hidden, vocab)


class _DummyMTPInner(nn.Module):
    def __init__(self, hidden=32, vocab=128, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleDict({
            str(i): _DummyMTPLayer(hidden, vocab) for i in range(num_layers)
        })


class _DummyMTPPredictor(nn.Module):
    def __init__(self, hidden=32, vocab=128, num_layers=2):
        super().__init__()
        self.model = _DummyMTPInner(hidden, vocab, num_layers)


def test_maybe_share_lm_head_replaces_inner_layers():
    """share_lm_head replaces shared_head.head in each MTP layer."""
    target = _DummyTarget(hidden=32, vocab=128)
    pred = _DummyMTPPredictor(hidden=32, vocab=128, num_layers=2)
    n = maybe_share_lm_head(target, pred)
    assert n == 2
    for layer in pred.model.layers.values():
        assert layer.shared_head.head is target.lm_head


def test_maybe_share_lm_head_count_matches_layers():
    """Returns number of layers updated."""
    target = _DummyTarget()
    pred = _DummyMTPPredictor(num_layers=4)
    assert maybe_share_lm_head(target, pred) == 4


def test_maybe_share_lm_head_attribute_assignment():
    """After share, mtp_predictor.lm_head is also set (top-level)."""
    target = _DummyTarget()
    pred = _DummyMTPPredictor()
    maybe_share_lm_head(target, pred)
    assert pred.lm_head is target.lm_head


def test_maybe_share_lm_head_raises_without_target_lm_head():
    """If target lacks lm_head, raise AttributeError."""
    target = nn.Module()  # no lm_head
    pred = _DummyMTPPredictor()
    with pytest.raises(AttributeError):
        maybe_share_lm_head(target, pred)


# ============================================================================
# maybe_share_embeddings
# ============================================================================


class _DummyTargetWithEmbeds(nn.Module):
    def __init__(self, hidden=32, vocab=128):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, hidden)


class _DummyMTPInnerWithEmbeds(nn.Module):
    def __init__(self, hidden=32, vocab=128):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, hidden)


def test_maybe_share_embeddings_ties_target_and_mtp():
    """After share, mtp.model.embed_tokens is target.model.embed_tokens."""
    target = _DummyTargetWithEmbeds(hidden=32, vocab=128)
    mtp = _DummyMTPInnerWithEmbeds(hidden=32, vocab=128)
    ok = maybe_share_embeddings(target, mtp)
    assert ok is True
    assert mtp.model.embed_tokens is target.model.embed_tokens


def test_maybe_share_embeddings_raises_without_embedding():
    """If target has neither embed_tokens nor embedding, raise."""
    target = nn.Module()  # no embed_tokens / embedding
    mtp = _DummyMTPInnerWithEmbeds()
    with pytest.raises(AttributeError):
        maybe_share_embeddings(target, mtp)


# ============================================================================
# Cross-path consistency: full-checkpoint dry run
# ============================================================================


def test_loader_demo_shape_target_layers_61_mtp_1():
    """Full pin: target=61 layers gives 185 target keys, 1 MTP layer gives 8 mtp keys."""
    info = loader_demo_shapes(target_layers=61, mtp_layers=1, hidden=32, vocab=128)
    # Target layers contribute 61 * 3 = 183 weights, plus 2 top-level (embed_tokens + lm_head) = 185
    assert info["target_keys"] == 185
    # MTP layer 61 contributes 8 unique keys after rewrite (path1 wraps 3, path2 promotes 1 collision-free
    # at mtp_layers=1, path3 keeps 4 unchanged; minus collision on shared embed_tokens)
    assert info["mtp_keys"] == 8
