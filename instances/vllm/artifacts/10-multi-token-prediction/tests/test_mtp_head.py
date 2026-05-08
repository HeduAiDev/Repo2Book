"""Tests for mtp_head.py — DeepSeek MTP layer architecture mirror.

Pins Trap E (MTP head is heavy, not lightweight) by exercising:
  - RMSNorm shape/zero/scale-invariance
  - MultiHeadAttention shape/head_dim/qkv shapes
  - DenseFFN shape
  - MTPBlock (full transformer): attn + FFN; FFN dominates attn
  - DeepSeekMultiTokenPredictorLayer: enorm + hnorm + eh_proj fusion + mtp_block
  - SharedHead: norm + lm_head + share_lm_head_with weight tying
  - DeepSeekMultiTokenPredictor stack: forward_one_step, compute_logits, propose_K
  - parameter_count_mtp / parameter_count_medusa demo numerics (verbatim §3.5)
  - MTP/Medusa ratio demo pin (12.91x shared lm, 1.91x separate)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from implementation.mtp_head import (
    DeepSeekMultiTokenPredictor,
    DeepSeekMultiTokenPredictorLayer,
    MTPBlock,
    MTPLayerStats,
    RMSNorm,
    SharedHead,
    _DenseFFN,
    _MultiHeadAttention,
    parameter_count_medusa,
    parameter_count_mtp,
)


# ============================================================================
# RMSNorm
# ============================================================================


def test_rmsnorm_default_weight_is_ones():
    norm = RMSNorm(hidden_size=8)
    assert torch.equal(norm.weight, torch.ones(8))


def test_rmsnorm_default_eps():
    norm = RMSNorm(hidden_size=8)
    assert norm.eps == 1e-6


def test_rmsnorm_custom_eps():
    norm = RMSNorm(hidden_size=8, eps=1e-3)
    assert norm.eps == 1e-3


def test_rmsnorm_forward_preserves_2d_shape():
    norm = RMSNorm(hidden_size=8)
    out = norm(torch.randn(4, 8))
    assert out.shape == (4, 8)


def test_rmsnorm_forward_preserves_3d_shape():
    norm = RMSNorm(hidden_size=16)
    out = norm(torch.randn(2, 4, 16))
    assert out.shape == (2, 4, 16)


def test_rmsnorm_zero_input_zero_output():
    """Zero input → zero output (eps prevents NaN; 0 / sqrt(eps) = 0)."""
    norm = RMSNorm(hidden_size=8)
    out = norm(torch.zeros(4, 8))
    assert torch.allclose(out, torch.zeros(4, 8))


def test_rmsnorm_unit_input_passes_through():
    """[1,1,1,1] has RMS=1 → output ≈ [1,1,1,1] with weight=1."""
    norm = RMSNorm(hidden_size=4)
    out = norm(torch.tensor([[1.0, 1.0, 1.0, 1.0]]))
    assert torch.allclose(out, torch.tensor([[1.0, 1.0, 1.0, 1.0]]), atol=1e-3)


def test_rmsnorm_positive_scale_invariance():
    """RMSNorm(c*x) ≈ RMSNorm(x) for positive c (scale-invariant up to weight)."""
    norm = RMSNorm(hidden_size=4, eps=1e-12)
    v = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    a = norm(v * 0.5)
    b = norm(v * 50.0)
    assert torch.allclose(a, b, atol=1e-3)


# ============================================================================
# MultiHeadAttention
# ============================================================================


def test_mha_forward_shape():
    """MHA expects [T, H], outputs [T, H] (no batch dim — sequence-level)."""
    mha = _MultiHeadAttention(hidden_size=64, num_heads=4)
    x = torch.randn(8, 64)
    out = mha(x)
    assert out.shape == (8, 64)


def test_mha_single_token():
    mha = _MultiHeadAttention(hidden_size=32, num_heads=4)
    out = mha(torch.randn(1, 32))
    assert out.shape == (1, 32)


def test_mha_invalid_num_heads_asserts():
    """num_heads must divide hidden_size."""
    with pytest.raises(AssertionError):
        _MultiHeadAttention(hidden_size=64, num_heads=5)


def test_mha_head_dim():
    mha = _MultiHeadAttention(hidden_size=64, num_heads=4)
    assert mha.head_dim == 16


def test_mha_qkv_shape():
    mha = _MultiHeadAttention(hidden_size=32, num_heads=4)
    assert mha.qkv.weight.shape == (3 * 32, 32)


def test_mha_o_proj_shape():
    mha = _MultiHeadAttention(hidden_size=32, num_heads=4)
    assert mha.o_proj.weight.shape == (32, 32)


# ============================================================================
# DenseFFN (SwiGLU)
# ============================================================================


def test_dense_ffn_shape():
    ffn = _DenseFFN(hidden_size=32, intermediate_size=64)
    out = ffn(torch.randn(4, 32))
    assert out.shape == (4, 32)


def test_dense_ffn_zero_input_returns_zero():
    """SwiGLU(0) = 0 (silu(0)*x = 0)."""
    ffn = _DenseFFN(hidden_size=8, intermediate_size=16)
    out = ffn(torch.zeros(2, 8))
    assert torch.allclose(out, torch.zeros(2, 8))


def test_dense_ffn_param_breakdown():
    """SwiGLU has 3 projections: gate, up, down. Param = 2*h*inter + inter*h."""
    h, inter = 32, 64
    ffn = _DenseFFN(hidden_size=h, intermediate_size=inter)
    expected = 2 * h * inter + inter * h
    actual = sum(p.numel() for p in ffn.parameters())
    assert actual == expected


# ============================================================================
# MTPBlock — full transformer block (Trap E source of evidence)
# ============================================================================


def test_mtp_block_forward_returns_two_tensors():
    """MTPBlock.forward returns (hidden, residual) tuple per source."""
    blk = MTPBlock(hidden_size=64, intermediate_size=128, num_heads=4)
    h, residual = blk(torch.randn(8, 64))
    assert h.shape == (8, 64)
    assert residual.shape == (8, 64)


def test_mtp_block_has_attn_and_ffn():
    """Trap E: MTP block contains MHA + dense FFN, NOT just an MLP."""
    blk = MTPBlock(hidden_size=64, intermediate_size=128, num_heads=4)
    assert hasattr(blk, "attn")
    assert hasattr(blk, "mlp")
    assert hasattr(blk, "input_layernorm")
    assert hasattr(blk, "post_attention_layernorm")


def test_mtp_block_param_count_dominated_by_ffn():
    """At intermediate=4*hidden, FFN params dominate attn params."""
    blk = MTPBlock(hidden_size=64, intermediate_size=256, num_heads=4)
    attn_params = sum(p.numel() for p in blk.attn.parameters())
    ffn_params = sum(p.numel() for p in blk.mlp.parameters())
    assert ffn_params > attn_params


# ============================================================================
# SharedHead
# ============================================================================


def test_shared_head_forward_returns_normalized_not_logits():
    """SharedHead.forward returns RMSNorm output [T, hidden] — NOT logits."""
    sh = SharedHead(hidden_size=32, vocab_size=128)
    out = sh(torch.randn(4, 32))
    assert out.shape == (4, 32)


def test_shared_head_compute_logits_projects_to_vocab():
    """compute_logits returns [T, vocab]."""
    sh = SharedHead(hidden_size=32, vocab_size=128)
    h_norm = torch.randn(4, 32)
    logits = sh.compute_logits(h_norm)
    assert logits.shape == (4, 128)


def test_shared_head_has_norm_and_head_attributes():
    sh = SharedHead(hidden_size=32, vocab_size=128)
    assert hasattr(sh, "norm")
    assert hasattr(sh, "head")


def test_shared_head_share_lm_head_with_ties_weights():
    """share_lm_head_with: head.weight is target_lm.weight (same Parameter object)."""
    sh = SharedHead(hidden_size=32, vocab_size=128)
    target_lm = nn.Linear(32, 128, bias=False)
    sh.share_lm_head_with(target_lm)
    assert sh.head.weight is target_lm.weight


def test_shared_head_share_lm_head_param_dedup():
    """After sharing, only sh.norm.weight + target_lm.weight are unique parameters."""
    sh = SharedHead(hidden_size=32, vocab_size=128)
    target_lm = nn.Linear(32, 128, bias=False)
    sh.share_lm_head_with(target_lm)
    unique = {id(p) for p in list(sh.parameters()) + list(target_lm.parameters())}
    assert len(unique) == 2  # {target_lm.weight, sh.norm.weight}


# ============================================================================
# DeepSeekMultiTokenPredictorLayer
# ============================================================================


def test_mtp_layer_constructible():
    layer = DeepSeekMultiTokenPredictorLayer(
        hidden_size=64, intermediate_size=128, num_heads=4
    )
    assert layer is not None


def test_mtp_layer_has_required_attributes():
    """Layer has enorm + hnorm + eh_proj + mtp_block per source."""
    layer = DeepSeekMultiTokenPredictorLayer(
        hidden_size=64, intermediate_size=128, num_heads=4
    )
    assert hasattr(layer, "enorm")
    assert hasattr(layer, "hnorm")
    assert hasattr(layer, "eh_proj")
    assert hasattr(layer, "mtp_block")


def test_mtp_layer_eh_proj_shape():
    """eh_proj projects [emb_n; h_t] (concat) → hidden. Weight is [hidden, 2*hidden]."""
    layer = DeepSeekMultiTokenPredictorLayer(
        hidden_size=64, intermediate_size=128, num_heads=4
    )
    assert layer.eh_proj.weight.shape == (64, 128)
    assert layer.eh_proj.bias is None  # bias=False per source


def test_mtp_layer_forward_shape():
    """forward(inputs_embeds[T,h], prev_hidden[T,h], positions[T]) → [T, h]."""
    layer = DeepSeekMultiTokenPredictorLayer(
        hidden_size=32, intermediate_size=64, num_heads=4
    )
    T = 8
    inputs_embeds = torch.randn(T, 32)
    prev_hidden = torch.randn(T, 32)
    positions = torch.arange(T)
    out = layer(inputs_embeds, prev_hidden, positions)
    assert out.shape == (T, 32)


def test_mtp_layer_position_zero_zeros_inputs_embeds():
    """At position == 0, inputs_embeds is masked to 0 (no next-token yet)."""
    layer = DeepSeekMultiTokenPredictorLayer(
        hidden_size=32, intermediate_size=64, num_heads=4
    )
    T = 4
    prev_hidden = torch.zeros(T, 32)
    positions = torch.zeros(T, dtype=torch.long)
    out_with_embeds = layer(torch.ones(T, 32), prev_hidden, positions)
    out_with_zeros = layer(torch.zeros(T, 32), prev_hidden, positions)
    assert torch.allclose(out_with_embeds, out_with_zeros, atol=1e-5)


def test_mtp_layer_parameter_stats_layer_only_matches_sum():
    """parameter_stats fields (excl. shared_head) sum to layer.parameters() count."""
    layer = DeepSeekMultiTokenPredictorLayer(
        hidden_size=64, intermediate_size=128, num_heads=4
    )
    stats = layer.parameter_stats(vocab_size=1000)
    actual = sum(p.numel() for p in layer.parameters())
    layer_only = stats.enorm + stats.hnorm + stats.eh_proj + stats.mtp_block
    assert layer_only == actual


# ============================================================================
# MTPLayerStats accounting
# ============================================================================


def test_mtp_layer_stats_share_lm_head_excludes_term():
    """total(share_lm_head=True) excludes shared_head_lm; False includes it."""
    stats = MTPLayerStats(
        enorm=10, hnorm=10, eh_proj=100, mtp_block=1000,
        shared_head_norm=5, shared_head_lm=99999,
    )
    assert stats.total(share_lm_head=True) == 1125
    assert stats.total(share_lm_head=False) == 1125 + 99999


# ============================================================================
# DeepSeekMultiTokenPredictor (stack)
# ============================================================================


def test_predictor_layers_count():
    """Predictor has num_mtp_layers ModuleDict entries."""
    pred = DeepSeekMultiTokenPredictor(
        hidden_size=32, intermediate_size=64, num_heads=4,
        vocab_size=128, num_mtp_layers=3,
    )
    assert len(pred.layers) == 3
    assert "0" in pred.layers and "1" in pred.layers and "2" in pred.layers


def test_predictor_layers_count_K1():
    pred = DeepSeekMultiTokenPredictor(
        hidden_size=32, intermediate_size=64, num_heads=4,
        vocab_size=128, num_mtp_layers=1,
    )
    assert len(pred.layers) == 1


def test_predictor_has_embed_tokens_and_shared_head():
    pred = DeepSeekMultiTokenPredictor(
        hidden_size=32, intermediate_size=64, num_heads=4,
        vocab_size=128, num_mtp_layers=1,
    )
    assert hasattr(pred, "embed_tokens")
    assert hasattr(pred, "shared_head")


def test_predictor_forward_one_step_shape():
    """forward_one_step returns [T, hidden]."""
    pred = DeepSeekMultiTokenPredictor(
        hidden_size=32, intermediate_size=64, num_heads=4,
        vocab_size=128, num_mtp_layers=1,
    )
    T = 4
    out = pred.forward_one_step(
        torch.randint(0, 128, (T,)), torch.arange(T),
        torch.randn(T, 32), spec_step_idx=0,
    )
    assert out.shape == (T, 32)


def test_predictor_compute_logits_shape():
    """compute_logits returns [T, vocab]."""
    pred = DeepSeekMultiTokenPredictor(
        hidden_size=32, intermediate_size=64, num_heads=4,
        vocab_size=128, num_mtp_layers=1,
    )
    logits = pred.compute_logits(torch.randn(8, 32))
    assert logits.shape == (8, 128)


def test_predictor_propose_K_shape():
    """propose_K returns [T, K] int64."""
    pred = DeepSeekMultiTokenPredictor(
        hidden_size=32, intermediate_size=64, num_heads=4,
        vocab_size=128, num_mtp_layers=1,
    )
    T, K = 4, 3
    drafts = pred.propose_K(
        torch.randn(T, 32), torch.randint(0, 128, (T,)),
        torch.arange(T), K=K,
    )
    assert drafts.shape == (T, K)
    assert drafts.dtype == torch.int64


@pytest.mark.parametrize("K", [1, 2, 3, 4, 5])
def test_predictor_propose_K_varies(K):
    pred = DeepSeekMultiTokenPredictor(
        hidden_size=16, intermediate_size=32, num_heads=2,
        vocab_size=64, num_mtp_layers=1,
    )
    T = 3
    drafts = pred.propose_K(
        torch.randn(T, 16), torch.zeros(T, dtype=torch.long),
        torch.arange(T), K=K,
    )
    assert drafts.shape == (T, K)


def test_predictor_propose_K_drafts_in_vocab():
    """propose_K's argmax outputs must be valid token ids in [0, vocab)."""
    vocab = 64
    pred = DeepSeekMultiTokenPredictor(
        hidden_size=16, intermediate_size=32, num_heads=2,
        vocab_size=vocab, num_mtp_layers=1,
    )
    T, K = 3, 4
    drafts = pred.propose_K(
        torch.randn(T, 16), torch.randint(0, vocab, (T,)),
        torch.arange(T), K=K,
    )
    assert drafts.min().item() >= 0
    assert drafts.max().item() < vocab


# ============================================================================
# parameter_count_mtp — demo §3.5 verbatim
# ============================================================================


def test_param_count_mtp_demo_hidden2048_inter8192_vocab32k_K2():
    """Pin demo §3.5 verbatim numbers (lower-bound: dense FFN, no MoE)."""
    res = parameter_count_mtp(
        hidden_size=2048, intermediate_size=8192,
        vocab_size=32000, num_heads=16, num_mtp_layers=2,
    )
    # Demo §3.5: per-layer = 75,505,664
    assert res["per_layer"] == 75_505_664
    # Per-layer breakdown:
    bd = res["per_layer_breakdown"]
    assert bd["enorm"] == 2_048
    assert bd["hnorm"] == 2_048
    assert bd["eh_proj"] == 8_388_608  # 2 * 2048 * 2048
    assert bd["mtp_block_attn"] == 16_777_216  # 4 * 2048 * 2048
    assert bd["mtp_block_ffn"] == 50_331_648  # 3 * 2048 * 8192
    assert bd["mtp_block_norms"] == 4_096  # 2 * 2048
    # Demo §3.5: total with shared lm = 216,549,376
    assert res["total_with_shared_lm"] == 216_549_376
    # Demo §3.5: total with separate lm = 282_085_376
    assert res["total_with_separate_lm"] == 282_085_376


def test_param_count_mtp_separate_minus_shared_eq_vocab_times_hidden():
    """The diff is exactly vocab * hidden (single SharedHead in this impl)."""
    res = parameter_count_mtp(
        hidden_size=2048, intermediate_size=8192,
        vocab_size=32000, num_heads=16, num_mtp_layers=2,
    )
    diff = res["total_with_separate_lm"] - res["total_with_shared_lm"]
    # Predictor has ONE SharedHead (shared across MTP layers), so diff is one vocab*hidden.
    assert diff == 32000 * 2048


def test_param_count_mtp_per_layer_sums_components():
    """per_layer = sum(breakdown components)."""
    res = parameter_count_mtp(
        hidden_size=512, intermediate_size=2048,
        vocab_size=8192, num_heads=8, num_mtp_layers=1,
    )
    bd = res["per_layer_breakdown"]
    assert res["per_layer"] == sum(bd.values())


# ============================================================================
# parameter_count_medusa — demo §3.5 verbatim
# ============================================================================


def test_param_count_medusa_demo_hidden2048_vocab32k_K2():
    """Pin demo §3.5: per-head 73,924,608 = 8,388,608 (mlp) + 65,536,000 (lm)."""
    res = parameter_count_medusa(hidden_size=2048, vocab_size=32000, K=2)
    assert res["per_head"] == 73_924_608
    assert res["per_head_mlp"] == 8_388_608  # 2 * 2048 * 2048
    assert res["per_head_lm"] == 65_536_000  # 2048 * 32000
    assert res["total_with_separate_lm"] == 147_849_216  # 73.9M * 2
    assert res["total_with_shared_lm"] == 16_777_216  # 8.4M * 2


def test_param_count_medusa_K_scales_total():
    """total = per_head * K."""
    for K in [1, 2, 4, 8]:
        res = parameter_count_medusa(hidden_size=512, vocab_size=1024, K=K)
        assert res["total_with_separate_lm"] == res["per_head"] * K


# ============================================================================
# Trap E pin: MTP/Medusa ratio ~12.91x shared, ~1.91x separate
# ============================================================================


def test_trap_E_mtp_to_medusa_ratio_shared_lm():
    """Demo §3.5: MTP/Medusa shared_lm = 12.91x — MTP is NOT lightweight."""
    mtp = parameter_count_mtp(2048, 8192, 32000, num_heads=16, num_mtp_layers=2)
    medusa = parameter_count_medusa(2048, 32000, K=2)
    ratio = mtp["total_with_shared_lm"] / medusa["total_with_shared_lm"]
    assert ratio == pytest.approx(12.91, abs=0.01)


def test_trap_E_mtp_to_medusa_ratio_separate_lm():
    """Demo §3.5: MTP/Medusa separate_lm = 1.91x."""
    mtp = parameter_count_mtp(2048, 8192, 32000, num_heads=16, num_mtp_layers=2)
    medusa = parameter_count_medusa(2048, 32000, K=2)
    ratio = mtp["total_with_separate_lm"] / medusa["total_with_separate_lm"]
    assert ratio == pytest.approx(1.91, abs=0.01)


def test_trap_E_mtp_block_dominates_medusa_per_head_mlp():
    """MTP per-layer block (attn+ffn) >= 8x Medusa per-head MLP (Trap E pin).

    With hidden=2048, inter=8192, vocab=32000:
      mtp_block (attn+ffn) = 16,777,216 + 50,331,648 = 67,108,864
      medusa per-head MLP = 2 * 2048 * 2048   = 8,388,608
      ratio = 8.0x exactly.
    """
    mtp = parameter_count_mtp(2048, 8192, 32000, num_heads=16, num_mtp_layers=1)
    medusa = parameter_count_medusa(2048, 32000, K=1)
    bd = mtp["per_layer_breakdown"]
    block_total = bd["mtp_block_attn"] + bd["mtp_block_ffn"]
    assert block_total >= 8 * medusa["per_head_mlp"]
