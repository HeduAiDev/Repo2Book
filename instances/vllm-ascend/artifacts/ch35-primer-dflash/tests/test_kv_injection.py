"""
Tests for KV injection (arXiv:2602.06036 §4.1 + Appendix A.3): fusing target
context features, projecting them into every draft layer's K/V via a single
fused GEMM (vs. a per-layer loop), and the cross-attention Q/K/V split.
"""
import torch

from kv_injection import (
    build_fused_kv_weight,
    dflash_layer_attention,
    fuse_target_context_features,
    precompute_layer_kv_fused,
    precompute_layer_kv_looped,
    rms_norm,
)

torch.manual_seed(0)

HIDDEN = 8
TARGET_HIDDEN = 8
NUM_LAYERS = 3
NUM_KV_HEADS = 2
HEAD_DIM = 4
KV_SIZE = NUM_KV_HEADS * HEAD_DIM
NUM_SELECTED_LAYERS = 5


def _make_selected_target_layers(num_ctx):
    return [torch.randn(num_ctx, TARGET_HIDDEN) for _ in range(NUM_SELECTED_LAYERS)]


class TestRMSNorm:
    def test_output_has_unit_rms_before_weight_scaling(self):
        x = torch.randn(4, HIDDEN) * 5 + 1
        weight = torch.ones(HIDDEN)
        y = rms_norm(x, weight, eps=1e-6)
        rms = y.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


class TestFuseTargetContextFeatures:
    def test_output_shape(self):
        num_ctx = 6
        layers = _make_selected_target_layers(num_ctx)
        w_c = torch.randn(HIDDEN, NUM_SELECTED_LAYERS * TARGET_HIDDEN) * 0.1
        norm_weight = torch.ones(HIDDEN)
        h_t = fuse_target_context_features(layers, w_c, norm_weight)
        assert h_t.shape == (num_ctx, HIDDEN)

    def test_every_selected_layer_affects_the_fused_feature(self):
        # H_t = RMSNorm(W_c [H^(l1);...;H^(l5)]) must actually depend on all
        # five selected layers, not silently drop some of them.
        num_ctx = 3
        w_c = torch.randn(HIDDEN, NUM_SELECTED_LAYERS * TARGET_HIDDEN) * 0.1
        norm_weight = torch.ones(HIDDEN)
        base_layers = _make_selected_target_layers(num_ctx)
        h_t_base = fuse_target_context_features(base_layers, w_c, norm_weight)
        for i in range(NUM_SELECTED_LAYERS):
            perturbed = list(base_layers)
            perturbed[i] = perturbed[i] + 10.0
            h_t_perturbed = fuse_target_context_features(perturbed, w_c, norm_weight)
            assert not torch.allclose(h_t_base, h_t_perturbed), f"layer {i} had no effect"


class TestFusedVsLoopedKvEquivalence:
    def test_fused_gemm_matches_per_layer_loop(self):
        # precompute_and_store_context_kv's "one GEMM for all layers" is an
        # engineering optimization over looping per layer -- it must be
        # numerically identical, not just architecturally similar.
        num_ctx = 5
        h_t = torch.randn(num_ctx, HIDDEN)
        positions = torch.arange(num_ctx, dtype=torch.float32)

        k_weights = [torch.randn(KV_SIZE, HIDDEN) * 0.1 for _ in range(NUM_LAYERS)]
        v_weights = [torch.randn(KV_SIZE, HIDDEN) * 0.1 for _ in range(NUM_LAYERS)]
        k_norm_weights = [torch.ones(HEAD_DIM) for _ in range(NUM_LAYERS)]

        fused_weight = build_fused_kv_weight(k_weights, v_weights)
        all_k_fused, all_v_fused = precompute_layer_kv_fused(
            h_t, fused_weight, k_norm_weights, positions,
            num_layers=NUM_LAYERS, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
        )
        all_k_looped, all_v_looped = precompute_layer_kv_looped(
            h_t, k_weights, v_weights, k_norm_weights, positions,
            num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
        )
        assert torch.allclose(all_k_fused, all_k_looped, atol=1e-5)
        assert torch.allclose(all_v_fused, all_v_looped, atol=1e-5)

    def test_fused_weight_shape(self):
        k_weights = [torch.randn(KV_SIZE, HIDDEN) for _ in range(NUM_LAYERS)]
        v_weights = [torch.randn(KV_SIZE, HIDDEN) for _ in range(NUM_LAYERS)]
        fused = build_fused_kv_weight(k_weights, v_weights)
        assert fused.shape == (NUM_LAYERS * 2 * KV_SIZE, HIDDEN)


class TestDflashLayerAttention:
    def _make_layer_weights(self):
        num_heads = 2
        return dict(
            w_q=torch.randn(num_heads * HEAD_DIM, HIDDEN) * 0.1,
            w_k=torch.randn(KV_SIZE, HIDDEN) * 0.1,
            w_v=torch.randn(KV_SIZE, HIDDEN) * 0.1,
            w_o=torch.randn(HIDDEN, num_heads * HEAD_DIM) * 0.1,
            q_norm_weight=torch.ones(HEAD_DIM),
            k_norm_weight=torch.ones(HEAD_DIM),
        )

    def test_output_shape(self):
        num_query, num_ctx = 4, 6
        h_d = torch.randn(num_query, HIDDEN)
        context_k = torch.randn(num_ctx, NUM_KV_HEADS, HEAD_DIM)
        context_v = torch.randn(num_ctx, NUM_KV_HEADS, HEAD_DIM)
        positions = torch.arange(num_query, dtype=torch.float32)
        out = dflash_layer_attention(
            h_d, context_k, context_v, positions=positions,
            num_heads=2, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
            **self._make_layer_weights(),
        )
        assert out.shape == (num_query, HIDDEN)

    def test_output_is_sensitive_to_injected_context_kv(self):
        # This is the whole point of KV injection: the draft layer's output
        # must depend on the target-derived context K/V, not just on the
        # draft's own query tokens.
        num_query, num_ctx = 3, 5
        h_d = torch.randn(num_query, HIDDEN)
        positions = torch.arange(num_query, dtype=torch.float32)
        weights = self._make_layer_weights()

        context_k = torch.randn(num_ctx, NUM_KV_HEADS, HEAD_DIM)
        context_v = torch.randn(num_ctx, NUM_KV_HEADS, HEAD_DIM)
        out_base = dflash_layer_attention(
            h_d, context_k, context_v, positions=positions,
            num_heads=2, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, **weights,
        )
        out_perturbed = dflash_layer_attention(
            h_d, context_k + 5.0, context_v + 5.0, positions=positions,
            num_heads=2, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, **weights,
        )
        assert not torch.allclose(out_base, out_perturbed)

    def test_query_only_derives_from_draft_hidden_states(self):
        # Appendix A.3: Q_i = W_i^Q H_d -- target features bypass the Q
        # projection entirely. We can't inspect Q directly through the
        # public function, but we can confirm the *keyword* contract: the
        # function never takes target hidden states as an input used for Q,
        # only as already-projected context_k/context_v.
        import inspect
        sig = inspect.signature(dflash_layer_attention)
        assert "target_hidden_states" not in sig.parameters
        assert {"context_k", "context_v", "h_d"}.issubset(sig.parameters)

    def test_masked_positions_see_each_other_bidirectionally(self):
        # Non-causal: perturbing one query position's input must be able to
        # affect another query position's output (block-internal visibility).
        num_query, num_ctx = 4, 3
        positions = torch.arange(num_query, dtype=torch.float32)
        weights = self._make_layer_weights()
        context_k = torch.randn(num_ctx, NUM_KV_HEADS, HEAD_DIM)
        context_v = torch.randn(num_ctx, NUM_KV_HEADS, HEAD_DIM)

        h_d = torch.randn(num_query, HIDDEN)
        out_base = dflash_layer_attention(
            h_d, context_k, context_v, positions=positions,
            num_heads=2, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, **weights,
        )
        h_d_perturbed = h_d.clone()
        h_d_perturbed[0] += 10.0  # perturb position 0 only
        out_perturbed = dflash_layer_attention(
            h_d_perturbed, context_k, context_v, positions=positions,
            num_heads=2, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, **weights,
        )
        # position 1..3's output should change too (not just position 0's).
        assert not torch.allclose(out_base[1:], out_perturbed[1:])
