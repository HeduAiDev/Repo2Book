"""Tests for feature_autoregression.py -- PAPER: arXiv:2401.15077 §2/§3.1/§3.2."""
import torch

from feature_autoregression import (
    AutoregressionHead,
    ToyTargetLLM,
    build_shifted_token_input,
    combined_loss,
)


class TestToyTargetLLM:
    def test_forward_prefix_shape(self):
        llm = ToyTargetLLM(vocab_size=6, hidden_dim=4, seed=0)
        token_ids = torch.tensor([0, 1, 2])
        feats = llm.forward_prefix(token_ids)
        assert feats.shape == (3, 4)

    def test_next_token_distribution_is_a_valid_distribution(self):
        llm = ToyTargetLLM(vocab_size=6, hidden_dim=4, seed=0)
        feats = llm.forward_prefix(torch.tensor([0, 1, 2]))
        probs = llm.next_token_distribution(feats[-1])
        assert probs.shape == (6,)
        assert torch.all(probs >= 0)
        assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)

    def test_deterministic_given_same_seed(self):
        llm_a = ToyTargetLLM(vocab_size=6, hidden_dim=4, seed=7)
        llm_b = ToyTargetLLM(vocab_size=6, hidden_dim=4, seed=7)
        toks = torch.tensor([1, 2, 3])
        feats_a = llm_a.forward_prefix(toks)
        feats_b = llm_b.forward_prefix(toks)
        assert torch.allclose(feats_a, feats_b)


class TestBuildShiftedTokenInput:
    def test_matches_vllm_shift_and_splice_example(self):
        # PAPER §3.1 / llm_base_proposer.py:L664-669 -- shift left by one,
        # splice the newly-sampled token into the last slot.
        token_ids = torch.tensor([1, 2, 3])
        shifted = build_shifted_token_input(token_ids, sampled_next_token=4)
        assert shifted.tolist() == [2, 3, 4]

    def test_single_token_prefix(self):
        token_ids = torch.tensor([9])
        shifted = build_shifted_token_input(token_ids, sampled_next_token=5)
        assert shifted.tolist() == [5]


class TestAutoregressionHead:
    def test_fc_receives_concatenated_2h_dim(self):
        hidden_dim = 4
        head = AutoregressionHead(hidden_dim=hidden_dim, seed=1)
        assert head.fc.weight.shape == (hidden_dim, 2 * hidden_dim)

    def test_forward_shape_single_vector(self):
        hidden_dim = 4
        head = AutoregressionHead(hidden_dim=hidden_dim, seed=1)
        token_embed = torch.randn(hidden_dim)
        feature = torch.randn(hidden_dim)
        out = head(token_embed, feature)
        assert out.shape == (hidden_dim,)

    def test_forward_shape_batched(self):
        hidden_dim = 4
        seq_len = 3
        head = AutoregressionHead(hidden_dim=hidden_dim, seed=1)
        token_embeds = torch.randn(seq_len, hidden_dim)
        features = torch.randn(seq_len, hidden_dim)
        out = head(token_embeds, features)
        assert out.shape == (seq_len, hidden_dim)


class TestTrainingLosses:
    def test_zero_loss_when_prediction_matches_target_exactly(self):
        # CE(p, p) is the entropy of p, which is only exactly 0 when p is a
        # one-hot (degenerate) distribution -- so use sharply peaked logits
        # (softmax ~= one-hot) rather than arbitrary random logits here.
        feature = torch.randn(4)
        logits = torch.tensor([50.0, -50.0, -50.0, -50.0, -50.0, -50.0])
        total, l_reg, l_cls = combined_loss(feature, feature, logits, logits)
        assert torch.isclose(l_reg, torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(l_cls, torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(total, torch.tensor(0.0), atol=1e-5)

    def test_combined_uses_w_cls_point_one_weighting(self):
        pred_feature = torch.zeros(4)
        target_feature = torch.ones(4)
        pred_logits = torch.tensor([2.0, 0.0, 0.0])
        target_logits = torch.tensor([0.0, 2.0, 0.0])
        total, l_reg, l_cls = combined_loss(pred_feature, target_feature, pred_logits, target_logits)
        assert torch.isclose(total, l_reg + 0.1 * l_cls, atol=1e-6)
        assert l_reg.item() > 0.0
        assert l_cls.item() > 0.0
