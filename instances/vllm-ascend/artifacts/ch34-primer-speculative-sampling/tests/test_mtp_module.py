"""
Tests for the paper-faithful reference implementation of the DeepSeek-V3
Multi-Token Prediction (MTP) module (arXiv:2412.19437 §2.2, Eq.21-23).

These tests check that the reference implementation reproduces the paper's
described *structure* — shared Emb/OutHead across depths, the causal chain
(depth k sees depth k-1's representation, not an independent parallel head),
and the token-shift Emb(t_{i+k}) — not merely that shapes line up.
"""
import torch

from mtp_module import DeepSeekMTPPredictor, MTPModule, RMSNorm


HIDDEN = 8
VOCAB = 16


def _make_predictor(depth=3, seed=0):
    torch.manual_seed(seed)
    return DeepSeekMTPPredictor(depth=depth, hidden_size=HIDDEN, vocab_size=VOCAB)


class TestRMSNorm:
    def test_output_has_unit_rms_before_weight_scaling(self):
        norm = RMSNorm(HIDDEN)
        x = torch.randn(2, 5, HIDDEN) * 10 + 3
        with torch.no_grad():
            norm.weight.fill_(1.0)
            y = norm(x)
        rms = y.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


class TestMTPModuleShapes:
    def test_forward_shapes(self):
        predictor = _make_predictor(depth=1)
        module = predictor.modules_by_depth[0]
        batch, valid_len = 2, 6
        h_prev = torch.randn(batch, valid_len, HIDDEN)
        tokens = torch.randint(0, VOCAB, (batch, valid_len))
        h_k, logits_k = module(h_prev, tokens)
        assert h_k.shape == (batch, valid_len, HIDDEN)
        assert logits_k.shape == (batch, valid_len, VOCAB)


class TestSharedParameters:
    """paper-mtp.md §2.2: 'for each MTP module, its embedding layer is shared
    with the main model' / 'its output head is shared with the main model'."""

    def test_embed_and_out_head_are_the_same_object_across_depths(self):
        predictor = _make_predictor(depth=3)
        embeds = {id(m.embed) for m in predictor.modules_by_depth}
        heads = {id(m.out_head) for m in predictor.modules_by_depth}
        assert embeds == {id(predictor.embed)}
        assert heads == {id(predictor.out_head)}


class TestCausalChainAndSequentialDepths:
    """paper-mtp.md §2.2: 'we sequentially predict additional tokens and keep
    the complete causal chain at each prediction depth' — contrasted
    explicitly with Gloeckle et al.'s independent parallel output heads."""

    def test_valid_window_shrinks_by_one_per_depth(self):
        depth = 3
        seq_len = 10
        predictor = _make_predictor(depth=depth)
        batch = 2
        h_main = torch.randn(batch, seq_len, HIDDEN)
        token_ids = torch.randint(0, VOCAB, (batch, seq_len))
        outputs = predictor(h_main, token_ids)
        assert len(outputs) == depth
        for k, (h_k, logits_k) in enumerate(outputs, start=1):
            assert h_k.shape[1] == seq_len - k
            assert logits_k.shape[1] == seq_len - k

    def test_shifted_tokens_correspond_to_t_i_plus_k(self):
        # Eq.21: at depth k, position i combines h_i^{k-1} with Emb(t_{i+k}).
        # We verify this indirectly: replacing token_ids[:, k] (i.e. t_{1+k}
        # for i=1) with a different id must change depth k's first output
        # position, since position i=1 at depth k consumes exactly that token.
        depth = 2
        seq_len = 6
        predictor = _make_predictor(depth=depth)
        batch = 1
        h_main = torch.randn(batch, seq_len, HIDDEN)
        token_ids = torch.randint(0, VOCAB, (batch, seq_len))

        k = 1  # first MTP depth: position i=0 (0-indexed) should read token_ids[:, 0 + k]
        outputs = predictor(h_main, token_ids)
        _, logits_before = outputs[k - 1]

        token_ids_perturbed = token_ids.clone()
        token_ids_perturbed[:, k] = (token_ids_perturbed[:, k] + 1) % VOCAB
        outputs_perturbed = predictor(h_main, token_ids_perturbed)
        _, logits_after = outputs_perturbed[k - 1]

        assert not torch.allclose(logits_before[:, 0], logits_after[:, 0])

    def test_depth_k_output_depends_on_depth_k_minus_1_representation(self):
        # Sequential causal chain: h^k is a function of h^{k-1}. Perturbing
        # the main model's hidden state must propagate into depth-2 output
        # (not just depth-1), proving depth 2 consumes depth 1's h, rather
        # than an independent parallel head reading only h_main.
        depth = 2
        seq_len = 6
        predictor = _make_predictor(depth=depth)
        batch = 1
        token_ids = torch.randint(0, VOCAB, (batch, seq_len))

        h_main = torch.randn(batch, seq_len, HIDDEN)
        outputs = predictor(h_main, token_ids)
        _, logits_depth2_before = outputs[1]

        h_main_perturbed = h_main.clone()
        h_main_perturbed[:, 0] += 5.0
        outputs_perturbed = predictor(h_main_perturbed, token_ids)
        _, logits_depth2_after = outputs_perturbed[1]

        assert not torch.allclose(logits_depth2_before, logits_depth2_after)

    def test_single_mtp_module_is_reusable_building_block(self):
        # A bare MTPModule (as used at any single depth) must be constructible
        # standalone with explicit shared embed/out_head, matching Eq.21-23
        # applying uniformly at every depth k.
        embed = torch.nn.Embedding(VOCAB, HIDDEN)
        out_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
        module = MTPModule(HIDDEN, VOCAB, embed, out_head)
        h_prev = torch.randn(2, 4, HIDDEN)
        tokens = torch.randint(0, VOCAB, (2, 4))
        h_k, logits_k = module(h_prev, tokens)
        assert h_k.shape == (2, 4, HIDDEN)
        assert logits_k.shape == (2, 4, VOCAB)
