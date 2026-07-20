"""Tests for chain_drafting.py -- PAPER: arXiv:2401.15077 §3.1 drafting phase,
chain special case (dossier paper_origin_note: vLLM v1's default eagle path,
llm_base_proposer.py:L392-L592 propose() / L646-L678 set_inputs_first_pass()).
"""
import torch

from chain_drafting import greedy_sample, propose_chain
from feature_autoregression import AutoregressionHead, ToyTargetLLM, build_shifted_token_input


def _setup(vocab_size=6, hidden_dim=4):
    target_llm = ToyTargetLLM(vocab_size=vocab_size, hidden_dim=hidden_dim, seed=0)
    draft_head = AutoregressionHead(hidden_dim=hidden_dim, seed=1)
    return target_llm, draft_head


class TestGreedySample:
    def test_returns_argmax_token_and_its_own_probability_as_confidence(self):
        target_llm, _ = _setup()
        feature = torch.randn(4)
        token_id, confidence = greedy_sample(target_llm, feature)
        probs = target_llm.next_token_distribution(feature)
        assert token_id == int(torch.argmax(probs).item())
        assert confidence == probs[token_id].item()

    def test_confidence_is_a_valid_probability(self):
        target_llm, _ = _setup()
        feature = torch.randn(4)
        _, confidence = greedy_sample(target_llm, feature)
        assert 0.0 <= confidence <= 1.0


class TestProposeChain:
    def test_returns_requested_number_of_draft_tokens(self):
        target_llm, draft_head = _setup()
        prefix_token_ids = torch.tensor([1, 2, 3])
        prefix_features = target_llm.forward_prefix(prefix_token_ids)
        draft_ids, confidences = propose_chain(
            target_llm, draft_head, prefix_token_ids, prefix_features,
            next_token_id=4, num_speculative_tokens=3,
        )
        assert len(draft_ids) == 3
        assert len(confidences) == 3
        assert all(0.0 <= c <= 1.0 for c in confidences)

    def test_single_speculative_token_early_exit_shape(self):
        target_llm, draft_head = _setup()
        prefix_token_ids = torch.tensor([1, 2])
        prefix_features = target_llm.forward_prefix(prefix_token_ids)
        draft_ids, confidences = propose_chain(
            target_llm, draft_head, prefix_token_ids, prefix_features,
            next_token_id=3, num_speculative_tokens=1,
        )
        assert len(draft_ids) == 1
        assert len(confidences) == 1

    def test_deterministic_given_same_seeds(self):
        target_llm_a, draft_head_a = _setup()
        target_llm_b, draft_head_b = _setup()
        prefix = torch.tensor([0, 1, 2])
        feats_a = target_llm_a.forward_prefix(prefix)
        feats_b = target_llm_b.forward_prefix(prefix)
        ids_a, conf_a = propose_chain(target_llm_a, draft_head_a, prefix, feats_a, 3, 3)
        ids_b, conf_b = propose_chain(target_llm_b, draft_head_b, prefix, feats_b, 3, 3)
        assert ids_a == ids_b
        assert conf_a == conf_b

    def test_first_pass_feature_matches_hand_rolled_shifted_input(self):
        # PAPER §3.1/Fig.3-5 -- the first draft token must come from feeding
        # the *shifted* token embeddings (not the raw prefix tokens) fused
        # with the prefix features. Recompute that first step by hand from
        # feature_autoregression building blocks and check it matches what
        # propose_chain's internals would produce (same seeds -> identical
        # weights -> identical numbers).
        target_llm, draft_head = _setup()
        prefix_token_ids = torch.tensor([1, 2, 3])
        prefix_features = target_llm.forward_prefix(prefix_token_ids)
        next_token_id = 4

        shifted = build_shifted_token_input(prefix_token_ids, next_token_id)
        assert shifted.tolist() == [2, 3, 4]
        token_embeds = target_llm.embed_tokens(shifted)
        fused_features = draft_head(token_embeds, prefix_features)
        expected_first_feature = fused_features[-1]
        expected_token_id, expected_conf = greedy_sample(target_llm, expected_first_feature)

        draft_ids, confidences = propose_chain(
            target_llm, draft_head, prefix_token_ids, prefix_features,
            next_token_id, num_speculative_tokens=1,
        )
        assert draft_ids[0] == expected_token_id
        assert confidences[0] == expected_conf

    def test_chain_continuation_feeds_previous_token_and_feature_back(self):
        # PAPER §3.1: "f_3 and t_4 are concatenated ... to predict f_4 and
        # sample t_5" -- step 2 onward must consume the *previous step's*
        # (token embedding, feature), not the original prefix again.
        target_llm, draft_head = _setup()
        prefix_token_ids = torch.tensor([1, 2])
        prefix_features = target_llm.forward_prefix(prefix_token_ids)
        next_token_id = 3

        draft_ids, _ = propose_chain(
            target_llm, draft_head, prefix_token_ids, prefix_features,
            next_token_id, num_speculative_tokens=2,
        )
        # Hand-reproduce step 1 to get the feature step 2 must consume.
        shifted = build_shifted_token_input(prefix_token_ids, next_token_id)
        token_embeds = target_llm.embed_tokens(shifted)
        fused_features = draft_head(token_embeds, prefix_features)
        feature_1 = fused_features[-1]
        token_1, _ = greedy_sample(target_llm, feature_1)
        assert token_1 == draft_ids[0]

        token_embed_1 = target_llm.embed_tokens(torch.tensor(token_1))
        feature_2 = draft_head(token_embed_1, feature_1)
        token_2, _ = greedy_sample(target_llm, feature_2)
        assert token_2 == draft_ids[1]
