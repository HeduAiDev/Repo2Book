"""Tests — Ch10 Multi-Token Prediction."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
import pytest
from implementation.mtp import (
    mtp_speedup_analysis, MTPHead, MultiTokenPredictor, verify_mtp_tokens,
)


class TestMTPSpeedup:
    def test_perfect_prediction(self):
        """With 100% acceptance, speedup = num_steps."""
        r = mtp_speedup_analysis(num_steps=3, acceptance_rates=(1.0, 1.0))
        assert r["speedup"] == 3.0
        assert r["efficiency"] == 1.0

    def test_imperfect_prediction(self):
        """Realistic acceptance rates give sub-ideal speedup."""
        r = mtp_speedup_analysis(num_steps=3, acceptance_rates=(0.8, 0.6))
        assert 1.0 < r["speedup"] < 3.0
        assert r["efficiency"] < 1.0

    def test_diminishing_returns(self):
        """More steps → lower marginal gain."""
        r2 = mtp_speedup_analysis(2, (0.85,))
        r4 = mtp_speedup_analysis(4, (0.85, 0.70, 0.55))
        # 4-step efficiency should be lower than 2-step
        assert r4["efficiency"] < r2["efficiency"]


class TestMTPHead:
    def test_output_shape(self):
        head = MTPHead(d_model=256, vocab_size=1000)
        x = torch.randn(4, 256)
        logits = head(x)
        assert logits.shape == (4, 1000)


class TestMultiTokenPredictor:
    def test_multiple_heads(self):
        mtp = MultiTokenPredictor(d_model=256, vocab_size=1000, num_heads=3)
        x = torch.randn(4, 256)
        logits_list = mtp(x)
        assert len(logits_list) == 3
        assert all(l.shape == (4, 1000) for l in logits_list)


class TestVerification:
    def test_all_accept(self):
        draft = [1, 2, 3]
        verified = [1, 2, 3]
        accepted, n = verify_mtp_tokens(draft, verified)
        assert n == 3
        assert accepted == [1, 2, 3]

    def test_first_mismatch_stops(self):
        """First mismatch → accept verified version of that token, stop."""
        draft = [1, 9, 3]  # 9 is wrong
        verified = [1, 2, 3]
        accepted, n = verify_mtp_tokens(draft, verified)
        assert n == 2
        assert accepted == [1, 2]  # First accepted, second replaced with verified

    def test_all_reject(self):
        draft = [9, 9, 9]
        verified = [1, 2, 3]
        accepted, n = verify_mtp_tokens(draft, verified)
        assert n == 1  # Only first verified token accepted
        assert accepted == [1]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
