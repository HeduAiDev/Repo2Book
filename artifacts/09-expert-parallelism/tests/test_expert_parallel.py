"""Tests — Ch9 Expert Parallelism."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
import pytest
from implementation.expert_parallel import (
    TopKRouter, determine_expert_map, SimpleMoELayer,
    simulate_ep_dispatch_combine,
)


class TestTopKRouter:
    def test_shape(self):
        router = TopKRouter(num_experts=8, top_k=2)
        logits = torch.randn(4, 8)
        w, ids = router.route(logits)
        assert w.shape == (4, 2)
        assert ids.shape == (4, 2)

    def test_weights_sum_to_one(self):
        router = TopKRouter(num_experts=8, top_k=2)
        logits = torch.randn(4, 8)
        w, _ = router.route(logits)
        assert torch.allclose(w.sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_topk_correctness(self):
        router = TopKRouter(num_experts=4, top_k=2)
        logits = torch.tensor([[1.0, 2.0, 3.0, 0.5]])  # Expert 2 > 1 > 0 > 3
        _, ids = router.route(logits)
        assert set(ids[0].tolist()) == {2, 1}  # Top-2 experts


class TestExpertMap:
    def test_linear(self):
        n, experts = determine_expert_map(4, 0, 8, "linear")
        assert n == 2
        assert experts == [0, 1]

    def test_linear_remainder(self):
        """With uneven division, first ranks get +1 expert."""
        n0, e0 = determine_expert_map(3, 0, 8, "linear")
        n1, e1 = determine_expert_map(3, 1, 8, "linear")
        n2, e2 = determine_expert_map(3, 2, 8, "linear")
        assert n0 == 3  # 8//3=2, rem=2 → first 2 ranks get 3
        assert n2 == 2  # last rank gets 2


class TestSimpleMoE:
    def test_forward_shape(self):
        moe = SimpleMoELayer(hidden_size=32, intermediate_size=64,
                             num_experts=4, top_k=2)
        x = torch.randn(8, 32)
        y = moe(x)
        assert y.shape == x.shape

    def test_no_nan(self):
        moe = SimpleMoELayer(hidden_size=32, intermediate_size=64,
                             num_experts=4, top_k=2)
        x = torch.randn(8, 32)
        y = moe(x)
        assert not torch.isnan(y).any()

    def test_sparse_computation(self):
        """Output changes with different top_k (more experts = more computation)."""
        moe1 = SimpleMoELayer(hidden_size=32, intermediate_size=64,
                              num_experts=4, top_k=1)
        moe2 = SimpleMoELayer(hidden_size=32, intermediate_size=64,
                              num_experts=4, top_k=4)
        # Copy weights for fair comparison
        moe2.W1.data = moe1.W1.data.clone()
        moe2.W2.data = moe1.W2.data.clone()
        moe2.router.weight.data = moe1.router.weight.data.clone()

        x = torch.randn(8, 32)
        y1 = moe1(x)
        y2 = moe2(x)
        # All experts used (top_k=4) vs sparse (top_k=1) — outputs differ
        assert not torch.allclose(y1, y2)


class TestCommunication:
    def test_ep_simulation(self):
        r = simulate_ep_dispatch_combine(4096, 2048, 256, 8, 8)
        assert r["total_all2all_gb"] > 0
        assert r["dispatch_volume_gb"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
