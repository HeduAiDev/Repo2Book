"""
Expert Parallelism — Our Reimplementation.

REFERENCE sources:
    FusedMoE:              vllm/model_executor/layers/fused_moe/layer.py:L219
    FusedMoEParallelConfig: vllm/model_executor/layers/fused_moe/config.py:L999
    FusedTopKRouter:       vllm/model_executor/layers/fused_moe/router/fused_topk_router.py:L116
    GroupedTopKRouter:     vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py:L247
    AgRsAll2AllManager:    vllm/distributed/device_communicators/all2all.py:L40
    EPLBConfig:            vllm/config/parallel.py:L55
    determine_expert_map:  vllm/model_executor/layers/fused_moe/layer.py:L71
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


# ═══════════════════════════════════════════════════════════════════════════
# Top-K Router
# REFERENCE: vllm/model_executor/layers/fused_moe/router/fused_topk_router.py:L116
# ═══════════════════════════════════════════════════════════════════════════

class TopKRouter:
    """
    Standard top-k gating for MoE.

    REFERENCE: fused_topk_router.py:L116 — FusedTopKRouter

    For each token:
        1. Compute router_logits (linear projection hidden → num_experts)
        2. Apply softmax
        3. Select top-k experts by score
        4. Renormalize weights of selected experts
    """

    def __init__(self, num_experts: int, top_k: int):
        self.num_experts = num_experts
        self.top_k = top_k

    def route(self, router_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            router_logits: [num_tokens, num_experts] — raw logits from gate

        Returns:
            topk_weights: [num_tokens, top_k] — normalized routing weights
            topk_ids: [num_tokens, top_k] — selected expert indices
        """
        # Softmax over experts
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k
        topk_weights, topk_ids = torch.topk(router_probs, self.top_k, dim=-1)

        # Renormalize so selected weights sum to 1
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        return topk_weights, topk_ids


# ═══════════════════════════════════════════════════════════════════════════
# Expert Map — determine_expert_map
# REFERENCE: vllm/model_executor/layers/fused_moe/layer.py:L71
# ═══════════════════════════════════════════════════════════════════════════

def determine_expert_map(
    ep_size: int, ep_rank: int, global_num_experts: int,
    placement_strategy: str = "linear",
) -> Tuple[int, List[int]]:
    """
    Which experts live on this EP rank?

    REFERENCE: layer.py:L71-L157 — determine_expert_map()

    Two strategies:
        "linear": Contiguous blocks. Rank 0: [0,1], Rank 1: [2,3], ...
        "round_robin": Interleaved. Rank 0: [0,2], Rank 1: [1,3], ...

    Used with DeepEP low-latency or NIXL backends.
    """
    base = global_num_experts // ep_size
    rem = global_num_experts % ep_size
    local_num = base + (1 if ep_rank < rem else 0)

    if placement_strategy == "linear":
        # First rem ranks get base+1 experts, rest get base
        start = ep_rank * base + min(ep_rank, rem)
        local_experts = list(range(start, start + local_num))
    else:  # round_robin
        local_experts = [i for i in range(global_num_experts) if i % ep_size == ep_rank]

    return local_num, local_experts


# ═══════════════════════════════════════════════════════════════════════════
# MoE Layer (simplified, single GPU)
# ═══════════════════════════════════════════════════════════════════════════

class SimpleMoELayer(nn.Module):
    """
    Simplified MoE layer without actual EP communication.

    REFERENCE: vllm/model_executor/layers/fused_moe/layer.py:L219 — FusedMoE

    Key idea: Instead of one large FFN, have many small "experts", each a
    small FFN. The router sends each token to top-k experts only — sparse
    activation means cheaper inference.

    Math:
        y = sum_{i in top_k} g_i(x) * Expert_i(x)

    where g_i(x) is the routing weight (softmax probability) and
    Expert_i(x) = W2_i @ act(W1_i @ x) is a standard 2-layer FFN.
    """

    def __init__(self, hidden_size: int, intermediate_size: int,
                 num_experts: int, top_k: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k

        # Router: hidden → num_experts
        self.router = nn.Linear(hidden_size, num_experts, bias=False)

        # Experts: each is a small FFN
        # Shape: [num_experts, intermediate_size, hidden_size]  (W1)
        #        [num_experts, hidden_size, intermediate_size]  (W2)
        self.W1 = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.W2 = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))

        self.tok_router = TopKRouter(num_experts, top_k)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W1)
        nn.init.kaiming_uniform_(self.W2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_tokens, hidden_size]

        Returns:
            output: [num_tokens, hidden_size]
        """
        N, D = x.shape

        # 1. Route tokens
        router_logits = self.router(x)                      # [N, E]
        topk_weights, topk_ids = self.tok_router.route(router_logits)  # [N, K]

        # 2. Dispatch: for each token, compute only top-k experts
        # In real EP, this is where AllToAll communication happens
        output = torch.zeros(N, D, device=x.device, dtype=x.dtype)

        for k in range(self.top_k):
            expert_ids = topk_ids[:, k]                     # [N] — which expert
            weights = topk_weights[:, k].unsqueeze(-1)      # [N, 1] — routing weight

            # Select the expert weights
            w1_k = self.W1[expert_ids]                      # [N, intermediate, D]
            w2_k = self.W2[expert_ids]                      # [N, D, intermediate]

            # Expert computation (batched per expert ID)
            h = torch.bmm(x.unsqueeze(1), w1_k.transpose(1, 2)).squeeze(1)  # [N, I]
            h = F.silu(h)
            out_k = torch.bmm(h.unsqueeze(1), w2_k.transpose(1, 2)).squeeze(1)  # [N, D]

            output += weights * out_k

        return output


# ═══════════════════════════════════════════════════════════════════════════
# EP + AllToAll Flow Simulation
# REFERENCE: vllm/distributed/device_communicators/all2all.py:L83-L136
# ═══════════════════════════════════════════════════════════════════════════

def simulate_ep_dispatch_combine(
    tokens: int, hidden_size: int, num_experts: int, top_k: int, ep_size: int,
) -> dict:
    """
    Quantify the AllToAll communication volume for Expert Parallelism.

    The EP data flow:
        1. Each GPU computes router_logits locally
        2. DISPATCH: all-to-all (all_gatherv) of hidden_states + router data
           → Each GPU receives tokens routed to its local experts
        3. Each GPU computes its local experts
        4. COMBINE: all-to-all (reduce_scatterv) to send results back
    """
    # Per-token data to dispatch
    per_token_dispatch = hidden_size + num_experts + top_k * 4  # hidden + logits + ids
    dispatch_total = tokens * per_token_dispatch

    # Each GPU sends to every other GPU (all_gatherv = each GPU receives full data)
    dispatch_comm = dispatch_total * ep_size  # bytes

    # Combine: reduce_scatterv sends results back
    combine_comm = tokens * hidden_size * ep_size  # bytes

    return {
        "tokens": tokens,
        "ep_size": ep_size,
        "dispatch_volume_gb": round(dispatch_comm / (1024**3), 4),
        "combine_volume_gb": round(combine_comm / (1024**3), 4),
        "total_all2all_gb": round((dispatch_comm + combine_comm) / (1024**3), 4),
        "note": (
            "With DeepEP/FlashInfer NVLink backends, actual bandwidth utilization "
            "approaches NVLink peak (~900 GB/s bidirectional on H100), making "
            "AllToAll latency ~0.1-0.5ms for typical MoE configurations."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# EP + TP Combined Analysis
# REFERENCE: vllm/model_executor/layers/fused_moe/config.py:L1082-L1209
# ═══════════════════════════════════════════════════════════════════════════

def ep_tp_tradeoff_analysis(
    total_gpus: int, num_experts: int, top_k: int, hidden_size: int,
) -> dict:
    """
    Analyze EP vs TP trade-offs for MoE layers.

    REFERENCE: config.py:L1082 — FusedMoEParallelConfig.make()

    When EP is enabled, TP is DISABLED for MoE layers:
        - Each GPU owns FULL expert weights (no tensor sharding)
        - Each GPU only hosts a SUBSET of experts
        - Expert computation is not sharded — it's replicated per expert per GPU

    TP (Tensor Parallel):
        - Expert weights are sharded across GPUs
        - Every GPU participates in every expert's computation
        - Communication: AllReduce per expert layer (O(d_model) per layer)

    EP (Expert Parallel):
        - Expert weights are NOT sharded
        - Each GPU only computes its local experts
        - Communication: AllToAll per MoE layer (O(tokens * hidden) per layer)
    """
    # TP: each GPU gets 1/total_gpus of expert weights
    tp_expert_weight_per_gpu = (
        num_experts * 2 * hidden_size * (8/3 * hidden_size) // total_gpus
    )

    # EP: each GPU gets num_experts/total_gpus full expert weights
    ep_expert_weight_per_gpu = (
        (num_experts // total_gpus) * 2 * hidden_size * (8/3 * hidden_size)
    )

    # Communication
    # TP: all-reduce per transformer block (2× for standard block)
    tp_comm_per_layer = 2 * hidden_size  # simplified

    # EP: all-to-all dispatch + combine
    # With top-k routing, each token goes to k experts → data is expanded
    ep_comm_per_layer = 2 * top_k * hidden_size  # dispatch + combine (simplified)

    return {
        "tp_weight_per_gpu": tp_expert_weight_per_gpu,
        "ep_weight_per_gpu": ep_expert_weight_per_gpu,
        "tp_comm_per_layer": tp_comm_per_layer,
        "ep_comm_per_layer": ep_comm_per_layer,
        "recommendation": (
            "EP is better when tokens >> experts (communication overhead is small "
            "relative to compute savings from not sharding weights). "
            "TP is better when experts >> tokens (weight sharding is necessary "
            "because each GPU can't hold all expert weights)."
        ),
    }


def demonstrate():
    print("Top-K Router Demo")
    print("=" * 50)
    router = TopKRouter(num_experts=8, top_k=2)
    logits = torch.randn(4, 8)  # 4 tokens, 8 experts
    weights, ids = router.route(logits)
    print(f"Router logits: {logits.shape}")
    print(f"Top-2 weights: {weights.tolist()}")
    print(f"Top-2 expert ids: {ids.tolist()}")
    print(f"Sum per token (should be 1.0): {weights.sum(-1).tolist()}")
    print()

    print("Expert Map (EP=4, 8 experts, linear)")
    for r in range(4):
        n, experts = determine_expert_map(4, r, 8, "linear")
        print(f"  Rank {r}: {n} experts → {experts}")

    print()
    print("EP All2All Communication")
    r = simulate_ep_dispatch_combine(
        tokens=4096, hidden_size=2048, num_experts=256, top_k=8, ep_size=8
    )
    print(f"  Dispatch: {r['dispatch_volume_gb']} GB")
    print(f"  Combine:  {r['combine_volume_gb']} GB")
    print(f"  Total:    {r['total_all2all_gb']} GB")
    print(f"  Note:     {r['note']}")


if __name__ == "__main__":
    demonstrate()
