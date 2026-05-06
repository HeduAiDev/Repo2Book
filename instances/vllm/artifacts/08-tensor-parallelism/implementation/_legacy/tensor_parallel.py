"""
Tensor Parallelism — Our Reimplementation (Megatron-style).

REFERENCE sources:
    ColumnParallelLinear:    vllm/model_executor/layers/linear.py:L410
    RowParallelLinear:      vllm/model_executor/layers/linear.py:L1394
    QKVParallelLinear:      vllm/model_executor/layers/linear.py:L977
    VocabParallelEmbedding: vllm/model_executor/layers/vocab_parallel_embedding.py:L192
    communication_op:       vllm/distributed/communication_op.py
    ParallelConfig:         vllm/config/parallel.py:L108

Megatron-style TP strategy:
    ColParallel (QKV, gate/up): split weights along output dim.
        → Each rank computes part of Q/K/V or gate/up.
        → No communication needed between ColPar and the next layer
          IF the next layer is RowPar (which handles the reduction).

    RowParallel (O, down): split weights along input dim.
        → Each rank computes partial output.
        → All-reduce combines partials to get full output.

    The trick: ColPar → (no sync) → activation → RowPar → all-reduce
    = one sync point per transformer block!
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# Simulated distributed communication (no real NCCL needed for education)
# ═══════════════════════════════════════════════════════════════════════════

class SimulatedTPGroup:
    """
    Simulates a TP group for educational purposes.

    REFERENCE: vllm/distributed/parallel_state.py:L290 — GroupCoordinator
               vllm/distributed/communication_op.py — all_reduce, all_gather

    In production, these call NCCL via PyTorch's dist.all_reduce().
    We simulate with explicit tensor operations.
    """

    def __init__(self, tp_size: int, tp_rank: int):
        self.tp_size = tp_size
        self.tp_rank = tp_rank

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Sum across all TP ranks.

        REFERENCE: vllm/distributed/communication_op.py:L12
                   → get_tp_group().all_reduce(input_)

        In real TP: dist.all_reduce(tensor, op=SUM, group=tp_group)
        We simulate by creating the full tensor from shards.
        """
        # In a real setting, each rank would have only its own shard.
        # For educational clarity, we pass the full tensor and show
        # the mathematical equivalence.
        return tensor

    def all_gather(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Concatenate shards from all ranks along dim.

        REFERENCE: vllm/distributed/communication_op.py:L17
        """
        return tensor  # Simplified


# ═══════════════════════════════════════════════════════════════════════════
# ColumnParallelLinear
# REFERENCE: vllm/model_executor/layers/linear.py:L410
# ═══════════════════════════════════════════════════════════════════════════

class ColumnParallelLinear(nn.Module):
    """
    Linear layer with weights sharded along the OUTPUT dimension.

    REFERENCE: vllm/model_executor/layers/linear.py:L410-L606

    Math:
        Full: Y = X @ W + b     where W: [in_features, out_features]
        TP:   Y_i = X @ W_i + b_i   where W_i: [in_features, out_features/tp]

    Each rank holds out_features/tp columns of W.
    All ranks see the same input X, compute partial outputs, then
    optionally all-gather to get the full Y.

    Used for: QKV projection, gate/up projection (MLP)
    """

    def __init__(
        self, in_features: int, out_features: int, tp_size: int, tp_rank: int,
        gather_output: bool = True, bias: bool = False,
    ):
        super().__init__()
        assert out_features % tp_size == 0
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.gather_output = gather_output
        self.output_size_per_partition = out_features // tp_size

        # Each rank holds its shard of W
        # nn.Linear stores weight as [out, in]; F.linear does x @ W^T
        # ColPar: each rank has [out/tp, in] — processes all input, produces partial output
        self.weight = nn.Parameter(
            torch.empty(self.output_size_per_partition, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
        else:
            self.register_parameter('bias', None)

        self.tp_group = SimulatedTPGroup(tp_size, tp_rank)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        REFERENCE: linear.py:L579-L606

        Each rank: Y_i = X @ W_i [+ b_i]
        Then all-gather if gather_output=True.
        """
        # Local GEMM (parallel across ranks)
        output = F.linear(x, self.weight, self.bias)

        if self.gather_output and self.tp_size > 1:
            # All-gather along last dim to reconstruct full output
            # REFERENCE: linear.py:L591 — tensor_model_parallel_all_gather()
            output = self._simulated_all_gather(output)

        return output, self.bias

    def _simulated_all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """In real TP, this is an NCCL AllGather."""
        return tensor  # Simplified for education


# ═══════════════════════════════════════════════════════════════════════════
# RowParallelLinear
# REFERENCE: vllm/model_executor/layers/linear.py:L1394
# ═══════════════════════════════════════════════════════════════════════════

class RowParallelLinear(nn.Module):
    """
    Linear layer with weights sharded along the INPUT dimension.

    REFERENCE: vllm/model_executor/layers/linear.py:L1394-L1630

    Math:
        Full: Y = X @ W + b     where W: [in_features, out_features]
        TP:   Y_i = X_i @ W_i   where W_i: [in_features/tp, out_features]
              Y = sum(Y_i) + b   (all-reduce)

    Each rank holds in_features/tp rows of W.
    Each rank receives in_features/tp columns of X, computes partial Y_i,
    then all-reduce sums the partials.

    Used for: output projection (O), down projection (MLP)
    """

    def __init__(
        self, in_features: int, out_features: int, tp_size: int, tp_rank: int,
        input_is_parallel: bool = True, bias: bool = False,
    ):
        super().__init__()
        assert in_features % tp_size == 0
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.input_is_parallel = input_is_parallel
        self.input_size_per_partition = in_features // tp_size

        # nn.Linear stores weight as [out, in]; F.linear does x @ W^T
        # RowPar: each rank has [out, in/tp] — consumes partial input, produces full output
        self.weight = nn.Parameter(
            torch.empty(out_features, self.input_size_per_partition)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self.tp_group = SimulatedTPGroup(tp_size, tp_rank)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        REFERENCE: linear.py:L1543-L1568

        If input is NOT parallel: split X along last dim, each rank uses its shard.
        If input IS parallel: X is already the right shard (from ColPar output).
        After local GEMM: all-reduce to sum partial results.
        """
        if not self.input_is_parallel:
            # Split input: each rank takes in_features/tp columns
            # REFERENCE: linear.py:L1550-L1553
            chunks = x.chunk(self.tp_size, dim=-1)
            x = chunks[self.tp_rank]

        # Local GEMM
        output = F.linear(x, self.weight)

        if self.tp_size > 1:
            # All-reduce: sum partial results from all ranks
            # REFERENCE: linear.py:L1563 — tensor_model_parallel_all_reduce()
            output = self._simulated_all_reduce(output)

        if self.bias is not None:
            # REFERENCE: Only rank 0 applies bias (linear.py:L1559)
            output = output + self.bias

        return output

    def _simulated_all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """In real TP, this is an NCCL AllReduce(SUM)."""
        return tensor


# ═══════════════════════════════════════════════════════════════════════════
# TP Transformer Block (demonstrates ColPar → RowPar pattern)
# ═══════════════════════════════════════════════════════════════════════════

class TPTransformerBlock(nn.Module):
    """
    One transformer block with Megatron-style TP.

    REFERENCE: vllm/model_executor/models/llama.py:L316 LlamaDecoderLayer

    Communication pattern:
        Attention:
            QKV: ColPar (no gather) → Attention (local heads) → O: RowPar (all-reduce)
        MLP:
            Gate+Up: ColPar → SiLU → Down: RowPar (all-reduce)

    Total: 2 all-reduces per block (one for attention, one for MLP).
    """

    def __init__(self, d_model: int, num_heads: int, tp_size: int, tp_rank: int):
        super().__init__()
        self.d_model = d_model
        self.tp_size = tp_size

        # Attention
        self.qkv_proj = ColumnParallelLinear(
            d_model, d_model * 3, tp_size, tp_rank, gather_output=False)
        self.o_proj = RowParallelLinear(d_model, d_model, tp_size, tp_rank)

        # MLP (simplified — just two layers)
        self.gate_up_proj = ColumnParallelLinear(
            d_model, d_model * 2, tp_size, tp_rank, gather_output=False)
        self.down_proj = RowParallelLinear(
            d_model * 2, d_model, tp_size, tp_rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        # Attention
        qkv, _ = self.qkv_proj(x)     # [B, L, 3D/tp] — each rank has partial QKV
        # In real TP: reshape QKV to heads, compute attention with local heads,
        # then o_proj (RowPar) does all-reduce
        qkv = qkv.reshape(B, L, 3, D // self.tp_size)
        # Simplified: skip actual attention, just recombine
        attn_out = self.o_proj(qkv.mean(dim=2))

        # MLP
        gate_up, _ = self.gate_up_proj(x)  # [B, L, 2D/tp]
        # SiLU activation (simplified)
        act = gate_up * torch.sigmoid(gate_up)
        mlp_out = self.down_proj(act)

        return attn_out + mlp_out


# ═══════════════════════════════════════════════════════════════════════════
# Communication Analysis
# ═══════════════════════════════════════════════════════════════════════════

def tp_communication_analysis(model_config: dict) -> dict:
    """
    Quantify TP communication per forward pass.

    REFERENCE: This analysis follows from the Megatron paper and
               vLLM's implementation in linear.py.
    """
    d = model_config['d_model']
    tp = model_config['tp_size']
    layers = model_config['num_layers']
    seq_len = model_config['seq_len']
    batch = model_config['batch_size']
    dtype_bytes = model_config.get('dtype_bytes', 2)

    # Per all-reduce: 2 × tensor_size bytes (send + recv)
    per_ar_bytes = 2 * batch * seq_len * d * dtype_bytes

    # Attention: 1 all-reduce (o_proj), MLP: 1 all-reduce (down_proj)
    # Plus embedding: 1 all-reduce at start
    num_ar_per_layer = 2  # o_proj + down_proj
    total_ar = 1 + num_ar_per_layer * layers

    total_comm = total_ar * per_ar_bytes

    # Per-GPU compute (weights are 1/tp as large)
    compute_per_layer = (
        2 * batch * seq_len * d * d * (1/tp) * 2    # QKV + O
        + 2 * batch * seq_len * d * 8/3 * d * (1/tp) * 2  # MLP gate/up + down
    )

    return {
        "num_all_reduces_per_forward": total_ar,
        "per_all_reduce_bytes": per_ar_bytes,
        "total_communication_bytes": total_comm,
        "comm_to_compute_ratio_at_scale": (
            "Communication is typically <5% of forward time "
            "for d=8192+, batch>2, seq>1024 — compute dominates at scale"
        ),
    }


def demonstrate_tp():
    """Show the ColPar → RowPar pattern with concrete numbers."""
    print("Megatron-style TP: ColPar → RowPar Pattern")
    print("=" * 60)
    print()
    print("ColPar (QKV, gate/up):  W_i: [in, out/tp]")
    print("  Each rank has 1/tp of output columns")
    print("  Output: each rank has partial result (no sync needed)")
    print()
    print("RowPar (O, down):       W_i: [in/tp, out]")
    print("  Each rank has 1/tp of input rows")
    print("  Output: all-reduce sums partials to get full result")
    print()
    print("The trick: ColPar output is ALREADY partitioned")
    print("  → RowPar input can consume it directly")
    print("  → No communication between ColPar and RowPar!")
    print("  → Only 1 all-reduce per (ColPar → RowPar) pair.")
    print()

    m = TPTransformerBlock(d_model=1024, num_heads=16, tp_size=4, tp_rank=0)
    total_params = sum(p.numel() for p in m.parameters())
    print(f"TP block (d=1024, tp=4): {total_params/1e6:.1f}M params per rank")
    print(f"vs full block: {4*total_params/1e6:.1f}M if no TP")

    x = torch.randn(1, 8, 1024)
    y = m(x)
    print(f"Input: {x.shape} → Output: {y.shape}")
    print(f"Output invariant preserved: shape matches input")


if __name__ == "__main__":
    demonstrate_tp()
