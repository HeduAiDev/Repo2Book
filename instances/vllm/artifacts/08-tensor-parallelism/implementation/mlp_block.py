"""mlp_block — Llama-style SwiGLU MLP with TP, mirroring LlamaMLP.

Reproduces the structure of:

    instances/vllm/source/vllm/model_executor/models/llama.py:L81-L121
        class LlamaMLP:
            self.gate_up_proj = MergedColumnParallelLinear(
                input_size=hidden_size,
                output_sizes=[intermediate_size] * 2,
                ...
            )
            self.down_proj = RowParallelLinear(
                input_size=intermediate_size,
                output_size=hidden_size,
                ...
            )
            self.act_fn = SiluAndMul()

            def forward(self, x):
                x, _ = self.gate_up_proj(x)   # one fused col-parallel matmul
                x = self.act_fn(x)            # SiLU(gate) * up — element-wise
                x, _ = self.down_proj(x)      # row-parallel + ONE all-reduce
                return x

The Megatron insight (Trap-E recap): the col→row composition needs ONE
all-reduce per block. The intermediate `silu(gate) * up` stays SHARDED
across ranks throughout — no all-gather between gate/up and down. If you
naively all-gathered between them, you'd double the communication.

References:
- llama.py:L81-L121      LlamaMLP
- linear.py:L609-L976    MergedColumnParallelLinear
- linear.py:L1394-L1577  RowParallelLinear
- vllm/model_executor/layers/activation.py  SiluAndMul (we inline silu_and_mul here)
"""

from __future__ import annotations

import numpy as np

from .column_parallel import MergedColumnParallelLinear
from .row_parallel import RowParallelLinear


# REFERENCE: instances/vllm/source/vllm/model_executor/layers/activation.py
# REFERENCE: instances/vllm/source/vllm/model_executor/layers/activation.py
# class SiluAndMul — the production op. We inline a numpy version below.
def silu_and_mul(x: np.ndarray) -> np.ndarray:
    """SiluAndMul: input is [..., 2*ffn] (gate concat up); split, silu(gate)*up."""
    half = x.shape[-1] // 2
    gate = x[..., :half]
    up = x[..., half:]
    silu = gate / (1.0 + np.exp(-gate))  # x * sigmoid(x)
    return silu * up


def silu_and_mul_per_rank(x_shards: list[np.ndarray]) -> list[np.ndarray]:
    """Apply silu_and_mul to each rank's shard separately.

    This is element-wise so it works on sharded data without communication.
    Each rank's input shard has shape [..., 2*(ffn/p)] and we split internally
    into [gate_per_rank, up_per_rank] of shape [..., ffn/p] each.
    """
    return [silu_and_mul(x) for x in x_shards]


# REFERENCE: instances/vllm/source/vllm/model_executor/models/llama.py:L81-L121
class LlamaMLPTP:
    """SwiGLU MLP with TP: MergedColumn (gate+up) → SiluAndMul → RowParallel.

    Constants (from llama.py):
        gate_up_proj.output_sizes = [intermediate_size] * 2
        down_proj.input_size      = intermediate_size
        down_proj.output_size     = hidden_size
        hidden_act                = "silu"  (only silu supported in Llama)

    Communication accounting (the Megatron win):
        forward() executes EXACTLY ONE all-reduce per call (the down_proj's
        reduce_results=True). gate_up_proj does no collective; SiLU does
        no collective; down_proj does one all-reduce.

    The `count_collectives()` method exposes this for Demo §5 verification.
    """

    # REFERENCE: llama.py:L82-L115 — __init__
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        tp_size: int = 1,
        bias: bool = False,
        params_dtype: np.dtype = np.float32,
        prefix: str = "",
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.tp_size = tp_size
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size, intermediate_size],
            tp_size=tp_size,
            bias=bias,
            gather_output=False,
            params_dtype=params_dtype,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            tp_size=tp_size,
            bias=bias,
            input_is_parallel=True,
            reduce_results=True,
            params_dtype=params_dtype,
            prefix=f"{prefix}.down_proj",
        )
        self._collective_count = 0

    # REFERENCE: instances/vllm/source/vllm/model_executor/models/llama.py:L439-L443
    # vLLM's stacked_params_mapping splits checkpoint's gate_proj/up_proj into
    # gate_up_proj (shard_id 0,1) and qkv_proj into q/k/v. Our load_weights
    # below mimics the gate_up_proj fuse step.
    def load_weights(
        self,
        W_gate: np.ndarray,  # [hidden, intermediate]
        W_up: np.ndarray,    # [hidden, intermediate]
        W_down: np.ndarray,  # [intermediate, hidden]
    ) -> None:
        """Loads three vLLM-checkpoint-shaped weights into the fused TP layers.

        gate and up are concatenated into a single fused weight then passed
        through the MergedColumnParallel loader (which shards each segment
        independently).
        """
        W_gate_up = np.concatenate([W_gate, W_up], axis=-1)
        self.gate_up_proj.load_weight(A_full=W_gate_up)
        self.down_proj.load_weight(A_full=W_down)

    # REFERENCE: llama.py:L117-L121 — forward
    def forward(self, x: np.ndarray) -> tuple[np.ndarray, int]:
        """One Llama MLP forward pass with collective accounting.

        Returns (output, collectives_used_in_this_call). `collectives_used`
        equals 1 when tp_size > 1 (the all-reduce inside down_proj),
        otherwise 0.
        """
        before = self._collective_count
        # Step 1: column-parallel gate+up — NO collective.
        x_shards = self.gate_up_proj.forward(x)  # list of [..., 2*ffn/p]
        # Step 2: SiluAndMul element-wise per rank — NO collective.
        z_shards = silu_and_mul_per_rank(x_shards)  # list of [..., ffn/p]
        # Step 3: row-parallel down_proj — ONE all-reduce inside down_proj.forward.
        y = self.down_proj.forward(z_shards)
        if self.tp_size > 1:
            self._collective_count += 1
        return y, self._collective_count - before

    def count_collectives(self) -> int:
        """Total collectives observed across all forward() calls so far."""
        return self._collective_count

    def reset_collective_count(self) -> None:
        self._collective_count = 0


def reference_unsharded_mlp(
    x: np.ndarray, W_gate: np.ndarray, W_up: np.ndarray, W_down: np.ndarray
) -> np.ndarray:
    """Single-GPU reference MLP — used to verify TP equivalence."""
    gate = x @ W_gate
    up = x @ W_up
    silu = gate / (1.0 + np.exp(-gate))
    return (silu * up) @ W_down


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    H, F = 256, 512  # hidden, intermediate (ffn)
    batch_seq = 8
    x = rng.standard_normal((batch_seq, H)).astype(np.float32) * 0.5
    W_gate = rng.standard_normal((H, F)).astype(np.float32) * 0.02
    W_up = rng.standard_normal((H, F)).astype(np.float32) * 0.02
    W_down = rng.standard_normal((F, H)).astype(np.float32) * 0.02

    y_ref = reference_unsharded_mlp(x, W_gate, W_up, W_down)
    for tp in (1, 2, 4):
        mlp = LlamaMLPTP(hidden_size=H, intermediate_size=F, tp_size=tp)
        mlp.load_weights(W_gate, W_up, W_down)
        y_tp, ncoll = mlp.forward(x)
        diff = float(np.max(np.abs(y_ref - y_tp)))
        print(f"LlamaMLPTP  tp={tp}  collectives_per_forward={ncoll}  "
              f"max_abs_diff={diff:.3e}")
