"""qkv_parallel — pedagogical mirror of vLLM's QKVParallelLinear.

Reproduces the class shape and head-sharding semantics of:

    instances/vllm/source/vllm/model_executor/layers/linear.py:L977-L1393
        class QKVParallelLinear(ColumnParallelLinear)

This is **column-parallel along the HEAD dimension**, not arbitrary feature
columns (Trap-C). The math:

    total_num_heads     = total Q heads (e.g., 32 for Llama-7B)
    total_num_kv_heads  = total KV heads (== total_num_heads for MHA;
                          smaller for MQA/GQA — Llama-3-70B has 8)
    head_size           = per-head dim (e.g., 128)
    tp_size             = number of TP ranks

    num_heads_per_rank      = divide(total_num_heads, tp_size)
    if tp_size >= total_num_kv_heads:
        # GQA + small KV count: replicate KV across rank-groups.
        num_kv_heads_per_rank   = 1
        num_kv_head_replicas    = divide(tp_size, total_num_kv_heads)
    else:
        num_kv_heads_per_rank   = divide(total_num_kv_heads, tp_size)
        num_kv_head_replicas    = 1

The fused QKV weight is shape [hidden, q_full + k_full + v_full] where
the three regions are head-major: [head0_q | head1_q | ... | head0_k | ...].

References:
- linear.py:L1029-L1043  num_heads / num_kv_heads / num_kv_head_replicas
- linear.py:L1043-L1047  output_sizes triple [q, k, v]
- linear.py:L1062-L1090  shard_id mapping ('q', 'k', 'v')
- llama.py:L142-L172     instantiation in LlamaAttention
"""

from __future__ import annotations

import numpy as np

from .tp_math import divide
from .column_parallel import ColumnParallelLinear


# REFERENCE: instances/vllm/source/vllm/model_executor/layers/linear.py:L977-L1393
class QKVParallelLinear(ColumnParallelLinear):
    """Fused Q/K/V projection with TP along the head dimension.

    Inherits from ColumnParallelLinear because at the linear-algebra level it
    IS column parallel — but the columns are blocked into heads, not arbitrary
    features. The `output_sizes` triple records the q/k/v segments so the
    parent's per-segment loader Just Works.
    """

    # REFERENCE: linear.py:L1005-L1060 — __init__
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        tp_size: int = 1,
        bias: bool = False,
        params_dtype: np.dtype = np.float32,
        prefix: str = "",
    ) -> None:
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads

        # REFERENCE: linear.py:L1030 — heads divided by tp_size
        self.num_heads = divide(self.total_num_heads, tp_size)

        # REFERENCE: linear.py:L1031-L1036 — GQA replication branch
        if tp_size >= self.total_num_kv_heads:
            # KV heads fewer than ranks → replicate.
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1

        # The fused output_size: q + k + v, all *tp_size to undo the per-rank
        # division so super().__init__'s `output_size_per_partition` lands on
        # exactly num_heads/num_kv_heads * head_size.
        # REFERENCE: linear.py:L1037-L1047
        input_size = self.hidden_size
        output_size = (
            self.num_heads * self.head_size
            + self.num_kv_heads * self.head_size  # K
            + self.num_kv_heads * self.head_size  # V (we use head_size for V too)
        ) * tp_size
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,       # q segment (full)
            self.num_kv_heads * self.head_size * tp_size,    # k segment (full)
            self.num_kv_heads * self.head_size * tp_size,    # v segment (full)
        ]

        # If KV heads are replicated (num_kv_head_replicas > 1), the "full" k/v
        # segment we provide for sharding is total_num_kv_heads * head_size,
        # but per-rank each rank still needs num_kv_heads (== 1) * head_size.
        # The trick: we synthesize a replicated full weight by tiling the
        # original kv weight `num_kv_head_replicas` times along the head dim
        # before sharding. See the load_weight override below.

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            tp_size=tp_size,
            bias=bias,
            gather_output=False,
            params_dtype=params_dtype,
            prefix=prefix,
        )

    # REFERENCE: linear.py:L1141-L1393 — weight_loader_v2 with shard_id 'q'/'k'/'v'.
    # In real vLLM the loader is called THREE times (once per shard_id) since
    # checkpoints store q_proj, k_proj, v_proj separately. Here we accept all
    # three as inputs to keep the demo simple.
    def load_qkv_weights(
        self,
        Wq_full: np.ndarray,  # [hidden, total_num_heads * head_size]
        Wk_full: np.ndarray,  # [hidden, total_num_kv_heads * head_size]
        Wv_full: np.ndarray,  # [hidden, total_num_kv_heads * head_size]
    ) -> None:
        """Replicate KV (when needed) and dispatch to the per-segment loader."""
        assert Wq_full.shape == (
            self.hidden_size, self.total_num_heads * self.head_size,
        )
        assert Wk_full.shape == (
            self.hidden_size, self.total_num_kv_heads * self.head_size,
        )
        assert Wv_full.shape == Wk_full.shape

        # REFERENCE: linear.py:L1031-L1036 — when tp_size >= total_num_kv_heads,
        # KV is replicated num_kv_head_replicas times. We synthesize this by
        # tiling the per-head KV blocks BEFORE we ask the parent loader to
        # shard along output_sizes (which are scaled to *tp_size).
        if self.num_kv_head_replicas > 1:
            # Original: [head_kv_0 | head_kv_1 | ... | head_kv_{Hkv-1}]
            # We need: [head_kv_0]*replicas | [head_kv_1]*replicas | ...
            #          (one replica per consumer rank within a kv-head's group)
            # Reshape to [hidden, total_num_kv_heads, head_size], then
            # repeat each KV head along axis=1 by `num_kv_head_replicas`.
            def replicate(W):
                W3 = W.reshape(self.hidden_size, self.total_num_kv_heads, self.head_size)
                W3r = np.repeat(W3, self.num_kv_head_replicas, axis=1)
                return W3r.reshape(self.hidden_size, -1)
            Wk_eff = replicate(Wk_full)
            Wv_eff = replicate(Wv_full)
        else:
            Wk_eff = Wk_full
            Wv_eff = Wv_full

        # Concat into one fused weight matching the parent's expected shape.
        W_fused = np.concatenate([Wq_full, Wk_eff, Wv_eff], axis=-1)
        assert W_fused.shape == (self.hidden_size, sum(self.output_sizes)), (
            f"fused QKV: got {W_fused.shape}, want "
            f"({self.hidden_size}, {sum(self.output_sizes)})"
        )
        # Defer to MergedColumn-style segment-aware loader inherited from the
        # parent (we're a column parallel with output_sizes set).
        super().load_weight(A_full=W_fused, b_full=None)

    # REFERENCE: instances/vllm/source/vllm/model_executor/models/llama.py:L228-L229
    # qkv, _ = self.qkv_proj(hidden_states)
    # q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    def split_qkv(self, qkv_per_rank: list[np.ndarray]) -> dict:
        """Split each rank's fused [..., q_local + k_local + v_local] output
        into per-rank q, k, v shards. Mirrors llama.py:L228-L229's
            qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        but does it for ALL ranks at once.
        """
        q_size = self.num_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size
        offsets = [0, q_size, q_size + kv_size, q_size + 2 * kv_size]
        out: dict[str, list[np.ndarray]] = {"q": [], "k": [], "v": []}
        for Y_r in qkv_per_rank:
            out["q"].append(Y_r[..., offsets[0] : offsets[1]])
            out["k"].append(Y_r[..., offsets[1] : offsets[2]])
            out["v"].append(Y_r[..., offsets[2] : offsets[3]])
        return out

    # REFERENCE: instances/vllm/source/vllm/model_executor/layers/linear.py:L1062-L1090
    # _get_shard_offset_mapping / _get_shard_size_mapping — vLLM's per-shard-id
    # offset table. Our split_qkv uses the same (q, k, v) layout: q first,
    # then k, then v. The "total" entry equals q+k+v_size_per_rank.
    def per_rank_summary(self) -> dict:
        """Diagnostic dict: how many heads each rank holds. Used by Demo §4."""
        return {
            "tp_size": self.tp_size,
            "total_num_heads": self.total_num_heads,
            "total_num_kv_heads": self.total_num_kv_heads,
            "num_heads_per_rank": self.num_heads,
            "num_kv_heads_per_rank": self.num_kv_heads,
            "num_kv_head_replicas": self.num_kv_head_replicas,
            "q_size_per_rank": self.num_heads * self.head_size,
            "kv_size_per_rank": self.num_kv_heads * self.head_size,
        }


if __name__ == "__main__":
    # Sanity: MHA case (Llama-7B-ish: 32 heads, 32 kv heads, head_size=128, tp=4).
    rng = np.random.default_rng(0)
    H = 4096
    qkv = QKVParallelLinear(
        hidden_size=H, head_size=128, total_num_heads=32,
        total_num_kv_heads=32, tp_size=4,
    )
    Wq = rng.standard_normal((H, 32 * 128)).astype(np.float32) * 0.02
    Wk = rng.standard_normal((H, 32 * 128)).astype(np.float32) * 0.02
    Wv = rng.standard_normal((H, 32 * 128)).astype(np.float32) * 0.02
    qkv.load_qkv_weights(Wq, Wk, Wv)
    X = rng.standard_normal((2, H)).astype(np.float32)
    Y = qkv.forward(X)  # list of 4 per-rank fused outputs
    splits = qkv.split_qkv(Y)
    # Reference: do the three matmuls directly on the unsharded weights.
    q_ref = X @ Wq
    k_ref = X @ Wk
    v_ref = X @ Wv
    # Concat per-rank q's, etc., compare to the unsharded direct matmul.
    q_all = np.concatenate(splits["q"], axis=-1)
    k_all = np.concatenate(splits["k"], axis=-1)
    v_all = np.concatenate(splits["v"], axis=-1)
    print("MHA tp=4:")
    print(f"  q diff = {np.max(np.abs(q_ref - q_all)):.3e}")
    print(f"  k diff = {np.max(np.abs(k_ref - k_all)):.3e}")
    print(f"  v diff = {np.max(np.abs(v_ref - v_all)):.3e}")
    print(f"  per_rank_summary: {qkv.per_rank_summary()}")

    # GQA case (Llama-3-70B-ish: 64 heads, 8 kv heads, head=128, tp=8).
    qkv2 = QKVParallelLinear(
        hidden_size=H, head_size=128, total_num_heads=64,
        total_num_kv_heads=8, tp_size=8,
    )
    Wq2 = rng.standard_normal((H, 64 * 128)).astype(np.float32) * 0.02
    Wk2 = rng.standard_normal((H, 8 * 128)).astype(np.float32) * 0.02
    Wv2 = rng.standard_normal((H, 8 * 128)).astype(np.float32) * 0.02
    qkv2.load_qkv_weights(Wq2, Wk2, Wv2)
    Y2 = qkv2.forward(X)
    splits2 = qkv2.split_qkv(Y2)
    q_ref2 = X @ Wq2
    q_all2 = np.concatenate(splits2["q"], axis=-1)
    print(f"GQA tp=8 (num_kv_heads=8 == tp_size): q diff = {np.max(np.abs(q_ref2 - q_all2)):.3e}")
    print(f"  per_rank_summary: {qkv2.per_rank_summary()}")

    # GQA + replication case: 8 KV heads, tp=16 -> each rank holds 1 KV head,
    # but 16 > 8 so KV is replicated 2× per rank-pair.
    qkv3 = QKVParallelLinear(
        hidden_size=H, head_size=128, total_num_heads=64,
        total_num_kv_heads=8, tp_size=16,
    )
    Wq3 = rng.standard_normal((H, 64 * 128)).astype(np.float32) * 0.02
    Wk3 = rng.standard_normal((H, 8 * 128)).astype(np.float32) * 0.02
    Wv3 = rng.standard_normal((H, 8 * 128)).astype(np.float32) * 0.02
    qkv3.load_qkv_weights(Wq3, Wk3, Wv3)
    print(f"GQA tp=16 (num_kv_heads=8 < tp): per_rank_summary = {qkv3.per_rank_summary()}")
    # Verify Q sharding still works.
    Y3 = qkv3.forward(X)
    splits3 = qkv3.split_qkv(Y3)
    q_ref3 = X @ Wq3
    q_all3 = np.concatenate(splits3["q"], axis=-1)
    print(f"  q diff = {np.max(np.abs(q_ref3 - q_all3)):.3e}")
