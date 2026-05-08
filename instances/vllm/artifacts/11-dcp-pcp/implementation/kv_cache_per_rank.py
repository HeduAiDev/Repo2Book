"""HBM-per-rank KV cache accounting under DCP and PCP.

This is the **capacity** axis — the reason CP exists at all. With
``total_cp = pcp * dcp``, each rank stores only ``1/total_cp`` of the
KV cache, multiplying the longest sequence the system can serve.

REFERENCE: vllm/v1/kv_cache_interface.py:L195-L205 (max_memory_usage_bytes)
REFERENCE: vllm/config/parallel.py:L115 (prefill_context_parallel_size)
REFERENCE: vllm/config/parallel.py:L310 (decode_context_parallel_size)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def cdiv(a: int, b: int) -> int:
    """Ceiling division — matches vllm.utils.cdiv."""
    return -(-a // b)


@dataclass(frozen=True)
class KVCacheSpec:
    """Subset of vLLM's ``AttentionSpec`` relevant for the chapter."""

    num_layers: int
    num_kv_heads: int  # K + V heads per layer (Llama uses GQA so this is reduced)
    head_size: int
    block_size: int
    dtype_bytes: int = 2  # bf16 / fp16

    @property
    def page_size_bytes(self) -> int:
        """Bytes per KV cache "page" (vLLM block).

        Each block stores ``block_size`` tokens. Each token has
        ``2 * num_kv_heads * head_size`` (K + V) cells of ``dtype_bytes``.
        """
        return 2 * self.block_size * self.num_kv_heads * self.head_size * self.dtype_bytes

    # REFERENCE: vllm/v1/kv_cache_interface.py:L196-L204
    def max_memory_usage_bytes(
        self,
        max_model_len: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> int:
        """Per-rank HBM bytes for KV cache holding the longest sequence.

        Source body (verbatim semantics)::

            max_model_len = vllm_config.model_config.max_model_len
            dcp = parallel_config.decode_context_parallel_size
            pcp = parallel_config.prefill_context_parallel_size
            if dcp * pcp > 1:
                max_model_len = cdiv(max_model_len, dcp * pcp)
            return cdiv(max_model_len, self.block_size) * self.page_size_bytes
        """
        # REFERENCE: vllm/v1/kv_cache_interface.py:L201-L203
        if dcp_world_size * pcp_world_size > 1:
            max_model_len = cdiv(max_model_len, dcp_world_size * pcp_world_size)
        # REFERENCE: vllm/v1/kv_cache_interface.py:L204
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes

    def per_layer_bytes_naive(self, seq_len: int) -> int:
        """Naive per-layer KV bytes — no CP, no block padding.

        ``seq_len * 2 * num_kv_heads * head_size * dtype_bytes``

        Used for §1 demo where we walk the headline 33.5 GB number
        for a 128K x 80-layer x 8-KV-head x bf16 model.
        """
        return seq_len * 2 * self.num_kv_heads * self.head_size * self.dtype_bytes


# Production-realistic specs used by the demos.
LLAMA_70B_KV_SPEC = KVCacheSpec(
    num_layers=80,
    num_kv_heads=8,  # GQA: 64 attention heads / 8 KV heads
    head_size=128,
    block_size=16,
    dtype_bytes=2,
)


# REFERENCE: vllm/v1/kv_cache_interface.py:L196-L199 (max_model_len from vllm_config)
# REFERENCE: vllm/v1/kv_cache_interface.py:L200-L202 (per-rank len comment "each dcp rank only need save")
def hbm_naive_total(seq_len: int, spec: KVCacheSpec) -> int:
    """Total KV bytes (no CP) summed across all layers.

    For Llama-70B at seq_len=128K::

        128K * 80 * 8 * 128 * 2 (K+V) * 2 (bf16)
        = 33,285,996,544 bytes  ~= 33.5 GB

    This is per-request; with B requests in flight it scales by B.
    """
    return seq_len * spec.num_layers * 2 * spec.num_kv_heads * spec.head_size * spec.dtype_bytes


def hbm_per_rank(
    seq_len: int,
    spec: KVCacheSpec,
    dcp: int,
    pcp: int,
) -> int:
    """Total per-rank KV bytes under (dcp, pcp).

    REFERENCE: vllm/v1/kv_cache_interface.py:L201-L203 (cdiv(max_model_len, dcp*pcp))
    REFERENCE: vllm/v1/kv_cache_interface.py:L204 (cdiv(...,block_size) * page_size_bytes)
    """
    total_cp = dcp * pcp
    if total_cp > 1:
        per_rank_len = cdiv(seq_len, total_cp)
    else:
        per_rank_len = seq_len
    # Sum across layers and apply block padding.
    blocks = cdiv(per_rank_len, spec.block_size)
    return spec.num_layers * blocks * spec.page_size_bytes


def fmt_gb(n_bytes: int) -> str:
    """Format bytes as GB to 1 decimal."""
    return f"{n_bytes / (1024 ** 3):.1f} GB"
