# REFERENCE: instances/vllm/source/vllm/v1/kv_cache_interface.py:L80-L205
"""KVCacheSpec — how big is one block, in bytes?

This is the formula that converts a model's attention configuration into a
single number — `page_size_bytes` — which the memory layout calculator needs
to translate "available bytes for KV cache" into "number of GPU blocks".

The formula (`AttentionSpec.real_page_size_bytes`):

    page_size = 2 * block_size * num_kv_heads * head_size * dtype_bytes
                ↑
                K and V

Field names match vLLM exactly. Quantization and NVFP4 layouts are noted but
not implemented — the standard fp16/bf16 path is the focus.
"""

from __future__ import annotations

from dataclasses import dataclass


# REFERENCE: instances/vllm/source/vllm/v1/kv_cache_interface.py:L80-L127
@dataclass(frozen=True)
class KVCacheSpec:
    """Base spec. Subclasses fill in `page_size_bytes` and `max_memory_usage_bytes`."""

    # REFERENCE: instances/vllm/source/vllm/v1/kv_cache_interface.py:L86-L87
    block_size: int  # tokens per block

    @property
    def page_size_bytes(self) -> int:
        # REFERENCE: instances/vllm/source/vllm/v1/kv_cache_interface.py:L89-L97
        raise NotImplementedError

    def max_memory_usage_bytes(self, max_model_len: int) -> int:
        # REFERENCE: instances/vllm/source/vllm/v1/kv_cache_interface.py:L103-L110
        raise NotImplementedError


# REFERENCE: instances/vllm/source/vllm/v1/kv_cache_interface.py:L129-L170
@dataclass(frozen=True)
class AttentionSpec(KVCacheSpec):
    """Standard attention KV cache layout.

    Field name match vLLM exactly. `head_size` (not `head_dim`) — vLLM's
    convention.
    """

    num_kv_heads: int
    head_size: int
    dtype_bytes: int = 2  # fp16/bf16. vLLM uses `torch.dtype` and `get_dtype_size`.

    @property
    def real_page_size_bytes(self) -> int:
        # REFERENCE: instances/vllm/source/vllm/v1/kv_cache_interface.py:L153-L170
        # 2 * block_size * num_kv_heads * head_size * dtype_bytes
        # ↑ K and V interleaved per-token
        # NOT IMPLEMENTED: nvfp4 packed layout (L154-L163), kv-quantization
        # per-token-head scales (L143-L146).
        return (
            2
            * self.block_size
            * self.num_kv_heads
            * self.head_size
            * self.dtype_bytes
        )

    @property
    def page_size_bytes(self) -> int:
        # REFERENCE: instances/vllm/source/vllm/v1/kv_cache_interface.py:L137-L150
        # NOT IMPLEMENTED: page_size_padded for alignment-padded specs.
        return self.real_page_size_bytes


# REFERENCE: instances/vllm/source/vllm/v1/kv_cache_interface.py:L173-L205
@dataclass(frozen=True)
class FullAttentionSpec(AttentionSpec):
    """The common case: dense attention over all tokens.

    `head_size_v` defaults to `head_size` (post-init in vLLM). We require
    callers to set them equal for the demo.
    """

    head_size_v: int | None = None
    sliding_window: int | None = None

    def __post_init__(self) -> None:
        # REFERENCE: instances/vllm/source/vllm/v1/kv_cache_interface.py:L192-L194
        if self.head_size_v is None:
            object.__setattr__(self, "head_size_v", self.head_size)

    # REFERENCE: instances/vllm/source/vllm/v1/kv_cache_interface.py:L196-L204
    def max_memory_usage_bytes(self, max_model_len: int) -> int:
        """Worst-case bytes for one request that fills `max_model_len` tokens.

        SIMPLIFIED: vLLM divides max_model_len by `dcp_world_size *
        pcp_world_size` for context-parallel sharding (L198-L203).
        """
        num_blocks = (max_model_len + self.block_size - 1) // self.block_size
        return num_blocks * self.page_size_bytes


# REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L930-L947
def get_num_blocks(available_memory: int, num_layers: int, page_size: int) -> int:
    """How many blocks fit in `available_memory` across `num_layers` layers?

    The same integer division vLLM uses at `kv_cache_utils.py:L945`. The result
    is the size of the BlockPool used by the scheduler. The "wasted" remainder
    bytes are usually < one block per layer — small, but worth tracking.
    """
    return max(int(available_memory // page_size // num_layers), 0)
