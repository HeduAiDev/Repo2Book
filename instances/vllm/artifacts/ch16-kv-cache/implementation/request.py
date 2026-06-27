# SPDX-License-Identifier: Apache-2.0
# 只做减法的精简版 —— 忠实子集，与 vLLM 同名同结构同控制流。
# 验收判据：把真实 vLLM 删掉所有 # SUBTRACTED 分支，应当 ≈ 得到本文件。
#
# 本文件提供本章 KV 缓存路径用到的最小 Request 视图与 KVCacheSpec 占位。
# 相对 ch15，本章新增 SlidingWindowSpec / ChunkedLocalAttentionSpec（多注意力类型）
# 与 KVCacheGroupSpec / KVCacheConfig（协调器按组拓扑构造），字段语义与真实
# vllm/v1/kv_cache_interface.py / vllm/v1/request.py 一致。
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


def cdiv(a: int, b: int) -> int:
    # SOURCE: vllm/utils/math_utils.py (cdiv — 向上取整除)
    return -(a // -b)


# SOURCE: vllm/v1/kv_cache_interface.py:L82 (class KVCacheSpec)
@dataclass(frozen=True, kw_only=True)
class KVCacheSpec:
    # SOURCE: vllm/v1/kv_cache_interface.py:L82 (class KVCacheSpec / AttentionSpec)
    """KV cache spec 抽象基类。本章只用到 block_size 与（子类的）窗口字段。"""
    # SUBTRACTED: num_kv_heads / head_size / dtype 等张量形状字段以及 page_size_bytes /
    # max_memory_usage_bytes 等显存计量方法 —— 决定显存布局，不影响分页/哈希/命中/
    # 准入推算的控制流。原 vllm/v1/kv_cache_interface.py:L82 (KVCacheSpec / AttentionSpec)。
    block_size: int


# SOURCE: vllm/v1/kv_cache_interface.py:L175 (class FullAttentionSpec)
@dataclass(frozen=True, kw_only=True)
class FullAttentionSpec(KVCacheSpec):
    # SOURCE: vllm/v1/kv_cache_interface.py:L175 (class FullAttentionSpec)
    """KV cache spec for a full-attention group. 本章只用到 block_size。"""
    # SUBTRACTED: sliding_window / attention_chunk_size 可选字段与 merge_* 合并逻辑
    # （L186-L257）—— 全注意力不带窗口；本章为对照另立独立的 Sliding/Chunked spec 类。
    pass


# SOURCE: vllm/v1/kv_cache_interface.py:L387 (class ChunkedLocalAttentionSpec)
@dataclass(frozen=True, kw_only=True)
class ChunkedLocalAttentionSpec(KVCacheSpec):
    attention_chunk_size: int

    # SOURCE: vllm/v1/kv_cache_interface.py:L390 (max_admission_blocks_per_request)
    def max_admission_blocks_per_request(
        self, max_num_batched_tokens: int, max_model_len: int
    ) -> int:
        """Per-request admission cap, in blocks.

        Single source of truth for both startup pool sizing
        (`max_memory_usage_bytes`) and the runtime admission gate, so requests
        admitted by startup can also be admitted at runtime.
        """
        # During chunked prefill, we hold KV for at most one chunk window.
        num_tokens = min(
            self.attention_chunk_size + max_num_batched_tokens, max_model_len
        )
        return cdiv(num_tokens, self.block_size)

    # SUBTRACTED: max_memory_usage_bytes（L404-L410）—— 启动期显存计量；与运行时
    # 准入闸共用同一 max_admission_blocks_per_request，本章只演示运行时一侧。


# SOURCE: vllm/v1/kv_cache_interface.py:L415 (class SlidingWindowSpec)
@dataclass(frozen=True, kw_only=True)
class SlidingWindowSpec(KVCacheSpec):
    sliding_window: int

    # SUBTRACTED: head_size_v / real_page_size_bytes（L416-L429）—— 显存布局字段，
    # 不影响窗口/分配控制流。

    # SOURCE: vllm/v1/kv_cache_interface.py:L443 (max_admission_blocks_per_request)
    def max_admission_blocks_per_request(
        self, max_num_batched_tokens: int, max_model_len: int
    ) -> int:
        """Per-request admission cap, in blocks.

        Single source of truth for both startup pool sizing
        (`max_memory_usage_bytes`) and the runtime admission gate. Per-request
        real-held blocks plateau at this bound because
        `SlidingWindowManager.remove_skipped_blocks` runs from `allocate_slots`
        before each chunk's `get_num_blocks_to_allocate`.
        """
        # During chunked prefill, we hold KV for the last `sliding_window-1`
        # computed tokens plus the newly scheduled tokens, and never more
        # than `max_model_len`.
        num_tokens = min(
            self.sliding_window - 1 + max_num_batched_tokens, max_model_len
        )
        # +1 because the sliding window may not start from the beginning of
        # the block. E.g. block size 4 and num_token 4 needs two blocks
        # [XXCD][EF] to store the 6-token window [CDEF].
        return cdiv(num_tokens, self.block_size) + 1

    # SUBTRACTED: max_memory_usage_bytes（L453-L462）—— 见 ChunkedLocal 同名说明。


# SOURCE: vllm/v1/kv_cache_interface.py:L756 (class KVCacheGroupSpec)
@dataclass
class KVCacheGroupSpec:
    # SOURCE: vllm/v1/kv_cache_interface.py:L756 (class KVCacheGroupSpec)
    """
    Represents a group of model layers that share the same KV cache block table.
    These layers are regarded as one layer in the KV cache manager.
    """
    # SUBTRACTED: layer_names —— 仅供 worker 侧按层名定位 KV 张量，与分配/协调控制流正交。
    kv_cache_spec: KVCacheSpec
    # Whether this group contains EAGLE/MTP draft attention layers.
    is_eagle_group: bool = False


# SOURCE: vllm/v1/kv_cache_interface.py:L771 (class KVCacheConfig)
@dataclass
class KVCacheConfig:
    # SOURCE: vllm/v1/kv_cache_interface.py:L771 (class KVCacheConfig)
    """The KV cache configuration of a model."""
    # SUBTRACTED: kv_cache_tensors / has_mamba_layers / needs_kv_cache_zeroing
    # （L767, L778-L784）—— 张量初始化布局与 Mamba 判定，本章协调器只读 num_blocks 与
    # kv_cache_groups 的拓扑。
    num_blocks: int
    kv_cache_groups: list[KVCacheGroupSpec]


# SOURCE: vllm/v1/request.py:L40 (class Request) —— 只保留本章 KV 缓存路径用到的字段
class Request:
    # SOURCE: vllm/v1/request.py:L41 (Request.__init__ — 只保留 KV 缓存相关字段)
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: Sequence[int],
        block_hasher: "Callable[[Request], list[Any]] | None" = None,
        mm_features: list | None = None,
        lora_request: Any | None = None,
        cache_salt: str | None = None,
        skip_reading_prefix_cache: bool = False,
    ) -> None:
        self.request_id = request_id
        # SOURCE: vllm/v1/request.py:L86
        self.lora_request = lora_request
        # SOURCE: vllm/v1/request.py:L147
        self.cache_salt: str | None = cache_salt
        # SOURCE: vllm/v1/request.py:L150
        self.mm_features = mm_features or []

        # SOURCE: vllm/v1/request.py:L134 / L155 (all_token_ids 视图)
        self._all_token_ids: list[int] = list(prompt_token_ids)
        self.all_token_ids = self._all_token_ids

        # SOURCE: vllm/v1/request.py:L146
        self.num_computed_tokens = 0
        # SOURCE: vllm/v1/request.py:L145
        self.spec_token_ids: list[int] = []
        # SOURCE: vllm/v1/request.py (num_preemptions — 抢占计数)
        self.num_preemptions = 0
        # SOURCE: vllm/v1/request.py:L129 (_prompt_embeds_per_block_hashes)
        self._prompt_embeds_per_block_hashes: dict[tuple[int, int], bytes] = {}
        # SUBTRACTED: prompt_embeds —— 仅 prompt-embeds 输入触发；常规 token 请求恒 None。
        self.prompt_embeds = None

        # SOURCE: vllm/v1/request.py:L172
        self.block_hashes: list[Any] = []
        # SOURCE: vllm/v1/request.py:L176
        self._block_hasher = block_hasher
        # SOURCE: vllm/v1/request.py:L177
        self.update_block_hashes()
        # SOURCE: vllm/v1/request.py:L179 (skip_reading_prefix_cache)
        self.skip_reading_prefix_cache = skip_reading_prefix_cache

    # SOURCE: vllm/v1/request.py:L217 (append_output_token_ids)
    def append_output_token_ids(self, token_ids: int | list[int]) -> None:
        if isinstance(token_ids, int):
            self._all_token_ids.append(token_ids)
        else:
            self._all_token_ids.extend(token_ids)
        # SOURCE: vllm/v1/request.py:L228
        self.update_block_hashes()

    # SOURCE: vllm/v1/request.py:L230 (update_block_hashes)
    def update_block_hashes(self) -> None:
        """Compute block hashes for any new full blocks and append them."""
        if self._block_hasher is not None:
            self.block_hashes.extend(self._block_hasher(self))

    @property
    def num_tokens(self) -> int:
        # SOURCE: vllm/v1/request.py:L240
        return len(self._all_token_ids)
