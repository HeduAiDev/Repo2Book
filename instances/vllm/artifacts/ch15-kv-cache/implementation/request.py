# SPDX-License-Identifier: Apache-2.0
# 只做减法的精简版 —— 忠实子集，与 vLLM 同名同结构同控制流。
# 验收判据：把真实 vLLM 删掉所有 # SUBTRACTED 分支，应当 ≈ 得到本文件。
#
# 本文件提供本章 KV 缓存路径用到的最小 Request 视图与 KVCacheSpec 占位，
# 使精简版可在 host 上运行、可打断点、可数值追踪（不 import vllm）。
# 只保留 block_pool / kv_cache_manager / single_type / coordinator 真正读到的字段，
# 字段语义与真实 vllm/v1/request.py 一致。
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any


# SOURCE: vllm/v1/kv_cache_interface.py (FullAttentionSpec — 本章只用到 block_size)
@dataclass
class FullAttentionSpec:
    """KV cache spec for a full-attention group. 本章只用到 block_size。"""
    # SOURCE: vllm/v1/kv_cache_interface.py (FullAttentionSpec — 本章只用到 block_size)
    # SUBTRACTED: num_kv_heads / head_size / dtype / sliding_window / attention_chunk_size
    # 等张量形状/窗口字段 —— 决定显存布局与窗口，不影响分页/哈希/命中控制流。
    # 原 vllm/v1/kv_cache_interface.py FullAttentionSpec。
    block_size: int


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
        # SOURCE: vllm/v1/request.py:L85
        self.lora_request = lora_request
        # SOURCE: vllm/v1/request.py:L146
        self.cache_salt: str | None = cache_salt
        # SOURCE: vllm/v1/request.py:L149
        self.mm_features = mm_features or []

        # SOURCE: vllm/v1/request.py:L133 / L155 (all_token_ids 视图)
        self._all_token_ids: list[int] = list(prompt_token_ids)
        self.all_token_ids = self._all_token_ids

        # SOURCE: vllm/v1/request.py:L145
        self.num_computed_tokens = 0
        # SOURCE: vllm/v1/request.py:L144
        self.spec_token_ids: list[int] = []
        # SOURCE: vllm/v1/request.py (num_preemptions — 抢占计数，f11)
        self.num_preemptions = 0
        # SOURCE: vllm/v1/request.py:L128 (_prompt_embeds_per_block_hashes)
        self._prompt_embeds_per_block_hashes: dict[tuple[int, int], bytes] = {}
        # SUBTRACTED: prompt_embeds —— 仅 prompt-embeds 输入触发（_gen_prompt_embeds_extra_hash_keys
        # 已 SUBTRACTED）；常规 token 请求恒 None。原 vllm/v1/request.py:L121。
        self.prompt_embeds = None

        # SOURCE: vllm/v1/request.py:L171
        self.block_hashes: list[Any] = []
        # SOURCE: vllm/v1/request.py:L175
        self._block_hasher = block_hasher
        # SOURCE: vllm/v1/request.py:L176
        self.update_block_hashes()
        # SOURCE: vllm/v1/request.py:L178 (skip_reading_prefix_cache — 需 prompt
        # logprobs / 全 pooling 时跳过前缀缓存读)
        self.skip_reading_prefix_cache = skip_reading_prefix_cache

    # SOURCE: vllm/v1/request.py:L211 (append_output_token_ids)
    def append_output_token_ids(self, token_ids: int | list[int]) -> None:
        if isinstance(token_ids, int):
            self._all_token_ids.append(token_ids)
        else:
            self._all_token_ids.extend(token_ids)
        # SOURCE: vllm/v1/request.py:L222
        self.update_block_hashes()

    # SOURCE: vllm/v1/request.py:L224 (update_block_hashes)
    def update_block_hashes(self) -> None:
        """Compute block hashes for any new full blocks and append them."""
        if self._block_hasher is not None:
            self.block_hashes.extend(self._block_hasher(self))

    @property
    def num_tokens(self) -> int:
        # SOURCE: vllm/v1/request.py:L234
        return len(self._all_token_ids)
