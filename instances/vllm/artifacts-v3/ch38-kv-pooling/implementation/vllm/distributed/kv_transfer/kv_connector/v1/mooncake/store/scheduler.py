# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# MooncakeStore 的调度器侧（本章消费面：lookup 查询路径）。
#
# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py:L1-L448
# SUBTRACTED（减法计划删除项 6：Mooncake 实现体不进精简版）：
#   · build_connector_meta 的 RequestTracker/ReqMeta 全套（L171-L395）——
#     chunk 账本与 partial-tail offload 是 worker 双线程实现体的对端。
#   · request_finished 的 delay_free 判定（L397-L421，依赖 tracker）与
#     reset_store 的 master remove_all RPC（L423-L448）。
#   · 细粒度命中对齐面（partial_hash_hits_enabled 与 _hash_block_size 推导、
#     resolve_kv_cache_block_sizes——ch14 定账 + coordinator 深枝）。
#   保留 get_num_new_matched_tokens 全文（lookup_async None=稍后再问、命中
#   差值 (N−computed, load_async) 与 OffloadingConnector 完全同构）。
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import LoadSpec
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (
    LookupKeyClient,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request

logger = init_logger(__name__)


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py:L50-L448
class MooncakeStoreScheduler:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py:L50-L51
    """Scheduler-side component for MooncakeStoreConnector."""

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py:L53-L79
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py:L58-L65
        assert vllm_config.kv_transfer_config is not None
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        kvc_extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        self.load_async = kvc_extra_config.get("load_async", True)
        self.lookup_async = kvc_extra_config.get("lookup_async", False)
        # Skips lookup CPU cost on instances that never load KV from the store.
        self.enable_lookup = kvc_extra_config.get("enable_lookup", True)
        self.client = LookupKeyClient(vllm_config)

        # SUBTRACTED: resolve_kv_cache_block_sizes 与 partial_hash_hits 的
        #   细粒度对齐面（L67-L73）——ch14 定账 + coordinator 深枝。

        # Per-request state
        self.load_specs: dict[str, LoadSpec] = {}  # to be loaded

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py:L81-L134
    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py:L81-L134
        """Check for external KV cache hit.

        Returns ``(None, False)`` when an async lookup is still in flight,
        signaling the scheduler to retry this request on a later step.
        """
        if not self.enable_lookup:
            return 0, False

        # Fine-grained hits may land on a hash boundary inside a block; without
        # partial hits, prefixes shorter than one physical block are skipped.
        # SUBTRACTED: align 对齐门（L96-L100）——细粒度命中对齐面未进精简版。
        num_external_hit_tokens = self.client.lookup(
            request.request_id,
            request.num_tokens,
            request.block_hashes,
            non_block=self.lookup_async,
        )
        if num_external_hit_tokens is None:
            # Lookup not ready yet; scheduler will retry on a later step.
            return None, False

        if num_external_hit_tokens < num_computed_tokens:
            need_to_allocate = 0
        else:
            need_to_allocate = num_external_hit_tokens - num_computed_tokens

        logger.debug(
            "Reqid: %s, Total tokens %d, kvpool hit tokens: %d, need to load: %d",
            request.request_id,
            request.num_tokens,
            num_external_hit_tokens,
            need_to_allocate,
        )

        if need_to_allocate <= 0:
            return 0, False

        self.load_specs[request.request_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            kvpool_cached_tokens=num_external_hit_tokens,
            can_load=False,
        )

        return need_to_allocate, self.load_async

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py:L136-L169
    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py:L136-L169
        """Update state after block allocation."""
        # SUBTRACTED: _unfinished_requests/_request_trackers 簿记（L143-L148）
        #   ——tracker 账本随 build_connector_meta 删除（删除项 6）。

        if request.request_id not in self.load_specs:
            return

        if num_external_tokens == 0:
            self.load_specs[request.request_id].can_load = False
            return

        assert (
            num_external_tokens > 0
            and num_external_tokens
            == self.load_specs[request.request_id].kvpool_cached_tokens
            - self.load_specs[request.request_id].vllm_cached_tokens
        ), (
            f"Mismatch in number of tokens: {num_external_tokens} vs "
            f"{self.load_specs[request.request_id].kvpool_cached_tokens} - "
            f"{self.load_specs[request.request_id].vllm_cached_tokens}"
            f" for request {request.request_id}"
        )

        self.load_specs[request.request_id].can_load = True

    # SUBTRACTED: build_connector_meta（L171-L395）/ request_finished
    #   （L397-L421）/ reset_store（L423-L448）——删除项 6：chunk 账本与
    #   master RPC 的实现体（worker 双线程）未进精简版。

    def build_connector_meta(self, scheduler_output) -> Any:  # noqa: D102
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py:L171-L395
        raise NotImplementedError(
            "MooncakeStoreScheduler.build_connector_meta requires the "
            "RequestTracker/ReqMeta pipeline (not part of this reduced build); "
            "see store/scheduler.py:L171-L395."
        )

    def request_finished(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, Any | None]:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py:L397-L421
        raise NotImplementedError(
            "MooncakeStoreScheduler.request_finished requires the tracker "
            "ledger (not part of this reduced build); see store/scheduler.py:L397-L421."
        )
