# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# MooncakeStore 的数据类（本章消费面：LoadSpec + meta 容器）。
#
# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py:L1-L450
# SUBTRACTED（减法计划删除项 6：Mooncake 实现体不进精简版）：
#   ChunkedTokenDatabase / BlobBlockHashes / KeyMetadata / PoolKey /
#   ReqMeta / RequestTracker 及指纹对账（m14 由 embed_excerpts+docs 承载
#   叙事）；保留 LoadSpec（查询命中登记）与 MooncakeStoreConnectorMetadata
#   容器骨架。
from dataclasses import dataclass, field
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py:L303-L311
@dataclass
class LoadSpec:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py:L303-L311
    """Specification for loading KV cache from external store."""

    vllm_cached_tokens: int
    kvpool_cached_tokens: int
    can_load: bool
    token_len: int = 0


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py:L437-L450
class MooncakeStoreConnectorMetadata(KVConnectorMetadata):
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py:L437-L450
    """Metadata passed from scheduler to worker."""

    def __init__(
        self,
        unfinished_request_ids: set[str],
        preempted_req_ids: set[str],
    ):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py:L440-L450
        self.requests: list[Any] = []
        self.unfinished_request_ids = unfinished_request_ids
        self.preempted_req_ids = preempted_req_ids

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py:L449-L450
    def add_request(self, req_meta: Any) -> None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py:L449-L450
        self.requests.append(req_meta)
