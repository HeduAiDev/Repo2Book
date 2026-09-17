# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""worker 回报结构：completed_jobs 回传信封就从这里进调度器。

# SOURCE: vllm/v1/outputs.py:L220-L250（KVConnectorOutput）
# SUBTRACTED: logprobs/池化/多模态/投机解码草稿等回报字段——本章的
#   update_connector_output 只读 kv_connector_worker_meta。
"""

from dataclasses import dataclass, field
from typing import Any


# SOURCE: vllm/v1/outputs.py:L223-L248
@dataclass
class KVConnectorOutput:
    # SOURCE: vllm/v1/outputs.py:L223-L248
    """Struct returned by the KV connector."""

    finished_sending: set[str] | None = None
    """[req_ids] finished sending"""
    finished_recving: set[str] | None = None
    """[req_ids] finished recving"""
    kv_connector_stats: Any | None = None
    """The stats from the KV connector."""
    kv_cache_events: Any | None = None
    """The KV cache events from the KV connector."""
    kv_connector_worker_meta: Any | None = None
    """The worker-side metadata (e.g. OffloadingWorkerMetadata)."""
    invalid_block_ids: set[int] = field(default_factory=set)
    """IDs of externally computed KV blocks that failed to load."""

    # SOURCE: vllm/v1/outputs.py:L240-L248
    def is_empty(self):
        # SOURCE: vllm/v1/outputs.py:L240-L248
        return (
            not self.finished_sending
            and not self.finished_recving
            and not self.kv_connector_stats
            and not self.kv_cache_events
            and not self.invalid_block_ids
            and not self.kv_connector_worker_meta
        )


__all__ = ["KVConnectorOutput"]
