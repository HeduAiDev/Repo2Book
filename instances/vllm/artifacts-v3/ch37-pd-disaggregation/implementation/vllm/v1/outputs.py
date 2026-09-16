# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""worker 回报结构：KV 传输的完成信号就从这里回调度器。

# SOURCE: vllm/v1/outputs.py:L220-L330（KVConnectorOutput / ModelRunnerOutput）
# SUBTRACTED: logprobs/池化/多模态/投机解码草稿等回报字段——本章的聚合器
#   （kv_connector/utils.py:L53-L173）只读 kv_connector_output。
"""

from dataclasses import dataclass, field
from typing import Any


# SOURCE: vllm/v1/outputs.py:L223-L248
@dataclass
class KVConnectorOutput:
    # [req_ids]
    finished_sending: set[str] | None = None
    finished_recving: set[str] | None = None
    kv_connector_stats: Any | None = None
    kv_cache_events: Any | None = None
    kv_connector_worker_meta: Any | None = None
    # IDs of externally computed KV blocks that failed to load.
    # Requests referencing these blocks should be rescheduled to recompute them
    invalid_block_ids: set[int] = field(default_factory=set)
    # Configuration describing how many finished sending/receiving
    # notifications should be expected for each request. This allows
    # handshake-based connectors like Nixl to update the KVOutputAggregator.
    # It captures a static setup info and should almost always remain constant
    # for a given connector after discovery. Default value entails no change.
    expected_finished_count: int = 0

    # SOURCE: vllm/v1/outputs.py:L240-L248
    def is_empty(self):
        return (
            not self.finished_sending
            and not self.finished_recving
            and not self.kv_connector_stats
            and not self.kv_cache_events
            and not self.invalid_block_ids
            and not self.kv_connector_worker_meta
        )


# SOURCE: vllm/v1/outputs.py:L260-L330
@dataclass
class ModelRunnerOutput:
    # [num_reqs]
    # SOURCE: vllm/v1/outputs.py:L260-L330
    req_ids: list[str]
    # req_id -> index
    req_id_to_index: dict[str, int]

    # num_reqs x num_generated_tokens
    sampled_token_ids: list[list[int]]
    # [num_reqs]
    logprobs: Any | None = None
    # [num_reqs]
    prompt_logprobs_dict: dict[str, Any] | None = None

    # [num_reqs]
    kv_connector_output: KVConnectorOutput | None = None


__all__ = ["KVConnectorOutput", "ModelRunnerOutput"]
