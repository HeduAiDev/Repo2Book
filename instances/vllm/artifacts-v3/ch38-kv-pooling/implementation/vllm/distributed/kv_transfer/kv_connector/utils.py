# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""connector 工具（本章消费面：yield_req_data——增量块号的唯一入口）。

# SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py:yield_req_data 段
# SUBTRACTED: KVOutputAggregator 聚合器与 attn backend 探测（get_current_
#   attn_backends）——聚合面归 ch16/ch37；本章只要「每请求增量块号」。
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


# SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py:yield_req_data
def yield_req_data(
    scheduler_output: "SchedulerOutput",
) -> Iterator[tuple[str, tuple[list[int], ...] | None, bool]]:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py:yield_req_data
    """
    Yields:
        (req_id, new_block_id_groups, preempted)
    """
    # new requests
    for req_data in scheduler_output.scheduled_new_reqs:
        yield req_data.req_id, req_data.block_ids, False

    # cached requests
    cached_reqs = scheduler_output.scheduled_cached_reqs
    yield from zip(
        cached_reqs.req_ids,
        cached_reqs.new_block_ids,
        (req_id in cached_reqs.resumed_req_ids for req_id in cached_reqs.req_ids),
    )


__all__ = ["yield_req_data"]
