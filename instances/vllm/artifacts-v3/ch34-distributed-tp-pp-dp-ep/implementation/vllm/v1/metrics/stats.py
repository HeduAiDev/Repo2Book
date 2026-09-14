# SOURCE: vllm/v1/metrics/stats.py
# ch34 切面：SchedulerStats（LB 打分与 wave 共识的统计载荷，L186-L218 字段子集）。
# SUBTRACTED：观测扩展字段族（prefix_cache/spec_decoding/lora/cudagraph/perf）——
# ch15/ch19/ch08 域。

from __future__ import annotations

from dataclasses import dataclass


# SOURCE: vllm/v1/metrics/stats.py:L186-L214 SchedulerStats —— 本章字段逐字
@dataclass
class SchedulerStats:
    """Stats associated with the scheduler."""
    # SOURCE: vllm/v1/metrics/stats.py:L186-L214（锚点双置）

    num_running_reqs: int = 0

    num_waiting_reqs: int = 0  # length of the "waiting" request queue
    # SUBTRACTED: num_skipped_waiting_reqs（L193）——ch11 域。

    # These are used for internal DP load-balancing.
    step_counter: int = 0
    current_wave: int = 0

    kv_cache_usage: float = 0.0
    # SUBTRACTED: iteration_details/prefix_cache_stats/connector_prefix_cache_
    #   stats/kv_cache_eviction_events/spec_decoding_stats/kv_connector_stats/
    #   waiting_lora_adapters/running_lora_adapters/cudagraph_stats/perf_stats
    #   （L200-L218）——各归其域。
