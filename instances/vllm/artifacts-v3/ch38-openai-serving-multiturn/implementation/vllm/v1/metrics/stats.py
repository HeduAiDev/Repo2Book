# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/v1/metrics/stats.py —— HOST SEAM（最小承载）：本章消费面
# 只需 RequestStateStats（RequestOutput.metrics 字段）、LoRARequestStates
# （abort_requests 的 lora 清理）、IterationStats/SchedulerStats/PrefillStats
# 形状（output_processor/async_llm 注释与签名位）；Prometheus 记录器族归
# 观测域。
from collections import defaultdict
from dataclasses import dataclass


# SOURCE: vllm/v1/metrics/stats.py:L218-L236 —— RequestStateStats 逐字段
@dataclass
class RequestStateStats:
    """Stats that need to be tracked across delta updates."""

    num_generation_tokens: int = 0

    # This is an engine frontend timestamp (wall-clock)
    arrival_time: float = 0.0

    # These are engine core timestamps (monotonic)
    queued_ts: float = 0.0
    scheduled_ts: float = 0.0
    first_token_ts: float = 0.0
    last_token_ts: float = 0.0

    # first token latency
    first_token_latency: float = 0.0

    # Track if this request is corrupted (NaNs in logits)
    is_corrupted: bool = False


# SOURCE: vllm/v1/metrics/stats.py:L186-L217 —— SchedulerStats 字段面
# （HOST SEAM：仅承载被引用字段；完整统计族归观测域）
@dataclass
class SchedulerStats:
    """Stats associated with the scheduler."""

    num_running_reqs: int = 0
    num_waiting_reqs: int = 0  # length of the "waiting" request queue
    num_skipped_waiting_reqs: int = 0  # length of the "skipped" waiting queue


# SOURCE: vllm/v1/metrics/stats.py:L259 —— PrefillStats 形状位（HOST SEAM）
@dataclass
class PrefillStats:
    pass


# SOURCE: vllm/v1/metrics/stats.py:L349 —— IterationStats 形状位
# （HOST SEAM：真实含 prometheus 记录；本章 output_handler 已删）
@dataclass
class IterationStats:
    pass


# SOURCE: vllm/v1/metrics/stats.py:L507-L529 —— LoRAStats 逐字
class LoRAStats:
    """Tracks waiting and running request IDs for a single LoRA."""

    # SOURCE: vllm/v1/metrics/stats.py:L511-L513
    def __init__(self):
        self.waiting: set[str] = set()
        self.running: set[str] = set()

    # SOURCE: vllm/v1/metrics/stats.py:L515-L525
    def update(self, req_id: str, waiting: bool, running: bool):
        assert not (waiting and running)
        if waiting:
            self.waiting.add(req_id)
        else:
            self.waiting.discard(req_id)

        if running:
            self.running.add(req_id)
        else:
            self.running.discard(req_id)

    # SOURCE: vllm/v1/metrics/stats.py:L527-L528
    @property
    def empty(self) -> bool:
        return not (self.waiting or self.running)


# SOURCE: vllm/v1/metrics/stats.py:L531-L563 —— LoRARequestStates 逐字
class LoRARequestStates:
    """A per-LoRA count of running and waiting requests."""

    # SOURCE: vllm/v1/metrics/stats.py:L535-L538
    def __init__(self, log_stats: bool = False):
        self.log_stats = log_stats
        self.requests: defaultdict[str, LoRAStats] = defaultdict(LoRAStats)

    # SOURCE: vllm/v1/metrics/stats.py:L540-L549
    def _request_update(
        self, req_id: str, lora_name: str | None, waiting: bool, running: bool
    ):
        if not self.log_stats or lora_name is None:
            return

        lora_stats = self.requests[lora_name]
        lora_stats.update(req_id, waiting, running)
        if lora_stats.empty:
            del self.requests[lora_name]

    # SOURCE: vllm/v1/metrics/stats.py:L551-L552
    def request_waiting(self, req_id: str, lora_name: str | None):
        self._request_update(req_id, lora_name, waiting=True, running=False)

    # SOURCE: vllm/v1/metrics/stats.py:L554-L555
    def request_running(self, req_id: str, lora_name: str | None):
        self._request_update(req_id, lora_name, waiting=False, running=True)

    # SOURCE: vllm/v1/metrics/stats.py:L557-L558
    def request_finished(self, req_id: str, lora_name: str | None):
        self._request_update(req_id, lora_name, waiting=False, running=False)
