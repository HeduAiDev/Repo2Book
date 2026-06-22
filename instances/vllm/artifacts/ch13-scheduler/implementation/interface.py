# SOURCE: vllm/v1/core/sched/interface.py
# SchedulerInterface 契约 + PauseState。明确‘每个调度步=一次 forward’、
# schedule() 产出 {req_id: num_tokens}。
from __future__ import annotations

import enum
from abc import ABC, abstractmethod


# SOURCE: vllm/v1/core/sched/interface.py:L22 PauseState
class PauseState(enum.IntEnum):
    """Scheduler pause state.

    - UNPAUSED: Normal operation
    - PAUSED_NEW: No new requests are scheduled, requests already in
                  running state are scheduled.
    - PAUSED_ALL: No requests are scheduled
    """

    UNPAUSED = 0
    PAUSED_NEW = 1
    PAUSED_ALL = 2


# SOURCE: vllm/v1/core/sched/interface.py:L36 SchedulerInterface
class SchedulerInterface(ABC):
    # SOURCE: vllm/v1/core/sched/interface.py:L52 schedule
    @abstractmethod
    def schedule(self) -> "object":
        # SOURCE: vllm/v1/core/sched/interface.py:L52
        """Schedule the requests to process in this scheduling step.

        The scheduling decision is made at the iteration level. Each scheduling
        step corresponds to a single forward pass of the model. Therefore, this
        method is called repeatedly by a busy loop in the engine.

        Essentially, the scheduler produces a dictionary of {req_id: num_tokens}
        that specifies how many tokens to process for each request in this
        scheduling step. For example, num_tokens can be as large as the number
        of prompt tokens for new requests, or it can be 1 for the requests that
        are auto-regressively generating new tokens one by one. Otherwise, it
        can be somewhere in between in case of chunked prefills, prefix caching,
        speculative decoding, etc.
        """
        raise NotImplementedError

    # SOURCE: vllm/v1/core/sched/interface.py: update_from_output
    @abstractmethod
    def update_from_output(self, scheduler_output, model_runner_output):
        # SOURCE: vllm/v1/core/sched/interface.py:update_from_output
        raise NotImplementedError

    # SUBTRACTED: add_request/finish_requests/get_num_unfinished_requests 等其余抽象
    #   方法（契约完整定义见原 interface.py:L36-L101）—— 本章只需 schedule/
    #   update_from_output 两个核心契约即可讲清连续批处理。
