# SOURCE: vllm/v1/core/sched/interface.py
# SchedulerInterface 契约 + PauseState。schedule() 的 docstring 是连续批处理的
# 一句话契约：调度决策在迭代级做、每步 = 一次 forward、产出 {req_id: num_tokens}。
# 只做减法：删 dossier.subtraction_plan.delete 批准的子系统的抽象方法
# （grammar bitmask/structured、draft token ids、finish_requests/生命周期、
# reset_prefix_cache/encoder cache、make_stats/shutdown、kv/ec connector getter）。
from __future__ import annotations

import enum
from abc import ABC, abstractmethod


# SOURCE: vllm/v1/core/sched/interface.py:L24 PauseState
class PauseState(enum.IntEnum):
    """Scheduler pause state.

    - UNPAUSED: Normal operation
    - PAUSE_NEW: No new requests are scheduled, requests already in
                 running state are scheduled.
    - PAUSE_ALL: No requests are scheduled
    """

    UNPAUSED = 0
    PAUSED_NEW = 1
    PAUSED_ALL = 2


# SOURCE: vllm/v1/core/sched/interface.py:L38 SchedulerInterface
class SchedulerInterface(ABC):
    # SOURCE: vllm/v1/core/sched/interface.py:L53-L67 schedule 契约
    @abstractmethod
    def schedule(self) -> "object":
        # SOURCE: vllm/v1/core/sched/interface.py:L54-L67 docstring
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

    # SUBTRACTED: throttle_prefills 参数（DP prefill balancing，dossier.delete 批准）。

    # SOURCE: vllm/v1/core/sched/interface.py:L135-L142 add_request 契约
    @abstractmethod
    def add_request(self, request: "object") -> None:
        # SOURCE: vllm/v1/core/sched/interface.py:L136-L142 docstring
        """Add a new request to the scheduler's internal queue.

        Args:
            request: The new request being added.
        """
        raise NotImplementedError

    # SUBTRACTED: 其余抽象方法（get_grammar_bitmask/structured L85-L89、
    #   update_from_output L91-L109 与 update_draft_token_ids* L111-L133（⑤ 拍
    #   语义 ch9 已讲、生命周期归 ch11）、finish_requests L144-L166、
    #   get_num_unfinished_requests/has_finished_requests L168-L191、
    #   pause_state property/set_pause_state L198-L206（三态全貌归 ch39）、
    #   reset_prefix_cache/reset_encoder_cache L208-L231、get_request_counts
    #   L233-L236、make_stats/shutdown L242-L253、kv/ec connector getter L255-L262）
    #   —— 分别属 dossier.delete 批准的 structured/observability 子系统或邻章
    #   （ch11/ch12）精简版范围。
