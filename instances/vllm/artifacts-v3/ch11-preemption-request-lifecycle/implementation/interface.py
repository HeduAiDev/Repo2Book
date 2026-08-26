# SOURCE: vllm/v1/core/sched/interface.py
# SchedulerInterface 契约 + PauseState。本章保留的生命周期面抽象方法：
# update_from_output（⑤ 拍状态推进——m11）与 finish_requests（外部死法——m17）
# 的 docstring 是两段契约原文。schedule 的 throttle_prefills 参数随 DP 节流
# 删除（dossier.delete 第 11 条）。
from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from collections.abc import Iterable


# SOURCE: vllm/v1/core/sched/interface.py:L24 PauseState
class PauseState(enum.IntEnum):
    """Scheduler pause state.

    - UNPAUSED: Normal operation
    - PAUSE_NEW: No new requests are scheduled, requests already in
                 running state are scheduled.
    - PAUSE_ALL: No requests are scheduled
    """

    # SOURCE: vllm/v1/core/sched/interface.py:L33-L35
    UNPAUSED = 0
    PAUSED_NEW = 1
    PAUSED_ALL = 2


# SOURCE: vllm/v1/core/sched/interface.py:L38 SchedulerInterface
class SchedulerInterface(ABC):
    # SOURCE: vllm/v1/core/sched/interface.py:L53 schedule 契约
    @abstractmethod
    def schedule(self) -> "object":
        # SOURCE: vllm/v1/core/sched/interface.py:L54-L71 docstring
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

        Additionally, the scheduler also returns useful data about each request
        or the batch as a whole. The model runner will use this information in
        preparing inputs to the model.
        """
        raise NotImplementedError

        # SUBTRACTED: throttle_prefills 参数与 Args 段（L54/L73-L77——DP
        #   prefill balancing，dossier.delete 第 11 条批准）。

    # SOURCE: vllm/v1/core/sched/interface.py:L91 update_from_output 契约
    @abstractmethod
    def update_from_output(
        self,
        scheduler_output: "object",
        model_runner_output: "object",
    ) -> dict[int, "object"]:
        # SOURCE: vllm/v1/core/sched/interface.py:L97-L108 docstring
        """Update the scheduler state based on the model runner output.

        This method is called after the model runner has processed the scheduled
        requests. The model runner output includes generated token ids, draft
        token ids for next step, etc. The scheduler uses this information to
        update its states, checks the finished requests, and returns the output
        for each request.

        Returns:
            A dict of client index to EngineCoreOutputs object containing the
            outputs for each request originating from that client.
        """
        raise NotImplementedError

    # SUBTRACTED: update_draft_token_ids / update_draft_token_ids_in_output
    #   （L111-L133——spec decode，第 6 条批准，归 ch12/spec 章）。

    # SOURCE: vllm/v1/core/sched/interface.py:L135 add_request 契约
    @abstractmethod
    def add_request(self, request: "object") -> None:
        # SOURCE: vllm/v1/core/sched/interface.py:L136-L142 docstring
        """Add a new request to the scheduler's internal queue.

        Args:
            request: The new request being added.
        """
        raise NotImplementedError

    # SOURCE: vllm/v1/core/sched/interface.py:L144 finish_requests 契约
    @abstractmethod
    def finish_requests(
        self,
        request_ids: str | Iterable[str] | None,
        finished_status: "object",
    ) -> "list":
        # SOURCE: vllm/v1/core/sched/interface.py:L150-L166 docstring
        """Finish the requests in the scheduler's internal queue. If the request
        is not in the queue, this method will do nothing for that request.

        This method is called in two cases:
        1. When the request is aborted by the client.
        2. When the frontend process detects a stop string of the request after
           de-tokenizing its generated tokens.

        Args:
            request_ids: A single or a list of request IDs, or None to finish all.
            finished_status: The finished status of the given requests.

        Returns:
            List of requests that were aborted. Will not include any that were
            already finished.
        """
        raise NotImplementedError

    # SOURCE: vllm/v1/core/sched/interface.py:L168 get_num_unfinished_requests
    @abstractmethod
    def get_num_unfinished_requests(self) -> int:
        # SOURCE: vllm/v1/core/sched/interface.py:L169-L170
        """Number of unfinished requests in the scheduler's internal queue."""
        raise NotImplementedError

    # SOURCE: vllm/v1/core/sched/interface.py:L173 has_unfinished_requests
    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests in the scheduler's
        internal queue."""
        # SOURCE: vllm/v1/core/sched/interface.py:L176
        return self.get_num_unfinished_requests() > 0

    # SOURCE: vllm/v1/core/sched/interface.py:L178 has_finished_requests
    @abstractmethod
    def has_finished_requests(self) -> bool:
        # SOURCE: vllm/v1/core/sched/interface.py:L179-L191
        """Returns True if there are finished requests that need to be cleared.
        NOTE: This is different from `not self.has_unfinished_requests()`.

        The scheduler maintains an internal list of the requests finished in the
        previous step. This list is returned from the next call to schedule(),
        to be sent to the model runner in the next step to clear cached states
        for these finished requests.

        This method checks if this internal list of finished requests is
        non-empty. This information is useful for DP attention.
        """
        raise NotImplementedError

    # SOURCE: vllm/v1/core/sched/interface.py:L193 has_requests
    def has_requests(self) -> bool:
        """Returns True if there are unfinished requests, or finished requests
        not yet returned in SchedulerOutputs."""
        # SOURCE: vllm/v1/core/sched/interface.py:L196
        return self.has_unfinished_requests() or self.has_finished_requests()

    # SOURCE: vllm/v1/core/sched/interface.py:L198 pause_state property
    @property
    @abstractmethod
    def pause_state(self) -> PauseState:
        # SOURCE: vllm/v1/core/sched/interface.py:L199-L202
        """Current pause state of the scheduler."""
        raise NotImplementedError

    # SUBTRACTED: 抽象 __init__（L39-L51——VllmConfig 装配，契约面换裸标量）、
    #   get_grammar_bitmask（L85-L89——structured，第 4 条）、set_pause_state
    #   （L204-L206——暂停机制，第 11 条：守卫 `not preempted_reqs and
    #   UNPAUSED` 的 UNPAUSED 半边保留，但 set 入口删除）、reset_prefix_cache/
    #   reset_encoder_cache（L208-L231——第 9/2 条）、make_stats/shutdown
    #   （L242-L253——第 10 条）、kv/ec connector getter（L255-L262——第 1 条）、
    #   get_kv_cache_usage（L238-L240——可观测性，第 10 条）。
