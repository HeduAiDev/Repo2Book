# REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L67-L998
"""Scheduler — the heart of vLLM's continuous batching.

The single insight (verbatim from scheduler.py:L353-L362):

    "There's no 'decoding phase' nor 'prefill phase' in the scheduler.
     Each request just has the num_computed_tokens and num_tokens_with_spec.
     ... At each step, the scheduler tries to assign tokens to the requests
     so that each request's num_computed_tokens can catch up its
     num_tokens_with_spec."

That single uniform treatment is what makes chunked prefill, prefix caching,
and speculative decoding all fall out of the same loop. This file implements
the loop in two phases:

    Phase 1 — RUNNING requests, FCFS, may preempt the lowest-priority running
              request on KV-cache OOM (scheduler.py:L388-L556).
    Phase 2 — WAITING requests, FCFS, admit until token budget or
              max_num_running_reqs runs out (scheduler.py:L568-L846).

`update_after_step` collapses what vLLM splits across `_update_after_schedule`
(L974) and `update_from_output` (L1290): advance num_computed_tokens, then
let the model's "sampled" token (a placeholder zero in our demo) be appended
once the prefill catches up.
"""

from __future__ import annotations

from .kv_cache_manager import SimpleKVCacheManager
from .output import SchedulerOutput
from .request import Request, RequestStatus
from .request_queue import FCFSRequestQueue


# REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L67-L176
class Scheduler:
    """Single-policy (FCFS) scheduler. Same class name as vLLM."""

    def __init__(
        self,
        max_num_scheduled_tokens: int,
        max_num_running_reqs: int,
        num_gpu_blocks: int,
        block_size: int = 16,
        enable_chunked_prefill: bool = True,
        long_prefill_token_threshold: int = 0,
    ) -> None:
        # Field names match vLLM's `self.scheduler_config.<name>` accesses.
        # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L106-L111
        self.max_num_scheduled_tokens = max_num_scheduled_tokens
        # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L106
        self.max_num_running_reqs = max_num_running_reqs
        # REFERENCE: instances/vllm/source/vllm/v1/config/scheduler.py (alongside L84)
        self.enable_chunked_prefill = enable_chunked_prefill
        # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L413-L414
        self.long_prefill_token_threshold = long_prefill_token_threshold

        self.kv_cache_manager = SimpleKVCacheManager(num_gpu_blocks, block_size)

        # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L158-L170
        self.requests: dict[str, Request] = {}
        self.waiting: FCFSRequestQueue = FCFSRequestQueue()
        self.running: list[Request] = []
        # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L176
        self.finished_req_ids: set[str] = set()

    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L1728-L1748
    def add_request(self, request: Request) -> None:
        """Enqueue a fresh request. Real vLLM additionally handles streaming
        input chunks and structured-output grammar init — both omitted."""
        self.requests[request.request_id] = request
        self.waiting.add_request(request)

    # ── The Core: schedule() ──────────────────────────────────────────────
    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L352-L945
    def schedule(self) -> SchedulerOutput:
        """One scheduling step. See module docstring for the two phases.

        Returns a SchedulerOutput describing what got scheduled, what got
        preempted, and what was newly admitted from waiting -> running.
        """
        token_budget = self.max_num_scheduled_tokens
        num_scheduled_tokens: dict[str, int] = {}
        preempted_req_ids: set[str] = set()
        newly_running_req_ids: list[str] = []

        # vLLM houses prefix-cache stats refresh here. No-op for our demo.
        # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L385
        self.kv_cache_manager.new_step_starts()

        # ── Phase 1: RUNNING requests ────────────────────────────────────
        # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L387-L556
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L408-L415
            num_new_tokens = request.num_new_tokens
            if 0 < self.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)

            if num_new_tokens == 0:
                # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L446-L462
                # `continue` (not `break`): we tolerate skipping a request to
                # let lower-priority requests still get scheduled.
                req_index += 1
                continue

            # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L465-L511
            # Try to allocate. On OOM, preempt the lowest-priority running
            # request (the LAST element of self.running under FCFS) and retry.
            new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens)
            while new_blocks is None:
                # FCFS preemption rule: pop the tail.
                # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L504
                preempted_req = self.running.pop()
                if preempted_req is request:
                    # We are the only candidate left — give up on this request.
                    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L508-L510
                    self._preempt_request(preempted_req)
                    preempted_req_ids.add(preempted_req.request_id)
                    new_blocks = None
                    break

                self._preempt_request(preempted_req)
                preempted_req_ids.add(preempted_req.request_id)
                # If the preempted request had already grabbed budget this
                # step, give the budget back.
                # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L485-L502
                returned = num_scheduled_tokens.pop(preempted_req.request_id, 0)
                token_budget += returned
                # Recompute num_new_tokens against the (possibly larger) budget.
                num_new_tokens = min(request.num_new_tokens, token_budget)
                if 0 < self.long_prefill_token_threshold < num_new_tokens:
                    num_new_tokens = self.long_prefill_token_threshold
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request, num_new_tokens
                )

            if new_blocks is None:
                # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L512-L514
                # Even after preemption we cannot schedule this request — bail.
                break

            # Record the scheduling decision.
            # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L516-L522
            request.block_ids.extend(new_blocks)
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

        # ── Phase 2: WAITING requests ────────────────────────────────────
        # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L568-L846
        #
        # Skip Phase 2 entirely if any request was preempted in Phase 1 — the
        # rationale (vLLM:L568) is that preempting then immediately admitting
        # someone new just wastes the freed blocks; let the next step retry.
        if not preempted_req_ids:
            while (
                self.waiting
                and token_budget > 0
                and len(self.running) < self.max_num_running_reqs
            ):
                request = self.waiting.peek_request()

                # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L673-L693
                num_new_tokens = request.num_new_tokens
                if 0 < self.long_prefill_token_threshold < num_new_tokens:
                    num_new_tokens = self.long_prefill_token_threshold

                # Chunked-prefill gate: if disabled and prompt is too long
                # for the remaining budget, stop admitting (don't skip; FCFS
                # means later requests can't jump ahead).
                # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L684-L690
                if (
                    not self.enable_chunked_prefill
                    and num_new_tokens > token_budget
                ):
                    break

                num_new_tokens = min(num_new_tokens, token_budget)
                if num_new_tokens <= 0:
                    # Nothing to do; pop and move on.
                    self.waiting.pop_request()
                    continue

                # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L744-L763
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request, num_new_tokens
                )
                if new_blocks is None:
                    # No KV cache room — stop admitting (no preemption in Phase 2).
                    break

                # Commit: pop from waiting, append to running.
                # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L785-L827
                self.waiting.pop_request()
                request.block_ids = list(new_blocks)
                request.status = RequestStatus.RUNNING
                self.running.append(request)
                num_scheduled_tokens[request.request_id] = num_new_tokens
                newly_running_req_ids.append(request.request_id)
                token_budget -= num_new_tokens

        # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L848-L853
        total = sum(num_scheduled_tokens.values())
        assert total <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs

        return SchedulerOutput(
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total,
            preempted_req_ids=preempted_req_ids,
            newly_running_req_ids=newly_running_req_ids,
            finished_req_ids=set(self.finished_req_ids),
        )

    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L952-L972
    def _preempt_request(self, request: Request) -> None:
        """Free blocks, reset progress, push back to the waiting queue.

        IMPORTANT: vLLM's _preempt_request does NOT remove the request from
        self.running — the caller must have already popped it. We follow the
        same contract; see the schedule() call site at L504 above.
        """
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L961
        self.kv_cache_manager.free(request)
        request.status = RequestStatus.PREEMPTED
        # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L964
        # KV cache is gone, so the work has to be redone from token 0.
        request.num_computed_tokens = 0
        # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L967
        request.num_preemptions += 1
        # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L972
        # Re-queued at the FRONT so it gets first crack next step.
        self.waiting.prepend_request(request)

    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L974-L998
    # +     instances/vllm/source/vllm/v1/core/sched/scheduler.py:L1290-L1551
    def update_after_step(self, output: SchedulerOutput) -> None:
        """Advance state after the model has run on the scheduled tokens.

        Combines `_update_after_schedule` (advance num_computed_tokens) and
        the post-forward bookkeeping that vLLM splits across
        `update_from_output` (sampled token append, stop-check, free).

        SIMPLIFIED: in vLLM the sampler returns the next token; here we
        append a placeholder 0 once a request finishes prefill. Everything
        about spec decode rejection, logprobs, and stop strings is omitted.
        """
        # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L984-L987
        for req_id, n in output.num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += n

            # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L1622-L1638
            # (_update_request_with_output): once prefill is done, every
            # forward step yields one sampled output token. We mimic with a
            # placeholder; tester replaces this with a real model in Ch20.
            if not request.is_prefill:
                if len(request.output_token_ids) < request.max_tokens:
                    request.output_token_ids.append(0)

            # check_stop: length cap reached -> finished.
            # REFERENCE: instances/vllm/source/vllm/v1/core/sched/utils.py — check_stop
            if len(request.output_token_ids) >= request.max_tokens:
                request.status = RequestStatus.FINISHED_LENGTH_CAPPED
                self._free_request(request)

        # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L995-L998
        # vLLM clears finished_req_ids each step so SchedulerOutput only
        # carries this step's freshly-finished IDs. We do the same after the
        # caller has had a chance to read them on the next schedule().
        self.finished_req_ids.clear()

    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L1813-L1834
    def _free_request(self, request: Request) -> None:
        """Release a finished request's resources and drop it from running.

        SIMPLIFIED: the real `_free_request` also coordinates with the KV
        connector for delayed freeing (asynchronous KV transfers), and frees
        the encoder cache. Both omitted.
        """
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        self.finished_req_ids.add(request.request_id)
        if request in self.running:
            self.running.remove(request)
        # vLLM also drops `del self.requests[request.request_id]` here.
        # We keep it for post-mortem inspection in tests.

    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L1724-L1726
    def get_request_counts(self) -> tuple[int, int]:
        """(num_running, num_waiting). Same name and return type as vLLM."""
        return len(self.running), len(self.waiting)


# ─────────────────────────────────────────────────────────────────────────
# Static-vs-continuous batching simulator — pedagogical, not in vLLM.
# Used by Ch04 narrative to quantify the GPU "bubble".
# ─────────────────────────────────────────────────────────────────────────


def static_batching_steps(workload: list[tuple[int, int]]) -> int:
    """Count steps for static batching: prefill all together, then decode together.

    Each tuple is (prompt_len, max_output_len). Static batching waits for the
    longest prompt (prefill bubble) and decodes uniformly until every request
    has produced its full output (decode bubble).
    """
    if not workload:
        return 0
    longest_prompt = max(p for p, _ in workload)
    longest_output = max(o for _, o in workload)
    # Prefill: one step per token of the longest prompt (all GPUs busy
    # only while the longest prompt is processing; shorter prompts idle).
    # Decode: one step per generated token until the longest finishes.
    return longest_prompt + longest_output


def continuous_batching_steps(
    workload: list[tuple[int, int]],
    token_budget: int,
) -> int:
    """Count steps for continuous batching: budget filled with mixed work each step."""
    remaining_prompt = [p for p, _ in workload]
    remaining_output = [o for _, o in workload]
    active = [True] * len(workload)
    steps = 0
    while any(active):
        budget = token_budget
        # Decode step yields 1 token per active-and-past-prefill request.
        for i in range(len(workload)):
            if active[i] and remaining_prompt[i] == 0 and budget > 0:
                remaining_output[i] -= 1
                budget -= 1
                if remaining_output[i] <= 0:
                    active[i] = False
        # Fill leftover with prefill chunks.
        for i in range(len(workload)):
            if active[i] and remaining_prompt[i] > 0 and budget > 0:
                take = min(remaining_prompt[i], budget)
                remaining_prompt[i] -= take
                budget -= take
        steps += 1
    return steps
