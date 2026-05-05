# Simplified Scheduler — the decision loop of vLLM
# REFERENCE: vllm/v1/core/sched/scheduler.py
#
# This captures the two-phase running-first scheduling algorithm:
#   1. Schedule RUNNING requests first (keep decoding going)
#   2. Schedule WAITING requests only if there's budget left
#   3. Preempt on OOM; recompute-only preemption
#
# SIMPLIFIED: No KV connectors, no multimodal, no speculative decoding,
#             no chunked prefill, no async scheduling, no structured output.
#             We use a toy block pool to model KV cache allocation.

from collections import deque
from collections.abc import Iterable

from output import SchedulerOutput
from request import Request, RequestStatus
from request_queue import RequestQueue, SchedulingPolicy, create_request_queue


# ─── Toy KV Cache Manager ───────────────────────────────────────────────────
# SIMPLIFIED: Real vLLM uses BlockPool + KVCacheCoordinator (covered in Ch05).
# Here we model just the bookkeeping the scheduler cares about: how many
# blocks are allocated to each request, and whether we can allocate more.


class ToyKVCacheManager:
    """A minimal KV cache block allocator.

    REFERENCE: vllm/v1/core/kv_cache_manager.py — KVCacheManager
    SIMPLIFIED: No prefix caching, no eviction ordering, no hash map.
                Just a pool of free blocks + per-request block tables.

    The scheduler treats the cache manager as an allocator with two ops:
      allocate_slots(req, num_new_tokens) -> Optional[blocks]
      free(req) -> releases all of the request's blocks
    """

    def __init__(self, num_gpu_blocks: int, block_size: int = 16) -> None:
        self.num_gpu_blocks = num_gpu_blocks
        self.block_size = block_size
        self.free_blocks: deque[int] = deque(range(num_gpu_blocks))
        # req_id -> list of block_ids owned by this request
        self.req_to_blocks: dict[str, list[int]] = {}

    def allocate_slots(self, request: Request, num_new_tokens: int) -> list[int] | None:
        """Allocate enough blocks for num_new_tokens more tokens.

        REFERENCE: kv_cache_manager.py:L227 — KVCacheManager.allocate_slots
        SIMPLIFIED: No computed blocks, no lookahead, no external tokens.

        Returns the list of newly allocated block_ids, or None if OOM.
        The scheduler uses a None return as the preemption trigger.
        """
        req_id = request.request_id
        current_blocks = self.req_to_blocks.setdefault(req_id, [])

        # cdiv: how many blocks does the request need in total after this step?
        total_tokens = request.num_computed_tokens + num_new_tokens
        num_blocks_needed = (total_tokens + self.block_size - 1) // self.block_size

        num_new_blocks = num_blocks_needed - len(current_blocks)
        if num_new_blocks <= 0:
            return []  # already have enough blocks

        if num_new_blocks > len(self.free_blocks):
            return None  # OOM — trigger preemption

        # Allocate. O(num_new_blocks), usually 0 or 1.
        new_blocks = [self.free_blocks.popleft() for _ in range(num_new_blocks)]
        current_blocks.extend(new_blocks)
        return new_blocks

    def free(self, request: Request) -> None:
        """Release all blocks owned by this request.

        REFERENCE: kv_cache_manager.py → KVCacheManager.free
        """
        blocks = self.req_to_blocks.pop(request.request_id, [])
        # Return to free pool in reverse order so the most recently used
        # block is reused first (helps with locality).
        self.free_blocks.extendleft(reversed(blocks))

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    @property
    def num_used_blocks(self) -> int:
        return self.num_gpu_blocks - self.num_free_blocks


# ─── The Scheduler ──────────────────────────────────────────────────────────


class Scheduler:
    """Main scheduling engine.

    REFERENCE: vllm/v1/core/sched/scheduler.py:L67 — Scheduler(SchedulerInterface)

    The engine calls schedule() in a busy loop:
        while has_unfinished_requests():
            output = scheduler.schedule()
            model_output = model_runner.execute(output)
            scheduler.update_from_output(output, model_output)

    This class produces SchedulerOutput — {req_id: num_tokens} plus metadata —
    and receives generated tokens back via update_from_output.
    """

    def __init__(
        self,
        max_num_running_reqs: int,
        max_num_scheduled_tokens: int,
        max_model_len: int,
        num_gpu_blocks: int,
        block_size: int = 16,
        policy: SchedulingPolicy = SchedulingPolicy.FCFS,
    ) -> None:
        # REFERENCE: scheduler.py:L106-L112 — scheduling constraints
        self.max_num_running_reqs = max_num_running_reqs
        self.max_num_scheduled_tokens = max_num_scheduled_tokens
        self.max_model_len = max_model_len
        self.policy = policy

        # REFERENCE: scheduler.py:L158-L176 — queues and request tracking
        self.requests: dict[str, Request] = {}
        self.waiting: RequestQueue = create_request_queue(policy)
        self.running: list[Request] = []
        self.finished_req_ids: set[str] = set()

        # REFERENCE: scheduler.py:L228-L240 — kv_cache_manager
        self.kv_cache_manager = ToyKVCacheManager(num_gpu_blocks, block_size)

    # ────────────── Request lifecycle ────────────────────────────────────────

    def add_request(self, request: Request) -> None:
        """Enqueue a new request.

        REFERENCE: scheduler.py:L1728 — Scheduler.add_request
        SIMPLIFIED: No streaming-session handling, no duplicate-id check.
        """
        assert request.request_id not in self.requests, (
            f"Duplicate request: {request.request_id}"
        )
        self.requests[request.request_id] = request
        self.waiting.add_request(request)

    def finish_requests(
        self,
        request_ids: str | Iterable[str],
        finished_status: RequestStatus = RequestStatus.FINISHED_ABORTED,
    ) -> None:
        """Abort requests from outside (e.g., client disconnect).

        REFERENCE: scheduler.py:L1750 — Scheduler.finish_requests
        """
        if isinstance(request_ids, str):
            request_ids = [request_ids]
        request_ids = set(request_ids)

        running_to_remove: list[Request] = []
        waiting_to_remove: list[Request] = []

        for req_id in request_ids:
            req = self.requests.get(req_id)
            if req is None or req.is_finished:
                continue
            if req.status == RequestStatus.RUNNING:
                running_to_remove.append(req)
            else:
                waiting_to_remove.append(req)

        if running_to_remove:
            to_remove_set = set(running_to_remove)
            self.running = [r for r in self.running if r not in to_remove_set]
        if waiting_to_remove:
            self.waiting.remove_requests(waiting_to_remove)

        for req in running_to_remove + waiting_to_remove:
            req.status = finished_status
            self._free_request(req)

    def _free_request(self, request: Request) -> None:
        """Free a finished request's resources.

        REFERENCE: scheduler.py:L1813 — Scheduler._free_request
        """
        self.kv_cache_manager.free(request)
        self.finished_req_ids.add(request.request_id)
        del self.requests[request.request_id]

    # ────────────── The main schedule() loop ─────────────────────────────────

    def schedule(self) -> SchedulerOutput:
        """One scheduling step — decide which requests to run next.

        REFERENCE: scheduler.py:L352-L945 — Scheduler.schedule()

        Two-phase running-first algorithm:
          Phase 1: Schedule RUNNING requests (decoding priority)
                   - Preempt on OOM
          Phase 2: Schedule WAITING requests (only if no preemptions)
                   - Stop early if can't allocate

        Returns a SchedulerOutput describing {req_id: num_tokens} for this step.
        """
        scheduled_new_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []
        num_scheduled_tokens: dict[str, int] = {}

        # REFERENCE: scheduler.py:L371 — token_budget = max_num_scheduled_tokens
        token_budget = self.max_num_scheduled_tokens

        # ───── Phase 1: Schedule RUNNING requests ────────────────────────────
        # REFERENCE: scheduler.py:L388-L556
        # Running-first: decoding requests always get priority. This keeps
        # the decoding going at near-peak throughput and prevents thrashing.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            # How many tokens does this request still need?
            # REFERENCE: scheduler.py:L408-L412
            num_new_tokens = request.num_new_tokens_needed
            num_new_tokens = min(num_new_tokens, token_budget)
            # Don't exceed max_model_len (guards against spec-decode overrun).
            # REFERENCE: scheduler.py:L419-L421
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len - 1 - request.num_computed_tokens,
            )

            if num_new_tokens <= 0:
                # Can happen when the request has already reached max_tokens
                # but hasn't been cleaned up yet. Skip without breaking.
                # REFERENCE: scheduler.py:L446-L462 — `continue` not `break`
                req_index += 1
                continue

            # Try to allocate KV cache blocks. If we can't, preempt.
            # REFERENCE: scheduler.py:L464-L510 — allocate-or-preempt loop
            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request, num_new_tokens
                )
                if new_blocks is not None:
                    break  # allocation succeeded

                # OOM: pick a victim and preempt it.
                # REFERENCE: scheduler.py:L478-L504
                if self.policy == SchedulingPolicy.PRIORITY:
                    # Preempt the lowest-priority running request.
                    # Highest (priority, arrival_time) value = lowest priority.
                    victim = max(
                        self.running, key=lambda r: (r.priority, r.arrival_time)
                    )
                    self.running.remove(victim)
                    # If victim was already scheduled this step, un-schedule it.
                    if victim in scheduled_running_reqs:
                        scheduled_running_reqs.remove(victim)
                        token_budget += num_scheduled_tokens.pop(victim.request_id)
                        req_index -= 1  # compensate for shifted indices
                else:
                    # FCFS: preempt the newest running request (last in list).
                    # REFERENCE: scheduler.py:L504 — self.running.pop()
                    victim = self.running.pop()

                self._preempt_request(victim)
                preempted_reqs.append(victim)

                if victim is request:
                    # We just preempted ourselves — no more victims to try.
                    # REFERENCE: scheduler.py:L508-L510
                    break

            if new_blocks is None:
                # Couldn't schedule this request even after preemption.
                # REFERENCE: scheduler.py:L512-L514
                break

            # Allocation succeeded — record the scheduling decision.
            scheduled_running_reqs.append(request)
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

        # ───── Phase 2: Schedule WAITING requests ────────────────────────────
        # REFERENCE: scheduler.py:L568-L846
        # Only run if no preemptions happened this step. Preemptions mean
        # the system is under memory pressure — adding more requests would
        # immediately re-preempt someone.
        # REFERENCE: scheduler.py:L568 — `if not preempted_reqs`
        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                # REFERENCE: scheduler.py:L572-L573 — max_num_running_reqs check
                if len(self.running) >= self.max_num_running_reqs:
                    break

                request = self.waiting.peek_request()

                # Compute num_new_tokens for this waiting request.
                # REFERENCE: scheduler.py:L673-L693
                # SIMPLIFIED: No prefix caching, no external (KV connector) tokens.
                num_new_tokens = request.num_tokens - request.num_computed_tokens

                # SIMPLIFIED: No chunked prefill support.
                # REFERENCE: scheduler.py:L684-L692 — if chunked_prefill disabled
                # and num_new_tokens > token_budget: break
                if num_new_tokens > token_budget:
                    break  # can't fit this prompt in the budget — defer

                # Try to allocate KV cache blocks for this prefill.
                # REFERENCE: scheduler.py:L744-L754
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request, num_new_tokens
                )
                if new_blocks is None:
                    # Memory is full — can't add any more waiting requests.
                    # REFERENCE: scheduler.py:L756-L763
                    break

                # Promote waiting → running.
                # REFERENCE: scheduler.py:L785-L813
                self.waiting.pop_request()
                self.running.append(request)

                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    # Preempted request resuming — counts as "new" for worker
                    # since its worker-side state was cleared on preemption.
                    # REFERENCE: scheduler.py:L814-L815
                    scheduled_new_reqs.append(request)
                else:
                    raise RuntimeError(f"Unexpected status: {request.status}")

                request.status = RequestStatus.RUNNING
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens

        # ───── Post-processing: invariants + build output ─────────────────────
        # REFERENCE: scheduler.py:L849-L859
        total_tokens = sum(num_scheduled_tokens.values())
        assert total_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs

        output = SchedulerOutput(
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_running_reqs=scheduled_running_reqs,
            preempted_reqs=preempted_reqs,
            num_scheduled_tokens=num_scheduled_tokens,
            finished_req_ids=self.finished_req_ids,
            total_num_scheduled_tokens=total_tokens,
        )

        # REFERENCE: scheduler.py:L974 — _update_after_schedule
        # Advance num_computed_tokens so the request is ready for the next
        # step. If model-runner output adjusts (e.g., spec rejection),
        # update_from_output will correct.
        for req_id, num_sched in num_scheduled_tokens.items():
            self.requests[req_id].num_computed_tokens += num_sched

        # Clear finished set — consumed by this output.
        # REFERENCE: scheduler.py:L995-L998
        self.finished_req_ids = set()

        return output

    # ────────────── Preemption ───────────────────────────────────────────────

    def _preempt_request(self, request: Request) -> None:
        """Kick a running request back to WAITING when KV cache is full.

        REFERENCE: scheduler.py:L952-L972 — Scheduler._preempt_request

        vLLM uses RECOMPUTE-only preemption:
        - Free all KV blocks (the request's work is discarded)
        - Reset num_computed_tokens = 0 (will redo the whole prefill)
        - Set status = PREEMPTED
        - Prepend to waiting queue so it resumes before newer arrivals

        Alternative strategies (not implemented here):
          SWAP: copy KV blocks CPU→GPU on preempt / GPU→CPU on resume.
                Saves recomputation but costs PCIe bandwidth.
          ABORT: kill the request. Only useful if client doesn't care.

        vLLM chose RECOMPUTE because:
          - GPU compute is cheaper than PCIe bandwidth at scale
          - Simpler implementation (no CPU-side storage pool)
          - Works uniformly with prefix caching (cached blocks remain hot)
        """
        assert request.status == RequestStatus.RUNNING, (
            f"Only running requests can be preempted, got {request.status}"
        )
        self.kv_cache_manager.free(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0  # must recompute everything
        request.num_preemptions += 1

        # Prepend so preempted requests resume before newer waiting requests.
        # REFERENCE: scheduler.py:L972 — self.waiting.prepend_request(request)
        # Under PRIORITY policy, "prepend" is equivalent to "add" — the heap
        # re-orders by (priority, arrival_time).
        self.waiting.prepend_request(request)

    # ────────────── update_from_output ───────────────────────────────────────

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        sampled_token_ids: dict[str, list[int]],
    ) -> list[str]:
        """Apply model-runner output back to the scheduler state.

        REFERENCE: scheduler.py:L1290-L1551 — Scheduler.update_from_output
        SIMPLIFIED: No logprobs, no spec decode, no KV connector stats, no
                    pooling output. Just token appending + stop check.

        Args:
            scheduler_output: The output of the last schedule() call.
            sampled_token_ids: {req_id: [new_token_ids]} from the model runner.

        Returns the list of request IDs that finished in this step.
        """
        finished_this_step: list[str] = []
        stopped_running: set[Request] = set()

        for req_id, num_scheduled in scheduler_output.num_scheduled_tokens.items():
            request = self.requests.get(req_id)
            if request is None or request.is_finished:
                # Request was aborted between scheduling and update.
                # REFERENCE: scheduler.py:L1339-L1347
                continue

            new_token_ids = sampled_token_ids.get(req_id, [])
            if not new_token_ids:
                # Still prefilling (prompt not done yet) — no output this step.
                continue

            # Append generated tokens and check for stop condition.
            # REFERENCE: scheduler.py:L1622-L1638 — _update_request_with_output
            stopped = False
            for token_id in new_token_ids:
                request.output_token_ids.append(token_id)
                if self._check_stop(request):
                    stopped = True
                    break

            if stopped:
                stopped_running.add(request)
                finished_this_step.append(req_id)

        # Remove stopped requests from running queue + free their blocks.
        # REFERENCE: scheduler.py:L1477-L1478
        if stopped_running:
            self.running = [r for r in self.running if r not in stopped_running]
            for req in stopped_running:
                self._free_request(req)

        return finished_this_step

    def _check_stop(self, request: Request) -> bool:
        """Check stop conditions and update request status.

        REFERENCE: vllm/v1/core/sched/utils.py:L94 — check_stop()
        SIMPLIFIED: No repetition detection, no custom stop tokens, no EOS.
                    Just length-based stopping.
        """
        if len(request.output_token_ids) >= request.max_tokens:
            request.status = RequestStatus.FINISHED_LENGTH_CAPPED
            return True
        if request.num_tokens >= self.max_model_len:
            request.status = RequestStatus.FINISHED_LENGTH_CAPPED
            return True
        return False

    # ────────────── Query helpers ────────────────────────────────────────────

    def get_num_unfinished_requests(self) -> int:
        """REFERENCE: scheduler.py:L1843 — get_num_unfinished_requests"""
        return len(self.waiting) + len(self.running)

    def has_unfinished_requests(self) -> bool:
        """REFERENCE: interface.py:L165 — has_unfinished_requests"""
        return self.get_num_unfinished_requests() > 0

    def get_request_counts(self) -> tuple[int, int]:
        """REFERENCE: scheduler.py:L1724 — get_request_counts"""
        return len(self.running), len(self.waiting)
