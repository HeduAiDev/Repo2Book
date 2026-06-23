"""DP wave consensus — subtract-only companion.

Faithful subset of the data-parallel engine's wave state machine:

  * `ParallelConfig.sync_dp_state` (vllm/config/parallel.py): one 2-element SUM
    all-reduce yields both "is any rank unfinished" (OR) and "did all ranks
    reach pause consensus" (SUM == dp_size).
  * `DPEngineCoreProc` (vllm/v1/engine/core.py): the busy loop that steps,
    publishes counts, runs a dummy batch to stay lock-step, all-reduces every
    32 steps to decide whether to keep running, and on global idle has rank 0
    report `wave_complete` and bumps `current_wave`. Plus START_DP_WAVE wakeup
    handling and the two-phase pause (`pending_pause` → `ignore_start_dp_wave`).

The dp_group all-reduce is backed by `_bridge.FakeDPGroup`; `EngineCoreOutputs`
and `SchedulerStats` are tiny faithful carriers (same fields the coordinator
reads). torch.distributed and the ZMQ output queue are the only externals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._bridge import FakeDPGroup, ReduceOp


@dataclass
class EngineCoreOutputs:  # SOURCE: vllm/v1/engine/__init__.py (EngineCoreOutputs)
    engine_index: int = 0
    scheduler_stats: "SchedulerStats | None" = None
    wave_complete: int | None = None
    start_wave: int | None = None


@dataclass
class SchedulerStats:  # SOURCE: vllm/v1/metrics/stats.py (SchedulerStats — DP fields)
    num_waiting_reqs: int = 0
    num_running_reqs: int = 0
    step_counter: int = 0
    current_wave: int = 0


# SOURCE: vllm/v1/engine/__init__.py (EngineCoreRequestType — DP wave subset)
class EngineCoreRequestType:
    ADD = b"\x00"
    START_DP_WAVE = b"\x05"


# SOURCE: vllm/config/parallel.py:L666 (ParallelConfig static DP-sync helpers)
class ParallelConfig:
    @staticmethod  # SOURCE: vllm/config/parallel.py:L655 (has_unfinished_dp)
    def has_unfinished_dp(dp_group: FakeDPGroup, rank: int, has_unfinished: bool) -> bool:
        # single-element MAX all-reduce ≡ OR across ranks
        tensor = [int(has_unfinished)]
        out = dp_group.all_reduce(rank, tensor, op=ReduceOp.MAX)
        return bool(out[0])

    @staticmethod  # SOURCE: vllm/config/parallel.py:L666 (sync_dp_state)
    def sync_dp_state(
        dp_group: FakeDPGroup, rank: int, has_unfinished: bool, pending_pause: bool
    ) -> tuple[bool, bool]:
        """Combined all-reduce for DP state synchronization.

        Single SUM all-reduce on a 2-element tensor:
          [0] = 1 if this rank has unfinished work. SUM > 0 ≡ OR → any work.
          [1] = 1 if this rank has a pending pause. SUM == dp_size ≡ consensus.
        """
        tensor = [int(has_unfinished), int(pending_pause)]
        out = dp_group.all_reduce(rank, tensor, op=ReduceOp.SUM)
        dp_size = dp_group.size()
        pause_count = out[1]
        has_unfinished_global = out[0] > 0 or pause_count % dp_size != 0
        return has_unfinished_global, pause_count == dp_size


# SOURCE: vllm/v1/engine/core.py:L1622 (DPEngineCoreProc — wave state machine)
class DPEngineCoreProc:
    """EngineCore running in a data-parallel context (MoE only).

    Holds the wave state machine; subtracts elastic-EP scaling entirely.
    """

    # SOURCE: vllm/v1/engine/core.py:L1626 / L1671 (__init__ + _init_data_parallel)
    def __init__(
        self,
        dp_rank: int,
        dp_group: FakeDPGroup,
        scheduler: Any,
        has_coordinator: bool = True,
        publish_dp_lb_stats: bool = True,
    ) -> None:
        # Counts forward-passes so we can sync finished state every N steps.
        self.step_counter = 0
        self.current_wave = 0
        self.last_counts = (0, 0)

        # Two-phase pause protocol state.
        self.pending_pause = False
        self.ignore_start_dp_wave = False
        # SUBTRACTED: eep_scaling_state (ElasticEPScalingState) — elastic-EP
        #   scaling is orthogonal to the wave main line · core.py:L1654-1656

        self.dp_rank = dp_rank
        self.engine_index = dp_rank
        self.dp_group = dp_group
        self.scheduler = scheduler
        self.has_coordinator = has_coordinator
        self.publish_dp_lb_stats = publish_dp_lb_stats
        self.engines_running = False

        # Stand-in for the ZMQ output_queue: front-end / coordinator drain it.
        self.output_queue: list[tuple[int, EngineCoreOutputs]] = []

    # SOURCE: vllm/v1/engine/core.py:L1710
    def add_request(self, request: Any, request_wave: int = 0) -> None:
        # SUBTRACTED: super().add_request(...) base enqueue — base-class request
        #   admission, not the wave-wake logic this section is about ·
        #   core.py:L1711
        if self.has_coordinator and request_wave != self.current_wave:
            if request_wave > self.current_wave:
                self.current_wave = request_wave
            elif not self.engines_running and self.scheduler.pause_state_unpaused():
                # Request received for an already-completed wave, notify
                # front-end that we need to start the next one.
                self.engines_running = True
                self.output_queue.append(
                    (-1, EngineCoreOutputs(start_wave=self.current_wave))
                )

    # SOURCE: vllm/v1/engine/core.py:L1757
    def _handle_client_request(self, request_type: bytes, request: Any) -> None:
        if request_type == EngineCoreRequestType.START_DP_WAVE:
            if self.ignore_start_dp_wave:
                return
            new_wave, exclude_eng_index = request
            if exclude_eng_index != self.engine_index and (
                new_wave >= self.current_wave
            ):
                self.current_wave = new_wave
                if not self.engines_running:
                    self.engines_running = True
        # SUBTRACTED: else → super()._handle_client_request (ADD/ABORT/etc base
        #   dispatch) — not the START_DP_WAVE wake path · core.py:L1774-1775

    # SOURCE: vllm/v1/engine/core.py:L1777
    def _maybe_publish_request_counts(self) -> None:
        if not self.publish_dp_lb_stats:
            return
        counts = self.scheduler.get_request_counts()
        if counts != self.last_counts:
            self.last_counts = counts
            stats = SchedulerStats(
                *counts, step_counter=self.step_counter, current_wave=self.current_wave
            )
            self.output_queue.append((-1, EngineCoreOutputs(scheduler_stats=stats)))

    # SOURCE: vllm/v1/engine/core.py:L1790
    def run_busy_loop(self) -> None:
        """Core busy loop of the EngineCore for data parallel case."""
        # SUBTRACTED: outer `while self._handle_shutdown()` + `_process_input_queue`
        #   (SIGINT/SIGTERM + ZMQ input poll) — driven one iteration at a time by
        #   the test so the per-step wave logic is observable · core.py:L1793-1796
        # SUBTRACTED: eep_scaling_state.progress() elastic-EP branch ·
        #   core.py:L1798-1804

        executed = self._process_engine_step()
        self._maybe_publish_request_counts()

        local_unfinished_reqs = self.scheduler.has_unfinished_requests()
        if not executed:
            if not local_unfinished_reqs and not self.engines_running:
                # All engines are idle.
                return

            # We are in a running state and so must execute a dummy pass
            # if the model didn't execute any ready requests.
            self.execute_dummy_batch()

        # 3) All-reduce operation to determine global unfinished reqs.
        self.engines_running = self._has_global_unfinished_reqs(local_unfinished_reqs)

        if not self.engines_running:
            if self.dp_rank == 0 or not self.has_coordinator:
                # In the coordinator case, dp rank 0 sends updates to the
                # coordinator. Otherwise each rank notifies its front-end.
                client_index = -1 if self.has_coordinator else 0
                self.output_queue.append(
                    (client_index, EngineCoreOutputs(wave_complete=self.current_wave))
                )
            # Increment wave count and reset step counter.
            self.current_wave += 1
            self.step_counter = 0

    # SOURCE: vllm/v1/engine/core.py:L1846
    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        # Optimization - only perform finish-sync all-reduce every 32 steps.
        self.step_counter += 1
        if self.step_counter % 32 != 0:
            return True

        has_unfinished, pause_consensus = ParallelConfig.sync_dp_state(
            self.dp_group,
            self.dp_rank,
            has_unfinished=local_unfinished,
            pending_pause=self.pending_pause,
        )

        if pause_consensus:
            self.ignore_start_dp_wave = True
            self.pending_pause = False

        return has_unfinished

    # --- step / dummy-batch hooks the scheduler/model_runner provide ---------
    # In vLLM these dispatch into the scheduler + executor. The companion lets a
    # fake scheduler decide whether a real step executed and count dummy passes.
    def _process_engine_step(self) -> bool:
        # SOURCE: vllm/v1/engine/core.py:L1806 (returns whether a batch executed)
        return self.scheduler.process_engine_step()

    # SOURCE: vllm/v1/engine/core.py (execute_dummy_batch — keep ranks lock-step)
    def execute_dummy_batch(self) -> None:
        self.scheduler.execute_dummy_batch()
