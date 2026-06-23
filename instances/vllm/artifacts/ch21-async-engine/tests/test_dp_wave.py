"""Tests for DP wave consensus (sync_dp_state + DPEngineCoreProc state machine).

Asserts the real vLLM behavior:
  * sync_dp_state: [0] SUM>0 ≡ OR (any unfinished), [1] SUM==dp_size ≡ pause
    consensus; non-divisible pause count keeps has_unfinished_global True.
  * _has_global_unfinished_reqs only all-reduces every 32 steps.
  * run_busy_loop: on global idle, rank0 emits wave_complete and bumps wave.
  * START_DP_WAVE wakes a non-excluded engine; ignore_start_dp_wave drops it.
"""

import threading

from implementation._bridge import FakeDPGroup
from implementation.dp_wave import (
    DPEngineCoreProc,
    EngineCoreRequestType,
    ParallelConfig,
)


class _FakeScheduler:
    def __init__(self, unfinished=False, step_executes=True, counts=(0, 0)):
        self._unfinished = unfinished
        self._step_executes = step_executes
        self._counts = counts
        self.dummy_batches = 0
        self.pause_state = "UNPAUSED"

    def has_unfinished_requests(self):
        return self._unfinished

    def process_engine_step(self):
        return self._step_executes

    def execute_dummy_batch(self):
        self.dummy_batches += 1

    def get_request_counts(self):
        return self._counts

    def pause_state_unpaused(self):
        return self.pause_state == "UNPAUSED"


# --- sync_dp_state primitive ---------------------------------------------
def _all_reduce_parallel(group, fn, n):
    results = [None] * n
    threads = []

    def run(r):
        results[r] = fn(r)

    for r in range(n):
        t = threading.Thread(target=run, args=(r,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    return results


def test_sync_dp_state_or_semantics():
    g = FakeDPGroup(3)
    # rank 1 has unfinished work, others don't; none pending pause.
    flags = {0: False, 1: True, 2: False}
    res = _all_reduce_parallel(
        g, lambda r: ParallelConfig.sync_dp_state(g, r, flags[r], False), 3
    )
    for has_unfinished, consensus in res:
        assert has_unfinished is True   # OR → any rank has work
        assert consensus is False       # not all pending pause


def test_sync_dp_state_pause_consensus():
    g = FakeDPGroup(3)
    res = _all_reduce_parallel(
        g, lambda r: ParallelConfig.sync_dp_state(g, r, False, True), 3
    )
    for has_unfinished, consensus in res:
        assert consensus is True            # SUM == dp_size
        assert has_unfinished is False      # nobody unfinished, count divisible


def test_sync_dp_state_partial_pause_keeps_running():
    g = FakeDPGroup(3)
    # only ranks 0,1 pending pause → pause_count=2, not divisible by 3.
    pp = {0: True, 1: True, 2: False}
    res = _all_reduce_parallel(
        g, lambda r: ParallelConfig.sync_dp_state(g, r, False, pp[r]), 3
    )
    for has_unfinished, consensus in res:
        assert consensus is False
        # pause_count % dp_size != 0 → has_unfinished_global stays True
        assert has_unfinished is True


def test_has_unfinished_dp_max_or():
    g = FakeDPGroup(2)
    flags = {0: True, 1: False}
    res = _all_reduce_parallel(
        g, lambda r: ParallelConfig.has_unfinished_dp(g, r, flags[r]), 2
    )
    assert all(res)  # MAX over ints == OR


# --- 32-step all-reduce throttle -----------------------------------------
def test_all_reduce_only_every_32_steps():
    g = FakeDPGroup(1)
    sched = _FakeScheduler(unfinished=True)
    proc = DPEngineCoreProc(0, g, sched)
    # First 31 calls short-circuit to True without touching the group.
    for _ in range(31):
        assert proc._has_global_unfinished_reqs(True) is True
    assert proc.step_counter == 31
    # 32nd call performs the all-reduce.
    assert proc._has_global_unfinished_reqs(True) is True
    assert proc.step_counter == 32


# --- run_busy_loop wave transition ---------------------------------------
def test_run_busy_loop_pauses_and_bumps_wave_on_global_idle():
    g = FakeDPGroup(1)
    # nothing executed, no local work, but engines_running so we go through
    # the all-reduce path; force step_counter to 31 so the next loop syncs.
    sched = _FakeScheduler(unfinished=False, step_executes=False)
    proc = DPEngineCoreProc(0, g, sched, has_coordinator=True)
    proc.engines_running = True
    proc.step_counter = 31
    proc.run_busy_loop()
    # Global idle → engines paused, rank0 reports wave_complete=0, wave bumped.
    assert proc.engines_running is False
    assert proc.current_wave == 1
    assert proc.step_counter == 0
    waves = [o.wave_complete for _, o in proc.output_queue if o.wave_complete is not None]
    assert waves == [0]


def test_run_busy_loop_runs_dummy_batch_when_running_but_no_ready():
    g = FakeDPGroup(1)
    sched = _FakeScheduler(unfinished=True, step_executes=False)
    proc = DPEngineCoreProc(0, g, sched)
    proc.engines_running = True
    proc.step_counter = 0  # won't reach 32 this loop → stays running
    proc.run_busy_loop()
    assert sched.dummy_batches == 1
    assert proc.engines_running is True


def test_idle_engines_skip_dummy_batch():
    g = FakeDPGroup(1)
    sched = _FakeScheduler(unfinished=False, step_executes=False)
    proc = DPEngineCoreProc(0, g, sched)
    proc.engines_running = False
    proc.run_busy_loop()
    assert sched.dummy_batches == 0


# --- two-phase pause + START_DP_WAVE -------------------------------------
def test_pause_consensus_sets_ignore_start_dp_wave():
    g = FakeDPGroup(1)
    sched = _FakeScheduler()
    proc = DPEngineCoreProc(0, g, sched)
    proc.pending_pause = True
    proc.step_counter = 31  # next call hits the all-reduce
    proc._has_global_unfinished_reqs(False)
    assert proc.ignore_start_dp_wave is True
    assert proc.pending_pause is False


def test_start_dp_wave_wakes_non_excluded_engine():
    g = FakeDPGroup(1)
    proc = DPEngineCoreProc(0, g, _FakeScheduler())
    proc.engine_index = 0
    proc.engines_running = False
    proc.current_wave = 2
    proc._handle_client_request(
        EngineCoreRequestType.START_DP_WAVE, (3, 1)  # new_wave=3, exclude=1
    )
    assert proc.current_wave == 3
    assert proc.engines_running is True


def test_start_dp_wave_excluded_engine_is_noop():
    g = FakeDPGroup(1)
    proc = DPEngineCoreProc(0, g, _FakeScheduler())
    proc.engine_index = 1
    proc.engines_running = False
    proc.current_wave = 2
    proc._handle_client_request(EngineCoreRequestType.START_DP_WAVE, (3, 1))
    # excluded == self → no wake
    assert proc.current_wave == 2
    assert proc.engines_running is False


def test_ignore_start_dp_wave_drops_wakeup():
    g = FakeDPGroup(1)
    proc = DPEngineCoreProc(0, g, _FakeScheduler())
    proc.ignore_start_dp_wave = True
    proc.engines_running = False
    proc.current_wave = 2
    proc._handle_client_request(EngineCoreRequestType.START_DP_WAVE, (3, 99))
    assert proc.current_wave == 2
    assert proc.engines_running is False


def test_publish_request_counts_only_on_change():
    g = FakeDPGroup(1)
    sched = _FakeScheduler(counts=(2, 1))
    proc = DPEngineCoreProc(0, g, sched)
    proc._maybe_publish_request_counts()
    assert len([o for _, o in proc.output_queue if o.scheduler_stats]) == 1
    # unchanged → no new publish
    proc._maybe_publish_request_counts()
    assert len([o for _, o in proc.output_queue if o.scheduler_stats]) == 1
