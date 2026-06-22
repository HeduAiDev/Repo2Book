# 纯单元测试（不 import vllm）—— 验证精简版复现真实 vLLM EngineCore 的可观察行为：
#   step() 编排顺序、grammar bitmask 夹在 execute_model 与 result 之间、deferred 采样、
#   执行期 abort 批量落地、run_busy_loop 调度/关停、生命周期 pause/resume/sleep/wake_up。
#
# 协作者（model_executor / scheduler）用记录调用序列的 spy；它们是 EngineCore 真实持有的
# 外部对象，spy 只记录真实方法被调用的事实与顺序，不杜撰算法。
import sys
from concurrent.futures import Future
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "implementation"))

from interfaces import (  # noqa: E402
    EngineCoreOutputs,
    EngineCoreRequestType,
    PauseState,
)
from core import EngineCore, EngineCoreProc, EngineShutdownState  # noqa: E402
from core_client import InprocClient  # noqa: E402


# --------------------------------------------------------------------------- #
# Spies / fakes for the two external collaborators.
# --------------------------------------------------------------------------- #
class FakeSchedulerOutput:
    def __init__(self, total=1, pending_structured=False):
        self.total_num_scheduled_tokens = total
        self.pending_structured_output_tokens = pending_structured


class FakeScheduler:
    def __init__(self):
        self.calls = []
        self._requests = []
        self.pause_state = PauseState.UNPAUSED
        self._next_schedule = FakeSchedulerOutput()
        self._update_result = {0: EngineCoreOutputs()}
        self._unfinished = False
        self.finished = []

    def has_requests(self):
        return bool(self._requests)

    def has_unfinished_requests(self):
        return self._unfinished

    def get_num_unfinished_requests(self):
        return len(self._requests)

    def add_request(self, request):
        self.calls.append("add_request")
        self._requests.append(request)

    def schedule(self):
        self.calls.append("schedule")
        return self._next_schedule

    def get_grammar_bitmask(self, scheduler_output):
        self.calls.append("get_grammar_bitmask")
        return "bitmask"

    def update_from_output(self, scheduler_output, model_output):
        self.calls.append("update_from_output")
        return self._update_result

    def update_draft_token_ids(self, ids):
        self.calls.append("update_draft_token_ids")

    def update_draft_token_ids_in_output(self, ids, so):
        self.calls.append("update_draft_token_ids_in_output")

    def finish_requests(self, ids, status):
        self.calls.append(("finish_requests", ids))
        self._requests = []
        return self.finished

    def set_pause_state(self, state):
        self.calls.append(("set_pause_state", state))
        self.pause_state = state

    def reset_prefix_cache(self, *a, **k):
        self.calls.append("reset_prefix_cache")
        return True

    def shutdown(self):
        self.calls.append("shutdown")


class FakeExecutor:
    def __init__(self, max_concurrent_batches=1, exec_returns_none=True):
        self.calls = []
        self.max_concurrent_batches = max_concurrent_batches
        self.supported_tasks = ("generate",)
        self.is_sleeping = False
        self._exec_returns_none = exec_returns_none

    def execute_model(self, scheduler_output, non_block=False):
        self.calls.append(("execute_model", non_block))
        f: Future = Future()
        f.set_result(None if self._exec_returns_none else "model_output")
        return f

    def sample_tokens(self, grammar_output, non_block=False):
        self.calls.append(("sample_tokens", grammar_output, non_block))
        if non_block:
            f: Future = Future()
            f.set_result("sampled_output")
            return f
        return "sampled_output"

    def take_draft_token_ids(self):
        self.calls.append("take_draft_token_ids")
        return [1]

    def sleep(self, level):
        self.calls.append(("sleep", level))

    def wake_up(self, tags):
        self.calls.append(("wake_up", tags))

    def shutdown(self):
        self.calls.append("shutdown")

    def from_engine_core_request(self, request):
        return request


class FakeConfig:
    use_spec_decode = False
    async_scheduling = False
    shutdown_timeout = 0


def make_core(executor=None, scheduler=None, cls=EngineCore, **kw):
    executor = executor or FakeExecutor()
    scheduler = scheduler or FakeScheduler()
    core = cls(FakeConfig(), executor, scheduler, **kw)
    return core, executor, scheduler


# --------------------------------------------------------------------------- #
# step() orchestration
# --------------------------------------------------------------------------- #
def test_step_returns_early_when_no_requests():
    core, ex, sch = make_core()
    outputs, executed = core.step()
    assert outputs == {} and executed is False
    assert sch.calls == []  # no schedule when nothing to do


def test_step_orchestration_order():
    core, ex, sch = make_core()
    sch._requests = ["r1"]
    outputs, executed = core.step()
    # schedule -> get_grammar_bitmask (between exec and result) -> update_from_output
    assert sch.calls == ["schedule", "get_grammar_bitmask", "update_from_output"]
    # execute_model invoked with non_block=True (async forward)
    assert ("execute_model", True) in ex.calls
    assert executed is True


def test_grammar_bitmask_computed_before_future_result():
    # The key overlap: get_grammar_bitmask must run AFTER execute_model dispatch
    # but its result is only consumed at sample_tokens (after future.result()).
    core, ex, sch = make_core()
    sch._requests = ["r1"]
    core.step()
    exec_idx = ex.calls.index(("execute_model", True))
    sample_idx = ex.calls.index(("sample_tokens", "bitmask", False))
    assert exec_idx < sample_idx
    # bitmask was computed (recorded) before sample consumed it
    assert "get_grammar_bitmask" in sch.calls


def test_step_skips_sample_when_executor_returns_output():
    # If execute_model already returns a non-None output, no main-proc sampling.
    core, ex, sch = make_core(executor=FakeExecutor(exec_returns_none=False))
    sch._requests = ["r1"]
    core.step()
    assert not any(c[0] == "sample_tokens" for c in ex.calls if isinstance(c, tuple))


def test_process_aborts_queue_before_update():
    core, ex, sch = make_core()
    sch._requests = ["r1"]
    core.aborts_queue.put_nowait(["r1"])
    core.step()
    # finish_requests (from abort batch) happens before update_from_output
    finish_idx = next(
        i for i, c in enumerate(sch.calls)
        if isinstance(c, tuple) and c[0] == "finish_requests"
    )
    update_idx = sch.calls.index("update_from_output")
    assert finish_idx < update_idx


def test_aborts_queue_batched_into_single_finish():
    core, ex, sch = make_core()
    sch._requests = ["r1"]
    core.aborts_queue.put_nowait(["a", "b"])
    core.aborts_queue.put_nowait(["c"])
    core._process_aborts_queue()
    finish_calls = [c for c in sch.calls if isinstance(c, tuple) and c[0] == "finish_requests"]
    assert len(finish_calls) == 1
    assert sorted(finish_calls[0][1]) == ["a", "b", "c"]


# --------------------------------------------------------------------------- #
# step_with_batch_queue (PP pipeline) & step_fn binding
# --------------------------------------------------------------------------- #
def test_step_fn_binds_to_plain_step_when_single_batch():
    core, ex, sch = make_core(executor=FakeExecutor(max_concurrent_batches=1))
    assert core.batch_queue is None
    assert core.step_fn == core.step


def test_step_fn_binds_to_batch_queue_when_pp():
    core, ex, sch = make_core(executor=FakeExecutor(max_concurrent_batches=2))
    assert core.batch_queue is not None
    assert core.step_fn == core.step_with_batch_queue


def test_batch_queue_fills_before_taking_result():
    # First call schedules a batch, queue not full, queue tail not done -> returns None, True.
    ex = FakeExecutor(max_concurrent_batches=2, exec_returns_none=False)
    sch = FakeScheduler()
    sch._requests = ["r1"]
    core, _, _ = make_core(executor=ex, scheduler=sch)
    # sample_tokens(non_block) returns a not-done future? our fake future is done.
    # Force the tail future to be 'not done' so the fill-pipeline branch triggers.
    pending: Future = Future()

    def sample_pending(grammar_output, non_block=False):
        return pending

    ex.sample_tokens = sample_pending
    out, executed = core.step_with_batch_queue()
    assert out is None and executed is True
    assert len(core.batch_queue) == 1


def test_batch_queue_blocks_and_updates_when_full_or_done():
    ex = FakeExecutor(max_concurrent_batches=2, exec_returns_none=False)
    sch = FakeScheduler()
    sch._requests = ["r1"]
    core, _, _ = make_core(executor=ex, scheduler=sch)
    out, executed = core.step_with_batch_queue()
    # sample future is already done -> proceeds to pop + update_from_output
    assert "update_from_output" in sch.calls
    assert out == sch._update_result


# --------------------------------------------------------------------------- #
# busy loop (EngineCoreProc)
# --------------------------------------------------------------------------- #
def make_proc(executor=None, scheduler=None):
    return make_core(executor=executor, scheduler=scheduler, cls=EngineCoreProc)


def test_process_engine_step_enqueues_outputs():
    core, ex, sch = make_proc()
    sch._requests = ["r1"]
    core._process_engine_step()
    # output (client_index, outputs) tuple pushed to output_queue
    item = core.output_queue.get_nowait()
    assert item == (0, EngineCoreOutputs())


def test_handle_client_request_add_dispatches():
    core, ex, sch = make_proc()

    class Req:
        request_id = "r1"
        client_index = 0
    core._handle_client_request(EngineCoreRequestType.ADD, (Req(), 0))
    assert "add_request" in sch.calls


def test_handle_client_request_abort_dispatches():
    core, ex, sch = make_proc()
    core._handle_client_request(EngineCoreRequestType.ABORT, ["r1"])
    assert any(isinstance(c, tuple) and c[0] == "finish_requests" for c in sch.calls)


def test_handle_client_request_wakeup_is_noop():
    core, ex, sch = make_proc()
    core._handle_client_request(EngineCoreRequestType.WAKEUP, None)
    assert sch.calls == []


def test_handle_client_request_executor_failed_raises():
    core, ex, sch = make_proc()
    with pytest.raises(RuntimeError, match="Executor failed"):
        core._handle_client_request(EngineCoreRequestType.EXECUTOR_FAILED, b"")


def test_utility_method_invoked_and_enqueued():
    core, ex, sch = make_proc()
    # call is_sleeping via UTILITY
    core._handle_client_request(
        EngineCoreRequestType.UTILITY, (0, 42, "is_sleeping", ())
    )
    client_idx, outputs = core.output_queue.get_nowait()
    assert client_idx == 0
    assert outputs.utility_output.call_id == 42
    assert outputs.utility_output.result.result is False


def test_has_work_reflects_requests_and_queue():
    core, ex, sch = make_proc()
    assert core.has_work() is False
    sch._requests = ["r1"]
    assert core.has_work() is True


def test_is_running_tracks_shutdown_state():
    core, ex, sch = make_proc()
    assert core.is_running() is True
    core.shutdown_state = EngineShutdownState.REQUESTED
    assert core.is_running() is False


def test_process_input_queue_dispatches_and_exits_on_work():
    core, ex, sch = make_proc()

    class Req:
        request_id = "r1"
        client_index = 0
    # ADD request makes scheduler have work -> add_request -> has_work True -> exit loop
    core.input_queue.put_nowait((EngineCoreRequestType.ADD, (Req(), 0)))
    core._process_input_queue()
    assert "add_request" in sch.calls
    assert core.has_work() is True


def test_handle_shutdown_three_states():
    core, ex, sch = make_proc()
    # RUNNING -> keep looping
    assert core._handle_shutdown() is True
    # REQUESTED with no work, timeout 0 -> aborts then SHUTTING_DOWN, no work -> exit
    core.shutdown_state = EngineShutdownState.REQUESTED
    assert core._handle_shutdown() is False
    assert core.shutdown_state == EngineShutdownState.SHUTTING_DOWN


def test_busy_loop_runs_step_then_exits_on_shutdown():
    # Drive one real iteration: one request, then request shutdown.
    core, ex, sch = make_proc()
    sch._requests = ["r1"]

    orig_step = core._process_engine_step

    def step_then_shutdown():
        result = orig_step()
        sch._requests = []  # request consumed
        core.shutdown_state = EngineShutdownState.REQUESTED
        return result

    core._process_engine_step = step_then_shutdown
    with pytest.raises(SystemExit):
        core.run_busy_loop()
    assert "schedule" in sch.calls


# --------------------------------------------------------------------------- #
# lifecycle: pause / resume / sleep / wake_up (in-proc EngineCore)
# --------------------------------------------------------------------------- #
def test_pause_scheduler_abort_sets_paused_new_and_finishes():
    core, ex, sch = make_core()
    ret = core.pause_scheduler(mode="abort")
    assert ret is None
    assert ("set_pause_state", PauseState.PAUSED_NEW) in sch.calls
    assert any(isinstance(c, tuple) and c[0] == "finish_requests" for c in sch.calls)


def test_pause_scheduler_keep_sets_paused_all():
    core, ex, sch = make_core()
    core.pause_scheduler(mode="keep")
    assert ("set_pause_state", PauseState.PAUSED_ALL) in sch.calls


def test_pause_scheduler_wait_rejected_inproc():
    core, ex, sch = make_core()
    with pytest.raises(ValueError, match="'wait' mode"):
        core.pause_scheduler(mode="wait")


def test_resume_scheduler_unpauses():
    core, ex, sch = make_core()
    core.resume_scheduler()
    assert ("set_pause_state", PauseState.UNPAUSED) in sch.calls


def test_sleep_level0_only_pauses_no_executor():
    core, ex, sch = make_core()
    ret = core.sleep(level=0)
    assert ret is None
    assert not any(isinstance(c, tuple) and c[0] == "sleep" for c in ex.calls)


def test_sleep_level1_delegates_to_executor():
    core, ex, sch = make_core()
    core.sleep(level=1)
    assert ("sleep", 1) in ex.calls


def test_wake_up_calls_executor_and_resumes():
    core, ex, sch = make_core()
    core.wake_up()
    assert any(c[0] == "wake_up" for c in ex.calls if isinstance(c, tuple))
    assert ("set_pause_state", PauseState.UNPAUSED) in sch.calls


def test_wake_up_scheduling_tag_skips_executor():
    core, ex, sch = make_core()
    core.wake_up(tags=["scheduling"])
    # only "scheduling" tag -> tags becomes [] -> executor.wake_up skipped
    assert not any(c[0] == "wake_up" for c in ex.calls if isinstance(c, tuple))
    assert ("set_pause_state", PauseState.UNPAUSED) in sch.calls


def test_is_sleeping_reflects_pause_or_executor():
    core, ex, sch = make_core()
    assert core.is_sleeping() is False
    sch.pause_state = PauseState.PAUSED_NEW
    assert core.is_sleeping() is True


# --------------------------------------------------------------------------- #
# EngineCoreProc.pause_scheduler async (idle-callback) version
# --------------------------------------------------------------------------- #
def test_proc_pause_completes_sync_when_no_work():
    core, ex, sch = make_proc()
    # no work -> _pause_complete True -> returns None synchronously
    ret = core.pause_scheduler(mode="keep")
    assert ret is None


def test_proc_pause_returns_future_when_work_pending():
    core, ex, sch = make_proc()
    sch._requests = ["r1"]  # has_work True -> pending
    fut = core.pause_scheduler(mode="keep")
    assert isinstance(fut, Future)
    assert not fut.done()
    # firing idle callbacks (e.g. when engine becomes idle) completes the future
    sch._requests = []
    core._notify_idle_state_callbacks()
    assert fut.done()


# --------------------------------------------------------------------------- #
# InprocClient (no busy loop): get_output drives step_fn directly
# --------------------------------------------------------------------------- #
def test_inproc_client_get_output_steps_engine():
    ex = FakeExecutor()
    sch = FakeScheduler()
    sch._requests = ["r1"]
    client = InprocClient(FakeConfig(), ex, sch)
    out = client.get_output()
    assert "schedule" in sch.calls
    assert isinstance(out, EngineCoreOutputs)


def test_inproc_client_add_request_preprocesses_and_adds():
    ex = FakeExecutor()
    sch = FakeScheduler()
    client = InprocClient(FakeConfig(), ex, sch)

    class Req:
        request_id = "r1"
        current_wave = 0
    client.add_request(Req())
    assert "add_request" in sch.calls


def test_inproc_client_sleep_wait_rejected():
    ex = FakeExecutor()
    sch = FakeScheduler()
    client = InprocClient(FakeConfig(), ex, sch)
    with pytest.raises(ValueError, match="'wait'"):
        client.sleep(mode="wait")
