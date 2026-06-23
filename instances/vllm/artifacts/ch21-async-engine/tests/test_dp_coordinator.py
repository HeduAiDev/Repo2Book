"""Tests for the DP coordinator wave state machine.

Asserts real vLLM behavior:
  * wave_complete from rank0 → current_wave = wave+1, engines_running=False,
    broadcast (None, wave, running) to front.
  * front-end new-request while paused → engines_running=True + START_DP_WAVE.
  * stale wave from front → exclude=None (all engines notified).
  * start_wave from an engine → wave advanced + START_DP_WAVE with exclude.
  * stats update the engine's [waiting, running] counts.
"""

from implementation._bridge import FakeSocket
from implementation.dp_coordinator import DPCoordinatorProc, EngineState
from implementation.dp_wave import (
    EngineCoreOutputs,
    EngineCoreRequestType,
    SchedulerStats,
)


def _coord(n=2):
    c = DPCoordinatorProc(engine_count=n)
    return c, FakeSocket("front"), FakeSocket("output"), FakeSocket("back")


def test_wave_complete_advances_wave_and_pauses():
    c, front, output, back = _coord()
    c.current_wave = 0
    c.engines_running = True
    output.send(EngineCoreOutputs(engine_index=0, wave_complete=0))
    assert c.process_events(front, output, back) is True
    assert c.current_wave == 1
    assert c.engines_running is False
    # broadcast to front-end with new state
    msg = front.recv()
    assert msg == (None, 1, False)


def test_stale_wave_complete_ignored():
    c, front, output, back = _coord()
    c.current_wave = 5
    output.send(EngineCoreOutputs(engine_index=0, wave_complete=2))
    c.process_events(front, output, back)
    # current_wave(5) > wave(2) → no change
    assert c.current_wave == 5


def test_front_request_while_paused_sends_start_wave():
    c, front, output, back = _coord()
    c.current_wave = 3
    c.engines_running = False
    # front-end frame: (engine_to_exclude, wave)
    front.send((1, 3))
    c.process_events(front, output, back)
    assert c.engines_running is True
    typ, payload = back.recv()
    assert typ == EngineCoreRequestType.START_DP_WAVE
    assert payload == (3, 1)  # current_wave, exclude=engine 1


def test_front_stale_wave_broadcasts_to_all_engines():
    c, front, output, back = _coord()
    c.current_wave = 4
    c.engines_running = False
    # request carrying a stale wave (2 < 4) → exclude must be cleared to None
    front.send((1, 2))
    c.process_events(front, output, back)
    typ, payload = back.recv()
    assert payload == (4, None)  # exclude set to None → all engines notified


def test_engine_start_wave_advances_and_excludes_reporter():
    c, front, output, back = _coord()
    c.current_wave = 1
    c.engines_running = False
    output.send(EngineCoreOutputs(engine_index=2, start_wave=3))
    c.process_events(front, output, back)
    assert c.current_wave == 3
    assert c.engines_running is True
    typ, payload = back.recv()
    assert payload == (3, 2)  # exclude the reporting engine


def test_scheduler_stats_update_counts():
    c, front, output, back = _coord(n=3)
    stats = SchedulerStats(num_waiting_reqs=7, num_running_reqs=2)
    output.send(EngineCoreOutputs(engine_index=1, scheduler_stats=stats))
    c.process_events(front, output, back)
    assert c.engines[1].request_counts == [7, 2]


def test_get_engine_counts_copy_is_independent():
    c, *_ = _coord(n=2)
    c.engines[0].request_counts = [3, 1]
    snap = c._get_engine_counts(do_copy=True)
    c.engines[0].request_counts[0] = 99
    assert snap[0] == [3, 1]  # copy unaffected
    live = c._get_engine_counts(do_copy=False)
    assert live[0][0] == 99  # live view reflects mutation


def test_engine_state_initial_counts():
    assert EngineState().request_counts == [0, 0]


def test_subscription_frames_ignored():
    c, front, output, back = _coord()
    front.send(b"\x01")
    # should be ignored, no broadcast
    c.process_events(front, output, back)
    assert not front.pending()
