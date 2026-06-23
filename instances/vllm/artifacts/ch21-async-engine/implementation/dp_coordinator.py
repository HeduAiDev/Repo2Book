"""DP coordinator — subtract-only companion.

Faithful subset of vLLM's DPCoordinator process (vllm/v1/engine/coordinator.py),
which sits between many front-end API servers and many DP engine ranks:

  * aggregates each engine's [waiting, running] load and broadcasts it to the
    front-ends for load-balancing;
  * tracks the global (current_wave, engines_running) state machine;
  * when engines are paused and a new request arrives, broadcasts START_DP_WAVE
    to wake all engines.

`process_input_socket` keeps the exact three-socket poll loop and wave state
transitions. ZMQ sockets are modeled by `_bridge.FakeSocket`; the elastic-EP
(SCALE_ELASTIC_EP) branches and the stats out-of-order detection are subtracted
per the approved delete plan.
"""

from __future__ import annotations

import copy
from typing import Any

from ._bridge import FakeSocket
from .dp_wave import EngineCoreOutputs, EngineCoreRequestType


# SOURCE: vllm/v1/engine/coordinator.py:L146
class EngineState:
    def __init__(self) -> None:  # SOURCE: vllm/v1/engine/coordinator.py:L147
        self.request_counts = [0, 0]  # [waiting, running]


# SOURCE: vllm/v1/engine/coordinator.py:L23
class DPCoordinator:
    """Front handle for the coordinator process (DP>1).

    Subtracts the real multiprocessing.Process spawn + ZMQ-addr handshake; keeps
    the address-holding role so the topology (front/back publish/output) is
    visible. The actual loop lives in DPCoordinatorProc.
    """

    # SOURCE: vllm/v1/engine/coordinator.py:L78
    def __init__(self, dp_size: int, enable_wave_coordination: bool = True) -> None:
        assert dp_size > 1, "Coordinator only used for data parallel"
        # SUBTRACTED: ZMQ address binding + Process spawn + addr-pipe handshake +
        #   weakref finalizer — process/socket lifecycle, not the coordination
        #   logic this chapter reads · coordinator.py:L84-131
        self.proc = DPCoordinatorProc(
            engine_count=dp_size, enable_wave_coordination=enable_wave_coordination
        )


# SOURCE: vllm/v1/engine/coordinator.py:L151
class DPCoordinatorProc:
    # SOURCE: vllm/v1/engine/coordinator.py:L152
    def __init__(
        self,
        engine_count: int,
        min_stats_update_interval_ms: int = 100,
        enable_wave_coordination: bool = True,
    ) -> None:
        # SUBTRACTED: set_process_title + zmq.Context() — process/runtime setup ·
        #   coordinator.py:L158-159
        self.engines = [EngineState() for _ in range(engine_count)]
        self.stats_update_interval_ms = min_stats_update_interval_ms
        self.enable_wave_coordination = enable_wave_coordination

        # wave state (in vLLM these are locals in process_input_socket; held on
        # self here so a test can step the loop one event at a time)
        self.current_wave = 0
        self.engines_running = False

    # SOURCE: vllm/v1/engine/coordinator.py:L194
    def process_input_socket(
        self,
        publish_front: FakeSocket,
        output_back: FakeSocket,
        publish_back: FakeSocket,
    ) -> None:
        """Three-socket poll loop: aggregate stats + wave state machine + broadcast.

        Driven event-by-event here (the test feeds frames + calls
        `process_events`); vLLM runs this under a zmq.Poller `while True`.
        """
        # SUBTRACTED: MsgpackDecoder construction, socket creation, the
        #   subscribe/READY handshake, and the poller-timeout periodic-publish
        #   branch — transport plumbing around the same state machine ·
        #   coordinator.py:L201-285
        # SUBTRACTED: last_stats_step/last_stats_wave/last_step_counts
        #   out-of-order stats detection — robustness for stat ordering, not the
        #   wave main line · coordinator.py:L209-211, L385-406
        while self.process_events(publish_front, output_back, publish_back):
            pass

    def process_events(
        self,
        publish_front: FakeSocket,
        output_back: FakeSocket,
        publish_back: FakeSocket,
    ) -> bool:
        """One iteration of the loop body. Returns False when no frames remain."""
        if not publish_front.pending() and not output_back.pending():
            return False

        wave_state_changed = False

        # SOURCE: vllm/v1/engine/coordinator.py:L306 (front-end XPUB events)
        if publish_front.pending():
            buffer = publish_front.recv()
            if buffer in (b"\x01", b"\x00"):
                # Ignore subscription messages.
                return True
            decoded = buffer
            # SUBTRACTED: SCALE_ELASTIC_EP front-end branch (dynamic engine
            #   add/remove) — orthogonal to the wave main line ·
            #   coordinator.py:L313-346
            if self.enable_wave_coordination:
                # A front-end sent a new request while engines are paused, so we
                # wake the others.
                engine_to_exclude, wave = decoded
                if not self.engines_running:
                    if wave < self.current_wave:
                        # Stale wave number — ensure all engines handle it.
                        engine_to_exclude = None
                    self.engines_running = True
                    wave_state_changed = True
                    self._send_start_wave(
                        publish_back, self.current_wave, engine_to_exclude
                    )

        # SOURCE: vllm/v1/engine/coordinator.py:L368 (engine output PULL events)
        if output_back.pending():
            outputs: EngineCoreOutputs = output_back.recv()
            eng_index = outputs.engine_index
            scheduler_stats = outputs.scheduler_stats
            if scheduler_stats:
                # 1. Updated request load stats.
                stats = self.engines[eng_index].request_counts
                stats[0] = scheduler_stats.num_waiting_reqs
                stats[1] = scheduler_stats.num_running_reqs

            # Wave coordination: handle wave completion and start notifications.
            if self.enable_wave_coordination:
                if (wave := outputs.wave_complete) is not None:
                    # 2. rank 0 reports we've moved into the global paused state.
                    if self.current_wave <= wave:
                        self.current_wave = wave + 1
                        self.engines_running = False
                        wave_state_changed = True
                elif (wave := outputs.start_wave) is not None and (
                    wave > self.current_wave
                    or (wave == self.current_wave and not self.engines_running)
                ):
                    # 3. Engine got a request for a non-current wave; push the
                    # others forward (race condition handling).
                    self.current_wave = wave
                    self.engines_running = True
                    wave_state_changed = True
                    self._send_start_wave(publish_back, wave, eng_index)

        if wave_state_changed:
            message = (None, self.current_wave, self.engines_running)
            publish_front.send(message)

        return True

    @staticmethod  # SOURCE: vllm/v1/engine/coordinator.py:L449 (_send_start_wave)
    def _send_start_wave(
        socket: FakeSocket, wave: int, exclude_engine_index: int | None
    ) -> None:
        """Broadcast START_DP_WAVE to all engines (with exclude de-dup)."""
        wave_encoded = (wave, exclude_engine_index)
        socket.send_multipart((EngineCoreRequestType.START_DP_WAVE, wave_encoded))

    # SOURCE: vllm/v1/engine/coordinator.py:L461
    def _get_engine_counts(self, do_copy: bool = False) -> list[list[int]]:
        """Return list of [waiting, running] count lists for each engine."""
        if do_copy:
            return [copy.copy(e.request_counts) for e in self.engines]
        return [e.request_counts for e in self.engines]
