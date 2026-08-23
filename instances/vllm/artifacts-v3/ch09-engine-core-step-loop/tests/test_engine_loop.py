# TDD tests for the v3 ch09 subtract-only companion (pin vLLM v0.27.1 / 6e448d0ea).
#
# These assert the *observable pinned vLLM behavior* this chapter teaches —
# the EngineCore five-beat step() and the EngineCoreProc busy loop — on a
# plain CPU host. Two layers:
#
# * In-process: EngineCore.step()/worker/scheduler contracts driven directly
#   (the five beats, the two-phase execute/sample contract, ExecuteModelState
#   State error, AsyncOutputFuture D2H-only waiting, the grammar-bitmask
#   window, the token account, abort/shutdown paths).
# * End-to-end over REAL ZMQ: one engine subprocess spawned through the real
#   launch_core_engines / CoreEngineProcManager path; tests play the ch05
#   front-end (ROUTER/PULL bind, byte-tag frames, two-layer handshake) and
#   script the model forward through the real UTILITY thin RPC. No forward
#   pass is faked inside the engine: the loop consumes exactly what the test
#   scripts, through the same dispatch vLLM uses.
#
# Mechanism coverage (dossier m1-m13):
#   m1  five-beat ordering in EngineCore.step, incl. non_block=True ②拍 and
#       aborts landing between ④ and ⑤;
#   m2  busy loop skeleton: idle parks in input_queue.get(block=True) (step
#       count frozen while parked), raise SystemExit as the only clean exit;
#   m3  two-phase contract: execute_model returns None + stashes
#       ExecuteModelState, State error on misuse, AsyncOutputFuture waits the
#       D2H event (not the computation);
#   m4  grammar bitmask window: computed after ② and applied to logits
#       strictly before greedy sampling; None fast path; prefill-chunk rows
#       excluded (F6 埋点);
#   m5  preprocess_add_request (Request 构造 + grammar_init) off the busy loop;
#   m6  output IO thread face: engine_index stamping + client_index routing
#       (in-process bucketing + 2-client e2e);
#   m7  abort dual channel: eager aborts_queue batch-merge before ⑤,
#       input_queue copy idempotent, aborted request skipped by
#       update_from_output (real comment case);
#   m8  two-layer startup handshake + EngineCoreReadyResponse post-init
#       config echo;
#   m9  shutdown abort/drain modes, WAKEUP sentinel, reject-add-in-shutdown,
#       ENGINE_CORE_DEAD single-frame death over the wire;
#   m10 InprocClient: same EngineCore.step, no busy loop;
#   m11 step_fn static binding (batch_queue_size=1 → step);
#   m12 update_from_output hot loop: append/judge-stop/free, client bucketing;
#   m13 schedule() iteration-level contract: {req_id: num_tokens} token
#       account, num_computed_tokens catching up num_tokens.
#
# Run:  cd instances/vllm/artifacts-v3/ch09-engine-core-step-loop
#       python -m pytest tests/ -q
#

import importlib
import queue as queue_mod
import sys
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future
from pathlib import Path

import pytest
import zmq

_IMPL_DIR = Path(__file__).resolve().parent.parent / "implementation"
sys.path.insert(0, str(_IMPL_DIR))

engine_loop = importlib.import_module("engine_loop")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mk_config(**over):
    """Build the seam VllmConfig (the full assembly line is ch03's product)."""
    model_over = dict(over.pop("model_config", {}) or {})
    cache_over = dict(over.pop("cache_config", {}) or {})
    par_over = dict(over.pop("parallel_config", {}) or {})
    sched_over = dict(over.pop("scheduler_config", {}) or {})
    return engine_loop.VllmConfig(
        model_config=engine_loop.ModelConfig(**model_over),
        cache_config=engine_loop.CacheConfig(**cache_over),
        parallel_config=engine_loop.ParallelConfig(**par_over),
        scheduler_config=engine_loop.SchedulerConfig(**sched_over),
        instance_id=over.pop("instance_id", f"test-{uuid.uuid4().hex[:8]}"),
        **over,
    )


def core_request(
    request_id="req-1",
    token_ids=(1, 2, 3),
    max_tokens=8,
    structured=False,
    stop_token_ids=None,
    client_index=0,
    **kw,
):
    sp = engine_loop.SamplingParams(max_tokens=max_tokens)
    if stop_token_ids is not None:
        sp.stop_token_ids = list(stop_token_ids)
    if structured:
        # the ch30 boundary: a wire-safe truthy marker (the real guided
        # decoding config fields decode into a StructuredOutputRequest)
        sp.structured_output_request = True
    return engine_loop.EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=list(token_ids),
        mm_features=None,
        sampling_params=sp,
        pooling_params=None,
        arrival_time=1.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        client_index=client_index,
        **kw,
    )


def add(ec, request_id="req-1", token_ids=(1, 2, 3), max_tokens=8, **kw):
    """Drive the real intake path: preprocess (input-thread work) → add."""
    req, wave = ec.preprocess_add_request(
        core_request(request_id, token_ids, max_tokens, **kw)
    )
    ec.add_request(req, wave)
    return req


def logits_row(favorite, vocab=8, value=5.0):
    """One sampling row: 'favorite' wins the greedy argmax."""
    row = [0.0] * vocab
    row[favorite] = value
    return row


def bare_proc(cfg=None, **cfg_over):
    """An EngineCoreProc with the real EngineCore interior but no ZMQ shell.

    EngineCoreProc.__init__ performs the startup handshake and starts the two
    IO threads (that surface is exercised e2e below); for in-process tests we
    build the instance via __new__ + a real EngineCore so the busy-loop
    methods (_process_input_queue / _handle_shutdown / run_busy_loop ...) run
    against genuine state.
    """
    cfg = cfg or mk_config(**cfg_over)
    proc = engine_loop.EngineCoreProc.__new__(engine_loop.EngineCoreProc)
    ec = engine_loop.EngineCore(cfg, engine_loop.UniProcExecutor, False)
    proc.__dict__.update(ec.__dict__)
    proc.input_queue = queue_mod.Queue()
    proc.output_queue = queue_mod.Queue()
    proc.engine_index = 0
    proc.engines_running = False
    proc.shutdown_state = engine_loop.EngineShutdownState.RUNNING
    proc.process_input_queue_block = True
    # a real (idle) thread for _send_engine_dead's join()
    t = threading.Thread(target=lambda: None, daemon=True)
    t.start()
    proc.output_thread = t
    return proc


def drain(queue_):
    out = []
    while not queue_.empty():
        out.append(queue_.get_nowait())
    return out


class _DoneAsync(engine_loop.AsyncModelRunnerOutput):
    def __init__(self):
        self._out = engine_loop.ModelRunnerOutput(
            req_ids=["r"], req_id_to_index={"r": 0}
        )

    def get_output(self):
        return self._out


class Lab:
    """One real engine subprocess + a raw-protocol front-end (ch05 topology).

    The engine is spawned through the real launch_core_engines /
    CoreEngineProcManager path (mp spawn). The front-end binds ROUTER (inputs)
    and PULL (outputs) with the tcp port-0 LAST_ENDPOINT writeback, completes
    the handshake context, collects the DEALER first-message
    EngineCoreReadyResponse, and speaks the byte-tag frame protocol. The MP
    clients themselves are ch05's product and are not part of this chapter's
    companion (dossier delete item 12).
    """

    def __init__(self, n_clients=1, **cfg_over):
        self.cfg = mk_config(**cfg_over)
        self.n_clients = n_clients
        self.ctx = zmq.Context()
        self.encoder = engine_loop.MsgpackEncoder()
        self.decoder = engine_loop.MsgpackDecoder(engine_loop.EngineCoreOutputs)
        self.utility_results: dict[int, Future] = {}
        self.pending_outputs: dict[int, deque] = {i: deque() for i in range(n_clients)}
        self.call_seq = 0

        addresses = engine_loop.get_engine_zmq_addresses(
            self.cfg, num_api_servers=n_clients
        )
        self.input_sockets = []
        self.output_sockets = []
        for i in range(n_clients):
            s = self.ctx.socket(zmq.ROUTER)
            s.bind(addresses.inputs[i])
            addresses.inputs[i] = s.getsockopt(zmq.LAST_ENDPOINT).decode()
            self.input_sockets.append(s)
        for i in range(n_clients):
            s = self.ctx.socket(zmq.PULL)
            s.bind(addresses.outputs[i])
            addresses.outputs[i] = s.getsockopt(zmq.LAST_ENDPOINT).decode()
            self.output_sockets.append(s)

        # __enter__ spawns the engine procs (they queue HELLO on the bound
        # handshake ROUTER and block awaiting the init message); we rebind the
        # data-socket endpoints above, then __exit__ resumes into
        # wait_for_engine_startup to complete HELLO → init → READY.
        self._launcher = engine_loop.launch_core_engines(
            self.cfg, engine_loop.UniProcExecutor, False, addresses
        )
        self.manager, self.addresses, _ = self._launcher.__enter__()
        self._launcher.__exit__(None, None, None)

        # m2/m8: the first message on each data DEALER is the
        # EngineCoreReadyResponse ("required before the front-end ROUTER
        # socket can send input messages back to us").
        self.identities = []
        self.ready_responses = []
        for s in self.input_sockets:
            ident, ready = s.recv_multipart()
            self.identities.append(ident)
            self.ready_responses.append(
                engine_loop.msgpack_ext.decode(
                    ready, type=engine_loop.EngineCoreReadyResponse
                )
            )

    def send(self, client: int, req_type, payload):
        frames = (
            self.identities[client],
            req_type.value,
            *self.encoder.encode(payload),
        )
        self.input_sockets[client].send_multipart(frames, copy=False)

    def add(self, request, client: int = 0):
        self.send(client, engine_loop.EngineCoreRequestType.ADD, request)

    def abort(self, request_ids, client: int = 0):
        self.send(client, engine_loop.EngineCoreRequestType.ABORT, request_ids)

    def _recv_raw(self, client: int, timeout=10.0):
        sock = self.output_sockets[client]
        if not sock.poll(timeout * 1000):
            raise TimeoutError(f"no output on client {client} within {timeout}s")
        frames = sock.recv_multipart(copy=False)
        if len(frames) == 1 and bytes(frames[0]) == (
            engine_loop.EngineCoreProc.ENGINE_CORE_DEAD
        ):
            raise RuntimeError("engine sent ENGINE_CORE_DEAD (fatal engine error)")
        return self.decoder.decode(frames)

    def recv(self, client: int = 0, timeout=10.0):
        """Next EngineCoreOutputs for this client (utility replies skipped)."""
        deadline = time.monotonic() + timeout
        while True:
            if self.pending_outputs[client]:
                return self.pending_outputs[client].popleft()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"no step output on client {client}")
            outputs = self._recv_raw(client, timeout=remaining)
            if outputs.utility_output is not None:
                self._resolve_utility(outputs.utility_output)
            else:
                return outputs

    def _resolve_utility(self, uo):
        fut = self.utility_results.pop(uo.call_id, None)
        if fut is None:
            return
        if uo.failure_message is not None:
            fut.set_exception(Exception(uo.failure_message))
        else:
            fut.set_result(uo.result.result)

    def call_utility(self, method: str, *args, client: int = 0, timeout=30.0):
        self.call_seq += 1
        call_id = self.call_seq
        fut: Future = Future()
        self.utility_results[call_id] = fut
        self.send(
            client,
            engine_loop.EngineCoreRequestType.UTILITY,
            (client, call_id, method, list(args)),
        )
        deadline = time.monotonic() + timeout
        while not fut.done():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"utility {method!r} did not answer in {timeout}s")
            outputs = self._recv_raw(client, timeout=remaining)
            if outputs.utility_output is not None:
                self._resolve_utility(outputs.utility_output)
            else:
                self.pending_outputs[client].append(outputs)
        return fut.result()

    def close(self):
        try:
            self.manager.shutdown(timeout=30)
        finally:
            for s in self.input_sockets + self.output_sockets:
                s.close(linger=300)
            self.ctx.term()


@pytest.fixture(scope="module")
def lab():
    lab = Lab()
    yield lab
    lab.close()


@pytest.fixture()
def two_client_lab():
    lab = Lab(n_clients=2)
    yield lab
    lab.close()


# ---------------------------------------------------------------------------
# m1: the five beats of EngineCore.step
# ---------------------------------------------------------------------------


class TestFiveBeats:
    def test_beat_order_and_non_block(self):
        ec = engine_loop.EngineCore(mk_config(), engine_loop.UniProcExecutor, False)
        add(ec, "req-1", token_ids=(1, 2, 3), max_tokens=8)
        ec.enqueue_forward_logits([{"req-1": logits_row(7)}])  # prefill step: 1 row
        # ②拍 must reach the executor with non_block=True (spy on the seam).
        seen_kwargs = {}
        orig_execute = ec.model_executor.execute_model

        def spy_execute(scheduler_output, non_block=False):
            seen_kwargs["non_block"] = non_block
            return orig_execute(scheduler_output, non_block=non_block)

        ec.model_executor.execute_model = spy_execute

        outputs, model_executed = ec.step()

        assert seen_kwargs["non_block"] is True
        sched = ec.scheduler.trace
        run = ec.model_executor.driver_worker.trace
        # ① schedule → ② execute_model(non_block=True) → ③ get_grammar_bitmask
        # → ④ future.result → (None → sample_tokens) → ⑤ update_from_output
        assert [e[1] for e in sched] == [
            "has_requests",
            "schedule",
            "get_grammar_bitmask",
            "update_from_output",
        ]
        assert [e[1] for e in run] == [
            "execute_model",
            "sample_tokens",
            "greedy_sample",
        ]
        # ③ computed while ②'s forward is already launched (timestamps):
        assert run[0][0] < sched[2][0] < run[1][0]
        # the sampled token comes from the scripted logits row (greedy)
        out = outputs[0].outputs[0]
        assert out.request_id == "req-1"
        assert out.new_token_ids == [7]
        assert out.finish_reason is None
        assert model_executed is True

    def test_idle_guard_returns_without_touching_executor(self):
        ec = engine_loop.EngineCore(mk_config(), engine_loop.UniProcExecutor, False)
        outputs, model_executed = ec.step()
        assert outputs == {}
        assert model_executed is False
        assert [e[1] for e in ec.scheduler.trace] == ["has_requests"]
        assert ec.model_executor.driver_worker.trace == []

    def test_zero_token_step_reports_not_executed_and_skips_sampling(self):
        # A finished-but-unflushed request keeps the engine stepping: schedule
        # flushes it with a 0-token batch, execute returns EMPTY (no forward),
        # `if model_output is None` is False → sample_tokens never runs.
        ec = engine_loop.EngineCore(mk_config(), engine_loop.UniProcExecutor, False)
        add(ec, "req-1", token_ids=(1,), max_tokens=1)
        ec.enqueue_forward_logits([{"req-1": logits_row(5)}])
        outputs, executed = ec.step()
        assert executed is True
        assert outputs[0].outputs[0].finish_reason is engine_loop.FinishReason.LENGTH
        ec.model_executor.driver_worker.trace.clear()
        outputs, executed = ec.step()  # flush step
        assert executed is False
        assert outputs == {}
        run = ec.model_executor.driver_worker.trace
        assert [e[1] for e in run] == ["execute_model"]  # no sample

    def test_throttle_prefills_flag_passed_through(self):
        # _should_throttle_prefills() is the DP prefill-balancing hook; the
        # base EngineCore never throttles (m11 domain boundary).
        ec = engine_loop.EngineCore(mk_config(), engine_loop.UniProcExecutor, False)
        assert ec._should_throttle_prefills() is False
        add(ec, "req-1", token_ids=(1,), max_tokens=2)
        ec.enqueue_forward_logits([{"req-1": logits_row(3)}, {"req-1": logits_row(3)}])
        ec.step()
        assert ec.scheduler.trace[-3][1] == "schedule"


# ---------------------------------------------------------------------------
# m3: the execute_model/sample_tokens two-phase contract (worker side)
# ---------------------------------------------------------------------------


def sched_output(req_ids=("req-1",), tokens=(3,)):
    new_reqs = [
        engine_loop.NewRequestData(
            req_id=rid,
            prompt_token_ids=[1] * t,
            mm_features=[],
            sampling_params=None,
            pooling_params=None,
            block_ids=(),
            num_computed_tokens=0,
            lora_request=None,
        )
        for rid, t in zip(req_ids, tokens)
    ]
    return engine_loop.SchedulerOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=engine_loop.CachedRequestData(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        ),
        num_scheduled_tokens=dict(zip(req_ids, tokens)),
        total_num_scheduled_tokens=sum(tokens),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


class TestTwoPhaseContract:
    def make_runner(self):
        return engine_loop.GPUModelRunner(mk_config())

    def test_state_error_raised_verbatim(self):
        runner = self.make_runner()
        runner.enqueue_logits([{"req-1": logits_row(1)}])
        assert runner.execute_model(sched_output()) is None
        with pytest.raises(
            RuntimeError,
            match="State error: sample_tokens\\(\\) must be called "
            "after execute_model\\(\\) returns None.",
        ):
            runner.execute_model(sched_output())

    def test_state_error_surfaces_through_executor_in_line(self):
        # UniProcExecutor.execute_model(non_block=True) surfaces a done
        # failure in-line (uniproc_executor.py:L117-L120).
        ex = engine_loop.UniProcExecutor(mk_config())
        runner = ex.driver_worker
        runner.enqueue_logits([{"req-1": logits_row(1)}])
        assert ex.execute_model(sched_output(), non_block=True) is not None
        with pytest.raises(RuntimeError, match="State error"):
            ex.execute_model(sched_output(), non_block=True)

    def test_stash_then_unpack_clears_state(self):
        runner = self.make_runner()
        runner.enqueue_logits([{"req-1": logits_row(1)}])
        runner.execute_model(sched_output())
        state = runner.execute_model_state
        assert isinstance(state, engine_loop.ExecuteModelState)
        assert state.scheduler_output.num_scheduled_tokens == {"req-1": 3}
        assert state.logits.shape == (1, 8)
        out = runner.sample_tokens(None)
        assert runner.execute_model_state is None  # cleared (consumed)
        assert out.req_ids == ["req-1"]
        assert out.sampled_token_ids == [[1]]

    def test_empty_batch_returns_empty_runner_output(self):
        runner = self.make_runner()
        out = runner.execute_model(sched_output(req_ids=(), tokens=()))
        assert out is engine_loop.EMPTY_MODEL_RUNNER_OUTPUT
        assert runner.execute_model_state is None

    def test_execute_model_state_is_the_ten_field_namedtuple(self):
        assert engine_loop.ExecuteModelState._fields == (
            "scheduler_output",
            "logits",
            "spec_decode_metadata",
            "spec_decode_common_attn_metadata",
            "hidden_states",
            "sample_hidden_states",
            "aux_hidden_states",
            "ec_connector_output",
            "cudagraph_stats",
            "slot_mappings",
        )


class TestAsyncOutputFuture:
    def test_result_waits_d2h_not_the_computation(self):
        # The forward "computation" is already done when the future is built;
        # result() only waits for async_output.get_output() — the D2H copy
        # event — to unblock.
        d2h = threading.Event()

        class SeamAsyncOutput(engine_loop.AsyncModelRunnerOutput):
            copy_done = False

            def get_output(self):
                d2h.wait(10)
                self.copy_done = True
                return engine_loop.ModelRunnerOutput(
                    req_ids=["r"], req_id_to_index={"r": 0}
                )

        seam = SeamAsyncOutput()
        fut = engine_loop.AsyncOutputFuture(seam, single_value=True)
        holder = {}
        t = threading.Thread(target=lambda: holder.setdefault("v", fut.result()))
        t.start()
        time.sleep(0.25)
        assert "v" not in holder  # still waiting on the D2H event
        assert not seam.copy_done
        d2h.set()
        t.join(10)
        assert holder["v"].req_ids == ["r"]
        assert seam.copy_done
        # Second result() is instant: the future is done now.
        assert fut.result().req_ids == ["r"]

    def test_timeout_not_implemented(self):
        fut = engine_loop.AsyncOutputFuture(_DoneAsync(), single_value=True)
        with pytest.raises(RuntimeError, match="timeout not implemented"):
            fut.result(timeout=1.0)

    def test_single_value_false_wraps_in_list_and_exceptions_propagate(self):
        class Exploding(engine_loop.AsyncModelRunnerOutput):
            def get_output(self):
                raise ValueError("d2h copy failed")

        fut = engine_loop.AsyncOutputFuture(_DoneAsync(), single_value=False)
        assert isinstance(fut.result(), list)
        fut2 = engine_loop.AsyncOutputFuture(Exploding(), single_value=True)
        with pytest.raises(ValueError, match="d2h copy failed"):
            fut2.result()

    def test_executor_sample_tokens_non_block_waits_d2h_when_async(self):
        # With async scheduling on, sample_tokens wraps the output in the
        # async container; collective_rpc turns it into an AsyncOutputFuture
        # whose result() waits only the D2H event (uniproc_executor.py:L97-106).
        cfg = mk_config(scheduler_config={"async_scheduling": True})
        ex = engine_loop.UniProcExecutor(cfg)
        runner = ex.driver_worker
        runner.enqueue_logits([{"req-1": logits_row(4)}])
        runner.execute_model(sched_output())
        fut = ex.sample_tokens(None, non_block=True)
        assert isinstance(fut, engine_loop.AsyncOutputFuture)
        assert not fut.done()
        runner.release_async_copies()  # the D2H copy completes
        out = fut.result()
        assert out.sampled_token_ids == [[4]]


# ---------------------------------------------------------------------------
# m4: the grammar bitmask window (F6)
# ---------------------------------------------------------------------------


class TestGrammarBitmaskWindow:
    def test_none_fast_path_without_structured_requests(self):
        ec = engine_loop.EngineCore(mk_config(), engine_loop.UniProcExecutor, False)
        add(ec, "req-1", token_ids=(1, 2), max_tokens=4)
        ec.enqueue_forward_logits([{"req-1": logits_row(3)}])
        ec.step()
        # ③ ran (always does) but the grammar manager was never consulted.
        assert ec.structured_output_manager.trace == []
        assert [e[1] for e in ec.model_executor.driver_worker.trace][-1] == (
            "greedy_sample"
        )

    def test_prefill_chunk_rows_excluded(self):
        # grammar_bitmask only covers requests that are NOT still-prefilling
        # (scheduler.py:L1654-L1659). A non-final prefill chunk → no manager
        # call, and no partial-prefill output is emitted either.
        ec = engine_loop.EngineCore(
            mk_config(scheduler_config={"max_num_batched_tokens": 2}),
            engine_loop.UniProcExecutor,
            False,
        )
        add(ec, "req-1", token_ids=(1, 2, 3), max_tokens=4, structured=True)
        ec.enqueue_forward_logits([{"req-1": logits_row(3)}])
        outputs, executed = ec.step()
        assert executed is True
        assert outputs == {}  # invariant: no partial prefill outputs
        # budget=2 < prompt=3 → first step is a non-final chunk.
        assert ec.scheduler.requests["req-1"].is_prefill_chunk is True
        assert "grammar_bitmask" not in ec.structured_output_manager.trace

    def test_bitmask_masks_argmax_before_greedy_sample(self):
        # The real apply_grammar_bitmask + real greedy argmax: token 5 has the
        # highest logit but the bitmask only allows {1, 4} → 4 is sampled.
        ec = engine_loop.EngineCore(mk_config(), engine_loop.UniProcExecutor, False)
        add(ec, "req-1", token_ids=(1, 2), max_tokens=4, structured=True)
        ec.enqueue_forward_logits([{"req-1": [0.0, 3.0, 0.0, 0.0, 7.0, 9.0, 0.0, 0.0]}])
        ec.enqueue_grammar_bitmask([[0b0000000000010010]])  # bits 1 and 4 set
        outputs, _ = ec.step()
        assert outputs[0].outputs[0].new_token_ids == [4]

    def test_bitmask_window_between_execute_and_sample(self):
        ec = engine_loop.EngineCore(mk_config(), engine_loop.UniProcExecutor, False)
        add(ec, "req-1", token_ids=(1, 2), max_tokens=4, structured=True)
        ec.enqueue_forward_logits([{"req-1": logits_row(5)}])
        ec.enqueue_grammar_bitmask([[0b10]])
        ec.step()
        run = ec.model_executor.driver_worker.trace
        mgr = ec.structured_output_manager.trace
        assert [e[1] for e in run] == [
            "execute_model",
            "sample_tokens",
            "apply_bitmask",
            "greedy_sample",
        ]
        assert [t for t in mgr if t == "grammar_bitmask"] == ["grammar_bitmask"]
        assert [t for t in mgr if t.startswith("grammar_init")] == ["grammar_init:req-1"]
        # The mask is computed on the CPU (③) *after* the forward was
        # launched (②) and applied to logits *before* sampling (④).
        sched = ec.scheduler.trace
        assert run[0][0] < sched[2][0] < run[2][0] < run[3][0]


# ---------------------------------------------------------------------------
# m13 + seam token account: schedule() contract
# ---------------------------------------------------------------------------


class TestScheduleTokenAccount:
    def test_prefill_then_decode_account(self):
        ec = engine_loop.EngineCore(mk_config(), engine_loop.UniProcExecutor, False)
        add(ec, "req-1", token_ids=(1, 2, 3), max_tokens=5)
        ec.enqueue_forward_logits([{"req-1": logits_row(6)}] * 4)
        ec.step()  # whole prompt as one (final) prefill chunk
        assert ec.scheduler.last_output.num_scheduled_tokens == {"req-1": 3}
        assert ec.scheduler.last_output.total_num_scheduled_tokens == 3
        req = ec.scheduler.requests["req-1"]
        assert req.num_computed_tokens == 3  # num_computed catches up num_tokens
        ec.step()
        assert ec.scheduler.last_output.num_scheduled_tokens == {"req-1": 1}
        assert req.num_computed_tokens == 4

    def test_budget_caps_prefill_into_chunks(self):
        ec = engine_loop.EngineCore(
            mk_config(scheduler_config={"max_num_batched_tokens": 2}),
            engine_loop.UniProcExecutor,
            False,
        )
        add(ec, "req-1", token_ids=(1, 2, 3), max_tokens=5)
        ec.enqueue_forward_logits([{"req-1": logits_row(6)}] * 4)
        ec.step()
        assert ec.scheduler.last_output.num_scheduled_tokens == {"req-1": 2}
        req = ec.scheduler.requests["req-1"]
        assert req.is_prefill_chunk is True  # 2 of 3 prompt tokens computed
        ec.step()
        assert ec.scheduler.last_output.num_scheduled_tokens == {"req-1": 1}
        assert req.is_prefill_chunk is False  # final chunk completes the prompt

    def test_schedule_produces_token_dict_not_request_count(self):
        # Part III hook: the scheduler speaks a *token* account. Two requests
        # in different phases → {req_id: num_tokens} mix.
        ec = engine_loop.EngineCore(mk_config(), engine_loop.UniProcExecutor, False)
        add(ec, "a", token_ids=(1, 2), max_tokens=4)
        both = {"a": logits_row(6), "b": logits_row(6)}
        ec.enqueue_forward_logits([{"a": logits_row(6)}] + [both] * 5)
        ec.step()  # a prefills (2 tokens)
        add(ec, "b", token_ids=(7, 8, 9, 10), max_tokens=4)
        ec.step()  # b prefills (4), a decodes (1)
        acct = ec.scheduler.last_output.num_scheduled_tokens
        assert acct == {"a": 1, "b": 4}


# ---------------------------------------------------------------------------
# m12: update_from_output hot loop
# ---------------------------------------------------------------------------


class TestUpdateFromOutput:
    def test_appends_token_and_finishes_on_max_tokens(self):
        ec = engine_loop.EngineCore(mk_config(), engine_loop.UniProcExecutor, False)
        add(ec, "req-1", token_ids=(1, 2), max_tokens=2)
        ec.enqueue_forward_logits([{"req-1": logits_row(7, vocab=16)}, {"req-1": logits_row(8, vocab=16)}])
        outs1, _ = ec.step()
        assert outs1[0].outputs[0].new_token_ids == [7]
        assert outs1[0].outputs[0].finish_reason is None
        outs2, _ = ec.step()
        out = outs2[0].outputs[0]
        assert out.new_token_ids == [8]
        assert out.finish_reason is engine_loop.FinishReason.LENGTH
        assert "req-1" not in ec.scheduler.requests  # freed

    def test_stop_token_finishes_stopped_with_reason(self):
        ec = engine_loop.EngineCore(mk_config(), engine_loop.UniProcExecutor, False)
        req, _ = ec.preprocess_add_request(
            core_request("req-1", token_ids=(1, 2), max_tokens=8, stop_token_ids=[9])
        )
        ec.add_request(req, 0)
        ec.enqueue_forward_logits([{"req-1": logits_row(9, vocab=16)}])
        outs, _ = ec.step()
        out = outs[0].outputs[0]
        assert out.new_token_ids == [9]
        assert out.finish_reason is engine_loop.FinishReason.STOP
        assert out.stop_reason == 9

    def test_aborted_during_execution_is_skipped(self):
        # The real comment case (scheduler.py:L1747-1755): aborted while the
        # model was executing it → the request is gone by update time → row
        # skipped, no output emitted for it.
        ec = engine_loop.EngineCore(mk_config(), engine_loop.UniProcExecutor, False)
        add(ec, "req-1", token_ids=(1, 2), max_tokens=8)
        ec.enqueue_forward_logits([{"req-1": logits_row(7)}])
        # Simulate the eager aborts_queue landing between ④ and ⑤:
        ec.aborts_queue.put_nowait(["req-1"])
        outs, _ = ec.step()
        assert all(
            o.request_id != "req-1" for eco in outs.values() for o in eco.outputs
        )
        assert ec.scheduler.finish_calls == [["req-1"]]  # one batch abort
        assert "req-1" not in ec.scheduler.requests

    def test_buckets_by_client_index(self):
        ec = engine_loop.EngineCore(mk_config(), engine_loop.UniProcExecutor, False)
        add(ec, "c0-req", token_ids=(1,), max_tokens=4, client_index=0)
        add(ec, "c1-req", token_ids=(2,), max_tokens=4, client_index=1)
        ec.enqueue_forward_logits([{"c0-req": logits_row(5), "c1-req": logits_row(6)}] * 4)
        outs, _ = ec.step()
        assert set(outs) == {0, 1}
        assert outs[0].outputs[0].request_id == "c0-req"
        assert outs[1].outputs[0].request_id == "c1-req"


# ---------------------------------------------------------------------------
# m7: abort dual channel
# ---------------------------------------------------------------------------


class TestAbortDualChannel:
    def test_aborts_queue_batch_merge(self):
        ec = engine_loop.EngineCore(mk_config(), engine_loop.UniProcExecutor, False)
        add(ec, "a", token_ids=(1,), max_tokens=4)
        add(ec, "b", token_ids=(1,), max_tokens=4)
        ec.aborts_queue.put_nowait("a")  # string form is tolerated
        ec.aborts_queue.put_nowait(["b"])
        ec._process_aborts_queue()
        # one merged batch, not one call per entry (order is set-deduped)
        assert len(ec.scheduler.finish_calls) == 1
        assert set(ec.scheduler.finish_calls[0]) == {"a", "b"}
        assert set(ec.scheduler.requests) == set()

    def test_abort_is_idempotent_in_scheduler(self):
        ec = engine_loop.EngineCore(mk_config(), engine_loop.UniProcExecutor, False)
        add(ec, "a", token_ids=(1,), max_tokens=4)
        first = ec.scheduler.finish_requests(
            ["a"], engine_loop.RequestStatus.FINISHED_ABORTED
        )
        assert [r.request_id for r in first] == ["a"]
        second = ec.scheduler.finish_requests(
            ["a"], engine_loop.RequestStatus.FINISHED_ABORTED
        )
        assert second == []  # "aborting in the scheduler is idempotent"

    def test_wakeup_sentinel_is_a_noop(self):
        proc = bare_proc()
        proc._handle_client_request(engine_loop.EngineCoreRequestType.WAKEUP, None)
        assert proc.scheduler.finish_calls == []


# ---------------------------------------------------------------------------
# m2 + m9: busy loop, shutdown, death
# ---------------------------------------------------------------------------


class TestBusyLoop:
    def test_idle_parks_in_blocking_get_not_spinning(self):
        proc = bare_proc()
        steps_before = proc.scheduler.current_step
        t = threading.Thread(target=proc._process_input_queue, daemon=True)
        t.start()
        time.sleep(0.4)
        # Parked: no schedule() happened during the quiet window.
        assert proc.scheduler.current_step == steps_before
        # A queued request wakes it and is dispatched via input_queue.
        req, wave = proc.preprocess_add_request(core_request("r"))
        proc.input_queue.put_nowait(
            (engine_loop.EngineCoreRequestType.ADD, (req, wave))
        )
        t.join(10)
        assert not t.is_alive()
        assert "r" in proc.scheduler.requests

    def test_non_block_mode_breaks_after_one_pass(self):
        proc = bare_proc()
        proc.process_input_queue_block = False
        proc._process_input_queue()  # returns despite empty queue + no work
        assert proc.scheduler.current_step == 0

    def test_run_busy_loop_raises_system_exit_when_drained(self):
        proc = bare_proc()
        outcome = {}

        def run():
            try:
                proc.run_busy_loop()
            except SystemExit:
                outcome["exit"] = True

        t = threading.Thread(target=run, daemon=True)
        t.start()
        time.sleep(0.2)
        proc.shutdown_state = engine_loop.EngineShutdownState.REQUESTED
        proc.input_queue.put_nowait((engine_loop.EngineCoreRequestType.WAKEUP, None))
        t.join(10)
        assert outcome.get("exit") is True


class TestShutdown:
    def test_abort_mode_finishes_inflight_and_notifies_clients(self):
        proc = bare_proc(shutdown_timeout=0)
        add(proc, "a", token_ids=(1,), max_tokens=100, client_index=2)
        proc.shutdown_state = engine_loop.EngineShutdownState.REQUESTED
        # First arbitration: abort-all happens (outputs are sent), but the
        # finished-flush work keeps the loop alive one more step.
        assert proc._handle_shutdown() is True
        assert proc.shutdown_state == engine_loop.EngineShutdownState.SHUTTING_DOWN
        sent = drain(proc.output_queue)
        # the abort output is routed to the owning client (client_index=2)
        assert sent[0][0] == 2
        assert set(sent[0][1].finished_requests) == {"a"}
        assert sent[0][1].outputs[0].finish_reason is engine_loop.FinishReason.ABORT
        proc.step()  # flush step: 0-token schedule → finished set drained
        assert proc._handle_shutdown() is False  # drained → exit

    def test_drain_mode_keeps_stepping_until_empty(self):
        proc = bare_proc(shutdown_timeout=300)
        add(proc, "a", token_ids=(1,), max_tokens=100)
        proc.enqueue_forward_logits([{"a": logits_row(5)}] * 2)
        proc.shutdown_state = engine_loop.EngineShutdownState.REQUESTED
        assert proc._handle_shutdown() is True  # work remains → keep going
        assert proc.shutdown_state == engine_loop.EngineShutdownState.SHUTTING_DOWN
        proc.step()  # drain continues stepping
        # still unfinished (max_tokens=100): loop must continue
        assert proc._handle_shutdown() is True

    def test_add_rejected_during_shutdown(self):
        proc = bare_proc()
        proc.shutdown_state = engine_loop.EngineShutdownState.SHUTTING_DOWN
        req, wave = proc.preprocess_add_request(core_request("late"))
        proc._handle_client_request(engine_loop.EngineCoreRequestType.ADD, (req, wave))
        assert "late" not in proc.scheduler.requests
        sent = drain(proc.output_queue)
        assert sent and sent[0][1].outputs[0].finish_reason is engine_loop.FinishReason.ABORT

    def test_executor_failed_sentinel_raises(self):
        proc = bare_proc()
        with pytest.raises(RuntimeError, match="Executor failed."):
            proc._handle_client_request(
                engine_loop.EngineCoreRequestType.EXECUTOR_FAILED, b""
            )

    def test_engine_dead_single_frame_join(self):
        proc = bare_proc()
        proc._send_engine_dead()
        assert (
            proc.output_queue.get_nowait()
            == engine_loop.EngineCoreProc.ENGINE_CORE_DEAD
        )

    def test_freeze_gc_heap_is_callable_and_idempotent(self):
        import gc

        engine_loop.freeze_gc_heap()
        engine_loop.freeze_gc_heap()
        gc.unfreeze()  # pair with EngineCore.shutdown's gc.unfreeze()


# ---------------------------------------------------------------------------
# m11: step_fn static binding; m5: preprocess off the busy loop
# ---------------------------------------------------------------------------


class TestBindingsAndPreprocess:
    def test_step_fn_bound_to_step_without_batch_queue(self):
        ec = engine_loop.EngineCore(mk_config(), engine_loop.UniProcExecutor, False)
        assert ec.batch_queue_size == 1
        assert ec.batch_queue is None
        assert ec.step_fn == ec.step  # static binding, zero per-step branching

    def test_batch_queue_built_when_concurrency_gt_one(self):
        cfg = mk_config(max_concurrent_batches=2)
        with pytest.raises(AttributeError, match="step_with_batch_queue"):
            # The overlap version is ch12's product: with the batch queue
            # enabled the static binding references a method this companion
            # does not carry (documented structural hole).
            engine_loop.EngineCore(cfg, engine_loop.UniProcExecutor, False)

    def test_preprocess_builds_request_and_starts_grammar_init(self):
        ec = engine_loop.EngineCore(mk_config(), engine_loop.UniProcExecutor, False)
        req, wave = ec.preprocess_add_request(
            core_request("r", token_ids=(1, 2), max_tokens=3)
        )
        assert isinstance(req, engine_loop.Request)
        assert req.num_prompt_tokens == 2
        assert req.use_structured_output is False
        assert wave == 0
        ec.preprocess_add_request(
            core_request("s", token_ids=(1,), max_tokens=3, structured=True)
        )
        # grammar_init runs here, on the input thread — not on the busy loop
        assert ec.structured_output_manager.trace == ["grammar_init:s"]


# ---------------------------------------------------------------------------
# m10: InprocClient — the same heart without a busy loop
# ---------------------------------------------------------------------------


class TestInprocClient:
    def test_docstring_advertises_no_busy_loop(self):
        assert "no busy loop" in engine_loop.InprocClient.__doc__

    def test_get_output_steps_the_engine_core_directly(self):
        client = engine_loop.InprocClient(
            mk_config(), engine_loop.UniProcExecutor, False
        )
        client.add_request(core_request("r", token_ids=(1,), max_tokens=2))
        client.engine_core.enqueue_forward_logits([{"r": logits_row(4)}, {"r": logits_row(5)}])
        outs1 = client.get_output()
        assert outs1.outputs[0].new_token_ids == [4]
        outs2 = client.get_output()
        assert outs2.outputs[0].finish_reason is engine_loop.FinishReason.LENGTH
        client.abort_requests(["r"])  # idempotent face
        client.shutdown()


# ---------------------------------------------------------------------------
# e2e over the real wire: m8 handshake, m1/m4 loop, m6/m7 IO, m9 death
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_ready_response_carries_post_init_config(self, lab):
        ready = lab.ready_responses[0]
        # post-init config that only the engine knows after construction
        assert ready.num_gpu_blocks == 128  # seam value of memory profiling
        assert ready.block_size == lab.cfg.cache_config.block_size
        assert ready.max_model_len == lab.cfg.model_config.max_model_len
        assert ready.max_num_seqs == lab.cfg.scheduler_config.max_num_seqs
        assert ready.vllm_version == "0.27.1"
        assert ready.data_parallel_rank == 0
        assert ready.dp_stats_address is None

    def test_generate_to_completion_over_the_wire(self, lab):
        lab.call_utility("clear_forward_scripts")
        # script the forward FIRST (the engine steps as soon as the ADD lands)
        lab.call_utility(
            "enqueue_forward_logits",
            [
                {"e2e-1": logits_row(7, vocab=16)},
                {"e2e-1": logits_row(8, vocab=16)},
                {"e2e-1": logits_row(9, vocab=16)},
            ],
        )
        lab.add(core_request("e2e-1", token_ids=(1, 2, 3), max_tokens=3))
        got = []
        for _ in range(3):
            outs = lab.recv()
            assert outs.engine_index == 0  # m6: stamped by the output IO thread
            got.append(outs.outputs[0])
        assert [o.new_token_ids for o in got] == [[7], [8], [9]]
        assert got[-1].finish_reason is engine_loop.FinishReason.LENGTH
        # engine parks once idle: step count freezes (m2)
        c1 = lab.call_utility("get_step_count")
        time.sleep(0.4)
        c2 = lab.call_utility("get_step_count")
        assert c1 == c2

    def test_structured_output_bitmask_constrains_sampling_e2e(self, lab):
        lab.call_utility("clear_forward_scripts")
        lab.call_utility("enqueue_forward_logits", [{"e2e-str": logits_row(5)}])
        lab.call_utility("enqueue_grammar_bitmask", [[0b100]])  # only token 2
        lab.add(core_request("e2e-str", token_ids=(1, 2), max_tokens=1, structured=True))
        outs = lab.recv()
        assert outs.outputs[0].new_token_ids == [2]

    def test_abort_midflight_silences_the_request(self, lab):
        # One long request, generously scripted: the abort lands within a few
        # steps of the wire (forward ≈ 5ms each) — between steps or eagerly
        # during one (aborts_queue); after it the request never resurfaces
        # (update_from_output's real skip-comment case) and the engine parks.
        lab.call_utility("clear_forward_scripts")
        lab.call_utility(
            "enqueue_forward_logits", [{"e2e-abort": logits_row(3)}] * 100
        )
        lab.add(core_request("e2e-abort", token_ids=(1,), max_tokens=200))
        outs = lab.recv()  # first token
        assert outs.outputs[0].new_token_ids == [3]
        lab.abort(["e2e-abort"])
        # drain any tokens already in flight, then require silence
        deadline = time.monotonic() + 10
        silent = False
        while time.monotonic() < deadline:
            if not lab.output_sockets[0].poll(600):
                silent = True
                break
            lab._recv_raw(0, timeout=0.1)
        assert silent
        # idempotent double-abort over the wire (dual-channel echo)
        lab.abort(["e2e-abort"])
        assert lab.call_utility("get_request_info", "e2e-abort") is None
    def test_utility_rpc_pairing_and_failure_path(self, lab):
        # msgpack carries the tuple back as a list (real msgspec Any decode)
        assert lab.call_utility("get_supported_tasks") == ["generate"]
        with pytest.raises(Exception, match="Call to boom_method method failed"):
            lab.call_utility("boom_method")

    def test_engine_dead_single_frame_on_fatal_error(self):
        lab = Lab()
        try:
            lab.add(core_request("dying", token_ids=(1,), max_tokens=5))
            # No logits scripted: the seam forward runs dry → the real error
            # path: busy loop raise → _send_engine_dead → single-frame
            # sentinel over the PUSH socket (linger=4000 keeps it ahead of
            # the socket close).
            frames = None
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if lab.output_sockets[0].poll(500):
                    raw = lab.output_sockets[0].recv_multipart(copy=False)
                    if bytes(raw[0]) == engine_loop.EngineCoreProc.ENGINE_CORE_DEAD:
                        frames = raw
                        break
            assert frames is not None
            lab.manager.processes[0].join(15)
            assert lab.manager.processes[0].exitcode != 0
        finally:
            try:
                lab.close()
            except Exception:
                pass

    def test_two_frontends_routed_by_client_index(self, two_client_lab):
        lab = two_client_lab
        lab.call_utility(
            "enqueue_forward_logits", [{"c0-r": logits_row(6), "c1-r": logits_row(7)}] * 4
        )
        lab.add(core_request("c0-r", token_ids=(1,), max_tokens=2), client=0)
        lab.add(
            core_request("c1-r", token_ids=(2,), max_tokens=2, client_index=1),
            client=1,
        )
        seen = {0: [], 1: []}
        for _ in range(2):
            for c in (0, 1):
                outs = lab.recv(client=c)
                seen[c].extend(o.request_id for o in outs.outputs)
        assert seen[0] == ["c0-r", "c0-r"]
        assert seen[1] == ["c1-r", "c1-r"]
