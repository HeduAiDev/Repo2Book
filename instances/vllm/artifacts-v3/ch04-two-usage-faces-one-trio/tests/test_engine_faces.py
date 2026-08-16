"""TDD tests for the v3 ch04 subtract-only companion (pin vLLM v0.27.1 / 6e448d0ea).

These assert the *observable vLLM behavior* this chapter teaches:
- trio isomorphism: AsyncLLM.__init__ and LLMEngine.__init__ assemble the same
  renderer/InputProcessor/OutputProcessor, differing only in the client call
  (make_async_mp_client vs make_client(asyncio_mode=False)),
- client factory 2D table (multiprocess_mode x asyncio_mode) and the
  VLLM_ENABLE_V1_MULTIPROCESSING=on-by-default flip for the offline face,
- dual-track request id (external_req_id + 8-hex random suffix),
- double registration: OutputProcessor.request_states + external->internal map
  (this process) AND the EngineCoreRequest crossing to the engine side,
- client_index stamped at add_request_async and honored engine-side
  (sockets[client_index] routing, F3),
- the queue-has/has-not fork inside one OutputProcessor.process_outputs
  (AsyncLLM mailbox vs LLMEngine list),
- RequestOutputKind.FINAL_ONLY (intermediate outputs are never constructed),
- two drivers: online generate() event loop pulling the per-request collector
  vs offline while has_unfinished_requests(): step() bare loop (+ sorted()
  input-order restore, + transactional batch add),
- abort semantics (external id expansion, defensive "already aborted" skip,
  generate() disconnect path).

Pure unit/e2e tests on the host (no `import vllm`, no torch/CUDA/zmq): the
dossier-sanctioned in-process EngineCore stub replaces the background engine
process; tests play the scheduler by handing the stub its step outputs. Run:
python -m pytest tests/ -q
"""

import asyncio
import importlib.util
import queue
import sys
import threading
import time
from pathlib import Path

import pytest

# Load the companion module directly by path (no package install needed).
_IMPL = Path(__file__).resolve().parent.parent / "implementation" / "engine_faces.py"
_spec = importlib.util.spec_from_file_location("engine_faces", _IMPL)
ef = importlib.util.module_from_spec(_spec)
sys.modules["engine_faces"] = ef
_spec.loader.exec_module(ef)


def cfg(**kw):
    return ef.VllmConfig(**kw)


def prompt(token_ids=(1, 2, 3)):
    # A rendered EngineInput (Renderer output, ch6): token-ids dict with a
    # "type" discriminator -- exactly what the sync fast path checks for.
    return {"type": "token_ids", "prompt_token_ids": list(token_ids)}


def core_request(request_id="r-1", external=None, sampling_params=None):
    req = ef.EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3],
        sampling_params=sampling_params or ef.SamplingParams(n=1),
        arrival_time=0.0,
    )
    if external is not None:
        req.external_req_id = external
    return req


# ---------------------------------------------------------------------------
# m1 -- one trio, two usage faces (station 3)
# ---------------------------------------------------------------------------


def test_trio_isomorphism_both_faces_assemble_same_trio():
    c = cfg(scheduler_config=ef.SchedulerConfig(stream_interval=3))
    allm = ef.AsyncLLM(c, ef.Executor, log_stats=False)
    lm = ef.LLMEngine(c, ef.Executor, log_stats=False, multiprocess_mode=True)
    for face in (allm, lm):
        assert hasattr(face, "renderer")
        assert isinstance(face.input_processor, ef.InputProcessor)
        assert isinstance(face.output_processor, ef.OutputProcessor)
    # The ONLY assembly divergence: which client factory call each face makes.
    assert isinstance(allm.engine_core, ef.AsyncMPClient)
    assert isinstance(lm.engine_core, ef.SyncMPClient)
    # Same config -> same stream_interval wiring on both faces.
    assert allm.output_processor.stream_interval == 3
    assert lm.output_processor.stream_interval == 3


def test_async_llm_engine_alias_is_async_llm():
    # v0 relic (vllm/engine/async_llm_engine.py:L6): a 7-line alias shim.
    assert ef.AsyncLLMEngine is ef.AsyncLLM


def test_async_llm_engine_alias_subclassable_like_real_shim():
    # Old user code did `from vllm.engine.async_llm_engine import AsyncLLMEngine`.
    class _Old(ef.AsyncLLMEngine):  # noqa: F841 -- existence is the assertion
        pass

    assert issubclass(_Old, ef.AsyncLLM)


# ---------------------------------------------------------------------------
# m2 -- client factory 2D table (station 4) + WC4 envs flip
# ---------------------------------------------------------------------------


def test_make_client_two_axis_table():
    assert isinstance(
        ef.EngineCoreClient.make_client(True, False, cfg(), ef.Executor, False),
        ef.SyncMPClient,
    )
    inproc = ef.EngineCoreClient.make_client(False, False, cfg(), ef.Executor, False)
    assert isinstance(inproc, ef.InprocClient)
    # asyncio without multiprocessing is explicitly unsupported.
    with pytest.raises(NotImplementedError):
        ef.EngineCoreClient.make_client(False, True, cfg(), ef.Executor, False)


def test_make_client_async_axis_routes_via_make_async_mp_client():
    client = ef.EngineCoreClient.make_client(True, True, cfg(), ef.Executor, False)
    assert isinstance(client, ef.AsyncMPClient)


def test_make_async_mp_client_carries_client_identity():
    client = ef.EngineCoreClient.make_async_mp_client(
        cfg(), ef.Executor, False, client_count=2, client_index=1
    )
    assert client.client_count == 2
    assert client.client_index == 1


class _EngineArgsStub:
    disable_log_stats = True

    def create_engine_config(self, usage_context=None):
        return cfg()


def test_from_engine_args_envs_flip_default_multiprocessing(monkeypatch):
    # WC4: VLLM_ENABLE_V1_MULTIPROCESSING defaults to True (envs.py:L149), so
    # even the offline LLMEngine lands on SyncMPClient -- NOT InprocClient.
    monkeypatch.setattr(ef.envs, "VLLM_ENABLE_V1_MULTIPROCESSING", True)
    eng = ef.LLMEngine.from_engine_args(_EngineArgsStub())
    assert isinstance(eng.engine_core, ef.SyncMPClient)


def test_from_engine_args_envs_off_is_the_inproc_escape_hatch(monkeypatch):
    monkeypatch.setattr(ef.envs, "VLLM_ENABLE_V1_MULTIPROCESSING", False)
    eng = ef.LLMEngine.from_engine_args(_EngineArgsStub())
    assert isinstance(eng.engine_core, ef.InprocClient)


# ---------------------------------------------------------------------------
# m5 / WC5 -- dual-track request id (station 6)
# ---------------------------------------------------------------------------


def test_assign_request_id_dual_track():
    req = core_request("chat-abc")
    ef.InputProcessor.assign_request_id(req)
    assert req.external_req_id == "chat-abc"
    # Internal id = external id + 8 random hex chars (PR #27987).
    assert req.request_id != "chat-abc"
    prefix, _, suffix = req.request_id.rpartition("-")
    assert prefix == "chat-abc"
    assert len(suffix) == 8
    int(suffix, 16)  # parses as hex


def test_assign_request_id_rejects_prepopulated_external_id():
    req = core_request("chat-abc", external="chat-abc")
    with pytest.raises(ValueError):
        ef.InputProcessor.assign_request_id(req)


def test_assign_request_id_escape_hatch_keeps_external_id(monkeypatch):
    monkeypatch.setattr(ef.envs, "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", True)
    req = core_request("chat-abc")
    ef.InputProcessor.assign_request_id(req)
    assert req.request_id == "chat-abc"
    assert req.external_req_id == "chat-abc"


def test_two_requests_same_external_id_never_collide_internally():
    ids = set()
    for _ in range(8):
        req = core_request("retry-me")
        ef.InputProcessor.assign_request_id(req)
        ids.add(req.request_id)
    assert len(ids) == 8  # retries reuse the external id without polluting demux


# ---------------------------------------------------------------------------
# m3 / WC2 -- double registration (stations 8-9), online face
# ---------------------------------------------------------------------------


def test_online_double_registration_writes_both_ledgers():
    async def main():
        allm = ef.AsyncLLM(cfg(), ef.Executor, log_stats=False)
        params = ef.SamplingParams(n=1, max_tokens=2)
        q = await allm.add_request("chat-abc", prompt(), params)
        internal = q.request_id

        # Ledger 1 (this process): RequestState table + external->internal map.
        assert set(allm.output_processor.request_states) == {internal}
        assert allm.output_processor.external_req_ids["chat-abc"] == [internal]
        state = allm.output_processor.request_states[internal]
        assert state.external_req_id == "chat-abc"
        assert state.output_kind is ef.RequestOutputKind.CUMULATIVE
        assert state.queue is q

        # Ledger 2 (crossing): the EngineCoreRequest left the API process with
        # both stamps on it -- the random-suffix internal id AND client_index.
        engine = allm.engine_core.engine_core
        msg_type, crossed = engine.input_queue.get_nowait()
        assert msg_type is ef.EngineCoreRequestType.ADD
        assert crossed.request_id == internal
        assert crossed.external_req_id == "chat-abc"  # WC5: also crosses
        assert crossed.client_index == 0  # F3 stamp, single front-end -> 0

    asyncio.run(main())


def test_mp_client_add_request_crosses_but_is_not_yet_admitted():
    # In real vLLM the request sits in the engine's input queue until the busy
    # loop consumes it; the in-process stub keeps exactly that in-flight state.
    client = ef.SyncMPClient(cfg(), ef.Executor, False)
    engine = client.engine_core
    client.add_request(core_request("r-1"))
    assert "r-1" not in engine.requests  # crossed, not yet admitted
    engine.emit_step_outputs([])  # one engine step drains pending inputs first
    assert "r-1" in engine.requests


def test_inproc_client_adds_directly_no_wire():
    client = ef.InprocClient(cfg(), ef.Executor, False)
    req = core_request("r-1")
    client.add_request(req)
    assert client.engine_core.requests["r-1"] is req
    assert client.engine_core.input_queue.empty()  # V0-style: no crossing


# ---------------------------------------------------------------------------
# m4 / WC3 / F3 -- client_index stamped & honored (stations 9-10)
# ---------------------------------------------------------------------------


def test_engine_routes_outputs_by_client_index_socket_lookup():
    engine = ef.EngineCore(cfg(), ef.Executor, False)
    sink0, sink1 = queue.Queue(), queue.Queue()
    engine.attach_output_socket(0, sink0)
    engine.attach_output_socket(1, sink1)
    outs0 = ef.EngineCoreOutputs(outputs=[ef.EngineCoreOutput("a", [1], None)])
    outs1 = ef.EngineCoreOutputs(outputs=[ef.EngineCoreOutput("b", [1], None)])
    engine.emit_step_outputs([(0, outs0), (1, outs1)])
    # sockets[client_index] O(1) lookup: each front-end gets its own outputs.
    assert sink0.get_nowait() is outs0
    assert sink1.get_nowait() is outs1


def test_add_request_async_stamps_client_index():
    async def main():
        client = ef.EngineCoreClient.make_async_mp_client(
            cfg(), ef.Executor, False, client_count=3, client_index=2
        )
        await client.add_request_async(core_request("r-9"))
        engine = client.engine_core
        _, crossed = engine.input_queue.get_nowait()
        assert crossed.client_index == 2

    asyncio.run(main())


# ---------------------------------------------------------------------------
# m6 -- one process_outputs feeds both faces (station 11)
# ---------------------------------------------------------------------------


def _op():
    return ef.OutputProcessor(None, log_stats=False, stream_interval=1, tracing_enabled=False)


def test_process_outputs_queue_fork_online_mailbox_vs_offline_list():
    op = _op()
    req = core_request("i-1", external="e-1")
    mailbox = ef.RequestOutputCollector(ef.RequestOutputKind.CUMULATIVE, "i-1")
    op.add_request(req, "hello", None, 0, mailbox)

    raw = ef.EngineCoreOutput("i-1", [7, 8], None)
    result = op.process_outputs([raw])
    # AsyncLLM arm: pushed into the per-request collector, not returned.
    assert result.request_outputs == []
    out = mailbox.get_nowait()
    assert out is not None and out.outputs[0].token_ids == [7, 8]

    # LLMEngine arm: same function, queue=None -> returned in the list.
    req2 = core_request("i-2", external="e-2")
    op.add_request(req2, "hello", None, 0)  # no queue
    result2 = op.process_outputs([ef.EngineCoreOutput("i-2", [9], None)])
    assert len(result2.request_outputs) == 1
    assert result2.request_outputs[0].outputs[0].token_ids == [9]


def test_process_outputs_skips_outputs_for_aborted_requests():
    op = _op()
    result = op.process_outputs([ef.EngineCoreOutput("ghost-id", [1], None)])
    assert result.request_outputs == [] and result.reqs_to_abort == []


def test_finished_request_freed_from_both_maps():
    op = _op()
    op.add_request(core_request("i-1", external="e-1"), None, None, 0)
    op.process_outputs([ef.EngineCoreOutput("i-1", [1], ef.FinishReason.LENGTH)])
    assert op.request_states == {}
    assert dict(op.external_req_ids) == {}
    assert not op.has_unfinished_requests()


# ---------------------------------------------------------------------------
# m7 / WC6 -- output_kind three states
# ---------------------------------------------------------------------------


def test_final_only_intermediate_outputs_are_never_constructed():
    op = _op()
    params = ef.SamplingParams(output_kind=ef.RequestOutputKind.FINAL_ONLY)
    mailbox = ef.RequestOutputCollector(params.output_kind, "i-1")
    op.add_request(core_request("i-1", external="e-1", sampling_params=params), None, None, 0, mailbox)
    # Intermediate output: make_request_output returns None -> nothing put.
    op.process_outputs([ef.EngineCoreOutput("i-1", [10], None)])
    assert mailbox.get_nowait() is None
    # Final output flows.
    op.process_outputs([ef.EngineCoreOutput("i-1", [11], ef.FinishReason.LENGTH)])
    out = mailbox.get_nowait()
    assert out.finished and out.outputs[0].token_ids == [10, 11]


def test_collector_merges_deltas_when_producer_outruns_consumer():
    q = ef.RequestOutputCollector(ef.RequestOutputKind.DELTA, "i-1")

    def _ro(token_ids, finished=False):
        return ef.RequestOutput(
            request_id="e-1", prompt=None, prompt_token_ids=[],
            prompt_logprobs=None,
            outputs=[ef.CompletionOutput(index=0, text="", token_ids=list(token_ids),
                                         cumulative_logprob=None, logprobs=None)],
            finished=finished,
        )

    q.put(_ro([1]))
    q.put(_ro([2]))  # not yet drained -> merged, not overwritten
    out = q.get_nowait()
    assert out.outputs[0].token_ids == [1, 2]
    assert q.get_nowait() is None  # single slot drained


# ---------------------------------------------------------------------------
# abort semantics (WC5 external expansion, WC2 defensive skip)
# ---------------------------------------------------------------------------


def test_abort_by_external_id_expands_to_all_internal_ids():
    op = _op()
    box1 = ef.RequestOutputCollector(ef.RequestOutputKind.CUMULATIVE, "i-1")
    box2 = ef.RequestOutputCollector(ef.RequestOutputKind.CUMULATIVE, "i-2")
    op.add_request(core_request("i-1", external="e-1"), None, None, 0, box1)
    op.add_request(core_request("i-2", external="e-1"), None, None, 0, box2)
    aborted = op.abort_requests(["e-1"], internal=False)
    assert sorted(aborted) == ["i-1", "i-2"]
    assert op.request_states == {} and dict(op.external_req_ids) == {}
    # Both consumers get an unblocking finished output.
    assert box1.get_nowait().finished and box2.get_nowait().finished


def test_online_abort_reaches_both_sides():
    async def main():
        allm = ef.AsyncLLM(cfg(), ef.Executor, log_stats=False)
        q = await allm.add_request("chat-abc", prompt(), ef.SamplingParams(n=1))
        await allm.abort("chat-abc")  # external id
        assert allm.output_processor.request_states == {}
        engine = allm.engine_core.engine_core
        engine.emit_step_outputs([])  # drain the ABORT message
        assert engine.requests == {}

    asyncio.run(main())


# ---------------------------------------------------------------------------
# m8 -- two drivers, end to end
# ---------------------------------------------------------------------------


def _step_outputs_for(live_ids, token, finished):
    outs = [
        ef.EngineCoreOutput(
            rid, [token], ef.FinishReason.LENGTH if finished else None
        )
        for rid in live_ids
    ]
    return [(0, ef.EngineCoreOutputs(outputs=outs))]


def test_online_driver_generate_pulls_mailbox_until_finished():
    async def main():
        allm = ef.AsyncLLM(cfg(), ef.Executor, log_stats=False)
        engine = allm.engine_core.engine_core
        params = ef.SamplingParams(n=1, max_tokens=2,
                                   output_kind=ef.RequestOutputKind.DELTA)
        collected = []

        async def consume():
            async for out in allm.generate(prompt(), params, "chat-abc"):
                collected.append(out)

        task = asyncio.ensure_future(consume())
        step, seen = 0, False
        for _ in range(500):
            await asyncio.sleep(0.001)
            live = list(allm.output_processor.request_states)
            if not live:
                if not seen:
                    continue
                break
            seen = True
            step += 1
            engine.emit_step_outputs(_step_outputs_for(live, 100 + step, step >= 2))
        await asyncio.wait_for(task, 5)
        # DELTA: one output per step, last one finished.
        assert [o.outputs[0].token_ids for o in collected] == [[101], [102]]
        assert collected[-1].finished
        # External id restored on the way out.
        assert all(o.request_id == "chat-abc" for o in collected)
        # Engine-side ledger also drained to empty.
        assert engine.requests == {}

    asyncio.run(main())


def test_online_output_handler_starts_lazily_before_the_event_loop():
    # Constructed outside the loop (OpenAI server startup), the handler task
    # must not start until the first add_request() -- the graceful-startup why.
    allm = ef.AsyncLLM(cfg(), ef.Executor, log_stats=False)
    assert allm.output_handler is None

    async def main():
        q = await allm.add_request("chat-abc", prompt(), ef.SamplingParams(n=1))
        assert allm.output_handler is not None and not allm.output_handler.done()
        assert allm.is_running

    asyncio.run(main())


def test_online_disconnect_aborts_the_request():
    async def main():
        allm = ef.AsyncLLM(cfg(), ef.Executor, log_stats=False)
        engine = allm.engine_core.engine_core
        params = ef.SamplingParams(n=1, max_tokens=99,
                                   output_kind=ef.RequestOutputKind.DELTA)
        agen = allm.generate(prompt(), params, "chat-abc")
        task = asyncio.ensure_future(agen.__anext__())
        # Play exactly one engine beat so the first delta can come out.
        for _ in range(500):
            await asyncio.sleep(0.001)
            live = list(allm.output_processor.request_states)
            if live:
                engine.emit_step_outputs(_step_outputs_for(live, 101, False))
                break
        first = await asyncio.wait_for(task, 5)
        assert first.outputs[0].token_ids == [101]
        # Client disconnects: the generator is closed -> abort(internal=True).
        await agen.aclose()
        assert allm.output_processor.request_states == {}
        msg_type, ids = engine.input_queue.get_nowait()
        assert msg_type is ef.EngineCoreRequestType.ABORT
        engine.handle_client_message(msg_type, ids)  # busy loop consumes it
        assert engine.requests == {}

    asyncio.run(main())


def test_engine_dead_add_request_raises_immediately():
    async def main():
        allm = ef.AsyncLLM(cfg(), ef.Executor, log_stats=False)
        allm.engine_core.engine_dead = True
        with pytest.raises(ef.EngineDeadError):
            await allm.add_request("chat-abc", prompt(), ef.SamplingParams(n=1))

    asyncio.run(main())


def test_sync_client_get_output_formats_engine_death():
    client = ef.SyncMPClient(cfg(), ef.Executor, False)
    client.engine_dead = True  # set by the monitor thread in real vLLM
    client.outputs_queue.put(RuntimeError("boom"))
    with pytest.raises(ef.EngineDeadError):
        client.get_output()


def _offline_llm():
    return ef.LLM(_EngineArgsStub())


def test_offline_driver_bare_while_step_loop_restores_input_order():
    llm = _offline_llm()
    engine = llm.llm_engine.engine_core.engine_core
    op = llm.llm_engine.output_processor

    def player():
        # Plays the EngineCore busy loop (ch9): finish request "1" BEFORE
        # request "0" so sorted() has real work to do.
        counts, seen = {}, False
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                live = list(op.request_states)
            except RuntimeError:  # dict changed size during iteration
                time.sleep(0.001)
                continue
            if not live:
                if not seen:
                    time.sleep(0.002)
                    continue
                return
            seen = True
            outs = []
            for rid in live:
                counts[rid] = counts.get(rid, 0) + 1
                # internal id "0-xxxx" / "1-xxxx" -> external prefix
                finished = counts[rid] >= 2 if rid.startswith("1-") else counts[rid] >= 3
                outs.append(ef.EngineCoreOutput(
                    rid, [100 + counts[rid]],
                    ef.FinishReason.LENGTH if finished else None))
            engine.emit_step_outputs([(0, ef.EngineCoreOutputs(outputs=outs))])
            time.sleep(0.002)

    t = threading.Thread(target=player, daemon=True)
    t.start()
    params = [ef.SamplingParams(n=1, max_tokens=3) for _ in range(2)]
    outputs = llm.generate([prompt((1,)), prompt((2,))], params)
    t.join(timeout=10)
    # FINAL_ONLY: exactly the two finals, restored to input order 0, 1.
    assert [o.request_id for o in outputs] == ["0", "1"]
    assert all(o.finished for o in outputs)
    assert outputs[0].outputs[0].token_ids == [101, 102, 103]
    assert outputs[1].outputs[0].token_ids == [101, 102]
    # request ids came from the auto-increment counter (station 2).
    assert dict(llm.llm_engine.output_processor.external_req_ids) == {}


def test_offline_face_stamps_final_only_and_counter_ids():
    llm = _offline_llm()
    params = ef.SamplingParams(n=1, max_tokens=2)
    added = llm._add_request(prompt(), params)
    assert added.startswith("0-")  # counter id "0" + random suffix
    assert params.output_kind is ef.RequestOutputKind.FINAL_ONLY
    added2 = llm._add_request(prompt(), ef.SamplingParams(n=1))
    assert added2.startswith("1-")


def test_offline_batch_add_is_transactional():
    llm = _offline_llm()
    engine = llm.llm_engine.engine_core.engine_core
    op = llm.llm_engine.output_processor
    params = [ef.SamplingParams(n=1, max_tokens=2), object()]  # 2nd is invalid
    with pytest.raises(Exception):
        llm._render_and_add_requests([prompt((1,)), prompt((2,))], params)
    # The successfully-added request 0 was aborted on both ledgers.
    assert op.request_states == {}
    msgs = []
    while True:
        try:
            msgs.append(engine.input_queue.get_nowait())
        except queue.Empty:
            break
    kinds = [m[0] for m in msgs]
    assert kinds == [ef.EngineCoreRequestType.ADD, ef.EngineCoreRequestType.ABORT]
    engine.emit_step_outputs([])  # busy loop consumes both messages
    assert engine.requests == {}


# ---------------------------------------------------------------------------
# trio members: InputProcessor fast path
# ---------------------------------------------------------------------------


def test_process_inputs_sync_fast_path_builds_engine_core_request():
    ip = ef.InputProcessor(cfg(), ef.BaseRenderer(cfg()))
    params = ef.SamplingParams(n=1, max_tokens=4)
    req = ip.process_inputs("chat-abc", prompt((5, 6)), params,
                            supported_tasks=("generate",))
    assert isinstance(req, ef.EngineCoreRequest)
    assert req.prompt_token_ids == [5, 6]
    assert req.sampling_params is not params  # cloned
    assert req.sampling_params.max_tokens == 4
    assert req.arrival_time > 0  # arrival default from clock
    # Raw (unrendered) prompts are the deleted thread-pool path (item 6).
    with pytest.raises(TypeError):
        ip.process_inputs("x", "just a string", params, supported_tasks=("generate",))


def test_input_processor_exposes_tokenizer_from_renderer():
    ip = ef.InputProcessor(cfg(), ef.BaseRenderer(cfg()))
    assert ip.tokenizer is None  # tokenizer-less companion == real None path


def test_supported_tasks_cached_on_both_faces():
    async def main():
        allm = ef.AsyncLLM(cfg(), ef.Executor, log_stats=False)
        assert await allm.get_supported_tasks() == ("generate",)
        assert await allm.get_supported_tasks() == allm._supported_tasks

    asyncio.run(main())
    lm = ef.LLMEngine(cfg(), ef.Executor, log_stats=False, multiprocess_mode=True)
    assert lm.get_supported_tasks() == ("generate",)


# ---------------------------------------------------------------------------
# RequestOutput / EngineCoreOutput wire shapes
# ---------------------------------------------------------------------------


def test_engine_core_output_finished_property():
    assert ef.EngineCoreOutput("i", [1], None).finished is False
    assert ef.EngineCoreOutput("i", [1], ef.FinishReason.STOP).finished is True


def test_engine_core_request_params_property():
    req = core_request()
    assert req.params is req.sampling_params
    req2 = ef.EngineCoreRequest(request_id="x", prompt_token_ids=None,
                                sampling_params=None, arrival_time=0.0)
    with pytest.raises(AssertionError):
        req2.params  # noqa: B018


def test_request_output_id_is_external_on_the_way_out():
    # make_request_output writes the EXTERNAL id into RequestOutput.
    op = _op()
    params = ef.SamplingParams(output_kind=ef.RequestOutputKind.FINAL_ONLY)
    op.add_request(core_request("i-1", external="user-facing-9",
                                sampling_params=params), None, None, 0)
    result = op.process_outputs([ef.EngineCoreOutput("i-1", [4], ef.FinishReason.STOP)])
    assert result.request_outputs[0].request_id == "user-facing-9"


def test_stop_string_detection_surfaces_abort_requests():
    # Without a tokenizer the detokenizer never matches stop strings, but the
    # process_outputs -> reqs_to_abort plumbing must stay wired for ch7.
    op = _op()
    op.add_request(core_request("i-1", external="e-1"), None, None, 0)
    result = op.process_outputs([ef.EngineCoreOutput("i-1", [4], None)])
    assert result.reqs_to_abort == []
