"""OutputProcessor.process_outputs — the single loop (chapter main axis).

Asserts vLLM's observable behavior:
 - already-aborted request (no state) is silently skipped;
 - is_prefilling flips false on first output, num_cached_tokens recorded;
 - AsyncLLM path (queue set) pushes to the per-request collector and returns an
   empty request_outputs list; LLMEngine path (queue None) returns the list;
 - stop-string detected by detokenizer sets finish_reason=STOP and, when
   EngineCore had not finished, adds the req to reqs_to_abort (reverse abort);
 - finishing deregisters the request from all three maps (_finish_request);
 - FINAL_ONLY emits nothing until finished;
 - DELTA path pops prompt logprobs once at end of prefill.
"""

from conftest import FakeRequest, FakeSamplingParams, char_tokenizer, ids

from implementation._types import (
    EngineCoreOutput,
    FinishReason,
    PrefillStats,
    RequestOutputKind,
)
from implementation.output_processor import OutputProcessor, RequestOutputCollector


def _op():
    return OutputProcessor(char_tokenizer, stream_interval=1)


def _add(op, req_id, *, kind=RequestOutputKind.CUMULATIVE, queue=False, **sp_kw):
    sp = FakeSamplingParams(output_kind=kind, **sp_kw)
    req = FakeRequest(request_id=req_id, external_req_id=req_id, sampling_params=sp)
    q = RequestOutputCollector(kind, req_id) if queue else None
    op.add_request(req, prompt=None, queue=q)
    return q


def test_skip_already_aborted():
    op = _op()
    # no state registered for "ghost"
    res = op.process_outputs([EngineCoreOutput(request_id="ghost", new_token_ids=ids("a"))])
    assert res.request_outputs == []
    assert res.reqs_to_abort == []


def test_prefill_flip_and_num_cached():
    op = _op()
    _add(op, "r0")
    state = op.request_states["r0"]
    assert state.is_prefilling is True
    op.process_outputs([
        EngineCoreOutput(
            request_id="r0",
            new_token_ids=ids("hi"),
            prefill_stats=PrefillStats(num_cached_tokens=7),
        )
    ])
    assert state.is_prefilling is False
    assert state.num_cached_tokens == 7


def test_llmengine_path_returns_list():
    op = _op()
    _add(op, "r0", queue=False)  # LLMEngine path
    res = op.process_outputs([EngineCoreOutput(request_id="r0", new_token_ids=ids("hi"))])
    assert len(res.request_outputs) == 1
    assert res.request_outputs[0].outputs[0].text == "hi"


def test_async_path_pushes_to_queue_and_empties_list():
    op = _op()
    q = _add(op, "r0", queue=True)  # AsyncLLM path
    res = op.process_outputs([EngineCoreOutput(request_id="r0", new_token_ids=ids("hi"))])
    assert res.request_outputs == []  # nothing returned; pushed to queue
    out = q.get_nowait()
    assert out.outputs[0].text == "hi"


def test_stop_string_triggers_reverse_abort():
    op = _op()
    _add(op, "r0", stop=["STOP"])
    # EngineCore did NOT finish (finished=False), but detokenizer sees "STOP"
    res = op.process_outputs([
        EngineCoreOutput(
            request_id="r0",
            new_token_ids=ids("abSTOP"),
            finished=False,
        )
    ])
    assert res.reqs_to_abort == ["r0"]
    ro = res.request_outputs[0]
    assert ro.finished is True
    assert ro.outputs[0].finish_reason == "stop"
    assert ro.outputs[0].stop_reason == "STOP"
    # finished request deregistered
    assert "r0" not in op.request_states


def test_engine_finished_no_reverse_abort():
    op = _op()
    _add(op, "r0")
    res = op.process_outputs([
        EngineCoreOutput(
            request_id="r0",
            new_token_ids=ids("hi"),
            finish_reason=FinishReason.LENGTH,
            finished=True,
        )
    ])
    assert res.reqs_to_abort == []  # EngineCore already finished
    assert "r0" not in op.request_states


def test_finish_request_deregisters_all_maps():
    op = _op()
    _add(op, "r0")
    assert op.has_unfinished_requests()
    assert "r0" in op.external_req_ids
    op.process_outputs([
        EngineCoreOutput(
            request_id="r0",
            new_token_ids=ids("x"),
            finish_reason=FinishReason.LENGTH,
            finished=True,
        )
    ])
    assert op.get_num_unfinished_requests() == 0
    assert "r0" not in op.external_req_ids


def test_final_only_emits_only_at_finish():
    op = _op()
    _add(op, "r0", kind=RequestOutputKind.FINAL_ONLY)
    # intermediate step -> no output
    res = op.process_outputs([EngineCoreOutput(request_id="r0", new_token_ids=ids("ab"))])
    assert res.request_outputs == []
    # final step -> output, full text
    res = op.process_outputs([
        EngineCoreOutput(
            request_id="r0",
            new_token_ids=ids("cd"),
            finish_reason=FinishReason.LENGTH,
            finished=True,
        )
    ])
    assert len(res.request_outputs) == 1
    assert res.request_outputs[0].outputs[0].text == "abcd"


def test_delta_pops_prompt_logprobs_once():
    op = _op()
    _add(op, "r0", kind=RequestOutputKind.DELTA, prompt_logprobs=1)
    state = op.request_states["r0"]
    # simulate prompt logprobs accumulated during prefill
    state.logprobs_processor.prompt_logprobs = ["p0", "p1"]
    res = op.process_outputs([EngineCoreOutput(request_id="r0", new_token_ids=ids("a"))])
    assert res.request_outputs[0].prompt_logprobs == ["p0", "p1"]
    # subsequent output: prompt logprobs already forgotten
    res = op.process_outputs([EngineCoreOutput(request_id="r0", new_token_ids=ids("b"))])
    assert res.request_outputs[0].prompt_logprobs == []
