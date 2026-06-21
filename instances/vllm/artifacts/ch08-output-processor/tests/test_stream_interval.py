"""stream_interval throttling + DELTA sent_tokens_offset correctness.

Asserts vLLM's observable behavior with stream_interval = k:
 - the first token is always emitted;
 - intermediate steps that haven't accumulated k new tokens are suppressed
   (make_request_output returns None);
 - once >= k tokens accumulated, an output is emitted;
 - completion always forces an emit;
 - in DELTA mode, sent_tokens_offset advances so emitted token slices never
   overlap (no duplicate tokens across emits).
"""

from conftest import FakeRequest, FakeSamplingParams, char_tokenizer, ids

from implementation._types import EngineCoreOutput, FinishReason, RequestOutputKind
from implementation.output_processor import OutputProcessor, RequestOutputCollector


def _setup(kind, stream_interval):
    op = OutputProcessor(char_tokenizer, stream_interval=stream_interval)
    sp = FakeSamplingParams(output_kind=kind)
    req = FakeRequest(request_id="r0", external_req_id="r0", sampling_params=sp)
    q = RequestOutputCollector(kind, "r0")
    op.add_request(req, prompt=None, queue=q)
    return op, q


def _step(op, q, text, finished=False):
    op.process_outputs([
        EngineCoreOutput(
            request_id="r0",
            new_token_ids=ids(text),
            finish_reason=FinishReason.LENGTH if finished else None,
            finished=finished,
        )
    ])
    return q.get_nowait()


def test_first_token_always_emitted():
    op, q = _setup(RequestOutputKind.DELTA, stream_interval=4)
    out = _step(op, q, "a")  # token #0 first -> emit
    assert out is not None
    assert out.outputs[0].text == "a"


def test_intermediate_suppressed_until_interval():
    op, q = _setup(RequestOutputKind.DELTA, stream_interval=4)
    _step(op, q, "a")        # first -> emit (offset advances to 1)
    assert _step(op, q, "b") is None   # 2-1 < 4 suppressed
    assert _step(op, q, "c") is None   # 3-1 < 4 suppressed
    assert _step(op, q, "d") is None   # 4-1 < 4 suppressed
    out = _step(op, q, "e")  # 5-1 >= 4 -> emit
    assert out is not None
    # DELTA: emitted slice is the unsent tail b..e, no overlap with first 'a'
    assert out.outputs[0].text == "bcde"


def test_completion_forces_emit():
    op, q = _setup(RequestOutputKind.DELTA, stream_interval=10)
    _step(op, q, "a")  # first emit
    assert _step(op, q, "b") is None
    out = _step(op, q, "c", finished=True)  # finish forces emit even below interval
    assert out is not None
    assert out.finished is True
    # delta tail since last send
    assert out.outputs[0].text == "bc"


def test_delta_offset_no_overlap_full_sequence():
    op, q = _setup(RequestOutputKind.DELTA, stream_interval=2)
    collected = ""
    for ch in "abcdef":
        out = _step(op, q, ch)
        if out is not None:
            collected += out.outputs[0].text
    # flush remainder on finish
    out = _step(op, q, "g", finished=True)
    if out is not None:
        collected += out.outputs[0].text
    assert collected == "abcdefg"  # exact, no duplication, no loss
