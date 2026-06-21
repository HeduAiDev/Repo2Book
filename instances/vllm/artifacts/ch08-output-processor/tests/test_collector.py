"""f1: RequestOutputCollector — per-request single-slot mailbox + DELTA merge.

Asserts vLLM's observable behavior:
 - empty slot accepts a put and sets the Event;
 - DELTA collector merges deltas when producer outruns consumer (text/token
   concatenation via RequestOutput.add);
 - CUMULATIVE collector replaces (no concat) for same index;
 - Exception preempts and is re-raised on get;
 - get blocks until put; get_nowait returns None when empty.
"""

import asyncio

import pytest

from implementation._types import CompletionOutput, RequestOutputKind
from implementation.outputs import RequestOutput
from implementation.output_processor import RequestOutputCollector


def _ro(req_id, index, text, token_ids, finished=False):
    return RequestOutput(
        request_id=req_id,
        prompt=None,
        prompt_token_ids=None,
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=index,
                text=text,
                token_ids=list(token_ids),
                cumulative_logprob=None,
                logprobs=None,
            )
        ],
        finished=finished,
    )


def test_put_sets_event_and_get_nowait():
    c = RequestOutputCollector(RequestOutputKind.DELTA, "r0")
    assert c.get_nowait() is None
    assert not c.ready.is_set()
    c.put(_ro("r0", 0, "ab", [1, 2]))
    assert c.ready.is_set()
    out = c.get_nowait()
    assert out.outputs[0].text == "ab"
    # slot cleared after take
    assert c.get_nowait() is None
    assert not c.ready.is_set()


def test_delta_merge_when_producer_ahead():
    # aggregate=True because DELTA
    c = RequestOutputCollector(RequestOutputKind.DELTA, "r0")
    assert c.aggregate is True
    c.put(_ro("r0", 0, "ab", [1, 2]))
    c.put(_ro("r0", 0, "cd", [3, 4]))  # consumer hasn't drained yet -> merge
    out = c.get_nowait()
    assert out.outputs[0].text == "abcd"
    assert out.outputs[0].token_ids == [1, 2, 3, 4]


def test_cumulative_replaces_same_index():
    c = RequestOutputCollector(RequestOutputKind.CUMULATIVE, "r0")
    assert c.aggregate is False
    c.put(_ro("r0", 0, "ab", [1, 2]))
    c.put(_ro("r0", 0, "abcd", [1, 2, 3, 4]))  # full replace, not concat
    out = c.get_nowait()
    assert out.outputs[0].text == "abcd"
    assert out.outputs[0].token_ids == [1, 2, 3, 4]


def test_different_index_do_not_override():
    c = RequestOutputCollector(RequestOutputKind.DELTA, "r0")
    c.put(_ro("r0", 0, "a", [1]))
    c.put(_ro("r0", 1, "x", [9]))  # different index -> appended, not merged
    out = c.get_nowait()
    by_index = {o.index: o.text for o in out.outputs}
    assert by_index == {0: "a", 1: "x"}


def test_exception_preempts_and_reraises():
    c = RequestOutputCollector(RequestOutputKind.DELTA, "r0")
    c.put(_ro("r0", 0, "a", [1]))
    err = RuntimeError("engine dead")
    c.put(err)  # exception overrides existing output
    with pytest.raises(RuntimeError, match="engine dead"):
        c.get_nowait()


def test_get_blocks_until_put():
    async def scenario():
        c = RequestOutputCollector(RequestOutputKind.DELTA, "r0")

        async def producer():
            await asyncio.sleep(0.01)
            c.put(_ro("r0", 0, "hi", [1, 2]))

        async def consumer():
            return await c.get()

        _, out = await asyncio.gather(producer(), consumer())
        return out

    out = asyncio.run(scenario())
    assert out.outputs[0].text == "hi"
