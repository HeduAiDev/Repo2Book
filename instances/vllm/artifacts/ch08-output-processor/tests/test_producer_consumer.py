"""f3: output_handler (single producer) <-> generate() (per-request consumer).

Asserts vLLM's observable behavior:
 - output_handler pulls a batch from EngineCore, chunks it by chunk_size, runs
   process_outputs on each chunk and pushes RequestOutputs to per-request
   queues (returning nothing itself);
 - stop-string finished reqs are reverse-aborted on EngineCore;
 - generate() drains its own queue with get_nowait()/get() and stops on
   out.finished;
 - one producer fans out to N independent consumer queues.
"""

import asyncio

from conftest import FakeRequest, FakeSamplingParams, char_tokenizer, ids

from implementation._types import EngineCoreOutput, FinishReason, RequestOutputKind
from implementation.output_processor import (
    OutputProcessor,
    RequestOutputCollector,
    generate,
    output_handler,
)
from implementation.outputs import STREAM_FINISHED


class FakeEngineCore:
    """Yields a scripted sequence of output batches, then blocks forever."""

    def __init__(self, batches):
        self.batches = list(batches)
        self.aborted = []
        self._i = 0

    async def get_output_async(self):
        if self._i < len(self.batches):
            b = self.batches[self._i]
            self._i += 1
            return b
        # No more output: block so the handler loop parks.
        await asyncio.Event().wait()

    async def abort_requests_async(self, req_ids):
        self.aborted.extend(req_ids)


class FakeOutputs:
    def __init__(self, outputs):
        self.outputs = outputs
        self.timestamp = 0.0


def _register(op, req_ids):
    queues = {}
    for rid in req_ids:
        sp = FakeSamplingParams(output_kind=RequestOutputKind.DELTA)
        req = FakeRequest(request_id=rid, external_req_id=rid, sampling_params=sp)
        q = RequestOutputCollector(RequestOutputKind.DELTA, rid)
        op.add_request(req, prompt=None, queue=q)
        queues[rid] = q
    return queues


def test_fanout_one_producer_many_consumers():
    async def scenario():
        op = OutputProcessor(char_tokenizer, stream_interval=1)
        queues = _register(op, ["r0", "r1"])

        batch = FakeOutputs([
            EngineCoreOutput(request_id="r0", new_token_ids=ids("A"),
                             finish_reason=FinishReason.LENGTH, finished=True),
            EngineCoreOutput(request_id="r1", new_token_ids=ids("B"),
                             finish_reason=FinishReason.LENGTH, finished=True),
        ])
        engine = FakeEngineCore([batch])

        handler = asyncio.create_task(output_handler(engine, op, chunk_size=8))
        res0, res1 = await asyncio.gather(
            generate(queues["r0"], STREAM_FINISHED),
            generate(queues["r1"], STREAM_FINISHED),
        )
        handler.cancel()
        return res0, res1

    res0, res1 = asyncio.run(scenario())
    assert [o.outputs[0].text for o in res0] == ["A"]
    assert [o.outputs[0].text for o in res1] == ["B"]


def test_chunking_splits_large_batch():
    async def scenario():
        op = OutputProcessor(char_tokenizer, stream_interval=1)
        queues = _register(op, ["r0"])
        queue = queues["r0"]
        # 3 outputs for the same req across one batch, chunk_size=1 forces 3 chunks
        batch = FakeOutputs([
            EngineCoreOutput(request_id="r0", new_token_ids=ids("a")),
            EngineCoreOutput(request_id="r0", new_token_ids=ids("b")),
            EngineCoreOutput(request_id="r0", new_token_ids=ids("c"),
                             finish_reason=FinishReason.LENGTH, finished=True),
        ])
        engine = FakeEngineCore([batch])
        handler = asyncio.create_task(output_handler(engine, op, chunk_size=1))
        res = await generate(queue, STREAM_FINISHED)
        handler.cancel()
        return res

    res = asyncio.run(scenario())
    # DELTA deltas may be merged in the mailbox if producer outran consumer;
    # the concatenation of all received text must be the full sequence.
    assert "".join(o.outputs[0].text for o in res) == "abc"


def test_reverse_abort_on_stop_string():
    async def scenario():
        op = OutputProcessor(char_tokenizer, stream_interval=1)
        sp = FakeSamplingParams(output_kind=RequestOutputKind.DELTA, stop=["STOP"])
        req = FakeRequest(request_id="r0", external_req_id="r0", sampling_params=sp)
        q = RequestOutputCollector(RequestOutputKind.DELTA, "r0")
        op.add_request(req, prompt=None, queue=q)

        batch = FakeOutputs([
            EngineCoreOutput(request_id="r0", new_token_ids=ids("abSTOP"), finished=False),
        ])
        engine = FakeEngineCore([batch])
        handler = asyncio.create_task(output_handler(engine, op, chunk_size=8))
        res = await generate(q, STREAM_FINISHED)
        handler.cancel()
        return engine.aborted, res

    aborted, res = asyncio.run(scenario())
    assert aborted == ["r0"]  # EngineCore got the reverse abort
    assert res[-1].finished is True
