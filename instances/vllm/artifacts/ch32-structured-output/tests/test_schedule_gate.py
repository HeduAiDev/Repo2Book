"""m01: scheduler-side gate -- has_structured_output_requests is only set for
requests that (a) use structured output and (b) are NOT mid-prefill-chunk;
get_grammar_bitmask early-returns None whenever that gate is off, and its
row order follows scheduler_output.num_scheduled_tokens iteration order.

SOURCE anchors: vllm/v1/core/sched/scheduler.py:L932-951 (_update_after_schedule),
                vllm/v1/core/sched/scheduler.py:L1224-1246 (get_grammar_bitmask)
"""
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "implementation")
)

from conftest import FakeBackend, FakeGrammar, make_request, make_scheduler_output

from output import SchedulerOutput  # noqa: E402
from scheduler import Scheduler  # noqa: E402
from structured_output_manager import StructuredOutputManager  # noqa: E402


def _make_scheduler(max_num_seqs=8):
    manager = StructuredOutputManager(max_num_seqs=max_num_seqs)
    manager.backend = FakeBackend(vocab_size=64)
    return Scheduler(structured_output_manager=manager)


def test_prefill_chunk_does_not_set_gate():
    sched = _make_scheduler()
    grammar = FakeGrammar(allowed_at=[{1, 2}])
    req = make_request("r0", grammar)
    sched.requests["r0"] = req
    req.num_prompt_tokens = 100  # bigger than what we schedule -> prefill chunk

    so = make_scheduler_output(num_scheduled_tokens={"r0": 10})
    # num_tokens tracks len(_all_token_ids); make it big so the request is
    # still mid-prefill after this step's tokens are counted.
    req._all_token_ids = [0] * 100
    sched._update_after_schedule(so)

    assert req.is_prefill_chunk is True
    assert so.has_structured_output_requests is False


def test_gate_set_when_not_prefill_chunk():
    sched = _make_scheduler()
    grammar = FakeGrammar(allowed_at=[{1, 2}])
    req = make_request("r0", grammar)
    sched.requests["r0"] = req
    req._all_token_ids = [0, 1, 2]  # num_tokens == 3

    so = make_scheduler_output(num_scheduled_tokens={"r0": 3})
    sched._update_after_schedule(so)

    assert req.is_prefill_chunk is False
    assert so.has_structured_output_requests is True


def test_get_grammar_bitmask_none_when_gate_off():
    sched = _make_scheduler()
    so = make_scheduler_output(
        num_scheduled_tokens={"r0": 1}, has_structured_output_requests=False
    )
    assert sched.get_grammar_bitmask(so) is None


def test_get_grammar_bitmask_row_order_follows_schedule_iteration_order():
    sched = _make_scheduler()
    for i, rid in enumerate(["b", "a", "c"]):
        grammar = FakeGrammar(allowed_at=[{1, 2}])
        req = make_request(rid, grammar)
        req._all_token_ids = [0, 1, 2]
        sched.requests[rid] = req

    # num_scheduled_tokens is an ordinary dict -- Python 3.7+ preserves
    # insertion order, which is exactly the "schedule order" the real
    # scheduler relies on.
    so = make_scheduler_output(
        num_scheduled_tokens={"b": 3, "a": 3, "c": 3},
        has_structured_output_requests=True,
    )
    grammar_output = sched.get_grammar_bitmask(so)
    assert grammar_output is not None
    assert grammar_output.structured_output_request_ids == ["b", "a", "c"]
    assert grammar_output.grammar_bitmask.shape[0] == 3


def test_prefill_chunk_request_excluded_from_bitmask_rows():
    sched = _make_scheduler()
    g_ready = FakeGrammar(allowed_at=[{1, 2}])
    req_ready = make_request("ready", g_ready)
    req_ready._all_token_ids = [0, 1, 2]
    req_ready.is_prefill_chunk = False
    sched.requests["ready"] = req_ready

    g_chunk = FakeGrammar(allowed_at=[{1, 2}])
    req_chunk = make_request("chunk", g_chunk)
    req_chunk.is_prefill_chunk = True
    sched.requests["chunk"] = req_chunk

    so = make_scheduler_output(
        num_scheduled_tokens={"ready": 3, "chunk": 5},
        has_structured_output_requests=True,
    )
    grammar_output = sched.get_grammar_bitmask(so)
    assert grammar_output.structured_output_request_ids == ["ready"]
