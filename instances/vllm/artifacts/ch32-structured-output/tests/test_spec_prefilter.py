"""m12: speculative coupling before assembly -- update_draft_token_ids_in_output
filters drafts through validate_tokens, pads the rejected tail with -1, and
records num_invalid_spec_tokens; make_spec_decoding_stats then consumes that
dict to correct the acceptance-rate denominator.

SOURCE anchors: vllm/v1/core/sched/scheduler.py:L1623-1657
                (update_draft_token_ids_in_output),
                vllm/v1/core/sched/scheduler.py:L1901-1917
                (make_spec_decoding_stats)
"""
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "implementation")
)

from conftest import FakeBackend, FakeGrammar, make_request, make_scheduler_output

from output import DraftTokenIds  # noqa: E402
from scheduler import Scheduler, SpecDecodingStats  # noqa: E402
from structured_output_manager import StructuredOutputManager  # noqa: E402


def _make_scheduler():
    manager = StructuredOutputManager(max_num_seqs=4, max_num_spec_tokens=2)
    manager.backend = FakeBackend(vocab_size=64)
    return Scheduler(structured_output_manager=manager, num_spec_tokens=2)


def test_invalid_drafts_padded_with_minus_one_and_counted():
    sched = _make_scheduler()
    # Only token 5 is legal at position 0; the draft proposes [5, 99] so the
    # second one must be rejected.
    grammar = FakeGrammar(allowed_at=[{5}])
    req = make_request("r0", grammar)
    sched.requests["r0"] = req

    so = make_scheduler_output(scheduled_spec_decode_tokens={"r0": [-1, -1]})
    draft = DraftTokenIds(req_ids=["r0"], draft_token_ids=[[5, 99]])

    sched.update_draft_token_ids_in_output(draft, so)

    # 1 of 2 drafts survived validate_tokens; the rest is padded to the
    # *original* scheduled length (2), and the padding count is recorded.
    assert so.scheduled_spec_decode_tokens["r0"] == [5, -1]
    assert so.num_invalid_spec_tokens == {"r0": 1}


def test_fully_valid_drafts_produce_no_invalid_count():
    sched = _make_scheduler()
    grammar = FakeGrammar(allowed_at=[{5}, {6}])
    req = make_request("r0", grammar)
    sched.requests["r0"] = req

    so = make_scheduler_output(scheduled_spec_decode_tokens={"r0": [-1, -1]})
    draft = DraftTokenIds(req_ids=["r0"], draft_token_ids=[[5, 6]])

    sched.update_draft_token_ids_in_output(draft, so)

    assert so.scheduled_spec_decode_tokens["r0"] == [5, 6]
    assert so.num_invalid_spec_tokens == {}


def test_validate_tokens_not_called_when_should_advance_is_false():
    # should_advance()==False for a request whose structured_output_request
    # is None (use_structured_output is False) -- drafts pass through
    # unfiltered instead of being validated against a nonexistent grammar.
    sched = _make_scheduler()
    req = make_request("r0", grammar=None)
    req.structured_output_request = None
    sched.requests["r0"] = req

    so = make_scheduler_output(scheduled_spec_decode_tokens={"r0": [-1]})
    draft = DraftTokenIds(req_ids=["r0"], draft_token_ids=[[123]])

    sched.update_draft_token_ids_in_output(draft, so)
    assert so.scheduled_spec_decode_tokens["r0"] == [123]
    assert so.num_invalid_spec_tokens == {}


def test_make_spec_decoding_stats_subtracts_invalid_from_draft_count():
    sched = _make_scheduler()
    stats = sched.make_spec_decoding_stats(
        spec_decoding_stats=None,
        num_draft_tokens=4,
        num_accepted_tokens=2,
        num_invalid_spec_tokens={"r0": 1},
        request_id="r0",
    )
    assert isinstance(stats, SpecDecodingStats)
    # 4 draft tokens scheduled, but 1 was already filtered out by the
    # grammar before it ever reached the acceptance check -- the denominator
    # used for the acceptance-rate stat must not double-penalize it.
    assert stats.num_draft_tokens == 3
    assert stats.num_accepted_tokens == 2


def test_make_spec_decoding_stats_no_op_when_no_draft_tokens():
    sched = _make_scheduler()
    stats = sched.make_spec_decoding_stats(
        spec_decoding_stats=None,
        num_draft_tokens=0,
        num_accepted_tokens=0,
        num_invalid_spec_tokens=None,
        request_id="r0",
    )
    assert stats is None
