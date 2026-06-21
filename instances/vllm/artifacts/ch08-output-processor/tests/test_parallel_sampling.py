"""ParentRequest.get_outputs — n>1 child -> parent aggregation.

Asserts vLLM's observable behavior:
 - streaming (non FINAL_ONLY): each child output forwarded immediately;
   child_requests shrinks as children finish; finished only when all done;
   an already-finished-and-returned child yields [] (no double send);
 - FINAL_ONLY: outputs withheld ([]) until all n children finished, then the
   whole index-ordered aggregate returns at once.
"""

from conftest import FakeSamplingParams

from implementation._types import CompletionOutput, RequestOutputKind
from implementation.parallel_sampling import ParentRequest


def _co(index, finished):
    return CompletionOutput(
        index=index,
        text="t",
        token_ids=[1],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="stop" if finished else None,
    )


def test_streaming_forwards_each_child():
    sp = FakeSamplingParams(output_kind=RequestOutputKind.DELTA, n=3)
    parent = ParentRequest("pid", "ext", sp)
    parent.child_requests = {"0_pid", "1_pid", "2_pid"}

    out, fin = parent.get_outputs("0_pid", _co(0, finished=False))
    assert [o.index for o in out] == [0]
    assert fin is False  # children still pending

    # child 1 finishes -> forwarded, removed from set
    out, fin = parent.get_outputs("1_pid", _co(1, finished=True))
    assert [o.index for o in out] == [1]
    assert "1_pid" not in parent.child_requests
    assert fin is False


def test_streaming_finished_when_all_children_done():
    sp = FakeSamplingParams(output_kind=RequestOutputKind.DELTA, n=2)
    parent = ParentRequest("pid", "ext", sp)
    parent.child_requests = {"0_pid", "1_pid"}
    parent.get_outputs("0_pid", _co(0, finished=True))
    out, fin = parent.get_outputs("1_pid", _co(1, finished=True))
    assert fin is True
    assert not parent.child_requests


def test_streaming_already_returned_child_not_resent():
    sp = FakeSamplingParams(output_kind=RequestOutputKind.DELTA, n=1)
    parent = ParentRequest("pid", "ext", sp)
    parent.child_requests = set()  # already finished & removed previously
    out, fin = parent.get_outputs("0_pid", _co(0, finished=True))
    assert out == []  # not sent again
    assert fin is True


def test_final_only_withholds_until_all_done():
    sp = FakeSamplingParams(output_kind=RequestOutputKind.FINAL_ONLY, n=3)
    parent = ParentRequest("pid", "ext", sp)
    parent.child_requests = {"0_pid", "1_pid", "2_pid"}
    assert len(parent.output_aggregator) == 3

    out, fin = parent.get_outputs("1_pid", _co(1, finished=True))
    assert out == []  # nothing yet
    assert fin is False
    out, fin = parent.get_outputs("0_pid", _co(0, finished=True))
    assert out == []
    out, fin = parent.get_outputs("2_pid", _co(2, finished=True))
    # all done -> whole aggregate returned, index-ordered
    assert [o.index for o in out] == [0, 1, 2]
    assert fin is True
