"""m07 + m15/9b: mask-vs-forward overlap and the async-scheduling deferred
sampling chain.

step(): get_grammar_bitmask is called right after execute_model(non_block=True)
launches the forward pass, so CPU mask-filling overlaps the GPU forward.

step_with_batch_queue(): when scheduler_output.pending_structured_output_tokens
is set (async scheduling + spec decode + structured output all engaged), the
bitmask computation must be deferred until *next* call, after
take_draft_token_ids() -> update_draft_token_ids_in_output() has run --
matching the source comment: "we need to get the draft token ids from the
prior step before we can compute the grammar bitmask for the deferred
request."

SOURCE anchors: vllm/v1/engine/core.py:L406-433 (step),
                vllm/v1/engine/core.py:L447-561 (step_with_batch_queue)
"""
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "implementation")
)

from engine_core import EngineCore  # noqa: E402
from output import DraftTokenIds, SchedulerOutput  # noqa: E402


class FakeFuture:
    def __init__(self, value, is_done=False):
        self._value = value
        self._is_done = is_done

    def result(self):
        return self._value

    def done(self):
        return self._is_done


class CallOrderRecorder:
    """Fake scheduler + fake model_executor that both append their name to a
    shared call-order list, so tests can assert on *when* get_grammar_bitmask
    fires relative to execute_model."""

    def __init__(self):
        self.calls: list[str] = []
        self.has_requests_value = True
        self.scheduler_output_factory = lambda: SchedulerOutput(
            num_scheduled_tokens={"r0": 1}, total_num_scheduled_tokens=1
        )
        self.grammar_output_value = "GRAMMAR_OUTPUT"
        self.model_output_after_result = "MODEL_OUTPUT"
        # If set, execute_model's future.result() returns None first (so
        # step() falls through to sample_tokens for the result).
        self.execute_model_returns_none = False

    # -- scheduler-shaped surface --
    def has_requests(self):
        return self.has_requests_value

    def schedule(self):
        self.calls.append("schedule")
        return self.scheduler_output_factory()

    def get_grammar_bitmask(self, scheduler_output):
        self.calls.append("get_grammar_bitmask")
        return self.grammar_output_value

    def update_from_output(self, scheduler_output, model_output):
        self.calls.append("update_from_output")
        return {"outputs": model_output}

    # -- model_executor-shaped surface --
    def execute_model(self, scheduler_output, non_block=True):
        self.calls.append("execute_model")
        return FakeFuture(None if self.execute_model_returns_none else "EXEC_RESULT")

    def sample_tokens(self, grammar_output, non_block=False):
        self.calls.append(f"sample_tokens({grammar_output})")
        if non_block:
            return FakeFuture(self.model_output_after_result)
        # step() calls this without non_block=True -- blocking call, returns
        # the ModelRunnerOutput directly (no future to unwrap).
        return self.model_output_after_result


def test_step_computes_bitmask_after_launching_forward_non_blocking():
    rec = CallOrderRecorder()
    rec.execute_model_returns_none = True  # forces step() to call sample_tokens
    core = EngineCore(scheduler=rec, model_executor=rec)

    outputs, model_executed = core.step()

    # execute_model must be launched *before* get_grammar_bitmask -- that is
    # the entire point of m07 (CPU mask work hides behind the GPU forward).
    assert rec.calls.index("execute_model") < rec.calls.index("get_grammar_bitmask")
    assert rec.calls == [
        "schedule",
        "execute_model",
        "get_grammar_bitmask",
        f"sample_tokens({rec.grammar_output_value})",
        "update_from_output",
    ]
    assert model_executed is True
    assert outputs == {"outputs": rec.model_output_after_result}


def test_step_returns_early_when_no_requests():
    rec = CallOrderRecorder()
    rec.has_requests_value = False
    core = EngineCore(scheduler=rec, model_executor=rec)
    assert core.step() == ({}, False)
    assert rec.calls == []


class DeferredChainRecorder(CallOrderRecorder):
    """Adds take_draft_token_ids / update_draft_token_ids_in_output to the
    executor+scheduler surface, and lets the test flip
    pending_structured_output_tokens on the *next* scheduler_output produced."""

    def __init__(self):
        super().__init__()
        self.pending_next = False
        self.draft_token_ids_value = DraftTokenIds(req_ids=["r0"], draft_token_ids=[[5]])

    def schedule(self):
        self.calls.append("schedule")
        so = SchedulerOutput(
            num_scheduled_tokens={"r0": 1},
            total_num_scheduled_tokens=1,
            pending_structured_output_tokens=self.pending_next,
        )
        return so

    def take_draft_token_ids(self):
        self.calls.append("take_draft_token_ids")
        return self.draft_token_ids_value

    def update_draft_token_ids_in_output(self, draft_token_ids, scheduler_output):
        self.calls.append("update_draft_token_ids_in_output")
        scheduler_output.num_invalid_spec_tokens = {}


def test_deferred_chain_waits_for_draft_tokens_before_computing_bitmask():
    rec = DeferredChainRecorder()
    core = EngineCore(
        scheduler=rec, model_executor=rec, use_spec_decode=True, batch_queue_size=2
    )

    # Round 1: nothing pending -- schedules normally, bitmask computed
    # immediately, queued.
    out1, executed1 = core.step_with_batch_queue()
    assert executed1 is True
    assert "get_grammar_bitmask" in rec.calls
    assert "take_draft_token_ids" not in rec.calls

    # Round 2: this round's scheduler_output has pending_structured_output_
    # tokens set -- the bitmask for it must NOT be computed until the
    # deferred branch runs later in *this same call*, after
    # take_draft_token_ids/update_draft_token_ids_in_output.
    rec.calls.clear()
    rec.pending_next = True
    out2, executed2 = core.step_with_batch_queue()

    assert "take_draft_token_ids" in rec.calls
    order = rec.calls
    assert order.index("take_draft_token_ids") < order.index(
        "update_draft_token_ids_in_output"
    )
    assert order.index("update_draft_token_ids_in_output") < order.index(
        "get_grammar_bitmask"
    )
    # And get_grammar_bitmask must come strictly after take_draft_token_ids --
    # this *is* the deferred-chain invariant from the source comment.
    assert order.index("take_draft_token_ids") < order.index("get_grammar_bitmask")


def test_deferred_chain_without_spec_decode_skips_draft_token_fetch():
    rec = DeferredChainRecorder()
    core = EngineCore(
        scheduler=rec, model_executor=rec, use_spec_decode=False, batch_queue_size=2
    )
    core.step_with_batch_queue()
    rec.calls.clear()
    rec.pending_next = True
    core.step_with_batch_queue()
    assert "take_draft_token_ids" not in rec.calls
    assert "get_grammar_bitmask" in rec.calls
