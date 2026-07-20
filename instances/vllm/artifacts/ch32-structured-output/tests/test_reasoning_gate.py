"""m16: reasoning-model coupling -- should_fill_bitmask / should_advance keep
the grammar unconstrained during a reasoning segment, gated by reasoner
presence, enable_in_reasoning, and reasoning_ended.

SOURCE anchors: vllm/v1/structured_output/__init__.py:L99-112 (_get_reasoner),
                L301-319 (should_fill_bitmask), L321-357 (should_advance)
"""
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "implementation")
)

from conftest import FakeGrammar, make_request

from structured_output_manager import StructuredOutputManager  # noqa: E402


class FakeReasoner:
    def __init__(self, end_at: int):
        self.end_at = end_at

    def is_reasoning_end(self, prompt_token_ids) -> bool:
        return False

    def is_reasoning_end_streaming(self, all_token_ids, new_slice) -> bool:
        return len(list(all_token_ids)) >= self.end_at


def _manager_with_reasoner(end_at=5):
    manager = StructuredOutputManager(max_num_seqs=4)
    manager.reasoner_cls = lambda **kwargs: FakeReasoner(end_at=end_at)
    return manager


def test_no_reasoner_always_fills_and_advances():
    manager = StructuredOutputManager(max_num_seqs=4)  # reasoner_cls stays None
    req = make_request("r0", FakeGrammar(allowed_at=[{1}]))
    assert manager.should_fill_bitmask(req) is True
    assert manager.should_advance(req) is True


def test_reasoner_present_blocks_bitmask_until_reasoning_ends():
    manager = _manager_with_reasoner(end_at=100)
    req = make_request("r0", FakeGrammar(allowed_at=[{1}]))
    req.prompt_token_ids = [0, 1, 2]  # short prompt -> is_reasoning_end() False
    assert manager.should_fill_bitmask(req) is False
    # reasoning_ended got cached to False on the request so we don't
    # re-derive it every call.
    assert req.structured_output_request.reasoning_ended is False


def test_enable_in_reasoning_overrides_reasoner_gate():
    manager = _manager_with_reasoner(end_at=100)
    manager.enable_in_reasoning = True
    req = make_request("r0", FakeGrammar(allowed_at=[{1}]))
    assert manager.should_fill_bitmask(req) is True
    assert manager.should_advance(req) is True


def test_should_advance_flips_true_once_reasoning_ends_this_step():
    manager = _manager_with_reasoner(end_at=3)
    req = make_request("r0", FakeGrammar(allowed_at=[{1}]))
    req.structured_output_request.reasoning_ended = False
    req._all_token_ids = [9, 9, 9]  # len == end_at -> reasoning ends now
    req.num_computed_tokens = 3
    req.num_output_placeholders = 0

    # The step where reasoning ends: we must NOT advance yet (next pass will).
    assert manager.should_advance(req) is False
    assert req.structured_output_request.reasoning_ended is True

    # Next call: reasoning_ended is now True -> advance freely.
    assert manager.should_advance(req) is True


def test_should_advance_false_for_non_structured_request():
    manager = StructuredOutputManager(max_num_seqs=4)
    req = make_request("r0", grammar=None)
    req.structured_output_request = None
    assert manager.should_advance(req) is False
