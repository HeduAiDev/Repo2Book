"""m03/m04/m05/m06: bitmask buffer budget, parallel-fill structural gate,
serial fill + speculative rollback, and the "-1 = fully allowed" residual-bit
cleanup on cross-step buffer reuse.

SOURCE anchors: vllm/v1/structured_output/__init__.py:L60-68 (parallel-fill
threshold), L203-234 (shape budget), L236-299 (parallel/serial branches),
L185-201 (_fill_bitmasks / full-mask reset)
"""
import math
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "implementation")
)

from conftest import FakeBackend, FakeGrammar, make_request

from structured_output_manager import StructuredOutputManager  # noqa: E402


def test_parallel_fill_is_structurally_dead_at_or_below_threshold():
    # __init__.py:62 -- `if self.fill_bitmask_parallel_threshold < max_num_seqs`
    # only constructs executor_for_fillmask when max_num_seqs > 128. At or
    # below 128, the parallel branch in grammar_bitmask() can never fire
    # because len(structured_output_request_ids) <= max_num_seqs <= 128.
    manager = StructuredOutputManager(max_num_seqs=128)
    assert not hasattr(manager, "executor_for_fillmask")

    manager_big = StructuredOutputManager(max_num_seqs=256)
    assert hasattr(manager_big, "executor_for_fillmask")


def test_bitmask_shape_budget_rows_times_columns():
    vocab_size = 65
    max_num_seqs = 4
    num_spec_tokens = 2
    manager = StructuredOutputManager(
        max_num_seqs=max_num_seqs, max_num_spec_tokens=num_spec_tokens
    )
    manager.backend = FakeBackend(vocab_size=vocab_size)

    grammar = FakeGrammar(allowed_at=[{1, 2}])
    req = make_request("r0", grammar)
    bitmask = manager.grammar_bitmask(
        requests={"r0": req},
        structured_output_request_ids=["r0"],
        scheduled_spec_decode_tokens={},
    )
    assert bitmask is not None
    # Allocation happens once, sized for the worst case (every seq uses all
    # spec positions), even though only 1 row is actually returned this call.
    assert manager._grammar_bitmask.shape == (
        max_num_seqs * (1 + num_spec_tokens),
        math.ceil(vocab_size / 32),
    )
    assert bitmask.shape == (1, math.ceil(vocab_size / 32))


def test_serial_fill_advances_then_rolls_back_speculative_positions():
    # A 3-step grammar: token 5 legal at position 0, token 6 at position 1,
    # token 7 at position 2. Draft proposes [5, 6] (both legal).
    manager = StructuredOutputManager(max_num_seqs=4, max_num_spec_tokens=2)
    manager.backend = FakeBackend(vocab_size=64)

    grammar = FakeGrammar(allowed_at=[{5}, {6}, {7}])
    req = make_request("r0", grammar)

    bitmask = manager.grammar_bitmask(
        requests={"r0": req},
        structured_output_request_ids=["r0"],
        scheduled_spec_decode_tokens={"r0": [5, 6]},
    )
    # 2 draft tokens + 1 bonus/padding row = 3 rows for this one request.
    assert bitmask.shape[0] == 3
    # accept_tokens was called for both draft tokens (advancing state twice),
    # then rolled back by exactly that many steps -- net position is 0.
    assert grammar.accept_log == [("r0", 5), ("r0", 6)]
    assert grammar.rollback_log == [2]
    assert grammar.position == 0
    # fill_bitmask was called once per row (3 rows), not once per accepted
    # token.
    assert grammar.fill_log == [0, 1, 2]


def test_serial_fill_stops_advancing_after_padding_sentinel():
    # itertools.chain(req_tokens, (-1,)) always appends a trailing -1
    # sentinel row. Once that sentinel is reached, apply_bitmask is flipped
    # False *before* the accept_tokens branch runs for that row, so
    # accept_tokens/rollback are never invoked at all for a request with no
    # real draft tokens this step.
    manager = StructuredOutputManager(max_num_seqs=4, max_num_spec_tokens=1)
    manager.backend = FakeBackend(vocab_size=64)

    grammar = FakeGrammar(allowed_at=[{5}, {6}])
    req = make_request("r0", grammar)

    bitmask = manager.grammar_bitmask(
        requests={"r0": req},
        structured_output_request_ids=["r0"],
        scheduled_spec_decode_tokens={"r0": [-1]},
    )
    # -1 is the immediate padding sentinel: apply_bitmask flips False right
    # away, so accept_tokens/rollback are never invoked at all.
    assert bitmask.shape[0] == 2
    assert grammar.accept_log == []
    assert grammar.rollback_log == []


def test_unconstrained_row_filled_with_full_mask_not_left_stale():
    # Reuse across two calls: seed a residual bit pattern in row 0, then make
    # a call where the grammar is terminated (should_fill_bitmask False path
    # is not directly reachable here without a reasoner; use is_terminated
    # instead, which _fill_bitmasks also treats as "don't constrain").
    manager = StructuredOutputManager(max_num_seqs=2, max_num_spec_tokens=0)
    manager.backend = FakeBackend(vocab_size=64)

    grammar = FakeGrammar(allowed_at=[{5}])
    grammar.terminated = True  # is_terminated() -> True
    req = make_request("r0", grammar)

    # Seed garbage into the buffer before the call to prove _fill_bitmasks
    # overwrites it rather than leaving stale bits from a previous step.
    manager._grammar_bitmask = manager.backend.allocate_token_bitmask(2)
    manager._grammar_bitmask[0].fill_(0)  # 0 = "everything masked out"

    bitmask = manager.grammar_bitmask(
        requests={"r0": req},
        structured_output_request_ids=["r0"],
        scheduled_spec_decode_tokens={},
    )
    # _full_mask is -1 (all bits set = all-ones = fully allowed).
    assert (bitmask[0] == -1).all()
