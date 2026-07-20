"""Rollback upper bound: max_rollback_tokens is pinned to num_speculative_tokens
when the xgrammar matcher is constructed -- this is the *reason* the serial
fill's rollback(state_advancements) in m05 is always safe (state_advancements
can be at most num_speculative_tokens). Also covers allocate_token_bitmask,
the entry point StructuredOutputManager.grammar_bitmask uses on its first
call to size the shared buffer.

ch31 already covers the full backend-selection/compile dispatch; this test
only exercises the two symbols this chapter's payoff depends on.

SOURCE anchors: vllm/v1/structured_output/backend_xgrammar.py:L115-125
"""
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "implementation")
)

import backend_xgrammar  # noqa: E402
from backend_xgrammar import XgrammarBackend  # noqa: E402


class FakeMatcher:
    def __init__(self, ctx, max_rollback_tokens):
        self.ctx = ctx
        self.max_rollback_tokens = max_rollback_tokens


class FakeXgr:
    GrammarMatcher = FakeMatcher

    @staticmethod
    def allocate_token_bitmask(max_num_seqs, vocab_size):
        return ("bitmask", max_num_seqs, vocab_size)


def test_max_rollback_tokens_pinned_to_num_speculative_tokens(monkeypatch):
    monkeypatch.setattr(backend_xgrammar, "xgr", FakeXgr)
    backend = XgrammarBackend(vocab_size=1000, num_speculative_tokens=4)

    grammar = backend.compile_grammar_tail(ctx="some-compiled-ctx")

    assert grammar.matcher.max_rollback_tokens == 4
    assert grammar.matcher.ctx == "some-compiled-ctx"


def test_allocate_token_bitmask_sizes_by_batch_and_vocab(monkeypatch):
    monkeypatch.setattr(backend_xgrammar, "xgr", FakeXgr)
    backend = XgrammarBackend(vocab_size=1000, num_speculative_tokens=0)

    result = backend.allocate_token_bitmask(256)
    assert result == ("bitmask", 256, 1000)
