"""Shared fixtures/fakes for ch32 tests.

FakeGrammar implements the StructuredOutputGrammar six-method contract
(ch31 built the real xgrammar/guidance implementations; here we only need a
transparent, hand-inspectable stand-in whose accept/rollback/validate
bookkeeping we can assert on directly) so this chapter's tests exercise the
*real* vLLM control flow in structured_output_manager.py / scheduler.py /
worker code, with only the grammar's internal FSM faked out.
"""
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "implementation")
)

import numpy as np
import pytest
import torch

from backend_types import StructuredOutputGrammar  # noqa: E402
from output import SchedulerOutput  # noqa: E402
from request import Request  # noqa: E402
from sampling_params import SamplingParams, StructuredOutputsParams  # noqa: E402
from so_request import StructuredOutputRequest  # noqa: E402


class FakeGrammar(StructuredOutputGrammar):
    """Deterministic grammar: accepts tokens in `allowed`, tracks a simple
    integer "position" that accept_tokens/rollback move forward/back, and
    fill_bitmask writes 0 (allowed) at column `token` and -1 (mask out)
    elsewhere -- enough to make fill_bitmask/rollback/accept_tokens
    observable without needing a real FSM.
    """

    def __init__(self, allowed_at):
        # allowed_at: list of sets, one per "position" -- allowed[i] is the
        # set of tokens legal when `position == i`.
        self.allowed_at = allowed_at
        self.position = 0
        self.terminated = False
        self.accept_log: list[tuple[str, int]] = []
        self.rollback_log: list[int] = []
        self.fill_log: list[int] = []

    def accept_tokens(self, request_id, tokens) -> bool:
        for tok in tokens:
            allowed = self.allowed_at[min(self.position, len(self.allowed_at) - 1)]
            if tok not in allowed:
                return False
            self.accept_log.append((request_id, tok))
            self.position += 1
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        out = []
        pos = self.position
        for tok in tokens:
            allowed = self.allowed_at[min(pos, len(self.allowed_at) - 1)]
            if tok not in allowed:
                break
            out.append(tok)
            pos += 1
        return out

    def rollback(self, num_tokens: int) -> None:
        self.rollback_log.append(num_tokens)
        self.position -= num_tokens
        assert self.position >= 0

    def fill_bitmask(self, bitmask: torch.Tensor, batch_index: int) -> None:
        self.fill_log.append(batch_index)
        allowed = self.allowed_at[min(self.position, len(self.allowed_at) - 1)]
        # Column j holds bits for tokens [32*j, 32*j+31]; bit==1 means
        # "legal", bit==0 means "illegal" -- the real xgrammar/kernel
        # convention (matches _full_mask == -1, all-bits-set, meaning
        # "fully allowed" for an unconstrained row).
        bitmask[batch_index].fill_(0)
        for tok in allowed:
            col = tok // 32
            bit = tok % 32
            bitmask[batch_index, col] |= np.int32(1 << bit).item()

    def is_terminated(self) -> bool:
        return self.terminated

    def reset(self):
        self.position = 0
        self.terminated = False


class FakeBackend:
    """Stand-in for StructuredOutputManager.backend -- only
    allocate_token_bitmask is used by grammar_bitmask()."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def allocate_token_bitmask(self, max_num_seqs: int) -> torch.Tensor:
        cols = -(-self.vocab_size // 32)
        return torch.full((max_num_seqs, cols), -1, dtype=torch.int32)


def make_request(req_id: str, grammar: FakeGrammar | None = None) -> Request:
    params = StructuredOutputsParams(regex="x")
    sp = SamplingParams(max_tokens=16, structured_outputs=params)
    req = Request(req_id, prompt_token_ids=[1, 2, 3], sampling_params=sp)
    if grammar is not None:
        req.structured_output_request.grammar = grammar
    return req


def make_scheduler_output(**kwargs) -> SchedulerOutput:
    return SchedulerOutput(**kwargs)


@pytest.fixture
def fake_grammar_factory():
    return FakeGrammar
