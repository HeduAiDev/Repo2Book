"""Shared lightweight stand-ins for the Stage 3 companion tests.

These mirror the *shape* of the real EngineCoreRequest / SamplingParams that
the output processor consumes — only the fields Stage 3 reads. No ``import
vllm``; pure unit tests of the subtract-only companion's faithfulness to vLLM's
observable behavior.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

# Make the chapter's implementation package importable as `implementation`.
CHAPTER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHAPTER_DIR))

from implementation._types import RequestOutputKind  # noqa: E402


@dataclass
class FakeSamplingParams:
    output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE
    n: int = 1
    stop: object = None
    min_tokens: int = 0
    include_stop_str_in_output: bool = False
    num_logprobs: object = None
    prompt_logprobs: object = None


@dataclass
class FakeRequest:
    request_id: str
    external_req_id: str
    prompt_token_ids: list = field(default_factory=list)
    sampling_params: FakeSamplingParams = field(default_factory=FakeSamplingParams)


def char_tokenizer(token_id: int) -> str:
    """A trivial 1-token-1-char decode: token id N -> chr(N)."""
    return chr(token_id)


@pytest.fixture
def char_tok():
    return char_tokenizer


def ids(s: str) -> list:
    return [ord(c) for c in s]
