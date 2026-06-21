"""Fast-path (tokenizers DecodeStream) tests.

These need the real ``tokenizers`` (>=0.22.0) + ``transformers`` libraries and a
real fast tokenizer, so they are skipped on host and run inside the vLLM
container:

  scripts/vllm_docker.sh -m pytest \
    /work/instances/vllm/artifacts/ch09-detokenization/tests/test_fast_detokenizer.py -v

They assert the real observable behaviour: native-prefill incremental decode
matches a full decode, and _protected_step rebuilds the DecodeStream on an
"Invalid prefix encountered" error.
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")
from packaging import version  # noqa: E402

# Guard against the lightweight fake modules other host tests inject into
# sys.modules: require the *real* libraries (a real DecodeStream + AutoTokenizer).
if not hasattr(transformers, "AutoTokenizer") or not callable(
    getattr(tokenizers.decoders.DecodeStream, "step", None)
):
    pytest.skip(
        "real tokenizers/transformers required — run inside the vLLM container",
        allow_module_level=True,
    )

if version.parse(tokenizers.__version__) < version.parse("0.22.0"):
    pytest.skip(
        "tokenizers >= 0.22.0 required for DecodeStream native prefill",
        allow_module_level=True,
    )

from transformers import AutoTokenizer  # noqa: E402

from implementation._types import EngineCoreRequest, SamplingParams  # noqa: E402
from implementation.detokenizer import (  # noqa: E402
    FastIncrementalDetokenizer,
    INVALID_PREFIX_ERR_MSG,
    IncrementalDetokenizer,
)


MODEL = os.environ.get("CH09_TOKENIZER", "gpt2")


@pytest.fixture(scope="module")
def fast_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if not tok.is_fast:
        pytest.skip(f"{MODEL} did not load a fast tokenizer")
    return tok


def _req(tok, prompt, **params):
    prompt_ids = tok.encode(prompt)
    return EngineCoreRequest(
        request_id="r0",
        sampling_params=SamplingParams(**params),
        prompt_token_ids=prompt_ids,
    )


def test_factory_picks_fast_path(fast_tokenizer):
    d = IncrementalDetokenizer.from_new_request(
        fast_tokenizer, _req(fast_tokenizer, "Hello")
    )
    assert isinstance(d, FastIncrementalDetokenizer)


def test_incremental_matches_full_decode(fast_tokenizer):
    tok = fast_tokenizer
    completion = " world, this is vLLM."
    out_ids = tok.encode(completion, add_special_tokens=False)
    d = IncrementalDetokenizer.from_new_request(tok, _req(tok, "Hello"))
    streamed = "".join(d.decode_next(i) for i in out_ids)
    # Incremental detok with prompt prefill should reproduce a one-shot decode
    # of the same continuation (modulo leading-space cleanup the prefill primes).
    assert streamed == tok.decode(out_ids)


def test_protected_step_recovers_on_invalid_prefix(fast_tokenizer, monkeypatch):
    d = IncrementalDetokenizer.from_new_request(
        fast_tokenizer, _req(fast_tokenizer, "Hi")
    )
    calls = {"n": 0}
    real_stream = d.stream

    class FlakyStream:
        def step(self, tokenizer, token_id):
            calls["n"] += 1
            if calls["n"] == 1:
                raise Exception(INVALID_PREFIX_ERR_MSG + " (simulated)")
            return real_stream.step(tokenizer, token_id)

    d.stream = FlakyStream()
    # First step raises invalid-prefix => _protected_step rebuilds the stream and
    # retries, so decode_next does not propagate the exception.
    token = d.decode_next(fast_tokenizer.encode(" ok", add_special_tokens=False)[0])
    assert isinstance(token, str)
    # stream was rebuilt (no longer the FlakyStream instance).
    assert not isinstance(d.stream, FlakyStream)


def test_protected_step_swallows_overflow(fast_tokenizer):
    d = IncrementalDetokenizer.from_new_request(
        fast_tokenizer, _req(fast_tokenizer, "Hi")
    )

    class OverflowStream:
        def step(self, tokenizer, token_id):
            raise OverflowError("simulated")

    d.stream = OverflowStream()
    # decode_next returns "" instead of raising.
    assert d.decode_next(0) == ""


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
