"""Host-runnable tests for the slow-path incremental detokenization algorithm.

These exercise the prefix/read double-window and the UTF-8 byte-fallback boundary
in ``detokenizer_utils.detokenize_incrementally`` / ``convert_prompt_ids_to_tokens``
against the *real vLLM observable behaviour*, using a deterministic stub tokenizer
(a test fixture is allowed to fake the tokenizer; the algorithm under test is the
real subtract-only port).

Run on host: ``python3 -m pytest instances/vllm/artifacts/ch09-detokenization/tests/test_detokenize_incrementally.py``
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from implementation.detokenizer_utils import (  # noqa: E402
    INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET,
    convert_prompt_ids_to_tokens,
    detokenize_incrementally,
)


class WordTokenizer:
    """A fast-style tokenizer whose token piece for id ``i`` is ``vocab[i]``.

    ``convert_tokens_to_string`` joins pieces directly (no extra cleanup), which is
    enough to drive the prefix/read window arithmetic. A leading space inside a
    piece models the SentencePiece "▁ => space" convention.
    """

    is_fast = True

    def __init__(self, vocab: list[str]):
        self.vocab = vocab

    def __len__(self):
        return len(self.vocab)

    def get_added_vocab(self):
        return {}

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        return [self.vocab[i] if 0 <= i < len(self.vocab) else None for i in ids]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)


class ByteFallbackTokenizer(WordTokenizer):
    """Splits a single multi-byte char across two ids: each yields half the bytes.

    ``convert_tokens_to_string`` decodes the accumulated raw bytes with
    errors="replace", so a half sequence ends with U+FFFD until completed — exactly
    the byte-fallback boundary detokenize_incrementally must hold back.
    """

    def __init__(self, vocab_bytes: list[bytes]):
        self.vocab_bytes = vocab_bytes

    is_fast = True

    def __len__(self):
        return len(self.vocab_bytes)

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        # token piece carries its raw bytes verbatim (latin-1 round-trips bytes)
        return [
            self.vocab_bytes[i].decode("latin-1") if 0 <= i < len(self.vocab_bytes)
            else None
            for i in ids
        ]

    def convert_tokens_to_string(self, tokens):
        raw = "".join(tokens).encode("latin-1")
        return raw.decode("utf-8", errors="replace")


def test_convert_prompt_ids_to_tokens_offsets():
    vocab = [f"w{i}" for i in range(20)]
    tok = WordTokenizer(vocab)
    prompt_ids = list(range(10))
    tokens, prefix_offset, read_offset = convert_prompt_ids_to_tokens(
        tok, prompt_ids
    )
    # Only the last OFFSET+2 ids are converted.
    assert len(tokens) == INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET + 2
    assert read_offset == len(tokens)
    assert prefix_offset == max(
        read_offset - INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET, 0
    )


def test_incremental_emits_only_new_text():
    vocab = ["He", "llo", " wor", "ld"]
    tok = WordTokenizer(vocab)
    # First iteration: prev_tokens=None forces priming from all but last id.
    all_ids = [0]
    new_tokens, text, prefix_offset, read_offset = detokenize_incrementally(
        tok, all_ids, prev_tokens=None, prefix_offset=0, read_offset=0
    )
    acc = text
    prev = new_tokens
    for nid in [1, 2, 3]:
        all_ids.append(nid)
        new_tokens, text, prefix_offset, read_offset = detokenize_incrementally(
            tok, all_ids, prev_tokens=prev, prefix_offset=prefix_offset,
            read_offset=read_offset,
        )
        prev = prev + new_tokens
        acc += text
    assert acc == "Hello world"


def test_utf8_holdback_then_release():
    # id0 = first half of a 2-byte utf-8 char (e.g. 'é' = b'\xc3\xa9'),
    # id1 = second half; only together do they decode.
    tok = ByteFallbackTokenizer([b"\xc3", b"\xa9"])

    all_ids = [0]
    # prime
    new_tokens, text, prefix_offset, read_offset = detokenize_incrementally(
        tok, all_ids, prev_tokens=None, prefix_offset=0, read_offset=0
    )
    # First half: new_text ends with U+FFFD => held back (empty emitted).
    assert text == ""
    held_prefix, held_read = prefix_offset, read_offset
    prev = new_tokens

    all_ids.append(1)
    new_tokens, text, prefix_offset, read_offset = detokenize_incrementally(
        tok, all_ids, prev_tokens=prev, prefix_offset=held_prefix,
        read_offset=held_read,
    )
    # Second half completes the char => 'é' now emitted.
    assert text == "é"


def test_out_of_bounds_token_id_emits_empty_token():
    vocab = ["a", "b"]
    tok = WordTokenizer(vocab)
    all_ids = [0, 999]  # 999 >= len(tok)
    new_tokens, text, _, _ = detokenize_incrementally(
        tok, all_ids, prev_tokens=["a"], prefix_offset=0, read_offset=1
    )
    assert new_tokens == [""]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
