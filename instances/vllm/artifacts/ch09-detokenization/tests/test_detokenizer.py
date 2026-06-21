"""Tests for the IncrementalDetokenizer hierarchy: stop strings, holdback,
min_tokens guard, stop-token exclusion, and the three-way factory dispatch.

``detokenizer.py`` imports ``tokenizers``/``transformers`` at module top (faithful
to real vLLM). The pure update / stop / holdback / Slow-path logic does not use
those libraries at *runtime*, so when they are absent on host we inject lightweight
fake modules into ``sys.modules`` purely to satisfy the import — the code under
test is the unmodified subtract-only port. Fast-path DecodeStream tests live in
``test_fast_detokenizer.py`` and are skipped unless the real libs are present.
"""

import os
import sys
import types

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)


def _ensure_import_deps():
    """Provide minimal stand-ins for tokenizers/transformers if missing on host."""
    if "tokenizers" not in sys.modules:
        try:
            import tokenizers  # noqa: F401
        except ImportError:
            tk = types.ModuleType("tokenizers")
            tk.__version__ = "0.22.0"
            tk.Tokenizer = type("Tokenizer", (), {})
            dec = types.ModuleType("tokenizers.decoders")
            dec.DecodeStream = type("DecodeStream", (), {})
            tk.decoders = dec
            sys.modules["tokenizers"] = tk
            sys.modules["tokenizers.decoders"] = dec
    if "transformers" not in sys.modules:
        try:
            import transformers  # noqa: F401
        except ImportError:
            tf = types.ModuleType("transformers")
            tf.PreTrainedTokenizerFast = type("PreTrainedTokenizerFast", (), {})
            sys.modules["transformers"] = tf


_ensure_import_deps()

from implementation._types import (  # noqa: E402
    EngineCoreRequest,
    SamplingParams,
)
from implementation.detokenizer import (  # noqa: E402
    BaseIncrementalDetokenizer,
    IncrementalDetokenizer,
    SlowIncrementalDetokenizer,
    check_stop_strings,
)


class _CharDetokenizer(BaseIncrementalDetokenizer):
    """Concrete Base subclass whose decode_next maps id -> a single char.

    Lets us drive update()/holdback/min_tokens/stop logic deterministically
    without any real tokenizer (Fast/Slow decode_next backends are covered
    elsewhere).
    """

    def __init__(self, request, id_to_char):
        super().__init__(request)
        self.id_to_char = id_to_char

    def decode_next(self, next_token_id: int) -> str:
        return self.id_to_char.get(next_token_id, "")


def _req(**params):
    return EngineCoreRequest(
        request_id="r0",
        sampling_params=SamplingParams(**params),
        prompt_token_ids=[],
    )


# ----- check_stop_strings (unit) ---------------------------------------------

def test_check_stop_strings_no_new_chars_returns_none():
    assert check_stop_strings("hello", 0, ["lo"], False) is None


def test_check_stop_strings_exclude_truncates_to_start():
    # "abXY" with stop "XY", exclude => truncate to index of 'X' (2)
    res = check_stop_strings("abXY", 2, ["XY"], include_in_output=False)
    assert res == ("XY", 2)


def test_check_stop_strings_include_truncates_to_end():
    # include_in_output and stop is not at very end => truncate after stop
    res = check_stop_strings("abXYcd", 4, ["XY"], include_in_output=True)
    assert res == ("XY", 4)


def test_check_stop_strings_include_at_end_no_truncation():
    res = check_stop_strings("abXY", 2, ["XY"], include_in_output=True)
    assert res == ("XY", -1)


def test_check_stop_strings_window_skips_old_text():
    # stop existed in already-searched region (new_char_count small) => miss
    assert check_stop_strings("XYabcdef", 1, ["XY"], False) is None


# ----- BaseIncrementalDetokenizer.update -------------------------------------

def test_update_detects_stop_and_truncates_excluded():
    # tokens spell "ab" then "STOP"; stop excluded.
    chars = {1: "a", 2: "b", 3: "S", 4: "T", 5: "O", 6: "P"}
    d = _CharDetokenizer(_req(stop="STOP"), chars)
    assert d.update([1, 2], stop_terminated=False) is None
    assert d.output_text == "ab"
    stop = d.update([3, 4, 5, 6], stop_terminated=False)
    assert stop == "STOP"
    # excluded => truncated to before the stop string
    assert d.output_text == "ab"


def test_update_stop_included_keeps_stop_text():
    chars = {1: "a", 3: "S", 4: "T", 5: "O", 6: "P", 7: "x"}
    d = _CharDetokenizer(_req(stop="STOP", include_stop_str_in_output=True), chars)
    stop = d.update([1, 3, 4, 5, 6, 7], stop_terminated=False)
    assert stop == "STOP"
    assert d.output_text == "aSTOP"  # truncated to end of stop, drops trailing 'x'


def test_min_tokens_guard_blocks_early_stop():
    # stop "ab" would match at 2 tokens, but min_tokens=4 forbids stopping early.
    chars = {1: "a", 2: "b", 3: "c", 4: "d"}
    d = _CharDetokenizer(_req(stop="ab", min_tokens=4), chars)
    assert d.update([1, 2], stop_terminated=False) is None  # blocked by min_tokens
    assert d.output_text == "ab"
    # past min_tokens, stop is allowed again on next matching window
    chars2 = {1: "a", 2: "b", 3: "x", 4: "y", 5: "a", 6: "b"}
    d2 = _CharDetokenizer(_req(stop="ab", min_tokens=4), chars2)
    assert d2.update([1, 2, 3, 4], stop_terminated=False) is None
    stop = d2.update([5, 6], stop_terminated=False)
    assert stop == "ab"


def test_stop_terminated_token_excluded_from_text_but_kept_in_ids():
    chars = {1: "a", 2: "b", 9: "Z"}
    d = _CharDetokenizer(_req(), chars)  # no stop strings
    d.update([1, 2, 9], stop_terminated=True)  # last token is the stop token
    assert d.output_text == "ab"  # 'Z' not decoded into text
    assert d.token_ids == [1, 2, 9]  # but kept in token_ids


def test_stop_terminated_with_include_keeps_token_text():
    chars = {1: "a", 9: "Z"}
    d = _CharDetokenizer(_req(include_stop_str_in_output=True), chars)
    d.update([1, 9], stop_terminated=True)
    assert d.output_text == "aZ"


# ----- holdback (get_next_output_text) ---------------------------------------

def test_stop_buffer_length_is_max_len_minus_one():
    d = _CharDetokenizer(_req(stop=["ab", "abcd"]), {})
    assert d.stop_buffer_length == 3  # max(2,4) - 1


def test_holdback_withholds_tail_until_finished():
    chars = {i: c for i, c in enumerate("hello", start=1)}
    d = _CharDetokenizer(_req(stop=["XYZ"]), chars)  # buffer_length = 2
    d.update([1, 2, 3, 4, 5], stop_terminated=False)  # "hello"
    # streaming (not finished): hold back last 2 chars
    assert d.get_next_output_text(finished=False, delta=False) == "hel"
    # finished: release everything
    assert d.get_next_output_text(finished=True, delta=False) == "hello"


def test_holdback_delta_mode_emits_incrementally():
    chars = {i: c for i, c in enumerate("hello", start=1)}
    d = _CharDetokenizer(_req(stop=["XYZ"]), chars)  # buffer_length = 2
    d.update([1, 2, 3], stop_terminated=False)  # "hel"
    assert d.get_next_output_text(finished=False, delta=True) == "h"  # 3-2=1
    d.update([4, 5], stop_terminated=False)  # "hello"
    # length = 5 - 2 = 3; emit output_text[1:3] = "el", offset 1 -> 3
    assert d.get_next_output_text(finished=False, delta=True) == "el"
    # finished: buffer_length 0, length = 5; emit output_text[3:5] = "lo"
    assert d.get_next_output_text(finished=True, delta=True) == "lo"


# ----- factory dispatch ------------------------------------------------------

def test_from_new_request_none_tokenizer_is_empty_shell():
    d = IncrementalDetokenizer.from_new_request(None, _req())
    assert type(d) is IncrementalDetokenizer
    assert d.update([1, 2, 3], stop_terminated=False) is None
    assert d.get_next_output_text(finished=True, delta=False) == ""


def test_from_new_request_non_fast_tokenizer_is_slow():
    class NotFast:  # not a PreTrainedTokenizerFast instance
        is_fast = False

        def __len__(self):
            return 100

        def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
            return ["x"] * len(ids)

        def get_added_vocab(self):
            return {}

        def convert_tokens_to_string(self, tokens):
            return "".join(tokens)

    req = EngineCoreRequest(
        request_id="r0",
        sampling_params=SamplingParams(),
        prompt_token_ids=[1, 2, 3],
    )
    d = IncrementalDetokenizer.from_new_request(NotFast(), req)
    assert isinstance(d, SlowIncrementalDetokenizer)


# ----- Slow path output-token accounting (prompt excluded) -------------------

def test_slow_path_excludes_prompt_from_output_tokens():
    class Tok:
        is_fast = True

        def __len__(self):
            return 100

        def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
            return [str(i) for i in ids]

        def get_added_vocab(self):
            return {}

        def convert_tokens_to_string(self, tokens):
            return "".join(tokens)

    req = EngineCoreRequest(
        request_id="r0",
        sampling_params=SamplingParams(),
        prompt_token_ids=[10, 11, 12],
    )
    d = SlowIncrementalDetokenizer(Tok(), req)
    assert d.num_output_tokens() == 0  # prompt does not count as output
    assert d.output_token_ids == []
    d.update([20, 21], stop_terminated=False)
    assert d.num_output_tokens() == 2
    assert d.output_token_ids == [20, 21]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
