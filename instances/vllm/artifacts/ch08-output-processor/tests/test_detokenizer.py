"""IncrementalDetokenizer.update + check_stop_strings + get_next_output_text.

Asserts vLLM's observable behavior:
 - incremental text accumulation;
 - stop-string detection truncates output and returns the matched string;
 - include_stop_str_in_output keeps the stop string;
 - stop-terminated last token excluded from detokenization but kept in token_ids;
 - min_tokens defers stop-string activation;
 - get_next_output_text delta vs full + stop_buffer_length hold-back;
 - the no-tokenizer base detokenizer only tracks token_ids.
"""

from conftest import FakeRequest, FakeSamplingParams, char_tokenizer, ids

from implementation._types import RequestOutputKind
from implementation.detokenizer import IncrementalDetokenizer, check_stop_strings


def _det(stop=None, min_tokens=0, include=False):
    req = FakeRequest(
        request_id="r",
        external_req_id="r",
        sampling_params=FakeSamplingParams(
            stop=stop, min_tokens=min_tokens, include_stop_str_in_output=include
        ),
    )
    return IncrementalDetokenizer.from_new_request(char_tokenizer, req)


def test_incremental_accumulation():
    d = _det()
    assert d.update(ids("ab"), False) is None
    assert d.output_text == "ab"
    assert d.update(ids("cd"), False) is None
    assert d.output_text == "abcd"
    assert d.output_token_ids == ids("abcd")
    assert d.num_output_tokens() == 4


def test_stop_string_truncates_and_returns():
    d = _det(stop=["STOP"])
    d.update(ids("hello"), False)
    # feed the rest so output_text becomes "helloSTOPx"
    ss = d.update(ids("STOPx"), False)
    assert ss == "STOP"
    # excluded from output by default -> truncate to before stop string
    assert d.output_text == "hello"


def test_stop_string_included_in_output():
    d = _det(stop=["END"], include=True)
    d.update(ids("ab"), False)
    ss = d.update(ids("ENDz"), False)
    assert ss == "END"
    assert d.output_text == "abEND"


def test_stop_terminated_excludes_last_token_but_keeps_id():
    # When EngineCore says STOP, the last token is not detokenized into text
    # (unless include_stop_str_in_output) but is still recorded in token_ids.
    d = _det()
    d.update(ids("abc"), stop_terminated=True)
    assert d.output_text == "ab"  # last char 'c' not decoded
    assert d.output_token_ids == ids("abc")  # but id retained


def test_min_tokens_defers_stop():
    d = _det(stop=["x"], min_tokens=3)
    # 'x' appears within first 3 tokens -> not allowed to stop yet
    assert d.update(ids("axb"), False) is None
    # now beyond min_tokens; a later 'x' triggers
    ss = d.update(ids("cx"), False)
    assert ss == "x"


def test_get_next_output_text_delta_and_full():
    d = _det()
    d.update(ids("abcd"), False)
    # delta returns only new chars and advances offset
    assert d.get_next_output_text(finished=False, delta=True) == "abcd"
    d.update(ids("ef"), False)
    assert d.get_next_output_text(finished=False, delta=True) == "ef"
    # full returns everything
    assert d.get_next_output_text(finished=False, delta=False) == "abcdef"


def test_stop_buffer_length_holds_back_until_finished():
    # stop_buffer_length = max(len(stop)) - 1 = 2 for "abc"
    d = _det(stop=["abc"])
    assert d.stop_buffer_length == 2
    d.update(ids("xyZZ"), False)
    # not finished -> last 2 chars held back
    assert d.get_next_output_text(finished=False, delta=False) == "xy"
    # finished -> full text
    assert d.get_next_output_text(finished=True, delta=False) == "xyZZ"


def test_no_tokenizer_base_only_tracks_token_ids():
    req = FakeRequest(request_id="r", external_req_id="r")
    d = IncrementalDetokenizer.from_new_request(None, req)
    assert d.update(ids("ab"), False) is None
    assert d.output_text == "" if hasattr(d, "output_text") else True
    assert d.output_token_ids == ids("ab")
    assert d.get_next_output_text(False, True) == ""


def test_check_stop_strings_window():
    # No new chars searched -> None
    assert check_stop_strings("hello", 0, ["lo"], False) is None
    # Match within new window, exclude -> truncate to match start
    res = check_stop_strings("helloXX", new_char_count=2, stop=["XX"], include_in_output=False)
    assert res == ("XX", 5)
    # include in output, at end -> no truncation (-1)
    res = check_stop_strings("helloXX", new_char_count=2, stop=["XX"], include_in_output=True)
    assert res == ("XX", -1)
