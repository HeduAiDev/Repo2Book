"""TDD tests for the v3 ch07 subtract-only companion (pin vLLM v0.27.1 / 6e448d0ea).

These assert the *observable vLLM behavior* the chapter teaches — the API
process uplink swimlane: the string of token ids the engine ships back is
detokenized incrementally, demuxed per request, gated by the three-state
RequestOutputKind contract (+ stream_interval / n>1 aggregation), staged in a
single-slot RequestOutputCollector and streamed out by generate(); on client
disconnect the request is aborted back into the engine. Runs on a plain CPU
host without the vllm package; the only stand-ins are the documented HOST
SEAMs in uplink.py (tokenizers backend wrapper / Slow-path TokenizerLike /
msgspec / output socket / engine input sink), each anchored to the real vLLM
interface it mirrors:

- m1  arrival: EngineCoreOutputQueueTask turns ZMQ frames into a queue of
      EngineCoreOutputs (real msgpack bytes here); exceptions and the dead
      sentinel ride the same queue;
- m2  output_handler: one resident task, chunks of
      VLLM_V1_OUTPUT_PROC_CHUNK_SIZE with asyncio.sleep(0) between slices;
      eager start in-loop, lazy start on first add_request;
- m3  process_outputs: the only loop over the batch; demux by internal id,
      already-aborted ids skipped idempotently;
- m4  incremental detokenization update: skip stop token -> decode_next per
      token -> min_tokens window -> check_stop_strings truncation;
- m5  three-way detokenizer factory (null shell / Fast DecodeStream / Slow
      pure-python);
- m6  Fast path: real Rust tokenizers DecodeStream with native prefill;
      'Invalid prefix encountered' recovery rebuilds the stream;
- m7  Slow path: prefix/read double-offset window (initial window = last 7
      prompt tokens, prefix back 5);
- m8  byte-fallback UTF-8 boundary: tail '�' or no growth -> empty
      string, read offset frozen until the next token completes the char;
- m9  stop-string holdback: stop_buffer_length = max(len(s)) - 1 held back
      while unfinished; released on finish;
- m10 check_stop_strings: windowed find + earliest-completion arbitration;
- m11 min_tokens guards stop checking (PR #22014);
- m12 stop-token exclusion: text and token id ledgers kept separate;
- m13 RequestOutputKind: FINAL_ONLY constructs nothing until finish;
- m14 stream_interval throttle (DELTA from sent_tokens_offset, clamp max);
- m15 n>1 fan-out (idx_ prefixed child ids, one collector) + parent
      aggregation (stream forwards, FINAL_ONLY collects all n);
- m16 single-slot collector: put merges via RequestOutput.add, exceptions
      preempt; get_nowait() or await get() fast path;
- m17 producer/consumer: output_handler feeds N collectors, generate()
      consumes and yields;
- m18 disconnect -> CancelledError -> abort(internal=True): local state
      removed + ABORT terminal output, then cross-process ABORT frame;
- m19 stop-string reverse abort (reqs_to_abort rides the handler loop);
- m20 propagate_error: a handler crash reaches every waiting consumer;
- m21 sync face: queue=None returns the list instead of putting;
- m22 dual-track id uplink exit: outputs carry the external id; abort works
      on external (expands) / internal (single) / parent (cascades) ids.

Run:  cd instances/vllm/artifacts-v3/ch07-uplink-token-to-text
      python -m pytest tests/ -q
"""

import asyncio
import contextlib
import importlib
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import tokenizers
import tokenizers.decoders

_IMPL_DIR = Path(__file__).resolve().parent.parent / "implementation"
sys.path.insert(0, str(_IMPL_DIR))

uplink = importlib.import_module("uplink")


# ---------------------------------------------------------------------------
# Fixtures: byte-level token streams (one token per raw UTF-8 byte) so the
# Fast path rides the REAL Rust DecodeStream and the Slow path rides a
# TokenizerLike seam with the same byte-fallback semantics.
# ---------------------------------------------------------------------------


def _bytes_to_unicode():
    # GPT-2 byte-level alphabet: printables + ¡..¬ (161-172 inclusive) + ®..ÿ
    bs = list(range(33, 127)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


_B2U = _bytes_to_unicode()
_BYTE_VOCAB = {ch: byte for byte, ch in _B2U.items()}


def b(text: str) -> list[int]:
    """Byte-level token ids: one token per raw UTF-8 byte."""
    return list(text.encode("utf-8"))


def fast_backend():
    """A real tokenizers.Tokenizer wrapped in the impl's TokenizersBackend —
    the exact surface FastIncrementalDetokenizer touches (._tokenizer)."""
    tk = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab=_BYTE_VOCAB, unk_token=None)
    )
    tk.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False
    )
    tk.decoder = tokenizers.decoders.ByteLevel()
    return uplink.TokenizersBackend(tk)


class SlowByteTokenizer:
    """TokenizerLike seam for the Slow path: one token per raw byte;
    convert_tokens_to_string joins and UTF-8-decodes with replacement — the
    byte-fallback semantics the real Slow path sees from the HF tokenizer.
    id 0 is an empty control token (the no-increment case)."""

    is_fast = False

    def __len__(self):
        return 256

    def get_added_vocab(self):
        return {}

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        out = []
        for i in ids:
            if i == 0:
                out.append("")
            elif 0 < i < 256:
                out.append(chr(i))
            else:
                out.append(None)  # out-of-vocab: caller replaces with ""
        return out

    def convert_tokens_to_string(self, tokens):
        raw = "".join(t for t in tokens).encode("latin-1", errors="replace")
        return raw.decode("utf-8", errors="replace")


def sp(**kw):
    """SamplingParams seam with the real defaults vLLM ships."""
    base = dict(
        n=1,
        stop=(),
        min_tokens=0,
        include_stop_str_in_output=False,
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
        detokenize=True,
        output_kind=uplink.RequestOutputKind.DELTA,
        stream_interval=None,
        seed=None,
        max_tokens=32,
    )
    base.update(kw)
    return uplink.SamplingParams(**base)


def make_request(ext_id, prompt_ids, params):
    return uplink.EngineCoreRequest(
        request_id=ext_id,
        prompt_token_ids=list(prompt_ids),
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        arrival_time=1.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def eco(rid, new_ids, finish=None, stop_reason=None, prefill=None):
    """One EngineCoreOutput for request `rid` (the INTERNAL id)."""
    return uplink.EngineCoreOutput(
        request_id=rid,
        new_token_ids=list(new_ids),
        finish_reason=finish,
        stop_reason=stop_reason,
        prefill_stats=prefill,
    )


def ro(text, token_ids, index=0, finished=False, request_id="ext-1"):
    """A RequestOutput with a single CompletionOutput."""
    return uplink.RequestOutput(
        request_id=request_id,
        prompt=None,
        prompt_token_ids=[1],
        prompt_logprobs=None,
        outputs=[
            uplink.CompletionOutput(
                index=index,
                text=text,
                token_ids=list(token_ids),
                cumulative_logprob=None,
                logprobs=None,
                finish_reason="stop" if finished else None,
                stop_reason=None,
            )
        ],
        finished=finished,
    )


def fast_detok(prompt="Hi", **param_kw):
    req = make_request("int-1", b(prompt), sp(**param_kw))
    return uplink.IncrementalDetokenizer.from_new_request(fast_backend(), req)


def slow_detok(prompt="hi", **param_kw):
    req = make_request("int-1", b(prompt), sp(**param_kw))
    return uplink.IncrementalDetokenizer.from_new_request(SlowByteTokenizer(), req)


class Harness:
    """AsyncLLM + seam AsyncMPClient wired to a scriptable output socket."""

    def __init__(self, tokenizer=None, stream_interval=1):
        self.socket = uplink.SeamOutputSocket()
        self.client = uplink.AsyncMPClient(self.socket)
        self.llm = uplink.AsyncLLM(
            self.client, tokenizer=tokenizer, stream_interval=stream_interval
        )

    async def add(self, ext_id, prompt_ids, params):
        return await self.llm.add_request(
            ext_id, make_request(ext_id, prompt_ids, params), params
        )

    def feed(self, outputs, engine_index=0, timestamp=123.0):
        msg = uplink.EngineCoreOutputs(
            engine_index=engine_index, outputs=list(outputs), timestamp=timestamp
        )
        self.socket.feed([uplink.SeamFrame(uplink.msgspec.msgpack.encode(msg))])

    def feed_dead(self):
        self.socket.feed([uplink.SeamFrame(b"ENGINE_CORE_DEAD")])

    def add_frames(self):
        return [
            r
            for t, r in self.client.sent
            if t is uplink.EngineCoreRequestType.ADD
        ]

    def abort_frames(self):
        return [
            r
            for t, r in self.client.sent
            if t is uplink.EngineCoreRequestType.ABORT
        ]

    async def close(self):
        tasks = [self.llm.output_handler, self.client.resources.output_queue_task]
        for t in tasks:
            if t is not None:
                t.cancel()
        for t in tasks:
            if t is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    await t


async def wait_for_add(h, n=1):
    for _ in range(500):
        if len(h.add_frames()) >= n:
            return
        await asyncio.sleep(0)
    raise AssertionError("ADD frame never sent")


def run(coro_factory):
    asyncio.run(coro_factory())


# ---------------------------------------------------------------------------
# m16 — RequestOutputCollector: the single-slot mailbox
# ---------------------------------------------------------------------------


def test_collector_aggregate_flag_follows_output_kind():
    q_delta = uplink.RequestOutputCollector(uplink.RequestOutputKind.DELTA, "int-1")
    q_cumul = uplink.RequestOutputCollector(
        uplink.RequestOutputKind.CUMULATIVE, "int-1"
    )
    q_final = uplink.RequestOutputCollector(
        uplink.RequestOutputKind.FINAL_ONLY, "int-1"
    )
    assert q_delta.aggregate is True
    assert q_cumul.aggregate is False
    assert q_final.aggregate is False
    assert q_delta.request_id == "int-1"  # holds the INTERNAL id (abort uses it)


def test_collector_delta_merges_in_slot():
    async def main():
        q = uplink.RequestOutputCollector(uplink.RequestOutputKind.DELTA, "int-1")
        q.put(ro("He", [1]))
        q.put(ro("llo", [2, 3]))
        out = await q.get()
        assert out.outputs[0].text == "Hello"
        assert list(out.outputs[0].token_ids) == [1, 2, 3]
        # single slot: both puts coalesced, nothing left queued
        assert q.get_nowait() is None

    run(main)


def test_collector_delta_merge_unions_finished_and_new_index():
    async def main():
        q = uplink.RequestOutputCollector(uplink.RequestOutputKind.DELTA, "int-1")
        q.put(ro("He", [1], index=0))
        q.put(ro("llo", [2], index=0, finished=True))
        q.put(ro("world", [9], index=1))  # n>1: different index appends, no override
        out = await q.get()
        assert out.finished is True
        assert [(c.index, c.text) for c in out.outputs] == [(0, "Hello"), (1, "world")]

    run(main)


def test_collector_cumulative_replaces_in_slot():
    async def main():
        q = uplink.RequestOutputCollector(
            uplink.RequestOutputKind.CUMULATIVE, "int-1"
        )
        q.put(ro("He", [1]))
        q.put(ro("Hello", [1, 2]))
        out = await q.get()
        assert out.outputs[0].text == "Hello"  # snapshot replaced, not appended
        assert q.get_nowait() is None

    run(main)


def test_collector_exception_preempts_occupied_slot():
    async def main():
        q = uplink.RequestOutputCollector(uplink.RequestOutputKind.DELTA, "int-1")
        q.put(ro("He", [1]))
        q.put(RuntimeError("engine exploded"))  # errors preempt unconditionally
        with pytest.raises(RuntimeError, match="engine exploded"):
            await q.get()

    run(main)


def test_collector_get_blocks_until_put():
    async def main():
        q = uplink.RequestOutputCollector(uplink.RequestOutputKind.DELTA, "int-1")
        assert q.get_nowait() is None

        async def late_put():
            await asyncio.sleep(0.01)
            q.put(ro("x", [1]))

        task = asyncio.ensure_future(late_put())
        out = await asyncio.wait_for(q.get(), 1)
        assert out.outputs[0].text == "x"
        await task

    run(main)


def test_get_nowait_or_await_fast_path_yields_first_output_without_switch():
    async def main():
        q = uplink.RequestOutputCollector(uplink.RequestOutputKind.DELTA, "int-1")
        q.put(ro("He", [1]))
        # generate()'s exact drain spell: non-blocking first, blocking fallback
        out = q.get_nowait() or await q.get()
        assert out.outputs[0].text == "He"
        assert q.get_nowait() is None

    run(main)


# ---------------------------------------------------------------------------
# RequestOutput.add — the merge implementation (outputs.py:L152-181)
# ---------------------------------------------------------------------------


def test_request_output_add_replaces_for_cumulative():
    base = ro("He", [1])
    base.add(ro("Hello", [1, 2, 3]), aggregate=False)
    assert base.outputs[0].text == "Hello"
    assert list(base.outputs[0].token_ids) == [1, 2, 3]


def test_request_output_add_extends_token_ids_in_place():
    base = ro("He", [1])
    base.add(ro("llo", [2]), aggregate=True)
    assert base.outputs[0].text == "Hello"
    assert list(base.outputs[0].token_ids) == [1, 2]


# ---------------------------------------------------------------------------
# m5 — detokenizer factory & null shell
# ---------------------------------------------------------------------------


def test_factory_null_shell_without_tokenizer():
    req = make_request("int-1", b("Hi"), sp())
    detok = uplink.IncrementalDetokenizer.from_new_request(None, req)
    assert type(detok) is uplink.IncrementalDetokenizer
    assert detok.update(b("hello"), False) is None  # only accumulates ids
    assert detok.get_next_output_text(False, True) == ""
    assert detok.get_next_output_text(True, False) == ""
    assert detok.output_token_ids == b("hello")
    assert detok.num_output_tokens() == 5


def test_factory_fast_vs_slow_dispatch():
    fast = uplink.IncrementalDetokenizer.from_new_request(
        fast_backend(), make_request("int-1", b("Hi"), sp())
    )
    slow = uplink.IncrementalDetokenizer.from_new_request(
        SlowByteTokenizer(), make_request("int-1", b("hi"), sp())
    )
    assert isinstance(fast, uplink.FastIncrementalDetokenizer)
    assert isinstance(slow, uplink.SlowIncrementalDetokenizer)
    assert uplink.USE_FAST_DETOKENIZER is True  # host tokenizers >= 0.22.0


def test_detokenize_false_yields_null_shell():
    # RequestState.from_new_request: detokenize=False -> tokenizer=None (L224-225)
    op = uplink.OutputProcessor(fast_backend(), log_stats=False, stream_interval=1)
    req = make_request("ext-1", b("Hi"), sp(detokenize=False))
    req.external_req_id = "ext-1"
    q = uplink.RequestOutputCollector(uplink.RequestOutputKind.DELTA, req.request_id)
    op.add_request(req, "Hi", None, 0, q)
    state = op.request_states[req.request_id]
    assert type(state.detokenizer) is uplink.IncrementalDetokenizer  # null shell


# ---------------------------------------------------------------------------
# m4/m6/m8/m9/m12 — incremental detokenization (Fast path = real DecodeStream)
# ---------------------------------------------------------------------------


def test_fast_delta_and_cumulative_text():
    detok = fast_detok(prompt="Hi")
    deltas, fulls = [], []
    for token in b("hello"):
        detok.update([token], False)
        deltas.append(detok.get_next_output_text(False, True))
        fulls.append(detok.get_next_output_text(False, False))
    assert deltas == list("hello")
    assert fulls == ["h", "he", "hel", "hell", "hello"]
    assert detok.output_token_ids == b("hello")  # Fast: ids exclude the prompt
    assert detok.num_output_tokens() == 5


def test_fast_native_prefill_prompt_not_in_output():
    detok = fast_detok(prompt="Hi")
    detok.update([111], False)  # 'o'
    assert detok.output_text == "o"
    assert detok.num_output_tokens() == 1


def test_fast_byte_fallback_multibyte_waits_for_completion():
    detok = fast_detok(prompt="Hi")
    detok.update([0xE4], False)
    assert detok.get_next_output_text(False, True) == ""  # half a char: nothing
    detok.update([0xB8], False)
    assert detok.get_next_output_text(False, True) == ""
    detok.update([0xAD], False)
    assert detok.get_next_output_text(False, True) == "中"  # 中 completes
    assert detok.output_token_ids == [0xE4, 0xB8, 0xAD]


def test_fast_invalid_prefix_recovery_rebuilds_stream():
    detok = fast_detok(prompt="Hi")

    class BoomStream:
        def step(self, tokenizer, token_id):
            raise RuntimeError("Invalid prefix encountered")

    detok.stream = BoomStream()
    token = detok.decode_next(72)  # 'H' — must recover, not crash
    assert token == "H"
    assert isinstance(detok.stream, tokenizers.decoders.DecodeStream)
    # the rebuilt stream is FRESH (prefill context lost — real semantics) and
    # keeps stepping; 'H' above was decoded outside update's loop, so only
    # what update() decodes lands in output_text
    assert detok.update([101], False) is None
    assert detok.output_text == "e"


def test_fast_overflow_and_typeerror_swallowed():
    detok = fast_detok(prompt="Hi")

    class BadIdStream:
        def step(self, tokenizer, token_id):
            raise TypeError("bad id")

    detok.stream = BadIdStream()
    assert detok.decode_next(72) == ""  # swallowed -> empty contribution
    assert detok.output_text == ""


def test_fast_stranded_continuation_byte_replacement_char():
    # a stranded continuation byte decodes to a REAL replacement char (m8:
    # mid-text '�' is genuine invalid output, only the TAIL is held back)
    detok = fast_detok(prompt="Hi")
    detok.update([0xAD], False)
    detok.update([65], False)
    assert detok.output_text == "�A"


def test_stop_string_hit_truncates_and_returns_stop():
    detok = fast_detok(prompt="Hi", stop=["STOP"])
    assert detok.update(b("hello"), False) is None
    assert detok.stop_buffer_length == 3  # max(len("STOP")) - 1
    stop = detok.update(b("STOP"), False)
    assert stop == "STOP"
    assert detok.output_text == "hello"  # truncated at stop-string start


def test_holdback_withholds_tail_until_finished():
    detok = fast_detok(prompt="Hi", stop=["STOP"])
    detok.update(b("hello"), False)
    # unfinished: last stop_buffer_length (3) chars are held back
    assert detok.get_next_output_text(False, True) == "he"
    assert detok.get_next_output_text(False, False) == "he"
    # finished: everything flows
    detok.update(b("!"), False)
    assert detok.get_next_output_text(True, True) == "llo!"


def test_holdback_zero_when_including_stop_string():
    detok = fast_detok(prompt="Hi", stop=["STOP"], include_stop_str_in_output=True)
    assert detok.stop_buffer_length == 0
    detok.update(b("hi"), False)
    assert detok.get_next_output_text(False, True) == "hi"


def test_stop_include_in_output_keeps_stop_text():
    detok = fast_detok(prompt="Hi", stop=["STOP"], include_stop_str_in_output=True)
    detok.update(b("hi"), False)
    stop = detok.update(b("STOP"), False)
    assert stop == "STOP"
    assert detok.output_text == "hiSTOP"  # kept through the END of the stop


def test_stop_token_exclusion_keeps_id_but_not_text():
    detok = fast_detok(prompt="Hi")
    # engine stop-terminated: last id is a stop TOKEN, excluded from text
    detok.update([65, 2], True)
    assert detok.output_token_ids == [65, 2]
    assert detok.output_text == "A"
    assert "\x02" not in detok.output_text

    incl = fast_detok(prompt="Hi", include_stop_str_in_output=True)
    incl.update([65, 2], True)
    assert incl.output_text == "A\x02"  # included: id decoded too


def test_empty_update_is_noop():
    detok = fast_detok(prompt="Hi", stop=["STOP"])
    assert detok.update([], True) is None


# ---------------------------------------------------------------------------
# m10 — check_stop_strings (module function)
# ---------------------------------------------------------------------------


def test_check_stop_strings_earliest_completion_wins():
    # both "AND" (ends at 4) and "DAN" (ends at 6) match; earliest wins
    hit = uplink.check_stop_strings("BANDANA", 7, ["DAN", "AND"], False)
    assert hit == ("AND", 1)


def test_check_stop_strings_include_truncates_to_end():
    hit = uplink.check_stop_strings("hello STOP", 5, ["STOP"], True)
    assert hit == ("STOP", -1)  # stop completes exactly at the tail: no trunc


def test_check_stop_strings_window_skips_already_searched_text():
    # only 1 new char: find starts at len-4, the old "STOP" is out of window
    assert uplink.check_stop_strings("XSTOPY", 1, ["STOP"], False) is None
    # 2 new chars bring the tail of an old match back into the window
    assert uplink.check_stop_strings("XSTOPY", 2, ["STOP"], False) == ("STOP", 1)


def test_check_stop_strings_no_new_chars_or_no_stops():
    assert uplink.check_stop_strings("STOP", 0, ["STOP"], False) is None
    assert uplink.check_stop_strings("abc", 3, [], False) is None


# ---------------------------------------------------------------------------
# m11 — min_tokens guards stop checking (PR #22014)
# ---------------------------------------------------------------------------


def test_min_tokens_swallows_stop_inside_guard_window():
    detok = fast_detok(prompt="Hi", stop=["AB"], min_tokens=3)
    assert detok.update([65, 66], False) is None  # "AB" but only 2 tokens
    assert detok.update([67], False) is None  # 3 tokens: still <= min
    assert detok.update([68], False) is None  # 4 tokens, but "AB" outside window
    assert detok.output_text == "ABCD"


def test_min_tokens_exceeded_stop_detected():
    detok = fast_detok(prompt="Hi", stop=["AB"], min_tokens=1)
    assert detok.update([120], False) is None  # "x"
    stop = detok.update([65, 66], False)  # "AB" after the guard: detected
    assert stop == "AB"
    assert detok.output_text == "x"


# ---------------------------------------------------------------------------
# m7/m8 — Slow path: double-offset window, UTF-8 boundary, null increments
# ---------------------------------------------------------------------------


def test_convert_prompt_ids_initial_window():
    tok = SlowByteTokenizer()
    prompt = list(range(100, 120))  # 20 ids
    tokens, prefix_offset, read_offset = uplink.convert_prompt_ids_to_tokens(
        tok, prompt, skip_special_tokens=False
    )
    assert len(tokens) == 7  # INITIAL..OFFSET (5) + 2
    assert read_offset == 7
    assert prefix_offset == 2
    assert uplink.INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET == 5


def test_convert_prompt_ids_replaces_out_of_vocab_with_empty():
    tok = SlowByteTokenizer()
    tokens, _, _ = uplink.convert_prompt_ids_to_tokens(tok, [9999, 9998], False)
    assert tokens == ["", ""]


def test_slow_delta_text_and_prompt_excluded_ids():
    detok = slow_detok(prompt="hi")
    deltas = []
    for token in b("ello"):
        detok.update([token], False)
        deltas.append(detok.get_next_output_text(False, True))
    assert deltas == list("ello")
    assert detok.output_text == "ello"
    assert detok.num_output_tokens() == 4  # Slow: counts exclude the prompt
    assert detok.output_token_ids == b("ello")


def test_slow_byte_fallback_freezes_read_offset():
    detok = slow_detok(prompt="hi")
    detok.update([0xE4], False)  # tail replacement char: empty contribution
    assert detok.output_text == ""
    assert detok.read_offset == 2  # frozen at the prompt window edge
    detok.update([0xB8], False)
    assert detok.output_text == ""
    assert detok.read_offset == 2
    detok.update([0xAD], False)  # 中 completes: window jumps past the bytes
    assert detok.output_text == "中"
    assert detok.read_offset == 5  # [h, i, \xe4, \xb8, \xad]


def test_slow_empty_token_yields_no_increment():
    detok = slow_detok(prompt="hi")
    detok.update([65], False)
    assert detok.get_next_output_text(False, True) == "A"  # consume the delta
    before = detok.read_offset
    detok.update([0], False)  # control token decodes to ""
    assert detok.get_next_output_text(False, True) == ""  # no text increment
    assert detok.read_offset == before  # window frozen across the empty token


def test_slow_out_of_vocab_id_decodes_empty():
    detok = slow_detok(prompt="hi")
    detok.update([9999], False)
    assert detok.output_text == ""
    assert detok.output_token_ids == [9999]


def test_fast_and_slow_agree_on_the_same_byte_stream():
    text = "hello 中文 world"  # ascii + multibyte mix
    fast = fast_detok(prompt="Hi", stop=["ZZZ"])
    slow = slow_detok(prompt="hi", stop=["ZZZ"])
    for token in b(text):
        fast.update([token], False)
        slow.update([token], False)
    assert fast.output_text == text
    assert slow.output_text == text
    assert fast.output_token_ids == slow.output_token_ids


# ---------------------------------------------------------------------------
# m13/m14/m21/m22 — OutputProcessor: gates, throttle, fork, external id
# ---------------------------------------------------------------------------


_OP_STATE_SEQ = iter(range(1, 1000))


def make_op_state(op, ext_id, prompt_ids, params, queue):
    req = make_request(ext_id, prompt_ids, params)
    req.external_req_id = ext_id
    req.request_id = f"{ext_id}-{next(_OP_STATE_SEQ):08x}"  # 8-hex internal id
    op.add_request(req, "prompt", None, 0, queue)
    return req.request_id


def test_final_only_constructs_nothing_until_finish():
    async def main():
        op = uplink.OutputProcessor(fast_backend(), log_stats=False, stream_interval=1)
        q = uplink.RequestOutputCollector(uplink.RequestOutputKind.FINAL_ONLY, "int")
        iid = make_op_state(op, "ext-1", b("Hi"), sp(output_kind=uplink.RequestOutputKind.FINAL_ONLY), q)
        res = op.process_outputs([eco(iid, b("hel"))])
        assert res.request_outputs == []  # queue face: list empty...
        assert q.get_nowait() is None  # ...and the collector slot never touched
        res = op.process_outputs([eco(iid, b("lo"), finish=uplink.FinishReason.LENGTH)])
        assert res.request_outputs == []  # queue face: the final rides the collector
        out = q.get_nowait()
        assert out.finished and out.outputs[0].text == "hello"
        assert out.outputs[0].finish_reason == "length"

    run(main)


def test_delta_vs_cumulative_snapshots():
    async def main():
        for kind in (uplink.RequestOutputKind.DELTA, uplink.RequestOutputKind.CUMULATIVE):
            op = uplink.OutputProcessor(
                fast_backend(), log_stats=False, stream_interval=1
            )
            q = uplink.RequestOutputCollector(kind, "int")
            iid = make_op_state(op, "ext-1", b("Hi"), sp(output_kind=kind), q)
            texts, id_lists = [], []
            for token in b("hi!"):
                op.process_outputs([eco(iid, [token])])
                out = q.get_nowait()
                texts.append(out.outputs[0].text)
                id_lists.append(list(out.outputs[0].token_ids))
            if kind == uplink.RequestOutputKind.DELTA:
                assert texts == list("hi!")
                assert id_lists == [[104], [105], [33]]
            else:
                assert texts == ["h", "hi", "hi!"]
                assert id_lists == [[104], [104, 105], [104, 105, 33]]

    run(main)


def test_stream_interval_throttles_but_delta_stays_complete():
    async def main():
        op = uplink.OutputProcessor(fast_backend(), log_stats=False, stream_interval=3)
        q = uplink.RequestOutputCollector(uplink.RequestOutputKind.DELTA, "int")
        iid = make_op_state(op, "ext-1", b("Hi"), sp(), q)
        # engine-level interval 3: request has none -> clamp keeps 3
        state = op.request_states[iid]
        assert state.stream_interval == 3
        step_texts = []
        for ch in "abcdefg":  # 7 tokens: fires at 1st, 4th, 7th
            op.process_outputs([eco(iid, [ord(ch)])])
            if (out := q.get_nowait()) is not None:
                step_texts.append(out.outputs[0].text)
                assert out.finished is False
        assert step_texts == ["a", "bcd", "efg"]
        # concatenation of throttled deltas == the whole text: no loss/overlap
        assert "".join(step_texts) == "abcdefg"

    run(main)


def test_stream_interval_first_token_and_finish_always_fire():
    async def main():
        op = uplink.OutputProcessor(fast_backend(), log_stats=False, stream_interval=5)
        q = uplink.RequestOutputCollector(uplink.RequestOutputKind.DELTA, "int")
        iid = make_op_state(op, "ext-1", b("Hi"), sp(), q)
        op.process_outputs([eco(iid, [65])])  # first token fires
        assert q.get_nowait().outputs[0].text == "A"
        op.process_outputs([eco(iid, [66])])  # below interval: nothing
        assert q.get_nowait() is None
        op.process_outputs(
            [eco(iid, [67], finish=uplink.FinishReason.LENGTH)]
        )  # finish fires
        out = q.get_nowait()
        assert out.finished and out.outputs[0].text == "BC"

    run(main)


def test_per_request_interval_clamped_to_engine_level():
    op = uplink.OutputProcessor(fast_backend(), log_stats=False, stream_interval=4)
    q = uplink.RequestOutputCollector(uplink.RequestOutputKind.DELTA, "int")
    iid = make_op_state(op, "ext-1", b("Hi"), sp(stream_interval=1), q)
    assert op.request_states[iid].stream_interval == 4  # max(1, 4)
    q2 = uplink.RequestOutputCollector(uplink.RequestOutputKind.DELTA, "int2")
    iid2 = make_op_state(op, "ext-2", b("Hi"), sp(stream_interval=9), q2)
    assert op.request_states[iid2].stream_interval == 9  # max(9, 4)


def test_external_id_written_back_and_demux():
    async def main():
        op = uplink.OutputProcessor(fast_backend(), log_stats=False, stream_interval=1)
        q1 = uplink.RequestOutputCollector(uplink.RequestOutputKind.DELTA, "i1")
        q2 = uplink.RequestOutputCollector(uplink.RequestOutputKind.DELTA, "i2")
        i1 = make_op_state(op, "user-id-A", b("Hi"), sp(), q1)
        i2 = make_op_state(op, "user-id-B", b("Hi"), sp(), q2)
        assert op.external_req_ids["user-id-A"] == [i1]
        # one batch carrying BOTH requests: single loop demuxes by internal id
        op.process_outputs([eco(i1, b("a")), eco(i2, b("b")), eco("ghost", b("zz"))])
        assert q1.get_nowait().request_id == "user-id-A"
        assert q2.get_nowait().request_id == "user-id-B"
        # unknown id = already aborted elsewhere: silently skipped
        assert op.get_num_unfinished_requests() == 2

    run(main)


def test_prefill_stats_records_cached_tokens():
    async def main():
        op = uplink.OutputProcessor(fast_backend(), log_stats=False, stream_interval=1)
        q = uplink.RequestOutputCollector(uplink.RequestOutputKind.DELTA, "int")
        iid = make_op_state(op, "ext-1", b("Hi"), sp(), q)
        state = op.request_states[iid]
        assert state.is_prefilling is True
        prefill = SimpleNamespace(num_cached_tokens=5, num_cache_creation_tokens=2)
        op.process_outputs([eco(iid, [65], prefill=prefill)])
        out = q.get_nowait()
        assert out.num_cached_tokens == 5
        assert out.num_cache_creation_tokens == 2
        assert state.is_prefilling is False
        op.process_outputs([eco(iid, [66])])  # later outputs keep the values
        assert q.get_nowait().num_cached_tokens == 5

    run(main)


def test_stop_string_finish_sets_reason_and_requests_engine_abort():
    async def main():
        op = uplink.OutputProcessor(fast_backend(), log_stats=False, stream_interval=1)
        q = uplink.RequestOutputCollector(uplink.RequestOutputKind.DELTA, "int")
        iid = make_op_state(op, "ext-1", b("Hi"), sp(stop=["STOP"]), q)
        # engine has NOT finished (no finish_reason): text-layer stop only
        res = op.process_outputs([eco(iid, b("helloSTOPworld"))])
        out = q.get_nowait()
        assert out.finished is True
        assert out.outputs[0].finish_reason == "stop"
        assert out.outputs[0].stop_reason == "STOP"
        assert out.outputs[0].text == "hello"
        assert res.reqs_to_abort == [iid]  # reverse abort owed to the engine
        assert op.get_num_unfinished_requests() == 0  # state already freed

    run(main)


def test_queue_none_returns_list_llm_engine_face():
    op = uplink.OutputProcessor(fast_backend(), log_stats=False, stream_interval=1)
    req = make_request("ext-1", b("Hi"), sp(output_kind=uplink.RequestOutputKind.CUMULATIVE))
    req.external_req_id = "ext-1"
    op.add_request(req, "prompt", None, 0, None)  # queue=None -> sync face
    res = op.process_outputs([eco(req.request_id, b("ok"))])
    assert [o.outputs[0].text for o in res.request_outputs] == ["ok"]
    assert res.reqs_to_abort == []


def test_finish_frees_all_three_tables():
    async def main():
        op = uplink.OutputProcessor(fast_backend(), log_stats=False, stream_interval=1)
        q = uplink.RequestOutputCollector(uplink.RequestOutputKind.DELTA, "int")
        iid = make_op_state(op, "ext-1", b("Hi"), sp(), q)
        op.process_outputs([eco(iid, [65], finish=uplink.FinishReason.LENGTH)])
        assert op.request_states == {}
        assert dict(op.external_req_ids) == {}
        assert op.parent_requests == {}

    run(main)


# ---------------------------------------------------------------------------
# m15 — n>1 fan-out and ParentRequest aggregation
# ---------------------------------------------------------------------------


def test_add_request_fanout_idx_prefix_and_shared_collector():
    async def main():
        h = Harness(tokenizer=fast_backend())
        try:
            params = sp(n=3)
            q = await h.add("ext-1", b("Hi"), params)
            await wait_for_add(h, 3)
            child_ids = sorted(r.request_id for r in h.add_frames())
            assert all(re.fullmatch(r"[012]_ext-1-[0-9a-f]{8}", cid) for cid in child_ids)
            assert len(h.llm.output_processor.request_states) == 3
            assert len(h.llm.output_processor.parent_requests) == 1
            parent = next(iter(h.llm.output_processor.parent_requests.values()))
            assert parent.child_requests == set(child_ids)
            # children carry n=1 params (fan-out splits the sampling)
            assert all(r.sampling_params.n == 1 for r in h.add_frames())
            # collector keyed by the parent's INTERNAL id; children are idx_ prefixed
            assert child_ids == [f"{i}_{q.request_id}" for i in range(3)]
        finally:
            await h.close()

    run(main)


def test_fanout_seed_clones_per_child_or_caches_shared():
    async def seeded():
        h = Harness(tokenizer=None)
        try:
            await h.add("s", b("Hi"), sp(n=2, seed=42))
            await wait_for_add(h, 2)
            seeds = sorted(r.sampling_params.seed for r in h.add_frames())
            assert seeds == [42, 43]  # seed + index per child
        finally:
            await h.close()

    async def unseeded():
        h2 = Harness(tokenizer=None)
        try:
            await h2.add("n", b("Hi"), sp(n=3))
            await wait_for_add(h2, 3)
            frames = h2.add_frames()
            # no seed: the child params object is cached and REUSED
            assert frames[0].sampling_params is frames[1].sampling_params
        finally:
            await h2.close()

    run(seeded)
    run(unseeded)


def test_parent_streaming_forwards_each_child():
    async def main():
        op = uplink.OutputProcessor(fast_backend(), log_stats=False, stream_interval=1)
        parent_req = make_request("ext-1", b("Hi"), sp(n=2))
        parent_req.external_req_id = "ext-1"
        parent = uplink.ParentRequest(parent_req)
        q = uplink.RequestOutputCollector(uplink.RequestOutputKind.DELTA, parent_req.request_id)
        child0, p0 = parent.get_child_info(0)
        child1, p1 = parent.get_child_info(1)
        req0 = make_request(child0, b("Hi"), p0)
        req0.external_req_id = "ext-1"
        req1 = make_request(child1, b("Hi"), p1)
        req1.external_req_id = "ext-1"
        op.add_request(req0, None, parent, 0, q)
        op.add_request(req1, None, parent, 1, q)
        op.process_outputs([eco(child0, b("aA"))])
        out = q.get_nowait()
        assert out.request_id == "ext-1"
        assert [c.index for c in out.outputs] == [0]
        assert out.finished is False  # child 1 still running
        op.process_outputs([eco(child1, b("bB"))])
        out = q.get_nowait()
        assert [c.index for c in out.outputs] == [1]
        assert out.finished is False

    run(main)


def test_parent_final_only_collects_all_n():
    async def main():
        op = uplink.OutputProcessor(fast_backend(), log_stats=False, stream_interval=1)
        params = sp(n=2, output_kind=uplink.RequestOutputKind.FINAL_ONLY)
        parent_req = make_request("ext-1", b("Hi"), params)
        parent_req.external_req_id = "ext-1"
        parent = uplink.ParentRequest(parent_req)
        q = uplink.RequestOutputCollector(uplink.RequestOutputKind.FINAL_ONLY, parent_req.request_id)
        assert len(parent.output_aggregator) == 2  # preallocated for FINAL_ONLY
        children = []
        for idx in range(2):
            cid, cp = parent.get_child_info(idx)
            req = make_request(cid, b("Hi"), cp)
            req.external_req_id = "ext-1"
            op.add_request(req, None, parent, idx, q)
            children.append(cid)
        op.process_outputs(
            [eco(children[0], b("one"), finish=uplink.FinishReason.LENGTH)]
        )
        assert q.get_nowait() is None  # nothing until all n are in
        op.process_outputs(
            [eco(children[1], b("two"), finish=uplink.FinishReason.LENGTH)]
        )
        out = await q.get()
        assert out.finished is True
        assert sorted(c.text for c in out.outputs) == ["one", "two"]
        assert [c.index for c in out.outputs] == [0, 1]

    run(main)


def test_parent_does_not_reemit_finished_child():
    op_parent_req = make_request("ext-1", b("Hi"), sp(n=2))
    op_parent_req.external_req_id = "ext-1"
    parent = uplink.ParentRequest(op_parent_req)
    c0, _ = parent.get_child_info(0)
    c1, _ = parent.get_child_info(1)
    done = uplink.CompletionOutput(
        index=0, text="x", token_ids=[1], cumulative_logprob=None, logprobs=None,
        finish_reason="length",
    )
    outputs, finished = parent.get_outputs(c0, done)
    assert outputs == [done] and finished is False
    # same child finishes AGAIN (duplicate delivery): not re-emitted
    outputs2, _ = parent.get_outputs(c0, done)
    assert outputs2 == []


# ---------------------------------------------------------------------------
# m18/m22 — abort: dual-track ids, parent cascade, ABORT terminal
# ---------------------------------------------------------------------------


def test_abort_internal_unblocks_consumer_with_abort_terminal():
    async def main():
        op = uplink.OutputProcessor(fast_backend(), log_stats=False, stream_interval=1)
        q = uplink.RequestOutputCollector(uplink.RequestOutputKind.DELTA, "int")
        iid = make_op_state(op, "ext-1", b("Hi"), sp(), q)
        aborted = op.abort_requests([iid], internal=True)
        assert aborted == [iid]
        assert op.request_states == {}  # state removed first...
        assert dict(op.external_req_ids) == {}
        out = await asyncio.wait_for(q.get(), 1)  # ...terminal output unblocks
        assert out.finished and out.outputs[0].finish_reason == "abort"
        assert out.outputs[0].text == ""

    run(main)


def test_abort_external_expands_to_all_internal_ids():
    async def main():
        op = uplink.OutputProcessor(fast_backend(), log_stats=False, stream_interval=1)
        # the same external id reused by a client retry: two internal ids
        q1 = uplink.RequestOutputCollector(uplink.RequestOutputKind.DELTA, "i1")
        q2 = uplink.RequestOutputCollector(uplink.RequestOutputKind.DELTA, "i2")
        i1 = make_op_state(op, "same-ext", b("Hi"), sp(), q1)
        i2 = make_op_state(op, "same-ext", b("Hi"), sp(), q2)
        assert op.external_req_ids["same-ext"] == [i1, i2]
        aborted = op.abort_requests(["same-ext"], internal=False)
        assert sorted(aborted) == sorted([i1, i2])
        assert dict(op.external_req_ids) == {}
        out1 = await asyncio.wait_for(q1.get(), 1)
        assert out1.outputs[0].finish_reason == "abort"
        out2 = await asyncio.wait_for(q2.get(), 1)
        assert out2.finished is True

    run(main)


def test_abort_parent_id_cascades_to_children():
    async def main():
        op = uplink.OutputProcessor(fast_backend(), log_stats=False, stream_interval=1)
        parent_req = make_request("ext-1", b("Hi"), sp(n=2))
        parent_req.external_req_id = "ext-1"
        parent = uplink.ParentRequest(parent_req)
        q = uplink.RequestOutputCollector(uplink.RequestOutputKind.DELTA, parent_req.request_id)
        children = []
        for idx in range(2):
            cid, cp = parent.get_child_info(idx)
            req = make_request(cid, b("Hi"), cp)
            req.external_req_id = "ext-1"
            op.add_request(req, None, parent, idx, q)
            children.append(cid)
        aborted = op.abort_requests([parent_req.request_id], internal=True)
        assert sorted(aborted) == sorted(children)
        assert op.parent_requests == {}
        assert op.request_states == {}

    run(main)


# ---------------------------------------------------------------------------
# m1/m2/m3/m17 — AsyncLLM end to end through real msgpack frames
# ---------------------------------------------------------------------------


def test_queue_task_is_named_enginecoreoutputqueuetask():
    async def main():
        h = Harness(tokenizer=fast_backend())
        try:
            await h.add("ext-1", b("Hi"), sp())
            await wait_for_add(h)
            task = h.client.resources.output_queue_task
            assert task is not None
            assert task.get_name() == "EngineCoreOutputQueueTask"
        finally:
            await h.close()

    run(main)


def test_generate_streams_deltas_to_external_id():
    async def main():
        h = Harness(tokenizer=fast_backend())
        try:
            got = asyncio.Queue()
            gen = h.llm.generate(make_request("ext-9", b("Hi"), sp()), sp(), "ext-9")

            async def consume():
                async for out in gen:
                    await got.put(out)

            task = asyncio.ensure_future(consume())
            await wait_for_add(h)
            iid = h.add_frames()[0].request_id
            assert re.fullmatch(r"ext-9-[0-9a-f]{8}", iid)
            for ch in "eli":  # 'H' came with... feed three decode steps
                h.feed([eco(iid, [ord(ch)])])
                out = await asyncio.wait_for(got.get(), 1)
                assert out.request_id == "ext-9"  # external id written back
                assert out.outputs[0].text == ch
                assert out.outputs[0].finish_reason is None
            h.feed([eco(iid, [33], finish=uplink.FinishReason.LENGTH)])  # '!'
            out = await asyncio.wait_for(got.get(), 1)
            assert out.finished and out.outputs[0].finish_reason == "length"
            assert out.outputs[0].text == "!"
            await task
            assert h.llm.output_processor.request_states == {}
        finally:
            await h.close()

    run(main)


def test_generate_final_only_e2e_single_output():
    async def main():
        h = Harness(tokenizer=fast_backend())
        try:
            params = sp(output_kind=uplink.RequestOutputKind.FINAL_ONLY)
            got = asyncio.Queue()
            gen = h.llm.generate(make_request("ext-1", b("Hi"), params), params, "ext-1")

            async def consume():
                async for out in gen:
                    await got.put(out)

            task = asyncio.ensure_future(consume())
            await wait_for_add(h)
            iid = h.add_frames()[0].request_id
            h.feed([eco(iid, b("not yet"))])  # intermediate: swallowed whole
            h.feed([eco(iid, b(" done"), finish=uplink.FinishReason.LENGTH)])
            out = await asyncio.wait_for(got.get(), 1)
            assert out.outputs[0].text == "not yet done"  # one final snapshot
            await task
        finally:
            await h.close()

    run(main)


def test_output_handler_chunks_batch_and_sleeps_between_slices(monkeypatch):
    async def main():
        monkeypatch.setattr(
            uplink.envs, "VLLM_V1_OUTPUT_PROC_CHUNK_SIZE", 2
        )
        assert uplink.envs.VLLM_V1_OUTPUT_PROC_CHUNK_SIZE == 2
        h = Harness(tokenizer=None)
        try:
            params = sp(output_kind=uplink.RequestOutputKind.CUMULATIVE)
            queues = [await h.add(f"ext-{i}", b("Hi"), params) for i in range(5)]
            await wait_for_add(h, 5)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            events = []
            real = h.llm.output_processor.process_outputs

            def spy(slice_outputs, *a, **k):
                events.append(("proc", len(slice_outputs)))
                return real(slice_outputs, *a, **k)

            h.llm.output_processor.process_outputs = spy

            async def watcher():
                events.append(("watch",))  # records as soon as it is scheduled

            h.feed(
                [
                    eco(h.add_frames()[i].request_id, [65 + i])
                    for i in range(5)
                ]
            )
            # wait for the FIRST slice, then arm the watcher: it can only run
            # while the handler is parked somewhere. If the handler yields
            # between slices (asyncio.sleep(0)), the watcher lands BETWEEN
            # two proc events; without the yield it lands after the batch.
            for _ in range(100):
                if any(e[0] == "proc" for e in events):
                    break
                await asyncio.sleep(0)
            watcher_task = asyncio.create_task(watcher())
            for _ in range(100):
                if len([e for e in events if e[0] == "proc"]) == 3:
                    break
                await asyncio.sleep(0)
            procs = [e for e in events if e[0] == "proc"]
            assert procs == [("proc", 2), ("proc", 2), ("proc", 1)]  # 128 -> 2
            proc_idx = [i for i, e in enumerate(events) if e[0] == "proc"]
            watch_idx = [i for i, e in enumerate(events) if e[0] == "watch"]
            assert proc_idx[0] < watch_idx[0] < proc_idx[-1]
            await watcher_task
        finally:
            await h.close()
            monkeypatch.undo()

    run(main)


def test_eager_start_inside_loop_lazy_outside():
    async def main():
        # inside a running loop: __init__ starts the handler eagerly
        h = Harness(tokenizer=None)
        try:
            assert h.llm.output_handler is not None
            assert h.client.resources.output_queue_task is not None
        finally:
            await h.close()

    run(main)

    # outside any loop: nothing starts until the first add_request
    h2 = Harness(tokenizer=None)

    async def main2():
        try:
            assert h2.llm.output_handler is None
            assert h2.client.resources.output_queue_task is None
            await h2.add("ext-1", b("Hi"), sp())
            assert h2.llm.output_handler is not None  # lazy start (L390-393)
        finally:
            await h2.close()

    assert h2.llm.output_handler is None
    run(main2)


def test_multi_request_batch_demuxes_to_own_collectors():
    async def main():
        h = Harness(tokenizer=fast_backend())
        try:
            qs = {}
            for ext in ("a", "b"):
                qs[ext] = await h.add(ext, b("Hi"), sp())
            await wait_for_add(h, 2)
            iids = [r.request_id for r in h.add_frames()]
            h.feed([eco(iids[0], b("one")), eco(iids[1], b("two"))])
            out_a = await asyncio.wait_for(qs["a"].get(), 1)
            out_b = await asyncio.wait_for(qs["b"].get(), 1)
            assert out_a.request_id == "a" and out_a.outputs[0].text == "one"
            assert out_b.request_id == "b" and out_b.outputs[0].text == "two"
        finally:
            await h.close()

    run(main)


# ---------------------------------------------------------------------------
# m1/m20 — death & error contracts ride the same queue
# ---------------------------------------------------------------------------


def test_dead_sentinel_raises_engine_dead_without_abort():
    async def main():
        h = Harness(tokenizer=fast_backend())
        try:
            got = asyncio.Queue()
            gen = h.llm.generate(make_request("ext-1", b("Hi"), sp()), sp(), "ext-1")

            async def consume():
                async for out in gen:
                    await got.put(out)

            task = asyncio.ensure_future(consume())
            await wait_for_add(h)
            h.feed_dead()
            with pytest.raises(uplink.EngineDeadError):
                await task
            assert h.client.resources.engine_dead is True
            assert h.abort_frames() == []  # dead engine: do NOT abort
            # states stay registered (propagate_error only wakes consumers);
            # the errored AsyncLLM now refuses new work
            with pytest.raises(uplink.EngineDeadError):
                await h.add("ext-2", b("Hi"), sp())
        finally:
            await h.close()

    run(main)


def test_socket_exception_reaches_every_consumer():
    async def main():
        h = Harness(tokenizer=fast_backend())
        try:
            gens = []
            tasks = []
            for ext in ("a", "b"):
                gen = h.llm.generate(make_request(ext, b("Hi"), sp()), sp(), ext)
                gens.append(gen)
                tasks.append(asyncio.ensure_future(drain(gen)))
            await wait_for_add(h, 2)
            h.socket.feed_exception(RuntimeError("boom"))
            for task in tasks:
                with pytest.raises(uplink.EngineGenerateError) as ei:
                    await task
                assert isinstance(ei.value.__cause__, RuntimeError)
            assert h.llm.output_processor.request_states == {}
        finally:
            await h.close()

    run(main)


async def drain(gen):
    async for _ in gen:
        pass


def test_queue_task_cancelled_feeds_engine_dead():
    async def main():
        h = Harness(tokenizer=fast_backend())
        try:
            got = asyncio.Queue()
            gen = h.llm.generate(make_request("ext-1", b("Hi"), sp()), sp(), "ext-1")
            task = asyncio.ensure_future(collect_into(gen, got))
            await wait_for_add(h)
            h.client.resources.output_queue_task.cancel()
            with pytest.raises(uplink.EngineDeadError):
                await task
        finally:
            await h.close()

    run(main)


async def collect_into(gen, q):
    async for out in gen:
        await q.put(out)


# ---------------------------------------------------------------------------
# m18/m19 — reverse aborts: stop-string & client disconnect
# ---------------------------------------------------------------------------


def test_stop_string_reverse_abort_e2e():
    async def main():
        h = Harness(tokenizer=fast_backend())
        try:
            params = sp(stop=["STOP"])
            got = asyncio.Queue()
            gen = h.llm.generate(make_request("ext-1", b("Hi"), params), params, "ext-1")
            task = asyncio.ensure_future(collect_into(gen, got))
            await wait_for_add(h)
            iid = h.add_frames()[0].request_id
            # engine did NOT finish: the stop lives only in the text layer
            h.feed([eco(iid, b("hiSTOPthere"))])
            out = await asyncio.wait_for(got.get(), 1)
            assert out.finished and out.outputs[0].text == "hi"
            assert out.outputs[0].stop_reason == "STOP"
            for _ in range(50):  # handler must issue the reverse abort
                if h.abort_frames():
                    break
                await asyncio.sleep(0)
            assert h.abort_frames() == [[iid]]  # engine told to stop computing
            await task
        finally:
            await h.close()

    run(main)


def test_disconnect_cancels_generate_and_aborts_engine():
    async def main():
        h = Harness(tokenizer=fast_backend())
        try:
            got = asyncio.Queue()
            gen = h.llm.generate(make_request("ext-1", b("Hi"), sp()), sp(), "ext-1")
            task = asyncio.ensure_future(collect_into(gen, got))
            await wait_for_add(h)
            iid = h.add_frames()[0].request_id
            h.feed([eco(iid, [72])])
            out = await asyncio.wait_for(got.get(), 1)
            assert out.outputs[0].text == "H"
            await asyncio.sleep(0.01)  # let generate() suspend on q.get()
            task.cancel()  # the client went away
            with pytest.raises(asyncio.CancelledError):
                await task
            for _ in range(50):
                if h.abort_frames():
                    break
                await asyncio.sleep(0)
            # two hops: local state removed + cross-process ABORT frame
            assert h.abort_frames() == [[iid]]
            assert h.llm.output_processor.request_states == {}
        finally:
            await h.close()

    run(main)


# ---------------------------------------------------------------------------
# Wire structs & entry registration details
# ---------------------------------------------------------------------------


def test_engine_core_outputs_round_trip_real_msgpack():
    msg = uplink.EngineCoreOutputs(
        engine_index=2,
        outputs=[eco("int-1", b("ok"), finish=uplink.FinishReason.STOP, stop_reason=7)],
        timestamp=0.5,
    )
    payload = uplink.msgspec.msgpack.encode(msg)
    decoded = uplink.msgspec.msgpack.decode(payload, type=uplink.EngineCoreOutputs)
    assert decoded.outputs[0].request_id == "int-1"
    assert decoded.outputs[0].new_token_ids == b("ok")
    assert decoded.outputs[0].finish_reason == uplink.FinishReason.STOP
    assert decoded.outputs[0].stop_reason == 7
    assert decoded.engine_index == 2
    assert decoded.outputs[0].finished is True


def test_engine_core_outputs_timestamp_defaults_to_monotonic():
    msg = uplink.EngineCoreOutputs()
    assert msg.timestamp > 0.0  # __post_init__ fills monotonic time


def test_add_request_assigns_dual_track_ids():
    async def main():
        h = Harness(tokenizer=None)
        try:
            await h.add("user-42", b("Hi"), sp())
            await wait_for_add(h)
            iid = h.add_frames()[0].request_id
            assert re.fullmatch(r"user-42-[0-9a-f]{8}", iid)  # 8 hex chars
            state = h.llm.output_processor.request_states[iid]
            assert state.external_req_id == "user-42"
        finally:
            await h.close()

    run(main)


def test_preset_external_req_id_rejected():
    req = make_request("ext-1", b("Hi"), sp())
    req.external_req_id = "ext-1"  # must not be pre-set by callers
    with pytest.raises(ValueError, match="external_req_id"):
        uplink.InputProcessor.assign_request_id(req)


def test_finish_reason_strings_are_the_external_api():
    assert (
        uplink.FinishReason.STOP == 0
        and uplink.FinishReason.LENGTH == 1
        and uplink.FinishReason.ABORT == 2
    )
    assert str(uplink.FinishReason.STOP) == "stop"
    assert str(uplink.FinishReason.ABORT) == "abort"
    assert uplink.RequestOutputKind.CUMULATIVE.value == 0
    assert uplink.RequestOutputKind.DELTA.value == 1
    assert uplink.RequestOutputKind.FINAL_ONLY.value == 2


def test_chunk_size_default_is_128():
    assert uplink.envs.VLLM_V1_OUTPUT_PROC_CHUNK_SIZE == 128
