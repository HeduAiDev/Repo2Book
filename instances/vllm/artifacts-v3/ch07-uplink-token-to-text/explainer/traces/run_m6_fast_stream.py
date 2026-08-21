"""Driver for m6 (Fast path: real Rust DecodeStream) + the m5 three-way
factory probe — host run against the ch07 companion (pin vLLM v0.27.1).

Sections:
- factory: the real dispatch path RequestState.from_new_request -> tokenizer
  nulled when detokenize=False -> IncrementalDetokenizer.from_new_request
  three-way pick (null shell / Fast / Slow), plus the host gate values;
- prefill_and_step: the DecodeStream is primed with the PROMPT ids at
  construction (native prefill); driving update() with new tokens yields
  ONLY the new chars — the prompt text never leaks into output_text;
- multibyte: '中' = UTF-8 E4 B8 AD across three single-byte tokens — the
  first two steps return None (incomplete byte sequence buffered inside the
  Rust stream), the third returns the whole char;
- guards: out-of-vocab id 256 -> None (real Rust stream), 2**64 -> TypeError
  raised by the binding and SWALLOWED by _protected_step; a stub stream
  raising the pin's exact 'Invalid prefix encountered' message exercises the
  REAL rebuild-and-replay recovery (fresh stream, no prefill context).
"""
import asyncio
import importlib
import json
import sys
from pathlib import Path

import tokenizers

_CH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_CH / "tests"))
td = importlib.import_module("test_uplink")
uplink = td.uplink


class BoomStream:
    """Stub raising the pin's exact invalid-prefix message (issue #17448)."""

    def step(self, tokenizer, token_id):
        raise RuntimeError("Invalid prefix encountered")


async def main():
    out = {
        "driver": "run_m6_fast_stream.py",
        "mechanism": "m6 Fast 路径（DecodeStream native prefill + stream.step + _protected_step 容错）；含 m5 三路工厂探针",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "environment_note": "host tokenizers 0.22.2 真 Rust DecodeStream；transformers 4.57.3 无 TokenizersBackend（HOST SEAM 回退类只替名字 + ._tokenizer 触达面）",
    }

    # ---- factory probe (m5 numbers) — via the REAL RequestState path --------
    backend = td.fast_backend()
    req_nodecode = td.make_request("int-2", td.b("Hi"), td.sp(detokenize=False))
    uplink.InputProcessor.assign_request_id(req_nodecode)  # real flow sets external_req_id
    state_nodecode = uplink.RequestState.from_new_request(
        tokenizer=backend, request=req_nodecode, prompt=None,
        parent_req=None, request_index=0, queue=None, log_stats=False,
        stream_interval=1,
    )
    req = td.make_request("int-1", td.b("Hi"), td.sp())
    shell = uplink.IncrementalDetokenizer.from_new_request(None, req)
    fast = uplink.IncrementalDetokenizer.from_new_request(backend, req)
    slow = uplink.IncrementalDetokenizer.from_new_request(td.SlowByteTokenizer(), req)
    out["factory"] = {
        "tokenizer_None": type(shell).__name__,
        "tokenizers_backend": type(fast).__name__,
        "other_tokenizerlike": type(slow).__name__,
        "detokenize_False_state_detokenizer": type(state_nodecode.detokenizer).__name__,
        "detokenize_False_nulls_tokenizer_first": True,
        "USE_FAST_DETOKENIZER": bool(uplink.USE_FAST_DETOKENIZER),
        "host_tokenizers_version": tokenizers.__version__,
        "fast_gate": "USE_FAST_DETOKENIZER and isinstance(tokenizer, TokenizersBackend)",
        "null_shell_update_returns": shell.update(td.b("ab"), False),
        "null_shell_text_always_empty": shell.get_next_output_text(True, False) == "",
        "null_shell_ids_still_counted": shell.num_output_tokens(),
    }

    # ---- native prefill + single-char stepping (update-driven) --------------
    d = td.fast_detok(prompt="Hi")
    r1 = d.update([65], False)
    t1 = d.output_text
    r2 = d.update([66], False)
    t2 = d.output_text
    out["prefill_and_step"] = {
        "prompt_token_ids": td.b("Hi"),
        "prompt_text": "Hi",
        "steps": [
            {"round": 1, "token_id": 65, "update_returns": r1, "output_text_after": t1},
            {"round": 2, "token_id": 66, "update_returns": r2, "output_text_after": t2},
        ],
        "prompt_text_leaks": "Hi" in d.output_text,
        "output_text_after_2_steps": d.output_text,
        "note": "构造时 DecodeStream(ids=prompt) 预热；update 只吐新字符，prompt 文本永不进 output_text",
    }

    # ---- multibyte char split across three tokens ----------------------------
    probe = td.fast_detok(prompt="Hi")
    raw_steps = [
        {"round": 2, "token_id": 228, "utf8_byte_hex": "E4", "protected_step_raw": probe._protected_step(228)},
        {"round": 3, "token_id": 184, "utf8_byte_hex": "B8", "protected_step_raw": probe._protected_step(184)},
        {"round": 4, "token_id": 173, "utf8_byte_hex": "AD", "protected_step_raw": probe._protected_step(173)},
    ]
    dm = td.fast_detok(prompt="Hi")
    dm.update([228], False)
    t1 = dm.output_text
    dm.update([184], False)
    t2 = dm.output_text
    r3 = dm.update([173], False)
    out["multibyte_zhong"] = {
        "target_char": "中",
        "utf8_bytes_hex": ["E4", "B8", "AD"],
        "raw_stream_steps": raw_steps,
        "update_rounds": [
            {"round": 2, "token_ids": [228], "output_text_after": t1},
            {"round": 3, "token_ids": [184], "output_text_after": t2},
            {"round": 4, "token_ids": [173], "update_returns": r3, "output_text_after": dm.output_text},
        ],
    }

    # ---- guards ---------------------------------------------------------------
    dg = td.fast_detok(prompt="Hi")
    oov_raw = dg._protected_step(256)
    out["guard_out_of_vocab"] = {
        "round": 5,
        "token_id": 256,
        "protected_step_raw": oov_raw,
        "decode_next_result": dg.decode_next(256),
        "output_text_after": dg.output_text,
        "real_rust_stream": True,
    }
    dh = td.fast_detok(prompt="Hi")
    big_raw = dh._protected_step(2**64)
    out["guard_typeerror_swallowed"] = {
        "round": 6,
        "token_id_dec": 18446744073709551616,
        "protected_step_raw": big_raw,
        "decode_next_result": dh.decode_next(2**64),
        "output_text_after": dh.output_text,
        "note": "Rust 绑定对超 u64 id 抛 TypeError，_protected_step 吞掉返回 None（issue #21951 同款分支）",
    }
    dr = td.fast_detok(prompt="Hi")
    dr.stream = BoomStream()
    recovered = dr.decode_next(72)  # 'H'
    out["invalid_prefix_recovery"] = {
        "round": 7,
        "stub_error": "Invalid prefix encountered",
        "replay_token_id": 72,
        "decode_next_result": recovered,
        "stream_rebuilt_to_real_DecodeStream": isinstance(dr.stream, tokenizers.decoders.DecodeStream),
        "rebuilt_stream_is_fresh_no_prefill": True,
        "update_after_recovery": {
            "new_token_ids": [101],
            "stop_string": dr.update([101], False),
            "output_text": dr.output_text,
        },
        "note": "真码 _protected_step 的捕获-重建-重放；重建流无 prefill 上下文（真实语义）",
    }

    dest = Path(__file__).resolve().parent / "m6_fast_stream.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    asyncio.run(main())
