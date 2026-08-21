"""Driver for m7 (Slow path: prefix/read double-offset window) — host run
against the ch07 companion (pin vLLM v0.27.1).

The SlowByteTokenizer seam is byte-level (one token = one raw byte, ids
latin-1-mapped, convert_tokens_to_string joins then UTF-8-decodes with
replacement) — exactly the byte-fallback surface the real Slow path sees
from an HF tokenizer; the window arithmetic under test is the verbatim pin
code. Records:
- initial window: convert_prompt_ids_to_tokens on a 10-token prompt keeps
  only the LAST 7 tokens (INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET 5 + 2),
  read_offset 7, prefix_offset 2;
- per-token decode rounds: window slice [prefix:read] -> prefix_text,
  full decode [prefix:] -> new_text, delta = new_text minus prefix_text,
  and the offsets handed to the next round (prefix <- old read,
  read <- len(output_tokens)); the per-step decode span stays bounded
  (read window + new token), never the whole sequence;
- the counterfactual cost of re-decoding the whole sequence every step.
"""
import asyncio
import importlib
import json
import sys
from pathlib import Path

_CH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_CH / "tests"))
td = importlib.import_module("test_uplink")
uplink = td.uplink


def window_span(detok):
    """Tokens the next convert_tokens_to_string pair will touch:
    [prefix_offset:] plus the incoming token."""
    return len(detok.token_ids) - detok.prefix_offset + 1


async def main():
    out = {
        "driver": "run_m7_slow_window.py",
        "mechanism": "m7 Slow 路径：prefix/read 双 offset 窗口对抗 decode 的空格 cleanup",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "tokenizer": "SlowByteTokenizer（HOST SEAM：byte 级 convert_ids_to_tokens / convert_tokens_to_string=latin-1→utf-8 replace）——窗口算术与判定为 pin 逐字代码",
        "prompt": "abcdefghij",
        "prompt_token_ids": td.b("abcdefghij"),
        "prompt_len": 10,
    }

    # ---- initial window -------------------------------------------------------
    tok = td.SlowByteTokenizer()
    tokens, prefix, read = uplink.convert_prompt_ids_to_tokens(
        tok, td.b("abcdefghij"), skip_special_tokens=True
    )
    out["initial_window"] = {
        "INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET": uplink.INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET,
        "prompt_len": len(td.b("abcdefghij")),
        "converted_token_count": len(tokens),
        "converted_tokens_are_prompt_tail": "".join(tokens) == "defghij",
        "read_offset": read,
        "prefix_offset": prefix,
        "prefix_back_from_read": read - prefix,
        "note": "只转 prompt 尾部 7 个 token（OFFSET 5 + 2），prefix 再退 5",
    }

    # ---- per-token decode rounds (update-driven, as the real flow runs) ------
    d = td.slow_detok(prompt="abcdefghij")
    rounds = []
    for tid in [107, 108, 109, 110, 111]:  # 'k'..'o'
        prefix_before = d.prefix_offset
        read_before = d.read_offset
        window_slice = "".join(d.tokens[prefix_before:read_before])
        text_before = d.output_text
        d.update([tid], False)
        delta = d.output_text[len(text_before):]
        out_tokens_total = len(d.tokens)
        rounds.append({
            "round": len(rounds) + 1,
            "token_id": tid,
            "char": chr(tid),
            "prefix_offset_before": prefix_before,
            "read_offset_before": read_before,
            "window_slice": window_slice,
            "window_slice_len": len(window_slice),
            "delta_returned": delta,
            "output_text_after": d.output_text,
            "prefix_offset_after": d.prefix_offset,
            "read_offset_after": d.read_offset,
            "decode_span_this_step": out_tokens_total - prefix_before,
            "sequence_len_so_far": out_tokens_total,
        })
    out["rounds"] = rounds
    out["bounded_span"] = {
        "sequence_len_final": len(d.tokens),
        "first_step_span": rounds[0]["decode_span_this_step"],
        "steady_state_span": rounds[1]["decode_span_this_step"],
        "max_decode_span_observed": max(r["decode_span_this_step"] for r in rounds),
        "full_redecode_span_would_be": len(d.tokens),
        "output_text_final": d.output_text,
        "num_output_tokens": d.num_output_tokens(),
        "note": "每步 decode 只触 [prefix:] 尾窗（首步盖初始窗、稳态=上下文1 token+新1 token），非全序列重解",
    }

    dest = Path(__file__).resolve().parent / "m7_slow_window.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    asyncio.run(main())
