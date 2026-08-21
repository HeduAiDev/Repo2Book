"""Driver for m4 (BaseIncrementalDetokenizer.update main flow) — host run
against the ch07 subtract-only companion (pin vLLM v0.27.1).

One small Fast detokenizer per scenario (byte-level WordLevel tokenizer: one
token per raw UTF-8 byte, so token id == byte value and readers can verify
every step by hand). Records the update() main flow rounds:
- scenario A (main line, stop=["END"], include=False): accumulate 2 rounds ->
  3rd round completes the stop IN-TEXT -> same-call truncation to the stop
  start + stop_string returned;
- scenario B (stop-token skip, stop_terminated=True): the popped stop token
  is absent from output_text but present in token_ids (two ledgers);
- scenario C (min_tokens=3 guard, stop=["AB"]): the stop completing INSIDE
  the guard window is swallowed (stop_check_offset pushed past it), and the
  first post-guard token cannot see it either (new_char_count window too
  short); a stop completing AFTER the guard is detected and truncated.
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


async def main():
    out = {
        "driver": "run_m4_update_flow.py",
        "mechanism": "m4 update 主流程（跳 stop token → 逐 token decode_next → min_tokens 推进 stop_check_offset → check_stop_strings 截断）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "tokenizer": "Fast 路径（真 Rust DecodeStream）：byte-level WordLevel，1 token = 1 字节，token id == 字节值（65='A' 66='B' 67='C' 68='D' 69='E' 78='N' 120='x'）",
        "scenario_A": {"prompt": "Hi", "stop": ["END"], "include_stop_str_in_output": False, "min_tokens": 0},
        "scenario_B": {"prompt": "Hi", "stop": [], "stop_terminated": True, "include_stop_str_in_output": False},
        "scenario_C": {"prompt": "Hi", "stop": ["AB"], "min_tokens": 3, "include_stop_str_in_output": False},
    }

    # ---- scenario A: accumulate -> stop completes in-text -> truncate --------
    dA = td.fast_detok(prompt="Hi", stop=["END"], include_stop_str_in_output=False)
    rounds_A = []
    for label, ids, terminated in [
        ("round1", [65, 66], False),   # "AB"
        ("round2", [69, 78], False),   # "EN"
        ("round3", [68], False),       # "D" completes "END" at index 2
    ]:
        before_len = len(dA.output_text)
        stop_string = dA.update(ids, terminated)
        rounds_A.append({
            "round": label,
            "new_token_ids": list(ids),
            "decoded_chars": "".join(chr(i) for i in ids),
            "output_text_before": None if before_len == 0 and not rounds_A else dA.output_text[: before_len] if rounds_A else "",
            "output_text_after": dA.output_text,
            "token_ids_ledger": list(dA.token_ids),
            "stop_string_returned": stop_string,
            "truncated_same_call": len(dA.output_text) < before_len + len(ids),
        })
    # fix output_text_before (first round starts from "")
    rounds_A[0]["output_text_before"] = ""
    out["A_rounds"] = rounds_A
    out["A_stop_buffer_length"] = dA.stop_buffer_length  # max(len("END"))-1 = 2
    out["A_final"] = {
        "output_text_len": len(dA.output_text),           # 2 ("AB")
        "num_output_tokens": dA.num_output_tokens(),       # 5 ids kept
        "text_and_ids_diverge": len(dA.output_text) != len(dA.token_ids),
    }

    # ---- scenario B: stop-token skip keeps two ledgers apart -----------------
    dB = td.fast_detok(prompt="Hi", stop=[])
    stop_string_B = dB.update([67, 69], True)  # stop_terminated, include=False
    out["B_round"] = {
        "round": "round4",
        "new_token_ids": [67, 69],
        "stop_terminated": True,
        "include_stop_str_in_output": False,
        "skipped_stop_token_id": 69,
        "output_text_after": dB.output_text,          # "C" only
        "token_ids_ledger": list(dB.token_ids),       # [67, 69] both
        "text_ledger_len": len(dB.output_text),       # 1
        "id_ledger_len": len(dB.token_ids),           # 2
        "stop_string_returned": stop_string_B,        # None
    }

    # ---- scenario C: min_tokens guard swallows / post-guard detects ----------
    dC = td.fast_detok(prompt="Hi", stop=["AB"], min_tokens=3)
    rounds_C = []
    for label, ids in [
        ("round5", [65, 66]),  # "AB" completes stop but only 2 tokens <= 3
        ("round6", [67]),      # 3 tokens: still <= min 3
        ("round7", [68]),      # 4 tokens > 3, but "AB" outside the window
        ("round8", [65, 66]),  # new "AB" after the guard -> detected
    ]:
        stop_string = dC.update(ids, False)
        rounds_C.append({
            "round": label,
            "new_token_ids": list(ids),
            "output_text_after": dC.output_text,
            "num_output_tokens": dC.num_output_tokens(),
            "min_tokens": 3,
            "stop_string_returned": stop_string,
        })
    out["C_rounds"] = rounds_C
    out["C_final"] = {
        "output_text_after_truncation": dC.output_text,  # "ABCD" -> truncated to stop start
        "output_text_len": len(dC.output_text),          # 4
        "num_output_tokens": dC.num_output_tokens(),     # 6 ids kept
    }

    dest = Path(__file__).resolve().parent / "m4_update_flow.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {dest}")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
