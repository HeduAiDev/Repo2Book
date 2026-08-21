"""Driver for m9 (stop-string holdback + get_next_output_text delta slicing)
— host run against the ch07 companion (pin vLLM v0.27.1).

Scenario A (stop=["END"], include=False): stop_buffer_length = max(len)-1 = 2.
While unfinished, the delta cut is len(output_text)-2 — so when the stop
string's PREFIX ("EN" of "END") reaches the tail it stays held back; when the
final byte completes it, update() truncates output_text to the stop start in
the SAME call, and the finished delta is computed from the truncated text.
Net: the consumer's concatenated deltas equal exactly the truncated text —
not one byte of "END" ever leaks. (Counterfactual computed for contrast:
holdback=0 would have leaked "EN" one round early.)
Scenario B (include_stop_str_in_output=True): holdback=0, nothing held; the
stop string is part of the deliverable; on hit no truncation (end >= len).
Scenario C (no stop strings at all): holdback=0 — the holdback is a cost paid
ONLY when exclusive stop strings exist.
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


def drive(detok, rounds):
    """Each round: update(ids) then a DELTA read (unfinished except last)."""
    emitted = ""
    rows = []
    for ids, finished in rounds:
        text_before = detok.output_text
        stop_string = detok.update(ids, False)
        delta = detok.get_next_output_text(finished, True)
        emitted += delta
        rows.append({
            "new_token_ids": list(ids),
            "output_text_after_update": detok.output_text,
            "stop_string_returned": stop_string,
            "finished_read": finished,
            "holdback_applied": 0 if finished else detok.stop_buffer_length,
            "last_output_text_offset_after": detok._last_output_text_offset,
            "delta_returned": delta,
            "cumulative_emitted": emitted,
        })
    return rows, emitted


async def main():
    out = {
        "driver": "run_m9_holdback.py",
        "mechanism": "m9 stop-string holdback：stop_buffer_length=max(len(s))-1 尾字符扣留 + get_next_output_text delta 切片",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "tokenizer": "Fast 路径 byte 级（1 token=1 字节）；update 与取文本同一拍内完成（make_request_output 的真实时序）",
    }

    # ---- scenario A: exclusive stop, holdback 2 ------------------------------
    dA = td.fast_detok(prompt="Hi", stop=["END"], include_stop_str_in_output=False)
    rounds, emitted = drive(
        dA,
        [([65, 66, 67], False), ([69], False), ([78], False), ([68], True)],
    )
    out["A_exclusive_stop"] = {
        "stop": ["END"],
        "stop_len": 3,
        "stop_buffer_length": dA.stop_buffer_length,
        "rounds": rounds,
        "emitted_total": emitted,
        "emitted_equals_truncated_text": emitted == dA.output_text,
        "stop_bytes_leaked": ("E" in emitted[len("ABC"):] if emitted.startswith("ABC") else True),
        "counterfactual_holdback_0_leak": "CEN",  # round 3 with buffer 0 would emit [2:5)
        "final_output_text": dA.output_text,
    }

    # ---- scenario B: include stop string in output ---------------------------
    dB = td.fast_detok(prompt="Hi", stop=["END"], include_stop_str_in_output=True)
    rounds_b, emitted_b = drive(
        dB,
        [([65, 66, 67], False), ([69], False), ([78], False), ([68], True)],
    )
    out["B_include_stop"] = {
        "stop": ["END"],
        "include_stop_str_in_output": True,
        "stop_buffer_length": dB.stop_buffer_length,
        "rounds": rounds_b,
        "emitted_total": emitted_b,
        "final_output_text": dB.output_text,
        "stop_string_included": emitted_b == "ABCEND",
    }

    # ---- scenario C: no stop strings -----------------------------------------
    dC = td.fast_detok(prompt="Hi", stop=())
    rounds_c, emitted_c = drive(dC, [([65, 66], False), ([67], True)])
    out["C_no_stop"] = {
        "stop": [],
        "stop_buffer_length": dC.stop_buffer_length,
        "rounds": rounds_c,
        "emitted_total": emitted_c,
        "note": "无 stop 串即零扣留——holdback 只在有『要排除的 stop 串』时才付",
    }

    dest = Path(__file__).resolve().parent / "m9_holdback.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    asyncio.run(main())
