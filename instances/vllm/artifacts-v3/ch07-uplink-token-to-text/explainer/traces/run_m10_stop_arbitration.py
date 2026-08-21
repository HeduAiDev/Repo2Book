"""Driver for m10 (check_stop_strings windowed find + earliest-completion
arbitration) — host run against the ch07 companion (pin vLLM v0.27.1).

Calls the verbatim function directly; also records what each candidate stop
string saw (index/end) for the table. Cases:
- multi-token step where two stops both match: the one that COMPLETES
  earliest wins even if it comes later in the stop list (v0.27.x semantics
  for speculative decoding's multi-token steps);
- tie (same completion end): broken by stop-list order — and the two orders
  produce DIFFERENT truncation points;
- window lower bound find(text, 1-new_char_count-L): an occurrence entirely
  in already-searched text is not re-found; new_char_count=0 short-circuits;
- include_in_output: truncate to stop END (or -1 = no truncation when the
  stop ends at the text tail) vs truncate to stop START when excluded.
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


def candidates(text, new_char_count, stop):
    """Observation-only mirror of the find loop (winner comes from the real fn)."""
    seen = []
    for s in stop:
        idx = text.find(s, 1 - new_char_count - len(s))
        seen.append({
            "stop_str": s,
            "find_start_param": 1 - new_char_count - len(s),
            "find_start_absolute": max(0, 1 - new_char_count - len(s)) if idx != -1 else None,
            "stop_index": idx,
            "end": idx + len(s) if idx != -1 else None,
        })
    return seen


async def main():
    out = {
        "driver": "run_m10_stop_arbitration.py",
        "mechanism": "m10 check_stop_strings 窗口查找 + 完成最早仲裁",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "fn": "check_stop_strings(output_text, new_char_count, stop, include_in_output) — verbatim pin code",
    }

    # ---- case 1: earliest completion beats list order ------------------------
    text1 = "XENDSTOP!"
    stop1 = ["STOP!", "END"]  # list order would try STOP! first
    res1 = uplink.check_stop_strings(text1, len(text1), stop1, False)
    out["case1_earliest_completion"] = {
        "round": 1,
        "output_text": text1,
        "new_char_count": len(text1),
        "stop_list": stop1,
        "candidates": candidates(text1, len(text1), stop1),
        "winner": res1[0] if res1 else None,
        "truncate_to": res1[1] if res1 else None,
        "truncated_text": text1[: res1[1]] if res1 else None,
        "note": "END 完成于 4、STOP! 完成于 9——完成最早者胜，与 stop 列表序无关（投机解码一步多 token 语义）",
    }

    # ---- case 2: tie broken by list order (two orders, two truncations) ------
    text2 = "xxab"
    res2a = uplink.check_stop_strings(text2, 4, ["b", "ab"], False)
    res2b = uplink.check_stop_strings(text2, 4, ["ab", "b"], False)
    out["case2_tie_list_order"] = {
        "round": 2,
        "output_text": text2,
        "new_char_count": 4,
        "order_b_first": {
            "stop_list": ["b", "ab"],
            "candidates": candidates(text2, 4, ["b", "ab"]),
            "winner": res2a[0],
            "truncate_to": res2a[1],
            "truncated_text": text2[: res2a[1]],
        },
        "order_ab_first": {
            "stop_list": ["ab", "b"],
            "candidates": candidates(text2, 4, ["ab", "b"]),
            "winner": res2b[0],
            "truncate_to": res2b[1],
            "truncated_text": text2[: res2b[1]],
        },
        "note": "两串完成位置同为 4（并列）——严格小于才替换，先按列表序查到的留下；两种列表序给出不同截断点",
    }

    # ---- case 3: window lower bound ------------------------------------------
    res3a = uplink.check_stop_strings("abxx", 2, ["ab"], False)  # "ab" is OLD text
    res3b = uplink.check_stop_strings("xxab", 2, ["ab"], False)  # "ab" is NEW text
    res3c = uplink.check_stop_strings("xxab", 0, ["ab"], False)  # no new chars
    out["case3_window_bound"] = {
        "round": 3,
        "old_occurrence": {
            "sub_round": 4,
            "output_text": "abxx",
            "new_char_count": 2,
            "find_start_param": 1 - 2 - 2,
            "result": res3a,
            "note": "起点 1-2-2=-3 → 绝对位置 1：只回看『新增 2 字符 + 拼 stop 可能用的 1 个旧字符』，旧文中的 ab 不重扫",
        },
        "new_occurrence": {
            "sub_round": 5,
            "output_text": "xxab",
            "new_char_count": 2,
            "find_start_param": 1 - 2 - 2,
            "result": res3b,
            "winner": res3b[0],
            "truncate_to": res3b[1],
        },
        "no_new_chars": {
            "sub_round": 6,
            "output_text": "xxab",
            "new_char_count": 0,
            "result": res3c,
            "note": "new_char_count=0 → 直接 None（每步只在新增窗口内找）",
        },
    }

    # ---- case 4: include_in_output truncation modes ---------------------------
    res4a = uplink.check_stop_strings("xxabyy", 6, ["ab"], True)   # text after stop
    res4b = uplink.check_stop_strings("xxab", 4, ["ab"], True)     # stop ends at tail
    res4c = uplink.check_stop_strings("xxabyy", 6, ["ab"], False)  # excluded
    out["case4_include_modes"] = {
        "round": 7,
        "include_text_after": {
            "output_text": "xxabyy",
            "result": res4a,
            "truncate_to": res4a[1],
            "truncated_text": "xxabyy"[: res4a[1]],
            "note": "include=True 截到串尾（best_end=4）",
        },
        "include_stop_at_tail": {
            "output_text": "xxab",
            "result": res4b,
            "truncate_to": res4b[1],
            "note": "best_end >= len(output_text) → -1 不截（串尾恰在文末）",
        },
        "exclude": {
            "output_text": "xxabyy",
            "result": res4c,
            "truncate_to": res4c[1],
            "truncated_text": "xxabyy"[: res4c[1]],
            "note": "include=False 截到串首（best_stop_index=2）",
        },
    }

    dest = Path(__file__).resolve().parent / "m10_stop_arbitration.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    asyncio.run(main())
