"""Driver for m8 (byte-fallback UTF-8 boundary) — host run against the ch07
companion (pin vLLM v0.27.1).

One byte stream driven through BOTH detokenizer paths (Fast = real Rust
DecodeStream; Slow = byte-level TokenizerLike seam) — they must agree:
  65 'A'      -> 'A'        (complete ASCII char, flows immediately)
  228 E4      -> ''         (first byte of '中' — incomplete, held)
  184 B8      -> ''         (second byte — still incomplete, held)
  173 AD      -> '中'       (third byte completes the 3-byte char)
  184 B8      -> ''         (stranded continuation byte — tail, held)
  66  'B'     -> '�B'  (B makes the tail decodable; the stranded byte is
                             now MID-text and passes through as a REAL
                             replacement char plus 'B')
For the Slow path each round also records prefix/read offsets: held rounds
return the OLD offsets (read_offset frozen — the half char is re-read next
round), completing rounds advance read_offset past every re-read token.
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

STREAM = [
    (65, "41", "A"),
    (228, "E4", "中-第1字节"),
    (184, "B8", "中-第2字节"),
    (173, "AD", "中-第3字节"),
    (184, "B8", "搁浅续字节"),
    (66, "42", "B"),
]


async def main():
    out = {
        "driver": "run_m8_byte_fallback.py",
        "mechanism": "m8 byte-fallback UTF-8 边界：末尾 U+FFFD/未增长 → 吐空串冻结 read_offset，等下一 token 补全；中间替换字符照常流出",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "stream": [
            {"token_id": t, "byte_hex": h, "meaning": m} for t, h, m in STREAM
        ],
        "target_char": "中",
        "utf8_of_zhong": ["E4", "B8", "AD"],
    }

    fast = td.fast_detok(prompt="Hi")
    slow = td.slow_detok(prompt="hi")
    rounds = []
    for tid, hx, meaning in STREAM:
        f_before = fast.output_text
        fast.update([tid], False)
        f_delta = fast.output_text[len(f_before):]
        s_before = slow.output_text
        s_read_before = slow.read_offset
        s_ids_before = len(slow.token_ids)
        slow.update([tid], False)
        s_delta = slow.output_text[len(s_before):]
        rounds.append({
            "round": len(rounds) + 1,
            "token_id": tid,
            "byte_hex": hx,
            "meaning": meaning,
            "fast_delta": f_delta,
            "slow_delta": s_delta,
            "slow_read_offset_before": s_read_before,
            "slow_read_offset_after": slow.read_offset,
            "slow_read_frozen": slow.read_offset == s_read_before,
            "slow_id_ledger_grew_even_when_held": len(slow.token_ids) == s_ids_before + 1,
            "fast_output_text_after": fast.output_text,
            "slow_output_text_after": slow.output_text,
        })
    out["rounds"] = rounds
    out["agreement"] = {
        "paths_agree_every_round": all(
            r["fast_delta"] == r["slow_delta"] for r in rounds
        ),
        "final_output_text": fast.output_text,
        "slow_final_output_text": slow.output_text,
        "held_rounds": [r["round"] for r in rounds if r["fast_delta"] == ""],
        "emitting_rounds": [r["round"] for r in rounds if r["fast_delta"] != ""],
        "slow_read_freeze_then_jump": [
            {"round": r["round"], "before": r["slow_read_offset_before"], "after": r["slow_read_offset_after"]}
            for r in rounds
        ],
        "tail_rule": "判定只看 new_text 末尾：末尾 '�' = 可能未完 → 空串+冻结；补全 token 到达后连同冻结段一次吐出",
    }

    dest = Path(__file__).resolve().parent / "m8_byte_fallback.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    asyncio.run(main())
