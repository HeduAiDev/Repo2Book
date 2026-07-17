#!/usr/bin/env python3
"""m9 — parseCtrl：64 位控制字位解码。
用 pin(3.2.0) 的真 parseCtrl 实跑两条真实 SLINE 编码，并把中间位字段全部打印出来
（stall/yield/wr-barrier/rd-barrier/wait-mask），格式化结果 = kernel.asm['sass'] 左列那串。
"""
import json
from triton.tools.disasm import parseCtrl, SLINE_RE


def decode_fields(hexstr):
    sline = f' /* 0x{hexstr} */ '
    enc = int(SLINE_RE.match(sline).group(1), 16)
    stall = (enc >> 41) & 0xf
    yld = (enc >> 45) & 0x1
    wrtdb = (enc >> 46) & 0x7
    readb = (enc >> 49) & 0x7
    watdb = (enc >> 52) & 0x3f
    return {
        "sline_hex": hexstr,
        "stall": stall,
        "yield_bit": yld,
        "wr_barrier": wrtdb,
        "rd_barrier": readb,
        "wait_mask": watdb,
        "formatted": parseCtrl(sline),   # 权威：调真函数
    }


if __name__ == "__main__":
    out = {
        "example_1": decode_fields("000e220000000800"),
        "example_2": decode_fields("002fda000780c0ff"),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
