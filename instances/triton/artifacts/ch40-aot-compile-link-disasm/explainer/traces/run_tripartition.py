#!/usr/bin/env python3
"""m1 — 签名字符串三分（hints / constants / signature）。
逐字复现 python/triton/tools/compile.py:L82-L102 的 constexpr + 三个字典推导，
版本无关（纯 stdlib），故可 host 直接跑。输入取 dossier m1 的具体参数。
"""
import json


# --- 逐字自 compile.py:L82-L96 的 constexpr（内嵌小函数） ---
def constexpr(s):
    try:
        ret = int(s)
        return ret
    except ValueError:
        pass
    try:
        ret = float(s)
        return ret
    except ValueError:
        pass
    return None


def tripartite(sig_str, arg_names):
    # compile.py:L71  signature = list(map(strip, split(",")))
    signature = list(map(lambda s: s.strip(" "), sig_str.split(",")))
    # compile.py:L94-L95  hints = {i: constexpr(s.split(":")[1]) ... if ":" in s}
    hints = {i: constexpr(s.split(":")[1]) for i, s in enumerate(signature) if ":" in s}
    hints = {k: v for k, v in hints.items() if v is not None}
    # compile.py:L96-L97  constants = {arg_names[i]: constexpr(s)}
    constants = {arg_names[i]: constexpr(s) for i, s in enumerate(signature)}
    constants = {k: v for k, v in constants.items() if v is not None}
    # compile.py:L98-L102  signature = {arg_names[i]: s.split(":")[0] if arg not in constants}
    signature_out = {
        arg_names[i]: s.split(":")[0]
        for i, s in enumerate(signature)
        if arg_names[i] not in constants
    }
    return signature, hints, constants, signature_out


if __name__ == "__main__":
    sig_str = "*fp32:16, i32:16, 1024, i32"
    arg_names = ["X", "N", "BLOCK", "stride"]
    raw, hints, constants, signature = tripartite(sig_str, arg_names)
    out = {
        "input_signature_string": sig_str,
        "arg_names": arg_names,
        "raw_split": raw,
        "num_segments": len(raw),
        "hints": {str(k): v for k, v in hints.items()},
        "constants": constants,
        "signature": signature,
        "num_hints": len(hints),
        "num_constants": len(constants),
        "num_runtime_args": len(signature),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
