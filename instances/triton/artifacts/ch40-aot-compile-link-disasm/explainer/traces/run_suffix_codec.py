#!/usr/bin/env python3
"""m4 — kernel_suffix 编码 ↔ _match_suffix 解码（特化的隐形通道，一对逆运算）。
用 pin(3.2.0) 的真函数实跑往返：
  编码 = triton.compiler.code_generator.kernel_suffix(signature, AttrsDescriptor)
  解码 = triton.tools.link.HeaderParser._match_suffix(suffix, c_sig)
参数：3 个运行期参数 (X, Y, N)，仅 N(第 2 位) 带 divisibility_16。
"""
import json
from triton.compiler.code_generator import kernel_suffix
from triton.backends.compiler import AttrsDescriptor
from triton.tools.link import HeaderParser


if __name__ == "__main__":
    # 3 个运行期参数；只有第 2 位(N)对齐 16 → hints={2:16}
    attrs = AttrsDescriptor.from_hints({2: 16})
    signature = ["*fp32", "*fp32", "i32"]   # X, Y, N —— 只关心长度(逐参数拼 index)
    suffix = kernel_suffix(signature, attrs)

    # 解码：link.py 的逆运算，c_sig 为 C 原型串（3 个参数）
    parser = HeaderParser()
    c_sig = "CUdeviceptr X, CUdeviceptr Y, int32_t N"
    num_specs, sizes = parser._match_suffix(suffix, c_sig)

    out = {
        "encode": {
            "signature_len": len(signature),
            "divisibility_16_params": attrs.arg_properties.get("tt.divisibility", []),
            "equal_to_1_params": attrs.arg_properties.get("tt.equal_to", []),
            "suffix": suffix,
        },
        "decode": {
            "suffix_in": suffix,
            "c_sig": c_sig,
            "num_specs": num_specs,
            "sizes": sizes,   # None|1|16 per param
        },
        "roundtrip_ok": (num_specs == 1 and sizes == [None, None, 16]),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
