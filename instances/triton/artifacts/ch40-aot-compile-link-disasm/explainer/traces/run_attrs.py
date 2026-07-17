#!/usr/bin/env python3
"""m2 — hints 物化成 AttrsDescriptor（AOT 特化路径）。
用 pin(3.2.0, 与钉版逐字节相同)的 AttrsDescriptor.from_hints 实跑，
观测 arg_properties 与 get_constants()。展示两条支路：
  (a) 全 :16   → 全进 tt.divisibility，get_constants 空
  (b) 含 :1    → 该位进 tt.equal_to 且 get_constants 把它并成常量(从运行期原型消失)
"""
import json
from triton.backends.compiler import AttrsDescriptor


def probe(hints):
    attrs = AttrsDescriptor.from_hints(hints)
    return {
        "hints_in": {str(k): v for k, v in hints.items()},
        "property_values": attrs.property_values,
        "divisibility_16_params": attrs.arg_properties.get("tt.divisibility", []),
        "equal_to_1_params": attrs.arg_properties.get("tt.equal_to", []),
        "get_constants": {str(k): v for k, v in attrs.get_constants().items()},
    }


if __name__ == "__main__":
    out = {
        "case_a_all_16": probe({0: 16, 1: 16}),
        "case_b_with_1": probe({0: 16, 2: 1}),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
