#!/usr/bin/env python3
"""ch36 explainer trace — CUDABackend.parse_options 的 fp8 能力启发式 + make_ttgir 的
capability//10 分档门控。纯整数/集合逻辑,逐字复刻自:
  third_party/nvidia/backend/compiler.py:L106      supported_fp8_dtypes 默认值
  third_party/nvidia/backend/compiler.py:L144-L155 parse_options 的 fp8 拼装
  third_party/nvidia/backend/compiler.py:L221      capability // 10 >= 8 门控
  third_party/nvidia/backend/compiler.py:L248      capability // 10 >= 9 门控
host python3 直接跑(不 import triton;不需要 CUDA)——只重放这段能力启发式的控制流。
"""

# 逐字复刻 CUDAOptions.supported_fp8_dtypes 默认值 (compiler.py:L106)
DEFAULT_SUPPORTED_FP8 = ("fp8e5", "fp8e4b15")


def parse_options_fp8(capability):
    """复刻 parse_options 里 fp8 清单 + max_num_imprecise_acc 的启发式 (compiler.py:L146-L155)。
    opts 为空 dict(用户未显式覆盖)——走全部默认分支。"""
    args = {}
    # L146-L150
    if "supported_fp8_dtypes" not in args:
        supported_fp8_dtypes = set(DEFAULT_SUPPORTED_FP8)
        if capability >= 89:                       # L148
            supported_fp8_dtypes.add("fp8e4nv")    # L149
        args["supported_fp8_dtypes"] = tuple(sorted(supported_fp8_dtypes))  # L150
    # L152-L154
    if "deprecated_fp8_dtypes" not in args:
        if capability >= 90:                       # L153
            args["deprecated_fp8_dtypes"] = ("fp8e4b15", )  # L154
    args.setdefault("deprecated_fp8_dtypes", ())
    # L158
    args["max_num_imprecise_acc_default"] = 2**30 if capability == 90 else 0
    return args


def ttgir_gates(capability):
    """复刻 make_ttgir 的两道分档门 (compiler.py:L221, L248)。"""
    return {
        "capability//10": capability // 10,
        "ge8_gate (L221: f32_dot_tc + warp-spec 四连 + add_pipeline)": capability // 10 >= 8,
        "ge9_gate (L248: fence_insertion + tma_lowering)": capability // 10 >= 9,
    }


def num_warps_is_pow2(n):
    """复刻 __post_init__ 的 2 的幂校验 (compiler.py:L122)。"""
    return n > 0 and (n & (n - 1)) == 0


if __name__ == "__main__":
    print("=== parse_options fp8 heuristic (opts={}, per capability) ===")
    for cap in (70, 80, 86, 89, 90):
        a = parse_options_fp8(cap)
        print(f"capability={cap}  cap//10={cap//10}")
        print(f"    supported_fp8_dtypes = {a['supported_fp8_dtypes']}")
        print(f"    deprecated_fp8_dtypes = {a['deprecated_fp8_dtypes']}")
        # 有效可用 = supported - deprecated
        eff = tuple(x for x in a['supported_fp8_dtypes'] if x not in a['deprecated_fp8_dtypes'])
        print(f"    effective_usable_fp8 = {eff}")
        print(f"    max_num_imprecise_acc_default = {a['max_num_imprecise_acc_default']}")

    print()
    print("=== make_ttgir capability//10 gates (per capability) ===")
    for cap in (70, 80, 86, 89, 90):
        g = ttgir_gates(cap)
        print(f"capability={cap}  cap//10={g['capability//10']}  "
              f"ge8={g['ge8_gate (L221: f32_dot_tc + warp-spec 四连 + add_pipeline)']}  "
              f"ge9={g['ge9_gate (L248: fence_insertion + tma_lowering)']}")

    print()
    print("=== num_warps power-of-2 check (__post_init__ L122) ===")
    for n in (1, 2, 3, 4, 6, 8):
        print(f"num_warps={n}  is_pow2={num_warps_is_pow2(n)}")
