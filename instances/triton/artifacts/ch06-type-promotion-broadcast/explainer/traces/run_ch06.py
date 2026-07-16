#!/usr/bin/env python3
"""ch06 类型提升与广播 —— pin 精确取证驱动脚本。

运行环境:instances/triton/v32 venv (pip triton==3.2.0,Python 前端与 pin
9641643 逐字节相同,见 INSTANCE.md ★pin 精确 IR 验证配方)。headless、无 GPU。

直接调 python/triton/language/semantic.py 的真函数:
  - computation_type_impl / integer_promote_impl  → 纯函数,无 builder,直接调
  - to_tensor / broadcast_impl_value / broadcast_impl_shape → 用 headless ir.builder 真跑
  - get_int_max_value/get_int_min_value → 真常量;溢出 int64 复算比对(assert 本身在 device)

用法:instances/triton/v32/bin/python run_ch06.py > ch06_run.json
"""
import json
import triton
import triton.language as tl
from triton.language import semantic
from triton._C.libtriton import ir

ctx = ir.context()
ir.load_dialects(ctx)
b = ir.builder(ctx)


def scalar(v, dt):
    return semantic.full((), v, dtype=dt, builder=b)


def block(shape, dt=tl.float32):
    return semantic.broadcast_impl_shape(scalar(0, dt), list(shape), b)


out = {"triton_version": triton.__version__, "note": "pin-compile headless direct-call"}
# 六档规则的档位索引(0..6)与 kind 序(bool<int<fp),供正文分档引用
out["rule_tiers"] = [0, 1, 2, 3, 4, 5, 6]
out["kind_order"] = {"bool": tl.int1.kind().value, "int": tl.int32.kind().value,
                     "fp": tl.float32.kind().value}

# ---- 1) computation_type_impl 六档 + 标量档 ----
# 每档一条:(a_dtype, a_is_scalar, b_dtype, b_is_scalar, div_or_mod)
comp_cases = [
    ("fp16",  tl.float16,  False, "fp32", tl.float32,  False, False),  # 档2:一侧fp32→fp32
    ("fp16",  tl.float16,  False, "fp16", tl.float16,  False, False),  # 档3:一侧fp16→fp16
    ("bf16",  tl.bfloat16, False, "fp8e5", tl.float8e5, False, False), # 档4:bf16非双侧→fp32
    ("bf16",  tl.bfloat16, False, "bf16", tl.bfloat16,  False, False), # 档4:双侧bf16→bf16
    ("fp8e4nv", tl.float8e4nv, False, "fp8e5", tl.float8e5, False, False),  # 档5:异fp8→fp16
    ("fp8e5", tl.float8e5, False, "fp8e5", tl.float8e5,  False, False),     # 档5:同fp8→自身
    ("int32", tl.int32,   False, "int64", tl.int64,     False, False),  # 档6:整数提升→int64
    ("fp16",  tl.float16,  False, "fp16", tl.float16,   False, True),   # 档3+除模→fp32
]
comp_rows = []
for an, aty, asc, bn, bty, bsc, dm in comp_cases:
    r = semantic.computation_type_impl(aty, asc, bty, bsc, dm)
    comp_rows.append({"a": an, "b": bn, "div_or_mod": dm, "result": str(r)})
out["computation_type"] = comp_rows

# ---- 2) 标量不拔高张量档次(step 0) ----
# 关键对照:同一个 fp32 常量,当"标量"vs"物化成fp32张量"两种身份,产出不同
scalar_cases = [
    ("fp16 tensor", tl.float16, False, "fp32 SCALAR", tl.float32, True,  False),  # →fp16(标量退让)
    ("fp16 tensor", tl.float16, False, "fp32 TENSOR", tl.float32, False, False),  # →fp32(物化上拉)
    ("fp16 tensor", tl.float16, False, "int SCALAR",  tl.int32,   True,  False),  # int标量→fp16
    ("int32 tensor", tl.int32,  False, "fp32 SCALAR", tl.float32, True,  False),  # fp标量>int张量→fp32
    ("fp16 tensor", tl.float16, False, "fp32 SCALAR", tl.float32, True,  True),   # 除模例外→fp32
]
scalar_rows = []
for an, aty, asc, bn, bty, bsc, dm in scalar_cases:
    r = semantic.computation_type_impl(aty, asc, bty, bsc, dm)
    scalar_rows.append({"tensor_side": an, "other_side": bn, "div_or_mod": dm,
                        "a_kind": aty.kind().value, "b_kind": bty.kind().value, "result": str(r)})
out["scalar_nonpromote"] = scalar_rows

# ---- 3) integer_promote_impl 整数 usual arithmetic conversions ----
int_cases = [
    ("int32", tl.int32, "int64", tl.int64),   # 同号取宽→int64
    ("uint32", tl.uint32, "int32", tl.int32),  # 异号,unsigned rank>=signed→uint32
    ("int8", tl.int8, "uint8", tl.uint8),      # 异号,unsigned rank>=signed→uint8
    ("int16", tl.int16, "int32", tl.int32),    # 同号取宽→int32
]
int_rows = []
for an, aty, bn, bty in int_cases:
    r = semantic.integer_promote_impl(aty, bty)
    int_rows.append({"a": an, "a_bits": aty.int_bitwidth, "a_signed": str(aty.int_signedness),
                     "b": bn, "b_bits": bty.int_bitwidth, "b_signed": str(bty.int_signedness),
                     "result": str(r)})
out["integer_promote"] = int_rows

# ---- 4) to_tensor 按值域定 dtype ----
tt_cases = [
    ("5", 5),
    ("3000000000", 3000000000),   # 3e9 ∈ [2^31,2^32) → uint32
    ("5000000000", 5000000000),   # 5e9 ∈ [2^32,2^63) → int64
    ("True(bool)", True),
    ("3.5", 3.5),
    ("1e300", 1e300),             # > max_float32 → float64
]
tt_rows = []
for label, v in tt_cases:
    t = semantic.to_tensor(v, b)
    tt_rows.append({"value": label, "dtype": str(t.type), "is_block": t.type.is_block()})
out["to_tensor"] = tt_rows
# 值域阈值常量(源码 L116-L138)
out["to_tensor_thresholds"] = {
    "int32_hi": 2**31, "uint32_hi": 2**32, "int64_hi": 2**63,
    "min_float32": 2**-126, "max_float32": (2 - 2**-23) * 2**127,
}

# ---- 5) broadcast_impl_value 两支 ----
bc_rows = []
# block-scalar → splat
lhs = block((128,), tl.float32); rhs = scalar(3.0, tl.float32)
o1, o2 = semantic.broadcast_impl_value(lhs, rhs, b)
bc_rows.append({"case": "block(128,) x scalar", "path": "splat",
                "lhs_out": list(o1.type.shape), "rhs_out": list(o2.type.shape)})
# block-block 等秩,尺寸1维互扩
l = block((128, 1)); r = block((1, 64))
o1, o2 = semantic.broadcast_impl_value(l, r, b)
bc_rows.append({"case": "(128,1) x (1,64)", "path": "broadcast",
                "lhs_out": list(o1.type.shape), "rhs_out": list(o2.type.shape)})
# block-block 补前导维
l = block((128,)); r = block((64, 128))
o1, o2 = semantic.broadcast_impl_value(l, r, b)
bc_rows.append({"case": "(128,) x (64,128)", "path": "expand_dims+broadcast",
                "lhs_out": list(o1.type.shape), "rhs_out": list(o2.type.shape)})
out["broadcast_value"] = bc_rows

# ---- 6) broadcast_impl_shape 单侧到目标 shape ----
bs_rows = []
o = semantic.broadcast_impl_shape(block((1, 64)), [128, 64], b)
bs_rows.append({"src": "(1,64)", "target": "(128,64)", "result": list(o.type.shape), "ok": True})
try:
    semantic.broadcast_impl_shape(block((2, 64)), [128, 64], b)
    bs_rows.append({"src": "(2,64)", "target": "(128,64)", "result": None, "ok": True})
except ValueError as e:
    bs_rows.append({"src": "(2,64)", "target": "(128,64)", "result": "ValueError",
                    "ok": False, "msg": str(e)[:120]})
out["broadcast_shape"] = bs_rows

# ---- 7) sanitize_overflow int64 复算比对(assert 本身在 device) ----
ov_rows = []
for label, dt, x, y in [("int8", tl.int8, 100, 100), ("int32", tl.int32, 1000000, 1)]:
    mx = dt.get_int_max_value()
    mn = dt.get_int_min_value()
    ret = int(x) + int(y)          # int64 头顶复算(Python int 任意精度 ⊇ int64)
    cond = (ret <= mx) and (ret >= mn)
    ov_rows.append({"dtype": label, "expr": f"{x}+{y}", "int64_recompute": ret,
                    "max": mx, "min": mn, "cond_in_range": cond,
                    "verdict": "pass" if cond else "assert fails (device)"})
out["sanitize_overflow"] = ov_rows

print(json.dumps(out, indent=2, ensure_ascii=False))
