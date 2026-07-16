#!/usr/bin/env python3
"""ch06 素材 trace 驱动脚本 —— 取证 semantic.py 的类型提升 / 广播 / 溢出检查。

为何这样跑：本章 skip_impl（无精简版）。真正的取证对象是 triton
python/triton/language/semantic.py 里的几个纯 Python 函数。pin 的 3.2.0 源码树
(instances/triton/source) 缺编译产物 triton._C.libtriton，`import triton` 会失败，
无法直接 import 该树里的 semantic 模块。

解法：computation_type_impl / integer_promote_impl 是**纯类型代数**——只调用
tl.dtype 的 kind()/is_fpXX()/int_bitwidth/int_signedness 等方法与 tl.floatXX 常量，
不碰 IR builder。这些 dtype API 在 triton 3.2.0→3.6.0 之间稳定。故本脚本把 pin 的
3.2.0 版这两个函数**逐字复制**（与 dossier.embed_excerpts 的 L61-L108 / L45-L58
完全一致），跑在本机已装的 triton 3.6.0 的 dtype 对象上，得到真实产出 dtype。

to_tensor 的值域→dtype 判定（L115-L138）、broadcast 的形状代数（L726-L794）、
sanitize_overflow 的 int64 复算比对（L199-L215）同样是纯 Python 逻辑，此处按 pin
源码逐分支复刻并跑出真值。builder 相关的 create_splat/create_broadcast 等只是 IR
落点，不改变 dtype/shape 结果，故不需要。

输出：run_semantic.json（原始真值，供 explainer 表格逐格溯源 + lint 核数字）。
"""
import json

import triton.language as tl

SIGNED = tl.dtype.SIGNEDNESS.SIGNED
UNSIGNED = tl.dtype.SIGNEDNESS.UNSIGNED


# ======================================================================
# 逐字复制自 pin 的 python/triton/language/semantic.py:L45-L58 (triton 3.2.0)
# ======================================================================
def integer_promote_impl(a_ty, b_ty):
    a_rank = a_ty.int_bitwidth
    b_rank = b_ty.int_bitwidth
    a_sn = a_ty.int_signedness
    b_sn = b_ty.int_signedness
    # Rules for signedness taken from "Usual arithmetic conversions" on
    # https://en.cppreference.com/w/c/language/conversion.
    if a_sn == b_sn:
        return a_ty if a_rank > b_rank else b_ty
    elif a_sn == tl.dtype.SIGNEDNESS.UNSIGNED:
        return a_ty if a_rank >= b_rank else b_ty
    elif b_sn == tl.dtype.SIGNEDNESS.UNSIGNED:
        return b_ty if b_rank >= a_rank else a_ty
    raise TypeError(f"unexpected signedness {a_sn} and {b_sn}")


# ======================================================================
# 逐字复制自 pin 的 python/triton/language/semantic.py:L61-L108 (triton 3.2.0)
# ======================================================================
def computation_type_impl(a_ty, a_is_scalar, b_ty, b_is_scalar, div_or_mod):
    # 0) For scalars we follow semantics similar to PyTorch, namely:
    # - If the scalar is of a lower or equal kind (bool < uint < int < fp),
    #   it doesn't participate in the pomotion
    if a_is_scalar != b_is_scalar:
        scalar_ty, tensor_ty = (a_ty, b_ty) if a_is_scalar else (b_ty, a_ty)
        if scalar_ty.kind().value <= tensor_ty.kind().value:
            # Upcast because of 3) and 4) below!
            if div_or_mod and (tensor_ty in (tl.float16, tl.bfloat16)):
                return tl.float32
            return tensor_ty

    # 1) if one operand is double, the other is implicitly
    #    converted to double
    if a_ty.is_fp64() or b_ty.is_fp64():
        return tl.float64
    # 2) if one operand is float, the other is implicitly
    #    converted to float
    if a_ty.is_fp32() or b_ty.is_fp32():
        return tl.float32
    # 3 ) if one operand is half, the other is implicitly converted to half
    if a_ty.is_fp16() or b_ty.is_fp16():
        if div_or_mod:
            return tl.float32
        else:
            return tl.float16
    # 4) return bf16 only if both operands are of bf16
    if a_ty.is_bf16() or b_ty.is_bf16():
        if div_or_mod:
            return tl.float32
        if a_ty.is_bf16() and b_ty.is_bf16():
            return tl.bfloat16
        return tl.float32
    # 5) return fp16 if operands are different fp8
    if a_ty.is_fp8() and b_ty.is_fp8():
        return a_ty if a_ty == b_ty else tl.float16
    if not a_ty.is_int() or not b_ty.is_int():
        raise TypeError(f"unexpected type {a_ty} and {b_ty}")
    # 6 ) both operands are integer and undergo integer promotion
    if div_or_mod and a_ty.int_signedness != b_ty.int_signedness:
        raise TypeError("different signedness")
    return integer_promote_impl(a_ty, b_ty)


# ======================================================================
# 复刻 to_tensor 的值域→dtype 判定 python/triton/language/semantic.py:L115-L138
# (只复刻 dtype 选择这一纯逻辑；full() 建 handle 需 builder，不影响 dtype 结论)
# ======================================================================
def to_tensor_dtype(x):
    if isinstance(x, bool):
        return "int1"
    elif isinstance(x, int):
        if -2**31 <= x < 2**31:
            return "int32"
        elif 2**31 <= x < 2**32:
            return "uint32"
        elif -2**63 <= x < 2**63:
            return "int64"
        elif 2**63 <= x < 2**64:
            return "uint64"
        else:
            raise ValueError(f"Nonrepresentable integer {x}.")
    elif isinstance(x, float):
        min_float32 = 2**-126
        max_float32 = (2 - 2**-23) * 2**127
        abs_x = abs(x)
        if abs_x == float("inf") or abs_x == 0.0 or x != x or min_float32 <= abs_x <= max_float32:
            return "float32"
        else:
            return "float64"


# ======================================================================
# 复刻 broadcast_impl_value 的形状代数 semantic.py:L744-L794
# (create_expand_dims/create_broadcast/create_splat 只是 IR 落点，不改变结果 shape)
# ======================================================================
def broadcast_value_shapes(lhs_shape, rhs_shape):
    """返回 (lhs_out_shape, rhs_out_shape, path)。shape=None 表示 0 维标量(非 block)。"""
    lhs_block = lhs_shape is not None
    rhs_block = rhs_shape is not None
    if lhs_block and not rhs_block:
        return list(lhs_shape), list(lhs_shape), "block-scalar:splat rhs"
    if not lhs_block and rhs_block:
        return list(rhs_shape), list(rhs_shape), "scalar-block:splat lhs"
    if lhs_block and rhs_block:
        L = list(lhs_shape)
        R = list(rhs_shape)
        # 补前导维(补 1)到等秩：create_expand_dims(handle, 0) 在最前面加轴
        while len(L) < len(R):
            L = [1] + L
        while len(R) < len(L):
            R = [1] + R
        ret = []
        for i, left in enumerate(L):
            right = R[i]
            if left == 1:
                ret.append(right)
            elif right == 1 or right == left:
                ret.append(left)
            else:
                raise ValueError(f"incompatible dimensions at index {i}: {left} and {right}")
        return ret, ret, f"block-block: pad->{L}/{R} broadcast->{ret}"
    return None, None, "scalar-scalar:untouched"


# ======================================================================
# 复刻 broadcast_impl_shape 单侧广播判据 semantic.py:L726-L741
# ======================================================================
def broadcast_shape_to(src_shape, dst_shape):
    if src_shape is None:  # non-block -> splat
        return list(dst_shape), "splat"
    if len(src_shape) != len(dst_shape):
        raise ValueError(f"rank mismatch {src_shape} {dst_shape}")
    if list(src_shape) == list(dst_shape):
        return list(src_shape), "identity"
    for i, item in enumerate(src_shape):
        if dst_shape[i] != item and item != 1:
            raise ValueError(f"non-singleton dim {i}: existing {item} vs expanded {dst_shape[i]}")
    return list(dst_shape), "broadcast"


# ======================================================================
# 复刻 binary_op_sanitize_overflow_impl 的 int64 复算比对 semantic.py:L199-L215
# ======================================================================
def sanitize_overflow(dtype, a, b, op="add"):
    """升 int64 复算 → 和该 dtype 的 max/min 比对 → cond(是否不溢出)。"""
    if dtype.int_bitwidth >= 64:
        return {"skipped": True, "reason": "int_bitwidth>=64，无更宽类型复算，函数开头 return"}
    mx = dtype.get_int_max_value()
    mn = dtype.get_int_min_value()
    if op == "add":
        ret = a + b       # int64 头顶空间足够容纳 <64bit 二元运算真值
    elif op == "sub":
        ret = a - b
    elif op == "mul":
        ret = a * b
    cond = (ret <= mx) and (ret >= mn)
    return {"skipped": False, "op": op, "a": a, "b": b, "int64_recompute": ret,
            "type_max": mx, "type_min": mn, "cond_no_overflow": cond}


# ======================================================================
# 跑所有 worked example
# ======================================================================
def nm(d):
    return d.name if hasattr(d, "name") else str(d)


out = {}

# --- m06-computation-type：六档主表 ---
comp_cases = [
    ("fp16 张量 × fp32 张量", tl.float16, False, tl.float32, False, False),
    ("fp16 张量 × fp16 张量", tl.float16, False, tl.float16, False, False),
    ("bf16 张量 × bf16 张量", tl.bfloat16, False, tl.bfloat16, False, False),
    ("bf16 张量 × fp8e5 张量", tl.bfloat16, False, tl.float8e5, False, False),
    ("fp8e5 张量 × fp8e4nv 张量", tl.float8e5, False, tl.float8e4nv, False, False),
    ("fp8e5 张量 × fp8e5 张量", tl.float8e5, False, tl.float8e5, False, False),
    ("int32 张量 × int64 张量", tl.int32, False, tl.int64, False, False),
    ("fp64 张量 × int32 张量", tl.float64, False, tl.int32, False, False),
    ("fp16 张量 × fp16 张量 (div_or_mod)", tl.float16, False, tl.float16, False, True),
    ("bf16 张量 × bf16 张量 (div_or_mod)", tl.bfloat16, False, tl.bfloat16, False, True),
]
out["m06-computation-type"] = []
for label, a, asc, b, bsc, dm in comp_cases:
    r = computation_type_impl(a, asc, b, bsc, dm)
    out["m06-computation-type"].append({
        "label": label, "a": nm(a), "a_is_scalar": asc, "b": nm(b), "b_is_scalar": bsc,
        "div_or_mod": dm, "a_kind": a.kind().value, "b_kind": b.kind().value, "result": nm(r)})

# --- m06-scalar-nonpromote：标量 vs 物化张量 对照 ---
scalar_cases = [
    ("fp16 张量 × Python 浮点标量 3.14 (is_scalar=True)", tl.float16, False, tl.float32, True, False),
    ("fp16 张量 × 物化成 fp32 张量 (is_scalar=False)", tl.float16, False, tl.float32, False, False),
    ("fp16 张量 × int 标量 5 (is_scalar=True)", tl.float16, False, tl.int32, True, False),
    ("int32 张量 × Python 浮点标量 (is_scalar=True)", tl.int32, False, tl.float32, True, False),
    ("fp16 张量 × 浮点标量 (div_or_mod=True)", tl.float16, False, tl.float32, True, True),
]
out["m06-scalar-nonpromote"] = []
for label, a, asc, b, bsc, dm in scalar_cases:
    r = computation_type_impl(a, asc, b, bsc, dm)
    scalar_ty, tensor_ty = (a, b) if asc else (b, a)
    step0 = (asc != bsc) and (scalar_ty.kind().value <= tensor_ty.kind().value)
    out["m06-scalar-nonpromote"].append({
        "label": label, "a": nm(a), "a_is_scalar": asc, "b": nm(b), "b_is_scalar": bsc,
        "div_or_mod": dm, "scalar_kind": scalar_ty.kind().value, "tensor_kind": tensor_ty.kind().value,
        "step0_fires": step0, "result": nm(r)})

# --- m06-integer-promote ---
int_cases = [
    ("int32 × int64 (同号,取更宽 rank)", tl.int32, tl.int64),
    ("uint32 × int32 (异号,rank 相等,取无符号)", tl.uint32, tl.int32),
    ("int8 × uint8 (异号,rank 相等,取无符号)", tl.int8, tl.uint8),
    ("int64 × uint32 (异号,有符号更宽)", tl.int64, tl.uint32),
    ("int16 × int32 (同号,取 int32)", tl.int16, tl.int32),
]
out["m06-integer-promote"] = []
for label, a, b in int_cases:
    r = integer_promote_impl(a, b)
    out["m06-integer-promote"].append({
        "label": label, "a": nm(a), "a_rank": a.int_bitwidth, "a_sn": str(a.int_signedness),
        "b": nm(b), "b_rank": b.int_bitwidth, "b_sn": str(b.int_signedness), "result": nm(r)})

# --- m06-to-tensor ---
tt_cases = [True, 5, -7, 2**31, 2**40, 2**63, 3.14, 1e300, 0.0, float("inf")]
out["m06-to-tensor"] = []
for x in tt_cases:
    out["m06-to-tensor"].append({"input": repr(x), "dtype": to_tensor_dtype(x)})

# --- m06-broadcast-value ---
bv_cases = [
    ("(128,) × 标量()", (128,), None),
    ("(128,1) × (1,64)", (128, 1), (1, 64)),
    ("(128,) × (64,128)", (128,), (64, 128)),
    ("(1,64) × (128,64)", (1, 64), (128, 64)),
]
out["m06-broadcast-value"] = []
for label, ls, rs in bv_cases:
    lo, ro, path = broadcast_value_shapes(ls, rs)
    out["m06-broadcast-value"].append({
        "label": label, "lhs_in": list(ls) if ls else None, "rhs_in": list(rs) if rs else None,
        "lhs_out": lo, "rhs_out": ro, "path": path})

# 报错分支
try:
    broadcast_value_shapes((128, 3), (128, 4))
    out["m06-broadcast-value-error"] = "no error!?"
except ValueError as e:
    out["m06-broadcast-value-error"] = {"label": "(128,3) × (128,4)", "error": str(e)}

# --- m06-broadcast-shape ---
bs_cases = [
    ("(1,64) -> (128,64) 合法", (1, 64), (128, 64)),
    ("标量() -> (128,64) splat", None, (128, 64)),
    ("(128,64) -> (128,64) identity", (128, 64), (128, 64)),
]
out["m06-broadcast-shape"] = []
for label, src, dst in bs_cases:
    res, path = broadcast_shape_to(src, dst)
    out["m06-broadcast-shape"].append({
        "label": label, "src": list(src) if src else None, "dst": list(dst), "result": res, "path": path})
try:
    broadcast_shape_to((3, 64), (128, 64))
    out["m06-broadcast-shape-error"] = "no error!?"
except ValueError as e:
    out["m06-broadcast-shape-error"] = {"label": "(3,64) -> (128,64) 报错", "error": str(e)}

# --- m06-sanitize-overflow ---
so_cases = [
    (tl.int8, 100, 100, "add"),
    (tl.int8, 60, 50, "add"),
    (tl.int32, 2000000000, 2000000000, "add"),
    (tl.int32, 100, 100, "add"),
    (tl.int64, 100, 100, "add"),
]
out["m06-sanitize-overflow"] = []
for dt, a, b, op in so_cases:
    r = sanitize_overflow(dt, a, b, op)
    r["dtype"] = nm(dt)
    r["int_bitwidth"] = dt.int_bitwidth
    out["m06-sanitize-overflow"].append(r)

print(json.dumps(out, ensure_ascii=False, indent=2))
