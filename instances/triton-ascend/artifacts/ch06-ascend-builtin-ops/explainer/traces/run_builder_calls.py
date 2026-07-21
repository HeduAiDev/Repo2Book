"""ch06 素材驱动 ②:内建算子发到 builder 上的调用序列(m2/m3/m4/m5/m6/m9/m10/m11/m12)。

跑的是本章精简版(implementation/,只做减法),用 tests/conftest.py 里的 FakeBuilder
站在真实 C++ `ir.builder` 位置上(host 无昇腾 NPU/CANN,见 INSTANCE.md)——记录的是
"Python 语言层到底给 builder 发了哪些调用、参数位宽是什么",不是 MLIR/真机语义。

输出:builder_calls.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CH = HERE.parents[1]
sys.path.insert(0, str(CH / "tests"))

import conftest  # noqa: E402
from conftest import FakeBuilder, make_tensor, make_gm_ptr  # noqa: E402

gen = conftest.env.__wrapped__()
mods = next(gen)
core = mods.core
constexpr = core.constexpr

report = {
    "driver": "explainer/traces/run_builder_calls.py",
    "pin": "2badfc89e70a9b7a5e88463a116c2feddce4b101 (v3.2.1)",
    "under_test": "implementation/(本章精简版,只做减法)+ tests/conftest.py 的 FakeBuilder",
    "environment": "host, 无昇腾 NPU/CANN;记录 Python 层发给 builder 的调用,不是真机数值",
    "scenarios": {},
}


def kinds(b):
    return [c[0] for c in b.calls]


def count(b, name):
    return len([c for c in b.calls if c[0] == name])


def repr_calls(b):
    return [repr(c) for c in b.calls]


# ======================= m6 / m2 / m3 / m4:位宽契约 =======================
widths = []

b = FakeBuilder()
src = make_gm_ptr(mods, core.float32, name="src")
idx22 = make_tensor(mods, [2, 2], core.int32)
mods.mem_ops.gather_out_to_ub(src=src, index=idx22, index_boundary=4, dim=0,
                              src_stride=(3, 1), end_offset=(2, 2), start_offset=(0, 0),
                              _builder=b)
c = [x for x in b.calls if x[0] == "create_gather_out_to_ub"][0]
widths.append({"op": "gather_out_to_ub", "index_dtype": "int32",
               "stride_width": c[5][0][0], "end_offset_width": c[6][0][0],
               "start_offset_width": c[7][0][0], "has_index_boundary": True,
               "index_boundary": c[3], "switch": "硬编码 stride=True / offsets=False",
               "raw": repr(c)})

b = FakeBuilder()
ptr = make_gm_ptr(mods, core.float32)
val22 = make_tensor(mods, [2, 2], core.float32)
mods.mem_ops.scatter_ub_to_out(ptr=ptr, value=val22, index=idx22, index_boundary=4, dim=0,
                               dst_stride=(3, 1), end_offset=(2, 2), start_offset=(0, 0),
                               _builder=b)
c = [x for x in b.calls if x[0] == "create_scatter_ub_to_out"][0]
widths.append({"op": "scatter_ub_to_out", "index_dtype": "int32",
               "stride_width": c[6][0][0], "end_offset_width": c[7][0][0],
               "start_offset_width": c[8][0][0], "has_index_boundary": True,
               "index_boundary": c[4], "switch": "硬编码 stride=True / offsets=False",
               "raw": repr(c)})

for idx_dtype, name in ((core.int32, "int32"), (core.int64, "int64")):
    b = FakeBuilder()
    ptr = make_gm_ptr(mods, core.float32)
    val = make_tensor(mods, [2, 2], core.float32)
    idx1d = make_tensor(mods, [2], idx_dtype)
    mods.mem_ops.index_put(ptr=ptr, index=idx1d, value=val, dim=0, index_boundary=4,
                           end_offset=(2, 2), start_offset=(0, 0), dst_stride=(3, 1),
                           _builder=b)
    c = [x for x in b.calls if x[0] == "create_index_put"][0]
    widths.append({"op": "index_put", "index_dtype": name,
                   "stride_width": c[8][0][0], "end_offset_width": c[6][0][0],
                   "start_offset_width": c[7][0][0], "has_index_boundary": True,
                   "index_boundary": c[5],
                   "switch": "require_i64 = index.dtype.is_int64() 单开关同时决定三者",
                   "raw": repr(c)})

b = FakeBuilder()
src = make_gm_ptr(mods, core.float32, name="src")
idx4 = make_tensor(mods, [4], core.int32)
mods.mem_ops.index_select_simd(src=src, dim=1, index=idx4,
                               src_shape=(8, 100, 256), src_offset=(4, -1, 128),
                               read_shape=(4, -1, 128), _builder=b)
c = [x for x in b.calls if x[0] == "create_index_select_simd"][0]
widths.append({"op": "index_select_simd", "index_dtype": "int32",
               "stride_width": None, "end_offset_width": None, "start_offset_width": None,
               "has_index_boundary": False, "index_boundary": None,
               "switch": "无 stride/offset 三元组,也无 index_boundary 参数",
               "raw": repr(c)})

report["scenarios"]["m6_width_contract"] = {"rows": widths}

# ======================= m4:index 摊平 =======================
b = FakeBuilder()
ptr = make_gm_ptr(mods, core.float32)
val = make_tensor(mods, [2, 2], core.float32)
idx_2d = make_tensor(mods, [2, 1], core.int32)
mods.mem_ops.index_put(ptr=ptr, index=idx_2d, value=val, dim=0, index_boundary=4,
                       end_offset=(2, 2), start_offset=(0, 1), dst_stride=(3, 1), _builder=b)
resh = [x for x in b.calls if x[0] == "create_reshape"]
report["scenarios"]["m4_index_flatten"] = {
    "index_shape_in": [2, 1], "index_shape_after_reshape": list(resh[0][2]),
    "value_shape": [2, 2], "dim": 0, "value_shape_at_dim": 2,
    "calls": repr_calls(b),
}

# ======================= m3:标量 value 自动广播 =======================
b = FakeBuilder()
ptr = make_gm_ptr(mods, core.float32)
mods.mem_ops.scatter_ub_to_out(ptr=ptr, value=0.0, index=idx22, index_boundary=4, dim=0,
                               dst_stride=(3, 1), end_offset=(2, 2), start_offset=(0, 0),
                               _builder=b)
splat = [x for x in b.calls if x[0] == "create_splat"]
report["scenarios"]["m3_scalar_broadcast"] = {
    "value_in": 0.0, "splat_calls": len(splat), "splat_shape": list(splat[0][2]),
    "calls": repr_calls(b),
}

# ======================= m5:返回 shape 推导 =======================
sel_rows = []
for dim, ishape, src_shape, src_off, read_shape in (
        (1, [4], (8, 100, 256), (4, -1, 128), (4, -1, 128)),
        (0, [3], (8, 100, 256), (-1, 0, 0), (-1, 50, 128)),
):
    b = FakeBuilder()
    src = make_gm_ptr(mods, core.float32, name="src")
    index = make_tensor(mods, ishape, core.int32)
    out = mods.mem_ops.index_select_simd(src=src, dim=dim, index=index, src_shape=src_shape,
                                         src_offset=src_off, read_shape=read_shape, _builder=b)
    c = [x for x in b.calls if x[0] == "create_index_select_simd"][0]
    sel_rows.append({"dim": dim, "index_len": ishape[0], "src_shape": list(src_shape),
                     "src_offset": list(src_off), "read_shape": list(read_shape),
                     "return_shape": [s.value for s in out.shape],
                     "read_shape_passed_through": list(c[6]), "raw": repr(c)})

errs = {}
b = FakeBuilder()
src = make_gm_ptr(mods, core.float32)
index = make_tensor(mods, [4], core.int32)
try:
    mods.mem_ops.index_select_simd(src=src, dim=2, index=index, src_shape=(8, 100, 256),
                                   src_offset=(4, -1, -1), read_shape=(4, -1, -1), _builder=b)
except AssertionError as e:
    errs["dim==ndim-1(尾轴)"] = str(e)
b = FakeBuilder()
index2d = make_tensor(mods, [2, 2], core.int32)
try:
    mods.mem_ops.index_select_simd(src=src, dim=0, index=index2d, src_shape=(8, 100),
                                   src_offset=(-1, 0), read_shape=(-1, 4), _builder=b)
except AssertionError as e:
    errs["index 非 1D"] = str(e)
report["scenarios"]["m5_return_shape"] = {"rows": sel_rows, "rejections": errs}

# ======================= m9:flip 双路径 =======================
flip_rows = []
for is_simt, shape, dim in ((False, [4, 8], 1), (False, [8], 0), (True, [4], 0), (True, [8], 0)):
    b = FakeBuilder(is_simt=is_simt)
    ptr = make_tensor(mods, shape, core.float32)
    out = mods.vec_ops.flip(ptr, dim=dim, _builder=b)
    flip_rows.append({
        "is_simt_mode": is_simt, "shape": shape, "dim": dim,
        "n": shape[dim],
        "create_flip": count(b, "create_flip"),
        "create_reduce": count(b, "create_reduce"),
        "create_xor": count(b, "create_xor"),
        "create_reshape": count(b, "create_reshape"),
        "create_bitcast": count(b, "create_bitcast"),
        "reshape_shapes": [list(c[2]) for c in b.calls if c[0] == "create_reshape"],
        "total_builder_calls": len(b.calls),
        "out_shape": list(out.type.shape) if hasattr(out.type, "shape") else None,
        "calls": kinds(b),
    })
b = FakeBuilder(is_simt=True)
ptr = make_tensor(mods, [3], core.int32)
flip_err = None
try:
    mods.vec_ops.flip(ptr, dim=0, _builder=b)
except AssertionError as e:
    flip_err = "static_assert(_is_power_of_two(3)) 失败: %s" % (str(e) or "AssertionError")
report["scenarios"]["m9_flip"] = {"rows": flip_rows, "non_power_of_two": flip_err}

# ======================= m10:sort =======================
sort_rows = []
for dtype, name, shape, dim in ((core.float32, "float32", [4, 8], 1),
                                (core.int8, "int8", [4], 0),
                                (core.int16, "int16", [4], 0),
                                (core.int32, "int32", [4], 0)):
    b = FakeBuilder()
    ptr = make_tensor(mods, shape, dtype)
    out = mods.vec_ops.sort(ptr, dim=dim, descending=False, _builder=b)
    hints = [c for c in b.calls if c[0] == "create_annotation_mark"]
    sort_rows.append({"dtype": name, "shape": shape, "dim": dim, "accepted": True,
                      "create_sort": count(b, "create_sort"),
                      "compile_hints": len(hints),
                      "hint": [hints[0][2], list(hints[0][3])] if hints else None,
                      "calls": kinds(b)})
sort_err = {}
b = FakeBuilder()
try:
    mods.vec_ops.sort(make_tensor(mods, [4, 8], core.float32), dim=0, _builder=b)
except ValueError as e:
    sort_err["dim=0(非末维)"] = str(e)
b = FakeBuilder()
try:
    mods.vec_ops.sort(make_tensor(mods, [4], core.uint8), dim=0, _builder=b)
except TypeError as e:
    sort_err["uint8(不在白名单)"] = str(e)
report["scenarios"]["m10_sort"] = {"rows": sort_rows, "rejections": sort_err}

# ======================= m11:cast 决策树 =======================
cast_rows = []


def cast_case(label, src_dtype, dst_dtype, on_910_95=False, overflow_mode=None):
    b = FakeBuilder()
    mods.vec_ops.is_compile_on_910_95 = on_910_95
    try:
        x = make_tensor(mods, [4], src_dtype)
        out = mods.vec_ops.ascend_cast_impl(x, dst_dtype, b, overflow_mode=overflow_mode)
        ok, err, dt = True, None, out.dtype.name
    except ValueError as e:
        ok, err, dt = False, str(e), None
    finally:
        mods.vec_ops.is_compile_on_910_95 = False
    conv = [c[0] for c in b.calls if c[0] not in ("create_annotation_mark",)]
    cast_rows.append({
        "case": label, "src": src_dtype.name, "dst": dst_dtype.name,
        "on_910_95": on_910_95, "overflow_mode": overflow_mode, "accepted": ok,
        "conversion_ops": len(conv), "ops": conv,
        "compile_hints": count(b, "create_annotation_mark"),
        "hint_names": [c[2] for c in b.calls if c[0] == "create_annotation_mark"],
        "result_dtype": dt, "error": err, "calls": kinds(b)})


cast_case("fp16 -> fp32(直转)", core.float16, core.float32)
cast_case("bf16 -> fp16(必经 fp32 中转)", core.bfloat16, core.float16)
cast_case("fp16 -> int32(直转)", core.float16, core.int32)
cast_case("fp32 -> int32(直转)", core.float32, core.int32)
cast_case("int32 -> bool(变 != 0)", core.int32, core.int1)
cast_case("uint32 -> int16 saturate @910_95", core.uint32, core.int16,
          on_910_95=True, overflow_mode="saturate")
cast_case("uint32 -> int16 saturate @非 910_95", core.uint32, core.int16,
          on_910_95=False, overflow_mode="saturate")
cast_case("fp8e4nv -> fp32 @非 910_95(直接拒绝)", core.float8e4nv, core.float32)
report["scenarios"]["m11_cast_tree"] = {"rows": cast_rows}

# ======================= m12:overflow_mode 校验与 compile_hint 挂载 =======================
ov_rows = []
for mode in (None, "trunc", "saturate", "sautrate"):
    b = FakeBuilder()
    x = make_tensor(mods, [4], core.int32)
    om = None if mode is None else constexpr(mode)
    try:
        mods.vec_ops.cast(x, core.int16, overflow_mode=om, _builder=b)
        ok, err = True, None
    except ValueError as e:
        ok, err = False, str(e)
    hints = [c for c in b.calls if c[0] == "create_annotation_mark"]
    ov_rows.append({"overflow_mode": mode, "accepted": ok,
                    "int_cast_calls": count(b, "create_int_cast"),
                    "compile_hints": len(hints),
                    "hint": [hints[0][2], list(hints[0][3])] if hints else None,
                    "error": err, "calls": kinds(b)})
docstring = mods.vec_ops.cast.__doc__
report["scenarios"]["m12_overflow_mode"] = {
    "rows": ov_rows,
    "docstring_spelling": "sautrate" if "sautrate" in docstring else "(not found)",
    "docstring_has_correct_saturate": "saturate" in docstring.replace("sautrate", ""),
    "real_whitelist": ["trunc", "saturate"],
    "whitelist_source": "third_party/ascend/language/cann/extension/vec_ops.py:L543 "
                        "(overflow_modes = [\"trunc\", \"saturate\"])",
    "hint_channel": "compile_hint_impl -> builder.create_annotation_mark "
                    "(third_party/ascend/ascend_ir.cc:L597-L603 落 annotation::MarkOp;"
                    "经 builder.py:L63-L86 的 setup_unified_builder 挂到主 builder)",
}

# ======================= m7/m8(非必需素材,顺带留证) =======================
b = FakeBuilder()
ful = make_tensor(mods, [4, 8], core.float32)
sub = make_tensor(mods, [1, 8], core.float32)
ins = mods.vec_ops.insert_slice(ful, sub, offsets=(2, 0), sizes=(1, 8), strides=(1, 1), _builder=b)
ext = mods.vec_ops.extract_slice(ful, offsets=(2, 0), sizes=(1, 8), strides=(1, 1), _builder=b)
el = mods.vec_ops.get_element(ful, (2, 3), _builder=b)
report["scenarios"]["m7_m8_slices"] = {
    "insert_slice_out_shape": list(ins.type.shape),
    "extract_slice_out_shape": list(ext.type.shape),
    "get_element_out_is_scalar": not hasattr(el.type, "shape"),
    "calls": repr_calls(b),
}

try:
    next(gen)
except StopIteration:
    pass

OUT = HERE / "builder_calls.json"
OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(json.dumps(report["scenarios"]["m6_width_contract"], ensure_ascii=False, indent=1, default=str))
print(json.dumps(report["scenarios"]["m9_flip"]["rows"], ensure_ascii=False, indent=1, default=str))
print(json.dumps(report["scenarios"]["m11_cast_tree"], ensure_ascii=False, indent=1, default=str))
print(json.dumps(report["scenarios"]["m12_overflow_mode"]["rows"], ensure_ascii=False, indent=1, default=str))
print("written:", OUT)
