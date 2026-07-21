"""ch06 测试脚手架。

本章机制（dossier.mechanisms m1-m14）落在五个纯 Python 文件里：
  - third_party/ascend/language/cann/extension/{mem_ops,vec_ops,_utils,aux_ops}.py
    （四个 GM<->UB 索引搬运内建 + 片上向量算子词汇表 + cast 决策树）
  - third_party/ascend/language/kernels/gather.py（gather_2d_simd，见
    test_gather_2d_simd.py 自己的桩，不复用本文件的 FakeBuilder）

它们真实调用的 `ir.builder`/`ascendnpu_ir_builder` 都是编译期生成的 C++ 绑定——本仓
构建产物只在有昇腾 NPU/CANN 工具链的机器上存在（见 INSTANCE.md），host 既没有真绑定，
也没有等价的可信 Python 替代（pip 装的官方 triton 是不同版本/不同 fork，用它对照会
静默引入版本漂移，同 ch04/ch05 的处理原则）。

所以这里在 sys.modules 里搭桩，做法与 ch04/ch05 conftest.py 一致：
  - 真正是 C++ 绑定、host 无法拥有的名字（triton._C.libtriton）——换成本文件的
    FakeBuilder 测试替身（只提供本章代码路径 duck-type 用到的方法，都做成「记录调用
    + 返回可预测的哨兵值」，不模拟 MLIR/hivm 语义——IR dump 级别的验证需要真机，不在
    本章测试范围）。
  - 真正是本仓 Python 源码的名字——按规范模块名把 implementation/ 下（已减法）的
    同名文件加载进 sys.modules，不是另造桩。
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

IMPL_DIR = Path(__file__).resolve().parent.parent / "implementation"


def _spec_and_mod(rel_path, modname):
    path = IMPL_DIR / rel_path
    is_pkg = path.name == "__init__.py"
    spec = importlib.util.spec_from_file_location(
        modname, path, submodule_search_locations=[str(path.parent)] if is_pkg else None)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    if "." in modname:
        parent = modname.rsplit(".", 1)[0]
        if parent in sys.modules:
            setattr(sys.modules[parent], modname.rsplit(".", 1)[1], mod)
    return mod, spec


def _exec(mod, spec):
    spec.loader.exec_module(mod)
    return mod


def _load(rel_path, modname):
    mod, spec = _spec_and_mod(rel_path, modname)
    _exec(mod, spec)
    return mod


class _Stubs:
    def __init__(self):
        self.added = []

    def mod(self, dotted):
        parts = dotted.split(".")
        for i in range(len(parts)):
            name = ".".join(parts[: i + 1])
            if name not in sys.modules:
                m = types.ModuleType(name)
                sys.modules[name] = m
                self.added.append(name)
                if i > 0:
                    setattr(sys.modules[".".join(parts[:i])], parts[i], m)
        return sys.modules[dotted]

    def cleanup(self):
        for n in reversed(self.added):
            sys.modules.pop(n, None)


# ---------------------------------------------------------------------------
# FakeBuilder —— 站在真实 `ir.builder`/`ascendnpu_ir_builder` 位置上的测试替身。
# 只记录调用 + 返回可预测的哨兵值，不模拟真实 MLIR 语义。
# ---------------------------------------------------------------------------
class FakeBuilder:
    def __init__(self, is_simt=False):
        self.calls = []
        self._is_simt = is_simt

    # ---- dtype.to_ir 下沉 ----
    def get_int1_ty(self):
        return "int1_ty"

    def get_int8_ty(self):
        return "int8_ty"

    def get_int16_ty(self):
        return "int16_ty"

    def get_int32_ty(self):
        return "int32_ty"

    def get_int64_ty(self):
        return "int64_ty"

    def get_half_ty(self):
        return "fp16_ty"

    def get_bf16_ty(self):
        return "bf16_ty"

    def get_float_ty(self):
        return "fp32_ty"

    def get_ptr_ty(self, elem_ty_ir, address_space):
        return ("ptr_ty", elem_ty_ir, address_space)

    def get_block_ty(self, elem_ty_ir, shape):
        return ("block_ty", elem_ty_ir, tuple(shape))

    # ---- _convert_elem_to_ir_value ----
    def get_int32(self, v):
        return ("i32_const", v)

    def get_int64(self, v):
        return ("i64_const", v)

    def create_int_cast(self, handle, ty, signed):
        self.calls.append(("create_int_cast", handle, ty, signed))
        return ("int_cast", handle, ty, signed)

    # ---- mem_ops 四件套 ----
    def create_index_put(self, ptr_h, index_h, value_h, dim, index_boundary,
                         end_offset, start_offset, dst_stride):
        self.calls.append(("create_index_put", ptr_h, index_h, value_h, dim,
                           index_boundary, tuple(end_offset), tuple(start_offset), tuple(dst_stride)))
        return "index_put_handle"

    def create_gather_out_to_ub(self, src_h, index_h, index_boundary, dim,
                                src_stride, end_offset, start_offset, other_h):
        self.calls.append(("create_gather_out_to_ub", src_h, index_h, index_boundary, dim,
                           tuple(src_stride), tuple(end_offset), tuple(start_offset), other_h))
        return "gather_handle"

    def create_scatter_ub_to_out(self, ptr_h, value_h, index_h, index_boundary, dim,
                                 dst_stride, end_offset, start_offset):
        self.calls.append(("create_scatter_ub_to_out", ptr_h, value_h, index_h, index_boundary, dim,
                           tuple(dst_stride), tuple(end_offset), tuple(start_offset)))
        return "scatter_handle"

    def create_index_select_simd(self, src_h, index_h, dim, src_shape, src_offset, read_shape, return_shape):
        self.calls.append(("create_index_select_simd", src_h, index_h, dim,
                           tuple(src_shape), tuple(src_offset), tuple(read_shape), tuple(return_shape)))
        return "index_select_handle"

    # ---- vec_ops: insert/extract/get_element ----
    def create_insert_slice(self, ful_h, sub_h, offsets, sizes, strides):
        self.calls.append(("create_insert_slice", ful_h, sub_h, tuple(offsets), tuple(sizes), tuple(strides)))
        return "insert_slice_handle"

    def create_extract_slice(self, ful_h, offsets, sizes, strides):
        self.calls.append(("create_extract_slice", ful_h, tuple(offsets), tuple(sizes), tuple(strides)))
        return "extract_slice_handle"

    def create_extract_scalar(self, src_h, indice):
        self.calls.append(("create_extract_scalar", src_h, tuple(indice)))
        return "scalar_handle"

    # ---- vec_ops: flip ----
    def is_simt_mode(self):
        return self._is_simt

    def create_flip(self, handle, dim):
        self.calls.append(("create_flip", handle, dim))
        return f"flipped({handle},{dim})"

    def create_reduce(self, handle, axis):
        # 简化桩：真实 xor_sum 内部的 combine-region/generator 机制不在本章范围，
        # 见 implementation/python/triton/language/standard.py 顶部注释。
        self.calls.append(("create_reduce", handle, axis))
        return f"reduced({handle},axis={axis})"

    def create_xor(self, lhs_h, rhs_h):
        self.calls.append(("create_xor", lhs_h, rhs_h))
        return f"xor({lhs_h},{rhs_h})"

    def create_reshape(self, handle, shape, can_reorder):
        self.calls.append(("create_reshape", handle, tuple(shape), can_reorder))
        return f"reshaped({handle},{tuple(shape)})"

    def create_splat(self, handle, shape):
        self.calls.append(("create_splat", handle, tuple(shape)))
        return f"splat({handle},{tuple(shape)})"

    # ---- vec_ops: sort ----
    def create_sort(self, handle, dim, descending):
        self.calls.append(("create_sort", handle, dim, descending))
        return f"sorted({handle},{dim},{descending})"

    # ---- vec_ops: cast / ascend_cast_impl ----
    # 注意：is_compile_on_910_95 是 ascend_cast_impl 读的模块级硬件探测全局量
    # （见 python/triton/tools/get_ascend_devices.py 的桩），不是 builder 的方法——
    # 这里不提供 is_910_95() 方法，测试用 monkeypatch 模块全局来切换该分支。
    def get_null_value(self, ty):
        return ("null", ty)

    def create_bitcast(self, handle, ty):
        self.calls.append(("create_bitcast", handle, ty))
        return f"bitcast({handle},{ty})"

    def create_fp_trunc(self, handle, ty):
        self.calls.append(("create_fp_trunc", handle, ty))
        return f"fp_trunc({handle},{ty})"

    def create_fp_ext(self, handle, ty):
        self.calls.append(("create_fp_ext", handle, ty))
        return f"fp_ext({handle},{ty})"

    def create_fp_to_si(self, handle, ty):
        self.calls.append(("create_fp_to_si", handle, ty))
        return f"fp_to_si({handle},{ty})"

    def create_fp_to_ui(self, handle, ty):
        self.calls.append(("create_fp_to_ui", handle, ty))
        return f"fp_to_ui({handle},{ty})"

    def create_ui_to_fp(self, handle, ty):
        self.calls.append(("create_ui_to_fp", handle, ty))
        return f"ui_to_fp({handle},{ty})"

    def create_si_to_fp(self, handle, ty):
        self.calls.append(("create_si_to_fp", handle, ty))
        return f"si_to_fp({handle},{ty})"

    def create_fcmpUNE(self, lhs_h, rhs_h):
        self.calls.append(("create_fcmpUNE", lhs_h, rhs_h))
        return f"fcmpUNE({lhs_h},{rhs_h})"

    def create_icmpNE(self, lhs_h, rhs_h):
        self.calls.append(("create_icmpNE", lhs_h, rhs_h))
        return f"icmpNE({lhs_h},{rhs_h})"

    def get_int32(self, v):  # noqa: F811  (与上面 _convert_elem_to_ir_value 共用)
        return ("i32_const", v)

    def get_fp32(self, v):
        return ("fp32_const", v)

    # ---- compile_hint_impl ----
    def get_bool_attr(self, v):
        return ("bool_attr", v)

    def get_unit_attr(self):
        return "unit_attr"

    def get_int32_attr(self, v):
        return ("int32_attr", v)

    def get_str_attr(self, v):
        return ("str_attr", v)

    def get_i64_array_attr(self, v):
        return ("i64_array_attr", tuple(v))

    def create_annotation_mark(self, handle, name, value):
        self.calls.append(("create_annotation_mark", handle, name, value))


@pytest.fixture
def env():
    stubs = _Stubs()

    # ---- triton._C.libtriton：编译期 C++ 绑定的桩 ---- #
    stubs.mod("triton")
    stubs.mod("triton._C")
    libtriton = stubs.mod("triton._C.libtriton")

    class _RoundingMode:
        RTNE = "RTNE"
        RTZ = "RTZ"

    libtriton.ir = types.SimpleNamespace(builder=FakeBuilder, ROUNDING_MODE=_RoundingMode)

    # ---- triton.language：本章自己的最小 tl 子集(core/semantic/standard) ---- #
    stubs.mod("triton.language")
    tl_core = _load("python/triton/language/core.py", "triton.language.core")
    tl_semantic = _load("python/triton/language/semantic.py", "triton.language.semantic")
    tl_standard = _load("python/triton/language/standard.py", "triton.language.standard")
    tl_pkg = _load("python/triton/language/__init__.py", "triton.language")

    # ---- triton.runtime.interpreter：InterpreterBuilder 类型标识 ---- #
    stubs.mod("triton.runtime")
    _load("python/triton/runtime/interpreter.py", "triton.runtime.interpreter")

    # ---- triton.tools.get_ascend_devices：is_compile_on_910_95 硬件探测的桩 ---- #
    stubs.mod("triton.tools")
    _load("python/triton/tools/get_ascend_devices.py", "triton.tools.get_ascend_devices")

    # ---- third_party/ascend/.../extension：mem_ops/vec_ops 与它们唯一的跨文件
    # 依赖 _utils/aux_ops/__init__ ---- #
    stubs.mod("triton.language.extra")
    stubs.mod("triton.language.extra.cann")
    ext_init_mod, ext_init_spec = _spec_and_mod(
        "third_party/ascend/language/cann/extension/__init__.py", "triton.language.extra.cann.extension")
    _exec(ext_init_mod, ext_init_spec)

    utils_mod = _load(
        "third_party/ascend/language/cann/extension/_utils.py", "triton.language.extra.cann.extension._utils")
    aux_ops_mod = _load(
        "third_party/ascend/language/cann/extension/aux_ops.py", "triton.language.extra.cann.extension.aux_ops")
    mem_ops_mod = _load(
        "third_party/ascend/language/cann/extension/mem_ops.py", "triton.language.extra.cann.extension.mem_ops")
    vec_ops_mod = _load(
        "third_party/ascend/language/cann/extension/vec_ops.py", "triton.language.extra.cann.extension.vec_ops")

    mods = types.SimpleNamespace(
        tl=tl_pkg,
        core=tl_core,
        semantic=tl_semantic,
        standard=tl_standard,
        utils=utils_mod,
        aux_ops=aux_ops_mod,
        mem_ops=mem_ops_mod,
        vec_ops=vec_ops_mod,
    )
    try:
        yield mods
    finally:
        stubs.cleanup()
        for n in (
            "triton.language.core",
            "triton.language.semantic",
            "triton.language.standard",
            "triton.language",
            "triton.runtime.interpreter",
            "triton.tools.get_ascend_devices",
            "triton.language.extra.cann.extension._utils",
            "triton.language.extra.cann.extension.aux_ops",
            "triton.language.extra.cann.extension.mem_ops",
            "triton.language.extra.cann.extension.vec_ops",
            "triton.language.extra.cann.extension",
        ):
            sys.modules.pop(n, None)


def make_tensor(mods, shape, dtype):
    """构造一个 shape/dtype 已知的 UB tile(block tensor)，handle 用可读字符串代替。"""
    tl = mods.core
    bt = tl.block_type(dtype, list(shape))
    return tl.tensor(f"h[{dtype.name}{list(shape)}]", bt)


def make_gm_ptr(mods, dtype, name="ptr"):
    """构造一个 GM 裸指针 tensor：type 是 pointer_type(scalar，非 block)。"""
    tl = mods.core
    pt = tl.pointer_type(dtype)
    return tl.tensor(f"h[{name}]", pt)
