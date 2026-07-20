"""ch05 测试脚手架。

本章的机制（dossier.mechanisms M1-M7）落在四个纯 Python 文件里：
  - python/triton/extension/buffer/language/{core,semantic}.py（buffer 语言：
    address_space/buffer_type/buffer + alloc/to_buffer/to_tensor/subview）
  - third_party/ascend/language/cann/extension/{core,semantic}.py（al.copy /
    al.copy_from_ub_to_l1 / al.fixpipe 与四组 Fixpipe 模式枚举）

但它们真实调用的 `ir.builder`/`ascendnpu_ir_builder`/`ascend_ir.AddressSpace` 都是
编译期生成的 C++ 绑定——本仓构建产物只在有昇腾 NPU/CANN 工具链的机器上存在（见
INSTANCE.md「运行验证需昇腾 NPU/CANN 工具链，宿主无此环境」），host 既没有真绑定，
也没有等价的可信 Python 替代（pip 装的官方 triton 是不同版本/不同 fork，用它对照
会静默引入版本漂移）。

所以这里在 sys.modules 里搭桩，做法与 ch04 conftest.py 一致：
  - 真正是 C++ 绑定、host 无法拥有的名字（triton._C.libtriton[.ascend]）——换成
    本文件的 FakeBuilder 测试替身（只提供本章代码路径 duck-type 用到的方法，都做成
    「记录调用 + 返回可预测的哨兵值」，不模拟 MLIR/hivm 语义——IR dump 级别的验证
    需要真机，不在本章测试范围）与 FakeAddressSpace/Fixpipe*（C++ 枚举的桩，成员
    名与 pybind 真导出的清单一致：L1/UB/L0A/L0B/L0C，见 ascend_ir.cc:L412-417）。
  - 真正是本仓 Python 源码的名字——按规范模块名把 implementation/ 下（已减法）的
    同名文件加载进 sys.modules，不是另造桩。

本章两对文件（buffer core↔semantic、ascend core↔semantic）在真实源码里都存在
"父模块引用尚在加载中的自身/兄弟模块"的循环导入——buffer/core.py 用
`importlib.import_module(".semantic", ...)` 延迟到文件末尾才拉 semantic，
ascend/semantic.py 顶部 `import triton.language.extra.cann.extension as al`
引用的正是自己所在的包。真实 Python 的处理方式是"模块对象在执行体之前就登记进
sys.modules"，本文件用 `_load_deferred`/`_exec` 两阶段显式复现这个次序，而不是
依赖真实文件系统 __path__ 的自动发现（同 ch04，manual 优先，避免宿主 sys.path
上任何同名包干扰）。
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

IMPL_DIR = Path(__file__).resolve().parent.parent / "implementation"


def _spec_and_mod(rel_path, modname):
    """构造 spec + 空壳 module 并登记进 sys.modules（含父包属性挂接），但不执行。"""
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
    """登记 + 立即执行（用于没有循环依赖风险的文件）。"""
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
# ascend_ir 桩：AddressSpace 反射源。成员名对齐的是 **pybind 真导出的清单**——
# third_party/ascend/ascend_ir.cc:L412-417 的 py::enum_<hivm::AddressSpace> 只
# .value() 了 L1/UB/L0A/L0B/L0C 这 5 个。HIVMAttrs.td:L188-194 里另有 Zero(0) 与
# GM(1)，但它们止步于 C++ 侧、不进 Python，所以桩里也不能有：照抄 .td 的 7 个就是
# 让替身凭空造出真 Python 层不存在的名字。
# 外加四组 Fixpipe 模式枚举的桩值。
# ---------------------------------------------------------------------------
class _FakeAddressSpace:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"AddressSpace.{self.name}"


def _make_address_space_enum():
    ns = types.SimpleNamespace()
    cls = _FakeAddressSpace
    for name in ("L1", "UB", "L0A", "L0B", "L0C"):
        setattr(cls, name, cls(name))
    return cls


def _make_sentinel_enum(*names):
    ns = types.SimpleNamespace()
    for n in names:
        setattr(ns, n, n)
    return ns


class FakeAscendIrBuilderType:
    """只作 ascend_ir.ascendnpu_ir_builder 类型注解的占位——从不被实例化。"""


# ---------------------------------------------------------------------------
# FakeBuilder —— 站在真实 `ir.builder` / `ascendnpu_ir_builder` 位置上的测试替身。
# 本章不模拟 ch04 的双 builder 分发（那是 ch04 的机制），只用一个对象承接本章
# 所有会被调用到的方法：记录调用 + 返回可预测的哨兵值。
# ---------------------------------------------------------------------------
class FakeBuilder:
    def __init__(self, is_910_95=True):
        self.calls = []
        self._is_910_95 = is_910_95

    # ---- dtype.to_ir 用到的六个标量类型下沉 ----
    def get_int1_ty(self):
        return "int1_ty"

    def get_int16_ty(self):
        return "int16_ty"

    def get_int32_ty(self):
        return "int32_ty"

    def get_half_ty(self):
        return "fp16_ty"

    def get_bf16_ty(self):
        return "bf16_ty"

    def get_float_ty(self):
        return "fp32_ty"

    # ---- 地址空间 ----
    def get_target_attribute(self, address_space_value):
        self.calls.append(("get_target_attribute", address_space_value))
        return f"attr({address_space_value.name})"

    def get_null_attr(self):
        return "null-attr"

    def get_unit_attr(self):
        return "unit-attr"

    def get_str_array_attr(self, values):
        return tuple(values)

    # ---- buffer_type.to_ir / alloc ----
    def get_buffer_ty(self, shape, element_ty_ir, addr_space_attr):
        return ("buffer_ty", tuple(shape), element_ty_ir, addr_space_attr)

    def get_buffer_ty_with_strides(self, shape, element_ty_ir, strides, addr_space_attr):
        return ("buffer_ty_strided", tuple(shape), element_ty_ir, tuple(strides), addr_space_attr)

    def alloc(self, memref_ty):
        self.calls.append(("alloc", memref_ty))
        return f"handle#{len(self.calls)}"

    def create_annotation_mark(self, handle, name, value):
        self.calls.append(("create_annotation_mark", handle, name, value))

    # ---- buffer<->tensor 桥 ----
    def create_bind_buffer(self, tensor_handle, buffer_handle):
        self.calls.append(("create_bind_buffer", tensor_handle, buffer_handle))

    def to_buffer(self, tensor_handle, addr_space_attr):
        self.calls.append(("to_buffer", tensor_handle, addr_space_attr))
        return f"buf-handle({tensor_handle})"

    def to_tensor(self, memref_handle, writable):
        self.calls.append(("to_tensor", memref_handle, writable))
        return f"tensor-handle({memref_handle})"

    def create_convert_layout(self, handle, ty):
        self.calls.append(("create_convert_layout", handle, ty))
        return f"converted({handle})"

    def subview(self, handle, offsets, sizes, strides):
        self.calls.append(("subview", handle, offsets, sizes, strides))
        return f"subview({handle},{offsets},{sizes},{strides})"

    # ---- al.copy / al.copy_from_ub_to_l1 ----
    def is_910_95(self):
        return self._is_910_95

    def create_copy_buffer(self, src_handle, dst_handle):
        self.calls.append(("create_copy_buffer", src_handle, dst_handle))

    # ---- al.fixpipe ----
    def create_fixpipe(self, src_handle, dst_handle, dma_mode, dual_dst_mode, pre_quant_mode, pre_relu_mode):
        self.calls.append((
            "create_fixpipe", src_handle, dst_handle, dma_mode, dual_dst_mode, pre_quant_mode, pre_relu_mode,
        ))


@pytest.fixture
def env():
    stubs = _Stubs()

    # ---- triton._C.libtriton[.ascend]：编译期 C++ 绑定的桩 ---- #
    stubs.mod("triton")
    stubs.mod("triton._C")
    libtriton = stubs.mod("triton._C.libtriton")
    _Ann = type("_Ann", (), {})
    libtriton.ir = types.SimpleNamespace(builder=FakeBuilder, attribute=_Ann, type=_Ann)

    ascend_pkg = stubs.mod("triton._C.libtriton.ascend")
    ascend_pkg.ir = types.SimpleNamespace(
        AddressSpace=_make_address_space_enum(),
        FixpipeDMAMode=_make_sentinel_enum("NZ2DN", "NZ2ND", "NZ2NZ"),
        FixpipeDualDstMode=_make_sentinel_enum("NO_DUAL", "COLUMN_SPLIT", "ROW_SPLIT"),
        FixpipePreQuantMode=_make_sentinel_enum("NO_QUANT", "F322BF16", "F322F16", "S322I8"),
        FixpipePreReluMode=_make_sentinel_enum("LEAKY_RELU", "NO_RELU", "NORMAL_RELU", "P_RELU"),
        ascendnpu_ir_builder=FakeAscendIrBuilderType,
    )

    # ---- triton.language.core / _utils（tl 的最小子集，见 implementation/ 本文件的
    # 顶部注释：与本章内存层级/copy/fixpipe 无关的算子/类型目录已省略）---- #
    stubs.mod("triton.language")
    _load("python/triton/language/_utils.py", "triton.language._utils")
    tl = _load("python/triton/language/core.py", "triton.language.core")

    # ---- python/triton/extension/buffer/language：core<->semantic 互相 import，
    # 用三阶两段式（先把 __init__/core/semantic 都登记进 sys.modules 但不执行，
    # 再按 semantic→core→__init__ 的次序真正执行）复现真实的循环导入次序——
    # semantic.py 的 `from . import core as bl` 需要包本身（__init__）已在
    # sys.modules 里才能被当成"包"解析，否则 Python 会报
    # "'triton.extension.buffer' is not a package"（见文件顶部注释）---- #
    stubs.mod("triton.extension")
    stubs.mod("triton.extension.buffer")
    bl_init_mod, bl_init_spec = _spec_and_mod(
        "python/triton/extension/buffer/language/__init__.py", "triton.extension.buffer.language")
    bl_core_mod, bl_core_spec = _spec_and_mod(
        "python/triton/extension/buffer/language/core.py", "triton.extension.buffer.language.core")
    bl_sem_mod, bl_sem_spec = _spec_and_mod(
        "python/triton/extension/buffer/language/semantic.py", "triton.extension.buffer.language.semantic")
    _exec(bl_sem_mod, bl_sem_spec)
    _exec(bl_core_mod, bl_core_spec)
    bl = _exec(bl_init_mod, bl_init_spec)

    # ---- third_party/ascend/.../extension：__init__ 顶层触发 core，core 顶层触发
    # semantic，semantic 顶层反过来 import 回自己所在的包（as al）——三阶两段式 ---- #
    stubs.mod("triton.language.extra")
    stubs.mod("triton.language.extra.cann")
    ext_init_mod, ext_init_spec = _spec_and_mod(
        "third_party/ascend/language/cann/extension/__init__.py", "triton.language.extra.cann.extension")
    ext_core_mod, ext_core_spec = _spec_and_mod(
        "third_party/ascend/language/cann/extension/core.py", "triton.language.extra.cann.extension.core")
    ext_sem_mod, ext_sem_spec = _spec_and_mod(
        "third_party/ascend/language/cann/extension/semantic.py", "triton.language.extra.cann.extension.semantic")
    _exec(ext_sem_mod, ext_sem_spec)   # al.ascend_address_space 只在调用期才被用到
    _exec(ext_core_mod, ext_core_spec)  # 此时 semantic 已完整加载
    _exec(ext_init_mod, ext_init_spec)  # 此时 core 已完整加载

    mods = types.SimpleNamespace(
        tl=tl,
        bl=bl,
        bl_core=bl_core_mod,
        bl_semantic=bl_sem_mod,
        al=ext_init_mod,
        al_core=ext_core_mod,
        al_semantic=ext_sem_mod,
    )
    try:
        yield mods
    finally:
        stubs.cleanup()
        for n in (
            "triton.language.core",
            "triton.language._utils",
            "triton.extension.buffer.language.core",
            "triton.extension.buffer.language.semantic",
            "triton.extension.buffer.language",
            "triton.language.extra.cann.extension.core",
            "triton.language.extra.cann.extension.semantic",
            "triton.language.extra.cann.extension",
        ):
            sys.modules.pop(n, None)
