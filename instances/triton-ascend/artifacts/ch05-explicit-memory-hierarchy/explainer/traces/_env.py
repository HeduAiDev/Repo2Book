"""Shared loader for ch05 explainer driver scripts.

Reuses the exact same sys.modules staging as tests/conftest.py (FakeBuilder
standing in for the C++ ir.builder / ascendnpu_ir_builder that the host lacks).
Returns an `env` namespace (tl / bl / al) plus the FakeBuilder class so each
driver can allocate buffers, run the reduced al.copy / al.fixpipe / bl.alloc,
and read the recorded builder.calls as the trace.
"""
import importlib.util
import sys
import types
from pathlib import Path

CH = Path(__file__).resolve().parent.parent.parent
IMPL_DIR = CH / "implementation"


def _spec_and_mod(rel_path, modname):
    path = IMPL_DIR / rel_path
    is_pkg = path.name == "__init__.py"
    spec = importlib.util.spec_from_file_location(
        modname, path,
        submodule_search_locations=[str(path.parent)] if is_pkg else None)
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


class _FakeAddressSpace:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"AddressSpace.{self.name}"


def _make_address_space_enum():
    cls = _FakeAddressSpace
    for name in ("Zero", "GM", "L1", "L0A", "L0B", "L0C", "UB"):
        setattr(cls, name, cls(name))
    return cls


def _make_sentinel_enum(*names):
    ns = types.SimpleNamespace()
    for n in names:
        setattr(ns, n, n)
    return ns


class FakeAscendIrBuilderType:
    pass


class FakeBuilder:
    def __init__(self, is_910_95=True):
        self.calls = []
        self._is_910_95 = is_910_95

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

    def get_target_attribute(self, address_space_value):
        self.calls.append(("get_target_attribute", address_space_value))
        return f"attr({address_space_value.name})"

    def get_null_attr(self):
        return "null-attr"

    def get_unit_attr(self):
        return "unit-attr"

    def get_str_array_attr(self, values):
        return tuple(values)

    def get_buffer_ty(self, shape, element_ty_ir, addr_space_attr):
        return ("buffer_ty", tuple(shape), element_ty_ir, addr_space_attr)

    def get_buffer_ty_with_strides(self, shape, element_ty_ir, strides, addr_space_attr):
        return ("buffer_ty_strided", tuple(shape), element_ty_ir, tuple(strides), addr_space_attr)

    def alloc(self, memref_ty):
        self.calls.append(("alloc", memref_ty))
        return f"handle#{len(self.calls)}"

    def create_annotation_mark(self, handle, name, value):
        self.calls.append(("create_annotation_mark", handle, name, value))

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

    def is_910_95(self):
        return self._is_910_95

    def create_copy_buffer(self, src_handle, dst_handle):
        self.calls.append(("create_copy_buffer", src_handle, dst_handle))

    def create_fixpipe(self, src_handle, dst_handle, dma_mode, dual_dst_mode, pre_quant_mode, pre_relu_mode):
        self.calls.append((
            "create_fixpipe", src_handle, dst_handle, dma_mode, dual_dst_mode, pre_quant_mode, pre_relu_mode,
        ))


def load_env():
    """Stage sys.modules exactly like conftest.env and return (env, FakeBuilder)."""
    stubs = _Stubs()

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

    stubs.mod("triton.language")
    _load("python/triton/language/_utils.py", "triton.language._utils")
    tl = _load("python/triton/language/core.py", "triton.language.core")

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

    stubs.mod("triton.language.extra")
    stubs.mod("triton.language.extra.cann")
    ext_init_mod, ext_init_spec = _spec_and_mod(
        "third_party/ascend/language/cann/extension/__init__.py", "triton.language.extra.cann.extension")
    ext_core_mod, ext_core_spec = _spec_and_mod(
        "third_party/ascend/language/cann/extension/core.py", "triton.language.extra.cann.extension.core")
    ext_sem_mod, ext_sem_spec = _spec_and_mod(
        "third_party/ascend/language/cann/extension/semantic.py", "triton.language.extra.cann.extension.semantic")
    _exec(ext_sem_mod, ext_sem_spec)
    _exec(ext_core_mod, ext_core_spec)
    _exec(ext_init_mod, ext_init_spec)

    return types.SimpleNamespace(tl=tl, bl=bl, al=ext_init_mod), FakeBuilder
