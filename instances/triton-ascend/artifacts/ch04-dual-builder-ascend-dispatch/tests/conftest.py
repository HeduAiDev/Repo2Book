"""ch04 测试脚手架。

本章的机制（dossier.mechanisms m1-m6）落在 python/triton/compiler/code_generator.py
与 third_party/ascend/language/cann/extension/ 几个纯 Python 文件里，但它们真实的
`ir.builder`/`ascendnpu_ir_builder` 是编译期生成的 C++ 绑定——本仓的构建产物只在有
昇腾 NPU/CANN 工具链的机器上存在（见 INSTANCE.md「运行验证需昇腾 NPU/CANN
工具链，宿主无此环境」），host 既没有真绑定，也没有能替代它的等价 Python 实现
（pip 装的官方 triton 是不同版本，`ir.builder.__init__` 签名都不兼容，用它对照会
静默引入版本漂移，比没有验证更危险）。

所以这里在 sys.modules 里搭桩：
  - 真正是 C++ 绑定、host 无法拥有的名字（triton._C.libtriton[.ascend]）——
    换成本文件里的 FakeBuilder/FakeAscendBuilder 测试替身，只提供本章代码路径
    duck-type 用到的方法（get_insertion_point/restore_insertion_point/get_loc/
    set_loc/create_scope_op/... ）。
  - 真正是本仓 Python 源码、只是不在本章 dossier 范围内的名字（triton.language.core
    的 tensor/constexpr/is_builtin，triton.runtime.JITFunction，
    triton.compiler.errors.CompilationError）——按**规范模块名**把
    implementation/ 下（已减法）的同名文件加载进 sys.modules，不是另造桩。

然后把 implementation/ 下本章的六个文件也按规范模块名加载，使它们互相 import
（如 code_generator.py 的 `import triton.language.extra.cann.extension as
extension`）解析到彼此、而不是解析到 pip 装的官方 triton。
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

IMPL_DIR = Path(__file__).resolve().parent.parent / "implementation"


def _load(rel_path, modname):
    """按规范模块名加载精简版文件并登记进 sys.modules（含父包链接）。

    __init__.py 按包加载（submodule_search_locations 非空）——extension/__init__.py
    自己也用相对导入 `from .core import ...`，需要 __package__ 等于它自己这个包名，
    不是父包名，否则相对导入会解析错层级。
    """
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
    spec.loader.exec_module(mod)
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
# FakeBuilder / FakeAscendBuilder —— 站在真实 ir.builder / ascendnpu_ir_builder
# 位置上的测试替身。只提供本章代码路径实际调用到的方法，都做成「记录调用 + 返回
# 可预测的哨兵值」，不模拟任何 MLIR 语义（本章要看的是『调用被路由到哪个对象』，
# 不是『IR 长什么样』——IR dump 级别的验证按 INSTANCE.md 约束需要真机，不在本章
# 测试范围内）。
# ---------------------------------------------------------------------------
class FakeBlock:
    def __init__(self, region=None):
        self.region = region
        self.erased = False

    def erase(self):
        self.erased = True


class FakeRegion:
    pass


class FakeOp:
    """create_scope_op 的返回值替身：记录属性/结果类型，get_region/get_result 可用。"""

    def __init__(self, attrs, result_types):
        self.attrs = attrs
        self.result_types = result_types
        self.region = FakeRegion()

    def get_region(self, i):
        assert i == 0
        return self.region

    def get_result(self, i):
        return f"scope-result-{i}"


class _InsertionPointMixin:
    """两个 builder 都要能记 loc/插入点，供 create_builder_method_wrapper /
    visit_Call 的插入点接力逻辑（m6）验证『切 builder 前后是否真的搬运了』。"""

    def __init__(self, name):
        self.name = name
        self.calls = []
        self._loc = None
        self._ip = f"{name}-ip-0"
        self._ip_seq = 0

    def get_loc(self):
        self.calls.append(("get_loc",))
        return self._loc

    def set_loc(self, *args):
        self.calls.append(("set_loc", args))
        self._loc = args

    def get_insertion_point(self):
        self.calls.append(("get_insertion_point",))
        return self._ip

    def restore_insertion_point(self, ip):
        self.calls.append(("restore_insertion_point", ip))
        self._ip = ip

    def get_insertion_block(self):
        return f"{self.name}-block"

    def set_insertion_point_to_start(self, block):
        self.calls.append(("set_insertion_point_to_start", block))
        self._ip_seq += 1
        self._ip = f"{self.name}-ip-{self._ip_seq}"

    def set_insertion_point_to_end(self, block):
        self.calls.append(("set_insertion_point_to_end", block))
        self._ip_seq += 1
        self._ip = f"{self.name}-ip-{self._ip_seq}"

    def create_block(self):
        self.calls.append(("create_block",))
        return FakeBlock()

    def create_block_with_parent(self, region, args):
        self.calls.append(("create_block_with_parent", region, args))
        return FakeBlock(region)


class FakeAscendBuilder(_InsertionPointMixin):
    """站在 self.ascend_builder（ascendnpu_ir_builder(context, arch)）位置上。"""

    def __init__(self, context, arch):
        super().__init__("ascend")
        self.context = context
        self.arch = arch

    # sub_vec_id builtin 直接调用的 ascend 方法
    def create_get_sub_vec_id(self):
        self.calls.append(("create_get_sub_vec_id",))
        return "sub-vec-id-handle"

    # setup_unified_builder 会把下面几个方法「挂」到主 builder 上；这里是被委托的
    # 真身，create_builder_method_wrapper 调用时会经过 get_insertion_point/
    # restore_insertion_point/get_loc/set_loc 做插入点同步。
    def create_scope_op(self, attrs, result_types):
        self.calls.append(("create_scope_op", attrs, result_types))
        return FakeOp(attrs, result_types)

    def scope_return(self, values):
        self.calls.append(("scope_return", values))

    def get_t_core_type_attr_name(self):
        return "t_core_type"

    def get_t_core_type_cube_attr(self):
        return "cube"

    def get_t_core_type_vector_attr(self):
        return "vector"

    def create_copy_buffer(self, *args, **kwargs):
        self.calls.append(("create_copy_buffer", args, kwargs))


class FakeBuilder(_InsertionPointMixin):
    """站在 self.builder（ir.builder(context, compile_mode=...)）位置上。"""

    def __init__(self, context, compile_mode):
        super().__init__("main")
        self.context = context
        self.compile_mode = compile_mode
        self.options = None
        self.codegen_fns = None
        self.module_map = None

    def create_module(self):
        return "fake-module"

    def get_unit_attr(self):
        return "unit-attr"


# ---------------------------------------------------------------------------
# module_from_spec 加载顺序：先纯 Python 支撑层，再 ascend extension 包，
# 最后 triton.compiler.code_generator（依赖前两者）。
# ---------------------------------------------------------------------------
@pytest.fixture
def env():
    stubs = _Stubs()

    # ---- triton._C.libtriton[.ascend]：编译期 C++ 绑定的桩，工厂函数返回上面的假 builder ---- #
    stubs.mod("triton")
    stubs.mod("triton._C")
    libtriton = stubs.mod("triton._C.libtriton")
    libtriton.ir = types.SimpleNamespace(builder=FakeBuilder)
    ascend_pkg = stubs.mod("triton._C.libtriton.ascend")
    ascend_pkg.ir = types.SimpleNamespace(ascendnpu_ir_builder=FakeAscendBuilder)

    # ---- 按依赖顺序加载精简版（覆盖任何同名桩）---- #
    stubs.mod("triton.language")
    core = _load("python/triton/language/core.py", "triton.language.core")

    stubs.mod("triton.runtime")
    jit = _load("python/triton/runtime/jit.py", "triton.runtime.jit")
    # SOURCE: python/triton/runtime/__init__.py:L4 `from .jit import JITFunction`
    sys.modules["triton.runtime"].JITFunction = jit.JITFunction

    stubs.mod("triton.compiler")
    errors = _load("python/triton/compiler/errors.py", "triton.compiler.errors")

    stubs.mod("triton.language.extra")
    stubs.mod("triton.language.extra.cann")
    stubs.mod("triton.language.extra.cann.extension")
    ext_core = _load("third_party/ascend/language/cann/extension/core.py",
                      "triton.language.extra.cann.extension.core")
    ext_scope = _load("third_party/ascend/language/cann/extension/scope.py",
                       "triton.language.extra.cann.extension.scope")
    ext_codegen = _load("third_party/ascend/language/cann/extension/code_generator.py",
                         "triton.language.extra.cann.extension.code_generator")
    ext_dispatch = _load("third_party/ascend/language/cann/extension/dispatch.py",
                          "triton.language.extra.cann.extension.dispatch")
    ext_builder = _load("third_party/ascend/language/cann/extension/builder.py",
                         "triton.language.extra.cann.extension.builder")
    extension = _load("third_party/ascend/language/cann/extension/__init__.py",
                       "triton.language.extra.cann.extension")

    code_generator = _load("python/triton/compiler/code_generator.py",
                            "triton.compiler.code_generator")

    mods = types.SimpleNamespace(
        core=core,
        jit=jit,
        errors=errors,
        ext_core=ext_core,
        ext_scope=ext_scope,
        ext_codegen=ext_codegen,
        ext_dispatch=ext_dispatch,
        ext_builder=ext_builder,
        extension=extension,
        code_generator=code_generator,
    )
    try:
        yield mods
    finally:
        stubs.cleanup()
        for n in (
            "triton.language.core",
            "triton.runtime.jit",
            "triton.compiler.errors",
            "triton.language.extra.cann.extension.core",
            "triton.language.extra.cann.extension.scope",
            "triton.language.extra.cann.extension.code_generator",
            "triton.language.extra.cann.extension.dispatch",
            "triton.language.extra.cann.extension.builder",
            "triton.language.extra.cann.extension",
            "triton.compiler.code_generator",
        ):
            sys.modules.pop(n, None)


def make_generator(mods, options=None):
    """按真实 CodeGenerator.__init__ 签名构造一个实例，其余参数用空占位填充
    （本章精简版已删掉与双 builder 无关的字段，见 code_generator.py 的 SUBTRACTED 注释）。"""
    CodeGenerator = mods.code_generator.CodeGenerator
    return CodeGenerator(
        context="ctx", prototype=None, gscope={}, attributes={}, constants={},
        function_name="kernel", jit_fn=types.SimpleNamespace(src="<kernel src>"),
        options=options or types.SimpleNamespace(), codegen_fns={}, module_map={},
    )
