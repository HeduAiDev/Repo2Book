"""ch18 测试脚手架：在 sys.modules 桩掉 vllm/vllm_ascend 的重运行时依赖（NPU/CANN host 不可跑），
再把（已减法的）implementation/ 模块按**规范模块名**注册进去，让它们彼此 import 解析到精简版本身。

本章三处核心都是纯 Python，故可在 host 验证与真实仓一致的可观察控制流：
  (1) NPUPlatform.get_attn_backend_cls 的三元 key → 4 后端路由；
  (2) @register_backend(CUSTOM,"ASCEND") 写进 _ATTN_OVERRIDES 的占位；
  (3) get_name 伪装 / 静态契约 4 个 @abstractmethod / get_impl·builder_cls 按 enable_cp 分流 /
      selector 点分路径 → resolve_obj_by_qualname → 后端类的端到端解析。
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

IMPL_DIR = Path(__file__).resolve().parent.parent / "implementation"


def _load(filename, modname):
    """按规范模块名加载精简版文件并登记进 sys.modules（含父包链接）。"""
    spec = importlib.util.spec_from_file_location(modname, IMPL_DIR / filename)
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


@pytest.fixture
def env():
    """搭桩 + 加载精简版模块，返回 (modules, knobs)。"""
    stubs = _Stubs()

    knobs = types.SimpleNamespace(
        use_v2_model_runner=True,   # 控 get_name 伪装开关
        cp_enabled=False,           # 控 get_impl/builder 的 CP 分流
    )

    # ---- vllm.envs：VLLM_USE_V2_MODEL_RUNNER 开关（get_name 伪装用）---- #
    # 用 ModuleType 子类的 __getattr__ 把环境量动态接到旋钮上，
    # 让 `import vllm.envs as envs_vllm; envs_vllm.VLLM_USE_V2_MODEL_RUNNER` 实时取值。
    stubs.mod("vllm")

    class _EnvsModule(types.ModuleType):
        def __getattr__(self, name):
            if name == "VLLM_USE_V2_MODEL_RUNNER":
                return knobs.use_v2_model_runner
            raise AttributeError(name)

    envs = _EnvsModule("vllm.envs")
    sys.modules["vllm.envs"] = envs
    setattr(sys.modules["vllm"], "envs", envs)
    stubs.added.append("vllm.envs")

    # ---- vllm_ascend.attention.utils：enable_cp 桩（CP 分流用）---- #
    stubs.mod("vllm_ascend")
    stubs.mod("vllm_ascend.attention")
    autils = stubs.mod("vllm_ascend.attention.utils")
    autils.enable_cp = lambda: knobs.cp_enabled

    # ---- 按依赖顺序加载精简版（覆盖任何同名桩）---- #
    stubs.mod("vllm")
    stubs.mod("vllm.v1")
    stubs.mod("vllm.v1.attention")
    stubs.mod("vllm.v1.attention.backends")

    backend = _load("backend.py", "vllm.v1.attention.backend")
    registry = _load("registry.py", "vllm.v1.attention.backends.registry")
    flash_attn = _load("flash_attn.py", "vllm.v1.attention.backends.flash_attn")
    selector = _load("selector.py", "vllm.v1.attention.selector")
    attention_v1 = _load("attention_v1.py", "vllm_ascend.attention.attention_v1")
    platform = _load("platform.py", "vllm_ascend.platform")

    # ---- vllm.platforms.current_platform = 昇腾平台（selector 端到端解析用）---- #
    plat = stubs.mod("vllm.platforms")
    NPUPlatform = platform.NPUPlatform
    NPUPlatform.device_name = "npu"
    plat.current_platform = NPUPlatform

    mods = types.SimpleNamespace(
        backend=backend,
        registry=registry,
        flash_attn=flash_attn,
        selector=selector,
        attention_v1=attention_v1,
        platform=platform,
    )
    try:
        yield mods, knobs
    finally:
        stubs.cleanup()
        for n in (
            "vllm.envs",
            "vllm_ascend.attention.utils",
            "vllm.v1.attention.backend",
            "vllm.v1.attention.backends.registry",
            "vllm.v1.attention.backends.flash_attn",
            "vllm.v1.attention.selector",
            "vllm_ascend.attention.attention_v1",
            "vllm_ascend.platform",
            "vllm.platforms",
        ):
            sys.modules.pop(n, None)


class _Cfg:
    """桩 AttentionSelectorConfig：基座只声明 use_mla/use_sparse；use_compress 仅在更新版 vLLM 上有。

    用普通类（保留默认 identity __hash__）而非 SimpleNamespace —— selector._cached_get_attn_backend
    带 @cache，要求 config 可哈希。
    """

    def __init__(self, use_mla=False, use_sparse=False, use_compress=None):
        self.use_mla = use_mla
        self.use_sparse = use_sparse
        if use_compress is not None:
            self.use_compress = use_compress


def make_cfg(use_mla=False, use_sparse=False, use_compress=None):
    return _Cfg(use_mla=use_mla, use_sparse=use_sparse, use_compress=use_compress)
