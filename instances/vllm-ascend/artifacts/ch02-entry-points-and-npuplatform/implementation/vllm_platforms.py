"""Subtract-only companion — vLLM 侧的平台选择 + 懒加载单例。

规范源码：vllm/platforms/__init__.py

resolve_current_platform_cls_qualname：把 builtin(cuda/rocm/tpu/xpu/cpu) 与 OOT 插件
用 chain() 串成一个循环逐个 func() 收集‘激活者’，再用集合交集分成两组，elif 链里
OOT 一组被先判 → 同时激活时 OOT 顶替 builtin。
__getattr__('current_platform')：首次访问时懒加载、缓存进 _current_platform 单例。

注：跨文件 import 路径已由真实的 `from vllm.plugins import …` / `from vllm.utils.import_utils
import …` 改写成本目录的扁平模块名，以便精简版独立可跑；符号与控制流保持一致。
"""
from itertools import chain

# 真实为 `from vllm.plugins import load_plugins_by_group`
from vllm_plugins import load_plugins_by_group
# 真实为 `from vllm.utils.import_utils import resolve_obj_by_qualname`
from vllm_import_utils import resolve_obj_by_qualname

# SOURCE: vllm/platforms/__init__.py:L40 —— PLATFORM_PLUGINS_GROUP 常量
PLATFORM_PLUGINS_GROUP = "vllm.platform_plugins"


# SOURCE: vllm/platforms/__init__.py:L60-L113
def cuda_platform_plugin() -> str | None:
    # SUBTRACTED: 原函数用 pynvml 探测是否有 GPU、排除 cpu-build/Jetson 等边界，在场返回
    #   "vllm.platforms.cuda.CudaPlatform"、否则 None。host 无 GPU → None。本章只需它的
    #   ‘形态’：builtin 插件会真去问硬件。原 vllm/platforms/__init__.py:L60-L113
    return None


# SOURCE: vllm/platforms/__init__.py（rocm_platform_plugin）
def rocm_platform_plugin() -> str | None:
    # SUBTRACTED: 同 cuda——探测 ROCm 硬件，不在场返回 None。
    return None


# SOURCE: vllm/platforms/__init__.py（tpu_platform_plugin）
def tpu_platform_plugin() -> str | None:
    # SUBTRACTED: 同上——探测 TPU，不在场返回 None。
    return None


# SOURCE: vllm/platforms/__init__.py（xpu_platform_plugin）
def xpu_platform_plugin() -> str | None:
    # SUBTRACTED: 同上——探测 Intel XPU，不在场返回 None。
    return None


# SOURCE: vllm/platforms/__init__.py:L190-L200（cpu_platform_plugin）
def cpu_platform_plugin() -> str | None:
    # SUBTRACTED: 原函数探测 cpu-build；本章把它也降为占位 None，让 OOT(ascend) 成为唯一激活者。
    return None


# SOURCE: vllm/platforms/__init__.py:L203-L209
builtin_platform_plugins = {
    "tpu": tpu_platform_plugin,
    "cuda": cuda_platform_plugin,
    "rocm": rocm_platform_plugin,
    "xpu": xpu_platform_plugin,
    "cpu": cpu_platform_plugin,
}


# SOURCE: vllm/platforms/__init__.py:L212-L252
def resolve_current_platform_cls_qualname() -> str:
    platform_plugins = load_plugins_by_group(PLATFORM_PLUGINS_GROUP)

    activated_plugins = []

    for name, func in chain(builtin_platform_plugins.items(), platform_plugins.items()):
        try:
            assert callable(func)
            platform_cls_qualname = func()
            if platform_cls_qualname is not None:
                activated_plugins.append(name)
        except Exception:
            pass

    activated_builtin_plugins = list(
        set(activated_plugins) & set(builtin_platform_plugins.keys())
    )
    activated_oot_plugins = list(set(activated_plugins) & set(platform_plugins.keys()))

    if len(activated_oot_plugins) >= 2:
        raise RuntimeError(
            "Only one platform plugin can be activated, but got: "
            f"{activated_oot_plugins}"
        )
    elif len(activated_oot_plugins) == 1:
        platform_cls_qualname = platform_plugins[activated_oot_plugins[0]]()
        # SUBTRACTED: logger.info("Platform plugin %s is activated", ...)
    elif len(activated_builtin_plugins) >= 2:
        raise RuntimeError(
            "Only one platform plugin can be activated, but got: "
            f"{activated_builtin_plugins}"
        )
    elif len(activated_builtin_plugins) == 1:
        platform_cls_qualname = builtin_platform_plugins[activated_builtin_plugins[0]]()
        # SUBTRACTED: logger.debug("Automatically detected platform %s.", ...)
    else:
        platform_cls_qualname = "vllm.platforms.interface.UnspecifiedPlatform"
        # SUBTRACTED: logger.debug("No platform detected, ... UnspecifiedPlatform")
    return platform_cls_qualname


# SOURCE: vllm/platforms/__init__.py:L257
_current_platform = None


# SOURCE: vllm/platforms/__init__.py:L262-L284
def __getattr__(name: str):
    if name == "current_platform":
        # lazy init current_platform.
        # 1. out-of-tree platform plugins need `from vllm.platforms import
        #    Platform` so that they can inherit `Platform` class. Therefore,
        #    we cannot resolve `current_platform` during the import of
        #    `vllm.platforms`.
        # 2. when users use out-of-tree platform plugins, they might run
        #    `import vllm`, some vllm internal code might access
        #    `current_platform` during the import, and we need to make sure
        #    `current_platform` is only resolved after the plugins are loaded
        #    (we have tests for this, if any developer violate this, they will
        #    see the test failures).
        global _current_platform
        if _current_platform is None:
            platform_cls_qualname = resolve_current_platform_cls_qualname()
            _current_platform = resolve_obj_by_qualname(platform_cls_qualname)()
            # SUBTRACTED: _init_trace = "".join(traceback.format_stack()) —— 仅调试时
            #   回溯‘平台在哪被首次解析’，与懒加载-缓存-返回单例语义无关。
            #   原 vllm/platforms/__init__.py:L280-L282
        return _current_platform
    elif name in globals():
        return globals()[name]
    else:
        raise AttributeError(f"No attribute named '{name}' exists in {__name__}.")
