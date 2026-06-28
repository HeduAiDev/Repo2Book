"""Subtract-only companion — 两组 entry point 的目标函数。

规范源码：vllm_ascend/__init__.py

两组 entry point 的‘回调时机/副作用’截然不同：
  platform_plugins 的 register() 是纯函数（只返回字符串、零副作用）——它在‘平台选择’这个
    极早期阶段被调用，此刻还不该 import torch_npu。
  general_plugins 的 4 个 register_* 由 vLLM 的 load_general_plugins() 在 engine-core 子进程里
    func() 调用，是有副作用的真注册，且每个都先 _ensure_global_patch() 把 platform 段
    monkey-patch 也补到子进程（幂等，靠 _GLOBAL_PATCH_APPLIED 标志位）。
"""

# SUBTRACTED: 文件顶部 `import vllm_ascend.logger`（原 vllm_ascend/__init__.py:L18）

# SOURCE: vllm_ascend/__init__.py:L20
_GLOBAL_PATCH_APPLIED = False


# SOURCE: vllm_ascend/__init__.py:L23-L37
def _ensure_global_patch():
    """Apply process-wide vLLM patches before engine-core initialization.

    vLLM loads general plugins in engine-core subprocesses. E2E test
    conftest hooks do not run there, so global patches that affect scheduler
    and engine code must also be applied through these plugin entry points.
    """
    global _GLOBAL_PATCH_APPLIED
    if _GLOBAL_PATCH_APPLIED:
        return

    # SUBTRACTED: `from vllm_ascend.utils import adapt_patch` + `adapt_patch(is_global_patch=True)`
    #   —— adapt_patch 的两段式(platform/worker) monkey-patch 是 ch03 主题，依赖 torch_npu，
    #   host 跑不动。本章只点出 register_* 会触发 platform 段。原 vllm_ascend/__init__.py:L34-L35
    _GLOBAL_PATCH_APPLIED = True


# SOURCE: vllm_ascend/__init__.py:L40-L43
def register():
    """Register the NPU platform."""

    return "vllm_ascend.platform.NPUPlatform"


# SOURCE: vllm_ascend/__init__.py:L46-L51
def register_connector():
    _ensure_global_patch()

    # SUBTRACTED: `from vllm_ascend.distributed.kv_transfer import register_connector` +
    #   register_connector() —— 真注册 KV connector，依赖 torch_npu/分布式，属后续章节。
    #   原 vllm_ascend/__init__.py:L49-L51


# SOURCE: vllm_ascend/__init__.py:L54-L61
def register_model_loader():
    _ensure_global_patch()

    # SUBTRACTED: netloader/rfork 两个 model loader 的真注册（依赖下游模块）。
    #   原 vllm_ascend/__init__.py:L57-L61


# SOURCE: vllm_ascend/__init__.py:L64-L69
def register_service_profiling():
    _ensure_global_patch()

    # SUBTRACTED: generate_service_profiling_config() 真生成 profiling 配置。
    #   原 vllm_ascend/__init__.py:L67-L69


# SOURCE: vllm_ascend/__init__.py:L72-L75
def register_model():
    # SUBTRACTED: from .models import register_model; register_model() —— 注册自定义模型。
    #   注意 register_model 是 general_plugins 里**唯一不先 _ensure_global_patch** 的回调。
    #   原 vllm_ascend/__init__.py:L73-L75
    pass
