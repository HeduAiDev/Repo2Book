"""接入点：把 ascend 的 sleep allocator 接进 vLLM 的 ModelConfig 校验门。

只做减法的忠实精简版。这是 ch03「五种重绑技法」里的库函数替换（换 vllm.config.model
模块里一个函数名的指向）——技法本身见 ch03，本章只看它的两个特殊之处：
  ① hasattr 守护 = 前向兼容 fallback：上游若有 is_cumem_allocator_available 校验函数
     就替换它，没有就 no-op；
  ② 当前 pin 的 vLLM v0.21.0 base 里没有 is_cumem_allocator_available（全仓 grep 零命中），
     所以在当前 base 上此 patch 是 no-op，真正放行 sleep mode 的是
     NPUPlatform.is_sleep_mode_available()=True 被 vllm/config/model.py 的校验命中。

唯一减法：host 无 vllm，把硬 import 降级为可选 import（缺失时 model_config_module=None，
patch 自然 no-op），其余逻辑原样保留。
"""
# SUBTRACTED: 文件头 Apache-2.0 许可证注释块（原 patch_camem_allocator.py:L1-L16）
# SUBTRACTED: import vllm.config.model as model_config_module（原 L17）——host 无 vllm，
#   降级为可选 import；缺失时 model_config_module=None，下方 hasattr 守护天然 no-op。
try:
    import vllm.config.model as model_config_module  # type: ignore
except ImportError:
    model_config_module = None


def _patched_is_cumem_allocator_available() -> bool:
    # SOURCE: vllm_ascend/patch/platform/patch_camem_allocator.py:L20-L24
    # NPUPlatform declares sleep mode support and vllm-ascend uses CaMemAllocator
    # in the worker path. Avoid importing the extension here because ModelConfig
    # validation runs before custom op initialization.
    return True


# SOURCE: vllm_ascend/patch/platform/patch_camem_allocator.py:L27-L28
# 注：原文是 `if hasattr(model_config_module, ...)`；这里多一个 `model_config_module is not None`
# 前置守护，仅因上方硬 import 被降级为可选 import（host 无 vllm），不改变语义。
if model_config_module is not None and hasattr(model_config_module, "is_cumem_allocator_available"):
    model_config_module.is_cumem_allocator_available = _patched_is_cumem_allocator_available
