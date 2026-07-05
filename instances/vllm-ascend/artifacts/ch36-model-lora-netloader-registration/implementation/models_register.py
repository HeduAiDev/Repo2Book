# ch30 变体(1) 模型注册 —— subtract-only 精简版
#
# 真实源码位于 vllm_ascend/models/__init__.py（整文件 7 行）。精简版改名为
# models_register.py 仅因 lint_fidelity 不扫描名为 __init__.py 的文件（会漏掉
# must_keep 的 register_model）；控制流与源码逐字一致，SOURCE 标注指向真实路径。
#
# 本文件零删除（整文件仅 7 行，全是把架构名映射到「<module>:<class>」懒加载字符串
# 的注册调用，无可删样板）。

# SOURCE: vllm_ascend/models/__init__.py:L1
from vllm import ModelRegistry


# SOURCE: vllm_ascend/models/__init__.py:L4-L7
def register_model():
    ModelRegistry.register_model("DeepseekV4ForCausalLM", "vllm_ascend.models.deepseek_v4:AscendDeepseekV4ForCausalLM")

    ModelRegistry.register_model("DeepSeekV4MTPModel", "vllm_ascend.models.deepseek_v4_mtp:DeepSeekV4MTP")
