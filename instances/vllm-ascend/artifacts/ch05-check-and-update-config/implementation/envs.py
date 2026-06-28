# 章 ch05 精简版 —— vllm_ascend/envs.py（subtract-only）
#
# 三级取值（additional_config→env→default）里 **env + default 两级的实现处**：
# 每个环境变量封成一个 lambda（os.getenv + 类型转换 + default），env 与 default 在 lambda 内塌缩；
# 模块级 __getattr__ 做懒求值——`ascend_envs.VLLM_ASCEND_*` 每次访问都重新读环境。
import os
from typing import Any, Callable

# begin-env-vars-definition

# SOURCE: vllm_ascend/envs.py:L30-L113 (env_variables 表)
# SUBTRACTED: 原表约三十项（MAX_JOBS / CMAKE_BUILD_TYPE / SOC_VERSION / …等构建与特性开关）。
#             范式完全同构：lambda: <类型转换>(os.getenv(KEY, <default>))。这里只保留本章
#             三级取值真正消费的两项（被 AscendConfig._get_config_value 取用），外加一项 None
#             默认、一项 int 默认的代表，足以演示「env+default 在 lambda 内塌缩」。
env_variables: dict[str, Callable[[], Any]] = {
    # The version of the Ascend chip; default None 代表（演示 default=None 的塌缩形态）。
    "SOC_VERSION": lambda: os.getenv("SOC_VERSION", None),
    # int 默认代表：演示 default 非 0 标量的塌缩形态。
    "VLLM_ASCEND_ENABLE_NZ": lambda: int(os.getenv("VLLM_ASCEND_ENABLE_NZ", 1)),
    # Whether to enable FlashComm optimization when tensor parallel is enabled.
    # DEPRECATED: use additional_config.enable_flashcomm1 instead.
    "VLLM_ASCEND_ENABLE_FLASHCOMM1": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0"))),
    # DEPRECATED: VLLM_ASCEND_BALANCE_SCHEDULING env var will be removed in a future release.
    # Use --additional-config '{"enable_balance_scheduling": true}' instead.
    "VLLM_ASCEND_BALANCE_SCHEDULING": lambda: bool(int(os.getenv("VLLM_ASCEND_BALANCE_SCHEDULING", "0"))),
}

# end-env-vars-definition


# SOURCE: vllm_ascend/envs.py:L118-L122
def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# SOURCE: vllm_ascend/envs.py:L125-L126
def __dir__():
    return list(env_variables.keys())
