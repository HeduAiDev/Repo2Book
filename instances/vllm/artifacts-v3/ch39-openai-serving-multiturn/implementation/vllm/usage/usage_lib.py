# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/usage/usage_lib.py —— HOST SEAM（最小承载）：本章消费面只有
# UsageContext.OPENAI_API_SERVER 枚举值（build_async_engine_client 的
# usage_context 默认参）。真实为 usage 统计上报库。
from enum import Enum


# SOURCE: vllm/usage/usage_lib.py —— UsageContext 枚举位（真实含
# ENGINE_CONTEXT/CLI_CONTEXT 等全族与上报链；精简面只承载被引用两值）
class UsageContext(Enum):
    # SOURCE: vllm/usage/usage_lib.py —— ENGINE_CONTEXT 位
    ENGINE_CONTEXT = "ENGINE_CONTEXT"
    # SOURCE: vllm/usage/usage_lib.py —— OPENAI_API_SERVER 位
    OPENAI_API_SERVER = "OPENAI_API_SERVER"


# SUBTRACTED: vllm/usage/usage_lib.py 其余（UsageContextReport/
# is_usage_context_enabled/上报链）——遥测域。
