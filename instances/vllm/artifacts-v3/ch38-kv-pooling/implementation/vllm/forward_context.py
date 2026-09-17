# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""前向上下文（本章只在签名面消费：start_load_kv 的第一参）。

# SOURCE: vllm/forward_context.py:ForwardContext 类头
# SUBTRACTED: 前向上下文的批装配/层名解析/静态上下文——本章不跑真前向。
"""


# SOURCE: vllm/forward_context.py:ForwardContext 类头
class ForwardContext:
    # SOURCE: vllm/forward_context.py:ForwardContext 类头
    """Holds context for the current forward pass（类型位）。"""


__all__ = ["ForwardContext"]
