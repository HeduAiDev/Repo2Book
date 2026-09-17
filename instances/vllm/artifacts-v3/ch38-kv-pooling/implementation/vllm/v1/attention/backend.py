# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""注意力后端类型位（本章只在注解/签名面消费）。

# SOURCE: vllm/v1/attention/backend.py:L1-L120（AttentionBackend/AttentionMetadata）
# SUBTRACTED: 后端实现族（FlashAttention/FlashInfer/TRT...）与元数据装配——
#   cross-layer 注册路径已按减法计划删除，本模块只承担类型面。
"""

from typing import Any


# SOURCE: vllm/v1/attention/backend.py:AttentionBackend 类头
class AttentionBackend:
    # SOURCE: vllm/v1/attention/backend.py:AttentionBackend 类头
    """Base class for attention backends（类型位）。"""


# SOURCE: vllm/v1/attention/backend.py:AttentionMetadata 类头
class AttentionMetadata(Any):
    # SOURCE: vllm/v1/attention/backend.py:AttentionMetadata 类头
    """Metadata for Attention layers（类型位）。"""


__all__ = ["AttentionBackend", "AttentionMetadata"]
