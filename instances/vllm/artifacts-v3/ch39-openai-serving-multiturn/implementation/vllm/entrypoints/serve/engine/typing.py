# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/entrypoints/serve/engine/typing.py —— HOST SEAM（最小承载）：
# AnyRequest 是 BaseServing 方法签名的请求联合类型面（真实为 Chat/
# Completion/Responses/Pooling 请求模型 Union）。
from typing import Any, TypeAlias

# SOURCE: vllm/entrypoints/serve/engine/typing.py —— AnyRequest 别名位
# （HOST SEAM：精简版退化为 Any 总形）
AnyRequest: TypeAlias = Any

# SUBTRACTED: AnyPoolingRequest——pooling 面类型（m18 平行面）。
