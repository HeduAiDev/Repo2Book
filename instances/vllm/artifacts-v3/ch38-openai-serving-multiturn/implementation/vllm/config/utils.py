# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/config/utils.py —— HOST SEAM（最小承载）：本章消费面只有
# entrypoints/openai/engine/protocol.py 的 structured_outputs_from_response_format
# 用 replace(structured_outputs, **overrides) 对 dataclass 做字段替换；
# 真实实现是对 msgspec.Struct 的专用替换工具，dataclass 场景与
# dataclasses.replace 语义一致。
from dataclasses import replace  # noqa: F401
