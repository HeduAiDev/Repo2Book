# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/v1/serial_utils.py —— HOST SEAM（最小承载）：真实 628 行
# 提供 msgspec 编解码器工厂与 UtilityResult 协议；本章消费面只有
# SamplingParams 的基类混入 PydanticMsgspecMixin（空混入，msgspec.Struct
# 本身提供序列化），host 上不需要跨进程编解码器。
# SUBTRACTED: MsgspecEncoder/Decoder 工厂、UtilityResult 与 zip 工具族
# （ch5 ZMQ 域消费）——本章单进程精简环境不跨进程传消息。


# SOURCE: vllm/v1/serial_utils.py:L513 —— HOST SEAM：空混入（真实为
# pydantic 兼容的 to_json/from_json 混入；msgspec.Struct 子类在单进程
# 内不需要它）
class PydanticMsgspecMixin:
    pass
