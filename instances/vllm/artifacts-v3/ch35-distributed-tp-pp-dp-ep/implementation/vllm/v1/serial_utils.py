# SOURCE: vllm/v1/serial_utils.py
# ch34 切面（ch05 域的序列化层，coordinator/core_client 消费）：MsgpackEncoder /
# MsgpackDecoder 的单缓冲形态。SUBTRACTED：torch/numpy 张量的零拷贝 OOB 机制
# （aux_buffers/ext_hook/enc_hook 全族，L1-L135、L140-L310）——多模态/张量过线域；
# 本章过线的载荷（EngineCoreOutputs/EngineCoreRequest）无张量字段。

from __future__ import annotations

from typing import Any

import msgspec.msgpack


# SOURCE: vllm/v1/serial_utils.py:L136+ MsgpackEncoder —— 单缓冲子集
class MsgpackEncoder:
    """Encoder with custom torch tensor and numpy array serialization.

    Note that unlike vanilla `msgspec` Encoders, this interface is generally
    not thread-safe when encoding tensors / numpy arrays.
    """

    # SOURCE: vllm/v1/serial_utils.py:L149-L164 __init__（OOB 消费者参数保留位）
    def __init__(
        self,
        size_threshold: int | None = None,
        oob_tensor_consumer=None,
    ):
        # SUBTRACTED: size_threshold / aux_buffers / OOB 张量通道（L154-L162）
        #   ——张量零拷贝域。
        self.oob_tensor_consumer = oob_tensor_consumer

    # SOURCE: vllm/v1/serial_utils.py:L166+ encode —— 返回 bytestr 序列
    def encode(self, obj: Any):
        return (msgspec.msgpack.encode(obj),)


# SOURCE: vllm/v1/serial_utils.py:L313+ MsgpackDecoder —— 单缓冲子集
class MsgpackDecoder:
    """Decoder with custom torch tensor and numpy array serialization.

    Note that unlike vanilla `msgspec` Decoders, this interface is generally
    not thread-safe when encoding tensors / numpy arrays.
    """

    # SOURCE: vllm/v1/serial_utils.py:L323-L338 __init__
    def __init__(
        self,
        t: Any | None = None,
        share_mem: bool = True,
        oob_tensor_provider=None,
    ):
        self.share_mem = share_mem
        args = () if t is None else (t,)
        # SUBTRACTED: ext_hook/dec_hook 的张量与 numpy 钩子（L332-L334）——
        #   零拷贝域；普通 msgspec 解码路径保留。
        self.decoder = msgspec.msgpack.Decoder(*args)
        self.oob_tensor_provider = oob_tensor_provider

    # SOURCE: vllm/v1/serial_utils.py:L340-L343 decode —— 单缓冲分支逐字
    def decode(self, bufs):
        if isinstance(bufs, (bytes, bytearray, memoryview)):
            return self.decoder.decode(bufs)
        # SUBTRACTED: 多缓冲（aux_buffers）张量重组分支——零拷贝域。
        return self.decoder.decode(bytes(bufs[0]))
