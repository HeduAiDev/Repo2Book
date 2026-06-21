# SPDX-License-Identifier: Apache-2.0
"""TDD: msgpack 多帧零拷贝编解码（aux_buffers / 内联阈值 / OOB 旁路）。

复现真实 vLLM 的可观察行为：
- 小张量/数组 < VLLM_MSGPACK_ZERO_COPY_THRESHOLD(256B) 内联进主帧（1 帧）。
- 大张量进 aux_buffers，多帧返回，索引零拷贝还原。
- 提供 OOBTensorConsumer 时大张量走旁路 dict 占位，主帧仍只 1 帧；
  解码端必须有 oob_tensor_provider，否则断言失败。
- encode_into 复用传入 bytearray 作为 bufs[0]。
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "implementation")
)

from serial_utils import (  # noqa: E402
    VLLM_MSGPACK_ZERO_COPY_THRESHOLD,
    MsgpackDecoder,
    MsgpackEncoder,
    OOBTensorConsumer,
    UtilityResult,
)


def test_small_tensor_inlined_single_frame():
    enc = MsgpackEncoder()
    dec = MsgpackDecoder(torch.Tensor)
    small = torch.arange(4, dtype=torch.int32)  # 16B < 256B
    assert small.nbytes < VLLM_MSGPACK_ZERO_COPY_THRESHOLD
    bufs = enc.encode(small)
    # Inline -> only the main struct frame, no aux backing-buffer frame.
    assert len(bufs) == 1
    out = dec.decode(bufs)
    assert torch.equal(out, small)


def test_large_tensor_uses_aux_buffers_multiframe():
    enc = MsgpackEncoder()
    dec = MsgpackDecoder(torch.Tensor)
    big = torch.arange(200, dtype=torch.int64)  # 1600B > 256B
    assert big.nbytes >= VLLM_MSGPACK_ZERO_COPY_THRESHOLD
    bufs = enc.encode(big)
    # Main frame + 1 aux backing-buffer frame (zero-copy).
    assert len(bufs) == 2
    out = dec.decode(bufs)
    assert torch.equal(out, big)


def test_large_ndarray_uses_aux_buffers():
    enc = MsgpackEncoder()
    dec = MsgpackDecoder(np.ndarray)
    arr = np.arange(100, dtype=np.float64)  # 800B
    bufs = enc.encode(arr)
    assert len(bufs) == 2
    out = dec.decode(bufs)
    assert np.array_equal(out, arr)


def test_encode_into_reuses_buffer_as_first_frame():
    enc = MsgpackEncoder()
    buf = bytearray()
    bufs = enc.encode_into(torch.arange(4, dtype=torch.int32), buf)
    # encode_into 的 bufs[0] 就是传入的那块 bytearray（被原地写入）。
    assert bufs[0] is buf
    assert len(buf) > 0


class _RecordingOOB(OOBTensorConsumer):
    """记录被旁路的张量，返回占位 dict（模拟 TensorIpcSender 的协议形状）。"""

    def __init__(self):
        self.sent = []
        self._msg = -1

    def new_message(self):
        self._msg += 1

    def __call__(self, tensor):
        meta = {"sender_id": "x", "message_id": self._msg, "tensor_id": len(self.sent)}
        self.sent.append((meta, tensor))
        return meta


def test_oob_consumer_bypasses_large_tensor():
    oob = _RecordingOOB()
    enc = MsgpackEncoder(oob_tensor_consumer=oob)

    def provider(dtype, shape, meta):
        # 按 meta 找回被旁路的张量。
        for m, t in oob.sent:
            if m == meta:
                return t
        raise KeyError(meta)

    dec = MsgpackDecoder(torch.Tensor, oob_tensor_provider=provider)
    big = torch.arange(200, dtype=torch.int64)
    bufs = enc.encode(big)
    # 张量本体走旁路，不进 ZMQ aux 帧 -> 主帧仍只 1 帧。
    assert len(bufs) == 1
    assert len(oob.sent) == 1
    out = dec.decode(bufs)
    assert torch.equal(out, big)


def test_oob_dict_without_provider_asserts():
    oob = _RecordingOOB()
    enc = MsgpackEncoder(oob_tensor_consumer=oob)
    dec = MsgpackDecoder(torch.Tensor)  # no provider
    bufs = enc.encode(torch.arange(200, dtype=torch.int64))
    with pytest.raises(AssertionError):
        dec.decode(bufs)


def test_utility_result_roundtrip():
    enc = MsgpackEncoder()
    dec = MsgpackDecoder(UtilityResult)
    bufs = enc.encode(UtilityResult(["generate", "embed"]))
    out = dec.decode(bufs)
    assert isinstance(out, UtilityResult)
    assert out.result == ["generate", "embed"]
