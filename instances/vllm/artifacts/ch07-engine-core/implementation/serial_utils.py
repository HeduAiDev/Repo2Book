# SPDX-License-Identifier: Apache-2.0
# 只做减法的忠实子集 —— 对应 vllm/v1/serial_utils.py。
# 源码 pin: f3fef123。保留多帧零拷贝编解码主线：
#   bufs[0] 主结构帧 + aux_buffers 追加大张量/数组 backing buffer + OOB dict 旁路。
# SUBTRACTED: MultiModalKwargsItems/slice/UtilityResult 的 pickle/cloudpickle 回退分支、
#   _encode_type_info_recursive 递归类型编码、_encode_mm_* / _decode_mm_*、ndarray 的
#   object/void pickle 回退 —— 本章只讲 tensor/ndarray 多帧零拷贝与 OOB 两条主线，
#   多模态嵌套与不安全 pickle 是 must_keep 之外的边缘分支（dossier.subtraction_plan.delete 第7项批准）。

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

import msgspec
import numpy as np
import torch
import zmq
from msgspec import msgpack

CUSTOM_TYPE_RAW_VIEW = 3


# SOURCE: vllm/v1/serial_utils.py:L129  UtilityResult
class UtilityResult:
    """Wrapper for special handling when serializing/deserializing."""

    def __init__(self, r: Any = None):
        # SOURCE: vllm/v1/serial_utils.py:L132  UtilityResult.__init__
        self.result = r


# 内联阈值：小于它的张量/数组直接进主帧（VLLM_MSGPACK_ZERO_COPY_THRESHOLD 默认 256B）。
# SUBTRACTED: 原从 envs.VLLM_MSGPACK_ZERO_COPY_THRESHOLD 读取，这里固化默认值。
#   原 vllm/v1/serial_utils.py:L155。
VLLM_MSGPACK_ZERO_COPY_THRESHOLD = 256

bytestr: TypeAlias = bytes | bytearray | memoryview | zmq.Frame


# SOURCE: vllm/v1/utils.py:L443  tensor_data
def tensor_data(tensor: torch.Tensor) -> memoryview:
    """Get the raw data of a tensor as a uint8 memoryview, useful for
    serializing and hashing."""
    return tensor.flatten().cpu().contiguous().view(torch.uint8).numpy().data


# SOURCE: vllm/v1/serial_utils.py:L57  OOBTensorConsumer
class OOBTensorConsumer(ABC):
    @abstractmethod
    def __call__(self, tensor: torch.Tensor) -> dict | None:
        # SOURCE: vllm/v1/serial_utils.py:L58  OOBTensorConsumer.__call__
        """
        Called with tensors for the current message.
        Returns None to reject the tensor (falls back to regular serialization),
        otherwise a dict with arbitrary placeholder data to be included
        in the serialized message.
        """
        return None

    @abstractmethod
    def new_message(self) -> None:
        # SOURCE: vllm/v1/serial_utils.py:L68  OOBTensorConsumer.new_message
        """Called at the start of each new encoded message."""
        pass


# SOURCE: vllm/v1/serial_utils.py:L75  OOBTensorProvider
# dtype, shape, metadata -> tensor
OOBTensorProvider = Callable[[str, tuple[int, ...], dict], torch.Tensor]


# SOURCE: vllm/v1/serial_utils.py:L136  MsgpackEncoder
class MsgpackEncoder:
    """Encoder with custom torch tensor and numpy array serialization.

    By default, arrays below 256B are serialized inline. Larger will get sent
    via dedicated messages. Note that this is a per-tensor limit.

    When a ``oob_tensor_consumer`` is provided, tensors (CUDA and CPU) will be
    offered to it for out-of-band handling.
    """

    # SOURCE: vllm/v1/serial_utils.py:L149  MsgpackEncoder.__init__
    def __init__(
        self,
        size_threshold: int | None = None,
        oob_tensor_consumer: OOBTensorConsumer | None = None,
    ):
        if size_threshold is None:
            size_threshold = VLLM_MSGPACK_ZERO_COPY_THRESHOLD
        self.encoder = msgpack.Encoder(enc_hook=self.enc_hook)
        # This is used as a local stash of buffers that we can then access from
        # our custom `msgspec` hook, `enc_hook`. We don't have a way to
        # pass custom data to the hook otherwise.
        self.aux_buffers: list[bytestr] | None = None
        self.size_threshold = size_threshold
        self.oob_tensor_consumer = oob_tensor_consumer
        # SUBTRACTED: VLLM_ALLOW_INSECURE_SERIALIZATION 警告。原 :L163。

    # SOURCE: vllm/v1/serial_utils.py:L166  MsgpackEncoder.encode
    def encode(self, obj: Any) -> Sequence[bytestr]:
        try:
            if self.oob_tensor_consumer is not None:
                self.oob_tensor_consumer.new_message()
            self.aux_buffers = bufs = [b""]
            bufs[0] = self.encoder.encode(obj)
            # This `bufs` list allows us to collect direct pointers to backing
            # buffers of tensors and np arrays, and return them along with the
            # top-level encoded buffer instead of copying their data into the
            # new buffer.
            return bufs
        finally:
            self.aux_buffers = None

    # SOURCE: vllm/v1/serial_utils.py:L180  MsgpackEncoder.encode_into
    def encode_into(self, obj: Any, buf: bytearray) -> Sequence[bytestr]:
        try:
            if self.oob_tensor_consumer is not None:
                self.oob_tensor_consumer.new_message()
            self.aux_buffers = [buf]
            bufs = self.aux_buffers
            self.encoder.encode_into(obj, buf)
            return bufs
        finally:
            self.aux_buffers = None

    # SOURCE: vllm/v1/serial_utils.py:L191  MsgpackEncoder.enc_hook
    def enc_hook(self, obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return self._encode_tensor(obj)

        # Fall back to pickle for object or void kind ndarrays.
        if isinstance(obj, np.ndarray) and obj.dtype.kind not in ("O", "V"):
            return self._encode_ndarray(obj)

        if isinstance(obj, UtilityResult):
            result = obj.result
            # SUBTRACTED: VLLM_ALLOW_INSECURE_SERIALIZATION 下的 _encode_type_info_recursive
            #   递归类型编码（delete 第7项）。安全默认路径只发 (None, result)。原 :L212-L219。
            return None, result

        # SUBTRACTED: slice / MultiModalKwargsItem(s) 分支 +
        #   VLLM_ALLOW_INSECURE_SERIALIZATION pickle/cloudpickle 回退。
        #   原 vllm/v1/serial_utils.py:L199-L235（delete 第7项批准）。
        raise TypeError(f"Object of type {type(obj)} is not serializable")

    # SOURCE: vllm/v1/serial_utils.py:L237  MsgpackEncoder._encode_ndarray
    def _encode_ndarray(
        self, obj: np.ndarray
    ) -> tuple[str, tuple[int, ...], int | memoryview]:
        assert self.aux_buffers is not None
        # If the array is non-contiguous, we need to copy it first
        arr_data = obj.data if obj.flags.c_contiguous else obj.tobytes()
        if not obj.shape or obj.nbytes < self.size_threshold:
            # Encode small arrays and scalars inline. Using this extension type
            # ensures we can avoid copying when decoding.
            data = msgpack.Ext(CUSTOM_TYPE_RAW_VIEW, arr_data)
        else:
            # Otherwise encode index of backing buffer to avoid copy.
            data = len(self.aux_buffers)
            self.aux_buffers.append(arr_data)

        # We serialize the ndarray as a tuple of native types.
        # The data is either inlined if small, or an index into a list of
        # backing buffers that we've stashed in `aux_buffers`.
        return obj.dtype.str, obj.shape, data

    # SOURCE: vllm/v1/serial_utils.py:L257  MsgpackEncoder._encode_tensor
    def _encode_tensor(
        self, obj: torch.Tensor
    ) -> tuple[str, tuple[int, ...], int | dict | memoryview]:
        oob_consumer = self.oob_tensor_consumer
        # view the tensor as a contiguous 1D array of bytes
        if obj.nbytes < self.size_threshold and obj.is_cpu:
            # Smaller tensors are encoded inline, just like ndarrays.
            data = msgpack.Ext(CUSTOM_TYPE_RAW_VIEW, tensor_data(obj))
        elif oob_consumer is not None and (data := oob_consumer(obj)) is not None:
            assert isinstance(data, dict)
        else:
            # Otherwise encode index of backing buffer to avoid copy.
            assert self.aux_buffers is not None
            data = len(self.aux_buffers)
            self.aux_buffers.append(tensor_data(obj))
        dtype = str(obj.dtype).removeprefix("torch.")
        return dtype, obj.shape, data


# SOURCE: vllm/v1/serial_utils.py:L313  MsgpackDecoder
class MsgpackDecoder:
    """Decoder with custom torch tensor and numpy array serialization.

    ``oob_tensor_provider`` must be used when an OOBTensorConsumer is used on the
    encoder side.
    """

    # SOURCE: vllm/v1/serial_utils.py:L323  MsgpackDecoder.__init__
    def __init__(
        self,
        t: Any | None = None,
        share_mem: bool = True,
        oob_tensor_provider: OOBTensorProvider | None = None,
    ):
        self.share_mem = share_mem
        # SUBTRACTED: is_pin_memory_available() pin 优化（CPU-only 测试无需）。原 :L330。
        self.pin_tensors = False
        args = () if t is None else (t,)
        self.decoder = msgpack.Decoder(
            *args, ext_hook=self.ext_hook, dec_hook=self.dec_hook
        )
        self.aux_buffers: Sequence[bytestr] = ()
        self.oob_tensor_provider = oob_tensor_provider
        # SUBTRACTED: VLLM_ALLOW_INSECURE_SERIALIZATION 警告。原 :L337。

    # SOURCE: vllm/v1/serial_utils.py:L340  MsgpackDecoder.decode
    def decode(self, bufs: bytestr | Sequence[bytestr]) -> Any:
        if isinstance(bufs, bytestr):  # type: ignore
            return self.decoder.decode(bufs)

        self.aux_buffers = bufs
        try:
            return self.decoder.decode(bufs[0])
        finally:
            self.aux_buffers = ()

    # SOURCE: vllm/v1/serial_utils.py:L350  MsgpackDecoder.dec_hook
    def dec_hook(self, t: type, obj: Any) -> Any:
        # Given native types in `obj`, convert to type `t`.
        from inspect import isclass

        if isclass(t):
            if issubclass(t, np.ndarray):
                return self._decode_ndarray(obj)
            if issubclass(t, torch.Tensor):
                return self._decode_tensor(obj)
            if t is UtilityResult:
                return self._decode_utility_result(obj)
            # SUBTRACTED: slice / MultiModalKwargsItem(s) 解码分支。
            #   原 vllm/v1/serial_utils.py:L357-L362（delete 第7项批准）。
        return obj

    # SOURCE: vllm/v1/serial_utils.py:L367  MsgpackDecoder._decode_utility_result
    def _decode_utility_result(self, obj: Any) -> UtilityResult:
        result_type, result = obj
        # SUBTRACTED: result_type 非空时的 _decode_type_info_recursive 递归还原
        #   （仅 VLLM_ALLOW_INSECURE_SERIALIZATION，delete 第7项）。原 :L369-L378。
        return UtilityResult(result)

    # SOURCE: vllm/v1/serial_utils.py:L389  MsgpackDecoder._decode_ndarray
    def _decode_ndarray(self, arr: Any) -> np.ndarray:
        dtype, shape, data = arr
        # zero-copy decode. We assume the ndarray will not be kept around,
        # as it now locks the whole received message buffer in memory.
        buffer = self.aux_buffers[data] if isinstance(data, int) else data
        arr = np.frombuffer(buffer, dtype=dtype)
        if not self.share_mem:
            arr = arr.copy()
        return arr.reshape(shape)

    # SOURCE: vllm/v1/serial_utils.py:L399  MsgpackDecoder._decode_tensor
    def _decode_tensor(self, arr: Any) -> torch.Tensor:
        dtype, shape, data = arr
        if isinstance(data, dict):
            assert self.oob_tensor_provider, (
                "Received OOB tensor but tensor provider is not set"
            )
            return self.oob_tensor_provider(dtype, shape, data)

        is_aux = isinstance(data, int)
        buffer = self.aux_buffers[data] if is_aux else data
        buffer = buffer if isinstance(buffer, memoryview) else memoryview(buffer)
        torch_dtype = getattr(torch, dtype)
        assert isinstance(torch_dtype, torch.dtype)
        if not buffer.nbytes:  # torch.frombuffer doesn't like empty buffers
            assert 0 in shape
            return torch.empty(shape, dtype=torch_dtype)
        # Create uint8 array
        arr = torch.frombuffer(buffer, dtype=torch.uint8)
        # Clone ensures tensor is backed by pytorch-owned memory for safe
        # future async CPU->GPU transfer.
        # Pin larger tensors for more efficient CPU->GPU transfer.
        if not is_aux:
            arr = arr.clone()
        elif not self.share_mem:
            arr = arr.pin_memory() if self.pin_tensors else arr.clone()
        # Convert back to proper shape & type
        return arr.view(torch_dtype).view(shape)

    # SOURCE: vllm/v1/serial_utils.py:L473  MsgpackDecoder.ext_hook
    def ext_hook(self, code: int, data: memoryview) -> Any:
        if code == CUSTOM_TYPE_RAW_VIEW:
            return data
        # SUBTRACTED: CUSTOM_TYPE_PICKLE / CUSTOM_TYPE_CLOUDPICKLE 不安全反序列化分支。
        #   原 vllm/v1/serial_utils.py ext_hook（delete 第7项批准）。
        raise NotImplementedError(f"Extension type code {code} is not supported")
