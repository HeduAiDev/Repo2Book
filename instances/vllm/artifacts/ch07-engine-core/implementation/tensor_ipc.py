# SPDX-License-Identifier: Apache-2.0
# 只做减法的忠实子集 —— 对应 vllm/v1/engine/tensor_ipc.py。
# 源码 pin: f3fef123。多模态张量经队列共享内存零拷贝旁路 + drain-and-buffer 乱序重组。
#
# 与真实 vLLM 的唯一差别：真实代码用 torch.multiprocessing.Queue（跨进程共享内存句柄）；
# 本精简版在单进程内用 queue.Queue 跑通同一套 sender/receiver 协议与 drain-and-buffer 逻辑，
# 控制流、字段、乱序处理 1:1。原 vllm/v1/engine/tensor_ipc.py:L17,L52。

import dataclasses
import uuid
from collections import defaultdict
from dataclasses import field
from typing import Any

import torch

from serial_utils import OOBTensorConsumer

# SUBTRACTED: TensorIpcQueue = torch.multiprocessing.Queue 别名 —— 单进程测试用
#   标准库 queue.Queue 即可承载同一 put/get 语义。原 vllm/v1/engine/tensor_ipc.py:L27。


@dataclasses.dataclass
# SOURCE: vllm/v1/engine/tensor_ipc.py:L30  TensorIpcData
class TensorIpcData:
    """
    Data sent via torch.multiprocessing.Queue for zero-copy IPC.

    Contains the tensor_id and the actual tensor. The tensor is
    shared in memory (GPU or CPU) for efficient inter-process communication.
    """

    sender_id: str
    message_id: int
    tensor_id: int
    tensor: torch.Tensor


# SOURCE: vllm/v1/engine/tensor_ipc.py:L45  TensorIpcSender
class TensorIpcSender(OOBTensorConsumer):
    """Send-side logic for tensor IPC via torch.multiprocessing.Queue.

    Uses a single queue targeting rank 0 (the only rank that consumes
    multimodal tensors during TP>1 / PP>1. Note: DP>1 not supported).
    """

    def __init__(self, queue: Any):
        # SOURCE: vllm/v1/engine/tensor_ipc.py:L52  TensorIpcSender.__init__
        self.queue = queue
        self._tensor_id_counter = 0
        self._message_counter = 0
        self._sender_id = uuid.uuid4().hex[:8]

    # SUBTRACTED: set_target_engine（DP>1 不支持的守卫，本章单 engine）。
    #   原 vllm/v1/engine/tensor_ipc.py:L58。

    def new_message(self) -> None:
        # SOURCE: vllm/v1/engine/tensor_ipc.py:L65  TensorIpcSender.new_message
        self._message_counter += 1
        self._tensor_id_counter = 0

    # SOURCE: vllm/v1/engine/tensor_ipc.py:L69  TensorIpcSender.__call__
    def __call__(self, tensor: torch.Tensor) -> dict[str, Any] | None:
        """Send tensor via queue, return its handle. Returns None if failed."""
        try:
            # Move tensor to shared memory for IPC
            # This is required for proper inter-process communication
            if not tensor.is_shared():
                tensor = tensor.share_memory_()

            metadata = {
                "sender_id": self._sender_id,
                "message_id": self._message_counter,
                "tensor_id": self._tensor_id_counter,
            }

            self._tensor_id_counter += 1

            ipc_data = TensorIpcData(**metadata, tensor=tensor)  # type: ignore[arg-type]

            # Use a timeout to avoid blocking indefinitely
            self.queue.put(ipc_data, timeout=10.0)

            return metadata
        except Exception:
            # Falling back to standard serialization.
            return None


@dataclasses.dataclass
# SOURCE: vllm/v1/engine/tensor_ipc.py:L108  _Sender
class _Sender:
    current_message_id: int = -1
    tensors: dict[int, dict[int, torch.Tensor]] = field(default_factory=dict)


# SOURCE: vllm/v1/engine/tensor_ipc.py:L114  TensorIpcReceiver
class TensorIpcReceiver:
    """Receive-side logic for tensor IPC via torch.multiprocessing.Queue."""

    def __init__(self, queue: Any):
        # SOURCE: vllm/v1/engine/tensor_ipc.py:L120  TensorIpcReceiver.__init__
        self.queue = queue
        self._tensor_buffers = defaultdict[str, _Sender](_Sender)

    # SOURCE: vllm/v1/engine/tensor_ipc.py:L124  TensorIpcReceiver.__call__
    def __call__(
        self, dtype: str, shape: tuple[int, ...], meta: dict[str, Any]
    ) -> torch.Tensor:
        """Retrieve a tensor from torch.multiprocessing.Queue.

        Uses a drain-and-buffer pattern: drains all available tensors from
        the queue, buffering them, until the requested tensor is found.
        Works for CUDA and CPU.
        """

        # Create lookup key from handle
        sender_id: str = meta["sender_id"]
        message_id: int = meta["message_id"]
        tensor_id: int = meta["tensor_id"]

        # Drain all available tensors. We save them regardless if this is
        # the one we're waiting for as they may arrive out of order from
        # multiple producers.
        while True:
            sender = self._tensor_buffers.get(sender_id)
            if sender is not None:
                tensors = sender.tensors
                tensor = tensors.get(message_id, {}).pop(tensor_id, None)
                if tensor is not None:
                    if sender.current_message_id != message_id:
                        while tensors and (mid := next(iter(tensors))) < message_id:
                            sender.tensors.pop(mid)
                        sender.current_message_id = message_id
                    return tensor

            ipc_data: TensorIpcData = self.queue.get(timeout=10.0)

            # Store tensor
            sender = self._tensor_buffers[ipc_data.sender_id]
            if sender.current_message_id > ipc_data.message_id:
                # Ignoring stale tensor from sender.
                continue

            sender.tensors.setdefault(ipc_data.message_id, {})[ipc_data.tensor_id] = (
                ipc_data.tensor
            )
