# SPDX-License-Identifier: Apache-2.0
"""TDD: TensorIpcSender/Receiver 共享内存旁路 + drain-and-buffer 乱序重组。

复现真实 vLLM 行为：
- sender 把张量 put 进队列，返回 {sender_id,message_id,tensor_id} 元数据。
- new_message 推进 message_id 并把 tensor_id 计数归零。
- receiver 用 drain-and-buffer：从队列排空、缓冲，直到找到目标张量；
  能处理多张量乱序到达（同一 message 内 tensor_id 乱序）。
"""
import os
import queue
import sys

import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "implementation")
)

from tensor_ipc import TensorIpcReceiver, TensorIpcSender  # noqa: E402


def test_sender_returns_metadata_and_enqueues():
    q = queue.Queue()
    sender = TensorIpcSender(q)
    sender.new_message()
    t = torch.arange(8, dtype=torch.float32)
    meta = sender(t)
    assert set(meta) == {"sender_id", "message_id", "tensor_id"}
    # new_message() 在首次发送前把 _message_counter 从 0 推进到 1。
    assert meta["message_id"] == 1
    assert meta["tensor_id"] == 0
    data = q.get_nowait()
    assert torch.equal(data.tensor, t)


def test_new_message_resets_tensor_counter():
    q = queue.Queue()
    sender = TensorIpcSender(q)
    sender.new_message()
    sender(torch.zeros(2))
    m2 = sender(torch.zeros(2))
    assert m2["tensor_id"] == 1  # 同一 message 内递增
    sender.new_message()
    m3 = sender(torch.zeros(2))
    assert m3["message_id"] == 2  # 第二次 new_message -> message_id 2
    assert m3["tensor_id"] == 0  # 新 message 归零


def test_receiver_roundtrip():
    q = queue.Queue()
    sender = TensorIpcSender(q)
    receiver = TensorIpcReceiver(q)
    sender.new_message()
    t = torch.arange(5, dtype=torch.float32)
    meta = sender(t)
    out = receiver("float32", tuple(t.shape), meta)
    assert torch.equal(out, t)


def test_receiver_handles_out_of_order_arrival():
    q = queue.Queue()
    sender = TensorIpcSender(q)
    receiver = TensorIpcReceiver(q)
    sender.new_message()
    a = torch.full((3,), 1.0)
    b = torch.full((3,), 2.0)
    meta_a = sender(a)  # tensor_id 0
    meta_b = sender(b)  # tensor_id 1
    # 先请求后到达的 tensor_id=1：receiver 必须 drain 整个队列并缓冲 0，再返回 1。
    out_b = receiver("float32", (3,), meta_b)
    assert torch.equal(out_b, b)
    # 0 已被缓冲，后续请求直接命中缓冲。
    out_a = receiver("float32", (3,), meta_a)
    assert torch.equal(out_a, a)
