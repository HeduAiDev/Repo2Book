# 环境桥接（NOT a vLLM abstraction）—— 进程内的 MessageQueue 替身。
#
# 真实 vLLM 用 vllm/distributed/device_communicators/shm_broadcast.py:MessageQueue：
# 跨进程共享内存的『单写多读广播队列』——executor 单次 enqueue，所有 worker 都能
# dequeue 到同一条消息（rpc_broadcast_mq）；以及 1 写 1 读的点对点队列（response_mq）。
#
# host 上为了无依赖可运行，这里用 threading 复刻它在本章用到的接口与语义：
#   - enqueue / dequeue(timeout|indefinite) / export_handle / create_from_handle /
#     wait_until_ready。
# 广播语义：每个 reader 通过 export_handle/create_from_handle 拿到带 reader_id 的
# 视图，各自维护读游标，互不消费对方的消息——这正是 shm_broadcast 的广播本质。
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass


# SOURCE: vllm/utils/system_utils.py:L168
def get_mp_context():
    # 真实代码用 vllm.utils 的 get_mp_context()（spawn 上下文）；本章线程模型下
    # 仅需一个能 .Lock() 的对象占位（spawn/Process 相关已 SUBTRACTED）。
    import multiprocessing

    return multiprocessing.get_context("spawn")


@dataclass
class Handle:
    # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py:L343
    # host 替身只保留 export/create_from_handle 闭环所需字段。
    """export_handle 的返回值：让其它 reader 连回同一条广播队列。"""

    mq: "MessageQueue"


# SOURCE: vllm/distributed/device_communicators/shm_broadcast.py:L353
class MessageQueue:
    """单写多读广播队列的进程内替身。

    n_reader 个 reader 各有一条独立的 queue.Queue；enqueue 把同一条消息分发给
    所有 reader（广播）。reader 侧通过 create_from_handle(handle, reader_id) 绑定
    到自己的那条 queue。1 写 1 读时（n_reader==1）退化为普通点对点队列。
    """

    # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py:L354
    def __init__(self, n_reader: int, n_local_reader: int | None = None, **kwargs):
        self.n_reader = n_reader
        # 每个 reader 一条队列；广播 = 往所有队列各放一份。
        self._queues = [queue.Queue() for _ in range(max(1, n_reader))]
        self._ready = threading.Event()
        self._ready.set()
        # 本视图绑定的 reader 槽位（writer 视图为 None）。
        self._reader_id: int | None = None

    # --- writer 侧 ---
    # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py:L703
    def enqueue(self, item) -> None:
        for q in self._queues:
            q.put(item)

    # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py:L445
    def export_handle(self) -> Handle:
        return Handle(mq=self)

    # --- reader 侧 ---
    @classmethod
    def create_from_handle(cls, handle: Handle, reader_id: int) -> "MessageQueue":
        # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py:L449
        # 返回一个绑定到 reader_id 槽位的视图，共享底层队列。
        view = object.__new__(cls)
        view.n_reader = handle.mq.n_reader
        view._queues = handle.mq._queues
        view._ready = handle.mq._ready
        view._reader_id = reader_id % max(1, handle.mq.n_reader)
        return view

    # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py:L748
    def dequeue(self, timeout: float | None = None, indefinite: bool = False):
        rid = self._reader_id if self._reader_id is not None else 0
        q = self._queues[rid]
        if indefinite:
            return q.get()
        try:
            return q.get(timeout=timeout)
        except queue.Empty as e:
            raise TimeoutError("MessageQueue.dequeue timed out") from e

    # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py:L496
    def wait_until_ready(self) -> None:
        self._ready.wait()
