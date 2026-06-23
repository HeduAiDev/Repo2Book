# 只做减法的忠实精简版 —— 镜像 vllm/distributed/device_communicators/shm_broadcast.py 的
# MessageQueue 对外契约（pin f3fef123）：enqueue / dequeue / export_handle /
# create_from_handle / wait_until_ready / shutdown。
# 与 vLLM 同名、同结构、同控制流；只删不增。
#
# 减法边界（subtract-only 的唯一结构性裁剪，明确注明）：真实 MessageQueue 是基于
# 共享内存环形缓冲（POSIX shm + ZMQ 远程 fallback）的『1 写 N 读』零拷贝**广播**队列，机制有数百行。
# 本章关心的是它的**可观察语义**——FIFO、一次 enqueue 被**所有** reader 看到、export/import handle
# 跨进程复用同一队列——而非 shm 字节布局。故 SUBTRACTED 掉 shm 环形缓冲实现，用 multiprocessing
# (spawn 上下文) 的 Queue 背书等价语义：
#   - 广播 MQ（n_reader=N）：内部 N 个 FIFO Queue（每 reader 一个），enqueue 向全部 put，
#     reader r 只读自己的队列 → 复现『一次 enqueue 被 N 个 reader 各看到一份』；
#   - 应答 MQ（n_reader=1）：单 FIFO Queue。
# dossier theory 的 FIFO 配对正确性论证只依赖 FIFO，不依赖 shm。控制平面 enqueue/dequeue/handle
# 控制流 1:1。底层 Queue 经 spawn 时只能由『进程参数继承』跨进程传递（不能再经 Pipe 二次 pickle），
# 故执行器把这些 Queue 作为子进程 kwargs 传入。
#
# SUBTRACTED: shm 环形缓冲读写、ShmRingBuffer、ZMQ 远程订阅、max_chunk_bytes 分块、
#   local/remote reader 区分等全部底层机制（vllm/distributed/device_communicators/shm_broadcast.py
#   整文件）—— 与本章控制平面广播/应答的 FIFO 语义正交。

import multiprocessing
from dataclasses import dataclass, field

# 用 spawn 上下文（与真实 vllm 默认启动法一致）创建底层 FIFO Queue。
_CTX = multiprocessing.get_context("spawn")


@dataclass
class Handle:
    # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py  Handle (字段裁剪)
    # SUBTRACTED: 真实 Handle 含 shm name / ZMQ 地址 / remote_subscribe_addr 等 shm/ZMQ 寻址字段——
    #   本镜像用底层 Queue 列表引用即可寻址同一队列。
    queues: list = field(default_factory=list)  # 每 reader 一个 FIFO（广播）；应答 MQ 为单元素
    local_reader_ranks: list = field(default_factory=lambda: [0])
    remote_subscribe_addr = None


class MessageQueue:
    # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py  MessageQueue (契约镜像)
    def __init__(
        self,
        n_reader,
        n_local_reader=None,
        max_chunk_bytes=None,
        connect_ip=None,
        _queues=None,
        _reader_rank=None,
    ):
        # SUBTRACTED: shm 环形缓冲分配、ZMQ socket 建立、reader 注册等（真实 __init__ 数十行）。
        self.n_reader = n_reader
        # writer 端持有全部 reader 的队列（广播时向全部 put）；reader 端只绑定自己那个。
        self._queues = (
            [_CTX.Queue() for _ in range(n_reader)] if _queues is None else _queues
        )
        self._reader_rank = _reader_rank  # None=writer 端；否则为本 reader 在 _queues 的下标

    # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py export_handle
    def export_handle(self) -> Handle:
        # 真实实现导出 shm name / ZMQ 地址供 reader 进程 attach；这里导出同一底层 FIFO 列表。
        return Handle(queues=self._queues)

    @staticmethod
    # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py create_from_handle
    def create_from_handle(handle: Handle, reader_rank) -> "MessageQueue":
        # 真实实现按 handle 把本进程注册为某 reader rank；这里把本进程绑定到 reader_rank 的队列。
        idx = reader_rank % len(handle.queues)
        return MessageQueue(
            len(handle.queues), _queues=handle.queues, _reader_rank=idx
        )

    # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py wait_until_ready
    def wait_until_ready(self) -> None:
        # SUBTRACTED: 真实的 writer↔readers 握手（确认所有 reader attach 完成）；FIFO 后端无需握手。
        return

    # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py enqueue
    def enqueue(self, obj) -> None:
        # 一次 enqueue 被所有 reader 看到：广播 MQ 向每个 reader 的 FIFO 各 put 一份。
        # SUBTRACTED: 真实按 n_reader 维护 shm 槽位的读计数，待所有 reader 读毕才回收槽位；
        #   shm 是一份数据零拷贝，这里以『各 reader 一份』表达等价的可观察语义。
        for q in self._queues:
            q.put(obj)

    # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py dequeue
    def dequeue(self, timeout=None, indefinite=False):
        # FIFO 取下一条；reader 只从自己绑定的队列取（writer 端默认取第 0 个，用于单 reader 应答 MQ）。
        q = self._queues[self._reader_rank if self._reader_rank is not None else 0]
        if indefinite:
            return q.get()
        return q.get(timeout=timeout)

    # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py shutdown
    def shutdown(self) -> None:
        # SUBTRACTED: 真实释放 shm / 关闭 ZMQ socket；FIFO 后端无显式资源需释放。
        return
