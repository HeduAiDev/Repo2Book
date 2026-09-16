# SOURCE: vllm/distributed/device_communicators/shm_broadcast.py
# HOST SEAM：MessageQueue —— 真实类是 SHM 环形缓冲 + ZMQ XPUB 的广播队列
# (shm_broadcast.py:L465+)；本章只消费 GroupCoordinator.__init__ 的
# create_from_process_group(cpu_group, 1<<22, 6) 与 broadcast_object 的可观察
# 契约（src==0 的对象广播、全员一致）。seam 以 torch.distributed（gloo cpu_group）
# 实现同一集合语义——不发明真实类没有的行为。

from __future__ import annotations

import torch
import torch.distributed as dist


# SOURCE: vllm/distributed/device_communicators/shm_broadcast.py:L465-L732 MessageQueue
class MessageQueue:  # HOST SEAM
    # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py:L466-L555
    #   __init__（属性面载体：真实 __init__ 交换 SHM/ZMQ handle；seam 只记组与
    #   writer）
    def __init__(self, group, writer_global_rank: int):
        # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py:L466-L555（锚点双置）
        self.group = group
        self.writer_global_rank = writer_global_rank
        self.n_reader = 0  # seam: 集合语义承载，无环形缓冲读者账

    # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py:L919-L923
    #   broadcast_object —— 对象广播（真实：SHM 写入/读取；seam：broadcast_
    #   object_list on cpu_group）
    def broadcast_object(self, obj):
        # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py:L919-L923（锚点双置）
        holder = [obj]
        dist.broadcast_object_list(
            holder, src=self.writer_global_rank, group=self.group
        )
        return holder[0]

    # SUBTRACTED: enqueue/dequeue/wait_until_ready/shutdown(L641-L646)/handle
    #   导出面（export_handle L557-L558 / create_from_handle L561-L606 /
    #   acquire_write·read L649-L822）——executor 控制面（ch17）的消费接口，
    #   本章不触达；真实 MessageQueue 亦无 destroy 方法（GroupCoordinator 销毁
    #   只置 mq_broadcaster=None，见 parallel_state.py:L1237-L1239 的对照版）。

    # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py:L978-L1059
    #   create_from_process_group —— 真实版：writer 构建 SHM/ZMQ handle 并经
    #   process group 广播给读者；seam 保留『经 cpu_group 的集合握手』这一步。
    @staticmethod
    def create_from_process_group(
        cpu_group,
        max_chunk_bytes,
        max_chunks,
        writer_rank: int = 0,
        external_writer_handle=None,
        blocking: bool = True,
    ):
        # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py:L978-L1059（锚点双置）
        ranks = dist.get_process_group_ranks(cpu_group)
        writer_global_rank = ranks[writer_rank]
        # 集合握手：writer 的 handle 广播给全员（handle 本体是 SHM 域细节）
        handle = {"writer": writer_global_rank, "shm": None}  # HOST SEAM
        dist.broadcast_object_list(
            [handle], src=writer_global_rank, group=cpu_group
        )
        return MessageQueue(cpu_group, writer_global_rank)
