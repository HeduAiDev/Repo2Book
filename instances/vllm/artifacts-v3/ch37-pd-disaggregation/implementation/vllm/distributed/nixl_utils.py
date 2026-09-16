# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL 传输库的宿主替身（本章唯一的 SEAM）。

# SOURCE: vllm/distributed/nixl_utils.py:L1-L99（真实实现在惰性 import 的 nixl 包里）
# SEAM: 真源码这里做的是「惰性导入外部 nixl 包的 nixl_agent」，本文件把它换成
#   一个**进程内**实现同一套契约的替身。契约逐条对照：

#   | 真 NIXL（RDMA 单边）                     | 本替身                            | 保持的可观察语义 |
#   |------------------------------------------|-----------------------------------|------------------|
#   | register_memory(descs) 把 KV 张量登记成可被远端读写的区域 | 记下 (addr,len,dev) 区域 | 只有登记过的区域能被传输 |
#   | get_agent_metadata()/add_remote_agent()  | msgpack 往返 agent 名+区域表      | 握手交换的是「基址/块长/块数」 |
#   | get_xfer_descs(Nx3 addr/len/dev)/prep_xfer_dlist | 同结构，按索引寻址        | **描述符 = (addr,len) 对**，传输按块号索引 |
#   | make_prepped_xfer('READ'/'WRITE')        | 记录一份待办传输                  | 方向语义：READ=远端→本地，WRITE=本地→远端 |
#   | transfer() 非阻塞发起                    | 立即执行本次搬迁（host memcpy 是瞬时的） | 发起后不阻塞调用方；notif 随传输完成送达对端 |
#   | check_xfer_state() -> DONE/PROC          | 恒 DONE（搬迁已完成）             | 轮询式完成判定 |
#   | send_notif()/get_new_notifs()            | 进程内收发箱                      | notif 在**传输完成时**到达对端，按发送方分组 |

#   偏离面（必须显式承认）：真 RDMA 传输是**双机异步**的，`check_xfer_state` 会先
#   返回若干次 PROC；本替身在同一进程内 memcpy，故 PROC 分支在多步轮询里基本观察
#   不到（代码路径逐字保留，见 base_worker.py 的 _pop_done_transfers）。除此之外，
#   「D 直接从 P 的显存地址取数、P 的 CPU 不参与」这一**单边语义**被完整保留：
#   传输只读/写描述符里的裸地址，不经 P 的任何 Python 代码路径。
"""

import ctypes
import threading
from typing import Any

import msgspec
import numpy as np

from vllm.logger import init_logger

logger = init_logger(__name__)

# 进程内 agent 表：真 NIXL 里这一层由 RDMA 子网 + agent 名寻址承担。
_AGENTS: dict[str, "_Agent"] = {}
_REGISTRY_LOCK = threading.RLock()


# SOURCE: nixl._api.nixl_agent（远端 agent 侧状态）
class _Agent:
    """一个 NIXL agent 的进程内状态：登记区域 + 收发箱 + 传输句柄。"""

    # SOURCE: nixl._api.nixl_agent（远端 agent 侧状态）
    def __init__(self, name: str) -> None:
        # SOURCE: nixl._api.nixl_agent（远端 agent 侧状态）
        self.name = name
        # (base_addr, size, device_id)
        self.regions: list[tuple[int, int, int]] = []
        # 发送方 agent 名 -> notif 列表
        self.notifs: dict[str, list[bytes]] = {}
        # dlist handle -> (agent_name | None, descs 数组)
        self.dlists: dict[int, tuple[str | None, np.ndarray]] = {}
        # xfer handle -> (op, local_dlist, local_idx, remote_dlist, remote_idx, notif)
        self.xfers: dict[int, tuple] = {}
        self.next_handle = 1


# SOURCE: vllm/distributed/nixl_utils.py:L13-L18（NixlWrapper 声明）+ L57-L76（惰性解析）
class NixlWrapper:
    """NIXL agent 句柄（契约见模块 docstring 的 SEAM 表）。"""

    # SOURCE: vllm/distributed/nixl_utils.py:L13-L18（NixlWrapper 声明）+ L57-L76（惰性解析）
    def __init__(self, agent_name: str, config: Any = None) -> None:
        self.agent_name = agent_name
        self.config = config
        with _REGISTRY_LOCK:
            _AGENTS[agent_name] = _Agent(agent_name)
        self._agent = _AGENTS[agent_name]

    # ---- 内存登记 ---------------------------------------------------- #

    # SOURCE: vllm/distributed/nixl_utils.py:L13-L18（NixlWrapper 声明）+ L57-L76（惰性解析）
    def get_reg_descs(self, caches_data: list, mem_type: str) -> "_RegDescs":
        return _RegDescs(list(caches_data), mem_type)

    # SOURCE: vllm/distributed/nixl_utils.py:L13-L18（NixlWrapper 声明）+ L57-L76（惰性解析）
    def register_memory(self, descs: "_RegDescs", backends: list[str] | None = None):
        with _REGISTRY_LOCK:
            for addr, size, dev, _ in descs.data:
                self._agent.regions.append((int(addr), int(size), int(dev)))

    # SOURCE: vllm/distributed/nixl_utils.py:L13-L18（NixlWrapper 声明）+ L57-L76（惰性解析）
    def deregister_memory(self, descs: "_RegDescs") -> None:
        with _REGISTRY_LOCK:
            for addr, _size, _dev, _ in descs.data:
                self._agent.regions = [
                    r for r in self._agent.regions if r[0] != int(addr)
                ]

    # SOURCE: vllm/distributed/nixl_utils.py:L13-L18（NixlWrapper 声明）+ L57-L76（惰性解析）
    def get_agent_metadata(self) -> bytes:
        with _REGISTRY_LOCK:
            payload = {
                "name": self.agent_name,
                "regions": [list(r) for r in self._agent.regions],
            }
        return msgspec.msgpack.encode(payload)

    # SOURCE: vllm/distributed/nixl_utils.py:L13-L18（NixlWrapper 声明）+ L57-L76（惰性解析）
    def add_remote_agent(self, metadata: bytes) -> str:
        """交换 agent 描述符：真 NIXL 在这里建立 RDMA 通路并校验远端可达。"""
        payload = msgspec.msgpack.decode(metadata)
        remote_name = payload["name"]
        with _REGISTRY_LOCK:
            remote = _AGENTS.setdefault(remote_name, _Agent(remote_name))
            remote.regions = [tuple(r) for r in payload["regions"]]
        return remote_name

    # SOURCE: vllm/distributed/nixl_utils.py:L13-L18（NixlWrapper 声明）+ L57-L76（惰性解析）
    def remove_remote_agent(self, agent_name: str) -> None:  # noqa: ARG002
        return None

    # ---- 传输 -------------------------------------------------------- #

    # SOURCE: vllm/distributed/nixl_utils.py:L13-L18（NixlWrapper 声明）+ L57-L76（惰性解析）
    def get_xfer_descs(self, blocks_data: np.ndarray, mem_type: str) -> "_XferDescs":
        return _XferDescs(np.asarray(blocks_data, dtype=np.uint64), mem_type)

    # SOURCE: vllm/distributed/nixl_utils.py:L13-L18（NixlWrapper 声明）+ L57-L76（惰性解析）
    def prep_xfer_dlist(self, agent_name: str, descs: "_XferDescs") -> int:
        with _REGISTRY_LOCK:
            handle = self._agent.next_handle
            self._agent.next_handle += 1
            # "NIXL_INIT_AGENT" = 本地描述符列表（真 NIXL 的同名哨兵）。
            peer = None if agent_name == "NIXL_INIT_AGENT" else agent_name
            self._agent.dlists[handle] = (peer, descs.data)
        return handle

    # SOURCE: vllm/distributed/nixl_utils.py:L13-L18（NixlWrapper 声明）+ L57-L76（惰性解析）
    def make_prepped_xfer(
        self,
        operation: str,
        local_xfer_side_handle: int,
        local_block_descs_ids: Any,
        remote_xfer_side_handle: int,
        remote_block_descs_ids: Any,
        notif_msg: bytes | None = None,
    ) -> int:
        with _REGISTRY_LOCK:
            handle = self._agent.next_handle
            self._agent.next_handle += 1
            self._agent.xfers[handle] = (
                operation,
                local_xfer_side_handle,
                np.asarray(local_block_descs_ids, dtype=np.int64),
                remote_xfer_side_handle,
                np.asarray(remote_block_descs_ids, dtype=np.int64),
                notif_msg,
            )
        return handle

    # SOURCE: vllm/distributed/nixl_utils.py:L13-L18（NixlWrapper 声明）+ L57-L76（惰性解析）
    def transfer(self, handle: int) -> str:
        """发起传输。真 NIXL 立即返回句柄、由网卡异步推进；替身当场搬迁完。"""
        with _REGISTRY_LOCK:
            (
                operation,
                local_handle,
                local_ids,
                remote_handle,
                remote_ids,
                notif_msg,
            ) = self._agent.xfers[handle]
            local_peer, local_descs = self._agent.dlists[local_handle]
            remote_peer, remote_descs = self._agent.dlists[remote_handle]

        if operation == "READ":
            src_descs, src_ids = remote_descs, remote_ids
            dst_descs, dst_ids = local_descs, local_ids
            peer_name = remote_peer
        elif operation == "WRITE":
            src_descs, src_ids = local_descs, local_ids
            dst_descs, dst_ids = remote_descs, remote_ids
            peer_name = remote_peer
        else:
            raise ValueError(f"Unsupported transfer operation: {operation}")

        assert len(src_ids) == len(dst_ids), (
            f"{operation}: {len(src_ids)} source descs != {len(dst_ids)} dst descs"
        )
        for s_idx, d_idx in zip(src_ids.tolist(), dst_ids.tolist()):
            src_addr, src_len, _ = src_descs[s_idx]
            dst_addr, dst_len, _ = dst_descs[d_idx]
            self._memmove(int(dst_addr), int(src_addr), min(int(src_len), int(dst_len)))

        # 单边操作的完成通知送给对端（真 NIXL：notif 随传输完成从本地网卡发出）。
        if notif_msg is not None and peer_name is not None:
            self._deliver_notif(peer_name, notif_msg)
        return "DONE"

    # SOURCE: vllm/distributed/nixl_utils.py:L13-L18（NixlWrapper 声明）+ L57-L76（惰性解析）
    def _deliver_notif(self, peer_name: str, notif_msg: bytes) -> None:
        with _REGISTRY_LOCK:
            peer = _AGENTS.get(peer_name)
            if peer is None:
                raise RuntimeError(f"Unknown NIXL agent {peer_name!r}")
            peer.notifs.setdefault(self.agent_name, []).append(bytes(notif_msg))

    # SOURCE: vllm/distributed/nixl_utils.py:L13-L18（NixlWrapper 声明）+ L57-L76（惰性解析）
    def check_xfer_state(self, handle: int) -> str:
        with _REGISTRY_LOCK:
            return "DONE" if handle in self._agent.xfers else "ERR"

    # SOURCE: vllm/distributed/nixl_utils.py:L13-L18（NixlWrapper 声明）+ L57-L76（惰性解析）
    def release_xfer_handle(self, handle: int) -> None:
        with _REGISTRY_LOCK:
            self._agent.xfers.pop(handle, None)

    # SOURCE: vllm/distributed/nixl_utils.py:L13-L18（NixlWrapper 声明）+ L57-L76（惰性解析）
    def release_dlist_handle(self, handle: int) -> None:
        with _REGISTRY_LOCK:
            self._agent.dlists.pop(handle, None)

    # ---- 通知 -------------------------------------------------------- #

    # SOURCE: vllm/distributed/nixl_utils.py:L13-L18（NixlWrapper 声明）+ L57-L76（惰性解析）
    def send_notif(self, agent_name: str, notif_msg: bytes) -> None:
        self._deliver_notif(agent_name, notif_msg)

    # SOURCE: vllm/distributed/nixl_utils.py:L13-L18（NixlWrapper 声明）+ L57-L76（惰性解析）
    def get_new_notifs(self) -> dict[str, list[bytes]]:
        with _REGISTRY_LOCK:
            notifs = self._agent.notifs
            self._agent.notifs = {}
        return notifs

    # SUBTRACTED: get_xfer_telemetry / get_partial_xfer_telemetry（NIXL 遥测）——
    #   遥测上报按减法计划删除（stats 旁路），调用点见 base_worker 的删除标记。

    # SOURCE: vllm/distributed/nixl_utils.py:L13-L18（xfer 后端契约位）
    @staticmethod
    def _memmove(dst_addr: int, src_addr: int, num_bytes: int) -> None:
        # SOURCE: vllm/distributed/nixl_utils.py:L13-L18（xfer 后端契约位）
        if num_bytes <= 0 or dst_addr == src_addr:
            return
        ctypes.memmove(dst_addr, src_addr, num_bytes)

    # SOURCE: vllm/distributed/nixl_utils.py:L13-L18（NixlWrapper 声明）+ L57-L76（惰性解析）
    def shutdown(self) -> None:
        with _REGISTRY_LOCK:
            _AGENTS.pop(self.agent_name, None)


# SOURCE: vllm/distributed/nixl_utils.py:L13-L18（nixl 属性契约位）
class _RegDescs:
    """register_memory 的入参：[(base_addr, size, device_id, "")]。"""

    # SOURCE: vllm/distributed/nixl_utils.py:L13-L18（nixl 属性契约位）
    def __init__(self, data: list, mem_type: str) -> None:
        # SOURCE: vllm/distributed/nixl_utils.py:L13-L18（nixl 属性契约位）
        self.data = data
        self.mem_type = mem_type


# SOURCE: vllm/distributed/nixl_utils.py:L13-L18（nixl 属性契约位）
class _XferDescs:
    """get_xfer_descs 的产物：按**块号索引**寻址的 (addr, len, dev) 数组。"""

    # SOURCE: vllm/distributed/nixl_utils.py:L13-L18（nixl 属性契约位）
    def __init__(self, data: np.ndarray, mem_type: str) -> None:
        # SOURCE: vllm/distributed/nixl_utils.py:L13-L18（nixl 属性契约位）
        self.data = data
        self.mem_type = mem_type


# SOURCE: nixl._api.nixl_agent_config
def nixl_agent_config(backends: list[str] | None = None, num_threads: int = 0, **kw):
    """真 NIXL 的 agent 配置构造器（本章只需把它透传给 wrapper）。"""
    return {"backends": backends, "num_threads": num_threads, **kw}


# SOURCE: nixl._api.nixlXferTelemetry
class nixlXferTelemetry:
    """遥测结构（减法计划已删其消费点，仅保留类型位）。"""


# SOURCE: vllm/distributed/nixl_utils.py:L88-L95
def is_nixl_available() -> bool:
    """本章恒可用（seam 是进程内实现）。"""
    return True


__all__ = [
    "NixlWrapper",
    "nixl_agent_config",
    "nixlXferTelemetry",
    "is_nixl_available",
]
