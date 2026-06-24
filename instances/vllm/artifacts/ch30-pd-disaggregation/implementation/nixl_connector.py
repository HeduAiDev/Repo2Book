# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
NIXL high-performance RDMA connector — facade 模式（顶层按 role 只建半边子对象）。

SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py
        + vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py
worker.start_load_kv 对每个待收请求发非阻塞 RDMA READ（make_prepped_xfer('READ')+
transfer），D 端主动从 P 端显存单边读 KV，P 端 CPU 不参与；首遇远端先后台握手交换
内存注册元数据。get_finished 轮询：本地 handle 转 DONE 算收完成、对端通知算发完成。
wait_for_layer_load/save_kv_layer 为 no-op（NIXL 不逐层、不显式 save）。
"""

import threading
from collections import defaultdict
from queue import Queue

from .base import KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:ReqMeta (nixl)
class NixlReqMeta:
    # SUBTRACTED: local_physical_block_ids/host buffer/异构 TP 字段（worker.py ReqMeta），
    #             对称 TP 主干只需 remote_engine_id + 本地/远端 block_ids。
    def __init__(self, req_id, remote_engine_id, local_block_ids, remote_block_ids):
        # SOURCE: vllm/.../nixl/connector.py:ReqMeta (nixl)
        self.req_id = req_id
        self.remote_engine_id = remote_engine_id
        self.local_block_ids = local_block_ids
        self.remote_block_ids = remote_block_ids


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:NixlConnectorMetadata
class NixlConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        # SOURCE: vllm/.../nixl/connector.py:NixlConnectorMetadata.__init__
        self.reqs_to_recv = {}   # req_id -> NixlReqMeta（D 侧本步要 READ 的请求）
        self.reqs_to_send = {}   # req_id -> expiration_time（P 侧待 D 来读的请求）

    def add_recv(self, meta: NixlReqMeta):
        # SOURCE: vllm/.../nixl/connector.py:NixlConnectorMetadata.add_new_req
        self.reqs_to_recv[meta.req_id] = meta


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/nixl_wrapper（NixlWrapper 封装）
class _FakeNixlWrapper:
    # SUBTRACTED: 真实是对 nixl 库 RDMA agent 的封装（注册内存、建 xfer descriptor、
    #             单边 READ）。精简版 loopback：make_prepped_xfer 记下『从哪个对端读』，
    #             transfer 立刻把对端 block 标为可读，handle 状态机 PROC→DONE 由测试
    #             显式推进（驱动 _pop_done_transfers 轮询），并支持 send_notif 把
    #             read-done 通知投递给对端 agent（驱动 _get_new_notifs）。忠实保留
    #             『非阻塞 READ + handle 轮询 + 通知』三件套，剥离真实 RDMA 细节。
    def __init__(self, agent_name):
        # SOURCE: vllm/.../nixl/nixl_wrapper.py:NixlWrapper.__init__
        self.agent_name = agent_name
        self._handles = {}            # handle_id -> "PROC"/"DONE"/"ERR"
        self._next_handle = 1
        self._inbox = defaultdict(list)  # agent_name -> [notif_msg,...]
        self._peers = {}              # agent_name -> _FakeNixlWrapper

    def connect(self, agent_name, peer):
        # SOURCE: vllm/.../nixl/nixl_wrapper.py:NixlWrapper.add_remote_agent
        self._peers[agent_name] = peer

    def make_prepped_xfer(self, op, local_side, local_descs, remote_side,
                          remote_descs, notif_msg=None):
        # SOURCE: vllm/.../nixl/nixl_wrapper.py:NixlWrapper.make_prepped_xfer
        assert op == "READ"
        h = self._next_handle
        self._next_handle += 1
        self._handles[h] = "PROC"
        return h

    def transfer(self, handle):
        # SOURCE: vllm/.../nixl/nixl_wrapper.py:NixlWrapper.transfer
        # 非阻塞：发起即返回，真正完成由 complete() 推进。
        return

    def complete(self, handle):
        # SOURCE: vllm/.../nixl/nixl_wrapper.py:NixlWrapper.check_xfer_state（DONE 状态）
        # 测试用钩子：把一个在途 READ 标为 DONE（模拟 RDMA 读完成）。
        self._handles[handle] = "DONE"

    def check_xfer_state(self, handle):
        # SOURCE: vllm/.../nixl/nixl_wrapper.py:NixlWrapper.check_xfer_state
        return self._handles.get(handle, "ERR")

    def release_xfer_handle(self, handle):
        # SOURCE: vllm/.../nixl/nixl_wrapper.py:NixlWrapper.release_xfer_handle
        self._handles.pop(handle, None)

    def send_notif(self, agent_name, notif_msg):
        # SOURCE: vllm/.../nixl/nixl_wrapper.py:NixlWrapper.send_notif
        self._peers[agent_name]._inbox[agent_name].append(notif_msg)

    def get_new_notifs(self):
        # SOURCE: vllm/.../nixl/nixl_wrapper.py:NixlWrapper.get_new_notifs
        out = dict(self._inbox)
        self._inbox = defaultdict(list)
        return out


class NixlConnectorWorker:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py:NixlConnectorWorker
    def __init__(self, engine_id, nixl_wrapper, world_size=1):
        # SOURCE: vllm/.../nixl/worker.py:NixlConnectorWorker.__init__
        # SUBTRACTED: kv_cache 注册 / transfer_topo / host buffer / MLA / Mamba 等大量
        #             worker 状态（worker.py __init__）—— dossier delete 项（异构 TP / SSM /
        #             主机缓冲进阶路径）。对称 TP 主干只需下列收发记账容器。
        self.engine_id = engine_id
        self.nixl_wrapper = nixl_wrapper
        self.world_size = world_size
        self.use_host_buffer = False
        self.copy_blocks = None
        self._remote_agents = {}                      # engine_id -> agent handle
        self._recving_transfers = defaultdict(list)   # req_id -> [xfer handle,...]
        self._recving_metadata = {}                   # req_id -> NixlReqMeta
        self._ready_requests = Queue()                # 握手完成后待发 READ 的 (req_id, meta)
        self._reqs_to_send = {}                       # P 侧 req_id -> 过期时间（兜底）
        self._reqs_to_process = set()
        self._failed_recv_reqs = set()
        self._handshake_lock = threading.Lock()

    def start_load_kv(self, metadata: NixlConnectorMetadata):
        # SOURCE: vllm/.../nixl/worker.py:L1840-L1895
        """Start loading by triggering non-blocking nixl_xfer.
        We check for these trnxs to complete in each step()."""
        for req_id, meta in metadata.reqs_to_recv.items():
            remote_engine_id = meta.remote_engine_id
            # always store metadata for failure recovery
            self._recving_metadata[req_id] = meta
            if remote_engine_id not in self._remote_agents:
                # Initiate handshake with remote engine to exchange metadata.
                with self._handshake_lock:
                    if remote_engine_id not in self._remote_agents:
                        self._background_nixl_handshake(req_id, remote_engine_id, meta)
                        continue

            # Handshake already completed, start async read xfer.
            self._read_blocks_for_req(req_id, meta)

        # Start transfers for requests whose handshakes have now finished.
        while not self._ready_requests.empty():
            self._read_blocks_for_req(*self._ready_requests.get_nowait())

        # SUBTRACTED: reqs_in_batch / reqs_not_processed 的过期管理（worker L1877-L1891）
        #             随异步调度错位兜底删去；保留 reqs_to_send 登记（P 侧防 D 失联占块）。
        for req_id, expiration_time in metadata.reqs_to_send.items():
            self._reqs_to_process.add(req_id)
            self._reqs_to_send[req_id] = expiration_time

    def _background_nixl_handshake(self, req_id, remote_engine_id, meta):
        # SOURCE: vllm/.../nixl/worker.py:_background_nixl_handshake
        # SUBTRACTED: 真实后台线程经 ZMQ 向远端拉 agent metadata 再 add_remote_agent
        #             （worker.py），精简版同步完成握手（记下远端 agent）并把请求放进
        #             _ready_requests 待 start_load_kv 末尾补发 READ —— 忠实保留『首遇
        #             远端先握手、握手完才发 READ』的次序。
        self._remote_agents[remote_engine_id] = self._peer_agent_name(remote_engine_id)
        self._ready_requests.put((req_id, meta))

    def add_remote_agent(self, remote_engine_id, agent_name):
        # SOURCE: vllm/.../nixl/worker.py:add_remote_agent
        self._remote_agents[remote_engine_id] = agent_name

    def _peer_agent_name(self, remote_engine_id):
        # SOURCE: vllm/.../nixl/worker.py:_remote_agents 查表（add_remote_agent 的反向）
        # 精简版：远端 agent 名即 engine_id（loopback 用它向对端 send_notif）。
        return self._remote_agents.get(remote_engine_id, remote_engine_id)

    def _read_blocks_for_req(self, req_id, meta):
        # SOURCE: vllm/.../nixl/worker.py:L1897-L1978
        # SUBTRACTED: remote_ranks 多读循环 / tp_ratio<0 MLA 单读 / mamba 块展开
        #             （worker L1900-L1978，dossier delete 项：异构 TP / MLA / SSM 路径）。
        #             对称 TP 下单远端 rank 直接 _read_blocks。
        self._read_blocks(
            local_block_ids=meta.local_block_ids,
            remote_block_ids=meta.remote_block_ids,
            dst_engine_id=meta.remote_engine_id,
            request_id=req_id,
            remote_request_id=req_id,
            remote_rank=0,
            local_xfer_side_handle=0,
            remote_xfer_side_handle=0,
        )

    def _read_blocks(self, local_block_ids, remote_block_ids, dst_engine_id,
                     request_id, remote_request_id, remote_rank,
                     local_xfer_side_handle, remote_xfer_side_handle):
        # SOURCE: vllm/.../nixl/worker.py:L1980-L2109
        """Post a READ point-to-point xfer request from a single local worker
        to a single remote worker."""
        # SUBTRACTED: block_size_ratio>1 重映射 / get_mapped_blocks（worker L1997-L2024）—
        #             异构 block_size 路径（dossier delete 项）。
        # Number of D TP workers that will read from dst P. Propagate info
        # on notification so that dst worker can wait before freeing blocks.
        notif_id = f"{remote_request_id}:{self.world_size}".encode()

        # Full prefix cache hit: do not need to read remote blocks,
        # just notify P worker that we have the blocks we need.
        if len(local_block_ids) == 0:
            agent_name = self._remote_agents[dst_engine_id]
            self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
            return

        # SUBTRACTED: partial-prefix 裁剪 / mamba 组保护 / descs id 计算（worker
        #             L2059-L2091）—— 对称 TP 整请求一把读，精简版直接发 READ。
        remote_block_descs_ids = remote_block_ids
        local_block_descs_ids = local_block_ids

        # Prepare transfer with Nixl.
        handle = None
        try:
            handle = self.nixl_wrapper.make_prepped_xfer(
                "READ",
                local_xfer_side_handle,
                local_block_descs_ids,
                remote_xfer_side_handle,
                remote_block_descs_ids,
                notif_msg=notif_id,
            )

            # Begin async xfer.
            self.nixl_wrapper.transfer(handle)

            # Use handle to check completion in future step().
            self._recving_transfers[request_id].append(handle)
        except Exception:
            # SUBTRACTED: _handle_failed_transfer 失败块标记（worker L2110+）。
            if handle is not None:
                self.nixl_wrapper.release_xfer_handle(handle)
            raise

    def get_finished(self):
        # SOURCE: vllm/.../nixl/worker.py:L1651-L1730
        """Get requests that are done sending or recving on this specific
        worker."""
        done_sending = self._get_new_notifs()
        done_recving = self._pop_done_transfers(self._recving_transfers)

        # add requests that skipped transfer to done_recving
        done_recving.update(self._failed_recv_reqs)
        self._failed_recv_reqs.clear()

        for req_id in done_recving:
            # clean up metadata for completed requests
            self._recving_metadata.pop(req_id, None)
        # SUBTRACTED: host buffer 回拷 / 异构 block_size·attn 后处理（worker L1681-L1709）。

        # Handle timeout to avoid stranding blocks on remote.
        # SUBTRACTED: reqs_to_send 基于 perf_counter 的过期回收循环（worker L1710-L1728）—
        #             兜底机制，精简版保留容器但不在此跑超时清扫（dossier delete 项）。

        return done_sending, done_recving

    def _get_new_notifs(self):
        # SOURCE: vllm/.../nixl/worker.py:L1732-L1775
        """Get req_ids which got a remote xfer message."""
        # SUBTRACTED: 多消费者 (异构 TP) 计数等齐再释放（worker L1755-L1773）。对称 TP
        #             下一条通知即代表该 P 侧请求被读完，可释放。
        notified_req_ids = set()
        for notifs in self.nixl_wrapper.get_new_notifs().values():
            for notif in notifs:
                req_id, _tp_size = notif.decode("utf-8").rsplit(":", 1)
                if req_id not in self._reqs_to_send and req_id not in self._reqs_to_process:
                    continue
                notified_req_ids.add(req_id)
                self._reqs_to_process.discard(req_id)
                self._reqs_to_send.pop(req_id, None)
        return notified_req_ids

    def _pop_done_transfers(self, transfers):
        # SOURCE: vllm/.../nixl/worker.py:L1777-L1822
        """Pop completed xfers by checking for DONE state. Returns set of
        req_ids that have all done xfers."""
        done_req_ids = set()
        for req_id, handles in list(transfers.items()):
            in_progress = []
            for handle in handles:
                xfer_state = self.nixl_wrapper.check_xfer_state(handle)
                if xfer_state == "DONE":
                    # SUBTRACTED: get_xfer_telemetry / xfer_stats 记账（worker L1792-L1794）。
                    self.nixl_wrapper.release_xfer_handle(handle)
                elif xfer_state == "PROC":
                    in_progress.append(handle)
                    continue
                else:
                    self._handle_failed_transfer(req_id, handle)

            if not in_progress:
                # Only report request as completed when all transfers are done.
                done_req_ids.add(req_id)
                del transfers[req_id]
            else:
                transfers[req_id] = in_progress
        return done_req_ids

    def _handle_failed_transfer(self, req_id, handle):
        # SOURCE: vllm/.../nixl/worker.py:L1824
        # SUBTRACTED: 把该请求所有逻辑块标 invalid 并记 stats（worker L1824+）。精简版
        #             只把请求记入 _failed_recv_reqs（让 get_finished 仍上报，满足契约）。
        self.nixl_wrapper.release_xfer_handle(handle)
        self._failed_recv_reqs.add(req_id)

    def get_block_ids_with_load_errors(self):
        # SOURCE: vllm/.../nixl/worker.py:get_block_ids_with_load_errors
        return set()


class NixlConnector(KVConnectorBase_V1):
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L87
    def __init__(self, vllm_config, role, kv_cache_config=None, *, engine_id="engine",
                 connector_worker=None):
        # SOURCE: vllm/.../nixl/connector.py:L87-L108
        super().__init__(vllm_config, role, kv_cache_config)
        self.engine_id = engine_id
        # facade：按 role 只建半边子对象。
        if role == KVConnectorRole.SCHEDULER:
            # SUBTRACTED: NixlConnectorScheduler 子对象（ch29 决策侧），本章只看 worker 半边。
            self.connector_scheduler = object()
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = connector_worker

    # ==============================
    # Worker Side Methods（全转发给 connector_worker，assert 保证决策侧进不来）
    # ==============================

    def start_load_kv(self, forward_context, **kwargs) -> None:
        # SOURCE: vllm/.../nixl/connector.py:L241-L244
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, NixlConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        # SOURCE: vllm/.../nixl/connector.py:L246-L248
        """NixlConnector does not do layerwise saving."""
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
        # SOURCE: vllm/.../nixl/connector.py:L250-L258
        """NixlConnector does not save explicitly."""
        pass

    def wait_for_save(self):
        # SOURCE: vllm/.../nixl/connector.py:L260-L264
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, NixlConnectorMetadata)
        if self.connector_worker.use_host_buffer and self.connector_worker.copy_blocks:
            # SUBTRACTED: save_kv_to_host host buffer 回拷（worker），host buffer 路径
            #             整组删去（dossier delete 项）。对称 GPU↔GPU READ 下此处不触发。
            pass

    def get_finished(self, finished_req_ids):
        # SOURCE: vllm/.../nixl/connector.py:L204-L207
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def get_block_ids_with_load_errors(self):
        # SOURCE: vllm/.../nixl/connector.py:L209-L212
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    # SUBTRACTED: set_xfer_handshake_metadata / set_host_xfer_buffer_ops /
    #             register_cross_layers_kv_cache / get_required_kvcache_layout 等扩展点转发
    #             （connector.py，dossier delete 项）—— facade 结构由上面五个 worker 契约
    #             方法的转发已充分体现。
