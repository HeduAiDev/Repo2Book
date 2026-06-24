# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Point-to-point NCCL connector — 单类自包含范式（不拆 scheduler/worker 子对象）。

SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py
        + vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py
靠 is_producer 区分 P/D：producer 在 save_kv_layer 里 extract_kv_from_layer +
send_tensor 发，consumer 在 start_load_kv 里逐请求逐层 recv_tensor + inject_kv_into_layer
写回 paged buffer。底层 P2pNcclEngine 的 send_tensor 三模式与 wait_for_sent 体现
『异步发为何要 fence』。
"""

import re
import threading
from collections import deque

import numpy as np

from .base import KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:ReqMeta
class ReqMeta:
    # SUBTRACTED: make_meta 里 token_ids→block_ids 对齐、chunked-prefill 切分（连接器
    #             scheduler 侧 build_connector_meta 的产物细节），本章 worker 侧只需
    #             request_id + block_ids。
    def __init__(self, request_id: str, block_ids):
        # SOURCE: vllm/.../p2p/p2p_nccl_connector.py:ReqMeta.make_meta
        self.request_id = request_id
        self.block_ids = block_ids


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:P2pNcclConnectorMetadata
class P2pNcclConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        # SOURCE: vllm/.../p2p/p2p_nccl_connector.py:P2pNcclConnectorMetadata.__init__
        self.requests = []

    def add_request(self, request_id: str, block_ids) -> None:
        # SOURCE: vllm/.../p2p/p2p_nccl_connector.py:add_request
        self.requests.append(ReqMeta(request_id, block_ids))


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py:SendQueueItem
class SendQueueItem:
    def __init__(self, tensor_id, remote_address, tensor):
        # SOURCE: vllm/.../p2p/p2p_nccl_engine.py:SendQueueItem
        self.tensor_id = tensor_id
        self.remote_address = remote_address
        self.tensor = tensor


class P2pNcclEngine:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py:P2pNcclEngine
    # SUBTRACTED: pynccl 点对点 comm 建链、ZMQ listener/ping/握手线程、TensorMemoryPool
    #             缓冲管理（dossier delete 项：控制面握手是底层网络细节）。精简版用一个
    #             进程内 loopback：PUT_ASYNC 入 send_queue → 后台 _send_thread 把 tensor
    #             投递到（同进程的）目标 engine recv_store，忠实保留『同步发 vs 异步入队
    #             后台发』+ wait_for_sent 等队空的语义。
    def __init__(self, send_type: str = "PUT_ASYNC"):
        # SOURCE: vllm/.../p2p/p2p_nccl_engine.py:P2pNcclEngine.__init__
        self.send_type = send_type
        self.rank = 0
        # tensor_id "#" layer_name -> 已收到的 tensor（consumer 侧）
        self.recv_store = {}
        self.recv_store_cv = threading.Condition()
        self.send_queue = deque()
        self.send_queue_cv = threading.Condition()
        # 进程内 loopback：remote_address -> 对端 engine（用于把发出的 tensor 投递过去）
        self._peers = {}
        # 完成记账：已发出 / 已收齐的 (request_id) 集合
        self._sent_req_layers = {}
        self._recv_req_layers = {}
        self._thread = None
        if self.send_type == "PUT_ASYNC":
            self._thread = threading.Thread(target=self._send_thread, daemon=True)
            self._thread.start()

    def connect(self, remote_address: str, peer: "P2pNcclEngine") -> None:
        # SOURCE: vllm/.../p2p/p2p_nccl_engine.py:create_connect
        # SUBTRACTED: create_connect/ZMQ socket 建链（p2p_nccl_engine.py），loopback 直接
        #             记下对端 engine 引用。
        self._peers[remote_address] = peer

    def send_tensor(self, tensor_id, tensor, remote_address=None) -> bool:
        # SOURCE: vllm/.../p2p/p2p_nccl_engine.py:L235-L258
        if remote_address is None:
            with self.recv_store_cv:
                self.recv_store[tensor_id] = tensor
                self.recv_store_cv.notify()
            return True

        item = SendQueueItem(
            tensor_id=tensor_id, remote_address=remote_address, tensor=tensor
        )

        if self.send_type == "PUT":
            return self.send_sync(item)

        if self.send_type == "PUT_ASYNC":
            with self.send_queue_cv:
                self.send_queue.append(item)
                self.send_queue_cv.notify()
            return True

        # SUBTRACTED: GET 模式（暂存 send_store 等对端来拉 + buffer LRU 驱逐，engine
        #             L260-L306，dossier delete 项）—— 第三种传输策略，与 worker 契约
        #             如何被填实的主线无关。PUT/PUT_ASYNC 已足够展示同步/异步 + fence。
        return True

    def send_sync(self, item: SendQueueItem) -> bool:
        # SOURCE: vllm/.../p2p/p2p_nccl_engine.py:L500
        # SUBTRACTED: 真实经 pynccl comm.send 走 NCCL p2p（engine send_sync 主体）。
        #             loopback 直接投递到对端 recv_store。
        peer = self._peers[item.remote_address]
        with peer.recv_store_cv:
            peer.recv_store[item.tensor_id] = item.tensor
            peer.recv_store_cv.notify_all()
        # 完成记账：标记本 request 的该层已发出
        req_id = item.tensor_id.split("#", 1)[0]
        self._sent_req_layers.setdefault(req_id, 0)
        self._sent_req_layers[req_id] += 1
        return True

    def _send_thread(self):
        # SOURCE: vllm/.../p2p/p2p_nccl_engine.py:_send_thread
        while True:
            with self.send_queue_cv:
                while not self.send_queue:
                    self.send_queue_cv.wait()
                item = self.send_queue.popleft()
                self.send_queue_cv.notify_all()
            self.send_sync(item)

    def recv_tensor(self, tensor_id, remote_address=None):
        # SOURCE: vllm/.../p2p/p2p_nccl_engine.py:L308-L335
        if self.send_type == "PUT" or self.send_type == "PUT_ASYNC":
            with self.recv_store_cv:
                while tensor_id not in self.recv_store:
                    self.recv_store_cv.wait()
                tensor = self.recv_store.pop(tensor_id)
            req_id = tensor_id.split("#", 1)[0]
            self._recv_req_layers.setdefault(req_id, 0)
            self._recv_req_layers[req_id] += 1
            return tensor
        # SUBTRACTED: GET 模式 recv（建链 + ZMQ 请求对端拉，engine L337+）随 GET 一并删。
        return None

    def wait_for_sent(self):
        # SOURCE: vllm/.../p2p/p2p_nccl_engine.py:L486-L498
        if self.send_type == "PUT_ASYNC":
            with self.send_queue_cv:
                while self.send_queue:
                    self.send_queue_cv.wait()

    def get_finished(self, finished_req_ids, no_compile_layers):
        # SOURCE: vllm/.../p2p/p2p_nccl_engine.py:L540
        # SUBTRACTED: 真实按『已发/已收的层数 == 模型层数』判某 request 是否传完
        #             （engine get_finished 主体逐层记账）。精简版保留同一判据：本进程
        #             引擎对 send/recv 各自记账（对称、双向可见），凡 finished_req_ids 里
        #             记账层数达标的报完成。
        num_layers = max(1, len(no_compile_layers))
        done_sending = {
            r for r in finished_req_ids
            if self._sent_req_layers.get(r, 0) >= num_layers
        }
        done_recving = {
            r for r in finished_req_ids
            if self._recv_req_layers.get(r, 0) >= num_layers
        }
        return done_sending or None, done_recving or None


class P2pNcclConnector(KVConnectorBase_V1):
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:L74
    def __init__(self, vllm_config, role, kv_cache_config=None, *, is_producer,
                 engine=None):
        # SOURCE: vllm/.../p2p/p2p_nccl_connector.py:L75-L105
        super().__init__(vllm_config, role, kv_cache_config)
        self.is_producer = is_producer
        self._rank = 0
        # SUBTRACTED: _block_size / _requests_need_load / chunked_prefill / get_world_group
        #             rank 推导（connector __init__）—— scheduler 侧状态与对称 TP rank 偏移，
        #             worker 收发主线不依赖。
        self.p2p_nccl_engine = (
            engine if role == KVConnectorRole.WORKER else None
        )

    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_kv(self, forward_context, **kwargs) -> None:
        # SOURCE: vllm/.../p2p/p2p_nccl_connector.py:L111-L229
        """Start loading the KV cache from the connector buffer to vLLM's
        paged KV buffer."""
        # Only consumer/decode loads KV Cache
        if self.is_producer:
            return

        assert self.p2p_nccl_engine is not None

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return

        def inject_kv_into_layer(layer, kv_cache, block_ids, request_id) -> None:
            # SOURCE: vllm/.../p2p/p2p_nccl_connector.py:L134-L193
            # SUBTRACTED: MLA / FlashInfer layout 分支（connector L163-L178，dossier
            #             delete 项二选一）。保留 FlashAttention 一支（layer.shape[0]==2，
            #             沿第二维按 block_ids 写回）。block_ids 不匹配的告警分支并入。
            if layer.shape[0] == 2:  # FlashAttention
                num_block = kv_cache.shape[1]
                if len(block_ids) == num_block:
                    layer[:, block_ids, ...] = kv_cache
                else:
                    layer[:, block_ids[:num_block], ...] = kv_cache

        # Get the metadata
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, P2pNcclConnectorMetadata)

        # Load the KV for each request each layer
        for request in metadata.requests:
            request_id = request.request_id
            ip, port = self.parse_request_id(request_id, False)
            remote_address = ip + ":" + str(port + self._rank)
            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]

                # Only process layers that have kv_cache attribute (attention
                # layers) Skip non-attention layers like FusedMoE
                kv_cache = getattr(layer, "kv_cache", None)
                if kv_cache is None:
                    continue

                layer = kv_cache

                kv_cache = self.p2p_nccl_engine.recv_tensor(
                    request.request_id + "#" + layer_name, remote_address
                )

                if kv_cache is None:
                    continue

                inject_kv_into_layer(
                    layer, kv_cache, request.block_ids, request.request_id
                )

    def wait_for_layer_load(self, layer_name: str) -> None:
        # SOURCE: vllm/.../p2p/p2p_nccl_connector.py:L231-L240
        # P2P 把 load 整体在 start_load_kv 里做完，逐层等待是 no-op。
        return

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
        # SOURCE: vllm/.../p2p/p2p_nccl_connector.py:L242-L307
        # Only producer/prefill saves KV Cache
        if not self.is_producer:
            return

        assert self.p2p_nccl_engine is not None

        def extract_kv_from_layer(layer, block_ids):
            # SOURCE: vllm/.../p2p/p2p_nccl_connector.py:L266-L295
            # SUBTRACTED: MLA / FlashInfer layout 分支（connector L287-L290）。保留
            #             FlashAttention 一支（layer.shape[0]==2，沿第二维按 block_ids 切片）。
            if layer.shape[0] == 2:  # FlashAttention
                return layer[:, block_ids, ...]
            return None

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, P2pNcclConnectorMetadata)
        for request in connector_metadata.requests:
            request_id = request.request_id
            ip, port = self.parse_request_id(request_id, True)
            remote_address = ip + ":" + str(port + self._rank)

            kv_cache = extract_kv_from_layer(kv_layer, request.block_ids)
            self.p2p_nccl_engine.send_tensor(
                request_id + "#" + layer_name, kv_cache, remote_address
            )

    def wait_for_save(self):
        # SOURCE: vllm/.../p2p/p2p_nccl_connector.py:L309-L312
        if self.is_producer:
            assert self.p2p_nccl_engine is not None
            self.p2p_nccl_engine.wait_for_sent()

    def get_finished(self, finished_req_ids, **kwargs):
        # SOURCE: vllm/.../p2p/p2p_nccl_connector.py:L314-L331
        assert self.p2p_nccl_engine is not None
        no_compile_layers = self._vllm_config.compilation_config.static_forward_context
        return self.p2p_nccl_engine.get_finished(finished_req_ids, no_compile_layers)

    @staticmethod
    def parse_request_id(request_id: str, is_prefill=True):
        # SOURCE: vllm/.../p2p/p2p_nccl_connector.py:L503-L518
        # 对端地址直接编码在 request_id 里（『请求 id 自带对端地址』）。
        if is_prefill:
            pattern = r"___decode_addr_(.*):(\d+)"
        else:
            pattern = r"___prefill_addr_(.*):(\d+)___"
        match = re.search(pattern, request_id)
        if match:
            ip = match.group(1)
            port = int(match.group(2))
            return ip, port
        raise ValueError(f"Request id {request_id} does not contain hostname and port")
