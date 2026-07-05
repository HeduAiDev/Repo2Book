# vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py
#   —— subtract-only companion（★ 入口：把池化接进 vLLM 引擎）
#
# AscendStoreConnector 子类化 vLLM 的 KVConnectorBase_V1(+SupportsHMA)，被引擎主循环通过标准钩子
#   驱动。它本身是「薄分发层」——几乎没有逻辑：按 role 把调度侧钩子转发给 KVPoolScheduler、worker 侧
#   钩子转发给 KVPoolWorker。rank0 的 worker 进程额外起一个 LookupKeyServer（zmq REP 服务端）：
#   调度进程没有 KV / 没有后端连接，必须经它跨进程问 worker「命中多少前缀」。
#
# host 无 vllm/NPU：基类与 zmq RPC 经 runtime_stub 接住；role 分派 / 转发 / 跨进程 lookup 协作
#   是纯 Python，可跑。
import threading

import zmq

from pool_scheduler import KVPoolScheduler, get_zmq_rpc_path_lookup
from pool_worker import KVPoolWorker
from runtime_stub import (
    KVConnectorBase_V1,
    KVConnectorRole,
    MsgpackDecoder,
    SupportsHMA,
    logger,
    make_zmq_socket,
)

# SUBTRACTED: import torch（仅 kv_caches: dict[str, torch.Tensor] 类型标注，已降级为无标注）；
#   vllm kv_events(KVEventAggregator 等) / ForwardContext / AttentionMetadata / BlockPool /
#   KVCacheBlocks / SchedulerOutput / KVConnectorOutput / Request / AscendStoreKVConnectorWorkerMetadata
#   （L1-L36）—— 仅服务被删的 kv-event / layerwise / mamba-finish 路径。原 ascend_store_connector.py:L1-L36

# SUBTRACTED: class AscendStoreKVEvents(KVConnectorKVEvents)（L39-L70）—— kv-event 聚合旁路遥测，
#   不参与 load/save 正确性。原 ascend_store_connector.py:L39-L70


class AscendStoreConnector(KVConnectorBase_V1, SupportsHMA):  # SOURCE: ascend_store_connector.py:L73
    # SUBTRACTED: requires_piecewise_for_cudagraph（L74-L80）—— layerwise 时要求 PIECEWISE cudagraph 模式；
    #   非 layerwise 不涉及。原 ascend_store_connector.py:L74-L80

    def __init__(self, vllm_config, role, kv_cache_config=None):  # SOURCE: ascend_store_connector.py:L82
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self.kv_role = vllm_config.kv_transfer_config.kv_role

        self.use_layerwise = vllm_config.kv_transfer_config.kv_connector_extra_config.get("use_layerwise", False)
        self.consumer_is_to_put = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "consumer_is_to_put", False
        )
        # SUBTRACTED: connector_name == "MooncakeConnectorStoreV1" 的弃用告警（L91-L96）—— 旧连接器迁移提示。
        #   原 ascend_store_connector.py:L91-L96

        self.kv_caches: dict = {}
        # SUBTRACTED: self._kv_cache_events / sended_but_unfinished_reqs（L99-L101）—— kv-event 聚合状态。

        # ★ role 二分派：SCHEDULER 进程起 KVPoolScheduler（「搬什么」端）；其余（WORKER）起 KVPoolWorker
        #   （「异步搬」端），且 rank0 额外起 LookupKeyServer（lookup RPC 服务端）。
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = KVPoolScheduler(vllm_config, self.use_layerwise, kv_cache_config)
        else:
            self.connector_worker = KVPoolWorker(
                vllm_config,
                self.use_layerwise,
                kv_cache_config,
            )

            assert self.connector_worker is not None
            if vllm_config.parallel_config.rank == 0:
                self.lookup_server = LookupKeyServer(self.connector_worker, vllm_config, self.use_layerwise)

    ############################################################
    # Scheduler Side Methods —— 转发给 connector_scheduler
    ############################################################

    def get_num_new_matched_tokens(self, request, num_computed_tokens: int) -> tuple[int, bool]:  # SOURCE: ascend_store_connector.py:L120
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request, blocks, num_external_tokens: int):  # SOURCE: ascend_store_connector.py:L124
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output):  # SOURCE: ascend_store_connector.py:L128
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    # SUBTRACTED: request_finished / request_finished_all_groups（L135-L149）—— 转发给 scheduler 的
    #   mamba 块异步释放 / HMA 多 group finish 回执；本主线 scheduler 不实现这些（hybrid 路径已删）。
    # SUBTRACTED: update_connector_output / take_events（L151-L185）—— kv-event 聚合旁路。
    #   原 ascend_store_connector.py:L135-L185

    ############################################################
    # Worker Side Methods —— 转发给 connector_worker
    ############################################################
    def register_kv_caches(self, kv_caches):  # SOURCE: ascend_store_connector.py:L190
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context, **kwargs) -> None:  # SOURCE: ascend_store_connector.py:L194
        assert self.connector_worker is not None
        metadata = self._get_connector_metadata()
        # SUBTRACTED: 大段 logger.debug（逐请求 load_spec 摘要，L197-L209）—— 调试日志，不参与控制流。
        self.connector_worker.start_load_kv(metadata)

    # SUBTRACTED: wait_for_layer_load / save_kv_layer（L212-L226）—— layerwise 逐层取/存钩子；
    #   非 layerwise 不涉及。原 ascend_store_connector.py:L212-L226

    def wait_for_save(self):  # SOURCE: ascend_store_connector.py:L228
        if self.kv_role == "kv_consumer" and not self.consumer_is_to_put:
            # Don't do save if the role is kv_consumer
            return

        if self.use_layerwise:
            return

        self.connector_worker.wait_for_save(self._get_connector_metadata())

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:  # SOURCE: ascend_store_connector.py:L238
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        done_sending, done_recving = self.connector_worker.get_finished(
            finished_req_ids, self._get_connector_metadata()
        )
        return done_sending, done_recving

    def get_block_ids_with_load_errors(self) -> set[int]:  # SOURCE: ascend_store_connector.py:L246
        """Return KV block IDs that failed to load on the worker."""
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    # SUBTRACTED: get_kv_connector_kv_cache_events / build_connector_worker_meta（L251-L269）——
    #   kv-event 收集 + mamba 块释放回执转发。原 ascend_store_connector.py:L251-L269

    def bind_gpu_block_pool(self, gpu_block_pool) -> None:  # SOURCE: ascend_store_connector.py:L263
        assert self.connector_scheduler is not None
        self.connector_scheduler.bind_gpu_block_pool(gpu_block_pool)


class LookupKeyServer:  # SOURCE: ascend_store_connector.py:L272
    def __init__(  # SOURCE: ascend_store_connector.py:L273
        self,
        pool_worker: KVPoolWorker,
        vllm_config,
        use_layerwise: bool,
    ):
        self.decoder = MsgpackDecoder()
        # SUBTRACTED: self.decoder_tensor = MsgpackDecoder(torch.Tensor)（L280）—— 仅 layerwise 解 tensor 帧用。
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lookup(vllm_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REP,  # type: ignore[attr-defined]
            bind=True,
        )

        self.pool_worker = pool_worker
        self.running = True
        self.use_layerwise = use_layerwise

        def process_request():  # SOURCE: ascend_store_connector.py:L294
            while self.running:
                all_frames = self.socket.recv_multipart(copy=False)
                token_len = int.from_bytes(all_frames[0], byteorder="big")
                kv_group_ids = self.decoder.decode([all_frames[1]])
                hash_frames = all_frames[2:]
                hashes_str = self.decoder.decode(hash_frames)
                # ★ 服务端把请求交给 KVPoolWorker.lookup_scheduler（它真正持有 m_store，调 m_store.exists 查池）
                result = self.pool_worker.lookup_scheduler(
                    token_len,
                    hashes_str,
                    kv_group_ids,
                    self.use_layerwise,
                )
                response = result.to_bytes(4, "big")
                self.socket.send(response)

        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()

    def close(self):  # SOURCE: ascend_store_connector.py:L319
        self.socket.close(linger=0)
        # TODO: close the thread!
