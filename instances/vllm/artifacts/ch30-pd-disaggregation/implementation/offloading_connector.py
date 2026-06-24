# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Offloading connector — CPU/磁盘卸载后端，同样 facade，但语义有别。

SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py
        + vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py
不是 P/D 跨节点搬，而是把 GPU KV 卸到 CPU/磁盘当二级缓存。关键语义差异：
wait_for_save→prepare_store_kv 只把 store job 入队（不真等完成），推迟到下一步
start_kv_transfers 开头才 transfer_async；get_finished 只为 load 报 finished_recving，
store 完成走 build_connector_worker_meta 的 completed_jobs + 调度侧 jobs_to_flush 围栏。
"""

from .base import KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:TransferResult
class TransferResult:
    def __init__(self, job_id, success=True):
        # SOURCE: vllm/.../offloading/worker.py:TransferResult
        self.job_id = job_id
        self.success = success
        # SUBTRACTED: transfer_time/transfer_size/transfer_type（stats 字段），可观测性。
        self.transfer_time = None
        self.transfer_size = None
        self.transfer_type = None


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:OffloadingWorker
class OffloadingWorker:
    # SUBTRACTED: register_handler / 真实 GPU↔CPU·磁盘 transfer handler（worker），精简版
    #             loopback：transfer_async 立刻把 job 记为可完成，get_finished 一把返回。
    #             忠实保留『提交即异步、靠后续 get_finished 收割』的接口形状。
    def __init__(self):
        # SOURCE: vllm/.../offloading/worker.py:OffloadingWorker.__init__
        self._inflight = []

    def transfer_async(self, job_id, transfer_spec) -> bool:
        # SOURCE: vllm/.../offloading/worker.py:OffloadingWorker.transfer_async
        self._inflight.append(TransferResult(job_id, success=True))
        return True

    def get_finished(self):
        # SOURCE: vllm/.../offloading/worker.py:OffloadingWorker.get_finished
        done = self._inflight
        self._inflight = []
        return done


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:OffloadingWorkerMetadata
class OffloadingWorkerMetadata:
    def __init__(self):
        # SOURCE: vllm/.../offloading/worker.py:OffloadingWorkerMetadata.__init__
        self.completed_jobs = set()

    def mark_completed(self, job_id):
        # SOURCE: vllm/.../offloading/worker.py:OffloadingWorkerMetadata.mark_completed
        self.completed_jobs.add(job_id)


# SOURCE: vllm/.../offloading_connector.py:LoadStoreSpec entry（load_jobs/store_jobs 项）
class _JobEntry:
    def __init__(self, transfer_spec, req_id=None):
        # SOURCE: vllm/.../offloading_connector.py:LoadStoreSpec entry（load_jobs/store_jobs 项）
        self.transfer_spec = transfer_spec
        self.req_id = req_id


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:OffloadingConnectorMetadata
class OffloadingConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        # SOURCE: vllm/.../offloading_connector.py:OffloadingConnectorMetadata.__init__
        self.load_jobs = {}    # job_id -> _JobEntry(req_id 非空)
        self.store_jobs = {}   # job_id -> _JobEntry
        self.jobs_to_flush = set()


class OffloadingConnectorWorker:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:L42-L53
    def __init__(self, spec=None):
        self.spec = spec
        self.worker = OffloadingWorker()
        # SUBTRACTED: kv_connector_stats（worker.py L49），可观测性。
        # job_id -> req_id for in-flight loads.
        self._load_jobs = {}
        self._unsubmitted_store_jobs = []
        self._connector_worker_meta = OffloadingWorkerMetadata()

    def start_kv_transfers(self, metadata: OffloadingConnectorMetadata):
        # SOURCE: vllm/.../offloading/worker.py:L295-L304
        for job_id, transfer_spec in self._unsubmitted_store_jobs:
            success = self.worker.transfer_async(job_id, transfer_spec)
            assert success
        self._unsubmitted_store_jobs.clear()

        for job_id, entry in metadata.load_jobs.items():
            self._load_jobs[job_id] = entry.req_id
            success = self.worker.transfer_async(job_id, entry.transfer_spec)
            assert success

    def prepare_store_kv(self, metadata: OffloadingConnectorMetadata):
        # SOURCE: vllm/.../offloading/worker.py:L306-L311
        for job_id, entry in metadata.store_jobs.items():
            # NOTE(orozery): defer the store to the beginning of the next
            # engine step, so that offloading starts AFTER transfers related
            # to token sampling, thereby avoiding delays to token generation.
            self._unsubmitted_store_jobs.append((job_id, entry.transfer_spec))

    def get_finished(self, finished_req_ids):
        # SOURCE: vllm/.../offloading/worker.py:L313-L344
        # Stores never emit finished_sending — the scheduler tracks store
        # completion via kv_connector_worker_meta.completed_jobs and fences any
        # block reuse via jobs_to_flush. Loads still emit finished_recving.
        finished_recving = set()
        for transfer_result in self.worker.get_finished():
            # we currently do not support job failures
            job_id = transfer_result.job_id
            assert transfer_result.success
            # SUBTRACTED: kv_connector_stats.record_transfer（worker L328-L337），可观测性。
            self._connector_worker_meta.mark_completed(job_id)
            req_id = self._load_jobs.pop(job_id, None)
            if req_id is not None:
                finished_recving.add(req_id)

        return set(), finished_recving

    def build_connector_worker_meta(self):
        # SOURCE: vllm/.../offloading/worker.py:L346-L352
        """Return completed transfer job IDs since the last call."""
        if not self._connector_worker_meta.completed_jobs:
            return None
        meta = self._connector_worker_meta
        self._connector_worker_meta = OffloadingWorkerMetadata()
        return meta


class OffloadingConnector(KVConnectorBase_V1):
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L46
    def __init__(self, vllm_config, role, kv_cache_config=None, *, spec=None):
        # SOURCE: vllm/.../offloading_connector.py:L51-L67
        super().__init__(vllm_config, role, kv_cache_config)
        # SUBTRACTED: OffloadingSpecFactory.create_spec（offloading_connector.py L60）—
        #             由配置造卸载策略（CPU/磁盘后端选择），精简版按需注入 spec。
        self.connector_scheduler = None
        self.connector_worker = None
        if role == KVConnectorRole.SCHEDULER:
            # SUBTRACTED: OffloadingConnectorScheduler 子对象（ch29 决策侧）。
            self.connector_scheduler = object()
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = OffloadingConnectorWorker(spec)

    # ==============================
    # Worker Side Methods（facade 转发）
    # ==============================

    def start_load_kv(self, forward_context, **kwargs) -> None:
        # SOURCE: vllm/.../offloading_connector.py:L90-L93
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, OffloadingConnectorMetadata)
        self.connector_worker.start_kv_transfers(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        # SOURCE: vllm/.../offloading_connector.py:L95-L96
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
        # SOURCE: vllm/.../offloading_connector.py:L98-L105
        pass

    def wait_for_save(self):
        # SOURCE: vllm/.../offloading_connector.py:L107-L110
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, OffloadingConnectorMetadata)
        self.connector_worker.prepare_store_kv(self._connector_metadata)

    def get_finished(self, finished_req_ids):
        # SOURCE: vllm/.../offloading_connector.py:L112-L114
        assert self.connector_worker is not None
        return self.connector_worker.get_finished(finished_req_ids)

    def build_connector_worker_meta(self):
        # SOURCE: vllm/.../offloading_connector.py:L116-L119
        if self.connector_worker is not None:
            return self.connector_worker.build_connector_worker_meta()
        return None

    # SUBTRACTED: handle_preemptions / register_kv_caches / take_events 等转发方法
    #             （offloading_connector.py，dossier delete 项）—— facade 结构由上面五个
    #             worker 契约方法的转发已充分体现。
