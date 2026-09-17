# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ch38 worker 半边：任意布局→canonical 规范化 / DMA 提交 / flush 围栏。
#
# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:L1-L397
# SUBTRACTED（按减法计划）：
#   · 删除项 5（cross-layer/packed 布局族）：register_cross_layers_kv_cache
#     整方法、register_kv_caches 的 packed 分支与 UniformTypeKVCacheSpecs
#     逐层分支——保留通用逐层路径一条（canonical 语义等价：tensors 多条
#     vs 一条）。
#   · 删除项 1（canonical_mapping）：_is_store_writer 的 replicated 判定化简
#     为恒 True 快路（字段定义保留——非 writer ack 分支在，默认路径不触）。
from collections import defaultdict
from dataclasses import replace

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.canonical_mapping import (
    derive_canonical_mappings,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    ReqId,
)
# SUBTRACTED: is_kv_cache_tensor_packed 导入——packed 分支删除项 5。
from vllm.logger import init_logger
# SUBTRACTED: AttentionBackend 导入——register_cross_layers 删除项 5。
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    MambaSpec,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingSpec,
    OffloadingWorker,
)

logger = init_logger(__name__)


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:L41-L396
class OffloadingConnectorWorker:
    """Implementation of Worker side methods"""

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:L44-L64
    def __init__(
        self,
        spec: OffloadingSpec,
        vllm_config: "VllmConfig",
        kv_cache_config: KVCacheConfig,
    ):
        self.spec = spec
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.worker: OffloadingWorker | None = None
        # Non-writers still ack: pending_count waits for world_size per job.
        # SUBTRACTED: replicated_layout 的 rank0 单写者判定（删除项 1）——
        #   复制页布局删除后每 rank 都是 writer，恒走 True 快路（字段位保留）。
        self._is_store_writer = True

        # job_id -> req_id for in-flight loads.
        self._load_jobs: dict[int, ReqId] = {}
        self._unsubmitted_store_jobs: list[
            tuple[int, GPULoadStoreSpec, LoadStoreSpec]
        ] = []
        self._connector_worker_meta = OffloadingWorkerMetadata()

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:L66-L67
    def _init_worker(self, kv_caches: CanonicalKVCaches) -> None:
        self.worker = self.spec.get_worker(kv_caches)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:L69-L243
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        kv_cache_config = self.kv_cache_config
        num_blocks = kv_cache_config.num_blocks
        mappings = derive_canonical_mappings(
            self.vllm_config, kv_cache_config, kv_caches
        )

        # SUBTRACTED: packed 布局的 layer_is_packed 表（L76-L84）——删除项 5；
        #   通用逐层路径的块步长恒 = page_size_bytes。

        # layer_name -> (num_blocks, page_size_bytes) tensor
        tensors_per_block: dict[str, tuple[torch.Tensor, ...]] = {}
        # layer_name -> size of (un-padded) page in bytes
        unpadded_page_size_bytes: dict[str, int] = {}
        # layer_name -> size of page in bytes
        page_size_bytes: dict[str, int] = {}
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            group_layer_names = kv_cache_group.layer_names
            group_kv_cache_spec = kv_cache_group.kv_cache_spec
            # SUBTRACTED: UniformTypeKVCacheSpecs 的逐层规格展开（L95-L98）
            #   ——删除项 5；通用逐层路径按组级规格走。
            per_layer_specs = {}
            for layer_name in group_layer_names:
                layer_kv_cache_spec = per_layer_specs.get(
                    layer_name, group_kv_cache_spec
                )
                if isinstance(layer_kv_cache_spec, AttentionSpec):
                    layer_kv_cache = kv_caches[layer_name]
                    assert isinstance(layer_kv_cache, torch.Tensor)

                    page = layer_kv_cache_spec.page_size_bytes
                    elem_size = layer_kv_cache.element_size()
                    byte_offset = layer_kv_cache.storage_offset() * elem_size
                    # SUBTRACTED: packed 的 stride(0) 块步长分支——删除项 5。
                    block_stride_bytes = page
                    raw = torch.empty(
                        0,
                        dtype=torch.int8,
                        device=layer_kv_cache.device,
                    ).set_(layer_kv_cache.untyped_storage())
                    tensors_per_block[layer_name] = (
                        torch.as_strided(
                            raw,
                            (num_blocks, page),
                            (block_stride_bytes, 1),
                            byte_offset,
                        ),
                    )
                    page_size_bytes[layer_name] = layer_kv_cache_spec.page_size_bytes
                    unpadded_page_size_bytes[layer_name] = (
                        layer_kv_cache_spec.unpadded_page_size_bytes
                    )

                elif isinstance(layer_kv_cache_spec, MambaSpec):
                    layer_kv_cache = kv_caches[layer_name]
                    assert layer_kv_cache.dtype == torch.int8
                    tensors_per_block[layer_name] = (
                        layer_kv_cache.view(
                            num_blocks, layer_kv_cache_spec.page_size_bytes
                        ),
                    )

                    page_size_bytes[layer_name] = layer_kv_cache_spec.page_size_bytes
                    unpadded_page_size_bytes[layer_name] = replace(
                        layer_kv_cache_spec, page_size_padded=None
                    ).page_size_bytes

                else:
                    raise NotImplementedError

        # SUBTRACTED: packed 单张量装配分支（L150-L175）——删除项 5
        #   （DSv4 packed 布局是传输加速变体，通用逐层路径语义等价）。

        block_tensors: list[CanonicalKVCacheTensor] = []
        block_data_refs: dict[str, list[CanonicalKVCacheRef]] = defaultdict(list)
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            # Filter to layers that were actually processed above.
            # Packed KV allocation emits KVCacheTensor entries for
            # every (tuple_idx, page_size) slot; slots where no group has a
            # layer at that index produce an empty shared_by (reserved memory
            # with no corresponding model layer).
            tensor_layer_names = [
                n for n in kv_cache_tensor.shared_by if n in tensors_per_block
            ]
            if not tensor_layer_names:
                continue

            # verify all layers in the group reference the exact same tensors
            assert len({len(tensors_per_block[n]) for n in tensor_layer_names}) == 1
            assert (
                len({tensors_per_block[n][0].data_ptr() for n in tensor_layer_names})
                == 1
            )
            assert (
                len({tensors_per_block[n][0].stride() for n in tensor_layer_names}) == 1
            )

            # pick the first layer to represent the group
            first_layer_name = tensor_layer_names[0]
            for tensor in tensors_per_block[first_layer_name]:
                block_tensors.append(
                    CanonicalKVCacheTensor(
                        tensor=tensor,
                        page_size_bytes=page_size_bytes[first_layer_name],
                    )
                )

                curr_tensor_idx = len(block_tensors) - 1
                for layer_name in tensor_layer_names:
                    mapping = (
                        mappings.get(layer_name)
                        if len(tensors_per_block[first_layer_name]) == 1
                        else None
                    )
                    assert (
                        mapping is None
                        or mapping.local_page_size_bytes
                        == unpadded_page_size_bytes[layer_name]
                    )
                    block_data_refs[layer_name].append(
                        CanonicalKVCacheRef(
                            tensor_idx=curr_tensor_idx,
                            page_size_bytes=(unpadded_page_size_bytes[layer_name]),
                            mapping=mapping,
                        )
                    )

        group_data_refs: list[list[CanonicalKVCacheRef]] = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            group_refs: list[CanonicalKVCacheRef] = []
            for layer_name in kv_cache_group.layer_names:
                group_refs += block_data_refs[layer_name]
            group_data_refs.append(group_refs)

        canonical_kv_caches = CanonicalKVCaches(
            tensors=block_tensors,
            group_data_refs=group_data_refs,
        )

        self._init_worker(canonical_kv_caches)

    # SUBTRACTED: register_cross_layers_kv_cache（L245-L290）——删除项 5
    #   （cross-layer 单张量布局；prefer_cross_layer_blocks 旗标仍保留在 facade）。


    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:L292-L317
    def handle_preemptions(self, kv_connector_metadata: OffloadingConnectorMetadata):
        assert self.worker is not None

        # Pop jobs_to_flush from store_jobs into _unsubmitted_store_jobs
        # so the existing submission loop below submits them before wait().
        if kv_connector_metadata.jobs_to_flush:
            for job_id in kv_connector_metadata.jobs_to_flush:
                entry = kv_connector_metadata.store_jobs.pop(job_id, None)
                if entry is not None:
                    if not self._is_store_writer:
                        self._connector_worker_meta.mark_completed(job_id)
                        continue
                    assert isinstance(entry.src_spec, GPULoadStoreSpec)
                    self._unsubmitted_store_jobs.append(
                        (job_id, entry.src_spec, entry.dst_spec)
                    )

        # Submit deferred stores from previous step (and jobs_to_flush above).
        for job_id, src_spec, dst_spec in self._unsubmitted_store_jobs:
            assert isinstance(src_spec, GPULoadStoreSpec)
            success = self.worker.submit_store(job_id, src_spec, dst_spec)
            assert success
        self._unsubmitted_store_jobs.clear()

        if kv_connector_metadata.jobs_to_flush:
            self.worker.wait(kv_connector_metadata.jobs_to_flush)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:L319-L330
    def start_kv_transfers(self, metadata: OffloadingConnectorMetadata):
        assert self.worker is not None
        for job_id, src_spec, dst_spec in self._unsubmitted_store_jobs:
            success = self.worker.submit_store(job_id, src_spec, dst_spec)
            assert success
        self._unsubmitted_store_jobs.clear()

        for job_id, entry in metadata.load_jobs.items():
            self._load_jobs[job_id] = entry.req_id
            assert isinstance(entry.dst_spec, GPULoadStoreSpec)
            success = self.worker.submit_load(job_id, entry.src_spec, entry.dst_spec)
            assert success

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:L332-L344
    def prepare_store_kv(self, metadata: OffloadingConnectorMetadata):
        for job_id, entry in metadata.store_jobs.items():
            if not self._is_store_writer:
                # Gate before queueing: no _unsubmitted_store_jobs entry.
                self._connector_worker_meta.mark_completed(job_id)
                continue
            # NOTE(orozery): defer the store to the beginning of the next
            # engine step, so that offloading starts AFTER transfers related
            # to token sampling, thereby avoiding delays to token generation.
            assert isinstance(entry.src_spec, GPULoadStoreSpec)
            self._unsubmitted_store_jobs.append(
                (job_id, entry.src_spec, entry.dst_spec)
            )

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:L346-L381
    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """
        Returns:
            tuple of (finished_sending, finished_recving). Stores never
            emit finished_sending — the scheduler tracks store completion
            via kv_connector_worker_meta.completed_jobs and fences any
            block reuse via jobs_to_flush. Loads still emit
            finished_recving so the base scheduler can resume requests
            blocked on remote KV (and free aborted-during-load reqs).
        """
        assert self.worker is not None
        finished_recving: set[str] = set()
        for transfer_result in self.worker.get_finished():
            # we currently do not support job failures
            job_id = transfer_result.job_id
            assert transfer_result.success
            is_load = job_id in self._load_jobs
            if (
                transfer_result.transfer_time is not None
                and transfer_result.transfer_size is not None
            ):
                if is_load:
                    stats = self._connector_worker_meta.transfer_stats.load
                else:
                    stats = self._connector_worker_meta.transfer_stats.store
                stats.record(
                    transfer_result.transfer_size,
                    transfer_result.transfer_time,
                )

            self._connector_worker_meta.mark_completed(job_id)
            req_id = self._load_jobs.pop(job_id, None)
            if req_id is not None:
                finished_recving.add(req_id)

        return set(), finished_recving

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:L383-L389
    def build_connector_worker_meta(self) -> OffloadingWorkerMetadata | None:
        """Return completed transfer job IDs since the last call."""
        if not self._connector_worker_meta.completed_jobs:
            return None
        meta = self._connector_worker_meta
        self._connector_worker_meta = OffloadingWorkerMetadata()
        return meta

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:L391-L396
    def shutdown(self) -> None:
        self._unsubmitted_store_jobs.clear()
        self._load_jobs.clear()
        self._connector_worker_meta = OffloadingWorkerMetadata()
        if self.worker is not None:
            self.worker.shutdown()
