# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""过线三件：TransferJob / OffloadingConnectorMetadata / OffloadingWorkerMetadata。"""

from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)
from vllm.v1.kv_offload.base import LoadStoreSpec

ReqId = str


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py:L14-L35
@dataclass(slots=True)
class DirectionalTransferStats:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py:L14-L18
    bytes: int = 0
    time: float = 0.0
    sizes: list[int | float] = field(default_factory=list)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py:L20-L27
    def aggregate(
        self, other: "DirectionalTransferStats"
    ) -> "DirectionalTransferStats":
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py:L20-L27
        return DirectionalTransferStats(
            bytes=self.bytes + other.bytes,
            time=self.time + other.time,
            sizes=[*self.sizes, *other.sizes],
        )

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py:L29-L32
    def record(self, num_bytes: int, time: float) -> None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py:L29-L32
        self.bytes += num_bytes
        self.time += time
        self.sizes.append(num_bytes)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py:L34-L35
    def is_empty(self) -> bool:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py:L34-L35
        return self.bytes == 0 and self.time == 0.0 and not self.sizes


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py:L38-L50
@dataclass(slots=True)
class TransferStats:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py:L38-L41
    load: DirectionalTransferStats = field(default_factory=DirectionalTransferStats)
    store: DirectionalTransferStats = field(default_factory=DirectionalTransferStats)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py:L43-L47
    def aggregate(self, other: "TransferStats") -> "TransferStats":
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py:L43-L47
        return TransferStats(
            load=self.load.aggregate(other.load),
            store=self.store.aggregate(other.store),
        )

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py:L49-L50
    def is_empty(self) -> bool:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py:L49-L50
        return self.load.is_empty() and self.store.is_empty()


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py:L53-L64
@dataclass
class TransferJob:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py:L53-L60
    """A transfer job bundling request context with transfer spec.

    Used for both loads and stores, keyed by scheduler-assigned job ID.
    The worker reports the job ID back when the transfer finishes,
    and the scheduler processes the completion.
    """

    req_id: ReqId
    src_spec: LoadStoreSpec
    dst_spec: LoadStoreSpec


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py:L67-L72
@dataclass
class OffloadingConnectorMetadata(KVConnectorMetadata):
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py:L67-L72
    # Keyed by scheduler-assigned job IDs.
    load_jobs: dict[int, TransferJob]
    store_jobs: dict[int, TransferJob]
    jobs_to_flush: set[int] | None = None


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py:L75-L104
@dataclass
class OffloadingWorkerMetadata(KVConnectorWorkerMetadata):
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py:L75-L83
    """Worker -> Scheduler metadata for completed transfer jobs.

    Each worker reports {job_id: 1} for newly completed transfer jobs
    (load or store). aggregate() sums counts across workers within a step.
    The scheduler accumulates across steps and processes
    a transfer completion only when count reaches num_workers.
    """

    completed_jobs: dict[int, int] = field(default_factory=dict)
    transfer_stats: TransferStats = field(default_factory=TransferStats)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py:L88-L90
    def mark_completed(self, job_id: int) -> None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py:L88-L90
        """Record a transfer job completion from this worker."""
        self.completed_jobs[job_id] = 1

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py:L92-L104
    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "KVConnectorWorkerMetadata":
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py:L92-L104
        assert isinstance(other, OffloadingWorkerMetadata)

        merged = dict(self.completed_jobs)
        for job_id, v in other.completed_jobs.items():
            merged[job_id] = merged.get(job_id, 0) + v

        return OffloadingWorkerMetadata(
            completed_jobs=merged,
            transfer_stats=self.transfer_stats.aggregate(other.transfer_stats),
        )
