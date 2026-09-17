# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ch38 生态对照④：MultiConnector 组合器（P/D+池化并存的官方形态）。
#
# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L1-L671
# SUBTRACTED: set_host_xfer_buffer_ops（xPU host buffer 旁路——ch37 面）；
#   其余逐字保留（首个命中数>0 者获加载权的判据 L399-L404 是 m14 锚点）。
import copy
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch

from vllm.config import VllmConfig
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBaseType
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    CopyBlocksOp,
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.distributed.kv_events import KVCacheEvent
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class MultiKVConnectorMetadata(KVConnectorMetadata):
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L46-L48
    metadata: tuple[KVConnectorMetadata, ...]
    extra_async_saves: dict[str, int] | None = None


@dataclass
class MultiKVConnectorWorkerMetadata(KVConnectorWorkerMetadata):
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L52-L68
    metadata: tuple[KVConnectorWorkerMetadata | None, ...]

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L55-L68
    def aggregate(self, other: KVConnectorWorkerMetadata) -> KVConnectorWorkerMetadata:
        assert isinstance(other, MultiKVConnectorWorkerMetadata)

        assert len(self.metadata) == len(other.metadata)
        metadata_list = []
        for metadata1, metadata2 in zip(self.metadata, other.metadata):
            if metadata1 is None:
                metadata_list.append(metadata2)
            elif metadata2 is None:
                metadata_list.append(metadata1)
            else:
                metadata_list.append(metadata1.aggregate(metadata2))

        return MultiKVConnectorWorkerMetadata(metadata=tuple(metadata_list))


@dataclass
class MultiKVConnectorStats(KVConnectorStats):
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L72-L104
    """
    Maintain a dict of KVConnectorStats objects, one for each connector.
    This is used to aggregate the stats from all connectors separately.
    """

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L78-L85
    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        for connector_id, stats in other.data.items():
            if connector_id not in self.data:
                self[connector_id] = stats
            else:
                assert isinstance(stats, type(self.data[connector_id]))
                self[connector_id] = self[connector_id].aggregate(stats)
        return self

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L87-L89
    def reset(self):
        for stats in self.data.values():
            stats.reset()

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L91-L95
    def reduce(self) -> dict[str, Any]:
        # TODO (NickLucche) Adjust for logging on separate lines
        return {
            connector_id: stats.reduce() for connector_id, stats in self.data.items()
        }

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L97-L98
    def is_empty(self) -> bool:
        return all(stats.is_empty() for stats in self.data.values())

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L100-L101
    def __getitem__(self, connector_id: str) -> KVConnectorStats:
        return self.data[connector_id]

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L103-L104
    def __setitem__(self, connector_id: str, stats: KVConnectorStats):
        self.data[connector_id] = stats


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L107-L125
class MultiKVConnectorPromMetrics(KVConnectorPromMetrics):
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L108-L117
    def __init__(
        self,
        vllm_config: "VllmConfig",
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
        prom_metrics: dict[str, KVConnectorPromMetrics],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        self._prom_metrics = prom_metrics

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L119-L125
    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        for connector_id, stats_data in transfer_stats_data.items():
            assert connector_id in self._prom_metrics, (
                f"{connector_id} is not contained in the list of registered connectors "
                f"with Prometheus metrics support: {self._prom_metrics.keys()}"
            )
            self._prom_metrics[connector_id].observe(stats_data["data"], engine_idx)


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L128-L671
class MultiConnector(KVConnectorBase_V1, SupportsHMA):
    """
    A wrapper for using multiple KVConnectors at the same time.

    The current logic is:
    - Load KV from the first connector that advertises available tokens from
      get_num_new_matched_tokens(), based on the order in the config.
    - Save to all connectors.
    """

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L139-L151
        """
        MultiConnector requires PIECEWISE CUDA graph mode if any of its
        child connectors require it.
        """
        connectors_config = extra_config.get("connectors", [])
        for conn_config in connectors_config:
            temp_ktc = KVTransferConfig(**conn_config)
            connector_cls = KVConnectorFactory.get_connector_class(temp_ktc)
            child_extra_config = conn_config.get("kv_connector_extra_config", {})
            if connector_cls.requires_piecewise_for_cudagraph(child_extra_config):
                return True
        return False

    @classmethod
    def all_children_support_hma(cls, kv_transfer_config: "KVTransferConfig") -> bool:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L154-L167
        """Return True only if every configured child connector supports HMA."""
        connectors_config = kv_transfer_config.kv_connector_extra_config.get(
            "connectors", []
        )
        if not connectors_config:
            return False
        for conn_config in connectors_config:
            child_config = KVTransferConfig(
                **{"engine_id": kv_transfer_config.engine_id, **conn_config}
            )
            if not KVConnectorFactory.supports_hma_config(child_config):
                return False
        return True

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L169-L204
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )

        self._connectors: list[KVConnectorBase_V1] = []
        self._ktc_kv_transfer_config = []
        for connector_cls, temp_config in self._get_connector_classes_and_configs(
            vllm_config
        ):
            self._connectors.append(connector_cls(temp_config, role, kv_cache_config))
            self._ktc_kv_transfer_config.append(temp_config.kv_transfer_config)

        assert vllm_config.kv_transfer_config is not None
        self._all_support_hma = MultiConnector.all_children_support_hma(
            vllm_config.kv_transfer_config
        )
        assert (
            vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            or self._all_support_hma
        ), "HMA should not be enabled unless all sub-connectors support it"

        # A mapping from request id to the index of the connector chosen to
        # load the request from (if any).
        self._requests_to_connector: dict[str, int] = {}

        # Keeps track of *additional* remaining async saves (beyond 1) to be
        # finished per request. Not needed for async loads since we only allow
        # a single connector to load.
        # Propagated from scheduler to worker side via the connector metadata.
        self._extra_async_saves: dict[str, int] = {}

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L207-L210
        if not self._connectors:
            return False
        return all(c.prefer_cross_layer_blocks for c in self._connectors)

    @property
    def requires_kv_delivery(self) -> bool:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L213-L214
        return any(c.requires_kv_delivery for c in self._connectors)

    @classmethod
    def _get_connector_classes_and_configs(
        cls, vllm_config: "VllmConfig"
    ) -> list[tuple[type[KVConnectorBaseType], "VllmConfig"]]:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L217-L240
        assert vllm_config.kv_transfer_config is not None
        ktcs = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "connectors"
        )
        assert ktcs is not None
        ret: list[tuple[type[KVConnectorBaseType], VllmConfig]] = []
        for ktc in ktcs:
            temp_config = copy.copy(vllm_config)
            engine_id = ktc.get("engine_id", vllm_config.kv_transfer_config.engine_id)
            temp_config.kv_transfer_config = KVTransferConfig(
                **ktc, engine_id=engine_id
            )
            ret.append(
                (
                    KVConnectorFactory.get_connector_class(
                        temp_config.kv_transfer_config
                    ),
                    temp_config,
                )
            )
        return ret

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L242-L247
    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type[AttentionBackend]
    ):
        # Register on all connectors
        for c in self._connectors:
            c.register_cross_layers_kv_cache(kv_cache, attn_backend)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L249-L251
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        for c in self._connectors:
            c.register_kv_caches(kv_caches)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L253-L255
    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        for c in self._connectors:
            c.bind_gpu_block_pool(gpu_block_pool)

    # We must override the base class method here because we need to bind
    # the metadata to each connector in the order of the connectors in the
    # MultiKVConnectorMetadata.
    #
    # Note: Call the base class method to ensure metadata is also set on the
    # MultiConnector instance itself; otherwise, `has_connector_metadata()` will
    # always return False.
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L264-L270
    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        assert isinstance(connector_metadata, MultiKVConnectorMetadata)
        if connector_metadata.extra_async_saves:
            self._extra_async_saves.update(connector_metadata.extra_async_saves)
        for c, cm in zip(self._connectors, connector_metadata.metadata):
            c.bind_connector_metadata(cm)
        super().bind_connector_metadata(connector_metadata)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L272-L275
    def clear_connector_metadata(self) -> None:
        for c in self._connectors:
            c.clear_connector_metadata()
        super().clear_connector_metadata()

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L277-L288
    def shutdown(self):
        exception: Exception | None = None
        for c in self._connectors:
            try:
                c.shutdown()
            except Exception as e:
                logger.exception(
                    "Exception during connector %s shutdown.", c.__class__.__name__
                )
                exception = e
        if exception:
            raise exception

    # ==============================
    # Worker-side methods
    # ==============================
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L293-L295
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        for c in self._connectors:
            c.start_load_kv(forward_context, **kwargs)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L297-L299
    def wait_for_layer_load(self, layer_name: str) -> None:
        for c in self._connectors:
            c.wait_for_layer_load(layer_name)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L301-L309
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> None:
        for c in self._connectors:
            c.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L311-L313
    def wait_for_save(self):
        for c in self._connectors:
            c.wait_for_save()

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L315-L340
    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        finished_sending: set[str] = set()
        finished_recving: set[str] = set()
        for c in self._connectors:
            sending, recving = c.get_finished(finished_req_ids)
            if not recving and not sending:
                continue
            # Aggregate finished recving request ids.
            finished_recving.update(recving or ())
            # Aggregate finished sending request ids - only include
            # once we've drained the "extra" count (for cases where
            # more than one connector is async-saving the same request).
            for req_id in sending or ():
                extra_pending = self._extra_async_saves.get(req_id)
                if extra_pending is None:
                    finished_sending.add(req_id)
                    continue
                assert extra_pending > 0
                if extra_pending == 1:
                    del self._extra_async_saves[req_id]
                else:
                    self._extra_async_saves[req_id] = extra_pending - 1

        return finished_sending or None, finished_recving or None

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L342-L346
    def get_block_ids_with_load_errors(self) -> set[int]:
        agg_block_ids: set[int] = set()
        for c in self._connectors:
            agg_block_ids |= c.get_block_ids_with_load_errors()
        return agg_block_ids

    # SUBTRACTED: set_host_xfer_buffer_ops（L348-L351）——xPU host buffer 旁路。

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L353-L357
    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata):
        """Handle preempted requests for all sub-connectors."""
        assert isinstance(kv_connector_metadata, MultiKVConnectorMetadata)
        for c, cm in zip(self._connectors, kv_connector_metadata.metadata):
            c.handle_preemptions(cm)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L359-L362
    def get_finished_count(self) -> int | None:
        # TODO(https://github.com/vllm-project/vllm/issues/33400)
        # Currently no connectors return non-None
        return None

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L364-L374
    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata | None:
        metadata_list: list[KVConnectorWorkerMetadata | None] | None = None
        for i, c in enumerate(self._connectors):
            kv_connector_worker_meta = c.build_connector_worker_meta()
            if metadata_list is None and kv_connector_worker_meta is not None:
                metadata_list = [None] * i
            if metadata_list is not None:
                metadata_list.append(kv_connector_worker_meta)
        if metadata_list is None:
            return None
        return MultiKVConnectorWorkerMetadata(metadata=tuple(metadata_list))

    # TODO: Add a generic implementation of 'get_kv_connector_kv_cache_events'
    # method for the MultiConnector. It should be able to get events from
    # multiple connectors, handling the case where only a subset of the
    # requested connectors implements the 'get_kv_connector_kv_cache_events'
    # WIP: https://github.com/vllm-project/vllm/pull/31811

    # ==============================
    # Scheduler-side methods
    # ==============================
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L385-L404
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        to_return = (0, False)
        for i, c in enumerate(self._connectors):
            toks, load_async = c.get_num_new_matched_tokens(
                request, num_computed_tokens
            )
            # If there is a connector still looking up the matches,
            # we return None to indicate that we are not done yet.
            if toks is None:
                return (None, False)
            # The first connector that has new matched tokens will be assigned
            # to this request.
            if to_return[0] == 0 and toks > 0:
                self._requests_to_connector[request.request_id] = i
                to_return = (toks, load_async)
        return to_return

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L406-L416
    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        chosen_connector = self._requests_to_connector.get(request.request_id, -1)
        for i, c in enumerate(self._connectors):
            if i == chosen_connector:
                # Forward call to the chosen connector (if any).
                c.update_state_after_alloc(request, blocks, num_external_tokens)
            else:
                # Other connectors still receive the request's real blocks
                c.update_state_after_alloc(request, blocks, 0)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L418-L420
    def on_new_request(self, request: "Request") -> None:
        for c in self._connectors:
            c.on_new_request(request)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L422-L433
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> MultiKVConnectorMetadata:
        metadata = MultiKVConnectorMetadata(
            metadata=tuple(
                c.build_connector_meta(scheduler_output) for c in self._connectors
            )
        )
        if self._extra_async_saves:
            metadata.extra_async_saves = self._extra_async_saves
            self._extra_async_saves = {}
        return metadata

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L435-L454
    def update_connector_output(self, connector_output: KVConnectorOutput):
        multi_connector_worker_meta: MultiKVConnectorWorkerMetadata | None = None
        if connector_output.kv_connector_worker_meta is not None:
            assert isinstance(
                connector_output.kv_connector_worker_meta,
                MultiKVConnectorWorkerMetadata,
            )
            multi_connector_worker_meta = connector_output.kv_connector_worker_meta

        try:
            for i, c in enumerate(self._connectors):
                if multi_connector_worker_meta is not None:
                    # set the connector-specific worker metadata
                    connector_output.kv_connector_worker_meta = (
                        multi_connector_worker_meta.metadata[i]
                    )
                c.update_connector_output(connector_output)
        finally:
            # restore kv_connector_worker_meta
            connector_output.kv_connector_worker_meta = multi_connector_worker_meta

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L456-L465
    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
        """
        Get the KVConnector handshake metadata from sub-connectors.
        Returns the first non-None metadata from sub-connectors.
        """
        for c in self._connectors:
            metadata = c.get_handshake_metadata()
            if metadata is not None:
                return metadata
        return None

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L467-L475
    def set_xfer_handshake_metadata(
        self, metadata: dict[int, KVConnectorHandshakeMetadata]
    ) -> None:
        """
        Set the KV connector handshake metadata for all sub-connectors.
        This is needed to start the NIXL listener thread for NixlConnector.
        """
        for c in self._connectors:
            c.set_xfer_handshake_metadata(metadata)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L477-L481
    def set_xfer_handshake_metadata_pp_aware(
        self, metadata: dict[tuple[int, int], KVConnectorHandshakeMetadata]
    ) -> None:
        for c in self._connectors:
            c.set_xfer_handshake_metadata_pp_aware(metadata)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L483-L512
    def _aggregate_request_finished(
        self,
        request: "Request",
        per_connector_fn: Callable[
            [KVConnectorBase_V1], tuple[bool, dict[str, Any] | None]
        ],
    ) -> tuple[bool, dict[str, Any] | None]:
        async_saves = 0
        kv_txfer_params = None
        for c in self._connectors:
            async_save, txfer_params = per_connector_fn(c)
            if async_save:
                async_saves += 1
            if txfer_params is not None:
                if kv_txfer_params is not None:
                    clashes = set(kv_txfer_params) & set(txfer_params)
                    if clashes:
                        raise RuntimeError(
                            "Key clash in kv_transfer_params from multiple "
                            f"connectors: {clashes}"
                        )
                    kv_txfer_params.update(txfer_params)
                else:
                    kv_txfer_params = txfer_params
        if async_saves > 1:
            self._extra_async_saves[request.request_id] = async_saves - 1

        self._requests_to_connector.pop(request.request_id, None)

        return async_saves > 0, kv_txfer_params

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L514-L522
    def request_finished(
        self,
        request: "Request",
        blocks: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self._aggregate_request_finished(
            request,
            lambda c: c.request_finished(request, blocks),
        )

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L524-L541
    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        if not self._all_support_hma:
            assert len(block_ids) == 1, (
                "HMA with multiple kv_cache_groups requires all "
                "sub-connectors to support HMA"
            )
            return self.request_finished(request, block_ids[0])

        return self._aggregate_request_finished(
            request,
            lambda c: cast(SupportsHMA, c).request_finished_all_groups(
                request, block_ids
            ),
        )

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L543-L545
    def take_events(self) -> Iterable["KVCacheEvent"]:
        for c in self._connectors:
            yield from c.take_events()

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L547-L548
    def has_pending_push_work(self) -> bool:
        return any(c.has_pending_push_work() for c in self._connectors)

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L551-L579
        """
        Get the required KV cache layout for this connector.
        Args:
            vllm_config (VllmConfig): the vllm config.

        Returns:
            str: the required KV cache layout. e.g. HND, or NHD.
            None if the connector does not require a specific layout.
        """
        assert vllm_config.kv_transfer_config is not None
        layouts: set[str] = set()
        for connector_cls, temp_config in cls._get_connector_classes_and_configs(
            vllm_config
        ):
            required_kvcache_layout = connector_cls.get_required_kvcache_layout(
                temp_config
            )
            if required_kvcache_layout is not None:
                layouts.add(required_kvcache_layout)

        if len(layouts) > 1:
            raise ValueError(
                f"KV cache layout mismatch: "
                f"found {len(layouts)} different layouts "
                f"({', '.join(layouts)})."
                f"All connectors must use the same layout."
            )
        return next(iter(layouts), None)

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L582-L619
        if data is None:
            return MultiKVConnectorStats()

        # data is a dict mapping connector name to their stats data.
        # The stats data can be either:
        # 1. Already-instantiated KVConnectorStats objects (same process)
        # 2. Serialized dicts (cross-process after serialization)
        # We need to reconstruct proper KVConnectorStats objects from dicts
        reconstructed_data = {}
        for connector_name, stats_value in data.items():
            # If already a KVConnectorStats object, use it directly
            if isinstance(stats_value, KVConnectorStats):
                reconstructed_data[connector_name] = stats_value
                continue

            # Otherwise, reconstruct from serialized dict
            # Get the connector class to reconstruct its stats
            connector_cls = KVConnectorFactory.get_connector_class_by_name(
                connector_name
            )

            # stats_value is the serialized dataclass which contains {'data': {...}}
            # We need to extract the inner 'data' field to avoid double-nesting
            assert isinstance(stats_value, dict) and "data" in stats_value, (
                f"Expected a dict with a 'data' field, got {stats_value}"
            )
            inner_data = stats_value["data"]

            # Use the connector's build_kv_connector_stats to reconstruct
            if reconstructed_stats := connector_cls.build_kv_connector_stats(
                data=inner_data
            ):
                reconstructed_data[connector_name] = reconstructed_stats

        return MultiKVConnectorStats(data=reconstructed_data)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L621-L638
    def get_kv_connector_stats(self) -> MultiKVConnectorStats | None:
        # Group connector stats by connector type.
        stats_by_connector: MultiKVConnectorStats | None = None
        for c in self._connectors:
            stats = c.get_kv_connector_stats()
            if stats is None:
                continue
            if stats_by_connector is None:
                # Lazy init to allow optional return value.
                stats_by_connector = MultiKVConnectorStats()
            connector_id = c.__class__.__name__
            if connector_id in stats_by_connector.data:
                stats_by_connector[connector_id] = stats_by_connector[
                    connector_id
                ].aggregate(stats)
            else:
                stats_by_connector[connector_id] = stats
        return stats_by_connector

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: "VllmConfig",
        metric_types: dict[type["PromMetric"], type["PromMetricT"]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L641-L667
        prom_metrics: dict[str, KVConnectorPromMetrics] = {}
        seen_classes: set[type] = set()
        for connector_cls, temp_config in cls._get_connector_classes_and_configs(
            vllm_config
        ):
            if connector_cls in seen_classes:
                continue
            seen_classes.add(connector_cls)
            connector_prom = connector_cls.build_prom_metrics(
                temp_config, metric_types, labelnames, per_engine_labelvalues
            )
            if connector_prom is not None:
                prom_metrics[connector_cls.__name__] = connector_prom
        return MultiKVConnectorPromMetrics(
            vllm_config,
            metric_types,
            labelnames,
            per_engine_labelvalues,
            prom_metrics,
        )

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L669-L671
    def reset_cache(self) -> bool:
        results = [c.reset_cache() is not False for c in self._connectors]
        return all(results)
