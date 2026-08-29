# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py
# **双面契约本体**（m1——本章第一主角，两半方法面逐行可读）：模块
# docstring 就是契约正文（两半原语逐条清单）；KVConnectorRole 枚举
# （role-split 的根）；SupportsHMA（混合模型逐组交接的门）；KVConnector_
# Metadata/KVConnectorWorkerMetadata（跨进程两份不透明信封）；
# KVConnectorBase_V1——调度器侧五原语（查/记账/交接/事件）+ worker 侧
# 六原语（逐层收发/完成上报）+ requires_kv_delivery（producer 交接护栏）。
# SUBTRACTED（dossier.delete 批准项的落点 + 邻章边界）：
#   第 4 条 P/D 跨引擎握手族：KVConnectorHandshakeMetadata、
#     get_handshake_metadata、set_xfer_handshake_metadata(_pp_aware)；
#   第 5 条 cross-layer/uniform 布局族：prefer_cross_layer_blocks、
#     register_cross_layers_kv_cache、set_host_xfer_buffer_ops、CopyBlocksOp；
#   第 6 条 requires_piecewise_for_cudagraph 与 get_required_kvcache_layout
#     （CUDA graph 交互 → ch19）；
#   第 3 条观测面：take_events、get_kv_connector_stats、
#     get_kv_connector_kv_cache_events、build_kv_connector_stats、
#     build_prom_metrics（第 7 条保留 shutdown/reset_cache/get_finished_
#     count 的签名与默认实现）。
import enum
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

from .output import SchedulerOutput
from .outputs import KVConnectorOutput

if TYPE_CHECKING:
    from .config import VllmConfig
    from .kv_cache_interface import KVCacheConfig
    from .kv_cache_manager import KVCacheBlocks
    from .request import Request
    from .block_pool import BlockPool

logger = logging.getLogger(__name__)  # LOGGER SEAM：vllm.logger.init_logger → stdlib


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L3-L41 模块
#   docstring——契约正文（两半原语逐条清单）
"""
KVConnectorBase_V1 Class for Distributed KV Cache & Hidden State
communication in vLLM v1

The class provides the following primitives:
    Scheduler-side: runs in the scheduler, binds metadata, which
    is used by the worker-side to load/save KV cache.
        get_num_new_matched_tokens() - get number of new tokens
            that exist in the remote KV cache. Might be called multiple
            times for a given request and should be side-effect free.
        update_state_after_alloc() - update KVConnector state after
            temporary buffer alloc by the CacheManager.
        update_connector_output() - update KVConnector state after
            output is received from worker-side connectors.
        request_finished() - called once when a request is finished,
            with the computed kv cache blocks for the request.
            Returns whether KV cache should be freed now or if the
            connector now assumes responsibility for freeing the
            the blocks asynchronously. Also optionally returns KV
            transfer params.
        take_events() - returns new KV events that were collected
            by the connector since the last call.

    Worker-side: runs in each worker, loads/saves KV cache to/from
    the Connector based on the metadata.
        handle_preemptions() - called for handling preempted requests
            or request evicted blocks before they are overwritten

        start_load_kv() - starts loading all KVs (maybe async)
        wait_for_layer_load() - blocks until layer i load is done

        save_kv_layer() - starts saving KV for layer i (maybe async)
        wait_for_save() - blocks until all saves are done

        get_finished() - called with ids of finished requests, returns
            ids of requests that have completed async sending/recving.
        build_connector_worker_meta() - builds metadata to be sent
            back to the scheduler-side connector
"""


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L85 SupportsHMA
class SupportsHMA(ABC):
    """
    The class that indicates the corresponding connector supports hybrid memory
    allocator (HMA).
    This is required to use the connector together with hybrid memory allocator.
    """

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L92
    @abstractmethod
    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called exactly once when a request has finished for all kv cache groups,
        before its blocks are freed for each group.

        NOTE(Kuntai): This function is only supported by connectors that support HMA.

        The connector may assumes responsibility for freeing the blocks
        asynchronously by returning True.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L114
        raise NotImplementedError


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L117 supports_hma
def supports_hma(connector: Any) -> bool:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L118-L121
    #   （类与实例两态判定——factory 把门用）
    if isinstance(connector, type):
        return issubclass(connector, SupportsHMA)
    else:
        return isinstance(connector, SupportsHMA)


# KVConnectorRole——role-split 的根（SCHEDULER=0 / WORKER=1）
# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L124
class KVConnectorRole(enum.Enum):
    # Connector running in the scheduler process
    SCHEDULER = 0

    # Connector running in the worker process
    WORKER = 1


# SUBTRACTED: KVConnectorHandshakeMetadata（L132-L138——第 4 条 P/D 握手）。


# KVConnectorMetadata——调度器→worker 的不透明计划信封（m6）
# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L141
class KVConnectorMetadata(ABC):  # noqa: B024
    """
    Abstract Metadata used to communicate
    Scheduler KVConnector -> Worker KVConnector.
    """

    pass


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L150
#   KVConnectorWorkerMetadata——worker→scheduler 的回传信封
class KVConnectorWorkerMetadata(ABC):
    """
    Abstract Metadata used to communicate back
    Worker KVConnector -> Scheduler KVConnector.

    Each worker can output its own metadata.
    For a single engine step, all metadata objects returned by workers
    will be aggregated using the `aggregate` method below, before
    being passed to the Scheduler KVConnector.
    """

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L161 aggregate
    @abstractmethod
    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "KVConnectorWorkerMetadata":
        """
        Aggregate metadata with another `KVConnectorWorkerMetadata` object.
        """
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L167
        pass


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L171 KVConnectorBase_V1
class KVConnectorBase_V1(ABC):
    """
    Base class for KV connectors.
    """

    # SUBTRACTED: prefer_cross_layer_blocks（L176-L182——第 5 条布局族）。

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L184
    #   requires_kv_delivery——producer 交接可靠性旗标（m13）
    @property
    def requires_kv_delivery(self) -> bool:
        """Whether this connector hands off KV that must be reliably delivered.

        If True, a request preempted while its hand-off is still pending is
        recomputed rather than allowed to finish and hand off blocks that the
        preemption already freed. Defaults to the producer role, since only a
        producer hands KV off when a request completes. Best-effort caches
        return False, as a dropped save is just a future cache miss.
        """
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L194
        return self._kv_transfer_config.is_kv_producer

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L196 __init__
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L202-L205
        #   （实验性警告——契约的自我声明）
        logger.warning(
            "Initializing KVConnectorBase_V1. This API is experimental and "
            "subject to change in the future as we iterate the design."
        )
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L206-L213
        self._connector_metadata: KVConnectorMetadata | None = None
        self._vllm_config = vllm_config
        if vllm_config.kv_transfer_config is not None:
            self._kv_transfer_config = vllm_config.kv_transfer_config
        else:
            raise ValueError("kv_transfer_config must be set for KVConnectorBase_V1")
        self._kv_cache_config = kv_cache_config
        self._role = role

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L215 role
    @property
    def role(self) -> KVConnectorRole:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L217
        #   （只读——角色终身不变）
        return self._role

    # ==============================
    # Worker-side methods
    # ==============================

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L223
    #   bind_connector_metadata——worker 收计划（mixin 一拍的第一步）
    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        """Set the connector metadata from the scheduler.

        This function should be called by the model runner every time
        before the model execution. The metadata will be used for runtime
        KV cache loading and saving.

        Args:
            connector_metadata (dict): the connector metadata.
        """
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L233
        self._connector_metadata = connector_metadata

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L235
    #   clear_connector_metadata——一拍收尾
    def clear_connector_metadata(self) -> None:
        """Clear the connector metadata.

        This function should be called by the model runner every time
        after the model execution.
        """
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L241
        self._connector_metadata = None

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L243
    #   _get_connector_metadata——connector 内部取计划
    def _get_connector_metadata(self) -> KVConnectorMetadata:
        """Get the connector metadata.

        This function should only be called inside the connector.

        Returns:
            ConnectorMetadata: the connector metadata.
        """
        # Should only be called while set to valid metadata.
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L252-L253
        assert self._connector_metadata is not None
        return self._connector_metadata

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L255
    #   has_connector_metadata——装饰器零开销直通的判据
    def has_connector_metadata(self) -> bool:
        """Check whether the connector metadata is currently set.

        Returns:
            bool: True if connector metadata exists, False otherwise.
        """
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L261
        return self._connector_metadata is not None

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L263
    #   register_kv_caches——worker 拿池张量（m14：按层名注册）
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Initialize with the KV caches. Useful for pre-registering the
        KV Caches in the KVConnector (e.g., for NIXL).

        Args:
            kv_caches: dictionary of layer names, kv cache
        """
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L271
        return

    # SUBTRACTED: register_cross_layers_kv_cache / set_host_xfer_buffer_ops
    #   （L273-L295——第 5 条布局族）。

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L297
    #   handle_preemptions——覆写前抢救（OffloadingConnector 的异步 save
    #   抢在被抢占/被驱逐的块被覆写之前存出去）
    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata):
        """
        Handle preempted requests or evicted blocks BEFORE they are overwritten.
        Needed for connectors which use async saves (e.g., OffloadingConnector)
        """
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L302
        return

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L304
    #   start_load_kv——异步加载发起（m8：forward context 内、前向开始前）
    @abstractmethod
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.

        """
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L320
        pass

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L322
    #   wait_for_layer_load——逐层重叠支点（m8：注意力层内阻塞到本层到位）
    @abstractmethod
    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L334
        pass

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L336
    #   save_kv_layer——层算完异步存出（m8）
    @abstractmethod
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """
        Start saving a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: arguments for the save operation.
        """
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L356
        pass

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L358
    #   wait_for_save——强制同步点（docstring 原话是正确性论据）
    @abstractmethod
    def wait_for_save(self):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L367
        pass

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L369
    #   get_finished——异步收/发完成上报（m9/m11：调度器据此放块）
    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens on the worker.
        The scheduler process (via the Executors) will use this output
        to track which workers are done.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L385
        return None, None

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L387
    #   get_block_ids_with_load_errors——失败块上报（m10：同步当拍报、
    #   异步最迟随 get_finished 同拍报）
    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Get the set of block IDs that failed to load.

        Returns:
            Set of block IDs that encountered load errors.
            Empty set if no load errors occurred.

        Notes:
            - Applies to both sync- and async-loading requests.
            - Async loading: failed blocks may be reported in any forward pass
              up to and including the pass where the request ID is returned by
              `get_finished()`. Even if failures occur, the request must still
              be reported via `get_finished()`, and the failed block IDs must
              appear here no later than that same pass.
            - Sync loading: failed blocks should be reported in the forward
              pass in which they are detected.
        """
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L405
        return set()

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L407 shutdown
    #   （第 7 条：保留签名与默认实现）
    def shutdown(self):
        """
        Shutdown the connector. This is called when the worker process
        is shutting down to ensure that all the async operations are
        completed and the connector is cleaned up properly.
        """
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L413
        return None

    # SUBTRACTED: get_kv_connector_stats / get_kv_connector_kv_cache_events
    #   / get_handshake_metadata（L415-L439——第 3/4 条观测与握手面）。

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L441
    #   build_connector_worker_meta——worker→scheduler 回传信封
    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata | None:
        """
        Build the KVConnector worker metadata for this engine step.

        Returns:
            KVConnectorWorkerMetadata: the worker metadata.
            None if no worker metadata is available.
        """
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L449
        return None

    # ==============================
    # Scheduler-side methods
    # ==============================

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L455
    #   bind_gpu_block_pool——调度器侧直读块池元数据（只读：inc/dec 引用、
    #   迭代前缀缓存块）
    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        """
        Bind the GPU block pool to the connector for per-GPU block status tracking.
        For example, inc/dec ref counts, or iterate over the prefix cache blocks.

        Args:
            gpu_block_pool: the GPU block pool.
        """
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L463
        #   （默认 no-op——后端覆写才落账；调度器侧的调用点在
        #   scheduler.py:L291-L294）
        return

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L465
    #   get_num_new_matched_tokens——外部命中查询（m2：第二个前缀缓存）
    @abstractmethod
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - An optional number of tokens that can be loaded from the
                  external KV cache beyond what is already computed.
                  If None, it means that the connector needs more time to
                  determine the number of matched tokens, and the scheduler
                  should query for this request again later.
                - `True` if external KV cache tokens will be loaded
                  asynchronously (between scheduler steps). Must be
                  'False' if the first element is 0.

        Notes:
            The connector should only consider the largest prefix of prompt-
            tokens for which KV cache is actually available at the time of the
            call. If the cache cannot be loaded for some tokens (e.g., due to
            connectivity issues or eviction), those tokens must not be taken
            into account.
        """
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L498
        pass

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L500
    #   update_state_after_alloc——分配后通知（判据 num_external_tokens
    #   而非 blocks 空否）
    @abstractmethod
    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.

        If get_num_new_matched_tokens previously returned True for a
        request, this function may be called twice for that same request -
        first when blocks are allocated for the connector tokens to be
        asynchronously loaded into, and second when any additional blocks
        are allocated, after the load/transfer is complete.

        Decide whether to load based on ``num_external_tokens``, not on
        whether ``blocks`` is empty: ``blocks`` may be non-empty even when
        ``num_external_tokens == 0`` (e.g. a non-chosen sub-connector of
        MultiConnector still receives the request's real blocks).

        Args:
            request (Request): the request object.
            blocks (KVCacheBlocks): the blocks allocated for the request.
            num_external_tokens (int): the number of tokens to load from the
                external KV cache. 0 means nothing should be loaded.
        """
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L524
        pass

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L526
    #   build_connector_meta——产不透明计划（m6：不许改 scheduler_output、
    #   调用即重置）
    @abstractmethod
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L539
        pass

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L541
    #   on_new_request——新请求登记钩子（scheduler.add_request 调用点）
    def on_new_request(self, request: "Request") -> None:
        """Called by the scheduler when a new request is added.

        Connectors can override this to inspect the request and perform
        bookkeeping. The default implementation is a no-op.
        """
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L547
        return

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L549
    #   update_connector_output——调度器侧消化 worker 回传
    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L557
        return

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L559
    #   request_finished——终局交接（True=接管异步释放，所有权转移②本体）
    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called exactly once when a request has finished, before its blocks are
        freed.

        The connector may assumes responsibility for freeing the blocks
        asynchronously by returning True.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L578
        return False, None

    # SUBTRACTED: take_events（L580-L587——第 3 条 kv events 面）。

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L589
    #   has_pending_push_work——push 型传输保活（m17：全部活请求结束后
    #   引擎还要步进等传输排空）
    def has_pending_push_work(self) -> bool:
        """Return True if the connector has push-mode work that requires
        the engine main loop to keep stepping (e.g. a P-side request whose
        KV blocks are waiting to be WRITTEN to a D node).

        Connectors that don't implement push-based KV transfer should
        leave this as False.
        """
        # TODO: replace with a more general connector hook for keeping the
        # scheduler alive (e.g. extend has_unfinished_requests).
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L599
        return False

    # SUBTRACTED: get_required_kvcache_layout / requires_piecewise_for_
    #   cudagraph（L601-L640——第 6 条 CUDA graph 交互 → ch19）；
    #   get_finished_count / build_kv_connector_stats / set_xfer_handshake_
    #   metadata(_pp_aware) / build_prom_metrics / reset_cache 的非默认分支
    #   （L642-L720——第 3/4/7 条：签名与默认实现的钩子面不进本章控制流，
    #   后端覆写归 ch36/37）。


# SOURCE: vllm/distributed/kv_transfer/kv_connector/base.py（v1 时代的类型
#   别名面：KVConnectorBase = KVConnectorBase_V1——mixin/factory 的类型注解
#   用它）
KVConnectorBase = KVConnectorBase_V1
