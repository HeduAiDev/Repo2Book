# vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py —— subtract-only companion
#
# 对照基座 vLLM v0.21.0 的 MultiConnector：一个把多个 KVConnector 串起来的 fan-out 包装器。
#   契约：「从第一个 advertise 命中 token 的子连接器 load；save 给所有子连接器。」
# 昇腾 AscendMultiConnector 子类化它（见 ascend_multi_connector.py），ch10 要看「加了什么」，
# 所以这里保留 base 的选举 + update_state_after_alloc + HMA 聚合 request_finished 三处对照点。
#
# host 无 vllm：基类 / 角色枚举 / HMA marker 经 runtime_stub 接住（标 # SOURCE 指向真实符号）。
from typing import TYPE_CHECKING, Any, cast

from runtime_stub import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
    logger,
    supports_hma,
)

# SUBTRACTED: import copy / torch / VllmConfig / KVTransferConfig / KVConnectorFactory /
#   metrics(KVConnectorStats/PromMetrics) 等 —— 仅服务被删的 stats/prom/worker-meta/工厂构造路径。
#   原 vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L3-L42

if TYPE_CHECKING:
    from runtime_stub import KVCacheBlocks, Request, SchedulerOutput


# SUBTRACTED: MultiKVConnectorMetadata / MultiKVConnectorWorkerMetadata / MultiKVConnectorStats /
#   MultiKVConnectorPromMetrics 四个 dataclass —— 指标聚合与 worker 侧元数据打包，ch10 不讲。
#   原 multi_connector.py:L45-L125


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L128
class MultiConnector(KVConnectorBase_V1, SupportsHMA):
    """
    A wrapper for using multiple KVConnectors at the same time.

    The current logic is:
    - Load KV from the first connector that advertises available tokens from
      get_num_new_matched_tokens(), based on the order in the config.
    - Save to all connectors.
    """

    # SUBTRACTED: requires_piecewise_for_cudagraph (L138-L151) —— cudagraph 编译模式探测，与分发无关。

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L153
    def __init__(
        self,
        vllm_config: "VllmConfig",  # noqa: F821
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",  # noqa: F821
    ):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)

        self._connectors: list[KVConnectorBase_V1] = []
        self._ktc_kv_transfer_config = []
        for connector_cls, temp_config in self._get_connector_classes_and_configs(vllm_config):
            self._connectors.append(connector_cls(temp_config, role, kv_cache_config))
            self._ktc_kv_transfer_config.append(temp_config.kv_transfer_config)

        self._all_support_hma = all(supports_hma(c) for c in self._connectors)
        assert vllm_config.scheduler_config.disable_hybrid_kv_cache_manager or self._all_support_hma, (
            "HMA should not be enabled unless all sub-connectors support it"
        )

        # A mapping from request id to the index of the connector chosen to
        # load the request from (if any).
        self._requests_to_connector: dict[str, int] = {}

        # Keeps track of *additional* remaining async saves (beyond 1) to be
        # finished per request.
        self._extra_async_saves: dict[str, int] = {}

    # SUBTRACTED: _get_connector_classes_and_configs 真实体（L193-L217）—— 从 vllm 配置工厂
    #   实例化子连接器；host 无 vllm 配置，测试直接装配 self._connectors。原 multi_connector.py:L193
    @classmethod
    def _get_connector_classes_and_configs(cls, vllm_config):  # SOURCE: multi_connector.py:L193
        return []

    # SUBTRACTED: prefer_cross_layer_blocks / register_cross_layers_kv_cache / register_kv_caches /
    #   bind_connector_metadata / clear_connector_metadata / shutdown （L187-L261）——
    #   元数据绑定 / 缓存注册 / 关闭，是 worker 进程生命周期，非 ch10 的「选举 + 分发」主线。

    # ==============================
    # Worker-side methods
    # ==============================
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L266
    def start_load_kv(self, forward_context, **kwargs) -> None:
        for c in self._connectors:
            c.start_load_kv(forward_context, **kwargs)

    # SUBTRACTED: wait_for_layer_load / save_kv_layer / wait_for_save （L270-L286）——
    #   逐层保存/等待的 fan-out，layerwise 子连接器的版本在 mooncake_layerwise_connector.py 讲透。

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L288
    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        finished_sending: set[str] = set()
        finished_recving: set[str] = set()
        for c in self._connectors:
            sending, recving = c.get_finished(finished_req_ids)
            if not recving and not sending:
                continue
            # Aggregate finished recving request ids.
            finished_recving.update(recving or ())
            # Aggregate finished sending request ids - only include once we've
            # drained the "extra" count (for >1 connector async-saving the same req).
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

    # SUBTRACTED: get_block_ids_with_load_errors / set_host_xfer_buffer_ops / handle_preemptions /
    #   get_finished_count / build_connector_worker_meta （L315-L347）—— worker 元数据与抢占处理。

    # ==============================
    # Scheduler-side methods
    # ==============================
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L358
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        to_return = (0, False)
        for i, c in enumerate(self._connectors):
            toks, load_async = c.get_num_new_matched_tokens(request, num_computed_tokens)
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

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L379
    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        chosen_connector = self._requests_to_connector.get(request.request_id, -1)
        empty_blocks = blocks.new_empty()
        for i, c in enumerate(self._connectors):
            if i == chosen_connector:
                # Forward call to the chosen connector (if any).
                c.update_state_after_alloc(request, blocks, num_external_tokens)
            else:
                # Call with empty blocks for other connectors.
                c.update_state_after_alloc(request, empty_blocks, 0)

    # SUBTRACTED: build_connector_meta / update_connector_output / get_handshake_metadata /
    #   set_xfer_handshake_metadata （L392-L445）—— 每步元数据打包 + 握手元数据分发。

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L447
    def _aggregate_request_finished(
        self,
        request: "Request",
        per_connector_fn,
    ) -> tuple[bool, dict[str, Any] | None]:
        async_saves = 0
        kv_txfer_params = None
        for c in self._connectors:
            async_save, txfer_params = per_connector_fn(c)
            if async_save:
                async_saves += 1
            if txfer_params is not None:
                if kv_txfer_params is not None:
                    raise RuntimeError("Only one connector can produce KV transfer params")
                kv_txfer_params = txfer_params
        if async_saves > 1:
            self._extra_async_saves[request.request_id] = async_saves - 1

        self._requests_to_connector.pop(request.request_id, None)

        return async_saves > 0, kv_txfer_params

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L475
    def request_finished(
        self,
        request: "Request",
        blocks: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self._aggregate_request_finished(
            request,
            lambda c: c.request_finished(request, blocks),
        )

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L485
    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        if not self._all_support_hma:
            assert len(block_ids) == 1, (
                "HMA with multiple kv_cache_groups requires all sub-connectors to support HMA"
            )
            return self.request_finished(request, block_ids[0])

        return self._aggregate_request_finished(
            request,
            lambda c: cast(SupportsHMA, c).request_finished_all_groups(request, block_ids),
        )

    # SUBTRACTED: take_events / get_required_kvcache_layout / build_kv_connector_stats /
    #   get_kv_connector_stats / build_prom_metrics / reset_cache （L504-L629）——
    #   事件流 / 布局校验 / 指标体系，与 ch10 的连接器分发主线无关。
