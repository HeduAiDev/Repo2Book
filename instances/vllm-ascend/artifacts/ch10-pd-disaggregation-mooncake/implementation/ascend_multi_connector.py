# vllm_ascend/distributed/kv_transfer/__init__.py (register_connector) +
# vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py (AscendMultiConnector)
#   —— subtract-only companion（ch10 第 1 层：连接器分发）
#
# register_connector：模块 init 钩子，把 vLLM 内置 'MultiConnector' 工厂项 pop 掉再指向
#   AscendMultiConnector，并按名注册三个 mooncake 连接器——配置字符串就此路由到昇腾类。
# AscendMultiConnector：MultiConnector + SupportsHMA 子类。继承 base 的「从首个命中子连接器
#   load、save 给全部」fan-out；只覆写两处分歧（update_state_after_alloc 让 layerwise 永远拿真
#   blocks；request_finished_all_groups 走 HMA 逐组聚合）。
from typing import TYPE_CHECKING, Any, cast

from runtime_stub import (
    KVConnectorFactory,
    KVConnectorRole,
    SupportsHMA,
    supports_hma,
)
from multi_connector import MultiConnector
from mooncake_layerwise_connector import MooncakeLayerwiseConnector

if TYPE_CHECKING:
    from runtime_stub import KVCacheBlocks, Request


# SOURCE: vllm_ascend/distributed/kv_transfer/__init__.py:L21
def register_connector():
    # override multi_connector as ascend_multi_connector
    if "MultiConnector" in KVConnectorFactory._registry:
        KVConnectorFactory._registry.pop("MultiConnector")
    KVConnectorFactory.register_connector(
        "MultiConnector", "vllm_ascend.distributed.kv_transfer.ascend_multi_connector", "AscendMultiConnector"
    )

    KVConnectorFactory.register_connector(
        "MooncakeConnectorV1", "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector", "MooncakeConnector"
    )

    KVConnectorFactory.register_connector(
        "MooncakeHybridConnector",
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector",
        "MooncakeConnector",
    )

    KVConnectorFactory.register_connector(
        "MooncakeConnectorStoreV1",
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector",
        "AscendStoreConnector",
    )

    KVConnectorFactory.register_connector(
        "AscendStoreConnector",
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector",
        "AscendStoreConnector",
    )

    KVConnectorFactory.register_connector(
        "MooncakeLayerwiseConnector",
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector",
        "MooncakeLayerwiseConnector",
    )
    # SUBTRACTED: UCMConnector / LMCacheAscendConnector / SimpleCPUOffloadConnector 注册
    #   （L57-L81）—— 同样的 register_connector(name, module, cls) 套路，不在 PD / 亲和路径上。
    #   原 vllm_ascend/distributed/kv_transfer/__init__.py:L57-L81


# SOURCE: vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py:L19
class AscendMultiConnector(MultiConnector, SupportsHMA):
    # SOURCE: vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py:L20
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole, kv_cache_config: "KVCacheConfig"):  # noqa: F821
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )

        self._all_support_hma = all(supports_hma(c) for c in self._connectors)
        assert vllm_config.scheduler_config.disable_hybrid_kv_cache_manager or self._all_support_hma, (
            "HMA should not be enabled unless all sub-connectors support it"
        )

    # SOURCE: vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py:L32
    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        chosen_connector = self._requests_to_connector.get(request.request_id, -1)
        empty_blocks = blocks.new_empty()
        for i, c in enumerate(self._connectors):
            if i == chosen_connector or isinstance(c, MooncakeLayerwiseConnector):
                # Forward call to the chosen connector (if any).
                c.update_state_after_alloc(request, blocks, num_external_tokens)
            else:
                # Call with empty blocks for other connectors.
                c.update_state_after_alloc(request, empty_blocks, 0)

    # SOURCE: vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py:L43
    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        if not self._all_support_hma:
            assert len(block_ids) == 1, "HMA with multiple kv_cache_groups requires all sub-connectors to support HMA"
            return super().request_finished(request, block_ids[0])

        async_saves = 0
        kv_txfer_params = None
        for c in self._connectors:
            async_save, txfer_params = cast(SupportsHMA, c).request_finished_all_groups(request, block_ids)
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
