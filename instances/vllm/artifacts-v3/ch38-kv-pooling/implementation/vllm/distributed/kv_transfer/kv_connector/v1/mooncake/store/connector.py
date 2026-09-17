# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# MooncakeStoreConnector 骨架：分布式池 facade（注册行 + role-split + 空契约面）。
#
# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py:L1-L354
# SUBTRACTED（减法计划删除项 6：Mooncake 实现体不进精简版——master 协调、
#   embedded/standalone 双模式、replica tier、_validate_kv_cache_config 的
#   混合模型门、MooncakeStoreKVEvents 聚合器）：保留注册行指向的 facade
#   骨架——role-split 建半边 + 『I/O 全押 get_finished』的空契约面 +
#   查询/加载的调度器委托。
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.forward_context import ForwardContext
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request

from .scheduler import MooncakeStoreScheduler
from .worker import MooncakeStoreWorker


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py:L87-L354
class MooncakeStoreConnector(KVConnectorBase_V1, SupportsHMA):
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py:L88
    """KV connector using MooncakeDistributedStore as shared KV pool."""

    # SUBTRACTED: prefer_cross_layer_blocks 属性与 _validate_kv_cache_config
    #   （L90-L126）——enable_cross_layers_blocks 开关与混合模型部署门
    #   （CrossAttention/Mamba align 校验）随实现体删除（删除项 6）。

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py:L128-L162
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py:L133-L151
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,  # type: ignore[arg-type]
        )
        assert vllm_config.kv_transfer_config is not None
        assert kv_cache_config is not None, "kv_cache_config is required"
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        # SUBTRACTED: _capacity_only 判定（L144-L150）——enable_lookup=false
        #   的容量贡献模式随实现体删除。

        self.connector_scheduler: MooncakeStoreScheduler | None = None
        self.connector_worker: MooncakeStoreWorker | None = None

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = MooncakeStoreScheduler(
                vllm_config, kv_cache_config
            )
        else:
            self.connector_worker = MooncakeStoreWorker(vllm_config, kv_cache_config)

    # ============================================================
    # Scheduler-side methods
    # ============================================================

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py:L183-L191
    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py:L183-L191
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py:L193-L202
    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py:L193-L202
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    # SUBTRACTED: build_connector_meta / request_finished / reset_cache 的
    #   调度器方法（L204-L241）——实现体随 scheduler 深枝删除（删除项 6）。

    # ============================================================
    # Worker-side methods
    # ============================================================

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py:L283-L285
    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py:L283-L285
        # No-op: loads are issued in get_finished() for compute overlap.
        pass

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py:L287-L289
    def wait_for_layer_load(self, layer_name: str) -> None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py:L287-L289
        # No layerwise support - no-op
        return

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py:L291-L299
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: Any,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py:L291-L299
        # No layerwise support - no-op
        return

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py:L301-L303
    def wait_for_save(self):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py:L301-L303
        # No-op: stores are issued in get_finished() for compute overlap.
        pass

    # SUBTRACTED: get_finished / get_block_ids_with_load_errors / 事件面
    #   （L305-L332）——worker 双线程实现体删除后无完成信号面（删除项 6）。
