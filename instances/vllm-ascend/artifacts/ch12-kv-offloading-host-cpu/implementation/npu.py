# vllm_ascend/kv_offload/npu.py —— subtract-only companion（标准路径接入点 NPUOffloadingSpec）
#
# 标准路径如何接 vLLM v1/kv_offload 框架：
#   · scheduler 侧 get_manager() —— 直接复用基座 CPUOffloadingManager（卸载的记账/分配/lookup
#     调度与硬件无关，昇腾无需重写），block_size = gpu_block_size * block_size_factor。
#   · worker 侧 get_handlers() —— 造一个昇腾自带 CpuNpuOffloadingHandler，并把它同时注册到
#     (GPU→CPU) 与 (CPU→GPU) 两个方向（双向收进同一个 handler）。
#   · num_cpu_blocks 必须经 kv_connector_extra_config 给定，否则开箱即抛。
#
# host 无 NPU/vllm：OffloadingSpec/OffloadingManager/CPUOffloadingManager/LoadStoreSpec/
#   GPULoadStoreSpec/CPULoadStoreSpec 由 runtime_stub 接住；get_manager/get_handlers 的
#   分发控制流是纯 Python，可跑可断言（实际搬运不真跑）。
from collections.abc import Iterator

import torch

from cpu_npu import CpuNpuOffloadingHandler
from runtime_stub import (
    AttentionBackend,
    CPULoadStoreSpec,
    CPUOffloadingManager,
    GPULoadStoreSpec,
    KVCacheConfig,
    LoadStoreSpec,
    OffloadingHandler,
    OffloadingManager,
    OffloadingSpec,
    VllmConfig,
)


class NPUOffloadingSpec(OffloadingSpec):  # SOURCE: vllm_ascend/kv_offload/npu.py:L16
    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig | None = None):  # SOURCE: vllm_ascend/kv_offload/npu.py:L17
        super().__init__(vllm_config, kv_cache_config)

        num_cpu_blocks = self.extra_config.get("num_cpu_blocks")
        if not num_cpu_blocks:
            raise Exception("num_cpu_blocks must be specified in kv_connector_extra_config")
        self.num_cpu_blocks: int = num_cpu_blocks

        # scheduler-side
        self._manager: OffloadingManager | None = None

        # worker-side
        self._handler: OffloadingHandler | None = None

    def get_manager(self) -> OffloadingManager:  # SOURCE: vllm_ascend/kv_offload/npu.py:L31
        if not self._manager:
            # SUBTRACTED: kv_events_config / enable_events 的读取（npu.py:L33-L34）——
            #   事件开关只透传给基座 CPUOffloadingManager 做 KV-cache-event 上报，
            #   与 device↔host 搬运机制无关；精简版固定 enable_events=False。
            assert len(self.gpu_block_size) == 1
            gpu_block_size = self.gpu_block_size[0]
            offloaded_block_size = gpu_block_size * self.block_size_factor
            self._manager = CPUOffloadingManager(
                block_size=offloaded_block_size,
                num_blocks=self.num_cpu_blocks,
                enable_events=False,  # SUBTRACTED: 原为 enable_events=enable_events（npu.py:L41）
            )
        return self._manager

    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:  # SOURCE: vllm_ascend/kv_offload/npu.py:L45
        if not self._handler:
            assert len(self.gpu_block_size) == 1
            gpu_block_size = self.gpu_block_size[0]
            self._handler = CpuNpuOffloadingHandler(
                # SUBTRACTED: attn_backends=attn_backends（npu.py:L54）——Handler 内从未读取该形参，
                #   死参，连同 Handler 侧形参一并删去。
                gpu_block_size=gpu_block_size,
                cpu_block_size=gpu_block_size * self.block_size_factor,
                num_cpu_blocks=self.num_cpu_blocks,
                gpu_caches=kv_caches,
            )

        assert self._handler is not None
        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handler
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handler
