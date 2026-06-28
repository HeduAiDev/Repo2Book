"""调用方证据：NPUWorker 在哪三处用上 CaMemAllocator —— 只做减法的忠实摘录。

本章主角是 camem.py；worker.py 在这里只作「谁来调 allocator」的旁证，按 dossier
减法计划只保留与 camem 相关的那几行，方法体其余（日志/字节统计/FRACTAL_NZ 校验/
w2_weight reshape 等昇腾权重格式细节）一律 # SUBTRACTED。

host 不跑这些方法（依赖 torch.npu / model_runner / vllm 运行时），仅作可读摘录。
"""
# SUBTRACTED: 文件头 Apache-2.0 许可证 + 大量 import（原 worker.py:L1-L60+）
from contextlib import nullcontext

from camem import CaMemAllocator


class NPUWorker:
    # SOURCE: vllm_ascend/worker/worker.py（NPUWorker 类，仅保留三处 camem 调用方法）
    # SUBTRACTED: NPUWorker 的其余 ~700 行（init_device/initialize_cache/execute_model 等），
    #   非本章主线（原 vllm_ascend/worker/worker.py）。

    def sleep(self, level: int = 1) -> None:
        # SOURCE: vllm_ascend/worker/worker.py:L200-L207
        # SUBTRACTED: free_bytes_before_sleep = torch.npu.mem_get_info()[0]（L201）——睡前字节统计。
        # Save the buffers before level 2 sleep
        if level == 2:
            model = self.model_runner.model  # noqa: F821
            self._sleep_saved_buffers = {name: buffer.cpu().clone() for name, buffer in model.named_buffers()}
        allocator = CaMemAllocator.get_instance()
        # 两档语义都在这一行 offload_tags 体现：
        #   level 1 → ('weights',)：weights 拷回 CPU、kv_cache 直接丢（唤醒后 KV 重算）；
        #   level 2 → tuple()：什么都不 offload，weights 也丢，靠 wake_up 重新 load_model 从磁盘读回。
        allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())
        # SUBTRACTED: 睡后 free/used 字节统计 + assert + logger.info（原 L208-L216）——日志，非主线。

    def wake_up(self, tags: list[str] | None = None) -> None:
        # SOURCE: vllm_ascend/worker/worker.py:L218-L226
        # SUBTRACTED: FRACTAL_NZ(weight_nz_mode) 校验（原 L219-L224）——昇腾权重格式细节，非主线。
        allocator = CaMemAllocator.get_instance()
        allocator.wake_up(tags=tags)
        # SUBTRACTED: w2_weight transpose/reshape 复原（原 L228 起）——昇腾权重格式细节，非主线。

    def load_model(self) -> None:
        # SOURCE: vllm_ascend/worker/worker.py:L544-L555
        if self.vllm_config.model_config.enable_sleep_mode:  # noqa: F821
            allocator = CaMemAllocator.get_instance()
            assert allocator.get_current_usage() == 0, "Sleep mode can only be used for one instance per process."
            context = allocator.use_memory_pool(tag="weights")
        else:
            context = nullcontext()  # type: ignore

        with context, set_current_vllm_config(self.vllm_config):  # noqa: F821
            self.model_runner.load_model()  # noqa: F821

    def initialize_from_config(self, kv_cache_config) -> None:
        # SOURCE: vllm_ascend/worker/worker.py:L762-L773
        """Allocate NPU KV cache with the specified kv_cache_config."""
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)  # noqa: F821
        if self.vllm_config.model_config.enable_sleep_mode:  # noqa: F821
            allocator = CaMemAllocator.get_instance()
            context = allocator.use_memory_pool(tag="kv_cache")
        else:
            context = nullcontext()  # type: ignore
        with context:
            self.model_runner.initialize_kv_cache(kv_cache_config)  # noqa: F821
        # SUBTRACTED: KV-zero metadata 构建（原 L774 起）——非本章主线。
