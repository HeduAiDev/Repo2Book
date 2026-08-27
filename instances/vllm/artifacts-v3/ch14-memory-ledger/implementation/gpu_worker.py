# SOURCE: vllm/v1/worker/gpu_worker.py
# Worker 的账本三步（m1/m3/m9/m16 站 1/3/7/12 的 worker 半边）：
#   init_device 尾段——分布式初始化刻意先于显存快照（NCCL 缓冲先落地，
#   不然测出的 available 偏大）→ init_snapshot → request_memory；
#   determine_available_memory——memory_profiling 里 profile_run 测 non_kv、
#   cudagraph 估计入账、一行减法定出 KV 池本金；get_kv_cache_spec 转发；
#   update_max_model_len——auto-fit 后同步 worker 缓存值；
#   initialize_from_config——num_gpu_blocks 写回 + CuMem tag="kv_cache" 池内
#   真分配。
# ENGINE SEAM：model_runner 由装配方注入（真实 Worker 自建 GPUModelRunner，
#   L408-L423——ch17 执行器装配；切面构造期直供同契约位）；
#   current_platform——HOST SEAM 平台位（host = CPU → 池上下文恒
#   nullcontext；is_cuda_alike 的 cudagraph 门在 host 上由测试直供
#   compilation_config.cudagraph_mode）。
# SUBTRACTED（dossier.delete 批准项的落点 + 邻章边界）：
#   第 2 条 kv_cache_memory_bytes 手动凌驾早退（L474-L496）与
#     maybe_apply_startup_plan（L472）/maybe_save_startup_plan（复跑建议
#     落盘，WC6）；
#   第 3 条 reserve_mm_ipc_gpu_memory 包装（L492-L496、L607-L611——mm IPC
#     预留；返回值语义 = 取 int）；
#   第 11 条 weight_transfer_config（L444-L450——RL 权重传输）与 debug/
#     等价 util 日志块（L550-L562、L568-L605——教学金句正文引用、精简版省）；
#   DSV4/设备选择/DP rank 修正/工作区/模型装载（L343-L427——ch03/05/17）；
#   ensure_kv_transfer_initialized（L657-L662——ch16）、routed experts
#     （L667-L668）、compile_or_warm_up 编排（→ ch19）。
import gc
from contextlib import nullcontext

import torch

from .config import CUDAGraphMode
from .envs import envs
from .mem_utils import MemorySnapshot, format_gib, memory_profiling
from .worker_utils import request_memory


# SOURCE: vllm/platforms/interface.py:L198-L220 current_platform（HOST SEAM
#   平台位：host = CPU——is_cpu 恒 True、is_cuda_alike/is_xpu 恒 False；
#   容器内为真平台）
class _HostPlatform:
    # SOURCE: vllm/platforms/interface.py:L198 is_xpu
    def is_xpu(self) -> bool:
        return False

    # SOURCE: vllm/platforms/interface.py:L201 is_cpu
    def is_cpu(self) -> bool:
        return True

    # SOURCE: vllm/platforms/interface.py:L220 is_cuda_alike
    def is_cuda_alike(self) -> bool:
        return False


current_platform = _HostPlatform()


# SOURCE: vllm/v1/worker/gpu_worker.py:L~180 Worker（账本切面——真实类为
#   完整 worker 装配，ch17 全文）
class Worker:
    # SOURCE: vllm/v1/worker/gpu_worker.py:L~200 __init__（切面装配）
    def __init__(
        self,
        vllm_config,
        model_runner,
    ) -> None:
        # SUBTRACTED: rank/device_config/distributed_init_method/use_v2 等
        #   装配面（L200-L230——ch05/17）；model_runner 自建（L408-L423
        #   ——ENGINE SEAM：切面直供）。
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.parallel_config = vllm_config.parallel_config
        self.device = torch.device("cpu")  # HOST SEAM：切面 CPU
        self.model_runner = model_runner
        # init_device 尾段的账位（真实在 L390-L396 拍）——测试直供或调
        # init_device_snapshot_tail
        self.init_snapshot: MemorySnapshot | None = None
        self.requested_memory: int | None = None

    # SOURCE: vllm/v1/worker/gpu_worker.py:L372-L396 init_device 尾段（站点
    #   抽块：分布式初始化刻意先于快照的注释脉络 + 快照/预算两步）
    def init_device_snapshot_tail(self) -> None:
        # SOURCE: vllm/v1/worker/gpu_worker.py:L372-L375（注释原话：NCCL
        #   缓冲先落地，不然后测出的 available 偏大——真实调
        #   init_worker_distributed_environment，L376-L382 → ch05）
        # Initialize the distributed environment BEFORE taking
        # memory snapshot
        # This ensures NCCL buffers are allocated before we measure
        # available memory
        # SUBTRACTED: init_worker_distributed_environment / V2 runner /
        #   random seed（L376-L388——ch05/17）

        # Now take memory snapshot after NCCL is initialized
        # SOURCE: vllm/v1/worker/gpu_worker.py:L390-L392
        gc.collect()
        torch.accelerator.empty_cache()

        # take current memory snapshot
        # SOURCE: vllm/v1/worker/gpu_worker.py:L394-L396
        self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)
        self.requested_memory = request_memory(init_snapshot, self.cache_config)

    # SOURCE: vllm/v1/worker/gpu_worker.py:L459 determine_available_memory
    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Profiles the peak memory usage of the model to determine how much
        memory can be used for KV cache without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculates the free memory that can be used for KV cache in
        bytes.

        Tip:
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # SUBTRACTED: maybe_apply_startup_plan（L472——第 2 条：复跑建议
        #   落盘/复用旁路，WC6）；kv_cache_memory_bytes 手动凌驾早退
        #   （L474-L496——第 2 条：手动指定 KV 内存不走 profile 主线）。

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        # SOURCE: vllm/v1/worker/gpu_worker.py:L498-L504（memory_profiling 里
        #   跑 profile_run——weights_memory 记模型装载）
        with memory_profiling(
            self.init_snapshot,
            weights_memory=int(self.model_runner.model_memory_usage),
        ) as profile_result:
            self.model_runner.profile_run()

        # Profile CUDA graph memory if graphs will be captured.
        # SUBTRACTED: ROCm/XPU 平台注释块（L506-L511——只说明平台差异）
        # SOURCE: vllm/v1/worker/gpu_worker.py:L512-L517（cudagraph_mode 门：
        #   NONE 则不估）
        cudagraph_memory_estimate = 0
        if (
            current_platform.is_cuda_alike()
            and self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory()

        # Respect the opt-in flag as originally designed.
        # SOURCE: vllm/v1/worker/gpu_worker.py:L519-L524（估计开关——默认开，
        #   关了只不入账不关测量）
        cudagraph_memory_estimate_applied = (
            cudagraph_memory_estimate
            if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS
            else 0
        )

        # SOURCE: vllm/v1/worker/gpu_worker.py:L526-L530（账本读数账位）
        self.total_consumed = profile_result.total_consumed
        self.peak_activation_memory = (
            profile_result.transient_peak_headroom + cudagraph_memory_estimate_applied
        )
        self.cudagraph_memory_estimate = cudagraph_memory_estimate

        # SOURCE: vllm/v1/worker/gpu_worker.py:L532-L543 快照 assert（他进程
        #   在 profile 期间释放显存 = 初始 free < 当前 free → 拒绝——profile
        #   是快照不是保证）
        free_gpu_memory = profile_result.after_profile.free_memory
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        assert self.init_snapshot.free_memory >= free_gpu_memory, (
            "Error in memory profiling. "
            f"Initial free memory {format_gib(self.init_snapshot.free_memory)} GiB, "
            f"current free memory {format_gib(free_gpu_memory)} GiB. "
            "This happens when other processes sharing the same container "
            "release GPU memory while vLLM is profiling during initialization. "
            "To fix this, ensure consistent GPU memory allocation or "
            "isolate vLLM in its own container."
        )
        # SOURCE: vllm/v1/worker/gpu_worker.py:L544-L548 一行减法定出 KV 池
        #   本金：requested − non_kv − cudagraph_applied
        self.available_kv_cache_memory_bytes = (
            self.requested_memory
            - profile_result.non_kv_cache_memory
            - cudagraph_memory_estimate_applied
        )

        # SUBTRACTED: debug 账目与等价 util 提示日志（L550-L605——dossier.
        #   delete 第 11 条：教学金句正文引用、精简版省）。

        # SUBTRACTED: reserve_mm_ipc_gpu_memory 包装（L607-L611——第 3 条：
        #   mm IPC 预留；返回值语义 = 取 int）
        # SOURCE: vllm/v1/worker/gpu_worker.py:L607-L611（去包装后直返）
        return int(self.available_kv_cache_memory_bytes)

    # SOURCE: vllm/v1/worker/gpu_worker.py:L634 get_kv_cache_spec
    def get_kv_cache_spec(self) -> dict:
        # SOURCE: vllm/v1/worker/gpu_worker.py:L635
        return self.model_runner.get_kv_cache_spec()

    # SOURCE: vllm/v1/worker/gpu_worker.py:L637 update_max_model_len
    def update_max_model_len(self, max_model_len: int) -> None:
        """Update max_model_len after auto-fit to GPU memory.
        This is called when max_model_len=-1 is used and the engine
        automatically determines the maximum context length that fits
        in GPU memory. Workers need to update their cached max_model_len
        to match the engine's decision.
        """
        # SOURCE: vllm/v1/worker/gpu_worker.py:L644-L647（model_config 直写 +
        #   runner 缓存同步——worker 先于 profile 启动、缓存的是旧值）
        self.model_config.max_model_len = max_model_len
        if self.model_runner is not None:
            self.model_runner.update_max_model_len(max_model_len)

    # SUBTRACTED: get_kv_connector_handshake_metadata（L613-L632——ch16）。

    # SOURCE: vllm/v1/worker/gpu_worker.py:L256 _maybe_get_memory_pool_context
    def _maybe_get_memory_pool_context(self, tag: str):
        # SOURCE: vllm/v1/worker/gpu_worker.py:L257-L261（CUDA-like 且未开
        #   cumem → 无池上下文）
        if (
            current_platform.is_cuda_alike()
            and not self.vllm_config.model_config.enable_cumem_allocator
        ):
            return nullcontext()

        # SOURCE: vllm/v1/worker/gpu_worker.py:L263-L270（XPU/CPU → 无池）
        if (
            current_platform.is_xpu()
            and not self.vllm_config.model_config.enable_sleep_mode
        ):
            return nullcontext()

        if current_platform.is_cpu():
            return nullcontext()

        # SUBTRACTED: allocator = get_mem_allocator_instance(); return
        #   allocator.use_memory_pool(tag=tag)（L272-L276——CuMem/XpuMem 池
        #   的真分配器接线（vllm/device_allocator/cumem.py:L313 use_memory_
        #   pool / xpumem.py:L275）——sleep/offload 域（ch16/19），本章只
        #   消费 tag="kv_cache" 的调用位；host 上不可达（is_cpu 先返））。
        raise RuntimeError("device allocator pool unavailable on this slice")

    # SOURCE: vllm/v1/worker/gpu_worker.py:L649 initialize_from_config
    def initialize_from_config(self, kv_cache_config) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""

        # Update local config with adjusted num blocks after profiling,
        # so that it's available to the warmup stage.
        # SOURCE: vllm/v1/worker/gpu_worker.py:L654-L655（num_gpu_blocks 写回
        #   worker 侧 cache_config——启动后全局可见的账）
        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks

        # SUBTRACTED: ensure_kv_transfer_initialized（L657-L662——ch16：kv
        #   cache connector 装配，须在 initialize_kv_cache 前）。
        # SOURCE: vllm/v1/worker/gpu_worker.py:L664-L665（CuMem tag="kv_cache"
        #   池内真分配——与 weights 两池隔离）
        with self._maybe_get_memory_pool_context(tag="kv_cache"):
            self.model_runner.initialize_kv_cache(kv_cache_config)

        # SUBTRACTED: routed experts capturer（L667-L668——单模型特路）。
