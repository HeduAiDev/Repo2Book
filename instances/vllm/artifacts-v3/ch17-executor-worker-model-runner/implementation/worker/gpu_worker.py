# Subtract-only companion for v3 ch17 — vllm/v1/worker/gpu_worker.py (pin
# v0.27.1 / 6e448d0ea). Same names, same structure, same control flow; only
# dossier-approved deletions, each marked `# SUBTRACTED:`.
#
# Deletions here (dossier subtraction_plan.delete + embed_excerpts elide 注):
#   #4  VLLM_ELASTIC_EP_SCALE_UP_LAUNCH 分支与 elastic_ep_executor 装配
#       （L149-L151、L645-L648 的调用位同族）、@instrument 装饰；
#   #5  Worker 生产特性面：sleep/wake_up/checkpoint_prepare/checkpoint_restore/
#       _get_sleep_mode_backend（L181-L254）、weight_transfer 相关（L159-L162、
#       L444-L450、L1207-L1313）、profiler/profile/annotate_profile（L164-L172、
#       L899-L1008、L1112-L1163）、worker_sentinel/handle_ft_command（L152-L154、
#       L429-L431）、SP/enable_sp 分支（L1035-L1062）、update_config/
#       reload_weights/update_max_model_len（L452-L457、L637-L647）、
#       _set_draft_weight_update_target（L867-L885）；
#   #7  compile_or_warm_up_model 内部：compile_ranges 补边界（L695-L703）、
#       cudagraph 估算对比日志（L719-L733）、startup_plan 建议与落盘
#       （L735-L791）、V2 warmup_kernels 分支（L793-L795）、pooling 分支
#       （L813-L814）；
#   （dossier embed_excerpts elide 注授权的小裁剪：assigned_physical_gpu_ids 块
#       整体压一行（L328-L357）、workspace manager（L404-L406）与 usage stats
#       （L425-L427）删。）
#   determine_available_memory 的账本主体（L472-L611）按 must_keep 的
#   『骨架保留，账本细节 ch14』裁为注释占位。

from __future__ import annotations

import gc
import os
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager, nullcontext
from datetime import timedelta
from types import NoneType
from typing import TYPE_CHECKING, Any

import regex as re
import torch
import torch.nn as nn

from .._host_seams import (
    CUDAGraphMode,
    GrammarOutput,
    IntermediateTensors,
    KVConnectorHandshakeMetadata,
    KVCacheConfig,
    KVCacheSpec,
    MemorySnapshot,
    SchedulerOutput,
    SupportedTask,
    TensorizerLoader,
    current_platform,
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_ec_transfer_initialized,
    ensure_ec_transfer_shutdown,
    ensure_kv_transfer_initialized,
    ensure_kv_transfer_shutdown,
    format_gib,
    get_kv_transfer_group,
    get_mem_allocator_instance,
    get_pp_group,
    get_tp_group,
    has_kv_transfer_group,
    init_batch_invariance,
    init_logger,
    kernel_warmup,
    override_envs_for_eplb,
    request_memory,
    set_current_vllm_config,
)
from .._host_seams import (
    CompilationMode,
    ModelRunnerOutput,
    AsyncModelRunnerOutput,
    VllmConfig,
    activate_jit_monitor,
    envs,
)
from ..utils.gc_utils import freeze_gc_heap, maybe_attach_gc_debug_callback
from ..utils.gpu_sync_debug import enable_gpu_sync_check, with_gpu_sync_check
from .worker_base import CompilationTimes, WorkerBase
from .._host_seams import (
    init_distributed_environment,
    ensure_model_parallel_initialized,
    set_custom_all_reduce,
)

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


# SOURCE: vllm/v1/worker/gpu_worker.py:L96-L125 AsyncIntermediateTensors — PP 懒同步
class AsyncIntermediateTensors(IntermediateTensors):
    """IntermediateTensors with lazy comm synchronization"""

    # SOURCE: vllm/v1/worker/gpu_worker.py:L99-L108 __init__
    def __init__(
        self,
        tensors: dict,
        comm_handles: list | None = None,
        comm_postprocess: list[Callable[[], None]] | None = None,
    ) -> None:
        super().__init__(tensors)
        self._comm_handles = comm_handles
        self._comm_postprocess = comm_postprocess
        self._comm_waited = False

    # SOURCE: vllm/v1/worker/gpu_worker.py:L110-L119 wait_for_comm
    def wait_for_comm(self) -> None:
        if self._comm_waited:
            return
        if self._comm_handles:
            for handle in self._comm_handles:
                handle.wait()
        if self._comm_postprocess:
            for fn in self._comm_postprocess:
                fn()
        self._comm_waited = True

    # SOURCE: vllm/v1/worker/gpu_worker.py:L121-L125 __getattribute__ 懒拦截
    def __getattribute__(self, name: str):
        # ensure `.tensors` is ready before use
        if name == "tensors" and not object.__getattribute__(self, "_comm_waited"):
            object.__getattribute__(self, "wait_for_comm")()
        return object.__getattribute__(self, name)


# SOURCE: vllm/v1/worker/gpu_worker.py:L128 Worker — 执行臂第二层的 GPU 实现
class Worker(WorkerBase):
    # SOURCE: vllm/v1/worker/gpu_worker.py:L129-L179 Worker.__init__
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        # configure float32 matmul precision according to vLLM env.
        precision = envs.VLLM_FLOAT32_MATMUL_PRECISION
        torch.set_float32_matmul_precision(precision)

        # SUBTRACTED: elastic_ep_executor 装配（gpu_worker.py:L149-L151——弹性
        #   EP 执行器；其唯一调用面 elastic_ep_execute 与 VLLM_ELASTIC_EP_
        #   SCALE_UP_LAUNCH 分支同属删除项 4/5）。
        # SUBTRACTED: worker_sentinel / enable_fault_tolerance（L152-L154——FT
        #   哨兵，删除项 5）。
        # SUBTRACTED: sleep 模式的缓冲保存字典（L155-L157——sleep/wake_up
        #   生产特性，删除项 5）。
        # SUBTRACTED: weight_transfer_engine 字段与开关（L159-L162——RL 权重
        #   热更，删除项 5）。
        # SUBTRACTED: profiler 包装与配置校验（L164-L172——删除项 5）。

        self.use_v2_model_runner = vllm_config.use_v2_model_runner
        # pending non-blocking PP send work from the previous iteration
        self._pp_send_work: list = []

        # SUBTRACTED: _sleep_mode_backend 惰性后端字段（L178-L179——删除项 5，
        #   其消费面 sleep/wake_up/_get_sleep_mode_backend 一并删除）。

    # SUBTRACTED: _get_sleep_mode_backend / sleep / wake_up /
    #   checkpoint_prepare / checkpoint_restore（gpu_worker.py:L181-L254——
    #   CuMem 池挂起恢复与检查点分布式状态，删除项 5）。

    # SOURCE: vllm/v1/worker/gpu_worker.py:L256-L277 _maybe_get_memory_pool_context
    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        if (
            current_platform.is_cuda_alike()
            and not self.vllm_config.model_config.enable_cumem_allocator
        ):
            return nullcontext()

        if (
            current_platform.is_xpu()
            and not self.vllm_config.model_config.enable_sleep_mode
        ):
            return nullcontext()

        if current_platform.is_cpu():
            return nullcontext()

        allocator = get_mem_allocator_instance()
        if tag == "weights":
            assert allocator.get_current_usage() == 0, (
                "CuMem allocator can only be used for one instance per process."
            )
        return allocator.use_memory_pool(tag=tag)
    @contextmanager
    # SOURCE: vllm/v1/worker/gpu_worker.py:L279-L301 _scoped_allocator_max_split
    def _scoped_allocator_max_split(self, max_split_size_mb: int):
        """Temporarily set max_split_size_mb to reduce allocator fragmentation at the
        cost of more cudaMalloc calls (negligible in practice). Restores the original
        value on exit."""
        if not current_platform.is_cuda():
            yield
            return

        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        match = re.search(r"max_split_size_mb:(\d+)", conf)
        original_value = match.group(1) if match else None

        torch._C._accelerator_setAllocatorSettings(
            f"max_split_size_mb:{max_split_size_mb}"
        )
        try:
            yield
        finally:
            # PyTorch defaults to SIZE_MAX (no limit).
            _SIZE_MAX_MB = (2**64 - 1) // (1024 * 1024)
            restore = original_value if original_value else str(_SIZE_MAX_MB)
            torch._C._accelerator_setAllocatorSettings(f"max_split_size_mb:{restore}")

    # SUBTRACTED: @instrument(span_name="Init device") 观测装饰（删除项 4）。
    # SOURCE: vllm/v1/worker/gpu_worker.py:L303-L427 init_device
    def init_device(self):
        if self.device_config.device_type == "cuda":
            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            parallel_config = self.parallel_config
            if (
                parallel_config.distributed_executor_backend
                not in ("ray", "external_launcher")
                and parallel_config.data_parallel_backend != "ray"
                and parallel_config.nnodes_within_dp == 1
            ):
                # Use local DP rank if available, otherwise use global DP rank.
                dp_local_rank = self.parallel_config.data_parallel_rank_local
                if dp_local_rank is None:
                    dp_local_rank = self.parallel_config.data_parallel_index

                tp_pp_world_size = (
                    self.parallel_config.pipeline_parallel_size
                    * self.parallel_config.tensor_parallel_size
                )

                # DP_LOCAL_RANK * TP_PP_WORLD_SIZE + TP_LOCAL_RANK
                self.local_rank += dp_local_rank * tp_pp_world_size

            # SUBTRACTED: assigned_physical_gpu_ids 发布与双段校验
            #   （gpu_worker.py:L328-L357——逻辑卡号到物理卡号的拓扑映射，
            #   elide 注授权压除；其 else 分支的设备数校验保留）。
            assert self.local_rank < torch.accelerator.device_count(), (
                f"DP adjusted local rank {self.local_rank} is out of "
                f"bounds for {torch.accelerator.device_count()} devices."
            )

            visible_device_index = (
                current_platform.logical_device_id_to_visible_device_id(self.local_rank)
            )
            self.device = torch.device(f"cuda:{visible_device_index}")
            torch.accelerator.set_device_index(self.device)

            current_platform.check_if_supports_dtype(self.model_config.dtype)

            # Initialize the distributed environment BEFORE taking
            # memory snapshot
            # This ensures NCCL buffers are allocated before we measure
            # available memory
            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
                current_platform.dist_backend,
            )

            if self.use_v2_model_runner:
                logger.info_once("Using V2 Model Runner")

            # Set random seed.
            set_random_seed(self.model_config.seed)

            # Now take memory snapshot after NCCL is initialized
            gc.collect()
            torch.accelerator.empty_cache()

            # take current memory snapshot
            self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)
            self.requested_memory = request_memory(init_snapshot, self.cache_config)
            logger.debug("worker init memory snapshot: %r", self.init_snapshot)
            logger.debug(
                "worker requested memory: %sGiB", format_gib(self.requested_memory)
            )
        else:
            raise RuntimeError(f"Unsupported device type: {self.device_config.device}")

        # SUBTRACTED: workspace manager 初始化（gpu_worker.py:L404-L406——
        #   num_ubatches 的 DBO 工作区，elide 注授权删除）。

        # Construct the model runner
        if self.use_v2_model_runner:
            from vllm.v1.worker.gpu.model_runner import (
                GPUModelRunner as GPUModelRunnerV2,
            )

            # HACK(woosuk): This is a temporary fix to avoid type errors.
            self.model_runner: GPUModelRunner = GPUModelRunnerV2(  # type: ignore
                self.vllm_config, self.device
            )
        else:
            from vllm.v1.worker.gpu_model_runner import (
                GPUModelRunner as GPUModelRunnerV1,
            )

            self.model_runner = GPUModelRunnerV1(self.vllm_config, self.device)

        # SUBTRACTED: rank 0 的 usage stats 上报（gpu_worker.py:L425-L427——
        #   elide 注授权删除）。

    # SUBTRACTED: handle_ft_command（gpu_worker.py:L429-L431——FT 哨兵命令面，
    #   删除项 5）。

    # FIXME(youkaichao & ywang96): Use TorchDispatchMode instead of memory pool
    # to hijack tensor allocation.
    # SOURCE: vllm/v1/worker/gpu_worker.py:L435-L450 load_model — 三锚点之一
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        with (
            self._maybe_get_memory_pool_context(tag="weights"),
            set_current_vllm_config(self.vllm_config),
            # 20 MiB is the minimum PyTorch allows for max_split_size_mb.
            self._scoped_allocator_max_split(max_split_size_mb=20),
        ):
            self.model_runner.load_model(load_dummy_weights=load_dummy_weights)

        # SUBTRACTED: weight_transfer_engine 创建（gpu_worker.py:L444-L450——
        #   RL 权重热更引擎，删除项 5）。

    # SUBTRACTED: update_config / reload_weights（gpu_worker.py:L452-L457——
    #   运行期微调接口，删除项 5）。
    @torch.inference_mode()
    # SOURCE: vllm/v1/worker/gpu_worker.py:L459-L611 determine_available_memory
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
        # SUBTRACTED: profile 账本主体（gpu_worker.py:L472-L611——
        #   maybe_apply_startup_plan、kv_cache_memory_bytes 直给分支、
        #   memory_profiling 上下文 + profile_run、cudagraph 显存估算与
        #   建议日志、available_kv_cache_memory_bytes 记账与返回值装配）——
        #   ch14 的精简版持有这本账；本章只认三锚点的归属：这一步在 Worker
        #   生命周期里的位置（executor.determine_available_memory →
        #   collective_rpc → 每个 worker 的此方法）与『返回可用于 KV cache
        #   的字节数』契约。宿主替身返回 0（HOST SEAM，账本在 ch14）。
        return 0  # HOST SEAM: profile 定出的可用 KV 字节（ch14 账本）

    # SOURCE: vllm/v1/worker/gpu_worker.py:L613-L632 get_kv_connector_handshake_metadata
    def get_kv_connector_handshake_metadata(
        self,
    ) -> dict[tuple[int, int], KVConnectorHandshakeMetadata] | None:
        """Get KV connector metadata from this worker if available.

        Returned dict is keyed by `(pp_rank, tp_rank)`.
        """

        if not has_kv_transfer_group():
            return None

        connector = get_kv_transfer_group()
        # Return None for connectors that don't need to exchange handshake
        # metadata across workers.
        if (metadata := connector.get_handshake_metadata()) is None:
            return None

        pp_rank = get_pp_group().rank_in_group
        tp_rank = get_tp_group().rank_in_group
        return {(pp_rank, tp_rank): metadata}

    # SOURCE: vllm/v1/worker/gpu_worker.py:L634-L635 get_kv_cache_spec
    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    # SUBTRACTED: update_max_model_len（gpu_worker.py:L637-L647——max_model_len=-1
    #   自适应后的 worker 侧同步，删除项 5）。

    # SUBTRACTED: @instrument(span_name="Allocate KV cache") 观测装饰（删除项 4）。
    # SOURCE: vllm/v1/worker/gpu_worker.py:L649-L676 initialize_from_config
    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""

        # Update local config with adjusted num blocks after profiling,
        # so that it's available to the warmup stage.
        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks

        # Init kv cache connector here, because it requires
        # `kv_cache_config`.
        # NOTE(Kuntai): This need to be done before `initialize_kv_cache`,
        # because `initialize_kv_cache` will inject kv cache groups not
        # related to kv cache connector (e.g. kv cache sharing layers).
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)

        with self._maybe_get_memory_pool_context(tag="kv_cache"):
            self.model_runner.initialize_kv_cache(kv_cache_config)

        if self.model_config.enable_return_routed_experts:
            self.model_runner.init_routed_experts_capturer()

        # Build KV-zero metadata outside the CuMem pool so the bookkeeping
        # GPU tensors (seg_addrs, block-id buffers) use the standard PyTorch
        # allocator and are not discarded during sleep/wake cycles.
        if kv_cache_config.needs_kv_cache_zeroing and hasattr(
            self.model_runner, "_init_kv_zero_meta"
        ):
            self.model_runner._init_kv_zero_meta()

    # SUBTRACTED: @instrument(span_name="Warmup (GPU)") 观测装饰（删除项 4）。
    # SOURCE: vllm/v1/worker/gpu_worker.py:L678-L853 compile_or_warm_up_model
    def compile_or_warm_up_model(self) -> CompilationTimes:
        warmup_sizes: list[int] = []

        if self.vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE:
            # warm up sizes that are not in cudagraph capture sizes,
            # but users still want to compile for better performance,
            # e.g. for the max-num-batched token size in chunked prefill.
            compile_sizes = self.vllm_config.compilation_config.compile_sizes
            warmup_sizes = compile_sizes.copy() if compile_sizes is not None else []  # type: ignore[assignment]
            cg_capture_sizes: list[int] = []

            if self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
                cg_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
                cg_capture_sizes = [] if cg_sizes is None else cg_sizes
                warmup_sizes = [x for x in warmup_sizes if x not in cg_capture_sizes]

            # SUBTRACTED: compile_ranges 区间补边界（gpu_worker.py:L695-L703——
            #   每个 compile_range 若无任何 size 落入则补区间端点，删除项 7）。

        # We skip EPLB here since we don't want to record dummy metrics
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size, skip_eplb=True, remove_lora=False)
        self.model_runner.maybe_remove_all_loras(self.model_runner.lora_config)

        # Warmup and tune the kernels used during model execution before
        # cuda graph capture.
        kernel_warmup(self)

        cuda_graph_memory_bytes = 0
        if not self.model_config.enforce_eager:
            cuda_graph_memory_bytes = self.model_runner.capture_model()

        # SUBTRACTED: cudagraph 实测 vs 估算对比日志（gpu_worker.py:L719-L733，
        #   删除项 7）。
        # SUBTRACTED: startup_plan 建议与落盘（gpu_worker.py:L735-L791——
        #   kv-cache-memory 建议文案与 maybe_save_startup_plan，删除项 7）。

        # SUBTRACTED: V2 warmup_kernels 分支（gpu_worker.py:L793-L795——V2 跑
        #   完整 execute_model+sample_tokens 以 JIT 编译 triton kernel，
        #   删除项 7）。
        if get_pp_group().is_last_rank:
            # V1: Warm up sampler and preallocate memory buffer for logits and other
            # sampling related tensors of max possible shape to avoid memory
            # fragmentation issue.
            # NOTE: This is called after `capture_model` on purpose to prevent
            # memory buffers from being cleared by `torch.accelerator.empty_cache`.
            max_num_reqs = min(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens,
            )

            # We skip EPLB here since we don't want to record dummy metrics
            hidden_states, last_hidden_states = self.model_runner._dummy_run(
                num_tokens=max_num_reqs,
                skip_eplb=True,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            # SUBTRACTED: pooling 分支（gpu_worker.py:L813-L814——pooling 模型
            #   的 _dummy_pooler_run，删除项 7）。
            self.model_runner._dummy_sampler_run(hidden_states=last_hidden_states)

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

        # Eagerly trigger inductor's once-per-process lazy inits during
        # warmup (rather than on a later compile cache-miss at runtime).
        c_config = self.compilation_config
        if c_config.mode != CompilationMode.NONE and c_config.backend == "inductor":
            from vllm.compilation.compiler_interface import (
                trigger_inductor_lazy_init,
            )

            trigger_inductor_lazy_init(self.device)

        # All warmup is done — start monitoring for unexpected JIT
        # compilations that would cause latency spikes during inference.
        from .._host_seams import activate_jit_monitor

        activate_jit_monitor(
            mode=self.observability_config.jit_monitor_mode,
            verbose=self.observability_config.jit_monitor_verbose,
        )

        # Freeze the worker heap so the GC won't scan static objects
        # (model weights, KV caches, CUDA graphs) during inference.
        freeze_gc_heap()
        maybe_attach_gc_debug_callback()

        # Warmup / first-compile is done — activate the `VLLM_GPU_SYNC_CHECK`
        # gate so subsequent `execute_model` / `sample_tokens` calls enforce it.
        enable_gpu_sync_check()

        return CompilationTimes(
            language_model=self.compilation_config.compilation_time,
            encoder=self.compilation_config.encoder_compilation_time,
        )

    # SOURCE: vllm/v1/worker/gpu_worker.py:L855-L856 reset_mm_cache
    def reset_mm_cache(self) -> None:
        self.model_runner.reset_mm_cache()

    # SOURCE: vllm/v1/worker/gpu_worker.py:L858-L859 reset_encoder_cache
    def reset_encoder_cache(self) -> None:
        self.model_runner.reset_encoder_cache()

    # SOURCE: vllm/v1/worker/gpu_worker.py:L861-L862 get_model
    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    # SOURCE: vllm/v1/worker/gpu_worker.py:L864-L865 get_draft_model
    def get_draft_model(self) -> nn.Module | None:
        return self.model_runner.get_draft_model()

    # SUBTRACTED: _set_draft_weight_update_target（gpu_worker.py:L867-L885——
    #   权重热更的 draft 目标设定，删除项 5）。

    # SOURCE: vllm/v1/worker/gpu_worker.py:L887-L888 get_supported_tasks
    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

    # SOURCE: vllm/v1/worker/gpu_worker.py:L890-L893 get_compilation_match_table
    def get_compilation_match_table(self) -> dict[str, int]:
        from vllm.compilation.passes.vllm_inductor_pass import get_match_table

        return get_match_table()

    # SOURCE: vllm/v1/worker/gpu_worker.py:L895-L897 get_encoder_timing_stats
    def get_encoder_timing_stats(self) -> dict[str, dict[str, float | int]]:
        """Get encoder timing stats from model runner."""
        return self.model_runner.get_encoder_timing_stats()

    # SUBTRACTED: annotate_profile（gpu_worker.py:L899-L1008——profiler 的迭代
    #   注解上下文，删除项 5；execute_model 里的 `with self.annotate_profile(...)`
    #   包装随之拆除）。
    @torch.inference_mode()
    @with_gpu_sync_check
    # SOURCE: vllm/v1/worker/gpu_worker.py:L1010-L1015 Worker.sample_tokens
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        return self.model_runner.sample_tokens(grammar_output)
    @torch.inference_mode()
    @with_gpu_sync_check
    # SOURCE: vllm/v1/worker/gpu_worker.py:L1017-L1107 Worker.execute_model
    def execute_model(
        self, scheduler_output: "SchedulerOutput"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        # ensure any previous non-blocking PP sends are complete
        if self._pp_send_work:
            for handle in self._pp_send_work:
                handle.wait()
            self._pp_send_work = []

        intermediate_tensors = None
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0
        all_gather_tensors = {}
        compilation_config = self.vllm_config.compilation_config
        parallel_config = self.vllm_config.parallel_config

        # SUBTRACTED: enable_sp 序列并行分支（gpu_worker.py:L1035-L1062——
        #   PP>1 且 pass_config.enable_sp 时的 batch padding 判定与
        #   all_gather_tensors 组装，删除项 5）。

        if forward_pass and not get_pp_group().is_first_rank:
            tensor_dict, comm_handles, comm_postprocess = (
                get_pp_group().irecv_tensor_dict(
                    all_gather_group=get_tp_group(),
                    all_gather_tensors=all_gather_tensors,
                )
            )
            assert tensor_dict is not None
            intermediate_tensors = AsyncIntermediateTensors(
                tensor_dict,
                comm_handles=comm_handles,
                comm_postprocess=comm_postprocess,
            )

        output = self.model_runner.execute_model(
            scheduler_output, intermediate_tensors
        )
        if (
            self.use_v2_model_runner
            and self.model_runner.is_pooling_model
            and output is None
        ):
            output = self.model_runner.pool()  # type: ignore
        if isinstance(
            output, ModelRunnerOutput | AsyncModelRunnerOutput | NoneType
        ):
            return output

        assert isinstance(output, IntermediateTensors)
        parallel_config = self.vllm_config.parallel_config
        assert (
            parallel_config.distributed_executor_backend != "external_launcher"
            and not get_pp_group().is_last_rank
        )

        # launch non-blocking send of intermediate tensors
        self._pp_send_work = get_pp_group().isend_tensor_dict(
            output.tensors,
            all_gather_group=get_tp_group(),
            all_gather_tensors=all_gather_tensors,
        )

        return None

    # SOURCE: vllm/v1/worker/gpu_worker.py:L1109-L1110 take_draft_token_ids
    def take_draft_token_ids(self) -> Any:
        return self.model_runner.take_draft_token_ids()

    # SUBTRACTED: profile（gpu_worker.py:L1112-L1163——torch/CUDA profiler 的
    #   启停编排，删除项 5）。

    # SOURCE: vllm/v1/worker/gpu_worker.py:L1165-L1167 execute_dummy_batch
    def execute_dummy_batch(self) -> None:
        num_tokens = getattr(self.model_runner, "uniform_decode_query_len", 1)
        self.model_runner._dummy_run(num_tokens, uniform_decode=True)

    # SOURCE: vllm/v1/worker/gpu_worker.py:L1169-L1170 add_lora
    def add_lora(self, lora_request) -> bool:
        return self.model_runner.add_lora(lora_request)

    # SOURCE: vllm/v1/worker/gpu_worker.py:L1172-L1173 remove_lora
    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    # SOURCE: vllm/v1/worker/gpu_worker.py:L1175-L1176 list_loras
    def list_loras(self) -> set[int]:
        return self.model_runner.list_loras()

    # SOURCE: vllm/v1/worker/gpu_worker.py:L1178-L1179 pin_lora
    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    # SOURCE: vllm/v1/worker/gpu_worker.py:L1181-L1183 check_health
    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return

    # SOURCE: vllm/v1/worker/gpu_worker.py:L1185-L1198 save_sharded_state
    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        from vllm.model_executor.model_loader import ShardedStateLoader

        ShardedStateLoader.save_model(
            self.model_runner.model,
            path,
            pattern=pattern,
            max_size=max_size,
        )

    # SOURCE: vllm/v1/worker/gpu_worker.py:L1200-L1205 save_tensorized_model
    def save_tensorized_model(self, tensorizer_config) -> None:
        TensorizerLoader.save_model(
            self.get_model(),
            tensorizer_config=tensorizer_config,
            model_config=self.model_config,
        )

    # SUBTRACTED: 权重热更全家（gpu_worker.py:L1207-L1313——
    #   _check_weight_transfer_engine / init_weight_transfer_engine /
    #   start_weight_update / start_draft_weight_update / _start_weight_update /
    #   update_weights / finish_weight_update，删除项 5）。

    # SOURCE: vllm/v1/worker/gpu_worker.py:L1315-L1340 Worker.shutdown
    def shutdown(self) -> None:
        gc.unfreeze()

        # has_kv_transfer_group can be None during interpreter shutdown.
        if ensure_kv_transfer_shutdown is not None:
            ensure_kv_transfer_shutdown()
        if ensure_ec_transfer_shutdown is not None:
            ensure_ec_transfer_shutdown()
        # SUBTRACTED: profiler.shutdown()（gpu_worker.py:L1323-L1324——删除项 5）。
        # SUBTRACTED: weight_transfer_engine.shutdown()（L1326-L1327——删除项 5）。

        # Release GPU resources held by the model runner so that memory
        # can be reclaimed when running in-process
        if model_runner := getattr(self, "model_runner", None):
            model_runner.shutdown()

        # Release kept-alive cumem pools while the pluggable allocator wrappers
        # and callbacks are still alive, so MemPool teardown is not deferred to
        # interpreter finalization (pytorch/pytorch#145168).
        if current_platform.is_cuda_alike():
            from vllm.device_allocator.cumem import CuMemAllocator

            if CuMemAllocator.instance is not None:
                CuMemAllocator.instance.release_pools()

    # SUBTRACTED: elastic_ep_execute（gpu_worker.py:L1343-L1344——弹性 EP 执行
    #   器的转发面，删除项 4）。


# SOURCE: vllm/utils/torch_utils.py set_random_seed — HOST SEAM 简化（真实实现
# 同时重置 torch/numpy/random 的全局种子；本档只需可复现性）
# SOURCE: (见 impl-notes.md §Source Map——worker/gpu_worker.py)
def set_random_seed(seed: int) -> None:  # HOST SEAM
    import random

    if seed is not None:
        random.seed(seed)


# SOURCE: vllm/v1/worker/gpu_worker.py:L1346-L1389 init_worker_distributed_environment
def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: str | None = None,
    local_rank: int = -1,
    backend: str = "nccl",
) -> None:
    """Initialize the distributed environment."""
    parallel_config = vllm_config.parallel_config
    init_batch_invariance()
    override_envs_for_eplb(
        parallel_config,
        moe_backend=None,
    )
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_method = distributed_init_method or "env://"

    timeout = None
    if parallel_config.distributed_timeout_seconds is not None:
        timeout = timedelta(seconds=parallel_config.distributed_timeout_seconds)

    init_distributed_environment(
        parallel_config.world_size,
        rank,
        init_method,
        local_rank,
        backend,
        timeout,
    )

    ensure_model_parallel_initialized(
        parallel_config.tensor_parallel_size,
        parallel_config.pipeline_parallel_size,
        parallel_config.prefill_context_parallel_size,
        parallel_config.decode_context_parallel_size,
    )

    # Init ec connector here before KV caches init
    # NOTE: We do not init KV caches for Encoder-only instance in EPD disagg mode
    ensure_ec_transfer_initialized(vllm_config)
