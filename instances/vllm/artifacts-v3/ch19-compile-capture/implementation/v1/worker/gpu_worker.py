# Subtract-only companion for v3 ch19 — vllm/v1/worker/gpu_worker.py
# (pin v0.27.1 / 6e448d0ea). Same names, same structure, same control flow;
# only dossier-approved deletions (each marked `# SUBTRACTED`), plus 本章
# 切面之外的域段以「SUBTRACTED + 归属章」注记收窄（impl-notes §范围裁剪）。
#
# 本章切面：Worker.compile_or_warm_up_model 启动编排（L678-L853，站 6/m15/
# m18）——warmup → kernel 调优 → capture → sampler 预热 → inductor lazy
# init → JIT 纠察 → freeze GC → sync 纠察，编译/捕获/warmup/防退化全部
# 前移启动期、运行期零惊喜。Worker 其余（init/分布式装配/execute 面族，
# L128-L677 与 L855 起）是 ch17/ch34 域，SUBTRACTED。
#
# Deletions here (dossier subtraction_plan.delete #7):
#   - startup_plan 落盘与 KV 内存建议段（L719-L791——maybe_save_startup_plan
#     与 --kv-cache-memory 建议日志，观测/提示域）；
#   - use_v2_model_runner 分支（L793-L795——V2 runner 实验态，全书锚定 V1）；
#   - @instrument 装饰（L678——tracing 观测域）。
# HOST 注记：activate_jit_monitor 真源为函数内 lazy import（L834）；伴读版
#   提升为模块级导入（HOST SEAM jit_monitor.activate 的 no-op 面，控制流不变）。
from __future__ import annotations

from ..._host_seams import (
    activate_jit_monitor,
    get_pp_group,
    init_logger,
    kernel_warmup,
    set_random_seed,
)
from ...compilation.compiler_interface import trigger_inductor_lazy_init
from ...config import CUDAGraphMode
from ...config.compilation import CompilationMode
from ...utils.gc_utils import freeze_gc_heap, maybe_attach_gc_debug_callback
from ...utils.gpu_sync_debug import enable_gpu_sync_check
from .worker_base import CompilationTimes

logger = init_logger(__name__)


# SOURCE: vllm/v1/worker/gpu_worker.py:L128-L136 Worker 类头（WorkerBase
#   控制面随 ch17 域 SUBTRACTED）
class Worker:
    # SUBTRACTED: __init__ 与 init 面族（L129-L677——分布式装配/模型加载/
    #   显存 profiling，ch17/ch34/ch14 域）。

    # SUBTRACTED: @instrument(span_name="Warmup (GPU)")（L678——tracing
    #   观测域）。
    def compile_or_warm_up_model(self) -> CompilationTimes:  # SOURCE: vllm/v1/worker/gpu_worker.py:L679-L853
        # SOURCE: vllm/v1/worker/gpu_worker.py:L680-L703 —— warmup 尺寸表：
        #   compile_sizes 减去 cg 捕获尺寸 + compile_ranges 补区间尾
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

            compile_ranges = self.vllm_config.compilation_config.get_compile_ranges()
            # For each compile_range, if none of the batch sizes
            # in warmup_sizes or cudagraph_capture_sizes are in the range,
            # add the end of the range to ensure compilation/warmup.
            all_sizes = set(cg_capture_sizes)
            all_sizes.update([x for x in warmup_sizes if isinstance(x, int)])
            for compile_range in compile_ranges:
                if not any(x in compile_range for x in all_sizes):
                    warmup_sizes.append(compile_range.end)

        # SOURCE: vllm/v1/worker/gpu_worker.py:L705-L713 —— 首个 dummy 前向
        #   （从大到小）触发 Dynamo trace 与切图编译 → kernel_warmup 调优
        # We skip EPLB here since we don't want to record dummy metrics
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size, skip_eplb=True, remove_lora=False)
        self.model_runner.maybe_remove_all_loras(self.model_runner.lora_config)

        # Warmup and tune the kernels used during model execution before
        # cuda graph capture.
        kernel_warmup(self)

        # SOURCE: vllm/v1/worker/gpu_worker.py:L715-L717 —— 捕获（eager 模式
        #   enforce_eager 跳过）
        cuda_graph_memory_bytes = 0
        if not self.model_config.enforce_eager:
            cuda_graph_memory_bytes = self.model_runner.capture_model()

        # SUBTRACTED: 图池 estimate 对比日志与 KV 内存建议/startup_plan 落盘
        #   段（L719-L791——delete[7]：maybe_save_startup_plan 属 ch17 启动计划
        #   域；--kv-cache-memory 建议是提示观测）。

        # SUBTRACTED: use_v2_model_runner 分支（L793-L795——delete[7]：V2
        #   runner 实验态，全书锚定 V1）。
        # SOURCE: vllm/v1/worker/gpu_worker.py:L796-L816 —— V1 收尾：末 PP 段
        #   sampler 预热（NOTE：刻意在 capture_model 之后，防 empty_cache 清掉
        #   logits/采样缓冲）
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
            if self.model_runner.is_pooling_model:
                self.model_runner._dummy_pooler_run(hidden_states)
            else:
                self.model_runner._dummy_sampler_run(hidden_states=last_hidden_states)

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        # SOURCE: vllm/v1/worker/gpu_worker.py:L818-L820
        set_random_seed(self.model_config.seed)

        # Eagerly trigger inductor's once-per-process lazy inits during
        # warmup (rather than on a later compile cache-miss at runtime).
        # SOURCE: vllm/v1/worker/gpu_worker.py:L822-L830
        c_config = self.compilation_config
        if c_config.mode != CompilationMode.NONE and c_config.backend == "inductor":
            trigger_inductor_lazy_init(self.device)

        # All warmup is done — start monitoring for unexpected JIT
        # compilations that would cause latency spikes during inference.
        # SOURCE: vllm/v1/worker/gpu_worker.py:L832-L839（真源为函数内 lazy
        #   import；伴读版提升为模块级 HOST SEAM 面，控制流不变）
        activate_jit_monitor(
            mode=self.observability_config.jit_monitor_mode,
            verbose=self.observability_config.jit_monitor_verbose,
        )

        # Freeze the worker heap so the GC won't scan static objects
        # (model weights, KV caches, CUDA graphs) during inference.
        # SOURCE: vllm/v1/worker/gpu_worker.py:L841-L844
        freeze_gc_heap()
        maybe_attach_gc_debug_callback()

        # Warmup / first-compile is done — activate the `VLLM_GPU_SYNC_CHECK`
        # gate so subsequent `execute_model` / `sample_tokens` calls enforce it.
        # SOURCE: vllm/v1/worker/gpu_worker.py:L846-L848
        enable_gpu_sync_check()

        # SOURCE: vllm/v1/worker/gpu_worker.py:L850-L853 —— 启动耗时回传
        return CompilationTimes(
            language_model=self.compilation_config.compilation_time,
            encoder=self.compilation_config.encoder_compilation_time,
        )

    # SUBTRACTED: execute_model/sample_tokens/reset_* 面族（L855 起——ch17
    #   执行域）。
