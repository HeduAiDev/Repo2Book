"""NPUWorker：从抽象 WorkerBase 重写执行主控（subtract-only 忠实摘录）。

# SOURCE: vllm_ascend/worker/worker.py
# 文件头物证（原 L16-L17）：
#   # This file is a part of the vllm-ascend project.
#   # Adapted from vllm-project/vllm/vllm/worker/gpu_worker.py
# —— 结构改编自 gpu_worker.py，但每一处设备调用换成 torch_npu / ATB / ACLGraph。

四步走完一个 Worker 生命周期：
  (1) init_device              —— npu set_device + torch_npu._inductor + 显存快照 + hccl 分布式
  (2) determine_available_memory —— memory_profiling 量峰值 + ACLGraph 显存估算 + KV 预算/回退建议
  (3) compile_or_warm_up_model —— warmup_sizes 预热 + capture_model + _warm_up_atb（ATB 预热）
  (4) execute_model            —— 派发到 NPUModelRunner

host 无 NPU/CANN：真实 torch.npu.set_device / memory_profiling / ATB 预热不真跑；
其中『显存回退决策』是纯 Python 算术，注入桩后 determine_available_memory 可跑验数值
（见 tests/）。其余方法体保留可读控制流。
"""
import gc
from types import NoneType

# SUBTRACTED: import copy（原 worker.py:L20）——只被 execute_model 的 PP/KV transfer 分支
#   （copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)）使用，该分支已 SUBTRACTED。

import torch

# 本地 worker_base 摘录：证明 NPUWorker 派生的是抽象 WorkerBase，而非 GPU 的 Worker。
from worker_base import CompilationTimes, WorkerBase

# SUBTRACTED: 文件头 Apache-2.0 许可证 + ~50 行 import（原 worker.py:L1-L78）——torch_npu /
#   vllm.* / vllm_ascend.* 运行时依赖 host 不可用；方法体内引用的 memory_profiling /
#   MemorySnapshot / CUDAGraphMode / NPUModelRunner / init_workspace_manager /
#   init_device_properties_triton / get_ascend_config / get_pp_group / set_random_seed /
#   AscendDeviceType / GiB_bytes / envs_vllm 等名字，由 tests 按需注入（host 不真跑设备路径）。
# SUBTRACTED: from vllm.logger import logger → 用 stdlib logging 占位，保持 logger.* 调用原样。
import logging
logger = logging.getLogger(__name__)


# SOURCE: vllm_ascend/worker/worker.py:L81
class NPUWorker(WorkerBase):
    # SOURCE: vllm_ascend/worker/worker.py:L82-L130
    def __init__(
        self,
        vllm_config,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        # Additional parameters for compatibility with vllm
        **kwargs,
    ):
        """Initialize the worker for Ascend."""
        # SUBTRACTED: COMPILE_CUSTOM_KERNELS 缺失时的 warning（原 L93-L97）——提醒文案，不影响主线。

        # register patch for vllm
        from vllm_ascend.utils import adapt_patch  # noqa: F401

        adapt_patch()  # noqa: F821

        # Register ops when worker init.
        from vllm_ascend import ops  # noqa: F401

        ops.register_dummy_fusion_op()  # noqa: F821
        if get_ascend_device_type() != AscendDeviceType.A5:  # noqa: F821
            _register_atb_extensions()  # noqa: F821
        register_ascend_customop(vllm_config)  # noqa: F821
        # init ascend config and soc version
        init_ascend_config(vllm_config)  # noqa: F821
        check_ascend_device_type()  # noqa: F821

        # 复用 WorkerBase 把 vllm_config 摊开成各 config 字段、记 local_rank/rank、
        # device 与 model_runner 置空——与 GPU 的 Worker 走的是同一段公共逻辑。
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        if self.cache_config.cache_dtype == "auto":
            self.cache_dtype = self.model_config.dtype
        else:
            self.cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[self.cache_config.cache_dtype]  # noqa: F821

        # Profiler is lazily initialized on first profile(is_start=True) call (RFC #6954)
        # 横切点：profiler 类型是 TorchNPUProfilerWrapper（torch_npu profiler 的薄包装），点名不展开。
        self.profiler_config = vllm_config.profiler_config
        self.profiler = None
        # SUBTRACTED: self.torch_reserved/torch_allocated/npugraph_memory_bytes 计数初始化（原 L131-133）
        #   ——分别服务 profile_memory / compile 诊断 log，二者本身已 SUBTRACTED。
        # SUBTRACTED: enable_sleep_mode 的 _sleep_saved_buffers 缓冲 dict（原 L134-136）——sleep mode 旁支。
        # SUBTRACTED: WEIGHT_LOADER_V2_SUPPORTED 的 FixMe 补丁（原 L138-142）——上游兼容性补丁。
        # SUBTRACTED: use_v2_model_runner 回退判断 + _pp_send_work 初始化（原 L144-148）
        #   ——v2 runner 开发中（主线走 v1）；_pp_send_work 服务 PP 收发（已 SUBTRACTED）。
        # SUBTRACTED: enable_npugraph_ex + static_kernel 的 SIGTERM/SIGINT 信号处理（原 L150-166）
        #   ——npugraph_ex 进程退出清理，与执行主控正确性无关。

    # SUBTRACTED: uninstall_static_kernel（原 L168-198）——static kernel 卸载，进程退出清理。
    # SUBTRACTED: sleep / wake_up（原 L200-254）——sleep mode（RL 场景显存腾挪）独立特性，不在四步主线。

    # SOURCE: vllm_ascend/worker/worker.py:L256-L258
    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    # ===== 第 1 步：设备层全换 =====
    # SOURCE: vllm_ascend/worker/worker.py:L260-L315
    def _init_device(self):
        device = torch.device(f"npu:{self.local_rank}")
        torch.npu.set_device(device)  # 对位基座 cuda:N / set_device_index

        # Import _inductor for graph mode execution with triton.
        # This lazy import avoids torch_npu re-initialization in patch. Note that
        # this should be imported AFTER torch.npu.set_device to avoid repeated
        # set_device in extra processes.
        from vllm.triton_utils import HAS_TRITON  # noqa: F401

        if HAS_TRITON:  # noqa: F821
            import torch_npu._inductor  # noqa: F401  —— 编译栈换成 torch_npu._inductor（triton graph 模式）

        gc.collect()
        torch.npu.empty_cache()

        # SUBTRACTED: A5 设备的 setup_ascend_local_comm_res 分支（原 L276-277）——特定 SoC。

        # take current memory snapshot —— 与基座最大差异：昇腾先拍快照、后初始化分布式，
        # 故 HCCL 通信 buffer 不计入快照基线（基座相反，NCCL buffer 计入基线，见 gpu_worker.py）。
        self.init_snapshot = MemorySnapshot()  # noqa: F821
        self.requested_memory = self.init_snapshot.total_memory * self.cache_config.gpu_memory_utilization
        if self.init_snapshot.free_memory < self.requested_memory:
            GiB = lambda b: round(b / GiB_bytes, 2)  # noqa: F821
            raise ValueError(
                # SUBTRACTED: 报错文案细节（原 L285-291）——启动空闲显存不足 gpu_memory_utilization 要求。
                f"Free memory on device ({GiB(self.init_snapshot.free_memory)} GiB) on startup "
                f"is less than desired GPU memory utilization."
            )

        # SUBTRACTED: data_parallel 下 visible_device_count 断言块（原 L294-306）——多卡 DP 环境校验。

        # Initialize the distributed environment.（HCCL，在快照之后才分配通信 buffer）
        self._init_worker_distributed_environment()
        # Set random seed.
        set_random_seed(self.model_config.seed)  # noqa: F821
        # Initialize device properties used by triton kernels.（给 triton kernel 喂昇腾设备属性）
        init_device_properties_triton()  # noqa: F821

        return device

    # SOURCE: vllm_ascend/worker/worker.py:L317-L332
    def init_device(self):
        # NOTE: KEEP device the member of `NPUWorker`, as it will be checked in
        # ray scenario. see https://github.com/vllm-project/vllm/pull/26845
        self.device = self._init_device()
        # Initialize workspace manager（昇腾 num_ubatches=1；基座 enable_dbo 时为 2）
        num_ubatches = 1
        init_workspace_manager(self.device, num_ubatches)  # noqa: F821
        # Init ModelRunner here, so that we have access to self.device.
        # SUBTRACTED: use_v2_model_runner 分支下 NPUModelRunnerV2 的 import（原 L326-330）——v2 开发中。
        self.model_runner = NPUModelRunner(self.vllm_config, self.device)  # noqa: F821

    # ===== 第 2 步：显存语义全换 + 复用基座算法骨架 =====
    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        # SOURCE: vllm_ascend/worker/worker.py:L334-L462
        """Profiles the peak memory usage of the model to determine how much
        memory can be used for KV cache without OOMs.
        """
        GiB = lambda b: b / GiB_bytes  # noqa: F821

        # SUBTRACTED: fast path —— 用户用 --kv-cache-memory 显式指定 KV 大小时只 profile_run
        #   编译模型、跳过显存估算直接返回（原 L345-361）。主线走自动 profiling。

        # Execute a forward pass with dummy inputs to profile the memory usage.
        with memory_profiling(  # noqa: F821
            self.init_snapshot,
            weights_memory=int(self.model_runner.model_memory_usage),
        ) as profile_result:
            self.model_runner.profile_run()

            # Record torch peak INSIDE the context and BEFORE graph capture, so
            # that graph pool allocations don't inflate the activation peak.
            profile_torch_peak = torch.npu.memory_stats(self.device).get("allocated_bytes.all.peak", 0)

            npugraph_memory_estimate = 0
            should_profile_npugraph_memory = (
                self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE  # noqa: F821
            )
            # SUBTRACTED: DeepSeek-V4 DSA 压缩注意力跳过 ACLGraph 显存估算的特例（原 L379-388）。
            if should_profile_npugraph_memory:
                npugraph_memory_estimate = self.model_runner.profile_cudagraph_memory()

        # Override torch_peak_increase with the pre-graph-capture value to avoid
        # double-counting graph pool memory as activation memory.
        profile_result.torch_peak_increase = profile_torch_peak - profile_result.before_profile.torch_peak
        profile_result.non_kv_cache_memory = (
            profile_result.non_torch_increase + profile_result.torch_peak_increase + profile_result.weights_memory
        )

        npugraph_memory_estimate_applied = (
            npugraph_memory_estimate if envs_vllm.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS else 0  # noqa: F821
        )

        # Save per-category memory for use in compile_or_warm_up_model() (step 3).
        self.peak_activation_memory = profile_result.torch_peak_increase
        self.non_torch_memory = profile_result.non_torch_increase
        self.npugraph_memory_estimate = npugraph_memory_estimate

        free_gpu_memory = profile_result.after_profile.free_memory
        assert self.init_snapshot.free_memory > free_gpu_memory, (
            "Error in memory profiling. "  # SUBTRACTED: 报错文案细节（原 L411-416）。
        )
        # KV cache 能吃多少显存 = requested − 非KV显存 − ACLGraph显存估算(受开关控制是否扣)。
        self.available_kv_cache_memory_bytes = (
            self.requested_memory - profile_result.non_kv_cache_memory - npugraph_memory_estimate_applied
        )

        logger.info("Available KV cache memory: %.2f GiB", GiB(self.available_kv_cache_memory_bytes))

        if npugraph_memory_estimate > 0:
            # 回退建议（纯 Python 算术）：ACLGraph 占总显存比例 delta = npugraph/total，
            # 建议把 --gpu-memory-utilization 提到 current+delta（封顶 1.0）以补回被 ACLGraph 占走的 KV。
            total_mem = self.init_snapshot.total_memory
            current_util = self.cache_config.gpu_memory_utilization
            ng_util_delta = npugraph_memory_estimate / total_mem
            suggested_util = min(
                round(current_util + ng_util_delta, 4),
                1.0,
            )
            if envs_vllm.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:  # noqa: F821
                equiv_util = round(current_util - ng_util_delta, 4)
                # SUBTRACTED: 完整提示文案（原 L438-448）——保留算式即可看懂回退建议如何算。
                logger.info("Increase --gpu-memory-utilization to %.4f", suggested_util)
            else:
                # SUBTRACTED: 完整 warning 文案（原 L450-460）。
                logger.warning("Increase --gpu-memory-utilization from %.4f to %.4f", current_util, suggested_util)

        return int(self.available_kv_cache_memory_bytes)

    # SUBTRACTED: profile_memory（原 L464-472）——execute_model() 里 torch 显存观测，删后仍能正确派发。

    # ===== 第 4 步：执行派发 =====
    # SOURCE: vllm_ascend/worker/worker.py:L474-L538
    def execute_model(
        self,
        scheduler_output,
    ):
        # SUBTRACTED: self.profile_memory() + msMonitor dp.step()（原 L478-481）——设备侧观测点。
        # SUBTRACTED: self._pp_send_work 完成等待（原 L483-486）——PP 发送握手，多卡旁支。

        intermediate_tensors = None
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0
        # SUBTRACTED: 非首 PP rank 接收 intermediate tensors 的 all_gather_group/irecv（原 L490-505）。

        if self.profiler is not None:
            self.profiler.step()

        # 真正的前向在 NPUModelRunner（留后续 ModelRunner 章）。单机最常路径就是这一调一返。
        output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)
        if isinstance(output, (ModelRunnerOutput, AsyncModelRunnerOutput, NoneType)):  # noqa: F821
            return output
        # SUBTRACTED: output 为 IntermediateTensors 时的 PP isend 转发 + kv_connector_output 透传
        #   （原 L514-538）——PP/KV transfer 多卡旁支，单机主线 forward_pass 必走上面的 return。

    # SUBTRACTED: sample_tokens（原 L540-542）——转发 model_runner.sample_tokens。

    # SOURCE: vllm_ascend/worker/worker.py:L544-L555
    def load_model(self) -> None:
        # SUBTRACTED: enable_sleep_mode 下 CaMemAllocator.use_memory_pool(tag="weights") 分支
        #   （原 L545-548）——sleep mode 内存池旁支，主线走 nullcontext 直路。
        from contextlib import nullcontext

        context = nullcontext()

        with context, set_current_vllm_config(self.vllm_config):  # noqa: F821
            self.model_runner.load_model()

    # ===== 第 3 步：编译预热 =====
    # SOURCE: vllm_ascend/worker/worker.py:L557-L659
    def compile_or_warm_up_model(self) -> CompilationTimes:
        # Note: need to adapt for graph mode.
        warmup_sizes = (self.vllm_config.compilation_config.compile_sizes or []).copy()
        if not self.model_config.enforce_eager:
            cg_capture_sizes: list[int] = []
            if self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:  # noqa: F821
                cg_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
                cg_capture_sizes = [] if cg_sizes is None else cg_sizes
                warmup_sizes = [x for x in warmup_sizes if x not in cg_capture_sizes]

            compile_ranges = self.vllm_config.compilation_config.get_compile_ranges()
            # For each compile_range, if none of the batch sizes in warmup_sizes or
            # cudagraph_capture_sizes are in the range, add the end of the range to
            # ensure compilation/warmup.
            all_sizes = set(cg_capture_sizes)
            all_sizes.update([x for x in warmup_sizes if isinstance(x, int)])
            for compile_range in compile_ranges:
                if not any(x in compile_range for x in all_sizes):
                    warmup_sizes.append(compile_range.end)

        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size)

        npugraph_memory_bytes = 0
        if not self.model_config.enforce_eager:
            npugraph_memory_bytes = self.model_runner.capture_model()  # 捕获 NPU/ACL graph，返回其显存占用

        # SUBTRACTED: ACLGraph 实测 vs 估算对比 log（原 L585-595）——诊断输出，不改控制流。
        # SUBTRACTED: --kv-cache-memory 建议 log（原 L597-634）——诊断输出，不改控制流。
        _ = npugraph_memory_bytes  # 上述被删 log 的唯一消费者；保留赋值以示其来源

        # Call ATB matmul to warm up; otherwise the first operation (ReshapeAndCache)
        # may cause performance degradation at runtime.（基座此处是 kernel_warmup，昇腾换成 _warm_up_atb）
        if get_ascend_device_type() != AscendDeviceType.A5:  # noqa: F821
            self._warm_up_atb()
        # SUBTRACTED: enable_cpu_binding 时 bind_cpus 的 try/except（原 L642-646）——NUMA 绑核优化。
        # Reset the seed to ensure that the random state is not affected by the
        # model initialization and profiling.
        set_random_seed(self.model_config.seed)  # noqa: F821
        return CompilationTimes(
            language_model=self.vllm_config.compilation_config.compilation_time,
            # `encoder_compilation_time` 为新版本字段，用 getattr 兜底兼容老版本 vLLM。
            encoder=getattr(
                self.vllm_config.compilation_config,
                "encoder_compilation_time",
                0.0,
            ),
        )

    # SOURCE: vllm_ascend/worker/worker.py:L661-L665
    def _warm_up_atb(self):
        # 昇腾特有：打一发 ATB 的 matmul_add，把 ATB 算子库首次初始化开销提前付掉，
        # 否则运行期第一个算子（ReshapeAndCache）会卡。
        x = torch.rand((2, 4), dtype=torch.float16).npu()
        weight = torch.rand((2, 4), dtype=torch.float16).npu()
        c = torch.rand((4, 4), dtype=torch.float32).npu()
        torch_npu._npu_matmul_add_fp32(x, weight, c)  # noqa: F821

    # SUBTRACTED: get_model / profile_prefill_latency / get_kv_connector_handshake_metadata /
    #   get_kv_cache_spec / update_max_model_len / profile / add_lora / remove_lora /
    #   list_loras / pin_lora / reset_encoder_cache / execute_dummy_batch /
    #   get_supported_(pooling_)tasks / take_draft_token_ids / check_health 等转发型/旁支方法
    #   （原 L667-883）——多为薄薄一行转发给 model_runner，不构成四步主线。

    # SOURCE: vllm_ascend/worker/worker.py:L836-L849
    def _init_worker_distributed_environment(self) -> None:
        """Initialize the distributed environment."""
        init_batch_invariance()  # noqa: F821
        # 后端是 "hccl"（昇腾），对位基座的 nccl —— 设备层重写的又一处物证。
        init_distributed_environment(  # noqa: F821
            self.parallel_config.world_size, self.rank, self.distributed_init_method, self.local_rank, "hccl"
        )
        ensure_model_parallel_initialized(  # noqa: F821
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_size,
            self.parallel_config.prefill_context_parallel_size,
            self.parallel_config.decode_context_parallel_size,
        )
        init_ascend_model_parallel(self.parallel_config)  # noqa: F821
        ensure_ec_transfer_initialized(self.vllm_config)  # noqa: F821

    # SOURCE: vllm_ascend/worker/worker.py:L762-L785
    def initialize_from_config(self, kv_cache_config) -> None:
        """Allocate NPU KV cache with the specified kv_cache_config."""
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)  # noqa: F821
        # SUBTRACTED: enable_sleep_mode 下 CaMemAllocator.use_memory_pool(tag="kv_cache") 分支
        #   （原 L765-767）——sleep mode 内存池旁支，主线走 nullcontext。
        from contextlib import nullcontext

        context = nullcontext()
        with context:
            self.model_runner.initialize_kv_cache(kv_cache_config)
            # SUBTRACTED: needs_kv_cache_zeroing 下 _init_kv_zero_meta 构建（原 L775-785）
            #   ——投机解码 KV-zero 元数据，旁支。
