# HOST SEAMS for the ch17 subtract-only companion (pin vLLM v0.27.1).
#
# vLLM itself does not install on this Windows host, so every vllm.* name the
# kept (subtract-only) code touches is mirrored here by a stdlib/torch-backed
# stand-in with the SAME observable interface subset. Each seam carries a
# `# SOURCE:` anchor into the pinned tree; none of them invents behavior the
# real module does not have on the paths this chapter exercises. The full
# inventory is registered in implementation/../impl-notes.md §Seam 清单.
#
# Companion files import these via package-relative imports (the ch10 v3
# convention): real `from vllm.x import y` becomes `from .._host_seams import y`
# when x is outside this chapter's subtract-only surface.

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import logging
import sys
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# logger — vllm/logger.py init_logger + the *_once wrappers
# ---------------------------------------------------------------------------


# SOURCE: vllm/logger.py init_logger — logging seam with the *_once helpers
def init_logger(name: str):
    log = logging.getLogger(name)
    if not log.handlers:
        log.addHandler(logging.NullHandler())
    seen: set[str] = set()

    # SOURCE: vllm/logger.py once-messaging wrapper (info_once/warning_once)
    class _Once:  # HOST SEAM
        # SOURCE: vllm/logger.py once-wrapper construction
        def __init__(self, fn):
            # HOST SEAM (real: functools-based dedup in vllm.logger)
            self._fn = fn

        # SOURCE: vllm/logger.py once-wrapper call
        def __call__(self, msg, *args):
            key = (self._fn.__name__, msg)
            if key not in seen:
                seen.add(key)
                self._fn(msg, *args)

    log.info_once = _Once(log.info)
    log.warning_once = _Once(log.warning)
    log.debug_once = _Once(log.debug)
    return log


# ---------------------------------------------------------------------------
# envs — vllm/envs.py flag seam (defaults per pin v0.27.1)
# ---------------------------------------------------------------------------


# SOURCE: vllm/envs.py:L226-L228 MQ / timeout flags — HOST SEAM (class attribute
# stand-in for the env-backed descriptor table; defaults are the pin's).
class envs:  # HOST SEAM
    # SOURCE: vllm/envs.py:L226 VLLM_MQ_MAX_CHUNK_BYTES_MB
    VLLM_MQ_MAX_CHUNK_BYTES_MB: int = 16
    # SOURCE: vllm/envs.py:L227 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS: int = 300
    # SOURCE: vllm/envs.py:L228 VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS
    VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS: int = 5
    # SOURCE: vllm/envs.py:L93 VLLM_GPU_SYNC_CHECK (None | "warn" | "error")
    VLLM_GPU_SYNC_CHECK = None
    # SOURCE: vllm/envs.py VLLM_WORKER_MULTIPROC_METHOD (fork on unix hosts)
    VLLM_WORKER_MULTIPROC_METHOD: str = "fork"
    # SOURCE: vllm/envs.py VLLM_FLOAT32_MATMUL_PRECISION (default "high")
    VLLM_FLOAT32_MATMUL_PRECISION: str = "high"
    # SOURCE: vllm/envs.py VLLM_GC_DEBUG
    VLLM_GC_DEBUG: str = ""


# SOURCE: vllm/envs.py enable_envs_cache — freeze the env table after worker
# bring-up; the real table is a frozen dict, here a no-op flag.
# SOURCE: (见 impl-notes.md §Source Map——_host_seams.py)
def enable_envs_cache() -> None:  # HOST SEAM
    return None


# ---------------------------------------------------------------------------
# current_platform — vllm/platforms/interface.py platform seam
# ---------------------------------------------------------------------------


# SOURCE: vllm/platforms/interface.py Platform — HOST SEAM: the interface
# subset the kept code touches. The CUDA platform is the chapter's narrative
# axis (platforms/cuda.py resolves worker_cls); device-op predicates report
# the host's reality (no CUDA control plane in the companion's tests).
class _PlatformSeam:  # HOST SEAM
    # SOURCE: vllm/platforms/cuda.py dist_backend — CUDA platform uses NCCL
    dist_backend = "nccl"

    # SOURCE: vllm/platforms/interface.py Platform.is_cuda_alike
    def is_cuda_alike(self) -> bool:
        # HOST SEAM: 控制面教学校本——宿主不走 CUDA 设备分支（真实语义按平台而定）
        return False

    # SOURCE: vllm/platforms/interface.py Platform.is_cuda
    def is_cuda(self) -> bool:
        return False

    # SOURCE: vllm/platforms/interface.py Platform.is_xpu
    def is_xpu(self) -> bool:
        return False

    # SOURCE: vllm/platforms/interface.py Platform.is_cpu
    def is_cpu(self) -> bool:
        return True

    # SOURCE: vllm/platforms/interface.py Platform.set_device
    def set_device(self, device) -> None:
        return None

    # SOURCE: vllm/platforms/interface.py Platform.update_block_size_for_backend
    def update_block_size_for_backend(self, vllm_config) -> None:
        return None

    # SOURCE: vllm/platforms/cuda.py Platform.check_if_supports_dtype
    def check_if_supports_dtype(self, dtype) -> None:
        return None

    # SOURCE: vllm/platforms/cuda.py Platform.logical_device_id_to_visible_device_id
    def logical_device_id_to_visible_device_id(self, local_rank: int) -> int:
        return local_rank


current_platform = _PlatformSeam()


# ---------------------------------------------------------------------------
# config carriers — vllm/config/* HOST SEAM (the real assembly line is ch03's)
# ---------------------------------------------------------------------------


# SOURCE: vllm/config/compilation.py CompilationMode — HOST SEAM enum subset
class CompilationMode(Enum):  # HOST SEAM
    NONE = "none"
    VLLM_COMPILE = "vllm"


# SOURCE: vllm/config/compilation.py CUDAGraphMode — HOST SEAM enum subset
class CUDAGraphMode(Enum):  # HOST SEAM
    NONE = "NONE"
    PIECEWISE = "PIECEWISE"
    FULL = "FULL"

@dataclass
# SOURCE: vllm/config/compilation.py PassConfig.enable_sp — HOST SEAM carrier
class _PassConfig:  # HOST SEAM
    enable_sp: bool = False


# SOURCE: vllm/config/compilation.py CompilationConfig — HOST SEAM field subset
@dataclass
class CompilationConfig:  # HOST SEAM
    mode: CompilationMode = CompilationMode.NONE
    backend: str = "inductor"
    compile_sizes: list[int] | None = None
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE
    cudagraph_capture_sizes: list[int] | None = None
    compilation_time: float = 0.0
    encoder_compilation_time: float = 0.0
    ir_enable_torch_wrap: bool = False
    pass_config: _PassConfig = field(default_factory=_PassConfig)

    # SOURCE: vllm/config/compilation.py CompilationConfig.get_compile_ranges
    def get_compile_ranges(self):  # HOST SEAM: 空表——区间补边界逻辑已按删除项 7 裁除
        return []

@dataclass
# SOURCE: vllm/config/parallel.py ParallelConfig — HOST SEAM field subset
class ParallelConfig:  # HOST SEAM
    world_size: int = 1
    local_world_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    prefill_context_parallel_size: int = 1
    decode_context_parallel_size: int = 1
    nnodes_within_dp: int = 1
    node_rank_within_dp: int = 0
    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    distributed_executor_backend: str | type = "uni"
    worker_cls: str | type = "auto"
    worker_extension_cls: str | None = None
    data_parallel_rank_local: int | None = None
    data_parallel_index: int = 0
    data_parallel_backend: str = "mp"
    assigned_physical_gpu_ids: list[int] | None = None
    enable_expert_parallel: bool = False
    distributed_timeout_seconds: int | None = None
    disable_custom_all_reduce: bool = True
    enable_dbo: bool = False
    rank: int = 0


# SOURCE: vllm/config/model.py ModelConfig — HOST SEAM field subset
@dataclass
class ModelConfig:  # HOST SEAM
    dtype: str = "float16"
    seed: int | None = 0
    max_model_len: int | None = None
    multimodal_config = None
    enable_return_routed_experts: bool = False
    enforce_eager: bool = False
    enable_cumem_allocator: bool = False
    enable_sleep_mode: bool = False

    # SOURCE: vllm/config/model.py ModelConfig.get_vocab_size
    def get_vocab_size(self) -> int:  # HOST SEAM
        return 0

@dataclass
# SOURCE: vllm/config/cache.py CacheConfig — HOST SEAM field subset
class CacheConfig:  # HOST SEAM
    gpu_memory_utilization: float = 0.9
    kv_cache_memory_bytes: int | None = None
    num_gpu_blocks: int | None = None

@dataclass
# SOURCE: vllm/config/scheduler.py SchedulerConfig — HOST SEAM field subset
class SchedulerConfig:  # HOST SEAM
    async_scheduling: bool = False
    max_num_seqs: int = 128
    max_num_batched_tokens: int = 2048


# SOURCE: vllm/config/device.py DeviceConfig — HOST SEAM field subset
@dataclass
class DeviceConfig:  # HOST SEAM
    device_type: str = "cuda"
    device = None

    # SOURCE: vllm/config/device.py DeviceConfig.__post_init__ — device 回填
    def __post_init__(self):  # HOST SEAM（真实版按平台解析 torch.device）
        if self.device is None:
            self.device = self.device_type

@dataclass
# SOURCE: vllm/config/observability.py ObservabilityConfig — HOST SEAM subset
class ObservabilityConfig:  # HOST SEAM
    jit_monitor_mode = None
    jit_monitor_verbose: bool = False


# SOURCE: vllm/config/kernel.py KernelConfig.ir_op_priority — HOST SEAM carrier
class _IROpPriority:  # HOST SEAM
    # SOURCE: vllm/config/kernel.py IROpPriority.set_default
    def set_default(self) -> None:
        return None

@dataclass
# SOURCE: vllm/config/kernel.py KernelConfig — HOST SEAM field subset
class KernelConfig:  # HOST SEAM
    ir_op_priority: _IROpPriority = field(default_factory=_IROpPriority)


# SOURCE: vllm/config/vllm.py VllmConfig — HOST SEAM field subset（装配线归 ch03）
@dataclass
class VllmConfig:  # HOST SEAM
    model_config: ModelConfig = field(default_factory=ModelConfig)
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    lora_config = None
    load_config = None
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    device_config: DeviceConfig = field(default_factory=DeviceConfig)
    speculative_config = None
    observability_config: ObservabilityConfig = field(
        default_factory=ObservabilityConfig
    )
    compilation_config: CompilationConfig = field(default_factory=CompilationConfig)
    kv_transfer_config = None
    weight_transfer_config = None
    profiler_config = None
    kernel_config: KernelConfig = field(default_factory=KernelConfig)
    use_v2_model_runner: bool = False

    # SOURCE: vllm/config/vllm.py VllmConfig.enable_trace_function_call_for_thread
    def enable_trace_function_call_for_thread(self) -> None:  # HOST SEAM
        return None


# SOURCE: vllm/config/__init__.py set_current_vllm_config — context manager
# publishing the config for worker/device init; seam is a nullcontext.
@contextlib.contextmanager
# SOURCE: (见 impl-notes.md §Source Map——_host_seams.py)
def set_current_vllm_config(vllm_config):  # HOST SEAM
    yield


# SOURCE: vllm/ir.set_default_torch_wrap — IR 编译轴（ch19 域）的宿主替身
def set_default_torch_wrap(enabled: bool) -> None:  # HOST SEAM
    return None


class ir:  # HOST SEAM namespace for `import vllm.ir`
    # SOURCE: vllm/ir/__init__.py set_default_torch_wrap
    set_default_torch_wrap = staticmethod(set_default_torch_wrap)


# ---------------------------------------------------------------------------
# schedule/output carriers — vllm/v1/core/sched/output.py HOST SEAM
# ---------------------------------------------------------------------------

@dataclass
# SOURCE: vllm/v1/core/sched/output.py:L68+ NewRequestData — field subset
class NewRequestData:  # HOST SEAM
    req_id: str
    num_computed_tokens: int = 0
    mm_features = None

@dataclass
# SOURCE: vllm/v1/core/sched/output.py:L108+ CachedRequestData — field subset
class CachedRequestData:  # HOST SEAM
    req_ids: list[str] = field(default_factory=list)
    num_computed_tokens: list[int] = field(default_factory=list)

@dataclass
# SOURCE: vllm/v1/core/sched/output.py:L193-L233 SchedulerOutput — carrier
class SchedulerOutput:  # HOST SEAM
    scheduled_new_reqs: list[NewRequestData] = field(default_factory=list)
    scheduled_cached_reqs: CachedRequestData = field(default_factory=CachedRequestData)
    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)
    total_num_scheduled_tokens: int = 0
    scheduled_spec_decode_tokens: dict[str, list[int]] = field(default_factory=dict)
    scheduled_encoder_inputs: dict[str, list[int]] = field(default_factory=dict)
    num_common_prefix_blocks: list[int] = field(default_factory=list)
    finished_req_ids: set[str] = field(default_factory=set)
    free_encoder_mm_hashes: list[str] = field(default_factory=list)

@dataclass
# SOURCE: vllm/v1/core/sched/output.py:L287-L291 GrammarOutput — carrier
class GrammarOutput:  # HOST SEAM
    structured_output_request_ids: list[str] = field(default_factory=list)
    grammar_bitmask: "object | None" = None  # npt.NDArray[np.int32]

@dataclass
# SOURCE: vllm/v1/kv_cache_interface.py KVCacheConfig — HOST SEAM subset
class KVCacheConfig:  # HOST SEAM
    num_blocks: int = 0
    needs_kv_cache_zeroing: bool = False


# SOURCE: vllm/v1/kv_cache_interface.py KVCacheSpec — opaque carrier
class KVCacheSpec:  # HOST SEAM
    pass


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py
#   KVConnectorHandshakeMetadata — opaque carrier
# SOURCE: (见 impl-notes.md §Source Map——_host_seams.py)
class KVConnectorHandshakeMetadata:  # HOST SEAM
    pass


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py KVConnectorOutput
@dataclass
class KVConnectorOutput:  # HOST SEAM
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py is_empty
    def is_empty(self) -> bool:  # HOST SEAM
        return True

@dataclass
# SOURCE: vllm/lora/request.py LoRARequest — carrier for the LoRA RPC faces
class LoRARequest:  # HOST SEAM
    lora_name: str = ""
    lora_int_id: int = 0
    lora_path = None


# SOURCE: vllm/v1/engine/__init__.py ReconfigureDistributedRequest — carrier
class ReconfigureDistributedRequest:  # HOST SEAM
    pass


# SOURCE: vllm/tasks.py SupportedTask — opaque carrier
class SupportedTask:  # HOST SEAM
    pass


# ---------------------------------------------------------------------------
# outputs — vllm/v1/outputs.py
# ---------------------------------------------------------------------------


# SOURCE: vllm/v1/outputs.py:L261-L331 ModelRunnerOutput — HOST SEAM field
# subset（ch08/ch29 持有 logprobs/采样细节；本档只要两段式的返回形状）
@dataclass
class ModelRunnerOutput:  # HOST SEAM
    req_ids: list[str] = field(default_factory=list)
    req_id_to_index: dict[str, int] = field(default_factory=dict)
    sampled_token_ids: list[list[int]] = field(default_factory=list)
    kv_connector_output: object | None = None
    @staticmethod
    # SOURCE: vllm/v1/outputs.py:L311-L323 with_kv_conn_output_only
    def with_kv_conn_output_only(
        kv_connector_output: object | None,
    ) -> "ModelRunnerOutput":
        """Return ModelRunnerOutput containing the provided KVConnectorOutput,
        otherwise empty. Returns None if kv_connector_output is passed as None.
        """
        if kv_connector_output is None or kv_connector_output.is_empty():
            return EMPTY_MODEL_RUNNER_OUTPUT
        output = _copy_empty_output()
        output.kv_connector_output = kv_connector_output
        return output


# SOURCE: vllm/v1/outputs.py EMPTY_MODEL_RUNNER_OUTPUT — module-level singleton
EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput()


# SOURCE: vllm/copy — copy.copy of the empty singleton (outputs.py uses copy())
def _copy_empty_output() -> ModelRunnerOutput:  # HOST SEAM
    out = ModelRunnerOutput()
    out.req_ids = list(EMPTY_MODEL_RUNNER_OUTPUT.req_ids)
    out.req_id_to_index = dict(EMPTY_MODEL_RUNNER_OUTPUT.req_id_to_index)
    return out


# SOURCE: vllm/v1/outputs.py:L325-L334 AsyncModelRunnerOutput — verbatim ABC
class AsyncModelRunnerOutput(ABC):
    @abstractmethod
    # SOURCE: vllm/v1/outputs.py:L326-L333 get_output abstract method
    def get_output(self) -> ModelRunnerOutput:
        """Get the ModelRunnerOutput for this async output.

        This is a blocking call that waits until the results are ready, which
        might involve copying device tensors to the host.
        This method should only be called once per AsyncModelRunnerOutput.
        """
        pass

@dataclass
# SOURCE: vllm/v1/outputs.py:L338-L343 DraftTokenIds — verbatim carrier
class DraftTokenIds:
    # [num_reqs]
    req_ids: list[str]
    # num_reqs x num_draft_tokens
    draft_token_ids: list[list[int]]


# ---------------------------------------------------------------------------
# sequence — vllm/sequence.py IntermediateTensors（PP 中间张量载体，ch34 域的
# 数据结构本体；本档以真实实现子集承载 AsyncIntermediateTensors）
# ---------------------------------------------------------------------------


# SOURCE: vllm/sequence.py:L11-L49 IntermediateTensors — 逐字子集
@dataclass
class IntermediateTensors:
    """For all pipeline stages except the last, we need to return the hidden
    states and residuals to be sent to the next stage. This data structure
    contains the hidden states and residuals for a request.
    """

    tensors: dict

    # SOURCE: vllm/sequence.py:L20-L28 __init__
    def __init__(
        self,
        tensors: dict,
    ) -> None:
        # manually define this function, so that
        # Dynamo knows `IntermediateTensors()` comes from this file.
        # Otherwise, dataclass will generate this function by evaluating
        # a string, and we will lose the information about the source file.
        self.tensors = tensors

    # SOURCE: vllm/sequence.py:L30-L34 __getitem__
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.tensors[key]
        elif isinstance(key, slice):
            return self.__class__({k: v[key] for k, v in self.tensors.items()})

    # SOURCE: vllm/sequence.py:L36-L37 __setitem__
    def __setitem__(self, key: str, value):
        self.tensors[key] = value

    # SOURCE: vllm/sequence.py:L39-L40 items
    def items(self):
        return self.tensors.items()

    # SOURCE: vllm/sequence.py:L42-L43 __len__
    def __len__(self):
        return len(self.tensors)


# ---------------------------------------------------------------------------
# distributed stand-ins — vllm/distributed/* HOST SEAM (ch34 owns the real thing)
# ---------------------------------------------------------------------------


# SOURCE: vllm/distributed/parallel_state.py GroupCoordinator — HOST SEAM: the
# interface subset kept code touches (world_size / is_first_rank / is_last_rank
# / tensor-dict send-recv faces raise into ch34's territory).
class _GroupSeam:  # HOST SEAM
    # SOURCE: (见 impl-notes.md §Source Map——_host_seams.py)
    def __init__(self, world_size: int = 1, rank: int = 0):
        self.world_size = world_size
        self.rank = rank
        self.ranks = list(range(world_size))
        self.is_first_rank = True
        self.is_last_rank = True

    # SOURCE: vllm/distributed/parallel_state.py irecv_tensor_dict — ch34 域
    def irecv_tensor_dict(self, **kwargs):  # HOST SEAM (structural hole -> ch34)
        raise NotImplementedError(
            "Pipeline-parallel transport is ch34's domain (pp group seam)"
        )

    # SOURCE: vllm/distributed/parallel_state.py isend_tensor_dict — ch34 域
    def isend_tensor_dict(self, tensors, **kwargs):  # HOST SEAM (-> ch34)
        raise NotImplementedError(
            "Pipeline-parallel transport is ch34's domain (pp group seam)"
        )

    # SOURCE: vllm/distributed/parallel_state.py broadcast_tensor_dict — ch34 域
    def broadcast_tensor_dict(self, dic, **kwargs):  # HOST SEAM (-> ch34)
        return dic


_TP_GROUP = _GroupSeam()
_PP_GROUP = _GroupSeam()
_DIST_ENV: dict = {}


# SOURCE: vllm/distributed/parallel_state.py get_tp_group
def get_tp_group() -> _GroupSeam:  # HOST SEAM
    return _TP_GROUP


# SOURCE: vllm/distributed/parallel_state.py get_pp_group
def get_pp_group() -> _GroupSeam:  # HOST SEAM
    return _PP_GROUP


# SOURCE: vllm/distributed/parallel_state.py has_kv_transfer_group
def has_kv_transfer_group() -> bool:  # HOST SEAM
    return False


# SOURCE: vllm/distributed/parallel_state.py get_kv_transfer_group
def get_kv_transfer_group():  # HOST SEAM
    raise NotImplementedError("KV transfer group seam: ch16/ch36 domain")


# SOURCE: vllm/distributed/__init__.py init_distributed_environment — HOST SEAM:
# 记录 world/rank/local_rank/backend（真实 NCCL/gloo 初始化归 ch34）。
# SOURCE: (见 impl-notes.md §Source Map——_host_seams.py)
def init_distributed_environment(
    world_size: int, rank: int, distributed_init_method, local_rank: int,
    backend: str, timeout=None,
) -> None:  # HOST SEAM
    _DIST_ENV.update(
        world_size=world_size,
        rank=rank,
        init_method=distributed_init_method,
        local_rank=local_rank,
        backend=backend,
    )


# SOURCE: vllm/distributed/__init__.py ensure_model_parallel_initialized —
# HOST SEAM: 初始化 TP/PP 组的单机退化形态（world=tp*pp）。
# SOURCE: (见 impl-notes.md §Source Map——_host_seams.py)
def ensure_model_parallel_initialized(tp: int, pp: int, pcp: int, dcp: int) -> None:
    global _TP_GROUP, _PP_GROUP
    _TP_GROUP = _GroupSeam(world_size=tp, rank=0)
    _PP_GROUP = _GroupSeam(world_size=pp, rank=pp - 1)
    _PP_GROUP.is_first_rank = pp == 1
    _PP_GROUP.is_last_rank = True


# SOURCE: vllm/distributed/__init__.py set_custom_all_reduce
def set_custom_all_reduce(enabled: bool) -> None:  # HOST SEAM
    return None


# SOURCE: vllm/distributed/__init__.py destroy_model_parallel
def destroy_model_parallel() -> None:  # HOST SEAM
    return None


# SOURCE: vllm/distributed/__init__.py destroy_distributed_environment
def destroy_distributed_environment() -> None:  # HOST SEAM
    return None


# SOURCE: vllm/distributed/kv_transfer/__init__.py ensure_kv_transfer_initialized
def ensure_kv_transfer_initialized(vllm_config, kv_cache_config) -> None:  # HOST SEAM
    return None


# SOURCE: vllm/distributed/kv_transfer/__init__.py ensure_kv_transfer_shutdown
def ensure_kv_transfer_shutdown() -> None:  # HOST SEAM
    return None


# SOURCE: vllm/distributed/ec_transfer.py ensure_ec_transfer_initialized
def ensure_ec_transfer_initialized(vllm_config) -> None:  # HOST SEAM
    return None


# SOURCE: vllm/distributed/ec_transfer.py ensure_ec_transfer_shutdown
def ensure_ec_transfer_shutdown() -> None:  # HOST SEAM
    return None


# SOURCE: vllm/model_executor/layers/batch_invariant.py init_batch_invariance
def init_batch_invariance() -> None:  # HOST SEAM
    return None


# SOURCE: vllm/distributed/eplb/eplb_utils.py override_envs_for_eplb
def override_envs_for_eplb(parallel_config, **kwargs) -> None:  # HOST SEAM
    return None


# ---------------------------------------------------------------------------
# memory faces — vllm/utils/mem_utils.py HOST SEAM (ch14 owns the real ledger)
# ---------------------------------------------------------------------------

@dataclass
# SOURCE: vllm/utils/mem_utils.py MemorySnapshot — HOST SEAM field subset
class MemorySnapshot:  # HOST SEAM
    device = None
    before_total: int = 0
    before_free: int = 0

    @property
    # SOURCE: (见 impl-notes.md §Source Map——_host_seams.py)
    def free_memory(self) -> int:  # HOST SEAM
        return self.before_free

    @property
    # SOURCE: (见 impl-notes.md §Source Map——_host_seams.py)
    def total_memory(self) -> int:  # HOST SEAM
        return self.before_total

    # SOURCE: (见 impl-notes.md §Source Map——_host_seams.py)
    def __repr__(self) -> str:  # HOST SEAM
        return "MemorySnapshot(seam)"


# SOURCE: vllm/utils/mem_utils.py request_memory — HOST SEAM: util * total 的
# 近似（真实公式按 free/total 与 weight 占比精算，归 ch14 账本）。
# SOURCE: (见 impl-notes.md §Source Map——_host_seams.py)
def request_memory(snapshot: MemorySnapshot, cache_config) -> int:  # HOST SEAM
    return 0


# SOURCE: vllm/utils/mem_utils.py format_gib — HOST SEAM
def format_gib(num_bytes: float) -> str:  # HOST SEAM
    gib = num_bytes / (1 << 30)
    return f"{gib:.2f}GiB"


# ---------------------------------------------------------------------------
# CuMem allocator — vllm/device_allocator HOST SEAM (tag 纪录仪)
# ---------------------------------------------------------------------------


# SOURCE: vllm/device_allocator/__init__.py get_mem_allocator_instance —
# HOST SEAM: 记录池 tag 的进出（weights / kv_cache 两锚点可观察），不做真实
# 显存池管理（sleep/wake 生产特性已按删除项 5 裁除）。
class _CuMemAllocatorSeam:  # HOST SEAM
    # SOURCE: (见 impl-notes.md §Source Map——_host_seams.py)
    def __init__(self) -> None:
        self.log: list[dict] = []

    # SOURCE: vllm/device_allocator/cumem.py CuMemAllocator.get_current_usage
    def get_current_usage(self) -> int:  # HOST SEAM
        return sum(e.get("bytes", 0) for e in self.log if e.get("entered"))
    @contextlib.contextmanager
    # SOURCE: vllm/device_allocator/cumem.py CuMemAllocator.use_memory_pool
    def use_memory_pool(self, tag: str):  # HOST SEAM
        self.log.append({"tag": tag, "entered": True})
        try:
            yield
        finally:
            self.log.append({"tag": tag, "entered": False})


CUMEM_SEAM = _CuMemAllocatorSeam()


# SOURCE: vllm/device_allocator/__init__.py get_mem_allocator_instance
def get_mem_allocator_instance():  # HOST SEAM
    return CUMEM_SEAM


# ---------------------------------------------------------------------------
# warmup / observability faces — ch19 domain
# ---------------------------------------------------------------------------


# SOURCE: vllm/model_executor/warmup/kernel_warmup.py kernel_warmup — HOST SEAM
# （kernel 调优细节归 ch19；编排调用位是本章 m9 的对象）
# SOURCE: (见 impl-notes.md §Source Map——_host_seams.py)
def kernel_warmup(worker) -> None:  # HOST SEAM
    return None


# SOURCE: vllm/utils/jit_monitor.py activate — HOST SEAM（JIT 纠察归 ch19）
def activate_jit_monitor(mode=None, verbose: bool = False) -> None:  # HOST SEAM
    return None


# SOURCE: vllm/compilation/compiler_interface.py trigger_inductor_lazy_init —
# 该函数本体在真实树内；宿主替身仅保留惰性初始化的调用面（inductor 分支在
# seam config mode=NONE 下不触发）。
# SOURCE: (见 impl-notes.md §Source Map——_host_seams.py)
def trigger_inductor_lazy_init(device) -> None:  # HOST SEAM
    return None


# SOURCE: vllm/v1/utils.py:L758-L772 record_function_or_nullcontext — HOST SEAM:
# 无 profiler 作用域时退化为 nullcontext（真实代码的默认路径）。
# SOURCE: (见 impl-notes.md §Source Map——_host_seams.py)
def record_function_or_nullcontext(name: str):  # HOST SEAM
    return contextlib.nullcontext()


# SOURCE: vllm/plugins/__init__.py load_general_plugins — HOST SEAM: 顺序约束
# envs→插件→实例化的中间一环（真实插件加载平台可插拔，宿主无插件可载）。
# SOURCE: (见 impl-notes.md §Source Map——_host_seams.py)
def load_general_plugins() -> None:  # HOST SEAM
    return None


# SOURCE: vllm/model_executor/model_loader TensorizerLoader — HOST SEAM 占位
# （save_tensorized 的生产面；宿主不实例化）
class TensorizerLoader:  # HOST SEAM
    # SOURCE: vllm/model_executor/model_loader TensorizerLoader.save_model
    def save_model(self, *args, **kwargs) -> None:  # HOST SEAM
        raise NotImplementedError("tensorizer save is out of the ch17 companion")


# SOURCE: vllm/utils/network_utils.py get_ip / get_open_port / loopback — HOST SEAM
def get_ip() -> str:  # HOST SEAM
    return "127.0.0.1"


# SOURCE: vllm/utils/network_utils.py get_loopback_ip
def get_loopback_ip() -> str:  # HOST SEAM
    return "127.0.0.1"


# SOURCE: vllm/utils/network_utils.py get_open_port
def get_open_port() -> int:  # HOST SEAM
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# SOURCE: vllm/utils/network_utils.py get_distributed_init_method
def get_distributed_init_method(ip: str, port: int) -> str:  # HOST SEAM
    return f"tcp://{ip}:{port}"


# ---------------------------------------------------------------------------
# qualname aliasing — HOST SEAM for `import vllm.…` qualnames
# ---------------------------------------------------------------------------

# HOST SEAM: 真实 vllm 包缺席时，vLLM 的规范 qualname 解析到本精简包：
#   "vllm.v1.worker.gpu_worker.Worker" -> implementation.worker.gpu_worker.Worker
# resolve_obj_by_qualname（vllm/utils/import_utils.py:L104-L110）逐字保留，
# importlib 命中 sys.modules 里预置的别名。
_ALIAS_TARGETS = (
    "vllm.v1.executor.abstract",
    "vllm.v1.executor.uniproc_executor",
    "vllm.v1.executor.multiproc_executor",
    "vllm.v1.worker.worker_base",
    "vllm.v1.worker.gpu_worker",
    "vllm.v1.worker.gpu_model_runner",
    "vllm.v1.serial_utils",
)


# SOURCE: (见 impl-notes.md §Source Map——_host_seams.py)
def install_vllm_module_aliases() -> None:  # HOST SEAM
    if importlib.util.find_spec("vllm") is not None:
        return  # 真实 vllm 在场：不劫持
    for name in ("vllm", "vllm.v1", "vllm.v1.executor", "vllm.v1.worker"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []  # namespace-package shape for importlib
            sys.modules[name] = mod
    for target in _ALIAS_TARGETS:
        if target in sys.modules:
            continue
        local = "implementation" + target[len("vllm.v1"):]
        sys.modules[target] = importlib.import_module(local)
    # 保留代码中的真实惰性导入面（HOST SEAM 一函数模块）：
    # vllm.compilation.compiler_interface.trigger_inductor_lazy_init
    if "vllm.compilation.compiler_interface" not in sys.modules:
        pkg = types.ModuleType("vllm.compilation")
        pkg.__path__ = []
        sys.modules.setdefault("vllm.compilation", pkg)
        iface = types.ModuleType("vllm.compilation.compiler_interface")
        iface.trigger_inductor_lazy_init = trigger_inductor_lazy_init
        sys.modules["vllm.compilation.compiler_interface"] = iface
