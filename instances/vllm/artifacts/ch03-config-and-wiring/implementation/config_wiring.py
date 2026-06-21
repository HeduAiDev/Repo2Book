# Subtract-only companion for ch03 "EngineArgs -> VllmConfig: Assembling the Stack".
#
# This module is a FAITHFUL SUBSET of the real vLLM configuration/wiring path,
# carved out so a reader can run it, set breakpoints and trace concrete values
# WITHOUT a GPU or an installed vLLM (host has no CUDA/vLLM). It keeps vLLM's
# names, structure and control flow; it only DELETES branches approved in the
# dossier subtraction_plan and marks every deletion with `# SUBTRACTED:`.
#
# Mapping rule, verbatim from the contract: take the real vLLM source, drop every
# `# SUBTRACTED:` branch, and you should get (approximately) this file.
#
# Source pin: f3fef123. Every def/class carries a `# SOURCE: vllm/...:Lxxx` ref.
#
# What is genuinely simplified vs. the real tree (and why it is still faithful):
#   * The sub-Config dataclasses (ModelConfig/CacheConfig/...) in real vLLM live
#     in vllm/config/*.py and carry hundreds of fields + heavy HF/platform IO.
#     Here they are reduced to the *fields this chapter actually reads* so the
#     two-level mapping (flat args -> structured config -> implementation class)
#     and the cross-config derivations stay observable. No field is invented:
#     every kept attribute corresponds 1:1 to a real vLLM attribute of the same
#     name (see impl-notes.md Source Map). Fields not exercised by this chapter's
#     control flow are SUBTRACTED, not renamed.
#   * `current_platform` HF reads / CUDA probes are replaced by a tiny injectable
#     `Platform` stub so the same control flow runs on a CPU host. This is NOT a
#     toy model forward; it is the minimal seam vLLM itself goes through
#     (current_platform.*), kept as a real decision input.

from __future__ import annotations

import time
from dataclasses import dataclass, field, is_dataclass, replace
from enum import IntEnum
from typing import Any, Callable, Optional, Union

# ----------------------------------------------------------------------------
# Enums mirrored from vllm/config/compilation.py (only the members this chapter
# observes are kept; values match the real enum semantics).
# ----------------------------------------------------------------------------


# SOURCE: vllm/config/compilation.py (CompilationMode)
class CompilationMode(IntEnum):
    NONE = 0
    # SUBTRACTED: STOCK_TORCH_COMPILE / DYNAMO_TRACE_ONCE intermediate modes
    #   (vllm/config/compilation.py) — this chapter only branches on NONE vs
    #   VLLM_COMPILE, so the in-between modes are dropped without changing the
    #   `mode is None -> VLLM_COMPILE/NONE` decision.
    VLLM_COMPILE = 3


# SOURCE: vllm/config/compilation.py (CUDAGraphMode)
class CUDAGraphMode(IntEnum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    # SUBTRACTED: FULL_DECODE_ONLY (vllm/config/compilation.py) — not referenced
    #   by any O0-O3 preset in this chapter.
    FULL_AND_PIECEWISE = 3

    # SOURCE: vllm/config/compilation.py CUDAGraphMode.requires_piecewise_compilation
    def requires_piecewise_compilation(self) -> bool:
        return self in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL_AND_PIECEWISE)


# ----------------------------------------------------------------------------
# Platform seam. Real vLLM calls module-global `current_platform`; we inject a
# stub so the *same* decision branches run on a CPU host.
# SOURCE: vllm/platforms/__init__.py current_platform (interface subset)
# ----------------------------------------------------------------------------


@dataclass
class Platform:
    # SOURCE: vllm/platforms/__init__.py current_platform (interface subset)
    device_type: str = "cuda"
    _is_cuda: bool = True
    _device_count: int = 1

    def is_cuda(self) -> bool:
        # SOURCE: vllm/platforms/interface.py Platform.is_cuda
        return self._is_cuda

    def is_rocm(self) -> bool:
        # SOURCE: vllm/platforms/interface.py Platform.is_rocm
        return False

    def is_tpu(self) -> bool:
        # SOURCE: vllm/platforms/interface.py Platform.is_tpu
        return False

    def device_count(self) -> int:
        # SOURCE: vllm/platforms/interface.py Platform.device_count
        return self._device_count

    def pre_register_and_update(self) -> None:
        # SOURCE: vllm/platforms/interface.py Platform.pre_register_and_update
        # SUBTRACTED: per-platform plugin registration / CLI option injection
        #   (vllm/platforms/*.pre_register_and_update) — host stub no-op; does
        #   not affect the assembly control flow traced in this chapter.
        return None

    def apply_config_platform_defaults(self, vllm_config: "VllmConfig") -> None:
        # SOURCE: vllm/platforms/interface.py Platform.apply_config_platform_defaults
        # SUBTRACTED: platform-specific config overrides (e.g. ROCm/TPU)
        #   (vllm/platforms/*.apply_config_platform_defaults) — no-op on the
        #   generic CUDA path this chapter follows.
        return None


# Injectable singleton (assign in tests to flip world_size/cuda decisions).
current_platform = Platform()


# ============================================================================
# Structured sub-configs (faithful field subsets).
# ============================================================================


# SOURCE: vllm/config/model.py ModelConfig (field subset)
@dataclass
class ModelConfig:
    model: str = "facebook/opt-125m"
    tokenizer: Optional[str] = None
    served_model_name: Optional[Union[str, list[str]]] = None
    max_model_len: int = 8192
    runner_type: str = "generate"  # "generate" | "pooling"
    is_multimodal_model: bool = False
    is_encoder_decoder: bool = False
    is_moe: bool = False
    is_attention_free: bool = False
    enforce_eager: bool = False
    model_weights: Optional[str] = None
    # SUBTRACTED: hundreds of fields (dtype/quantization/hf_config/rope/...) and
    #   the HF-config-reading machinery (vllm/config/model.py) — this chapter
    #   only reads the derived flags above to drive cross-config decisions.

    # SOURCE: vllm/config/model.py ModelConfig.verify_with_parallel_config
    def verify_with_parallel_config(self, parallel_config: "ParallelConfig") -> None:
        # SUBTRACTED: TP/PP head-count divisibility + quant compatibility checks
        #   (vllm/config/model.py) — kept as a no-op seam so __post_init__'s call
        #   site stays identical; raising paths are model-specific edge cases.
        return None

    # SOURCE: vllm/config/model.py ModelConfig.compute_hash
    def compute_hash(self) -> str:
        return _hash10(str((self.model, self.max_model_len, self.runner_type)))


# SOURCE: vllm/config/cache.py CacheConfig (field subset)
@dataclass
class CacheConfig:
    block_size: Optional[int] = None
    gpu_memory_utilization: float = 0.9
    cache_dtype: str = "auto"
    enable_prefix_caching: Optional[bool] = None
    num_gpu_blocks_override: Optional[int] = None
    sliding_window: Optional[int] = None
    is_attention_free: bool = False
    # SUBTRACTED: kv_offloading_* / mamba_* / prefix_caching_hash_algo /
    #   calculate_kv_scales / kv_sharing_fast_prefill and the rest of the
    #   CacheConfig fields (vllm/config/cache.py) — not read by this chapter's
    #   assembly/derivation flow.

    # SOURCE: vllm/config/cache.py CacheConfig.compute_hash
    def compute_hash(self) -> str:
        return _hash10(str((self.block_size, self.cache_dtype,
                            self.enable_prefix_caching)))


# SOURCE: vllm/config/parallel.py ParallelConfig (field subset)
@dataclass
class ParallelConfig:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    # SOURCE: vllm/config/parallel.py ParallelConfig.distributed_executor_backend
    distributed_executor_backend: Optional[Union[str, type]] = None
    data_parallel_backend: str = "mp"
    data_parallel_external_lb: bool = False
    nnodes: int = 1
    is_moe_model: bool = False
    # SUBTRACTED: context/expert/decode parallel sizes, placement_group,
    #   data_parallel_* LB fields, disable_nccl_for_dp_synchronization, etc.
    #   (vllm/config/parallel.py) — DP load-balancing + elastic-EP are an
    #   independent subsystem (dossier delete item), single-DP path unaffected.

    @property
    def world_size(self) -> int:
        # SOURCE: vllm/config/parallel.py ParallelConfig.world_size
        # SUBTRACTED: ... * decode_context_parallel_size factor (PCP) — fixed to
        #   1 in the single-DP path, so world_size = TP * PP here.
        return self.tensor_parallel_size * self.pipeline_parallel_size

    @property
    def world_size_across_dp(self) -> int:
        # SOURCE: vllm/config/parallel.py ParallelConfig.world_size_across_dp
        return self.world_size * self.data_parallel_size

    # SOURCE: vllm/config/parallel.py ParallelConfig.__post_init__ (backend derivation, L829-L874)
    def __post_init__(self) -> None:
        # SUBTRACTED: external_launcher early branch, max_parallel_loading_workers
        #   warning, allowed_backends/nnodes validation, elastic-EP
        #   (vllm/config/parallel.py:L825-L827, L876+) — orthogonal to backend
        #   selection traced here.
        if (self.distributed_executor_backend is None
                and self.world_size_across_dp > 1):
            # We use multiprocessing by default if world_size fits on the
            # current node and we aren't in a ray placement group.
            backend = "mp"
            # SUBTRACTED: ray_utils.ray_is_available() probe + the TPU/SPMD,
            #   data_parallel_backend=="ray", ray-initialized/placement-group
            #   branches (vllm/config/parallel.py:L833-L869) — Ray is a dossier
            #   delete item; single-node mp/uni path keeps its exact logic.
            if current_platform.is_cuda() and self.nnodes > 1:
                backend = "mp"
            elif (current_platform.is_cuda()
                  and current_platform.device_count() < self.world_size):
                gpu_count = current_platform.device_count()
                raise ValueError(
                    f"World size ({self.world_size}) is larger than the number "
                    f"of available GPUs ({gpu_count}) in this node."
                )
            self.distributed_executor_backend = backend

        if (self.distributed_executor_backend is None
                and self.world_size == 1):
            self.distributed_executor_backend = "uni"

    # SOURCE: vllm/config/parallel.py ParallelConfig.compute_hash
    def compute_hash(self) -> str:
        return _hash10(str((self.tensor_parallel_size,
                            self.pipeline_parallel_size,
                            self.data_parallel_size)))


# SOURCE: vllm/config/scheduler.py SchedulerConfig (field subset)
@dataclass
class SchedulerConfig:
    runner_type: str = "generate"
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 256
    max_model_len: int = 8192
    enable_chunked_prefill: bool = True
    is_multimodal_model: bool = False
    is_encoder_decoder: bool = False
    policy: str = "fcfs"
    scheduler_cls: Optional[Union[str, type]] = None
    # SOURCE: vllm/config/scheduler.py SchedulerConfig.async_scheduling
    async_scheduling: Optional[bool] = None  # tri-state: True / False / None(auto)
    # SUBTRACTED: max_num_partial_prefills / long_prefill_token_threshold /
    #   disable_hybrid_kv_cache_manager / stream_interval and friends
    #   (vllm/config/scheduler.py) — not read by the factory/derivation flow.

    # SOURCE: vllm/config/scheduler.py SchedulerConfig.get_scheduler_cls (L168-L188)
    def get_scheduler_cls(self) -> type:
        if self.scheduler_cls is None:
            if self.async_scheduling:
                # SOURCE: vllm/v1/core/sched/async_scheduler.py AsyncScheduler
                return AsyncScheduler
            # SOURCE: vllm/v1/core/sched/scheduler.py Scheduler
            return Scheduler

        # This warning can be removed once the Scheduler interface is finalized.
        # SUBTRACTED: logger.warning_once about non-public custom scheduler
        #   interface (vllm/config/scheduler.py:L181-L185) — logging only.
        if not isinstance(self.scheduler_cls, str):
            return self.scheduler_cls
        return _resolve_obj_by_qualname(self.scheduler_cls)

    # SOURCE: vllm/config/scheduler.py SchedulerConfig.compute_hash
    def compute_hash(self) -> str:
        return _hash10(str((self.max_num_batched_tokens, self.max_num_seqs,
                            self.policy, self.async_scheduling)))


# SOURCE: vllm/config/compilation.py PassConfig (fusion-flag subset)
@dataclass
class PassConfig:
    # SOURCE: vllm/config/compilation.py PassConfig (fusion-flag subset)
    fuse_norm_quant: Optional[bool] = None
    fuse_act_quant: Optional[bool] = None
    fuse_allreduce_rms: Optional[bool] = None
    fuse_attn_quant: Optional[bool] = None
    enable_sp: Optional[bool] = None
    fuse_gemm_comms: Optional[bool] = None
    fuse_act_padding: Optional[bool] = None
    fuse_mla_dual_rms_norm: Optional[bool] = None
    fuse_rope_kvcache: Optional[bool] = None


# SOURCE: vllm/config/compilation.py CompilationConfig (field subset)
@dataclass
class CompilationConfig:
    mode: Optional[CompilationMode] = None
    cudagraph_mode: Optional[CUDAGraphMode] = None
    use_inductor_graph_partition: Optional[bool] = None
    backend: str = "inductor"
    pass_config: PassConfig = field(default_factory=PassConfig)
    # SUBTRACTED: custom_ops list, ir_enable_torch_wrap, cudagraph_capture_sizes,
    #   compile_mm_encoder and the rest (vllm/config/compilation.py) — the
    #   custom_ops/ir_enable_torch_wrap derivations in __post_init__ are
    #   themselves SUBTRACTED below, so these fields are not read here.

    # SOURCE: vllm/config/compilation.py CompilationConfig.compute_hash
    def compute_hash(self) -> str:
        return _hash10(str((self.mode, self.cudagraph_mode)))


# SOURCE: vllm/config/kernel.py KernelConfig (field subset)
@dataclass
class KernelConfig:
    enable_flashinfer_autotune: Optional[bool] = None
    # SUBTRACTED: ir_op_priority + the rest (vllm/config/kernel.py) — only the
    #   autotune flag is observed by the optimization-level application here.

    # SOURCE: vllm/config/kernel.py KernelConfig.set_platform_defaults
    def set_platform_defaults(self, vllm_config: "VllmConfig") -> None:
        # SUBTRACTED: IR op-priority population from platform (vllm/config/kernel.py)
        #   — populates op priorities used by fusion predicates; the predicates
        #   themselves are SUBTRACTED (replaced by static bools), so this is a
        #   no-op seam preserving the call site before fusion defaults apply.
        return None

    # SOURCE: vllm/config/kernel.py KernelConfig.compute_hash
    def compute_hash(self) -> str:
        return _hash10(str((self.enable_flashinfer_autotune,)))


# SOURCE: vllm/config/device.py DeviceConfig (field subset)
@dataclass
class DeviceConfig:
    device: str = "cuda"

    # SOURCE: vllm/config/device.py DeviceConfig.compute_hash
    def compute_hash(self) -> str:
        return _hash10(str((self.device,)))


# ============================================================================
# Optimization levels — vllm/config/vllm.py:L68-L270
# ============================================================================


# SOURCE: vllm/config/vllm.py OptimizationLevel (L68-L80)
class OptimizationLevel(IntEnum):
    """Optimization level enum."""

    O0 = 0
    """O0 : No optimization. no compilation, no cudagraphs, just starting up."""
    O1 = 1
    """O1: Quick optimizations. Dynamo+Inductor + Piecewise cudagraphs."""
    O2 = 2
    """O2: Full optimizations. -O1 as well as Full and Piecewise cudagraphs."""
    O3 = 3
    """O3: Currently the same as -O2."""


PerformanceMode = str  # SOURCE: vllm/config/vllm.py PerformanceMode (Literal)

# SOURCE: vllm/config/vllm.py IS_QUANTIZED / IS_DENSE (L85-L92)
# These are deliberately constant False in current vLLM (see issue #25689).
IS_QUANTIZED = False
IS_DENSE = False


# Fusion predicates are functions in real vLLM (lazily evaluated against the
# VllmConfig). We keep the *function-valued preset* shape — the dossier marks
# the predicate BODIES as elided, not the mechanism — but the bodies are reduced
# to a faithful platform-independent default so the file runs on a CPU host.

# SOURCE: vllm/config/vllm.py enable_norm_fusion (L95-L103)
def enable_norm_fusion(cfg: "VllmConfig") -> bool:
    # SUBTRACTED: custom-op / kernel ir_op_priority probes (vllm/config/vllm.py)
    #   — depend on CompilationConfig.custom_ops + KernelConfig.ir_op_priority,
    #   both SUBTRACTED above; reduced to False so the *function-valued default*
    #   mechanism is still demonstrated end-to-end on host.
    return False


# SOURCE: vllm/config/vllm.py enable_act_fusion (L106-L116)
def enable_act_fusion(cfg: "VllmConfig") -> bool:
    # SUBTRACTED: custom-op / nvfp4 probes (vllm/config/vllm.py) — see above.
    return False


# SOURCE: vllm/config/vllm.py enable_allreduce_rms_fusion (L119-L147)
def enable_allreduce_rms_fusion(cfg: "VllmConfig") -> bool:
    # SUBTRACTED: ROCm aiter + Hopper/Blackwell + flashinfer + TP/DP/PP gating
    #   (vllm/config/vllm.py:L120-L147) — all platform/hardware probes; reduced
    #   to the TP>1 prerequisite which is the user-visible knob.
    return cfg.parallel_config.tensor_parallel_size > 1


# SUBTRACTED: enable_norm_pad_fusion / enable_mla_dual_rms_norm_fusion /
#   enable_rope_kvcache_fusion predicate bodies (vllm/config/vllm.py) — same
#   class of platform probes; represented below by the static defaults they
#   resolve to on the generic CUDA path.


# SOURCE: vllm/config/vllm.py OPTIMIZATION_LEVEL_00 (L184-L203)
OPTIMIZATION_LEVEL_00 = {
    "compilation_config": {
        "pass_config": {
            "fuse_norm_quant": False,
            "fuse_act_quant": False,
            "fuse_allreduce_rms": False,
            "fuse_attn_quant": False,
            "enable_sp": False,
            "fuse_gemm_comms": False,
            "fuse_act_padding": False,
            "fuse_mla_dual_rms_norm": False,
            "fuse_rope_kvcache": False,
        },
        "cudagraph_mode": CUDAGraphMode.NONE,
        "use_inductor_graph_partition": False,
    },
    "kernel_config": {
        "enable_flashinfer_autotune": False,
    },
}
# SOURCE: vllm/config/vllm.py OPTIMIZATION_LEVEL_01 (L204-L223)
OPTIMIZATION_LEVEL_01 = {
    "compilation_config": {
        "pass_config": {
            "fuse_norm_quant": enable_norm_fusion,
            "fuse_act_quant": enable_act_fusion,
            "fuse_allreduce_rms": False,
            "fuse_attn_quant": False,
            "enable_sp": False,
            "fuse_gemm_comms": False,
            # SUBTRACTED: enable_norm_pad_fusion predicate -> False default
            "fuse_act_padding": False,
            # SUBTRACTED: enable_mla_dual_rms_norm_fusion predicate -> False default
            "fuse_mla_dual_rms_norm": False,
            "fuse_rope_kvcache": False,
        },
        "cudagraph_mode": CUDAGraphMode.PIECEWISE,
        "use_inductor_graph_partition": False,
    },
    "kernel_config": {
        "enable_flashinfer_autotune": True,
    },
}
# SOURCE: vllm/config/vllm.py OPTIMIZATION_LEVEL_02 (L224-L243)
OPTIMIZATION_LEVEL_02 = {
    "compilation_config": {
        "pass_config": {
            "fuse_norm_quant": enable_norm_fusion,
            "fuse_act_quant": enable_act_fusion,
            "fuse_allreduce_rms": enable_allreduce_rms_fusion,
            "fuse_attn_quant": IS_QUANTIZED,
            "enable_sp": IS_DENSE,
            "fuse_gemm_comms": IS_DENSE,
            # SUBTRACTED: enable_norm_pad_fusion predicate -> False default
            "fuse_act_padding": False,
            # SUBTRACTED: enable_mla_dual_rms_norm_fusion predicate -> False default
            "fuse_mla_dual_rms_norm": False,
            # SUBTRACTED: enable_rope_kvcache_fusion predicate -> False default
            "fuse_rope_kvcache": False,
        },
        "cudagraph_mode": CUDAGraphMode.FULL_AND_PIECEWISE,
        "use_inductor_graph_partition": False,
    },
    "kernel_config": {
        "enable_flashinfer_autotune": True,
    },
}
# SOURCE: vllm/config/vllm.py OPTIMIZATION_LEVEL_03 (L244-L263) — same as O2.
OPTIMIZATION_LEVEL_03 = OPTIMIZATION_LEVEL_02

# SOURCE: vllm/config/vllm.py OPTIMIZATION_LEVEL_TO_CONFIG (L265-L270)
OPTIMIZATION_LEVEL_TO_CONFIG = {
    OptimizationLevel.O0: OPTIMIZATION_LEVEL_00,
    OptimizationLevel.O1: OPTIMIZATION_LEVEL_01,
    OptimizationLevel.O2: OPTIMIZATION_LEVEL_02,
    OptimizationLevel.O3: OPTIMIZATION_LEVEL_03,
}


# ============================================================================
# VllmConfig — the aggregate config + cross-config derivation hub.
# vllm/config/vllm.py:L274-L981
# ============================================================================


# SOURCE: vllm/config/vllm.py VllmConfig (L274-L366)
@dataclass
class VllmConfig:
    """Dataclass which contains all vllm-related configuration."""

    model_config: ModelConfig = None  # type: ignore[assignment]
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    device_config: DeviceConfig = field(default_factory=DeviceConfig)
    kernel_config: KernelConfig = field(default_factory=KernelConfig)
    compilation_config: CompilationConfig = field(default_factory=CompilationConfig)
    lora_config: Optional[Any] = None
    speculative_config: Optional[Any] = None
    quant_config: Optional[Any] = None
    instance_id: str = ""
    optimization_level: OptimizationLevel = OptimizationLevel.O2
    performance_mode: PerformanceMode = "balanced"
    # SUBTRACTED: load/offload/attention/mamba/structured_outputs/observability/
    #   profiler/kv_transfer/kv_events/ec_transfer/reasoning/additional/
    #   weight_transfer/shutdown_timeout fields (vllm/config/vllm.py:L291-L365)
    #   — carried verbatim into VllmConfig in real vLLM but not read by this
    #   chapter's derivation or by the three factories.

    # SOURCE: vllm/config/vllm.py VllmConfig.compute_hash (L367-L473)
    def compute_hash(self) -> str:
        """Provide a hash that uniquely identifies all the configs that affect
        the structure of the computation graph from input ids/embeddings to the
        final hidden states.
        """
        factors: list[Any] = []

        # summarize vllm config
        vllm_factors: list[Any] = []
        # SUBTRACTED: from vllm import __version__ (vllm/config/vllm.py:L383-L385)
        #   — host has no vllm package; use a fixed stand-in so the algorithm
        #   (collect factors -> hash -> first 10 chars) stays identical.
        vllm_factors.append("0.15.1")
        if self.model_config:
            vllm_factors.append(self.model_config.compute_hash())
            # SUBTRACTED: multimodal-encoder hash append (vllm/config/vllm.py:L388-L393)
        else:
            vllm_factors.append("None")
        if self.cache_config:
            vllm_factors.append(self.cache_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.parallel_config:
            vllm_factors.append(self.parallel_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.scheduler_config:
            vllm_factors.append(self.scheduler_config.compute_hash())
        else:
            vllm_factors.append("None")
        # SUBTRACTED: device/load/offload/attention/lora/speculative/
        #   structured_outputs/profiler/observability/quant/kv_transfer/
        #   ec_transfer/additional_config appends (vllm/config/vllm.py:L408-L467)
        #   — all the same if-present-append-hash-else-"None" shape; keeping
        #   model/cache/parallel/scheduler/compilation/kernel is enough to show
        #   the algorithm. (dossier delete item.)
        if self.compilation_config:
            vllm_factors.append(self.compilation_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.kernel_config:
            vllm_factors.append(self.kernel_config.compute_hash())
        else:
            vllm_factors.append(None)
        factors.append(vllm_factors)

        hash_str = _safe_hash(str(factors).encode())[:10]
        return hash_str

    # SOURCE: vllm/config/vllm.py VllmConfig._set_config_default (L637-L650)
    def _set_config_default(self, config_obj: Any, key: str, value: Any) -> None:
        """Set config attribute to default if not already set by user."""
        if getattr(config_obj, key) is None:
            # Static values are hard-coded; values that depend on user config are
            # callables decided at run time.
            setattr(config_obj, key, value(self) if callable(value) else value)

    # SOURCE: vllm/config/vllm.py VllmConfig._apply_optimization_level_defaults (L652-L679)
    def _apply_optimization_level_defaults(self, defaults: dict[str, Any]) -> None:
        """Apply optimization level defaults using self as root.

        Recursively applies values from defaults into nested config objects.
        Only fields present in defaults are overwritten. User specified fields
        will not be overridden by the default.
        """

        def apply_recursive(config_obj: Any, config_defaults: dict[str, Any]) -> None:
            # SOURCE: vllm/config/vllm.py _apply_optimization_level_defaults.apply_recursive (L667-L677)
            for key, value in config_defaults.items():
                if not hasattr(config_obj, key):
                    continue

                current = getattr(config_obj, key)
                if isinstance(value, dict) and is_dataclass(current):
                    apply_recursive(current, value)
                else:
                    self._set_config_default(config_obj, key, value)

        apply_recursive(self, defaults)

    # SOURCE: vllm/config/vllm.py VllmConfig.__post_init__ (L721-L981)
    def __post_init__(self):
        """Verify configs are valid & consistent with each other."""

        # To give each torch profile run a unique instance name.
        self.instance_id = f"{time.time_ns()}"

        # SUBTRACTED: performance_mode info_once log + try_verify_and_update_config
        #   (vllm/config/vllm.py:L727-L730) — the latter does architecture-specific
        #   model rewrites (HF-driven); no-op on this chapter's path.

        if self.model_config is not None:
            self.model_config.verify_with_parallel_config(self.parallel_config)
            # SUBTRACTED: verify_dual_chunk_attention_config (vllm/config/vllm.py:L734)
            self.parallel_config.is_moe_model = self.model_config.is_moe

        if self.lora_config is not None:
            self.lora_config.verify_with_model_config(self.model_config)

        # SUBTRACTED: mamba stochastic-rounding check + deep_gemm auto-disable
        #   (vllm/config/vllm.py:L741-L775) — hardware/feature edge cases.

        if self.quant_config is None and self.model_config is not None:
            # SUBTRACTED: VllmConfig._get_quantization_config(...) call
            #   (vllm/config/vllm.py:L753-L756) — reads model quant metadata;
            #   left None on the unquantized path this chapter traces.
            pass

        # ---- async_scheduling tri-state decision (L777-L852) ----
        executor_backend = self.parallel_config.distributed_executor_backend
        executor_class = Executor.get_class(self)
        executor_supports_async_sched = executor_class.supports_async_scheduling()

        if self.scheduler_config.async_scheduling:
            # Async scheduling explicitly enabled, hard fail incompatibilities.
            if self.speculative_config is not None:
                # SUBTRACTED: EAGLE/MTP/Draft/NGram method allow-list +
                #   disable_padded_drafter_batch hard checks
                #   (vllm/config/vllm.py:L787-L802) — speculative decoding is out
                #   of this chapter's single-path scope (speculative_config None).
                pass
            if not executor_supports_async_sched:
                raise ValueError(
                    f"`{executor_backend}` does not support async scheduling yet."
                )
        elif self.scheduler_config.async_scheduling is None:
            # Enable async scheduling unless there is an incompatible option.
            if (self.model_config is not None
                    and self.model_config.runner_type == "pooling"):
                # Async scheduling hurts pooling models, disable by default.
                self.scheduler_config.async_scheduling = False
            elif self.speculative_config is not None:
                # SUBTRACTED: split into method-allow-list + disable_padded_drafter
                #   branches (vllm/config/vllm.py:L819-L838) — collapsed because
                #   speculative_config is None on this path; behavior preserved:
                #   any speculative_config disables auto async scheduling here.
                self.scheduler_config.async_scheduling = False
            elif not executor_supports_async_sched:
                self.scheduler_config.async_scheduling = False
            else:
                self.scheduler_config.async_scheduling = True

        # SUBTRACTED: logger.info_once("Asynchronous scheduling is %s") (L849-L852)

        # SUBTRACTED: disable_nccl_for_dp_synchronization derivation, cascade-attn
        #   disable for async speculative, torch_shm spawn check, Turing float32
        #   warning (vllm/config/vllm.py:L854-L902) — DP/spec/hardware edge cases.

        # ---- compilation / cudagraph final resolution + opt-level apply (L904-L976) ----
        if self.model_config is not None and self.model_config.enforce_eager:
            # Enforce eager: disable torch.compile and CUDAGraphs (overrides opt level).
            self.compilation_config.mode = CompilationMode.NONE
            self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

        if _env_get("TORCH_COMPILE_DISABLE") == "1":
            self.compilation_config.mode = CompilationMode.NONE

        # SUBTRACTED: inductor-disabled warning, has_blocked_weights custom_ops
        #   append (vllm/config/vllm.py:L919-L944) — logging + quant edge case.

        current_platform.apply_config_platform_defaults(self)

        if self.compilation_config.mode is None:
            if self.optimization_level > OptimizationLevel.O0:
                self.compilation_config.mode = CompilationMode.VLLM_COMPILE
            else:
                self.compilation_config.mode = CompilationMode.NONE

        # SUBTRACTED: ir_enable_torch_wrap + custom_ops "all"/"none" defaulting
        #   (vllm/config/vllm.py:L954-L968) — depends on SUBTRACTED custom_ops field.

        # Populate IR op priorities before fusion defaults are applied.
        self.kernel_config.set_platform_defaults(self)

        default_config = OPTIMIZATION_LEVEL_TO_CONFIG[self.optimization_level]
        self._apply_optimization_level_defaults(default_config)
        if self.kernel_config.enable_flashinfer_autotune is None:
            raise ValueError(
                "KernelConfig.enable_flashinfer_autotune must be set after "
                "applying optimization level defaults."
            )

        # SUBTRACTED: cudagraph-mode requires_piecewise_compilation vs mode
        #   consistency guard + tail validations (vllm/config/vllm.py:L983+) —
        #   final-state assertions; the kept flow already lands the values.


# ============================================================================
# EngineArgs — flat user-facing args + the first-level mapping.
# vllm/engine/arg_utils.py
# ============================================================================


# SOURCE: vllm/engine/arg_utils.py EngineArgs (L403-L462, field subset)
@dataclass
class EngineArgs:
    """Arguments for vLLM engine.

    Each field default references the matching sub-Config attribute, so CLI
    defaults and sub-config defaults are a single source of truth.
    """

    model: str = ModelConfig.model
    tokenizer: Optional[str] = ModelConfig.tokenizer
    served_model_name: Optional[Union[str, list[str]]] = ModelConfig.served_model_name
    max_model_len: int = ModelConfig.max_model_len
    kv_cache_dtype: str = CacheConfig.cache_dtype
    block_size: Optional[int] = CacheConfig.block_size
    gpu_memory_utilization: float = CacheConfig.gpu_memory_utilization
    enable_prefix_caching: Optional[bool] = CacheConfig.enable_prefix_caching
    num_gpu_blocks_override: Optional[int] = CacheConfig.num_gpu_blocks_override
    tensor_parallel_size: int = ParallelConfig.tensor_parallel_size
    pipeline_parallel_size: int = ParallelConfig.pipeline_parallel_size
    data_parallel_size: int = ParallelConfig.data_parallel_size
    # Note: Specifying a custom executor backend by passing a class is intended
    # for expert use only. The API may change without notice.
    distributed_executor_backend: Optional[Union[str, type]] = \
        ParallelConfig.distributed_executor_backend
    max_num_batched_tokens: int = SchedulerConfig.max_num_batched_tokens
    max_num_seqs: int = SchedulerConfig.max_num_seqs
    enable_chunked_prefill: bool = SchedulerConfig.enable_chunked_prefill
    scheduling_policy: str = SchedulerConfig.policy
    scheduler_cls: Optional[Union[str, type]] = SchedulerConfig.scheduler_cls
    async_scheduling: Optional[bool] = SchedulerConfig.async_scheduling
    enforce_eager: bool = ModelConfig.enforce_eager
    optimization_level: OptimizationLevel = OptimizationLevel.O2
    performance_mode: PerformanceMode = "balanced"
    disable_log_stats: bool = False
    compilation_config: Any = field(default_factory=CompilationConfig)
    # SUBTRACTED: the several-hundred remaining EngineArgs fields — multimodal /
    #   LoRA / speculative / KV-transfer / quantization / load / observability /
    #   etc. (vllm/engine/arg_utils.py:L403-L690) — not on this chapter's
    #   single-DP, unquantized assembly path.

    # SOURCE: vllm/engine/arg_utils.py EngineArgs.__post_init__ (L690-L720)
    def __post_init__(self):
        # support `EngineArgs(compilation_config={...})` without having to
        # manually construct a CompilationConfig object.
        if isinstance(self.compilation_config, dict):
            self.compilation_config = CompilationConfig(**self.compilation_config)
        # SUBTRACTED: attention/mamba/kernel/eplb dict->Config promotions +
        #   plugin loading + HF-offline path replacement (vllm/engine/arg_utils.py
        #   :L697-L720) — same dict->Config promotion pattern; the one kept above
        #   is representative, the rest are dossier-elided fields.

    # SOURCE: vllm/engine/arg_utils.py EngineArgs.create_model_config (subset)
    def create_model_config(self) -> ModelConfig:
        # SUBTRACTED: HF config download/parse, dtype resolution, runner inference,
        #   multimodal/encoder-decoder detection (vllm/engine/arg_utils.py
        #   create_model_config + vllm/config/model.py) — the heaviest step in
        #   real vLLM (network + HF IO). Here we pack the flat fields into a
        #   ModelConfig directly; derived flags keep their declared defaults so
        #   the downstream cross-config decisions stay observable.
        return ModelConfig(
            model=self.model,
            tokenizer=self.tokenizer,
            served_model_name=self.served_model_name,
            max_model_len=self.max_model_len,
            enforce_eager=self.enforce_eager,
        )

    # SOURCE: vllm/engine/arg_utils.py EngineArgs.create_engine_config (L1622-L2177)
    def create_engine_config(self, usage_context: Optional[str] = None,
                             headless: bool = False) -> VllmConfig:
        """Create the VllmConfig.

        NOTE: If VllmConfig is incompatible, we raise an error.
        """
        current_platform.pre_register_and_update()

        device_config = DeviceConfig(device=current_platform.device_type)

        # SUBTRACTED: envs.validate_environ + speculator-model detection that can
        #   override model/tokenizer (vllm/engine/arg_utils.py:L1633-L1655).

        model_config = self.create_model_config()
        self.model = model_config.model
        self.model_weights = model_config.model_weights
        self.tokenizer = model_config.tokenizer

        # SUBTRACTED: _check_feature_supported / _set_default_reasoning_config_args
        #   and the chunked-prefill + prefix-caching default derivation helpers
        #   (vllm/engine/arg_utils.py:L1657-L1678) — capability-driven defaulting;
        #   on this path the EngineArgs defaults already hold.

        # CacheConfig is the representative "flat self.* -> structured sub-config"
        # repacking; ParallelConfig/SchedulerConfig below follow the same pattern.
        cache_config = CacheConfig(
            block_size=self.block_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            cache_dtype=self.kv_cache_dtype,
            is_attention_free=model_config.is_attention_free,
            num_gpu_blocks_override=self.num_gpu_blocks_override,
            enable_prefix_caching=self.enable_prefix_caching,
        )
        # SUBTRACTED: TurboQuant boundary-layer skip, Ray runtime-env / placement
        #   group collection, the entire DP hybrid/external load-balancer
        #   derivation block, and the mamba/kernel/offload/observability/attention
        #   per-field repacking (vllm/engine/arg_utils.py:L1700-L1880, L2018-L2144)
        #   — all dossier delete items; structurally identical repacking samples.

        parallel_config = ParallelConfig(
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            data_parallel_size=self.data_parallel_size,
            distributed_executor_backend=self.distributed_executor_backend,
        )

        scheduler_config = SchedulerConfig(
            runner_type=model_config.runner_type,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_seqs=self.max_num_seqs,
            max_model_len=model_config.max_model_len,
            enable_chunked_prefill=self.enable_chunked_prefill,
            is_multimodal_model=model_config.is_multimodal_model,
            is_encoder_decoder=model_config.is_encoder_decoder,
            policy=self.scheduling_policy,
            scheduler_cls=self.scheduler_cls,
            async_scheduling=self.async_scheduling,
        )

        # CompilationConfig deep-copy + cudagraph overrides (L2113-L2130).
        compilation_config = replace(self.compilation_config)

        config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            compilation_config=compilation_config,
            optimization_level=self.optimization_level,
            performance_mode=self.performance_mode,
        )

        return config


# SOURCE: vllm/engine/arg_utils.py AsyncEngineArgs (L... subclass)
@dataclass
class AsyncEngineArgs(EngineArgs):
    # SOURCE: vllm/engine/arg_utils.py AsyncEngineArgs
    """The async variant used by AsyncLLM / the OpenAI server."""

    enable_log_requests: bool = False
    # SUBTRACTED: async-only CLI extras (vllm/engine/arg_utils.py) — only the
    #   one distinguishing field is kept to show the subclass relationship.


# ============================================================================
# Factory #1: Executor.get_class — vllm/v1/executor/abstract.py:L47-L92
# ============================================================================


# SOURCE: vllm/v1/executor/abstract.py Executor (L37-L92)
class Executor:
    """Abstract base class for vLLM executors."""

    uses_ray: bool = False
    supports_pp: bool = False

    def __init__(self, vllm_config: VllmConfig = None):
        # SOURCE: vllm/v1/executor/abstract.py Executor.__init__ (L94+)
        # SUBTRACTED: collective_rpc worker spawn / device init body
        #   (vllm/v1/executor/abstract.py:L94-L150) — instantiating real workers
        #   needs CUDA/subprocesses; this chapter only observes which CLASS the
        #   factory selects, so the body is reduced to recording the config.
        self.vllm_config = vllm_config

    @staticmethod
    def get_class(vllm_config: VllmConfig) -> type:
        # SOURCE: vllm/v1/executor/abstract.py Executor.get_class (L47-L92)
        executor_class: type
        parallel_config = vllm_config.parallel_config
        distributed_executor_backend = parallel_config.distributed_executor_backend
        # distributed_executor_backend must be set in VllmConfig.__post_init__
        if isinstance(distributed_executor_backend, type):
            if not issubclass(distributed_executor_backend, Executor):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"Executor. Got {distributed_executor_backend}."
                )
            executor_class = distributed_executor_backend
        # SUBTRACTED: distributed_executor_backend == "ray" branch (V2/V1)
        #   (vllm/v1/executor/abstract.py:L60-L68) — Ray is a dossier delete item;
        #   mp/uni/external_launcher cover the single-node story.
        elif distributed_executor_backend == "mp":
            executor_class = MultiprocExecutor
        elif distributed_executor_backend == "uni":
            executor_class = UniProcExecutor
        elif distributed_executor_backend == "external_launcher":
            executor_class = ExecutorWithExternalLauncher
        # SUBTRACTED: isinstance(str) -> resolve_obj_by_qualname dynamic resolve
        #   (vllm/v1/executor/abstract.py:L81-L87) — advanced/custom usage.
        else:
            raise ValueError(
                f"Unknown distributed executor backend: {distributed_executor_backend}"
            )
        return executor_class

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        # SOURCE: vllm/v1/executor/abstract.py Executor.supports_async_scheduling (L367-L372)
        # Base class default: executors do NOT support async scheduling unless a
        # subclass overrides this to return True.
        return False


# SOURCE: vllm/v1/executor/multiproc_executor.py MultiprocExecutor (marker stub)
class MultiprocExecutor(Executor):
    @classmethod
    def supports_async_scheduling(cls) -> bool:
        # SOURCE: vllm/v1/executor/multiproc_executor.py MultiprocExecutor.supports_async_scheduling (L496-L498)
        return True


# SOURCE: vllm/v1/executor/uniproc_executor.py UniProcExecutor (marker stub)
class UniProcExecutor(Executor):
    @classmethod
    def supports_async_scheduling(cls) -> bool:
        # SOURCE: vllm/v1/executor/uniproc_executor.py UniProcExecutor.supports_async_scheduling (L139-L141)
        return True


# SOURCE: vllm/v1/executor/uniproc_executor.py ExecutorWithExternalLauncher (L144)
class ExecutorWithExternalLauncher(UniProcExecutor):
    # ExecutorWithExternalLauncher SUBCLASSES UniProcExecutor and does NOT
    # override supports_async_scheduling, so it INHERITS True (external launcher
    # DOES support async scheduling).
    # SUBTRACTED: _init_executor / _distributed_args / collective_rpc torchrun
    #   plumbing (vllm/v1/executor/uniproc_executor.py:L144-L210) — these spawn a
    #   real worker via torchrun env vars (RANK/LOCAL_RANK/MASTER_*); this chapter
    #   only observes which CLASS the factory selects and its inherited
    #   supports_async_scheduling, so the body is reduced to the subclass marker.
    pass


# ============================================================================
# Factory #2: SchedulerConfig.get_scheduler_cls — already defined on
# SchedulerConfig above. Scheduler / AsyncScheduler marker stubs below.
# ============================================================================


# SOURCE: vllm/v1/core/sched/scheduler.py Scheduler (marker stub)
class Scheduler:
    pass


# SOURCE: vllm/v1/core/sched/async_scheduler.py AsyncScheduler (marker stub)
class AsyncScheduler(Scheduler):
    pass


# ============================================================================
# Factory #3: EngineCoreClient.make_client — vllm/v1/engine/core_client.py:L80-L130
# ============================================================================


# SOURCE: vllm/v1/engine/core_client.py InprocClient (marker stub)
class InprocClient:
    def __init__(self, vllm_config, executor_class, log_stats):
        # SOURCE: vllm/v1/engine/core_client.py InprocClient.__init__
        self.vllm_config = vllm_config
        self.executor_class = executor_class
        self.log_stats = log_stats


# SOURCE: vllm/v1/engine/core_client.py SyncMPClient (marker stub)
class SyncMPClient(InprocClient):
    pass


# SOURCE: vllm/v1/engine/core_client.py AsyncMPClient (marker stub)
class AsyncMPClient(InprocClient):
    def __init__(self, vllm_config, executor_class, log_stats,
                 client_addresses=None, client_count=1, client_index=0):
        # SOURCE: vllm/v1/engine/core_client.py AsyncMPClient.__init__
        super().__init__(vllm_config, executor_class, log_stats)


# SOURCE: vllm/v1/engine/core_client.py DPAsyncMPClient (marker stub)
class DPAsyncMPClient(AsyncMPClient):
    pass


# SOURCE: vllm/v1/engine/core_client.py DPLBAsyncMPClient (marker stub)
class DPLBAsyncMPClient(AsyncMPClient):
    pass


# SOURCE: vllm/v1/engine/core_client.py EngineCoreClient (L... base)
class EngineCoreClient:
    """Abstract base class for EngineCore IPC clients."""

    @staticmethod
    def make_client(multiprocess_mode: bool, asyncio_mode: bool,
                    vllm_config: VllmConfig, executor_class: type,
                    log_stats: bool) -> "EngineCoreClient":
        # SOURCE: vllm/v1/engine/core_client.py EngineCoreClient.make_client (L80-L103)
        # TODO: support this for debugging purposes.
        if asyncio_mode and not multiprocess_mode:
            raise NotImplementedError(
                "Running EngineCore in asyncio without multiprocessing "
                "is not currently supported."
            )

        if multiprocess_mode and asyncio_mode:
            return EngineCoreClient.make_async_mp_client(
                vllm_config, executor_class, log_stats
            )

        if multiprocess_mode and not asyncio_mode:
            return SyncMPClient(vllm_config, executor_class, log_stats)

        return InprocClient(vllm_config, executor_class, log_stats)

    @staticmethod
    def make_async_mp_client(vllm_config: VllmConfig, executor_class: type,
                             log_stats: bool,
                             client_addresses: Optional[dict] = None,
                             client_count: int = 1,
                             client_index: int = 0) -> "AsyncMPClient":
        # SOURCE: vllm/v1/engine/core_client.py EngineCoreClient.make_async_mp_client (L105-L130)
        parallel_config = vllm_config.parallel_config
        client_args = (vllm_config, executor_class, log_stats,
                       client_addresses, client_count, client_index)
        if parallel_config.data_parallel_size > 1:
            if parallel_config.data_parallel_external_lb:
                # External load balancer - client per DP rank.
                return DPAsyncMPClient(*client_args)
            # Internal load balancer - client balances to all DP ranks.
            return DPLBAsyncMPClient(*client_args)
        return AsyncMPClient(*client_args)


# ============================================================================
# Convergence point: EngineCore.__init__ — vllm/v1/engine/core.py:L94-L153
# ============================================================================


# SOURCE: vllm/v1/engine/core.py EngineCore (L94+)
class EngineCore:
    """Inner core of the vLLM engine — where the three factories' products meet."""

    # SOURCE: vllm/v1/engine/core.py EngineCore.__init__ (L116-L153)
    def __init__(self, vllm_config: VllmConfig, executor_class: type,
                 log_stats: bool, executor_fail_callback=None):
        self.log_stats = log_stats

        # Setup Model — executor_class becomes an instance here (was a class).
        self.model_executor = executor_class(vllm_config)
        # SUBTRACTED: register_failure_callback, available_gpu_memory_for_kv_cache,
        #   elastic-ep early init, _initialize_kv_caches profiling,
        #   StructuredOutputManager, KV-connector handshake, batch_queue (PP),
        #   prefix hasher, GC freeze (vllm/v1/engine/core.py:L120-L229) — engine
        #   runtime preparation; this chapter stops at scheduler selection.

        # Setup scheduler — Factory #2 turns async_scheduling into a class.
        Scheduler = vllm_config.scheduler_config.get_scheduler_cls()
        self.scheduler = Scheduler  # SUBTRACTED: real Scheduler(...) instantiation
        #   needs kv_cache_config from profiling (above, SUBTRACTED); we keep the
        #   selected class so the factory result is observable on host.


# ============================================================================
# Entry facade: LLMEngine.from_engine_args — vllm/v1/engine/llm_engine.py:L151-L177
# ============================================================================


# SOURCE: vllm/v1/engine/llm_engine.py LLMEngine (class)
class LLMEngine:
    """v1 synchronous engine facade — the start of this chapter's data flow."""

    def __init__(self, vllm_config: VllmConfig, executor_class: type,
                 log_stats: bool, usage_context: Optional[str] = None,
                 stat_loggers=None, multiprocess_mode: bool = False):
        # SOURCE: vllm/v1/engine/llm_engine.py LLMEngine.__init__
        self.vllm_config = vllm_config
        # LLMEngine always passes asyncio_mode=False (AsyncLLM uses make_async_mp_client).
        self.engine_core = EngineCoreClient.make_client(
            multiprocess_mode=multiprocess_mode,
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
        )
        # SUBTRACTED: tokenizer / processor / output-processor / stat-logger setup
        #   (vllm/v1/engine/llm_engine.py __init__ body) — runtime plumbing beyond
        #   this chapter's "assembly complete" boundary.

    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs,
                         usage_context: str = "ENGINE_CONTEXT",
                         stat_loggers=None,
                         enable_multiprocessing: bool = False) -> "LLMEngine":
        # SOURCE: vllm/v1/engine/llm_engine.py LLMEngine.from_engine_args (L151-L177)
        """Creates an LLM engine from the engine arguments."""

        # Create the engine configs (first-level mapping).
        vllm_config = engine_args.create_engine_config(usage_context)
        # Factory #1: pick the executor *class* (not instance — see EngineCore).
        executor_class = Executor.get_class(vllm_config)

        # SUBTRACTED: envs.VLLM_ENABLE_V1_MULTIPROCESSING override of
        #   enable_multiprocessing (vllm/v1/engine/llm_engine.py:L166-L168) —
        #   env-driven; caller passes the flag explicitly here.

        # Create the LLMEngine.
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            multiprocess_mode=enable_multiprocessing,
        )


# ============================================================================
# Tiny host-side helpers (replace vllm.utils.* / hashlib usage). Not vLLM
# abstractions — pure infrastructure so the file runs without importing vllm.
# ============================================================================


def _hash10(s: str) -> str:
    # SOURCE: vllm/config/utils.py safe_hash (usage pattern) — sub-config hashes.
    import hashlib
    return hashlib.sha256(s.encode()).hexdigest()[:10]


def _safe_hash(b: bytes) -> str:
    # SOURCE: vllm/config/vllm.py safe_hash(...).hexdigest() (L470-L472)
    import hashlib
    return hashlib.sha256(b).hexdigest()


def _env_get(key: str) -> Optional[str]:
    # SOURCE: vllm/config/vllm.py os.environ.get usage (L912)
    import os
    return os.environ.get(key)


def _resolve_obj_by_qualname(qualname: str):
    # SOURCE: vllm/utils/import_utils.py resolve_obj_by_qualname
    module_name, obj_name = qualname.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)
