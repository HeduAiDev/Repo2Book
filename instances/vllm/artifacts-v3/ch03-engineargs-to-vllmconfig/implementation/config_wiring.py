# Subtract-only companion for v3 ch03 «从 EngineArgs 到 VllmConfig» (装配流).
#
# FAITHFUL SUBSET of the real vLLM configuration/wiring path at pin
# v0.27.1 (6e448d0ea). It keeps vLLM's names, structure and control flow;
# it only DELETES branches approved in the dossier subtraction_plan (plus the
# mechanical deletions listed in impl-notes.md) and marks every deletion with
# `# SUBTRACTED:`. Mapping rule: take the real vLLM source, drop every
# SUBTRACTED branch, and you should get (approximately) this file.
#
# Goal line (dossier subtraction_plan.note): a minimal assembly line that runs
#   LLM(model=..., enforce_eager=...) -> EngineArgs -> create_engine_config
#   -> VllmConfig (cross-checks + async tri-state + O0-O3) -> three factories
#   -> EngineCore (inproc-capable) — single GPU, uni/mp backends, no
#   Ray/DP/multimodal/LoRA/speculative on the traced path.
#
# Runs on a CPU host WITHOUT torch/vllm: platform/env/HF seams are injectable
# stubs (see impl-notes.md Source Map). Every def/class carries a
# `# SOURCE: vllm/...:Lxxx` ref into the pinned tree.

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import MISSING, InitVar, dataclass, field, fields, is_dataclass
from enum import Enum, IntEnum
from typing import Any, Callable, Literal, Optional, Union, get_args


# ============================================================================
# Host seams — stdlib stand-ins for vllm.* infrastructure so the assembly line
# runs without the vllm package. Each mirrors the real interface subset.
# ============================================================================


# SOURCE: vllm/utils/mem_constants.py GiB_bytes
GiB_bytes = 1024**3


# SOURCE: vllm/utils/hashing.py:L103 safe_hash — md5 with sha256 FIPS fallback
def safe_hash(data: bytes, usedforsecurity: bool = True):
    # SUBTRACTED: FIPS fallback path detail (vllm/utils/hashing.py:L114-L117)
    #   — same digest call on host.
    return hashlib.md5(data, usedforsecurity=usedforsecurity)


# SOURCE: vllm/config/utils.py:L360-L378 get_hash_factors — dataclass fields
# (minus an ignore set) as the hash factors
def get_hash_factors(config: Any, ignored_factors: set[str]) -> dict[str, object]:
    # SOURCE: vllm/config/utils.py:L360 get_hash_factors
    factors: dict[str, object] = {}
    for dc_field in fields(config):
        factor = dc_field.name
        if factor in ignored_factors:
            continue
        factors[factor] = getattr(config, factor, None)
    return factors


# SOURCE: vllm/config/utils.py:L381-L383 hash_factors — canonical JSON SHA-256
def hash_factors(items: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(items, sort_keys=True, default=str).encode()).hexdigest()


# SOURCE: vllm/config/utils.py:L83-L112 get_field — borrow a default-factory
# field from a Config dataclass (single source of truth mechanism)
def get_field(cls: type, name: str) -> Any:
    # SOURCE: vllm/config/utils.py:L83 get_field
    named_field = next(f for f in fields(cls) if f.name == name)
    default = named_field.default
    default_factory = named_field.default_factory
    if default is MISSING and default_factory is MISSING:
        return None
    if default_factory is not MISSING:
        return field(default_factory=default_factory)
    return default


# SOURCE: vllm/logger.py init_logger — logging seam with the *_once helpers
def init_logger(name: str):
    import logging

    log = logging.getLogger(name)
    if not log.handlers:
        log.addHandler(logging.NullHandler())
    seen: set[str] = set()

    # SOURCE: vllm/logger.py once-messaging — lru_cache'd _print_*_once
    #   helpers (L76-L94) + _VllmLogger.{info,warning}_once methods (L118-L145,
    #   patched onto Logger at L150-L151); the seen set below re-derives the
    #   same once-filter on this host seam.
    class _Once:
        # SOURCE: vllm/logger.py once-messaging wrapper
        def __init__(self, fn):
            self._fn = fn

        def __call__(self, msg, *args):  # SOURCE: vllm/logger.py once-wrapper call
            key = (self._fn.__name__, msg)
            if key not in seen:
                seen.add(key)
                self._fn(msg, *args)

    log.info_once = _Once(log.info)
    log.warning_once = _Once(log.warning)
    return log


logger = init_logger(__name__)


# SOURCE: vllm/envs.py envs — environment flag seam (vllm/envs.py:L149 etc.)
class envs:
    # SOURCE: vllm/envs.py:L149 VLLM_ENABLE_V1_MULTIPROCESSING
    VLLM_ENABLE_V1_MULTIPROCESSING: bool = True
    # SOURCE: vllm/envs.py:L170 VLLM_USE_BREAKABLE_CUDAGRAPH
    VLLM_USE_BREAKABLE_CUDAGRAPH: bool = False
    # SOURCE: vllm/envs.py:L66 VLLM_XLA_USE_SPMD
    VLLM_XLA_USE_SPMD: bool = False
    # SOURCE: vllm/envs.py:L279 VLLM_USE_V2_MODEL_RUNNER
    VLLM_USE_V2_MODEL_RUNNER: Optional[bool] = None
    # SOURCE: vllm/envs.py VLLM_DP_* offline SPMD fallbacks
    VLLM_DP_SIZE: int = 1
    VLLM_DP_RANK: int = 0
    VLLM_DP_RANK_LOCAL: Optional[int] = None
    VLLM_DP_MASTER_IP: str = "127.0.0.1"
    VLLM_DP_MASTER_PORT: int = 29500

    # SOURCE: vllm/envs.py validate_environ — env validation seam
    @staticmethod
    def validate_environ(fail_on_environ_validation: bool = False) -> None:
        # SOURCE: vllm/envs.py validate_environ
        # SUBTRACTED: deprecated/unrecognized env-var detection
        #   (vllm/envs.py validate_environ body) — host seam no-op.
        return None


# SOURCE: vllm/usage/usage_lib.py:L111-L117 UsageContext
class UsageContext(str, Enum):
    # SUBTRACTED: UNKNOWN_CONTEXT / API_SERVER / OPENAI_BATCH_RUNNER members
    #   (vllm/usage/usage_lib.py:L112/L114/L116) — not keys this chapter's
    #   default tables read.
    LLM_CLASS = "LLM_CLASS"
    OPENAI_API_SERVER = "OPENAI_API_SERVER"
    ENGINE_CONTEXT = "ENGINE_CONTEXT"


# SOURCE: vllm/platforms/interface.py Platform (interface subset)
@dataclass
class Platform:
    # SOURCE: vllm/platforms/interface.py Platform device probe attributes
    device_type: str = "cuda"
    _is_cuda: bool = True
    _device_count: int = 1
    _total_memory: Optional[int] = 80 * GiB_bytes
    _device_name: Optional[str] = "NVIDIA H100 80GB HBM3"

    # SOURCE: vllm/platforms/interface.py Platform.is_cuda
    def is_cuda(self) -> bool:
        return self._is_cuda

    # SOURCE: vllm/platforms/interface.py Platform.is_rocm
    def is_rocm(self) -> bool:
        return False

    # SOURCE: vllm/platforms/interface.py Platform.is_tpu
    def is_tpu(self) -> bool:
        return False

    # SOURCE: vllm/platforms/interface.py Platform.is_cpu
    def is_cpu(self) -> bool:
        return False

    # SOURCE: vllm/platforms/cuda.py:L608 CudaPlatform.device_count (interface.py declares no base device_count)
    def device_count(self) -> int:
        return self._device_count

    # SOURCE: vllm/platforms/interface.py Platform.get_device_total_memory
    def get_device_total_memory(self) -> int:
        if self._total_memory is None:
            raise RuntimeError("no device on the host platform seam")
        return self._total_memory

    # SOURCE: vllm/platforms/interface.py Platform.get_device_name
    def get_device_name(self) -> str:
        if self._device_name is None:
            raise RuntimeError("no device on the host platform seam")
        return self._device_name

    # SOURCE: vllm/platforms/interface.py:L972 Platform.get_cpu_architecture (CpuArchEnum: interface.py:L79)
    def get_cpu_architecture(self) -> str:
        return "x86"

    # SOURCE: vllm/platforms/interface.py Platform.pre_register_and_update
    def pre_register_and_update(self) -> None:
        # SUBTRACTED: per-platform plugin registration / CLI option injection
        #   (vllm/platforms/*.pre_register_and_update) — host seam no-op.
        return None

    # SOURCE: vllm/platforms/interface.py Platform.apply_config_platform_defaults
    def apply_config_platform_defaults(self, vllm_config: "VllmConfig") -> None:
        # SUBTRACTED: platform-specific config overrides (ROCm/TPU/CPU)
        #   (vllm/platforms/*.apply_config_platform_defaults) — no-op on the
        #   generic CUDA path this chapter traces.
        return None

    # SOURCE: vllm/platforms/interface.py Platform.check_and_update_config
    def check_and_update_config(self, vllm_config: "VllmConfig") -> None:
        # SUBTRACTED: platform-specific final config fixes
        #   (vllm/platforms/*.check_and_update_config) — no-op seam.
        return None

    # SOURCE: vllm/platforms/interface.py Platform.support_static_graph_mode
    def support_static_graph_mode(self) -> bool:
        # cudagraph-capable platforms return True (CUDA path this chapter).
        return self._is_cuda


# Injectable singleton (tests reassign to flip device/world decisions).
current_platform = Platform()


# SOURCE: vllm/plugins/__init__.py load_general_plugins — plugin seam
def load_general_plugins() -> None:
    # SUBTRACTED: entry-point plugin discovery (vllm/plugins/__init__.py) —
    #   plugins may inject new CLI options; host seam no-op.
    return None


# SOURCE: vllm/utils/torch_utils.py:L374-L392 resolve_kv_cache_dtype_string
def resolve_kv_cache_dtype_string(kv_cache_dtype: str, model_config: "ModelConfig") -> str:
    # SUBTRACTED: HF quantization_config probing for the "auto" resolution
    #   (vllm/utils/torch_utils.py:L383-L389) — reads hf_config off disk; on
    #   the unquantized path this chapter traces, "auto" stays "auto".
    if kv_cache_dtype != "auto":
        return kv_cache_dtype
    return "auto"


# SOURCE: vllm/platforms/cpu.py CpuArchEnum (members used by arg_utils)
class CpuArchEnum(Enum):
    RISCV = "riscv"


# ============================================================================
# Enums from vllm/config/compilation.py (members this chapter branches on).
# ============================================================================


# SOURCE: vllm/config/compilation.py CompilationMode (member subset)
class CompilationMode(IntEnum):
    NONE = 0
    # SUBTRACTED: STOCK_TORCH_COMPILE / DYNAMO_TRACE_ONCE intermediate modes
    #   (vllm/config/compilation.py) — this chapter only branches on
    #   NONE vs VLLM_COMPILE.
    VLLM_COMPILE = 3


# SOURCE: vllm/config/compilation.py CUDAGraphMode (member subset)
class CUDAGraphMode(IntEnum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    # SUBTRACTED: FULL_DECODE_ONLY (vllm/config/compilation.py) — only
    #   referenced by the encoder-decoder downgrade branch, a dossier delete.
    FULL_AND_PIECEWISE = 3

    # SOURCE: vllm/config/compilation.py CUDAGraphMode.requires_piecewise_compilation
    def requires_piecewise_compilation(self) -> bool:
        return self in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL_AND_PIECEWISE)


# ============================================================================
# Structured sub-configs — faithful field subsets (vllm/config/*.py). Fields
# this chapter's control flow does not read are SUBTRACTED, never renamed.
# ============================================================================


# SOURCE: vllm/config/cache.py:L44 CacheConfig (field subset)
@dataclass
class CacheConfig:
    """Configuration for the KV cache."""

    block_size: Optional[int] = None                       # L49
    prefix_match_unit: Optional[int] = None                # L56
    gpu_memory_utilization: float = 0.92                   # L68
    cache_dtype: str = "auto"                              # L76
    is_attention_free: bool = False                        # L84
    num_gpu_blocks_override: Optional[int] = None          # L87
    sliding_window: Optional[int] = None                   # L90
    enable_prefix_caching: bool = True                     # L93
    prefix_caching_hash_algo: str = "sha256"               # L95
    calculate_kv_scales: bool = False                      # L111
    kv_cache_dtype_skip_layers: list = field(default_factory=list)  # L116
    mamba_cache_dtype: str = "auto"                        # L131
    mamba_block_size: Optional[int] = None                 # L127
    mamba_ssm_cache_dtype: str = "auto"                    # L135
    mamba_cache_mode: str = "none"                         # L139
    replayssm_buffer_len: int = 16                         # L148
    use_replayssm: bool = False                            # L152
    kv_sharing_fast_prefill: bool = False                  # L174
    kv_cache_memory_bytes: Optional[int] = None            # L182
    kv_offloading_size: Optional[float] = None             # L191
    kv_offloading_backend: str = "native"                  # L197
    # SUBTRACTED: hybrid/allocator/fp8 cache fields (vllm/config/cache.py) —
    #   not read by this chapter's assembly path.

    # SOURCE: vllm/config/cache.py:L202 CacheConfig.compute_hash
    def compute_hash(self) -> str:
        # SUBTRACTED: real ignored-factor set (vllm/config/cache.py:L203+) —
        #   all kept fields are graph-shape-relevant on this path.
        return hash_factors(get_hash_factors(self, set()))


# SOURCE: vllm/config/model.py ModelArchConfig (seam for the HF-derived arch
# metadata the cross-check reads)
@dataclass
class ModelArchConfig:
    # SOURCE: vllm/config/model.py ModelArchConfig.total_num_attention_heads
    total_num_attention_heads: int = 16


# SOURCE: vllm/config/model.py:L125 ModelConfig (field subset)
@dataclass
class ModelConfig:
    model: str = "Qwen/Qwen3-0.6B"                         # L125
    model_weights: str = ""                                # L129
    tokenizer: Optional[str] = None                        # L140
    tokenizer_mode: str = "auto"                           # L143
    trust_remote_code: bool = False                        # L164
    dtype: Any = "auto"                                    # L167
    seed: int = 0                                          # L177
    hf_config_path: Optional[str] = None                   # L187
    revision: Optional[str] = None                         # L197
    max_model_len: Optional[int] = None                    # L208
    quantization: Optional[str] = None                     # L223
    enforce_eager: bool = False                            # L235
    max_logprobs: int = 20                                 # L242
    disable_sliding_window: bool = False                   # L262
    disable_cascade_attn: bool = True                      # L266
    skip_tokenizer_init: bool = False                      # L273
    served_model_name: Optional[Union[str, list]] = None   # L283
    config_format: str = "auto"                            # L291
    hf_token: Optional[Union[bool, str]] = None            # L298
    hf_overrides: dict = field(default_factory=dict)
    tokenizer_revision: Optional[str] = None
    model_class_overrides: dict = field(default_factory=dict)
    allow_deprecated_quantization: bool = False
    quantization_config: Any = None
    runner: str = "auto"                                   # L133
    convert: str = "auto"                                  # L136
    model_arch_config: ModelArchConfig = field(default_factory=ModelArchConfig)
    # SUBTRACTED: the HF-config read (download + transformers parse + dtype
    #   resolution + multimodal/encoder-decoder detection,
    #   vllm/config/model.py ModelConfig.__init__ body) — the heaviest step in
    #   real vLLM. Here ModelConfig packs the flat fields directly; the
    #   derived flags below keep their declared defaults so downstream
    #   cross-config decisions stay observable. Tests override them to flip
    #   branches (e.g. is_chunked_prefill_supported).
    is_moe: bool = False
    is_multimodal_model: bool = False
    is_encoder_decoder: bool = False
    is_attention_free: bool = False
    is_hybrid: bool = False
    is_chunked_prefill_supported: bool = True
    is_prefix_caching_supported: bool = True
    is_diffusion: bool = False
    enable_return_routed_experts: bool = False
    architectures: Optional[list] = None
    # SUBTRACTED: ~80 further ModelConfig fields (multimodal / pooler /
    #   generation / sleep-mode..., vllm/config/model.py) — not on this
    #   chapter's single-GPU unquantized assembly path.

    @property
    def runner_type(self) -> str:
        # SOURCE: vllm/config/model.py ModelConfig.runner_type property
        # SUBTRACTED: HF-config-driven resolution — "auto" maps to "generate"
        #   on the text-model path this chapter traces.
        if self.runner == "pooling":
            return "pooling"
        return "generate"

    @property
    def architecture(self) -> Optional[str]:
        # SOURCE: vllm/config/model.py:L964 ModelConfig.architecture property
        if not self.architectures:
            return None
        return self.architectures[0]

    # SOURCE: vllm/config/model.py:L1381 ModelConfig.get_sliding_window
    def get_sliding_window(self) -> Optional[int]:
        # SUBTRACTED: hf_config sliding_window read (vllm/config/model.py) —
        #   returns None on the plain full-attention path.
        return None

    # SOURCE: vllm/config/model.py:L1309 ModelConfig.verify_with_parallel_config
    def verify_with_parallel_config(self, parallel_config: "ParallelConfig") -> None:
        total_num_attention_heads = self.model_arch_config.total_num_attention_heads
        tensor_parallel_size = parallel_config.tensor_parallel_size
        if total_num_attention_heads % tensor_parallel_size != 0:
            raise ValueError(
                f"Total number of attention heads ({total_num_attention_heads})"
                " must be divisible by tensor parallel size "
                f"({tensor_parallel_size})."
            )
        # SUBTRACTED: expert-parallelism / PP registry / decode-context-
        #   parallelism / torch_shm checks (vllm/config/model.py:L1322-L1395)
        #   — EP/PP/DCP/DP edges outside this chapter's single-GPU path.

    # SOURCE: vllm/config/model.py:L398 ModelConfig.compute_hash
    def compute_hash(self) -> str:
        ignored_factors = {
            "convert",
            "tokenizer",
            "tokenizer_mode",
            "seed",
            "hf_config_path",
            "tokenizer_revision",
            "enforce_eager",
            "disable_cascade_attn",
            "skip_tokenizer_init",
            "served_model_name",
            "config_format",
            "hf_token",
            "hf_overrides",
            "model_class_overrides",
            "runner",
        }
        # SUBTRACTED: full real ignore set (vllm/config/model.py:L410-L429+) —
        #   subset covers the kept fields.
        return hash_factors(get_hash_factors(self, ignored_factors))


# SOURCE: vllm/config/load.py:L27 LoadConfig (field subset)
@dataclass
class LoadConfig:
    # SOURCE: vllm/config/load.py:L27 LoadConfig
    load_format: str = "auto"                              # L30
    download_dir: Optional[str] = None                     # L60
    model_loader_extra_config: dict = field(default_factory=dict)  # L94
    ignore_patterns: Any = field(default_factory=lambda: ["original/**/*"])  # L100
    use_tqdm_on_load: bool = True                          # L103
    # SUBTRACTED: safetensors strategy/prefetch + pt_load_map_location
    #   (vllm/config/load.py) — loader internals, not read here.


# SOURCE: vllm/config/attention.py AttentionConfig (field subset)
@dataclass
class AttentionConfig:
    # SOURCE: vllm/config/attention.py AttentionConfig
    backend: Optional[str] = None
    flash_attn_version: Optional[int] = None
    # SUBTRACTED: further attention knobs (vllm/config/attention.py) — the
    #   override sample that writes them is a dossier delete.


# SOURCE: vllm/config/mamba.py MambaConfig (field subset)
@dataclass
class MambaConfig:
    # SOURCE: vllm/config/mamba.py MambaConfig
    enable_stochastic_rounding: bool = False
    # SUBTRACTED: backend / ssu_algorithm / philox fields (vllm/config/mamba.py)
    #   — the override sample that writes them is a dossier delete.


# SOURCE: vllm/config/kernel.py IrOpPriorityConfig (marker)
@dataclass
class IrOpPriorityConfig:
    # SOURCE: vllm/config/kernel.py IrOpPriorityConfig
    pass


# SOURCE: vllm/config/kernel.py KernelConfig (field subset)
@dataclass
class KernelConfig:
    enable_flashinfer_autotune: Optional[bool] = None
    ir_op_priority: IrOpPriorityConfig = field(default_factory=IrOpPriorityConfig)
    # SUBTRACTED: backend selection fields (moe/linear,
    #   vllm/config/kernel.py) — the override sample that writes them is a
    #   dossier delete; only autotune is observed by the optimization-level
    #   application on this path.

    # SOURCE: vllm/config/kernel.py KernelConfig.set_platform_defaults
    def set_platform_defaults(self, vllm_config: "VllmConfig") -> None:
        # SUBTRACTED: IR op-priority population from platform
        #   (vllm/config/kernel.py) — feeds the fusion predicates, themselves
        #   reduced to platform-independent defaults (see below); no-op seam
        #   preserving the call site before fusion defaults apply.
        return None

    # SOURCE: vllm/config/kernel.py KernelConfig.compute_hash
    def compute_hash(self) -> str:
        return hash_factors(get_hash_factors(self, set()))


# SOURCE: vllm/config/parallel.py:L58 EPLBConfig (field subset)
@dataclass
class EPLBConfig:
    # SOURCE: vllm/config/parallel.py:L58 EPLBConfig
    communicator: Optional[str] = None                      # L92
    use_async: bool = True                                 # L84
    # SUBTRACTED: window/step/redundant-expert fields (vllm/config/parallel.py)
    #   — EP load-balancing internals (ch34).


# SOURCE: vllm/config/fault_tolerance.py FaultToleranceConfig (marker)
@dataclass
class FaultToleranceConfig:
    # SOURCE: vllm/config/fault_tolerance.py FaultToleranceConfig
    pass


# SOURCE: vllm/config/weight_transfer.py WeightTransferConfig (marker)
@dataclass
class WeightTransferConfig:
    # SOURCE: vllm/config/weight_transfer.py WeightTransferConfig
    pass


# SOURCE: vllm/config/reasoning.py ReasoningConfig (field subset)
@dataclass
class ReasoningConfig:
    # SOURCE: vllm/config/reasoning.py ReasoningConfig
    reasoning_parser: Optional[str] = None
    # SUBTRACTED: further reasoning flags (vllm/config/reasoning.py).


# SOURCE: vllm/config/structured_outputs.py StructuredOutputsConfig (subset)
@dataclass
class StructuredOutputsConfig:
    # SOURCE: vllm/config/structured_outputs.py StructuredOutputsConfig
    reasoning_parser: Optional[str] = None
    # SUBTRACTED: grammar backend knobs (vllm/config/structured_outputs.py)
    #   — structured-output backends are ch31/ch32 territory.


# SOURCE: vllm/config/profiler.py ProfilerConfig (marker subset)
@dataclass
class ProfilerConfig:
    # SOURCE: vllm/config/profiler.py ProfilerConfig
    profiler: Optional[str] = None


# SOURCE: vllm/config/kv_transfer.py KVTransferConfig (subset)
@dataclass
class KVTransferConfig:
    kv_connector: Optional[str] = None

    @property
    def is_kv_transfer_instance(self) -> bool:
        # SOURCE: vllm/config/kv_transfer.py KVTransferConfig.is_kv_transfer_instance
        # SUBTRACTED: kv_role-based resolution — no connector on this path.
        return False


# SOURCE: vllm/config/kv_events.py KVEventsConfig (marker)
@dataclass
class KVEventsConfig:
    # SOURCE: vllm/config/kv_events.py KVEventsConfig
    pass


# SOURCE: vllm/config/ec_transfer.py ECTransferConfig (marker)
@dataclass
class ECTransferConfig:
    # SOURCE: vllm/config/ec_transfer.py ECTransferConfig
    pass


# SOURCE: vllm/config/diffusion.py DiffusionConfig (marker)
@dataclass
class DiffusionConfig:
    # SOURCE: vllm/config/diffusion.py DiffusionConfig
    pass


# SOURCE: vllm/config/speculative.py:L83 SpeculativeConfig (marker subset —
# speculative decoding internals belong to their own chapters)
@dataclass
class SpeculativeConfig:
    # SOURCE: vllm/config/speculative.py:L83 SpeculativeConfig
    method: str = "eagle"
    disable_padded_drafter_batch: bool = False
    # SUBTRACTED: full spec-decode parsing/validation
    #   (vllm/config/speculative.py) — only the two fields the async
    #   tri-state reads are kept.


# SOURCE: vllm/config/speculative.py:L61-L66 NgramGPUTypes / EagleModelTypes
NgramGPUTypes = Literal["ngram_gpu"]
MTPModelTypes = Literal[
    "mtp",
    "eagle3_mtp",
    "deepseek_mtp",
    "qwen3_mtp",
    "gpt_oss_mtp",
    "glm4_mtp",
    "hy_mtp",
    "mini_max_mtp",
    "er4_mtp",
    "Llama4XMTPModel",
    "qwen3_next_mtp",
    "ernie_mtp",
    "map_mtp",
]
# SUBTRACTED: several further MTP aliases in the real literal
#   (vllm/config/speculative.py:L44-L60) — the membership check only needs
#   representative members; "medusa" must (and does) stay outside.
DFlashModelTypes = Literal["dflash"]
EagleModelTypes = Literal["eagle", "eagle3", "extract_hidden_states", MTPModelTypes, DFlashModelTypes]


# SOURCE: vllm/config/scheduler.py:L26 SchedulerConfig
@dataclass
class SchedulerConfig:
    """Scheduler configuration."""

    max_model_len: InitVar = None  # InitVar (real L29; default via default_factory)
    is_encoder_decoder: InitVar = None  # InitVar (real L35)

    DEFAULT_MAX_NUM_BATCHED_TOKENS: int = 2048              # ClassVar L42
    DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP: int = 256  # ClassVar L43
    DEFAULT_MAX_NUM_SEQS: int = 128                         # ClassVar L44

    runner_type: str = "generate"                           # L46
    max_num_batched_tokens: int = 2048                      # L49
    max_num_scheduled_tokens: Optional[int] = None          # L56
    max_num_seqs: int = 128                                 # L63
    long_prefill_token_threshold: int = 0                   # L70
    enable_chunked_prefill: bool = True                     # L74
    is_multimodal_model: bool = False                       # L82
    max_num_encoder_input_tokens: int = field(init=False, default=0)   # L86
    encoder_cache_size: int = field(init=False, default=0)              # L93
    policy: str = "fcfs"                                    # L99
    disable_chunked_mm_input: bool = False                  # L107
    scheduler_cls: Any = None                               # L117
    disable_hybrid_kv_cache_manager: Optional[bool] = None  # L122
    scheduler_reserve_full_isl: bool = True                 # L130
    watermark: float = 0.0                                  # L136
    prefill_schedule_interval: int = 1                      # L143
    async_scheduling: Optional[bool] = None                 # L148
    stream_interval: int = 1                                # L153
    # SUBTRACTED: ClassVar/validator plumbing differences only — field set
    #   matches the real SchedulerConfig this chapter reads.

    # SOURCE: vllm/config/scheduler.py:L159-L168 SchedulerConfig.default_factory
    @staticmethod
    def default_factory(**kwargs):
        # SOURCE: vllm/config/scheduler.py:L159 SchedulerConfig.default_factory
        if "max_model_len" not in kwargs:
            kwargs["max_model_len"] = 8192
        if "is_encoder_decoder" not in kwargs:
            kwargs["is_encoder_decoder"] = False
        return SchedulerConfig(**kwargs)

    # SOURCE: vllm/config/scheduler.py:L170-L191 SchedulerConfig.get_scheduler_cls
    def get_scheduler_cls(self) -> type["SchedulerInterface"]:
        if self.scheduler_cls is None:
            if self.async_scheduling:
                # SUBTRACTED: real lazy import (vllm/v1/core/sched/...)
                return AsyncScheduler
            return Scheduler

        # The first half of this warning can be removed once the Scheduler
        # interface is finalized and we can maintain support for scheduler
        # classes that implement it
        logger.warning_once(
            "Using custom scheduler class %s. This scheduler interface is not "
            "public and compatibility may not be maintained. If you have "
            "subclassed Scheduler instead of AsyncScheduler, you will see "
            "degraded performance due to async scheduling being disabled.",
            self.scheduler_cls,
        )
        if not isinstance(self.scheduler_cls, str):
            return self.scheduler_cls
        # SUBTRACTED: resolve_obj_by_qualname for string scheduler_cls
        #   (vllm/config/scheduler.py:L191) — OOT advanced usage.
        raise ValueError(f"Unknown scheduler_cls {self.scheduler_cls!r}")

    # SOURCE: vllm/config/scheduler.py:L193-L219 SchedulerConfig.compute_hash
    def compute_hash(self) -> str:
        """Hash of the configs that affect the computation graph structure."""
        factors: list[Any] = []

        # max_num_batched_tokens need to be included in the hash due
        # to two reasons:
        # 1. LoRA creates static buffers based on max_num_batched_tokens.
        #   The tensor sizes and strides get captured in the torch.compile
        #   graph explicitly.
        # 2. Inductor decides whether using 32-bit or 64-bit indexing integer
        #   based on the data sizes. `max_num_batched_tokens` has an
        #   impact on that. For more details, please check
        #   https://github.com/vllm-project/vllm/issues/29585
        factors.append(self.max_num_batched_tokens)

        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    # SOURCE: vllm/config/scheduler.py:L227-L245 SchedulerConfig.__post_init__
    def __post_init__(self, max_model_len: int, is_encoder_decoder: bool) -> None:
        if is_encoder_decoder:
            # Chunked prefill should be disabled for encoder-decoder models.
            self.disable_chunked_mm_input = True
            self.enable_chunked_prefill = False
            self.long_prefill_token_threshold = 0
            logger.info(
                "Encoder-decoder models do not support chunked prefill nor"
                " prefix caching; disabling both."
            )

        self.max_num_encoder_input_tokens = self.max_num_batched_tokens
        self.encoder_cache_size = self.max_num_batched_tokens

        if self.enable_chunked_prefill:
            logger.info_once(
                "Chunked prefill is enabled with max_num_batched_tokens=%d.",
                self.max_num_batched_tokens,
            )

        self.verify_max_model_len(max_model_len)

    # SOURCE: vllm/config/scheduler.py:L249-L285 SchedulerConfig.verify_max_model_len
    def verify_max_model_len(self, max_model_len: int):
        if (
            self.max_num_batched_tokens < max_model_len
            and not self.enable_chunked_prefill
        ):
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) is "
                f"smaller than max_model_len ({max_model_len}). "
                "This effectively limits the maximum sequence length to "
                "max_num_batched_tokens and makes vLLM reject longer "
                "sequences. Please increase max_num_batched_tokens or "
                "decrease max_model_len."
            )

        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
                "be greater than or equal to max_num_seqs "
                f"({self.max_num_seqs})."
            )

        if self.max_num_batched_tokens > self.max_num_seqs * max_model_len:
            logger.warning(
                "max_num_batched_tokens (%d) exceeds max_num_seqs "
                "* max_model_len (%d). This may lead to unexpected behavior.",
                self.max_num_batched_tokens,
                self.max_num_seqs * max_model_len,
            )

        if self.long_prefill_token_threshold > max_model_len:
            raise ValueError(
                "long_prefill_token_threshold "
                f"({self.long_prefill_token_threshold}) cannot be greater "
                f"than the max_model_len ({max_model_len})."
            )

        return self


# ============================================================================
# ParallelConfig — vllm/config/parallel.py:L118 (field subset) + the backend
# derivation this chapter opens (L911-L956) + its scoped compute_hash
# (L774-L830).
# ============================================================================


# SOURCE: vllm/config/parallel.py:L118 ParallelConfig (field subset)
@dataclass
class ParallelConfig:
    """Configuration for the distributed execution."""

    pipeline_parallel_size: int = 1                         # L122
    tensor_parallel_size: int = 1                           # L124
    prefill_context_parallel_size: int = 1                  # L126
    data_parallel_size: int = 1                             # L129
    data_parallel_size_local: int = 1                       # L132
    data_parallel_rank: int = 0                             # L136
    data_parallel_rank_local: Optional[int] = None          # L139
    data_parallel_master_ip: str = "127.0.0.1"              # L141
    data_parallel_rpc_port: int = 29550                     # L143
    data_parallel_master_port: int = 29500                  # L145
    data_parallel_backend: str = "mp"                       # L147
    data_parallel_external_lb: bool = False                 # L149
    data_parallel_hybrid_lb: bool = False                   # L156
    is_moe_model: Optional[bool] = None                     # L163
    enable_expert_parallel: bool = False                    # L165
    enable_ep_weight_filter: bool = False                   # L167
    enable_eplb: bool = False                               # L174
    eplb_config: EPLBConfig = field(default_factory=EPLBConfig)  # L176
    all2all_backend: str = "allgather_reducescatter"       # L188
    disable_custom_all_reduce: bool = False                 # L205
    enable_elastic_ep: bool = False                         # L208
    enable_dbo: bool = False                                # L211
    worker_cls: str = "auto"                                # L259
    master_addr: str = "127.0.0.1"                          # L270
    master_port: int = 29501                                # L273
    node_rank: int = 0                                      # L276
    nnodes: int = 1                                         # L279
    disable_nccl_for_dp_synchronization: Optional[bool] = None
    distributed_executor_backend: Any = None
    distributed_timeout_seconds: Optional[int] = None
    cpu_distributed_timeout_seconds: Optional[int] = None
    fault_tolerance_config: Optional[FaultToleranceConfig] = None
    enable_fault_tolerance: bool = False
    world_size: int = field(init=False, default=1)          # L327
    data_parallel_index: int = field(init=False, default=0)
    # SUBTRACTED: placement_group/ray_runtime_env/NUMA/CP/ubatch fields
    #   (vllm/config/parallel.py) — Ray/DP/NUMA plumbing is ch34+ territory.

    def __post_init__(self) -> None:
        # SOURCE: vllm/config/parallel.py:L831-L841 ParallelConfig.__post_init__ head
        # Continue with the rest of the initialization
        self.world_size = (
            self.pipeline_parallel_size
            * self.tensor_parallel_size
            * self.prefill_context_parallel_size
        )

        if self.distributed_executor_backend == "external_launcher":
            logger.info("Using external launcher for distributed inference.")
            self.world_size *= self.data_parallel_size

        # SUBTRACTED: elastic-EP validation block (vllm/config/parallel.py:
        #   L843-L865) — EP elasticity edges (ch34), inert at enable_elastic_ep
        #   =False and off the uni/mp single-node path.

        if self.data_parallel_size > 1 or self.data_parallel_size_local == 0:
            # Data parallel was specified in the engine args.
            # SUBTRACTED: external-launcher rank inference + open-port list
            #   (vllm/config/parallel.py:L869-L884) — DP>1 machinery; the
            #   chapter's DP=1 default never enters this branch.
            if not (0 <= self.data_parallel_rank < self.data_parallel_size):
                raise ValueError(
                    f"data_parallel_rank ({self.data_parallel_rank})"
                    f" must be in the range [0, {self.data_parallel_size})"
                )
        else:
            # Otherwise fall back to env vars (e.g. for offline SPMD case).
            self.data_parallel_size = envs.VLLM_DP_SIZE
            self.data_parallel_rank = envs.VLLM_DP_RANK
            self.data_parallel_rank_local = envs.VLLM_DP_RANK_LOCAL
            self.data_parallel_master_ip = envs.VLLM_DP_MASTER_IP
            self.data_parallel_master_port = envs.VLLM_DP_MASTER_PORT

            if self.data_parallel_size > 1 and self.is_moe_model is False:
                raise ValueError(
                    "Offline data parallel mode is not supported/useful"
                    " for dense models."
                )

        self.data_parallel_index = self.data_parallel_rank

        if self.distributed_executor_backend == "external_launcher":
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            logger.info("Disabling V1 multiprocessing for external launcher.")

        if self.distributed_executor_backend is None and self.world_size_across_dp > 1:
            # We use multiprocessing by default if world_size fits on the
            # current node and we aren't in a ray placement group.

            # SUBTRACTED: ray_utils import + ray_is_available probe
            #   (vllm/config/parallel.py:L915-L918) — host seam: no ray
            #   installed, mirrors ray-not-found on the traced path.
            ray_found = False
            backend: str = "mp"
            if current_platform.is_tpu() and envs.VLLM_XLA_USE_SPMD:
                backend = "uni"
            elif current_platform.is_cuda() and self.nnodes > 1:
                backend = "mp"
            elif (
                current_platform.is_cuda()
                and current_platform.device_count() < self.world_size
            ):
                gpu_count = current_platform.device_count()
                raise ValueError(
                    f"World size ({self.world_size}) is larger than the number of "
                    f"available GPUs ({gpu_count}) in this node. If this is "
                    "intentional and you are using:\n"
                    "- ray, set '--distributed-executor-backend ray'.\n"
                    "- multiprocessing, set '--nnodes' appropriately."
                )
            elif self.data_parallel_backend == "ray":
                logger.info(
                    "Using ray distributed inference because "
                    "data_parallel_backend is ray"
                )
                backend = "ray"
            elif ray_found:
                # SUBTRACTED: placement-group / ray-initialized inspection
                #   (vllm/config/parallel.py:L941-L951) — no ray on the host
                #   seam; keeps the branch shape for the reader.
                pass
            self.distributed_executor_backend = backend
            logger.debug("Defaulting to use %s for distributed inference", backend)

        if self.distributed_executor_backend is None and self.world_size == 1:
            self.distributed_executor_backend = "uni"

        if self.max_parallel_loading_workers is not None:
            logger.warning(
                "max_parallel_loading_workers is currently "
                "not supported and will be ignored."
            )
        allowed_backends = ("mp", "uni", "external_launcher")
        if (
            self.distributed_executor_backend not in allowed_backends
            and self.nnodes > 1
        ):
            raise ValueError(
                "nnodes > 1 can only be set when distributed executor "
                "backend is mp, uni or external_launcher."
            )

        # SUBTRACTED: EPLB communicator auto-selection
        #   (vllm/config/parallel.py:L973-L988) — needs NIXL probing; inert at
        #   enable_eplb=False on this path.
        # SUBTRACTED: _verify_args pydantic model_validator
        #   (vllm/config/parallel.py:L997-L1019) — backend-type check already
        #   enforced by Executor.get_class; host dataclass skips validators.

    @property
    def world_size_across_dp(self) -> int:
        # SOURCE: vllm/config/parallel.py:L549-L551 world_size_across_dp
        """Process world size across TP, PCP, PP, and DP."""
        return self.world_size * self.data_parallel_size

    @property
    def use_batched_dp_moe(self) -> bool:
        # SOURCE: vllm/config/parallel.py:L698-L707 use_batched_dp_moe
        return (
            self.all2all_backend
            in (
                "deepep_low_latency",
                "nixl_ep",
            )
            and self.enable_expert_parallel
            and self.data_parallel_size > 1
        )

    # SOURCE: vllm/config/parallel.py:L774-L829 ParallelConfig.compute_hash
    def compute_hash(self):
        """Hash of the configs that affect the computation graph structure.

        This hash is also used for DP worker configuration validation
        to prevent hangs from mismatched collective communication patterns.
        """
        ignored_factors = {
            # Derived/runtime topology, networking, or launch details
            "data_parallel_rank",
            "data_parallel_rank_local",
            "data_parallel_size_local",
            "data_parallel_index",
            "data_parallel_backend",
            "data_parallel_external_lb",
            "data_parallel_hybrid_lb",
            "data_parallel_master_ip",
            "data_parallel_master_port",
            # SUBTRACTED: "_data_parallel_master_port_list" / "_coord_store_port"
            #   entries (vllm/config/parallel.py) — ignored-factor rows for two
            #   init=False derived fields that are themselves subtracted here.
            "data_parallel_rpc_port",
            "rank",
            "master_addr",
            "master_port",
            "node_rank",
            "nnodes",
            "max_parallel_loading_workers",
            "disable_custom_all_reduce",
            "ray_workers_use_nsight",
            "ray_runtime_env",
            "placement_group",
            "distributed_executor_backend",
            "worker_cls",
            "sd_worker_cls",
            "worker_extension_cls",
            "_api_process_count",
            "_api_process_rank",
            # NUMA binding is per-rank host-side memory locality; it does
            # not affect collective-communication semantics. When numa_bind
            # is enabled with auto-detection, each DP rank stores its own
            # NUMA node in numa_bind_nodes (see vllm/utils/numa_utils.py
            # `_get_numa_node`), which would otherwise diverge the DP hash.
            "numa_bind",
            "numa_bind_nodes",
            "numa_bind_cpus",
            "assigned_physical_gpu_ids",
        }

        factors = get_hash_factors(self, ignored_factors)
        return hash_factors(factors)


# ParallelConfig references these after definition; patch in the one field the
# warning path reads (kept out of the field list above for reading order).
# SOURCE: vllm/config/parallel.py max_parallel_loading_workers
ParallelConfig.max_parallel_loading_workers = None


# ============================================================================
# Optimization levels — vllm/config/vllm.py:L104-L327.
# ============================================================================


# SOURCE: vllm/config/vllm.py:L104-L116 OptimizationLevel
class OptimizationLevel(IntEnum):
    """Optimization level enum."""

    O0 = 0
    """O0 : No optimization. no compilation, no cudagraphs, no other
    optimization, just starting up immediately"""
    O1 = 1
    """O1: Quick optimizations. Dynamo+Inductor compilation and Piecewise
    cudagraphs"""
    O2 = 2
    """O2: Full optimizations. -O1 as well as Full and Piecewise cudagraphs."""
    O3 = 3
    """O3: Currently the same as -O2s."""


# SOURCE: vllm/config/vllm.py:L119 PerformanceMode
PerformanceMode = Literal["balanced", "interactivity", "throughput"]

# SOURCE: vllm/config/vllm.py:L121-L128 IS_QUANTIZED / IS_DENSE
# These are deliberately constant False in current vLLM (see issue #25689).
IS_QUANTIZED = False
IS_DENSE = False


# Fusion predicates are FUNCTIONS in real vLLM, lazily evaluated against the
# VllmConfig ("preset values may be predicate functions" is a chapter theme).
# The function-valued preset shape is kept; the BODIES are reduced to the
# platform-independent generic-CUDA default so the file runs on a CPU host.

# SOURCE: vllm/config/vllm.py:L131-L139 enable_norm_fusion
def enable_norm_fusion(cfg: "VllmConfig") -> bool:
    # SUBTRACTED: custom-op / kernel ir_op_priority probes (vllm/config/vllm.py)
    #   — depend on CompilationConfig.custom_ops routing + KernelConfig
    #   .ir_op_priority; both reduce to False on the generic path.
    return False


# SOURCE: vllm/config/vllm.py:L142-L152 enable_act_fusion
def enable_act_fusion(cfg: "VllmConfig") -> bool:
    # SUBTRACTED: custom-op / nvfp4 probes (vllm/config/vllm.py) — see above.
    return False


# SOURCE: vllm/config/vllm.py:L155-L175 enable_allreduce_rms_fusion
def enable_allreduce_rms_fusion(cfg: "VllmConfig") -> bool:
    # SUBTRACTED: ROCm aiter + Hopper/Blackwell + flashinfer gating
    #   (vllm/config/vllm.py:L160-L174) — hardware probes; reduced to the
    #   TP>1 prerequisite which is the user-visible knob.
    return cfg.parallel_config.tensor_parallel_size > 1


# SUBTRACTED: enable_rope_kvcache_fusion / enable_rope_kvcache_mla_fusion /
#   enable_norm_pad_fusion / enable_mla_dual_rms_norm_fusion /
#   enable_qk_norm_rope_kvcache predicate bodies (vllm/config/vllm.py:L178-L226)
#   — ROCm/AITER platform probes; represented in the presets below by the
#   static values they resolve to on the generic CUDA path (False).


# SOURCE: vllm/config/vllm.py:L229-L251 OPTIMIZATION_LEVEL_00
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
            "fuse_qk_norm_rope_kvcache": False,
            "enable_qk_norm_rope_fusion": False,
            "fuse_rope_kvcache_cat_mla": False,
        },
        "cudagraph_mode": CUDAGraphMode.NONE,
        "use_inductor_graph_partition": False,
    },
    "kernel_config": {
        "enable_flashinfer_autotune": False,
    },
}
# SOURCE: vllm/config/vllm.py:L252-L274 OPTIMIZATION_LEVEL_01
OPTIMIZATION_LEVEL_01 = {
    "compilation_config": {
        "pass_config": {
            "fuse_norm_quant": enable_norm_fusion,
            "fuse_act_quant": enable_act_fusion,
            "fuse_allreduce_rms": False,
            "fuse_attn_quant": False,
            "enable_sp": False,
            "fuse_gemm_comms": False,
            # SUBTRACTED: enable_norm_pad_fusion predicate -> generic default
            "fuse_act_padding": False,
            # SUBTRACTED: enable_mla_dual_rms_norm_fusion predicate -> default
            "fuse_mla_dual_rms_norm": False,
            "fuse_rope_kvcache": False,
            "fuse_qk_norm_rope_kvcache": False,
            "enable_qk_norm_rope_fusion": False,
            "fuse_rope_kvcache_cat_mla": False,
        },
        "cudagraph_mode": CUDAGraphMode.PIECEWISE,
        "use_inductor_graph_partition": False,
    },
    "kernel_config": {
        "enable_flashinfer_autotune": True,
    },
}
# SOURCE: vllm/config/vllm.py:L275-L297 OPTIMIZATION_LEVEL_02
OPTIMIZATION_LEVEL_02 = {
    "compilation_config": {
        "pass_config": {
            "fuse_norm_quant": enable_norm_fusion,
            "fuse_act_quant": enable_act_fusion,
            "fuse_allreduce_rms": enable_allreduce_rms_fusion,
            "fuse_attn_quant": IS_QUANTIZED,
            "enable_sp": IS_DENSE,
            "fuse_gemm_comms": IS_DENSE,
            # SUBTRACTED: enable_norm_pad_fusion predicate -> generic default
            "fuse_act_padding": False,
            # SUBTRACTED: enable_mla_dual_rms_norm_fusion predicate -> default
            "fuse_mla_dual_rms_norm": False,
            # SUBTRACTED: enable_rope_kvcache_fusion predicate -> default
            "fuse_rope_kvcache": False,
            # SUBTRACTED: enable_qk_norm_rope_kvcache predicate -> default
            "fuse_qk_norm_rope_kvcache": False,
            "enable_qk_norm_rope_fusion": False,
            # SUBTRACTED: enable_rope_kvcache_mla_fusion predicate -> default
            "fuse_rope_kvcache_cat_mla": False,
        },
        "cudagraph_mode": CUDAGraphMode.FULL_AND_PIECEWISE,
        "use_inductor_graph_partition": False,
    },
    "kernel_config": {
        "enable_flashinfer_autotune": True,
    },
}
# SOURCE: vllm/config/vllm.py:L298-L320 OPTIMIZATION_LEVEL_03 — same as O2.
OPTIMIZATION_LEVEL_03 = {
    "compilation_config": {
        "pass_config": {
            "fuse_norm_quant": enable_norm_fusion,
            "fuse_act_quant": enable_act_fusion,
            "fuse_allreduce_rms": enable_allreduce_rms_fusion,
            "fuse_attn_quant": IS_QUANTIZED,
            "enable_sp": IS_DENSE,
            "fuse_gemm_comms": IS_DENSE,
            # SUBTRACTED: predicate -> generic default (see O2 notes)
            "fuse_act_padding": False,
            "fuse_mla_dual_rms_norm": False,
            "fuse_rope_kvcache": False,
            "fuse_qk_norm_rope_kvcache": False,
            "enable_qk_norm_rope_fusion": False,
            "fuse_rope_kvcache_cat_mla": False,
        },
        "cudagraph_mode": CUDAGraphMode.FULL_AND_PIECEWISE,
        "use_inductor_graph_partition": False,
    },
    "kernel_config": {
        "enable_flashinfer_autotune": True,
    },
}

# SOURCE: vllm/config/vllm.py:L322-L327 OPTIMIZATION_LEVEL_TO_CONFIG
OPTIMIZATION_LEVEL_TO_CONFIG = {
    OptimizationLevel.O0: OPTIMIZATION_LEVEL_00,
    OptimizationLevel.O1: OPTIMIZATION_LEVEL_01,
    OptimizationLevel.O2: OPTIMIZATION_LEVEL_02,
    OptimizationLevel.O3: OPTIMIZATION_LEVEL_03,
}


# ============================================================================
# VllmConfig — the aggregate config + cross-config derivation hub.
# vllm/config/vllm.py:L331-L1600 (field subset + kept derivation spine).
# ============================================================================


# SOURCE: vllm/config/vllm.py:L331 VllmConfig
@dataclass
class VllmConfig:
    """Dataclass which contains all vllm-related configuration. This
    simplifies passing around the distinct configurations in the codebase.
    """

    model_config: Any = None                                # L338
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    scheduler_config: SchedulerConfig = field(
        default_factory=SchedulerConfig.default_factory,
    )
    device_config: "DeviceConfig" = field(default_factory=lambda: DeviceConfig())
    load_config: LoadConfig = field(default_factory=LoadConfig)
    attention_config: AttentionConfig = field(default_factory=AttentionConfig)
    mamba_config: MambaConfig = field(default_factory=MambaConfig)
    kernel_config: KernelConfig = field(default_factory=KernelConfig)
    lora_config: Any = None                                 # L360
    speculative_config: Any = None                          # L362
    diffusion_config: Any = None                            # L364
    structured_outputs_config: StructuredOutputsConfig = field(
        default_factory=StructuredOutputsConfig
    )
    quant_config: Any = None                                # L375
    compilation_config: "CompilationConfig" = field(
        default_factory=lambda: CompilationConfig()
    )
    profiler_config: ProfilerConfig = field(default_factory=ProfilerConfig)
    kv_transfer_config: Any = None                          # L388
    kv_events_config: Any = None                            # L390
    ec_transfer_config: Any = None                          # L392
    additional_config: Any = field(default_factory=dict)    # L403
    instance_id: str = ""                                   # L407
    optimization_level: OptimizationLevel = OptimizationLevel.O2  # L409
    performance_mode: PerformanceMode = "balanced"          # L415
    weight_transfer_config: Any = None                      # L422
    shutdown_timeout: int = 0                               # L425
    # SUBTRACTED: offload/observability/reasoning/ec-manager fields
    #   (vllm/config/vllm.py:L352, L371, L394, L398) — their passthrough
    #   constructions are dossier delete items (arg_utils L2415-L2462) and no
    #   kept derivation reads them.

    @property
    def use_v2_model_runner(self) -> bool:
        # SOURCE: vllm/config/vllm.py:L578 VllmConfig.use_v2_model_runner
        use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER
        if use_v2_model_runner is not None:
            return use_v2_model_runner
        # SUBTRACTED: PCP / dspark / default-V2 architecture forcing
        #   (vllm/config/vllm.py:L583-L600) — off the traced path (PCP=1).
        return False

    @property
    def max_concurrent_batches(self) -> int:
        # SOURCE: vllm/config/vllm.py:L540-L550 max_concurrent_batches
        # PP requires PP-size concurrent batches to fill the pipeline.
        # Async scheduling requires 2 concurrent batches to overlap.
        pp_size = self.parallel_config.pipeline_parallel_size
        if self.scheduler_config.async_scheduling:
            if self.use_v2_model_runner:
                return pp_size + 1
            # V1 Model Runner does not fully support async scheduling with PP.
            if pp_size <= 1:
                return 2
        return pp_size

    # SOURCE: vllm/config/vllm.py:L431-L537 VllmConfig.compute_hash
    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []

        # summarize vllm config
        vllm_factors: list[Any] = []
        # SUBTRACTED: `from vllm import __version__` (vllm/config/vllm.py:L447)
        #   — host has no vllm package; fixed stand-in so the algorithm
        #   (collect factors -> hash -> first 10 chars) is identical.
        vllm_factors.append("0.27.1")
        if self.model_config:
            vllm_factors.append(self.model_config.compute_hash())
            if (
                self.compilation_config
                and getattr(self.compilation_config, "compile_mm_encoder", False)
                and getattr(self.model_config, "multimodal_config", None)
            ):
                vllm_factors.append("None")  # mm encoder hash — no mm config here
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
        #   structured_outputs/profiler/observability/quant if-present appends
        #   (vllm/config/vllm.py:L472-L504) — same if-present-append-else-None
        #   shape; dossier delete item keeps the representative six.
        if self.compilation_config:
            vllm_factors.append(self.compilation_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.kernel_config:
            vllm_factors.append(self.kernel_config.compute_hash())
        else:
            vllm_factors.append(None)
        # SUBTRACTED: kv_transfer / ec_transfer appends
        #   (vllm/config/vllm.py:L513-L520) — dossier delete item.
        if self.additional_config:
            if isinstance(additional_config := self.additional_config, dict):
                additional_config_hash = safe_hash(
                    json.dumps(additional_config, sort_keys=True).encode(),
                    usedforsecurity=False,
                ).hexdigest()
            else:
                additional_config_hash = additional_config.compute_hash()
            vllm_factors.append(additional_config_hash)
        else:
            vllm_factors.append("None")
        factors.append(vllm_factors)

        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()[
            :10
        ]
        return hash_str

    # SOURCE: vllm/config/vllm.py:L811-L824 VllmConfig._set_config_default
    def _set_config_default(self, config_obj: Any, key: str, value: Any) -> None:
        """Set config attribute to default if not already set by user.

        Args:
            config_obj: Configuration object to update.
            key: Attribute name.
            value: Default value (static or callable).
        """
        if getattr(config_obj, key) is None:
            # Some config values are known before initialization and are
            # hard coded.
            # Other values depend on the user given configuration, so they are
            # implemented with lambda functions and decided at run time.
            setattr(config_obj, key, value(self) if callable(value) else value)

    # SOURCE: vllm/config/vllm.py:L826-L853 VllmConfig._apply_optimization_level_defaults
    def _apply_optimization_level_defaults(self, defaults: dict[str, Any]) -> None:
        """Apply optimization level defaults using self as root.

        Recursively applies values from defaults into nested config objects.
        Only fields present in defaults are overwritten.

        If the user configuration does not specify a value for a default field
        and if the default field is still None after all user selections are
        applied, then default values will be applied to the field. User specified
        fields will not be overridden by the default.

        Args:
            defaults: Dictionary of default values to apply.
        """

        def apply_recursive(config_obj: Any, config_defaults: dict[str, Any]) -> None:
            # SOURCE: vllm/config/vllm.py:L841-L851 apply_recursive closure
            """Recursively apply defaults to config_obj, using self as root."""
            for key, value in config_defaults.items():
                if not hasattr(config_obj, key):
                    continue

                current = getattr(config_obj, key)
                if isinstance(value, dict) and is_dataclass(current):
                    apply_recursive(current, value)
                else:
                    self._set_config_default(config_obj, key, value)

        apply_recursive(self, defaults)

    # SOURCE: vllm/config/vllm.py (static) VllmConfig._get_quantization_config
    @staticmethod
    def _get_quantization_config(model_config, load_config):
        # SOURCE: vllm/config/vllm.py VllmConfig._get_quantization_config
        # SUBTRACTED: quant-method metadata resolution
        #   (vllm/config/vllm.py _get_quantization_config body) — reads HF
        #   quant config; unquantized path this chapter traces yields None.
        return None

    # SOURCE: vllm/config/vllm.py:L1728 VllmConfig._set_max_num_scheduled_tokens
    def _set_max_num_scheduled_tokens(self):
        """
        In most cases, the scheduler may schedule a batch with as many tokens
        as the worker is configured to handle. However for some speculative
        decoding methods, the drafter model may insert additional slots into
        the batch when drafting. To account for this, we need to decrease the
        max_num_scheduled_tokens by an upper bound on the number of slots
        that can be added.
        """
        if self.speculative_config is not None:
            # SUBTRACTED: spec-draft slot accounting + guards
            #   (vllm/config/vllm.py:L1737-L1770) — speculative decoding is
            #   off this chapter's traced path (speculative_config None).
            pass

    # SOURCE: vllm/config/vllm.py:L972 VllmConfig.__post_init__
    def __post_init__(self):
        """Verify configs are valid & consistent with each other."""

        # To give each torch profile run a unique instance name.
        self.instance_id = f"{time.time_ns()}"

        if self.performance_mode != "balanced":
            logger.info_once("Performance mode set to '%s'.", self.performance_mode)

        self.try_verify_and_update_config()

        if self.model_config is not None:
            self.model_config.verify_with_parallel_config(self.parallel_config)
            # SUBTRACTED: verify_dual_chunk_attention_config
            #   (vllm/config/vllm.py:L985) — DCP-adjacent edge check.

            self.parallel_config.is_moe_model = self.model_config.is_moe

        if (
            self.model_config is not None
            and self.model_config.enable_return_routed_experts
        ):
            # SUBTRACTED: routed_experts field default above — the check block
            #   below is kept verbatim from the real tree.
            if self.parallel_config.pipeline_parallel_size > 1:
                raise ValueError(
                    "--enable-return-routed-experts is incompatible with "
                    "pipeline parallelism (PP > 1)."
                )

            # Incompatible with any KV connector — covers both PD disaggregation
            # (kv_producer/kv_consumer: routing captured on P can't reach D) and
            # single-instance KV offload/sharing (kv_both: slot_mapping semantics
            # change when KV blocks live outside local GPU memory, breaking the
            # slot-indexed routed_experts buffer).
            if (
                self.kv_transfer_config is not None
                and self.kv_transfer_config.is_kv_transfer_instance
            ):
                raise ValueError(
                    "--enable-return-routed-experts is incompatible with KV "
                    "connectors (PD disaggregation, KV cache offload)."
                )

        if self.lora_config is not None:
            # SUBTRACTED: LoRAConfig construction is a dossier delete; the
            #   kept branch would call lora_config.verify_with_model_config.
            pass

        if (
            self.mamba_config.enable_stochastic_rounding
            and self.cache_config.mamba_ssm_cache_dtype != "float16"
        ):
            raise ValueError(
                "Stochastic rounding for Mamba cache requires "
                "the SSM cache to be float16. Please set it explicitly, "
                "by specifying `--mamba-ssm-cache-dtype float16`, or disable "
                "stochastic rounding by not specifying "
                "`--enable-mamba-cache-stochastic-rounding`."
            )

        if self.quant_config is None and self.model_config is not None:
            self.quant_config = VllmConfig._get_quantization_config(
                self.model_config, self.load_config
            )

        # SUBTRACTED: deep_gemm auto-disable for Blackwell model types
        #   (vllm/config/vllm.py:L1033-L1050) — dossier delete item
        #   (hardware/feature edge check).

        # ---- async_scheduling tri-state decision (L1052-L1143) ----
        # SUBTRACTED: `from vllm.platforms import current_platform` /
        #   `from vllm.v1.executor.abstract import Executor` local imports
        #   (vllm/config/vllm.py:L1052-L1053) — single-module companion
        #   references the globals directly.

        executor_backend = self.parallel_config.distributed_executor_backend
        executor_class = Executor.get_class(self)
        executor_supports_async_sched = executor_class.supports_async_scheduling()
        uses_rocm_deepep_ht_dbo = (
            current_platform.is_rocm()
            and self.parallel_config.enable_dbo
            and self.parallel_config.all2all_backend == "deepep_high_throughput"
        )

        if self.scheduler_config.async_scheduling:
            # Async scheduling explicitly enabled, hard fail any incompatibilities.
            # Currently, async scheduling only support eagle speculative
            # decoding.
            if uses_rocm_deepep_ht_dbo:
                raise ValueError(
                    "Async scheduling is not compatible with ROCm DeepEP "
                    "high-throughput DBO. Please use --no-async-scheduling or "
                    "select a different all2all backend."
                )
            if self.speculative_config is not None:
                if (
                    self.speculative_config.method not in get_args(EagleModelTypes)
                    and self.speculative_config.method not in get_args(NgramGPUTypes)
                    and self.speculative_config.method != "draft_model"
                    and self.speculative_config.method != "dspark"
                ):
                    raise ValueError(
                        "Currently, async scheduling is only supported "
                        "with EAGLE/MTP/Draft Model/NGram GPU/DSpark kind of "
                        "speculative decoding"
                    )
                if self.speculative_config.disable_padded_drafter_batch:
                    raise ValueError(
                        "Async scheduling is not compatible with "
                        "disable_padded_drafter_batch=True."
                    )
            if not executor_supports_async_sched:
                raise ValueError(
                    f"`{executor_backend}` does not support async scheduling yet."
                )
        elif self.scheduler_config.async_scheduling is None:
            # Enable async scheduling unless there is an incompatible option.
            if (
                self.model_config is not None
                and self.model_config.runner_type == "pooling"
            ):
                # The current implementation of asynchronous scheduling negatively
                # impacts performance of pooling models, so we disable by default.
                logger.debug(
                    "Disabling asynchronous scheduling by default for pooling model."
                )
                self.scheduler_config.async_scheduling = False
            elif (
                self.speculative_config is not None
                and self.speculative_config.method not in get_args(EagleModelTypes)
                and self.speculative_config.method not in get_args(NgramGPUTypes)
                and self.speculative_config.method != "dspark"
            ):
                logger.warning_once(
                    "Async scheduling not supported with %s-based "
                    "speculative decoding and will be disabled.",
                    self.speculative_config.method,
                )
                self.scheduler_config.async_scheduling = False
            elif (
                self.speculative_config is not None
                and self.speculative_config.disable_padded_drafter_batch
            ):
                logger.warning_once(
                    "Async scheduling is not compatible with "
                    "disable_padded_drafter_batch=True and will be disabled.",
                )
                self.scheduler_config.async_scheduling = False
            elif not executor_supports_async_sched:
                logger.warning_once(
                    "Async scheduling will be disabled because it is not supported "
                    "with the `%s` distributed executor backend. ",
                    executor_backend,
                )
                self.scheduler_config.async_scheduling = False
            elif uses_rocm_deepep_ht_dbo:
                logger.warning_once(
                    "Async scheduling is disabled for ROCm DeepEP "
                    "high-throughput DBO because that combination can corrupt "
                    "DP+EP generation accuracy."
                )
                self.scheduler_config.async_scheduling = False
            else:
                self.scheduler_config.async_scheduling = True

        if self.parallel_config.disable_nccl_for_dp_synchronization is None:
            if self.scheduler_config.async_scheduling:
                if self.parallel_config.data_parallel_size > 1 and (
                    self.model_config is None or self.model_config.is_moe
                ):
                    logger.info_once(
                        "Disabling NCCL for DP synchronization "
                        "when using async scheduling.",
                    )
                self.parallel_config.disable_nccl_for_dp_synchronization = True
            else:
                self.parallel_config.disable_nccl_for_dp_synchronization = False

        # SUBTRACTED: cascade-attention disable for async spec decoding +
        #   mm torch_shm spawn check + Turing float32 warning
        #   (vllm/config/vllm.py:L1158-L1191) — dossier delete items
        #   (spec/mm/hardware edge cases).

        # ---- compilation / cudagraph final resolution + opt-level apply ----
        if self.model_config is not None and self.model_config.enforce_eager:
            logger.warning_once(
                "Enforce eager set, disabling torch.compile and CUDAGraphs. "
                "This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none"
            )
            self.compilation_config.mode = CompilationMode.NONE
            self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

        if os.environ.get("TORCH_COMPILE_DISABLE") == "1":
            logger.warning_once(
                "TORCH_COMPILE_DISABLE is set, disabling torch.compile. "
                "This is equivalent to setting -cc.mode=none"
            )
            self.compilation_config.mode = CompilationMode.NONE

        # SUBTRACTED: breakable-cudagraph auto-enable for specific model
        #   classes + VLLM_USE_BREAKABLE_CUDAGRAPH mode disable
        #   (vllm/config/vllm.py:L1208-L1241) — dossier delete item.

        if self.compilation_config.backend == "eager" or (
            self.compilation_config.mode is not None
            and self.compilation_config.mode != CompilationMode.VLLM_COMPILE
        ):
            logger.warning_once(
                "Inductor compilation was disabled by user settings, "
                "optimizations settings that are only active during "
                "inductor compilation will be ignored."
            )

        # SUBTRACTED: has_blocked_weights -> "+quant_fp8" custom op
        #   (vllm/config/vllm.py:L1253-L1268) — dossier delete item.

        current_platform.apply_config_platform_defaults(self)

        if self.compilation_config.mode is None:
            if self.optimization_level > OptimizationLevel.O0:
                self.compilation_config.mode = CompilationMode.VLLM_COMPILE
            else:
                self.compilation_config.mode = CompilationMode.NONE

        # By default, enable torch wrapping only when using custom Inductor lowering
        if self.compilation_config.ir_enable_torch_wrap is None:
            self.compilation_config.ir_enable_torch_wrap = (
                self.compilation_config.mode == CompilationMode.VLLM_COMPILE
                and self.compilation_config.backend == "inductor"
            )

        if all(s not in self.compilation_config.custom_ops for s in ("all", "none")):
            if (
                self.compilation_config.backend == "inductor"
                and self.compilation_config.mode != CompilationMode.NONE
            ):
                self.compilation_config.custom_ops.append("none")
            else:
                self.compilation_config.custom_ops.append("all")

        # This populates IR op priorities,
        # must happen after compilation mode and backend are decided,
        # but before fusion defaults are applied as those may depend on op priority.
        self.kernel_config.set_platform_defaults(self)

        default_config = OPTIMIZATION_LEVEL_TO_CONFIG[self.optimization_level]
        self._apply_optimization_level_defaults(default_config)
        if self.kernel_config.enable_flashinfer_autotune is None:
            raise ValueError(
                "KernelConfig.enable_flashinfer_autotune must be set after applying "
                "optimization level defaults."
            )

        # SUBTRACTED: dynamic-speculative-decoding adjustments
        #   (vllm/config/vllm.py:L1307-L1308) — spec edges off this path.

        if (
            self.compilation_config.cudagraph_mode.requires_piecewise_compilation()
            and self.compilation_config.mode != CompilationMode.VLLM_COMPILE
            and not envs.VLLM_USE_BREAKABLE_CUDAGRAPH
        ):
            logger.info_once(
                "Cudagraph mode %s is not compatible with compilation mode %s."
                "Overriding to NONE.",
                self.compilation_config.cudagraph_mode,
                self.compilation_config.mode,
            )
            self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

        # SUBTRACTED: sequence-parallelism threshold derivation +
        #   fast_moe_cold_start resolution (vllm/config/vllm.py:L1323-L1367)
        #   — dossier delete items (hardware/feature edges).

        self._set_max_num_scheduled_tokens()

        if current_platform.support_static_graph_mode():
            # if cudagraph_mode has full cudagraphs, we need to check support
            # SUBTRACTED: pooling / encoder-decoder / KV-connector cudagraph
            #   downgrades (vllm/config/vllm.py:L1373-L1422) — dossier delete
            #   item; the enforce_eager branch below is kept per the plan.

            # disable cudagraph when enforce eager execution
            if self.model_config is not None and self.model_config.enforce_eager:
                logger.info_once("Cudagraph is disabled under eager mode")
                self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
                # override related settings when enforce eager
                self.compilation_config.max_cudagraph_capture_size = 0
                self.compilation_config.cudagraph_capture_sizes = []
            else:
                self.compilation_config.cudagraph_num_of_warmups = 1

            # SUBTRACTED: _set_cudagraph_sizes() (vllm/config/vllm.py:L1434)
            #   — capture-size computation is ch19's mechanism.

        else:
            self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

        # SUBTRACTED: __post_init__ tail (vllm/config/vllm.py:L1439-L1600+) —
        #   kv-sharing prefill guard, Whisper fork warning, kv-events
        #   warnings, platform check_and_update_config, v2-runner validation,
        #   compile-ranges recomputation, splitting-ops selection, SP/PP
        #   rms_norm routing, cascade-attention final checks, ubatching
        #   a2a checks — compilation internals (ch19) and feature edges
        #   beyond this chapter's "assembly complete" boundary.

    # SOURCE: vllm/config/vllm.py:L2055 VllmConfig.try_verify_and_update_config
    def try_verify_and_update_config(self):
        if self.model_config is None:
            return

        # Avoid running try_verify_and_update_config multiple times
        if getattr(self.model_config, "config_updated", False):
            return
        self.model_config.config_updated = True

        architecture = self.model_config.architecture
        if architecture is None:
            return

        # SUBTRACTED: `from vllm.model_executor.models.config import
        #   MODELS_CONFIG_MAP, HybridAttentionMambaModelConfig` +
        #   ModelRegistry._normalize_arch (vllm/config/vllm.py:L2068-L2082) —
        #   the real map holds per-architecture rewrite hooks; the host seam
        #   map is empty (every lookup misses), matching the traced path for
        #   architectures without a registered hook.
        cls = MODELS_CONFIG_MAP.get(architecture, None)
        if cls is not None:
            cls.verify_and_update_config(self)

        # SUBTRACTED: is_hybrid / convert_type="classify" / Run:ai-URI hook
        #   dispatch (vllm/config/vllm.py:L2086-L2115) — architecture-specific
        #   rewrites whose trigger flags default off on this path.


# SOURCE: vllm/model_executor/models/config.py MODELS_CONFIG_MAP (empty seam)
MODELS_CONFIG_MAP: dict = {}


# ============================================================================
# DeviceConfig + PassConfig + CompilationConfig — vllm/config/device.py,
# vllm/config/compilation.py (field subsets).
# ============================================================================


# SOURCE: vllm/config/device.py DeviceConfig (field subset)
@dataclass
class DeviceConfig:
    # SOURCE: vllm/config/device.py DeviceConfig
    device: str = "cuda"


# SOURCE: vllm/config/compilation.py:L107 PassConfig (fusion-flag subset)
@dataclass
class PassConfig:
    """Configuration for custom Inductor passes.

    This is separate from general `CompilationConfig` so that inductor passes
    don't all have access to full configuration - that would create a cycle as
    the `PassManager` is set as a property of config.

    You must pass PassConfig to VLLmConfig constructor via the CompilationConfig
    constructor. VllmConfig's post_init does further initialization.
    If used outside of the VllmConfig, some fields may be left in an
    improper state.
    """

    # SOURCE: vllm/config/compilation.py:L107 PassConfig
    fuse_norm_quant: Optional[bool] = None                  # L121
    fuse_act_quant: Optional[bool] = None
    fuse_allreduce_rms: Optional[bool] = None
    fuse_attn_quant: Optional[bool] = None
    enable_sp: Optional[bool] = None
    fuse_gemm_comms: Optional[bool] = None
    fuse_act_padding: Optional[bool] = None
    fuse_mla_dual_rms_norm: Optional[bool] = None
    fuse_rope_kvcache: Optional[bool] = None
    fuse_qk_norm_rope_kvcache: Optional[bool] = None
    enable_qk_norm_rope_fusion: Optional[bool] = None
    fuse_rope_kvcache_cat_mla: Optional[bool] = None
    # SUBTRACTED: sp_min_token_num + fusion counters
    #   (vllm/config/compilation.py) — SP threshold derivation is a dossier
    #   delete; counters are compile-time bookkeeping.


# SOURCE: vllm/config/compilation.py:L398 CompilationConfig (field subset)
@dataclass
class CompilationConfig:
    """Configuration for compilation.

    You must pass CompilationConfig to VllmConfig constructor.
    VllmConfig's post_init does further initialization. If used outside of the
    VllmConfig, some fields will be left in an improper state.

    It contains PassConfig, which controls the custom fusion/transformation passes.
    The rest has three parts:
    """
    # SUBTRACTED: field-index list + cudagraph-vs-inductor size rationale tail
    #   of the real docstring (vllm/config/compilation.py:L408-L444) — a map
    #   over compilation fields that are ch19 territory.

    mode: Optional[CompilationMode] = None                  # L447
    cudagraph_mode: Optional[CUDAGraphMode] = None          # L607
    use_inductor_graph_partition: Optional[bool] = None     # L669
    backend: str = "inductor"
    custom_ops: list = field(default_factory=list)          # L495
    ir_enable_torch_wrap: Optional[bool] = None
    cudagraph_capture_sizes: Optional[list] = None
    max_cudagraph_capture_size: Optional[int] = None
    cudagraph_num_of_warmups: Optional[int] = None
    pass_config: PassConfig = field(default_factory=PassConfig)
    # SUBTRACTED: splitting_ops / compile_ranges / partition_hashes /
    #   cache_dir and the full __post_init__ normalization
    #   (vllm/config/compilation.py:L700-L1330) — torch.compile mechanics are
    #   ch19; this chapter only reads/writes the fields above.

    # SOURCE: vllm/config/compilation.py CompilationConfig.compute_hash
    def compute_hash(self) -> str:
        # SUBTRACTED: full graph-partition factor set
        #   (vllm/config/compilation.py) — kept factors are the ones this
        #   chapter's presets write.
        return hash_factors(
            get_hash_factors(
                self,
                {"custom_ops", "cudagraph_capture_sizes", "cudagraph_num_of_warmups"},
            )
        )


# ============================================================================
# Factory #1: Executor.get_class — vllm/v1/executor/abstract.py:L37-L92.
# ============================================================================


# SOURCE: vllm/v1/executor/abstract.py:L37 Executor
class Executor:
    """Abstract base class for vLLM executors."

    An executor is responsible for executing the model on one device,
    or it can be a distributed executor that can execute the model on
    multiple devices.
    """

    uses_ray: bool = False  # whether the executor uses Ray for orchestration.
    supports_pp: bool = False  # whether the executor supports PP

    # SOURCE: vllm/v1/executor/abstract.py:L95 Executor.__init__ (reduced)
    def __init__(self, vllm_config: Optional[VllmConfig] = None) -> None:
        # SUBTRACTED: config extraction + _init_executor + failure-callback
        #   registration + sleeping/kv-aggregator state
        #   (vllm/v1/executor/abstract.py:L99-L112) — instantiating real
        #   workers needs CUDA/subprocesses; this chapter observes WHICH class
        #   the factory selects, so the body records the config.
        self.vllm_config = vllm_config

    # SOURCE: vllm/v1/executor/abstract.py:L47-L92 Executor.get_class
    @staticmethod
    def get_class(vllm_config: VllmConfig) -> type["Executor"]:
        # SOURCE: vllm/v1/executor/abstract.py:L47 Executor.get_class
        executor_class: type[Executor]
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
        elif distributed_executor_backend == "ray":
            # SUBTRACTED: envs.VLLM_USE_RAY_V2_EXECUTOR_BACKEND sub-switch
            #   (vllm/v1/executor/abstract.py:L61-L64) — dossier delete item.
            executor_class = RayDistributedExecutor
        elif distributed_executor_backend == "mp":
            executor_class = MultiprocExecutor
        elif distributed_executor_backend == "uni":
            executor_class = UniProcExecutor
        elif distributed_executor_backend == "external_launcher":
            # TODO: make v1 scheduling deterministic
            # to support external launcher
            executor_class = ExecutorWithExternalLauncher
        # SUBTRACTED: isinstance(str) -> resolve_obj_by_qualname dynamic
        #   resolution branch (vllm/v1/executor/abstract.py:L81-L87) —
        #   dossier delete item (OOT advanced usage).
        else:
            raise ValueError(
                f"Unknown distributed executor backend: {distributed_executor_backend}"
            )
        return executor_class

    # SOURCE: vllm/v1/executor/abstract.py:L363-L368 Executor.supports_async_scheduling
    @classmethod
    def supports_async_scheduling(cls) -> bool:
        """
        Whether the executor supports async scheduling.
        """
        # SOURCE: vllm/v1/executor/abstract.py:L364 Executor.supports_async_scheduling
        return False


# SOURCE: vllm/v1/executor/uniproc_executor.py UniProcExecutor (marker stub)
class UniProcExecutor(Executor):
    # SUBTRACTED: _init_executor / worker lifecycle / collective_rpc body
    #   (vllm/v1/executor/uniproc_executor.py) — instantiating a real worker
    #   needs a device; only the class identity matters on this path.

    # SOURCE: vllm/v1/executor/uniproc_executor.py:L145-L147 supports_async_scheduling
    @classmethod
    def supports_async_scheduling(cls) -> bool:
        # SOURCE: vllm/v1/executor/uniproc_executor.py:L146 UniProcExecutor.supports_async_scheduling
        return True


# SOURCE: vllm/v1/executor/multiproc_executor.py MultiprocExecutor (marker stub)
class MultiprocExecutor(Executor):
    # SUBTRACTED: process pool spawn / worker probe / collective_rpc body
    #   (vllm/v1/executor/multiproc_executor.py) — same treatment as uni.

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L525-L527 supports_async_scheduling
    @classmethod
    def supports_async_scheduling(cls) -> bool:
        # SOURCE: vllm/v1/executor/multiproc_executor.py:L526 MultiprocExecutor.supports_async_scheduling
        return True


# SOURCE: vllm/v1/executor/ray_executor.py RayDistributedExecutor (marker stub)
class RayDistributedExecutor(Executor):
    # SUBTRACTED: Ray actor orchestration body (vllm/v1/executor/ray_executor.py)
    #   — Ray deployments are out of this chapter's single-node scope.
    uses_ray = True


# SOURCE: vllm/v1/executor/uniproc_executor.py ExecutorWithExternalLauncher
class ExecutorWithExternalLauncher(UniProcExecutor):
    # SUBTRACTED: torchrun env plumbing (RANK/LOCAL_RANK/MASTER_*)
    #   (vllm/v1/executor/uniproc_executor.py) — subclasses UniProcExecutor and
    #   does NOT override supports_async_scheduling (inherits True).
    pass


# ============================================================================
# Factory #2 target classes — vllm/v1/core/sched/{scheduler,async_scheduler}.py
# (marker stubs: the busy loop belongs to ch9/ch12; only the selected class and
# its assembly kwargs are observed here).
# ============================================================================


# SOURCE: vllm/v1/core/sched/scheduler.py Scheduler (assembly-recording stub)
class Scheduler:
    # SOURCE: vllm/v1/core/sched/scheduler.py Scheduler.__init__ (signature)
    def __init__(self, vllm_config=None, kv_cache_config=None,
                 structured_output_manager=None, include_finished_set=False,
                 log_stats=False, block_size=None, hash_block_size=None, **kw):
        # SUBTRACTED: request queues / kv cache manager / connector /
        #   spec-decode state (vllm/v1/core/sched/scheduler.py Scheduler
        #   __init__ body) — scheduling mechanics are ch9-ch12; this chapter
        #   stops at "Scheduler(...) assembled".
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.structured_output_manager = structured_output_manager
        self.block_size = block_size
        self.hash_block_size = hash_block_size


# SOURCE: vllm/v1/core/sched/async_scheduler.py AsyncScheduler (marker stub)
class AsyncScheduler(Scheduler):
    # SUBTRACTED: optimistic-advance compensation machinery (placeholders /
    #   stale-output drain / defer_block_free,
    #   vllm/v1/core/sched/async_scheduler.py) — ch12 opens this box.
    pass


# ============================================================================
# EngineArgs — flat user-facing args + the first-level mapping.
# vllm/engine/arg_utils.py:L421-L753 (field subset).
# ============================================================================


# SOURCE: vllm/engine/arg_utils.py:L421 EngineArgs
@dataclass
class EngineArgs:
    """Arguments for vLLM engine."""

    model: str = ModelConfig.model
    enable_return_routed_experts: bool = ModelConfig.enable_return_routed_experts
    model_weights: str = ModelConfig.model_weights
    served_model_name: Optional[Union[str, list]] = ModelConfig.served_model_name
    tokenizer: Optional[str] = ModelConfig.tokenizer
    hf_config_path: Optional[str] = ModelConfig.hf_config_path
    runner: str = ModelConfig.runner
    convert: str = ModelConfig.convert
    skip_tokenizer_init: bool = ModelConfig.skip_tokenizer_init
    tokenizer_mode: str = ModelConfig.tokenizer_mode
    trust_remote_code: bool = ModelConfig.trust_remote_code
    download_dir: Optional[str] = LoadConfig.download_dir
    load_format: str = LoadConfig.load_format
    config_format: str = ModelConfig.config_format
    dtype: Any = ModelConfig.dtype
    kv_cache_dtype: str = CacheConfig.cache_dtype
    seed: int = ModelConfig.seed
    max_model_len: Optional[int] = ModelConfig.max_model_len
    # Note: Specifying a custom executor backend by passing a class
    # is intended for expert use only. The API may change without
    # notice.
    distributed_executor_backend: Any = ParallelConfig.distributed_executor_backend
    pipeline_parallel_size: int = ParallelConfig.pipeline_parallel_size
    master_addr: str = ParallelConfig.master_addr
    master_port: int = ParallelConfig.master_port
    nnodes: int = ParallelConfig.nnodes
    node_rank: int = ParallelConfig.node_rank
    distributed_timeout_seconds: Optional[int] = ParallelConfig.distributed_timeout_seconds
    cpu_distributed_timeout_seconds: Optional[int] = (
        ParallelConfig.cpu_distributed_timeout_seconds
    )
    tensor_parallel_size: int = ParallelConfig.tensor_parallel_size
    prefill_context_parallel_size: int = ParallelConfig.prefill_context_parallel_size
    data_parallel_size: int = ParallelConfig.data_parallel_size
    data_parallel_rank: Optional[int] = None
    data_parallel_start_rank: Optional[int] = None
    data_parallel_size_local: Optional[int] = None
    data_parallel_address: Optional[str] = None
    data_parallel_rpc_port: Optional[int] = None
    data_parallel_hybrid_lb: bool = False
    data_parallel_external_lb: bool = False
    data_parallel_backend: str = ParallelConfig.data_parallel_backend
    enable_expert_parallel: bool = ParallelConfig.enable_expert_parallel
    all2all_backend: str = ParallelConfig.all2all_backend
    enable_eplb: bool = ParallelConfig.enable_eplb
    eplb_config: EPLBConfig = get_field(ParallelConfig, "eplb_config")
    block_size: Optional[int] = None
    enable_prefix_caching: Optional[bool] = None
    prefix_caching_hash_algo: str = CacheConfig.prefix_caching_hash_algo
    gpu_memory_utilization: float = CacheConfig.gpu_memory_utilization
    kv_cache_memory_bytes: Optional[int] = CacheConfig.kv_cache_memory_bytes
    num_gpu_blocks_override: Optional[int] = CacheConfig.num_gpu_blocks_override
    max_num_batched_tokens: Optional[int] = None
    max_num_scheduled_tokens: Optional[int] = None
    long_prefill_token_threshold: int = SchedulerConfig.long_prefill_token_threshold
    max_num_seqs: Optional[int] = None
    disable_log_stats: bool = False
    revision: Optional[str] = ModelConfig.revision
    hf_token: Optional[Union[bool, str]] = ModelConfig.hf_token
    hf_overrides: dict = get_field(ModelConfig, "hf_overrides")
    model_class_overrides: dict = get_field(ModelConfig, "model_class_overrides")
    tokenizer_revision: Optional[str] = ModelConfig.tokenizer_revision
    quantization: Optional[str] = ModelConfig.quantization
    quantization_config: Any = None
    allow_deprecated_quantization: bool = ModelConfig.allow_deprecated_quantization
    enforce_eager: bool = ModelConfig.enforce_eager
    enable_chunked_prefill: Optional[bool] = None
    disable_chunked_mm_input: bool = SchedulerConfig.disable_chunked_mm_input
    scheduler_reserve_full_isl: bool = SchedulerConfig.scheduler_reserve_full_isl
    prefill_schedule_interval: int = SchedulerConfig.prefill_schedule_interval
    watermark: float = SchedulerConfig.watermark
    disable_hybrid_kv_cache_manager: Optional[bool] = (
        SchedulerConfig.disable_hybrid_kv_cache_manager
    )
    structured_outputs_config: StructuredOutputsConfig = get_field(
        VllmConfig, "structured_outputs_config"
    )
    reasoning_parser: Optional[str] = StructuredOutputsConfig.reasoning_parser
    speculative_config: dict[str, Any] | None = None
    spec_method: str | None = None
    spec_model: str | None = None
    spec_tokens: int | None = None
    diffusion_config: dict[str, Any] | None = None
    scheduling_policy: str = SchedulerConfig.policy
    scheduler_cls: Any = SchedulerConfig.scheduler_cls
    compilation_config: CompilationConfig = get_field(VllmConfig, "compilation_config")
    cudagraph_capture_sizes: Optional[list] = CompilationConfig.cudagraph_capture_sizes
    max_cudagraph_capture_size: Optional[int] = get_field(
        CompilationConfig, "max_cudagraph_capture_size"
    )
    attention_config: AttentionConfig = get_field(VllmConfig, "attention_config")
    mamba_config: MambaConfig = get_field(VllmConfig, "mamba_config")
    kernel_config: KernelConfig = get_field(VllmConfig, "kernel_config")
    worker_cls: str = ParallelConfig.worker_cls
    profiler_config: ProfilerConfig = get_field(VllmConfig, "profiler_config")
    kv_transfer_config: Optional[KVTransferConfig] = None
    kv_events_config: Optional[KVEventsConfig] = None
    ec_transfer_config: Optional[ECTransferConfig] = None
    # SUBTRACTED: real default borrows get_field(VllmConfig,
    #   "reasoning_config") (vllm/engine/arg_utils.py:L686) — the VllmConfig
    #   reasoning field is itself a delete-item casualty (passthrough), so the
    #   borrow reduces to the same None default.
    reasoning_config: Optional[ReasoningConfig] = None
    calculate_kv_scales: bool = CacheConfig.calculate_kv_scales
    kv_cache_dtype_skip_layers: list = get_field(CacheConfig, "kv_cache_dtype_skip_layers")
    mamba_cache_dtype: str = CacheConfig.mamba_cache_dtype
    mamba_ssm_cache_dtype: str = CacheConfig.mamba_ssm_cache_dtype
    mamba_block_size: Optional[int] = get_field(CacheConfig, "mamba_block_size")
    prefix_match_unit: Optional[int] = get_field(CacheConfig, "prefix_match_unit")
    mamba_cache_mode: str = CacheConfig.mamba_cache_mode
    replayssm_buffer_len: int = CacheConfig.replayssm_buffer_len
    use_replayssm: bool = CacheConfig.use_replayssm
    enable_fault_tolerance: bool = ParallelConfig.enable_fault_tolerance
    fault_tolerance_config: Optional[FaultToleranceConfig] = get_field(
        ParallelConfig, "fault_tolerance_config"
    )
    kv_offloading_size: Optional[float] = CacheConfig.kv_offloading_size
    kv_offloading_backend: str = CacheConfig.kv_offloading_backend
    ir_op_priority: IrOpPriorityConfig = get_field(KernelConfig, "ir_op_priority")
    additional_config: dict = get_field(VllmConfig, "additional_config")
    async_scheduling: Optional[bool] = SchedulerConfig.async_scheduling
    stream_interval: int = SchedulerConfig.stream_interval
    kv_sharing_fast_prefill: bool = CacheConfig.kv_sharing_fast_prefill
    disable_nccl_for_dp_synchronization: Optional[bool] = (
        ParallelConfig.disable_nccl_for_dp_synchronization
    )
    optimization_level: OptimizationLevel = VllmConfig.optimization_level
    performance_mode: PerformanceMode = VllmConfig.performance_mode
    tokens_only: bool = False
    shutdown_timeout: int = 0
    weight_transfer_config: Optional[WeightTransferConfig] = get_field(
        VllmConfig, "weight_transfer_config"
    )
    fail_on_environ_validation: bool = False
    model_loader_extra_config: dict = get_field(LoadConfig, "model_loader_extra_config")
    ignore_patterns: Any = get_field(LoadConfig, "ignore_patterns")
    use_tqdm_on_load: bool = LoadConfig.use_tqdm_on_load
    # SUBTRACTED: the several-hundred remaining EngineArgs fields — multimodal
    #   / LoRA / speculative extras / KV-transfer detail / observability /
    #   NUMA / CP / DBO / EP weight filter... (vllm/engine/arg_utils.py
    #   :L426-L753) — not on this chapter's single-GPU, unquantized,
    #   no-feature assembly path (dossier embed elide blesses the same cut).
    # SUBTRACTED: add_cli_args / from_cli / CLI parser plumbing
    #   (vllm/engine/arg_utils.py:L822-L1669) — argparse wiring needs
    #   FlexibleArgumentParser and cannot run on host; the assembly line
    #   under study starts at an already-constructed EngineArgs.

    # SOURCE: vllm/engine/arg_utils.py:L755 EngineArgs.__post_init__
    def __post_init__(self):
        # support `EngineArgs(compilation_config={...})`
        # without having to manually construct a
        # CompilationConfig object
        if isinstance(self.compilation_config, dict):
            self.compilation_config = CompilationConfig(**self.compilation_config)
        if isinstance(self.attention_config, dict):
            self.attention_config = AttentionConfig(**self.attention_config)
        if isinstance(self.mamba_config, dict):
            self.mamba_config = MambaConfig(**self.mamba_config)
        if isinstance(self.kernel_config, dict):
            self.kernel_config = KernelConfig(**self.kernel_config)
        if isinstance(self.eplb_config, dict):
            self.eplb_config = EPLBConfig(**self.eplb_config)
        if isinstance(self.weight_transfer_config, dict):
            self.weight_transfer_config = WeightTransferConfig(
                **self.weight_transfer_config
            )
        if isinstance(self.fault_tolerance_config, dict):
            if not self.enable_fault_tolerance:
                logger.warning(
                    "--fault-tolerance-config was passed. Fault tolerance is being "
                    "automatically enabled."
                )
                self.enable_fault_tolerance = True
            self.fault_tolerance_config = FaultToleranceConfig(
                **self.fault_tolerance_config
            )
        if isinstance(self.ir_op_priority, dict):
            self.ir_op_priority = IrOpPriorityConfig(**self.ir_op_priority)

        # SUBTRACTED: resolve_quantization_config (vllm/config/quantization.py)
        #   — online-shorthand resolution; the traced path passes quantization
        #   =None straight through.
        self.quantization_config = self.quantization_config

        # Setup plugins
        load_general_plugins()

        # SUBTRACTED: HF_HUB_OFFLINE model/tokenizer path replacement
        #   (vllm/engine/arg_utils.py:L796-L820) — dossier delete item
        #   (offline-HF branch).

    # SOURCE: vllm/engine/arg_utils.py:L1676 EngineArgs.create_model_config
    def create_model_config(self) -> ModelConfig:
        if not envs.VLLM_ENABLE_V1_MULTIPROCESSING:
            logger.warning(
                "The global random seed is set to %d. Since "
                "VLLM_ENABLE_V1_MULTIPROCESSING is set to False, this may "
                "affect the random state of the Python process that "
                "launched vLLM.",
                self.seed,
            )

        # SUBTRACTED: ~50 further kwargs (multimodal / pooler / generation /
        #   sleep-mode / video-pruning..., vllm/engine/arg_utils.py
        #   :L1720-L1752) — the HF-config read inside real ModelConfig is the
        #   heaviest step; the kept subset carries the fields this chapter's
        #   derivations read.
        return ModelConfig(
            model=self.model,
            model_weights=self.model_weights,
            hf_config_path=self.hf_config_path,
            runner=self.runner,
            convert=self.convert,
            tokenizer=self.tokenizer,
            tokenizer_mode=self.tokenizer_mode,
            trust_remote_code=self.trust_remote_code,
            dtype=self.dtype,
            seed=self.seed,
            revision=self.revision,
            hf_token=self.hf_token,
            hf_overrides=self.hf_overrides,
            model_class_overrides=self.model_class_overrides,
            tokenizer_revision=self.tokenizer_revision,
            max_model_len=self.max_model_len,
            quantization=self.quantization,
            quantization_config=self.quantization_config,
            allow_deprecated_quantization=self.allow_deprecated_quantization,
            enforce_eager=self.enforce_eager,
            skip_tokenizer_init=self.skip_tokenizer_init,
            enable_return_routed_experts=self.enable_return_routed_experts,
            served_model_name=self.served_model_name,
            config_format=self.config_format,
        )

    # SOURCE: vllm/engine/arg_utils.py:L1763 EngineArgs.create_load_config
    def create_load_config(self) -> LoadConfig:
        if self.quantization == "bitsandbytes":
            self.load_format = "bitsandbytes"

        # SUBTRACTED: tensorizer extra-config handling
        #   (vllm/engine/arg_utils.py:L1767-L1776 + validate_tensorizer_args)
        #   — external package path.

        return LoadConfig(
            load_format=self.load_format,
            download_dir=self.download_dir,
            model_loader_extra_config=self.model_loader_extra_config,
            ignore_patterns=self.ignore_patterns,
            use_tqdm_on_load=self.use_tqdm_on_load,
        )

    # SOURCE: vllm/engine/arg_utils.py:L1790 EngineArgs.create_speculative_config
    def create_speculative_config(
        self,
        target_model_config: ModelConfig,
        target_parallel_config: ParallelConfig,
    ) -> Optional[SpeculativeConfig]:
        """Initializes and returns a SpeculativeConfig object based on
        `speculative_config`.
        """
        for flag, key, value in (
            ("--spec-method", "method", self.spec_method),
            ("--spec-model", "model", self.spec_model),
            ("--spec-tokens", "num_speculative_tokens", self.spec_tokens),
        ):
            if value is None:
                continue
            if self.speculative_config is None:
                self.speculative_config = {}
            if key in self.speculative_config:
                raise ValueError(
                    f"{flag} and --speculative-config['{key}'] are mutually exclusive"
                )
            self.speculative_config[key] = value

        if self.speculative_config is None:
            return None

        self.speculative_config = {
            k.replace("-", "_"): v for k, v in self.speculative_config.items()
        }

        # Note(Shangming): These parameters are not obtained from the cli arg
        # '--speculative-config' and must be passed in when creating the engine
        # config.
        self.speculative_config.update(
            {
                "target_model_config": target_model_config,
                "target_parallel_config": target_parallel_config,
            }
        )
        # SUBTRACTED: SpeculativeConfig full validation — stub accepts the
        #   dict; only method/disable_padded_drafter_batch are read here.
        kwargs = {
            k: v
            for k, v in self.speculative_config.items()
            if k in ("method", "disable_padded_drafter_batch")
        }
        return SpeculativeConfig(**kwargs)

    # SOURCE: vllm/engine/arg_utils.py:L1872 EngineArgs.create_diffusion_config
    def create_diffusion_config(self) -> Optional[DiffusionConfig]:
        if self.diffusion_config is None:
            return None
        cfg = self.diffusion_config
        if isinstance(cfg, str):
            cfg = json.loads(cfg)
        return DiffusionConfig(**cfg)

    # SUBTRACTED: create_observability_config (vllm/engine/arg_utils.py
    #   :L1880-L1894) — observability passthrough is a dossier delete item.

    # SOURCE: vllm/engine/arg_utils.py:L1896 EngineArgs.create_engine_config
    def create_engine_config(
        self,
        usage_context: UsageContext | None = None,
        headless: bool = False,
    ) -> VllmConfig:
        """
        Create the VllmConfig.

        NOTE: If VllmConfig is incompatible, we raise an error.
        """
        current_platform.pre_register_and_update()

        device_config = DeviceConfig(device=current_platform.device_type)

        envs.validate_environ(self.fail_on_environ_validation)

        # SUBTRACTED: speculator detection & model/tokenizer override
        #   (vllm/engine/arg_utils.py:L1912-L1927) — dossier delete item
        #   (special-model branch; create_model_config behavior unchanged).

        model_config = self.create_model_config()
        self.model = model_config.model
        self.model_weights = model_config.model_weights
        self.tokenizer = model_config.tokenizer

        self._check_feature_supported()
        self._set_default_chunked_prefill_and_prefix_caching_args(model_config)
        self._set_default_reasoning_config_args()
        sliding_window: Optional[int] = None
        layer_types = None  # SUBTRACTED: hf_text_config.layer_types read
        if layer_types is None or all(lt == "sliding_attention" for lt in layer_types):
            # Only set CacheConfig.sliding_window if the model is all sliding
            # window. Otherwise CacheConfig.sliding_window will override the
            # global layers in interleaved sliding window models.
            sliding_window = model_config.get_sliding_window()

        # Resolve "auto" kv_cache_dtype to actual value from model config
        resolved_cache_dtype = resolve_kv_cache_dtype_string(
            self.kv_cache_dtype, model_config
        )

        assert self.enable_prefix_caching is not None, (
            "enable_prefix_caching must be set by this point"
        )

        cache_config = CacheConfig(
            block_size=self.block_size,  # type: ignore[arg-type]
            gpu_memory_utilization=self.gpu_memory_utilization,
            kv_cache_memory_bytes=self.kv_cache_memory_bytes,
            cache_dtype=resolved_cache_dtype,  # type: ignore[arg-type]
            is_attention_free=model_config.is_attention_free,
            num_gpu_blocks_override=self.num_gpu_blocks_override,
            sliding_window=sliding_window,
            enable_prefix_caching=self.enable_prefix_caching,
            prefix_caching_hash_algo=self.prefix_caching_hash_algo,
            calculate_kv_scales=self.calculate_kv_scales,
            kv_cache_dtype_skip_layers=self.kv_cache_dtype_skip_layers,
            kv_sharing_fast_prefill=self.kv_sharing_fast_prefill,
            mamba_cache_dtype=self.mamba_cache_dtype,
            mamba_ssm_cache_dtype=self.mamba_ssm_cache_dtype,
            mamba_block_size=self.mamba_block_size,
            prefix_match_unit=self.prefix_match_unit,
            mamba_cache_mode=self.mamba_cache_mode,
            replayssm_buffer_len=self.replayssm_buffer_len,
            use_replayssm=self.use_replayssm,
            kv_offloading_size=self.kv_offloading_size,
            kv_offloading_backend=self.kv_offloading_backend,
        )

        # SUBTRACTED: TurboQuant boundary-layer patch
        #   (vllm/engine/arg_utils.py:L1978-L1987) — dossier delete item.

        # SUBTRACTED: Ray runtime-env + placement-group collection
        #   (vllm/engine/arg_utils.py:L1989-L2014) — dossier delete item
        #   (Ray-only; single-node mp/uni never touches it).

        # SUBTRACTED: DP load-balancer derivation block (hybrid/external/rank/
        #   size_local/address, vllm/engine/arg_utils.py:L2016-L2189) —
        #   dossier delete item. DP=1 defaults substituted inline below
        #   (mirrors the else-branches at L2158-L2161/L2177-L2179/L2185-L2189;
        #   the address else-branch's `self.master_addr or` fallback folds
        #   away — master_addr defaults to None on this traced path).
        data_parallel_external_lb = (
            self.data_parallel_external_lb or self.data_parallel_rank is not None
        )
        data_parallel_size_local = (
            self.data_parallel_size_local
            if self.data_parallel_size_local is not None
            else self.data_parallel_size
        )
        data_parallel_address = (
            self.data_parallel_address
            if self.data_parallel_address is not None
            else ParallelConfig.data_parallel_master_ip
        )
        data_parallel_rpc_port = (
            self.data_parallel_rpc_port
            if self.data_parallel_rpc_port is not None
            else ParallelConfig.data_parallel_rpc_port
        )

        if self.tokens_only and not model_config.skip_tokenizer_init:
            model_config.skip_tokenizer_init = True
            logger.info("Skipping tokenizer initialization for tokens-only mode.")

        parallel_config = ParallelConfig(
            pipeline_parallel_size=self.pipeline_parallel_size,
            tensor_parallel_size=self.tensor_parallel_size,
            prefill_context_parallel_size=self.prefill_context_parallel_size,
            data_parallel_size=self.data_parallel_size,
            data_parallel_rank=self.data_parallel_rank or 0,
            data_parallel_external_lb=data_parallel_external_lb,
            data_parallel_size_local=data_parallel_size_local,
            master_addr=self.master_addr,
            master_port=self.master_port,
            nnodes=self.nnodes,
            node_rank=self.node_rank,
            distributed_timeout_seconds=self.distributed_timeout_seconds,
            cpu_distributed_timeout_seconds=self.cpu_distributed_timeout_seconds,
            data_parallel_master_ip=data_parallel_address,
            data_parallel_rpc_port=data_parallel_rpc_port,
            data_parallel_backend=self.data_parallel_backend,
            data_parallel_hybrid_lb=self.data_parallel_hybrid_lb,
            is_moe_model=model_config.is_moe,
            enable_expert_parallel=self.enable_expert_parallel,
            all2all_backend=self.all2all_backend,
            enable_eplb=self.enable_eplb,
            eplb_config=self.eplb_config,
            worker_cls=self.worker_cls,
            disable_nccl_for_dp_synchronization=self.disable_nccl_for_dp_synchronization,
            fault_tolerance_config=self.fault_tolerance_config,
            enable_fault_tolerance=self.enable_fault_tolerance,
            distributed_executor_backend=self.distributed_executor_backend,
        )
        # SUBTRACTED: ~25 further ParallelConfig kwargs (EPLB strategy / NUMA /
        #   fault tolerance detail / CP fields / ray nsight / device ids...,
        #   vllm/engine/arg_utils.py:L2214-L2245) — same repacking shape; the
        #   kept kwargs carry everything this chapter reads.

        speculative_config = self.create_speculative_config(
            target_model_config=model_config,
            target_parallel_config=parallel_config,
        )
        diffusion_config = self.create_diffusion_config()

        self._set_default_max_num_seqs_and_batched_tokens_args(
            usage_context,
            model_config,
            parallel_config,
        )

        assert self.max_num_batched_tokens is not None, (
            "max_num_batched_tokens must be set by this point"
        )
        assert self.max_num_seqs is not None, "max_num_seqs must be set by this point"
        assert self.enable_chunked_prefill is not None, (
            "enable_chunked_prefill must be set by this point"
        )
        assert model_config.max_model_len is not None, (
            "max_model_len must be set by this point"
        )
        scheduler_config = SchedulerConfig(
            runner_type=model_config.runner_type,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_scheduled_tokens=self.max_num_scheduled_tokens,
            max_num_seqs=self.max_num_seqs,
            max_model_len=model_config.max_model_len,
            enable_chunked_prefill=self.enable_chunked_prefill,
            disable_chunked_mm_input=self.disable_chunked_mm_input,
            is_multimodal_model=model_config.is_multimodal_model,
            is_encoder_decoder=model_config.is_encoder_decoder,
            policy=self.scheduling_policy,
            scheduler_cls=self.scheduler_cls,
            long_prefill_token_threshold=self.long_prefill_token_threshold,
            scheduler_reserve_full_isl=self.scheduler_reserve_full_isl,
            watermark=self.watermark,
            prefill_schedule_interval=self.prefill_schedule_interval,
            disable_hybrid_kv_cache_manager=self.disable_hybrid_kv_cache_manager,
            async_scheduling=self.async_scheduling,
            stream_interval=self.stream_interval,
        )

        # SUBTRACTED: LoRA config construction + LoRA/spec budget check
        #   (vllm/engine/arg_utils.py:L2291-L2329) — dossier delete item.

        # bitsandbytes pre-quantized model need a specific model loader
        if model_config.quantization == "bitsandbytes":
            self.quantization = self.load_format = "bitsandbytes"

        # SUBTRACTED: attention / mamba / kernel / ir_op_priority override
        #   samples (vllm/engine/arg_utils.py:L2335-L2411) — dossier delete
        #   item; the same flat->structured repacking pattern CacheConfig
        #   already shows.

        load_configs = self.create_load_config()

        # SUBTRACTED: reasoning-parser passthrough + observability config
        #   (vllm/engine/arg_utils.py:L2415-L2424) — dossier delete item.

        # Compilation config overrides
        from copy import deepcopy

        compilation_config = deepcopy(self.compilation_config)
        if self.cudagraph_capture_sizes is not None:
            if compilation_config.cudagraph_capture_sizes is not None:
                raise ValueError(
                    "cudagraph_capture_sizes and compilation_config."
                    "cudagraph_capture_sizes are mutually exclusive"
                )
            compilation_config.cudagraph_capture_sizes = self.cudagraph_capture_sizes
        if self.max_cudagraph_capture_size is not None:
            if compilation_config.max_cudagraph_capture_size is not None:
                raise ValueError(
                    "max_cudagraph_capture_size and compilation_config."
                    "max_cudagraph_capture_size are mutually exclusive"
                )
            compilation_config.max_cudagraph_capture_size = (
                self.max_cudagraph_capture_size
            )

        # SUBTRACTED: offload config + gdn/kda additional_config passthrough
        #   (vllm/engine/arg_utils.py:L2445-L2462) — dossier delete item.

        config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            load_config=load_configs,
            attention_config=self.attention_config,
            mamba_config=self.mamba_config,
            kernel_config=self.kernel_config,
            speculative_config=speculative_config,
            diffusion_config=diffusion_config,
            structured_outputs_config=self.structured_outputs_config,
            compilation_config=compilation_config,
            kv_transfer_config=self.kv_transfer_config,
            kv_events_config=self.kv_events_config,
            ec_transfer_config=self.ec_transfer_config,
            profiler_config=self.profiler_config,
            additional_config=self.additional_config,
            optimization_level=self.optimization_level,
            performance_mode=self.performance_mode,
            weight_transfer_config=self.weight_transfer_config,
            shutdown_timeout=self.shutdown_timeout,
        )
        # SUBTRACTED: lora/offload/observability/reasoning kwargs
        #   (vllm/engine/arg_utils.py:L2471, L2477-L2484) — their
        #   constructions are the dossier delete items above.

        return config

    # SOURCE: vllm/engine/arg_utils.py:L2495 EngineArgs._check_feature_supported
    def _check_feature_supported(self):
        """Raise an error if the feature is not supported."""
        if self.pipeline_parallel_size > 1:
            supports_pp = getattr(
                self.distributed_executor_backend, "supports_pp", False
            )
            if not supports_pp and self.distributed_executor_backend not in (
                ParallelConfig.distributed_executor_backend,
                "ray",
                "mp",
                "external_launcher",
            ):
                name = (
                    "Pipeline Parallelism without Ray distributed "
                    "executor or multiprocessing executor or external "
                    "launcher"
                )
                raise ValueError(f"Feature {name} is not supported.")

    # SOURCE: vllm/engine/arg_utils.py:L2515 EngineArgs.get_batch_defaults
    @classmethod
    def get_batch_defaults(  # SOURCE: vllm/engine/arg_utils.py:L2515 EngineArgs.get_batch_defaults
        cls,
        world_size: int,
    ) -> tuple[dict[UsageContext | None, int], dict[UsageContext | None, int]]:
        default_max_num_batched_tokens: dict[UsageContext | None, int]
        default_max_num_seqs: dict[UsageContext | None, int]

        # When no user override, set the default values based on the usage
        # context.
        # Use different default values for different hardware.

        # Try to query the device name on the current platform. If it fails,
        # it may be because the platform that imports vLLM is not the same
        # as the platform that vLLM is running on (e.g. the case of scaling
        # vLLM with Ray) and has no GPUs. In this case we use the default
        # values for non-H100/H200 GPUs.
        try:
            device_memory = current_platform.get_device_total_memory()
            device_name = current_platform.get_device_name().lower()
        except Exception:
            # This is only used to set default_max_num_batched_tokens
            device_memory = 0
            device_name = ""

        # NOTE(Kuntai): Setting large `max_num_batched_tokens` for A100 reduces
        # throughput, see PR #17885 for more details.
        # So here we do an extra device name check to prevent such regression.
        if device_memory >= 70 * GiB_bytes and "a100" not in device_name:
            # For GPUs like H100 and MI300x, use larger default values.
            default_max_num_batched_tokens = {
                UsageContext.LLM_CLASS: 16384,
                UsageContext.OPENAI_API_SERVER: 8192,
            }
            default_max_num_seqs = {
                UsageContext.LLM_CLASS: 1024,
                UsageContext.OPENAI_API_SERVER: 1024,
            }
        else:
            # TODO(woosuk): Tune the default values for other hardware.
            default_max_num_batched_tokens = {
                UsageContext.LLM_CLASS: 8192,
                UsageContext.OPENAI_API_SERVER: 2048,
            }
            default_max_num_seqs = {
                UsageContext.LLM_CLASS: 256,
                UsageContext.OPENAI_API_SERVER: 256,
            }

        # SUBTRACTED: TPU (V6E/V5E/V5P) and CPU platform default overrides
        #   (vllm/engine/arg_utils.py:L2565-L2594) — dossier delete item; the
        #   GPU path above is unchanged.

        return default_max_num_batched_tokens, default_max_num_seqs

    # SOURCE: vllm/engine/arg_utils.py:L2598 _set_default_chunked_prefill_and_prefix_caching_args
    def _set_default_chunked_prefill_and_prefix_caching_args(
        self, model_config: ModelConfig
    ) -> None:
        default_chunked_prefill = model_config.is_chunked_prefill_supported
        # Hybrid models support prefix caching but keep it opt-in for now
        # while the feature matures.
        default_prefix_caching = (
            model_config.is_prefix_caching_supported and not model_config.is_hybrid
        )

        if self.enable_chunked_prefill is None:
            self.enable_chunked_prefill = default_chunked_prefill

            logger.debug(
                "%s chunked prefill by default",
                "Enabling" if default_chunked_prefill else "Disabling",
            )
        elif (
            model_config.runner_type == "generate"
            and not self.enable_chunked_prefill
            and default_chunked_prefill
        ):
            logger.warning_once(
                "This model does not officially support disabling chunked prefill. "
                "Disabling this manually may cause the engine to crash "
                "or produce incorrect outputs.",
            )
        elif (
            model_config.runner_type == "pooling"
            and self.enable_chunked_prefill
            and not default_chunked_prefill
        ):
            logger.warning_once(
                "This model does not officially support chunked prefill. "
                "Enabling this manually may cause the engine to crash "
                "or produce incorrect outputs.",
            )

        if self.enable_prefix_caching is None:
            self.enable_prefix_caching = default_prefix_caching

            logger.debug(
                "%s prefix caching by default",
                "Enabling" if default_prefix_caching else "Disabling",
            )
        elif (
            model_config.runner_type == "pooling"
            and self.enable_prefix_caching
            and not default_prefix_caching
        ):
            logger.warning_once(
                "This model does not officially support prefix caching. "
                "Enabling this manually may cause the engine to crash "
                "or produce incorrect outputs.",
            )

        # Disable chunked prefill and prefix caching for:
        # RISCV CPUs in V1
        if current_platform.is_cpu() and current_platform.get_cpu_architecture() in (
            CpuArchEnum.RISCV,
        ):
            logger.info(
                "Chunked prefill is not supported for"
                "RISC-V CPUs; "
                "disabling it for V1 backend."
            )
            self.enable_chunked_prefill = False
            logger.info(
                "Prefix caching is not supported for "
                "RISC-V CPUs; "
                "disabling it for V1 backend."
            )
            self.enable_prefix_caching = False

    # SOURCE: vllm/engine/arg_utils.py:L2672 _set_default_reasoning_config_args
    def _set_default_reasoning_config_args(self):
        if not self.reasoning_parser:
            return
        if self.reasoning_config is None:
            self.reasoning_config = ReasoningConfig()
        self.reasoning_config.reasoning_parser = self.reasoning_parser

    # SUBTRACTED: _get_min_mm_batched_tokens (vllm/engine/arg_utils.py
    #   :L2679-L2710) — multimodal prefix-LM floor; needs MULTIMODAL_REGISTRY
    #   and is_multimodal_model=False on this path.

    # SOURCE: vllm/engine/arg_utils.py:L2712 _set_default_max_num_seqs_and_batched_tokens_args
    def _set_default_max_num_seqs_and_batched_tokens_args(
        self,
        usage_context: UsageContext | None,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ):
        world_size = self.pipeline_parallel_size * self.tensor_parallel_size
        (
            default_max_num_batched_tokens,
            default_max_num_seqs,
        ) = self.get_batch_defaults(world_size)

        orig_max_num_batched_tokens = self.max_num_batched_tokens
        orig_max_num_seqs = self.max_num_seqs

        if self.max_num_batched_tokens is None:
            if parallel_config.use_batched_dp_moe:
                self.max_num_batched_tokens = (
                    SchedulerConfig.DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP
                )
            else:
                self.max_num_batched_tokens = default_max_num_batched_tokens.get(
                    usage_context,
                    SchedulerConfig.DEFAULT_MAX_NUM_BATCHED_TOKENS,
                )

        if self.max_num_seqs is None:
            self.max_num_seqs = default_max_num_seqs.get(
                usage_context,
                SchedulerConfig.DEFAULT_MAX_NUM_SEQS,
            )

        # If throughput mode is set, double max_num_batched_tokens and max_num_seqs.
        if self.performance_mode == "throughput":
            if orig_max_num_batched_tokens is None:
                self.max_num_batched_tokens *= 2
            if orig_max_num_seqs is None:
                self.max_num_seqs *= 2

        if orig_max_num_batched_tokens is None:
            assert model_config.max_model_len is not None, (
                "max_model_len must be set by this point"
            )
            if not self.enable_chunked_prefill:
                # If max_model_len is too short, use the default for higher throughput.
                self.max_num_batched_tokens = max(
                    model_config.max_model_len,
                    self.max_num_batched_tokens,
                )

            # SUBTRACTED: multimodal prefix-LM token floor
            #   (vllm/engine/arg_utils.py:L2762-L2777) — is_multimodal_model
            #   False on this path.

            # When using default settings,
            # Ensure max_num_batched_tokens does not exceed model limit.
            # Some models (e.g., Whisper) have embeddings tied to max length.
            self.max_num_batched_tokens = min(
                self.max_num_seqs * model_config.max_model_len,
                self.max_num_batched_tokens,
            )

            logger.debug(
                "Defaulting max_num_batched_tokens to %d for %s usage context.",
                self.max_num_batched_tokens,
                usage_context.value if usage_context else None,
            )

        if orig_max_num_seqs is None:
            assert self.max_num_batched_tokens is not None  # For type checking
            self.max_num_seqs = min(self.max_num_seqs, self.max_num_batched_tokens)

            logger.debug(
                "Defaulting max_num_seqs to %d for %s usage context.",
                self.max_num_seqs,
                usage_context.value if usage_context else None,
            )


# SOURCE: vllm/engine/arg_utils.py:L2804 AsyncEngineArgs
@dataclass
class AsyncEngineArgs(EngineArgs):
    """Arguments for asynchronous vLLM engine."""
    # SOURCE: vllm/engine/arg_utils.py:L2804 AsyncEngineArgs

    enable_log_requests: bool = False

    # SUBTRACTED: async-only CLI extras + add_cli_args override
    #   (vllm/engine/arg_utils.py:L2810-L2838) — parser plumbing; the one
    #   distinguishing field shows the subclass relationship (vllm serve
    #   entry shares the same assembly line).


# ============================================================================
# Factory #3: EngineCoreClient.make_client — vllm/v1/engine/core_client.py.
# ============================================================================


# SOURCE: vllm/v1/engine/core_client.py:L77 EngineCoreClient
class EngineCoreClient:
    """EngineCoreClient: subclasses handle different methods for pushing
    and pulling from the EngineCore for asyncio / multiprocessing.

    Subclasses:
    * InprocClient: In process EngineCore (for V0-style LLMEngine use)
    * SyncMPClient: ZMQ + background proc EngineCore (for LLM)
    * AsyncMPClient: ZMQ + background proc EngineCore w/ asyncio (for AsyncLLM)
    """

    # SOURCE: vllm/v1/engine/core_client.py:L89-L112 EngineCoreClient.make_client
    @staticmethod
    def make_client(  # SOURCE: vllm/v1/engine/core_client.py:L89 EngineCoreClient.make_client
        multiprocess_mode: bool,
        asyncio_mode: bool,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
    ) -> "EngineCoreClient":
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

    # SOURCE: vllm/v1/engine/core_client.py:L114-L139 EngineCoreClient.make_async_mp_client
    # SUBTRACTED: @instrument(span_name="Overall Loading") decorator
    #   (vllm/v1/engine/core_client.py:L115) — observability seam.
    @staticmethod
    def make_async_mp_client(  # SOURCE: vllm/v1/engine/core_client.py:L116 EngineCoreClient.make_async_mp_client
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> "AsyncMPClient":
        parallel_config = vllm_config.parallel_config
        client_args = (
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )
        # SUBTRACTED: DP external/internal load-balancer split
        #   (vllm/v1/engine/core_client.py:L133-L138) — dossier delete item
        #   (DP details open in ch34); DP=1 lands on AsyncMPClient directly.
        return AsyncMPClient(*client_args)


# SOURCE: vllm/v1/engine/core_client.py:L306 InprocClient
class InprocClient(EngineCoreClient):
    """
    InprocClient: client for in-process EngineCore. Intended
    for use in LLMEngine for V0-style add_request() and step()
        EngineCore setup in this process (no busy loop).

        * pushes EngineCoreRequest directly into the EngineCore
        * pulls EngineCoreOutputs by stepping the EngineCore
    """

    # SOURCE: vllm/v1/engine/core_client.py:L316-L317 InprocClient.__init__
    def __init__(self, *args, **kwargs):
        # SOURCE: vllm/v1/engine/core_client.py:L316 InprocClient.__init__
        self.engine_core = EngineCore(*args, **kwargs)

    # SUBTRACTED: get_output/add_request/abort/shutdown surface
    #   (vllm/v1/engine/core_client.py:L319-L402) — request-path plumbing
    #   (ch4-ch7); this chapter stops once the EngineCore is assembled.


# SOURCE: vllm/v1/engine/core_client.py:L503 MPClient (marker stub)
class MPClient(EngineCoreClient):
    """MPClient: base client for multi-proc EngineCore.
    EngineCore runs in a background process busy loop, getting
    new EngineCoreRequests and returning EngineCoreOutputs

    * pushes EngineCoreRequests via input_socket
    * pulls EngineCoreOutputs via output_socket

    * AsyncMPClient subclass for AsyncLLM usage
    * SyncMPClient subclass for LLM usage
    """

    # SUBTRACTED: shared mp machinery — ZMQ context + input/output sockets,
    #   EngineCore process spawn + startup handshake, background output
    #   handling (vllm/v1/engine/core_client.py:L503-L801) — the mp topology
    #   is ch5; the hierarchy slot is kept so Sync/AsyncMPClient sit on their
    #   real base class.
    pass


# SOURCE: vllm/v1/engine/core_client.py:L802 SyncMPClient (marker stub)
class SyncMPClient(MPClient):
    """Synchronous client for multi-proc EngineCore."""

    # SUBTRACTED: @instrument decorator (vllm/v1/engine/core_client.py:L805)
    #   + super().__init__ ZMQ socket setup / EngineCore process spawn /
    #   background output thread (L503-L801 MPClient machinery) — the mp
    #   topology is ch5; on the host seam the client only records its inputs.
    def __init__(self, vllm_config, executor_class, log_stats):
        # SOURCE: vllm/v1/engine/core_client.py:L806 SyncMPClient.__init__
        self.vllm_config = vllm_config
        self.executor_class = executor_class
        self.log_stats = log_stats


# SOURCE: vllm/v1/engine/core_client.py:L974 AsyncMPClient (marker stub)
class AsyncMPClient(MPClient):
    """Asyncio-compatible client for multi-proc EngineCore."""

    # SUBTRACTED: @instrument decorator (vllm/v1/engine/core_client.py:L977)
    #   + super().__init__ ZMQ asyncio sockets / engine process spawn /
    #   output_handler task (L503-L801 MPClient machinery) — same treatment
    #   as sync; ch5 opens the topology.
    def __init__(self, vllm_config, executor_class, log_stats,
                 client_addresses=None, client_count=1, client_index=0):
        # SOURCE: vllm/v1/engine/core_client.py:L978 AsyncMPClient.__init__
        self.vllm_config = vllm_config
        self.executor_class = executor_class
        self.log_stats = log_stats


# ============================================================================
# Convergence point: EngineCore.__init__ — vllm/v1/engine/core.py:L103-L234.
# ============================================================================


# SOURCE: vllm/v1/structured_output/__init__.py StructuredOutputManager (stub)
class StructuredOutputManager:
    # SUBTRACTED: grammar backend + request cache init
    #   (vllm/v1/structured_output/__init__.py) — structured output is
    #   ch31/ch32; the manager is kept as an assembly milestone marker.
    def __init__(self, vllm_config: VllmConfig):
        # SOURCE: vllm/v1/structured_output/__init__.py StructuredOutputManager.__init__
        self.vllm_config = vllm_config


# SOURCE: vllm/v1/engine/core.py:L103 EngineCore
class EngineCore:
    """Inner loop of vLLM's Engine."""

    # SOURCE: vllm/v1/engine/core.py:L106-L248 EngineCore.__init__
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        executor_fail_callback: Callable | None = None,
        include_finished_set: bool = False,
    ):
        # plugins need to be loaded at the engine/scheduler level too
        load_general_plugins()

        self.vllm_config = vllm_config
        # SUBTRACTED: data_parallel_rank_local log gating (core.py:L120-L125)
        #   — DP>1 logging edge.
        logger.info(
            "Initializing a V1 LLM engine (v%s) with config: %s",
            "0.27.1",
            vllm_config,
        )

        self.log_stats = log_stats
        # Opaque weight version supplied by the caller.
        self._weight_version = "default"

        # Setup Model.
        self.model_executor = executor_class(vllm_config)
        self._pooler_config_logged = False
        # SUBTRACTED: executor failure-callback registration (core.py:L134-
        #   L135) — needs register_failure_callback on the executor stub.

        self.available_gpu_memory_for_kv_cache = -1

        # SUBTRACTED: elastic-EP scale-up early init (core.py:L139-L140) —
        #   EP elasticity edge (ch34).

        # Setup KV Caches and update CacheConfig after profiling.
        kv_cache_config = self._initialize_kv_caches(vllm_config)
        self.structured_output_manager = StructuredOutputManager(vllm_config)

        # Setup scheduler.
        Scheduler = vllm_config.scheduler_config.get_scheduler_cls()

        if len(kv_cache_config.kv_cache_groups) == 0:  # noqa: SIM102
            # Encoder models without KV cache don't support
            # chunked prefill. But do SSM models?
            if vllm_config.scheduler_config.enable_chunked_prefill:
                logger.warning("Disabling chunked prefill for model without KVCache")
                vllm_config.scheduler_config.enable_chunked_prefill = False

        scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )

        self.scheduler: SchedulerInterface = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self.structured_output_manager,
            include_finished_set=include_finished_set,
            log_stats=self.log_stats,
            block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
        )
        # SUBTRACTED: EngineCore.__init__ tail (vllm/v1/engine/core.py:L169-
        #   L248) — spec-decode flags, KV-connector handshake, batch queue
        #   (max_concurrent_batches), prefix hasher, step_fn static binding,
        #   idle-state callbacks + freeze_gc_heap: the busy loop and async
        #   overlap are ch9/ch12; this chapter ends at "Scheduler(...) assembled".

    # SOURCE: vllm/v1/engine/core.py EngineCore._initialize_kv_caches (seam)
    def _initialize_kv_caches(self, vllm_config: VllmConfig):
        # SUBTRACTED: profile_run + get_kv_cache_configs + CacheConfig update
        #   (vllm/v1/engine/core.py _initialize_kv_caches body) — the memory
        #   ledger's three-step accounting is ch14 and needs a GPU. Reduced to
        #   a stand-in config whose kv_cache_groups are non-empty so the
        #   chunked-prefill guard above stays on the text-model path.
        class _KVCacheConfig:
            # SOURCE: vllm/v1/kv_cache_interface.py KVCacheConfig (marker)
            def __init__(self):
                self.kv_cache_groups = [object()]  # non-empty marker
                self.block_size = 16

        return _KVCacheConfig()


# SOURCE: vllm/v1/core/kv_cache_utils.py:L626 resolve_kv_cache_block_sizes (seam)
def resolve_kv_cache_block_sizes(kv_cache_config, vllm_config):
    # SUBTRACTED: per-backend scheduler/hash block-size arithmetic
    #   (vllm/v1/core/kv_cache_utils.py:L626 resolve_kv_cache_block_sizes body) —
    #   hybrid-attention detail (ch14); reduced to the stand-in block size for
    #   both scheduler and hash granularity on the traced path.
    return kv_cache_config.block_size, kv_cache_config.block_size


# ============================================================================
# Entry facades: LLMEngine / AsyncLLM — vllm/v1/engine/{llm_engine,async_llm}.py.
# ============================================================================


# SOURCE: vllm/v1/engine/llm_engine.py:L48 LLMEngine
class LLMEngine:
    """Legacy LLMEngine for backwards compatibility."""

    # SOURCE: vllm/v1/engine/llm_engine.py:L51-L141 LLMEngine.__init__ (reduced)
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        aggregate_engine_logging: bool = False,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list = None,
        mm_registry: Any = None,
        multiprocess_mode: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.log_stats = log_stats
        # SUBTRACTED: tracing init + dp group setup + renderer/InputProcessor/
        #   OutputProcessor construction + stat loggers + v0 model_executor
        #   exposure + finalizer (vllm/v1/engine/llm_engine.py:L62-L141) — the
        #   request-path trio is ch4-ch7; this chapter keeps the client choice.

        # EngineCore (gets EngineCoreRequests and gives EngineCoreOutputs)
        self.engine_core = EngineCoreClient.make_client(
            multiprocess_mode=multiprocess_mode,
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
        )

    # SOURCE: vllm/v1/engine/llm_engine.py:L143-L158 LLMEngine.from_vllm_config
    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list = None,
        disable_log_stats: bool = False,
    ) -> "LLMEngine":
        # SOURCE: vllm/v1/engine/llm_engine.py:L143 LLMEngine.from_vllm_config
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            log_stats=(not disable_log_stats),
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            multiprocess_mode=envs.VLLM_ENABLE_V1_MULTIPROCESSING,
        )

    # SOURCE: vllm/v1/engine/llm_engine.py:L160-L186 LLMEngine.from_engine_args
    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list = None,
        enable_multiprocessing: bool = False,
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # SOURCE: vllm/v1/engine/llm_engine.py:L160 LLMEngine.from_engine_args

        # Create the engine configs.
        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = Executor.get_class(vllm_config)

        if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
            logger.debug("Enabling multiprocessing for LLMEngine.")
            enable_multiprocessing = True

        # Create the LLMEngine.
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            multiprocess_mode=enable_multiprocessing,
        )

    # SUBTRACTED: step()/generate()/request surface
    #   (vllm/v1/engine/llm_engine.py:L188+) — the request loop is ch2/ch9.


# SOURCE: vllm/v1/engine/async_llm.py AsyncLLM (facade subset)
class AsyncLLM:
    """An asynchronous wrapper for the vLLM engine."""

    # SOURCE: vllm/v1/engine/async_llm.py AsyncLLM.__init__ (reduced)
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        log_requests: bool = True,
        log_stats: bool = True,
        start_engine_loop: bool = True,
        stat_loggers: list = None,
        aggregate_engine_logging: bool = False,
        client_addresses: dict = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> None:
        self.vllm_config = vllm_config
        # SUBTRACTED: renderer/InputProcessor/OutputProcessor + output_handler
        #   task + profiler (vllm/v1/engine/async_llm.py:L100-L203) — the
        #   async request path is ch4-ch7.

        # EngineCore (starts the engine in background process).
        self.engine_core = EngineCoreClient.make_async_mp_client(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
        )

    # SOURCE: vllm/v1/engine/async_llm.py:L205-L232 AsyncLLM.from_vllm_config
    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list = None,
        enable_log_requests: bool = False,
        aggregate_engine_logging: bool = False,
        disable_log_stats: bool = False,
        client_addresses: dict = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> "AsyncLLM":
        # Create the LLMEngine.
        # SOURCE: vllm/v1/engine/async_llm.py:L205 AsyncLLM.from_vllm_config
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            start_engine_loop=start_engine_loop,
            stat_loggers=stat_loggers,
            log_requests=enable_log_requests,
            log_stats=not disable_log_stats,
            aggregate_engine_logging=aggregate_engine_logging,
            usage_context=usage_context,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
        )

    # SOURCE: vllm/v1/engine/async_llm.py:L234-L257 AsyncLLM.from_engine_args
    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list = None,
    ) -> "AsyncLLM":
        """Create an AsyncLLM from the EngineArgs."""
        # SOURCE: vllm/v1/engine/async_llm.py:L234 AsyncLLM.from_engine_args

        # Create the engine configs.
        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = Executor.get_class(vllm_config)

        # Create the AsyncLLM.
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_requests=engine_args.enable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )


# ============================================================================
# Offline user entry: LLM — vllm/entrypoints/llm.py.
# ============================================================================


# SOURCE: vllm/entrypoints/llm.py:L67 LLM (entry subset)
class LLM:
    # SUBTRACTED: mixin bases BeamSearchOfflineMixin / PoolingOfflineMixin /
    #   OfflineInferenceMixin (vllm/entrypoints/llm.py:L67) — their
    #   generate()/chat()/tokenize surfaces are the subtracted request path.

    """An LLM for generating texts from given prompts and sampling parameters.

    This class includes a tokenizer, a language model (possibly distributed
    across multiple GPUs), and GPU memory space allocated for intermediate
    states (aka KV cache). Given a batch of prompts and sampling parameters,
    this class generates texts from the model, using an intelligent batching
    mechanism and efficient memory management.
    """
    # SUBTRACTED: per-kwarg Args block of the real docstring
    #   (vllm/entrypoints/llm.py:L76-L174) — documents the ~90 kwargs that the
    #   entry subset below does not carry.

    # SOURCE: vllm/entrypoints/llm.py:L295-L341 EngineArgs collection in LLM.__init__ (def L177)
    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.92,
        enforce_eager: bool = False,
        max_model_len: Optional[int] = None,
        max_num_batched_tokens: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
        enable_chunked_prefill: Optional[bool] = None,
        async_scheduling: Optional[bool] = None,
        optimization_level: OptimizationLevel = OptimizationLevel.O2,
        performance_mode: str = "balanced",
        distributed_executor_backend: Any = None,
        compilation_config: Any = None,
        speculative_config: Optional[dict] = None,
        **kwargs,
    ):
        # SUBTRACTED: the data-parallel offline guard (llm.py:L288-L293) —
        #   DP offline needs the explicit multi-process example; the guard's
        #   raise is superseded by DP=1 on this path.
        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            enable_chunked_prefill=enable_chunked_prefill,
            async_scheduling=async_scheduling,
            optimization_level=optimization_level,
            performance_mode=performance_mode,
            distributed_executor_backend=distributed_executor_backend,
            speculative_config=speculative_config,
            **(
                {"compilation_config": compilation_config}
                if compilation_config is not None
                else {}
            ),
            **kwargs,
        )
        # SUBTRACTED: the further ~90 LLM kwargs + pooler/attention/
        #   structured/profiler instance pre-construction
        #   (vllm/entrypoints/llm.py:L160-L294) — same flat->EngineArgs
        #   repacking; log_non_default_args (L337) is a logging nicety.
        # Note: 9 of the explicit kwargs above (max_model_len /
        #   max_num_batched_tokens / max_num_seqs / enable_chunked_prefill /
        #   async_scheduling / optimization_level / performance_mode /
        #   distributed_executor_backend / speculative_config) are **kwargs-
        #   pass-throughs in the real entry (llm.py:L177-L294); they are
        #   promoted to explicit params here to show this chapter's knobs.
        #   compilation_config None routing: real _make_config(None) ->
        #   CompilationConfig() fresh default; omitted kwarg -> EngineArgs
        #   default_factory gives the same fresh default.

        self.llm_engine = LLMEngine.from_engine_args(
            engine_args=engine_args, usage_context=UsageContext.LLM_CLASS
        )
        self.model_config = self.llm_engine.model_config
        self.engine_class = type(self.llm_engine)

        self.request_counter = None  # SUBTRACTED: Counter() — request stats

    # SUBTRACTED: generate()/chat()/tokenize surface (vllm/entrypoints/
    #   llm.py + mixins) — the request path is ch2.
