# Subtract-only companion for v3 ch19 — vllm/config/compilation.py
# (pin v0.27.1 / 6e448d0ea). Same names, same structure, same control flow;
# only dossier-approved deletions (each marked `# SUBTRACTED:`), plus
# 章范围外域段以 SUBTRACTED+归属注记收窄（impl-notes §范围裁剪）。
#
# HOST 注记（pydantic→dataclass）：真源用 pydantic @config（vllm/config/utils.py
# 的 config 装饰器，ch03 域）＋field_validator；伴读版以纯 dataclass 承载同一
# 字段集与 __post_init__ 校验语义（str→enum 折进 __post_init__ 头部），行为
# 面等价——见 impl-notes §Host 决策。
from __future__ import annotations

import enum
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from .._host_seams import current_platform, envs, init_logger, round_up
from ..utils.import_utils import resolve_obj_by_qualname
from ..utils.torch_utils import is_torch_equal_or_newer
from .utils import Range

if TYPE_CHECKING:
    from .vllm import VllmConfig
else:
    VllmConfig = object

logger = init_logger(__name__)


# SOURCE: vllm/config/compilation.py:L37-L50 CompilationMode
class CompilationMode(enum.IntEnum):
    """The compilation approach used for torch.compile-based compilation of the
    model."""

    NONE = 0
    """No torch.compile compilation is applied, model runs in fully eager pytorch mode.
    The model runs as-is."""
    STOCK_TORCH_COMPILE = 1
    """The standard `torch.compile` compilation pipeline."""
    DYNAMO_TRACE_ONCE = 2
    """Single Dynamo trace through the model, avoiding recompilation."""
    VLLM_COMPILE = 3
    """Custom vLLM Inductor-based backend with caching, piecewise compilation,
    shape specialization, and custom passes."""


# SOURCE: vllm/config/compilation.py:L53-L103 CUDAGraphMode —— 三运行期模式
#   + 两组合档（tuple 值）；decode_mode/mixed_mode 拆解组合档
class CUDAGraphMode(enum.Enum):
    """Constants for the cudagraph mode in CompilationConfig.
    Meanwhile, the subset enum `NONE`, `PIECEWISE` and `FULL` are also
    treated as concrete runtime mode for cudagraph runtime dispatching.
    """

    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = (FULL, NONE)
    FULL_AND_PIECEWISE = (FULL, PIECEWISE)

    def decode_mode(self) -> "CUDAGraphMode":  # SOURCE: vllm/config/compilation.py:L65-L66
        return CUDAGraphMode(self.value[0]) if self.separate_routine() else self

    def mixed_mode(self) -> "CUDAGraphMode":  # SOURCE: vllm/config/compilation.py:L68-L69
        return CUDAGraphMode(self.value[1]) if self.separate_routine() else self

    def has_mode(self, mode: "CUDAGraphMode") -> bool:  # SOURCE: vllm/config/compilation.py:L71-L75
        assert not mode.separate_routine()
        if self.separate_routine():
            return mode.value in self.value
        return self == mode

    def requires_piecewise_compilation(self) -> bool:  # SOURCE: vllm/config/compilation.py:L77-L78
        return self.has_mode(CUDAGraphMode.PIECEWISE)

    def max_cudagraph_mode(self) -> "CUDAGraphMode":  # SOURCE: vllm/config/compilation.py:L80-L81
        return CUDAGraphMode(max(self.value)) if self.separate_routine() else self

    def has_full_cudagraphs(self) -> bool:  # SOURCE: vllm/config/compilation.py:L83-L84
        return self.max_cudagraph_mode() == CUDAGraphMode.FULL

    def has_piecewise_cudagraphs(self) -> bool:  # SOURCE: vllm/config/compilation.py:L86-L87
        return self.requires_piecewise_compilation()

    def separate_routine(self) -> bool:  # SOURCE: vllm/config/compilation.py:L89-L90
        return isinstance(self.value, tuple)

    @classmethod
    def valid_runtime_modes(cls) -> frozenset["CUDAGraphMode"]:  # SOURCE: vllm/config/compilation.py:L92-L94
        return frozenset({cls.NONE, cls.PIECEWISE, cls.FULL})

    def is_valid_runtime_mode(self) -> bool:  # SOURCE: vllm/config/compilation.py:L96-L97
        return self in CUDAGraphMode.valid_runtime_modes()

    def __str__(self) -> str:  # SOURCE: vllm/config/compilation.py:L99-L100
        return self.name

    def __bool__(self) -> bool:  # SOURCE: vllm/config/compilation.py:L102-L103
        return self != CUDAGraphMode.NONE


# SUBTRACTED: vllm/config/compilation.py PassConfig 的 flashinfer_max_size/
#   default_fi_allreduce_fusion_max_size_mb（L187-L216——flashinfer 融合
#   pass 消费侧，ch12/ch27 融合域）与 compute_hash/log_enabled_passes
#   （L218-L331——缓存 hash/观测，消费侧已随 delete[3]/观测删除）。


# SOURCE: vllm/config/compilation.py:L106-L184 PassConfig —— 融合 pass 开关
#   集（O2 预设按硬件谓词点亮的账本）
@dataclass
class PassConfig:
    """Configuration for custom Inductor passes.

    This is separate from general `CompilationConfig` so that inductor passes
    don't all have access to full configuration - that would create a cycle as
    the `PassManager` is set as a property of config.

    You must pass PassConfig to VLLMConfig constructor via the CompilationConfig
    constructor. VLLMConfig's post_init does further initialization.
    If used outside of the VLLMConfig, some fields may be left in an
    improper state.
    """

    # New flags
    fuse_norm_quant: bool = None  # type: ignore[assignment]
    """Fuse the custom RMSNorm + quant ops."""
    fuse_act_quant: bool = None  # type: ignore[assignment]
    """Fuse the custom SiluMul + quant ops."""
    fuse_attn_quant: bool = None  # type: ignore[assignment]
    """Fuse the custom Attention and MLAAttention + quant ops."""
    eliminate_noops: bool = True
    """Eliminate no-op ops."""
    enable_sp: bool = None  # type: ignore[assignment]
    """Enable sequence parallelism. Requires TP>1. Automatically disabled
    if the model's hidden_size is too small for SP to be beneficial
    (threshold is device-capability dependent)."""
    fuse_gemm_comms: bool = None  # type: ignore[assignment]
    """Enable async TP."""
    fuse_allreduce_rms: bool = None  # type: ignore[assignment]
    """Enable flashinfer allreduce fusion."""
    enable_qk_norm_rope_fusion: bool = None  # type: ignore[assignment]
    """Enable fused Q/K RMSNorm + RoPE pass."""
    fuse_rope_kvcache_cat_mla: bool = None  # type: ignore[assignment]
    """Enable fused MLA KV cache update with RoPE."""

    # ROCm/AITER specific fusions
    fuse_act_padding: bool = None  # type: ignore[assignment]
    """Fuse the custom RMSNorm + padding ops."""
    fuse_mla_dual_rms_norm: bool = None  # type: ignore[assignment]
    """Fuse paired q/kv RMS norms in MLA attention."""
    fuse_rope_kvcache: bool = None  # type: ignore[assignment]
    """Fuse the QK rope + KV cache ops."""
    fuse_qk_norm_rope_kvcache: bool = None  # type: ignore[assignment]
    """Fuse QK RMSNorm + RoPE + KV cache update into a single AITER HIP
    kernel. Supersedes both enable_qk_norm_rope_fusion and fuse_rope_kvcache
    for layers that support it. Auto-enabled at O1+ on ROCm for models
    with QK-norm (e.g. Qwen3-MoE)."""

    rope_kvcache_fusion_max_token_num: int = 256
    """The threshold for ROCm AITER RoPE+KVCache fusion e.g. for small batch decode.
    Larger batch sizes e.g. during prefill will use the unfused kernels.
    Also applies to the fused QK-Norm+RoPE+KVCache pass.
    """

    # SUBTRACTED: fi_allreduce_fusion_max_size_mb/sp_min_token_num 阈值字段
    #   （L161-L183——flashinfer/SP 融合阈值，融合 pass 消费侧）。

    # SUBTRACTED: pydantic _skip_none_validation field_validator
    #   （L227-L247——None-延迟初始化语义由纯 dataclass 默认值天然承载）。

    def __post_init__(self) -> None:  # SOURCE: vllm/config/compilation.py:L249-L310
        # Handle deprecation and defaults

        if not self.eliminate_noops:
            if self.fuse_norm_quant or self.fuse_act_quant:
                logger.warning_once(
                    "Fusion enabled but reshape elimination disabled. "
                    "RMSNorm/SiluMul + quant (fp8) fusion might not work"
                )
            if self.fuse_attn_quant:
                logger.warning_once(
                    "Fusion enabled but reshape elimination disabled. "
                    "Attention + quant (fp8) fusion might not work"
                )
            if self.fuse_allreduce_rms:
                logger.warning_once(
                    "Fusion enabled but reshape elimination disabled. "
                    "Allreduce + rms norm + quant (fp8) fusion might not work"
                )
            if self.fuse_act_padding:
                logger.warning_once(
                    "Fusion enabled but reshape elimination disabled. "
                    "RMSNorm + padding fusion might not work"
                )
        if self.enable_qk_norm_rope_fusion and not (
            current_platform.is_cuda_alike() or current_platform.is_xpu()
        ):
            logger.warning_once(
                "QK Norm + RoPE fusion enabled but the current platform is not "
                "CUDA, ROCm or XPU. The fusion will be disabled."
            )
            self.enable_qk_norm_rope_fusion = False
        if self.fuse_act_padding and not current_platform.is_rocm():
            logger.warning_once(
                "Padding fusion enabled but the current platform is not ROCm. "
                "The fusion will be disabled."
            )
            self.fuse_act_padding = False
        if self.fuse_mla_dual_rms_norm and not current_platform.is_rocm():
            logger.warning_once(
                "MLA dual RMS norm fusion requires ROCm/AITER. "
                "The fusion will be disabled."
            )
            self.fuse_mla_dual_rms_norm = False
        if self.fuse_rope_kvcache and not current_platform.is_rocm():
            logger.warning_once(
                "KV cache fusion currently only enabled on ROCm. "
                "The fusion will be disabled."
            )
            self.fuse_rope_kvcache = False
        if self.fuse_qk_norm_rope_kvcache and not current_platform.is_rocm():
            logger.warning_once(
                "QK-Norm+RoPE+KVCache fusion requires ROCm with AITER. "
                "The fusion will be disabled."
            )
            self.fuse_qk_norm_rope_kvcache = False
        if self.fuse_rope_kvcache_cat_mla and not current_platform.is_cuda_alike():
            logger.warning_once(
                "MLA KV cache update with RoPE fusion enabled but the "
                "current platform is not CUDA or ROCm. The fusion will be disabled."
            )
            self.fuse_rope_kvcache_cat_mla = False


# SOURCE: vllm/config/compilation.py:L334-L351 DynamicShapesType
class DynamicShapesType(str, enum.Enum):
    """Types of dynamic shapes handling in torch.compile().
    see  Dynamic shapes and vllm guard dropping in torch_compile.md
    for more details."""

    BACKED = "backed"
    """Use backed dynamic shapes. torch.compile() guards on backed dynamic
    shapes and may add guards. Symbols are specialized to 0, 1, or >=2 even
    without encountering branching on those ranges."""

    UNBACKED = "unbacked"
    """Use unbacked dynamic shapes. Guaranteed not to be guarded on and not
    0/1 specialized, but may throw data dependent errors when branches require
    their value without explicit unbacked handling."""

    BACKED_SIZE_OBLIVIOUS = "backed_size_oblivious"
    """Experimental flag that treats backed symbols as unbacked when explicit
    unbacked handling is defined."""


# SUBTRACTED: DynamicShapesConfig.compute_hash（L386-L394——缓存 hash 域）。


# SOURCE: vllm/config/compilation.py:L354-L394 DynamicShapesConfig
@dataclass
class DynamicShapesConfig:  # SOURCE: vllm/config/compilation.py:L354-L394
    """Configuration to control/debug torch compile dynamic shapes."""

    type: DynamicShapesType = DynamicShapesType.BACKED
    """Controls the type of dynamic shapes handling to use with torch.compile().

    - BACKED: Default PyTorch behavior with potential guards ignored.
    - UNBACKED: No guards guaranteed (most sound) but may throw
      data dependent errors.
    - BACKED_SIZE_OBLIVIOUS: Experimental safer alternative to
      backed/unbacked.
    """

    evaluate_guards: bool = False
    """
    A debug mode to detect and fail if Dynamo ever specializes a dynamic shape by
    guarding on it. When True, dynamic shape guards are not dropped from dynamo.
    And a failure will be triggered if a recompilation ever happens due to that.
    This mode requires VLLM_USE_BYTECODE_HOOK to be 0.
    Enabling this allow observing the dynamic shapes guards in the tlparse
    artifacts also.
    When type is backed, aot_compile must be disabled for this mode to work.
    until this change picked up https://github.com/pytorch/pytorch/pull/169239.
    """

    # SUBTRACTED: assume_32_bit_indexing（L380-L384——torch>=2.10 索引位宽
    #   实验开关，guard 消费侧）。

    assume_32_bit_indexing: bool = False


# SOURCE: vllm/config/compilation.py:L397-L756 CompilationConfig —— 本章主角
#   配置持有者：mode/splitting_ops/cudagraph_mode/custom_ops/cudagraph_
#   capture_sizes/static_forward_context 全在此（docstring 全文见真源）。
#   SUBTRACTED 字段（本章零消费）：debug_dump_path/cache_dir/compile_cache_
#   save_format（缓存域，随 delete[3]）、compile_mm_encoder/cudagraph_mm_
#   encoder/encoder_cudagraph_*（多模态扩展态）、cudagraph_copy_inputs
#   （codegen 尾段随 delete[3]）、cudagraph_specialize_lora（LoRA 专化，随
#   delete[2]）、inductor_passes/fast_moe_cold_start（pass/MoE 域）。
@dataclass
class CompilationConfig:
    """Configuration for compilation.

    You must pass CompilationConfig to VLLmConfig constructor.
    VllmConfig's post_init does further initialization. If used outside of the
    VllmConfig, some fields will be left in an improper state.

    It contains PassConfig, which controls the custom fusion/transformation
    passes. The rest has three parts: top-level compilation control, CUDA
    graph capture, and inductor compilation.
    """

    # Top-level Compilation control
    mode: CompilationMode = None  # type: ignore[assignment]
    """The compilation approach used for torch.compile-based compilation of
    the model. None → default (VLLM_COMPILE for V1 when -O1+)."""
    backend: str = ""
    """The backend for compilation: "" (default, "inductor" on CUDA-alike),
    "eager", or a qualified name."""
    custom_ops: list[str] = field(default_factory=list)
    """Fine-grained control over which custom ops to enable/disable. Use 'all'
    to enable all, 'none' to disable all, '+op'/'-op' to pin single ops."""
    ir_enable_torch_wrap: bool = None  # type: ignore[assignment]
    """If True, enable vllm_ir torch custom op wrapping during the forward
    pass. Defaults to True when using Inductor with vllm-compile."""

    splitting_ops: list[str] | None = None
    """A list of ops to exclude from cudagraphs, used in piecewise
    compilation. If None, defaults to attention ops for piecewise
    cudagraphs. If empty list [], no ops are excluded."""

    # Inductor capture
    compile_sizes: list[int | str] | None = None
    """Sizes to compile for inductor. In addition to integers, it also
    supports "cudagraph_capture_sizes" to specify the sizes for cudagraph
    capture."""
    compile_ranges_endpoints: list[int] | None = None
    """Endpoints for Inductor compile ranges. The compile ranges are
    [1, endpoints[0]], [endpoints[0] + 1, endpoints[1]], ...,
    [endpoints[-1] + 1, max_num_batched_tokens]."""
    inductor_compile_config: dict = field(default_factory=dict)
    """Additional configurations for inductor."""

    # CudaGraph compilation
    cudagraph_mode: CUDAGraphMode = None  # type: ignore[assignment]
    """The mode of the cudagraph: NONE / PIECEWISE / FULL /
    FULL_DECODE_ONLY / FULL_AND_PIECEWISE (v1 default)."""
    cudagraph_num_of_warmups: int = 0
    """Number of warmup runs for cudagraph. It means the first several runs
    will be treated as warmup runs. Only after that, the execution will be
    recorded, and the recorded cudagraph will be used for subsequent runs."""
    cudagraph_capture_sizes: list[int] = None  # type: ignore[assignment]
    """Sizes to capture cudagraph. None (default): capture sizes are inferred
    from vllm config."""

    use_inductor_graph_partition: bool = None  # type: ignore[assignment]
    """Use inductor graph partition to split the graph at cudagraph_unsafe
    ops (codegen-time partitioning; RoPE+KV fusion passes only run there)."""

    pass_config: PassConfig = field(default_factory=PassConfig)
    """Custom inductor passes, see PassConfig for more details"""

    max_cudagraph_capture_size: int = None  # type: ignore[assignment]
    """The maximum cudagraph capture size (largest entry of
    cudagraph_capture_sizes once post_init_cudagraph_sizes ran)."""

    dynamic_shapes_config: DynamicShapesConfig = field(
        default_factory=DynamicShapesConfig
    )
    """Configuration for dynamic shapes options"""

    # keep track of enabled and disabled custom ops
    enabled_custom_ops: Counter[str] = field(default_factory=Counter, init=False)
    """custom ops that are enabled"""
    disabled_custom_ops: Counter[str] = field(default_factory=Counter, init=False)
    """custom ops that are disabled"""
    traced_files: set[str] = field(default_factory=set, init=False)
    """files that are traced for compilation"""
    compilation_time: float = field(default=0.0, init=False)
    """time taken for compilation"""
    encoder_compilation_time: float = field(default=0.0, init=False)
    """time taken for multimodal encoder compilation"""

    static_forward_context: dict[str, Any] = field(default_factory=dict, init=False)
    """Per-model forward context
    Map from layer name to layer objects that need to be accessed outside
    model code, e.g., Attention, FusedMOE when dp_size>1."""

    static_all_moe_layers: list[str] = field(default_factory=list, init=False)
    """The names of all the MOE layers in the model
    """

    # Attention ops; used for piecewise cudagraphs
    # Use PyTorch operator format: "namespace::name"
    # SOURCE: vllm/config/compilation.py:L762-L778 _attention_ops —— 切图点
    #   清单（13 算子）
    _attention_ops: ClassVar[list[str]] = [
        "vllm::unified_attention_with_output",
        "vllm::unified_mla_attention_with_output",
        "vllm::mamba_mixer2",
        "vllm::mamba_mixer",
        "vllm::short_conv",
        "vllm::linear_attention",
        "vllm::qwen_gdn_attention_core",
        "vllm::gdn_attention_core_xpu",
        "vllm::olmo_hybrid_gdn_full_forward",
        "vllm::sparse_attn_indexer",
        "vllm::rocm_aiter_sparse_attn_indexer",
        "vllm::deepseek_v4_attention",
        "vllm::hpc_rope_norm_forward",
    ]

    # SUBTRACTED: compute_hash/__repr__/__str__（L780-L840——缓存 hash 与
    #   序列化 repr，消费侧随 delete[3] 缓存块删除）。

    def __post_init__(self) -> None:
        # HOST SEAM（pydantic→dataclass）:三条 field_validator 的 str→enum /
        #   dict→PassConfig 前置转换折进 __post_init__ 头（原语义逐条对应
        #   validate_mode_before L842-L861 / validate_cudagraph_mode_before
        #   L863-L869 / validate_pass_config_before L871-L877）。
        if isinstance(self.mode, str):
            mode_name = self.mode.upper()
            if mode_name not in CompilationMode.__members__:
                raise ValueError(
                    f"Invalid compilation mode: {self.mode}. "
                    f"Valid modes are: {', '.join(CompilationMode.__members__.keys())}"
                )
            self.mode = CompilationMode[mode_name]
        if isinstance(self.cudagraph_mode, str):
            self.cudagraph_mode = CUDAGraphMode[self.cudagraph_mode.upper()]
        if isinstance(self.pass_config, dict):
            self.pass_config = PassConfig(**self.pass_config)

        # SOURCE: vllm/config/compilation.py:L914-L916 auto_functionalized_v2
        KEY = "enable_auto_functionalized_v2"
        if KEY not in self.inductor_compile_config:
            self.inductor_compile_config[KEY] = False

        # SOURCE: vllm/config/compilation.py:L918-L935 inductor 运行期断言
        #   开关与 DEBUG 日志联动（torch<2.12 workaround）
        if not is_torch_equal_or_newer("2.12.0.dev"):
            enable_asserts = envs.VLLM_LOGGING_LEVEL == "DEBUG"
            for key in (
                "size_asserts",
                "alignment_asserts",
                "scalar_asserts",
            ):
                self.inductor_compile_config.setdefault(key, enable_asserts)

        # SUBTRACTED: inductor_passes 的 qualified-name 解析循环（L937-L952
        #   ——pass 装配域，随 delete[3] configure_post_pass 一族）与
        #   +rotary_embedding 三连（L954-L974）/combo_kernels（L976-L986
        #   ——融合 pass 域）。

        # SOURCE: vllm/config/compilation.py:L988-L995 inductor partition 的
        #   torch 版本闸
        if self.use_inductor_graph_partition and not is_torch_equal_or_newer(
            "2.9.0.dev"
        ):
            raise ValueError(
                "use_inductor_graph_partition is only "
                "supported with torch>=2.9.0.dev. Set "
                "use_inductor_graph_partition=False instead."
            )

        # SOURCE: vllm/config/compilation.py:L997-L1018 custom_ops 语法校验
        for op in self.custom_ops:
            if op not in {"all", "none"} and (len(op) < 2 or op[0] not in {"+", "-"}):
                raise ValueError(
                    f"Invalid syntax '{op}' for custom op, "
                    "must be 'all', 'none', '+op' or '-op' "
                    "(where 'op' is the registered op name)"
                )

        base_modes = [op for op in self.custom_ops if op in {"all", "none"}]
        if len(base_modes) > 1:
            raise ValueError(
                "custom_ops can contain only one base mode: 'all' or 'none'"
            )

        enabled_ops = {op[1:] for op in self.custom_ops if op.startswith("+")}
        disabled_ops = {op[1:] for op in self.custom_ops if op.startswith("-")}
        conflicting_ops = sorted(enabled_ops & disabled_ops)
        if conflicting_ops:
            raise ValueError(
                "custom_ops cannot both enable and disable the same operation(s): "
                f"{', '.join(conflicting_ops)}. Remove either the '+' or '-' directive"
            )

        # Currently only eager and inductor backend are supported.
        # for piecewise compilation. Custom backends are not supported for
        # piecewise compilation. Update when more backends are supported.
        # SOURCE: vllm/config/compilation.py:L1020-L1030
        if self.mode == CompilationMode.VLLM_COMPILE and self.backend not in [
            "",
            "eager",
            "inductor",
        ]:
            raise ValueError(
                f"Invalid backend for piecewise compilation: {self.backend}"
            )

        # SUBTRACTED: encoder cudagraph 配置校验（L1032-L1057——多模态
        #   扩展态）。

        # SOURCE: vllm/config/compilation.py:L1059-L1060 平台默认 backend
        if self.backend == "":
            self.backend = current_platform.get_compile_backend()

    # SOURCE: vllm/config/compilation.py:L1062-L1104 init_backend —— STOCK/
    #   DYNAMO 走 torch backend 名或 qualname；VLLM_COMPILE 走 VllmBackend
    def init_backend(  # SOURCE: vllm/config/compilation.py:L1062-L1104
        self,
        vllm_config: "VllmConfig",
        prefix: str = "",
        is_encoder: bool = False,
    ) -> str | Callable:
        """
        Initialize the backend for the compilation config from a vllm config.
        Arguments:
            vllm_config: The vllm config to initialize the backend from.
            prefix: Cache directory prefix for this compiled module.
            is_encoder: Whether this module is used in an encoder (as
                opposed to a text backbone).
        Returns:
            The backend for the compilation config.
        """
        if self.mode is None:
            raise ValueError(
                "No compilation mode is set. This method should only be "
                "called via vllm config where the level is set if none is "
                "provided."
            )
        if self.mode == CompilationMode.NONE:
            raise ValueError("No compilation mode is set.")

        from torch._dynamo.backends.registry import list_backends

        torch_backends = list_backends(exclude_tags=tuple())
        if self.mode in [
            CompilationMode.STOCK_TORCH_COMPILE,
            CompilationMode.DYNAMO_TRACE_ONCE,
        ]:
            if self.backend in torch_backends:
                return self.backend
            return resolve_obj_by_qualname(self.backend)

        assert self.mode == CompilationMode.VLLM_COMPILE
        if self.backend not in ["eager", "inductor"]:
            logger.info("Using OOT custom backend for compilation.")

        from ..compilation.backends import VllmBackend

        return VllmBackend(vllm_config, prefix=prefix, is_encoder=is_encoder)

    # SOURCE: vllm/config/compilation.py:L1106-L1131 post_init_cudagraph_sizes
    def post_init_cudagraph_sizes(self) -> None:
        """To complete the initialization after cudagraph related
        configs are set. This includes:
        - initialize compile_sizes
        """

        computed_compile_sizes: list[int] = []
        if self.compile_sizes is not None:
            # de-duplicate the sizes provided by the config
            self.compile_sizes = list(set(self.compile_sizes))
            for x in self.compile_sizes:
                if isinstance(x, str):
                    assert x == "cudagraph_capture_sizes", (
                        "Unrecognized size type in compile_sizes, "
                        f"expect 'cudagraph_capture_sizes', got {x}"
                    )
                    computed_compile_sizes.extend(self.cudagraph_capture_sizes)
                else:
                    assert isinstance(x, int)
                    computed_compile_sizes.append(x)
        self.compile_sizes = computed_compile_sizes  # type: ignore

        # make sure the sizes are in ascending order
        self.cudagraph_capture_sizes.sort()
        if self.cudagraph_capture_sizes:
            assert self.cudagraph_capture_sizes[-1] == self.max_cudagraph_capture_size

    # SOURCE: vllm/config/compilation.py:L1133-L1248 set_splitting_ops_for_v1
    #   —— splitting_ops 组装主支（delete[9]：fuse_attn_quant 支
    #   L1143-L1144+L1250-L1264、empty splitting_ops 降级 L1186-L1211、
    #   SP/fuse_gemm_comms 全图要求 L1213-L1230、DeepEP 禁图 L1232-L1248
    #   四条扩展分支整删；主支 L1146-L1184 原文保留）
    def set_splitting_ops_for_v1(  # SOURCE: vllm/config/compilation.py:L1133-L1248
        self, all2all_backend: str, data_parallel_size: int = 1
    ):
        # To compatible with OOT hardware plugin platform (for example vllm-ascend)
        # which currently only supports sequence parallelism in eager mode.
        if self.mode != CompilationMode.VLLM_COMPILE:
            if self.splitting_ops is None:
                self.splitting_ops = []
            return

        # SUBTRACTED: fuse_attn_quant 互斥支（L1143-L1144——delete[9]①：走
        #   set_splitting_ops_for_attn_fusion 的 attn 融合切点路线，正文一句
        #   话带过；方法体 L1250-L1264 随调用点一并删）。
        if self.splitting_ops is None:
            # NOTE: When using full cudagraph, instead of setting an empty
            # list and capture the full cudagraph inside the flattened fx
            # graph, we keep the piecewise fx graph structure but capture
            # the full cudagraph outside the fx graph. This reduces some
            # cpu overhead when the runtime batch_size is not cudagraph
            # captured. see https://github.com/vllm-project/vllm/pull/20059
            # for details. Make a copy to avoid mutating the class-level
            # list via reference.
            self.splitting_ops = list(self._attention_ops)

            # unified_kv_cache_update has a string param that prevents Inductor
            # from reusing piecewise graphs. Remove it from the compiled graph.
            # This has the side-effect of excluding cache from cudagraphs but
            # that doesn't seem to affect performance.
            # https://github.com/vllm-project/vllm/issues/33267
            if not self.use_inductor_graph_partition:
                if self.pass_config.fuse_rope_kvcache:
                    logger.warning_once(
                        "fuse_rope_kvcache is enabled, but splitting_ops is None "
                        "and Inductor graph partition is not enabled."
                        "Disabling fuse_rope_kvcache."
                        "Please either set splitting_ops to an empty list []"
                        "or set use_inductor_graph_partition to True "
                        "to enable RoPE+KV cache fusion."
                    )
                    self.pass_config.fuse_rope_kvcache = False
                if self.pass_config.fuse_qk_norm_rope_kvcache:
                    logger.warning_once(
                        "fuse_qk_norm_rope_kvcache is enabled, but "
                        "splitting_ops is None and Inductor graph partition "
                        "is not enabled. Disabling fuse_qk_norm_rope_kvcache. "
                        "Please either set splitting_ops to an empty list [] "
                        "or set use_inductor_graph_partition to True "
                        "to enable QK-Norm+RoPE+KV cache fusion."
                    )
                    self.pass_config.fuse_qk_norm_rope_kvcache = False
                self.splitting_ops.append("vllm::unified_kv_cache_update")
                self.splitting_ops.append("vllm::unified_mla_kv_cache_update")

        # SUBTRACTED: elif len(self.splitting_ops) == 0 的两段降级
        #   （L1186-L1211——delete[9]③：空 splitting_ops 的档位自降级）。

        # SUBTRACTED: SP/fuse_gemm_comms 全图要求（L1213-L1230——delete[9]②）
        #   与 DeepEP high-throughput 禁图（L1232-L1248——delete[9]④）。

    # SUBTRACTED: set_splitting_ops_for_attn_fusion（L1250-L1264——delete[9]①
    #   的方法体，随调用点整删）。

    # SOURCE: vllm/config/compilation.py:L1270-L1273 splitting_ops_contain_attention
    def splitting_ops_contain_attention(self) -> bool:
        return self.splitting_ops is not None and all(
            op in self.splitting_ops for op in self._attention_ops
        )

    # SOURCE: vllm/config/compilation.py:L1275-L1296 splitting_ops_contain_kv_cache_update
    def splitting_ops_contain_kv_cache_update(self) -> bool:
        # when using Dynamo partition while splitting ops is None
        # and attn+quant fusion disabled, the kv_cache_update_ops are
        # appended to splitting_ops in set_splitting_ops_for_v1 due to
        # https://github.com/vllm-project/vllm/issues/33267
        # In this case, we return True if the kv_cache_update_ops
        # are not in the splitting_ops yet, but will subsequently
        # be added to splitting_ops.
        if (
            not self.use_inductor_graph_partition
            and self.splitting_ops is None
            and not self.pass_config.fuse_attn_quant
        ):
            return True

        kv_cache_update_ops = [
            "vllm::unified_kv_cache_update",
            "vllm::unified_mla_kv_cache_update",
        ]
        return self.splitting_ops is not None and all(
            op in self.splitting_ops for op in kv_cache_update_ops
        )

    # SOURCE: vllm/config/compilation.py:L1298-L1307 is_attention_compiled_piecewise
    def is_attention_compiled_piecewise(self) -> bool:
        if not self.splitting_ops_contain_attention():
            return False

        if not self.use_inductor_graph_partition:
            # Dynamo-level FX split case
            return self.mode == CompilationMode.VLLM_COMPILE

        # Inductor partition case
        return self.backend == "inductor" and self.mode != CompilationMode.NONE

    # SOURCE: vllm/config/compilation.py:L1309-L1353 custom_op_log_check ——
    #   set_current_vllm_config(check_compile=True) 的收尾校验
    def custom_op_log_check(self):  # SOURCE: vllm/config/compilation.py:L1309-L1353
        """
        This method logs the enabled/disabled custom ops and checks that the
        passed custom_ops field only contains relevant ops.
        It is called at the end of set_current_vllm_config,
        after the custom ops have been instantiated.
        """

        if len(self.enabled_custom_ops) + len(self.disabled_custom_ops) == 0:
            logger.debug("No custom ops found in model.")
            return

        logger.debug("enabled custom ops: %s", self.enabled_custom_ops)
        logger.debug("disabled custom ops: %s", self.disabled_custom_ops)

        all_ops_in_model = self.enabled_custom_ops | self.disabled_custom_ops
        for op in self.custom_ops:
            if op in {"all", "none"}:
                continue

            assert op[0] in {"+", "-"}, (
                "Invalid custom op syntax (should be checked during init)"
            )

            # check if op name exists in model
            op_name = op[1:]
            if op_name not in all_ops_in_model:
                from ..model_executor.custom_op import op_registry

                # Does op exist at all or is it just not present in this model?
                # Note: Only imported op classes appear in the registry.
                missing_str = (
                    "doesn't exist (or wasn't imported/registered)"
                    if op_name not in op_registry
                    else "not present in model"
                )

                enable_str = "enabling" if op[0] == "+" else "disabling"
                logger.warning_once(
                    "Op '%s' %s, %s with '%s' has no effect",
                    op_name,
                    missing_str,
                    enable_str,
                    op,
                )

    # SOURCE: vllm/config/compilation.py:L1355-L1366 is_custom_op_enabled ——
    #   O2 预设谓词（enable_norm_fusion 等）的查询面
    def is_custom_op_enabled(self, op: str) -> bool:  # SOURCE: vllm/config/compilation.py:L1355-L1366
        count_all = self.custom_ops.count("all")
        count_none = self.custom_ops.count("none")
        if count_all + count_none != 1:
            raise ValueError(
                "custom_ops must contain exactly one base mode: 'all' or 'none'"
            )

        if count_all:
            return f"-{op}" not in self.custom_ops

        return f"+{op}" in self.custom_ops

    # SOURCE: vllm/config/compilation.py:L1368-L1516 resolve_cudagraph_mode_
    #   and_sizes —— 后端能力最弱链降级链（min_cg_support 接口归 ch21）
    def resolve_cudagraph_mode_and_sizes(  # SOURCE: vllm/config/compilation.py:L1368-L1516
        self,
        min_cg_support: "AttentionCGSupport",
        min_cg_attn_backend: str | None,
        uniform_decode_query_len: int = 1,
        use_v2_model_runner: bool = False,
        tensor_parallel_size: int = 1,
        kv_cache_config: "KVCacheConfig | None" = None,
        max_num_reqs: int | None = None,
        is_profiling: bool = False,
    ) -> CUDAGraphMode:
        from ..v1.attention.backend import AttentionCGSupport

        cudagraph_mode = self.cudagraph_mode
        if cudagraph_mode is None or cudagraph_mode == CUDAGraphMode.NONE:
            self.cudagraph_mode = CUDAGraphMode.NONE
            return CUDAGraphMode.NONE

        # Check cudagraph for mixed batch is supported
        if (
            cudagraph_mode.mixed_mode() == CUDAGraphMode.FULL
            and min_cg_support != AttentionCGSupport.ALWAYS
        ):
            msg = (
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported "
                f"with {min_cg_attn_backend} backend (support: "
                f"{min_cg_support})"
            )
            if min_cg_support == AttentionCGSupport.NEVER:
                # if not supported any full cudagraphs, just raise it.
                msg += (
                    "; please try cudagraph_mode=PIECEWISE, and "
                    "make sure compilation mode is VLLM_COMPILE"
                )
                raise ValueError(msg)

            # attempt to resolve the full cudagraph related mode
            if self.splitting_ops_contain_attention():
                msg += "; setting cudagraph_mode=FULL_AND_PIECEWISE"
                cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
            else:
                msg += "; setting cudagraph_mode=FULL_DECODE_ONLY"
                cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
            logger.warning(msg)

        # check that if we are doing decode full-cudagraphs it is supported
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and min_cg_support == AttentionCGSupport.NEVER
        ):
            msg = (
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported "
                f"with {min_cg_attn_backend} backend (support: "
                f"{min_cg_support})"
            )
            if self.mode == CompilationMode.VLLM_COMPILE and (
                self.splitting_ops_contain_attention()
                or self.use_inductor_graph_partition
            ):
                msg += (
                    "; setting cudagraph_mode=PIECEWISE because "
                    "attention is compiled piecewise"
                )
                cudagraph_mode = CUDAGraphMode.PIECEWISE
            else:
                msg += (
                    "; setting cudagraph_mode=NONE because "
                    "attention is not compiled piecewise"
                )
                cudagraph_mode = CUDAGraphMode.NONE
            logger.warning(msg)

        # check that if we are doing spec-decode + decode full-cudagraphs it is
        # supported
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and uniform_decode_query_len > 1
            and min_cg_support.value < AttentionCGSupport.UNIFORM_BATCH.value
        ):
            msg = (
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported"
                f" with spec-decode for attention backend "
                f"{min_cg_attn_backend} (support: {min_cg_support})"
            )
            if self.splitting_ops_contain_attention():
                msg += "; setting cudagraph_mode=PIECEWISE"
                cudagraph_mode = CUDAGraphMode.PIECEWISE
            else:
                msg += "; setting cudagraph_mode=NONE"
                cudagraph_mode = CUDAGraphMode.NONE
            logger.warning(msg)

        # double check that we can support full cudagraph if they are requested
        # even after automatic downgrades
        if (
            cudagraph_mode.has_full_cudagraphs()
            and min_cg_support == AttentionCGSupport.NEVER
        ):
            raise ValueError(
                f"CUDAGraphMode.{cudagraph_mode.name} is not "
                f"supported with {min_cg_attn_backend} backend ("
                f"support:{min_cg_support}) "
                "; please try cudagraph_mode=PIECEWISE, "
                "and make sure compilation mode is VLLM_COMPILE"
            )

        # MRV1 adjusts cudagraph sizes to be a multiple of uniform_decode_query_len
        # to avoid: https://github.com/vllm-project/vllm/issues/28207 and temp-fix:
        # https://github.com/vllm-project/vllm/issues/28207#issuecomment-3504004536
        # Will be removed in the near future when we have separate cudagraph capture
        # sizes for decode and mixed prefill-decode.
        # MRV2 handles cudagraph capture sizing in cudagraph_utils.py
        # and doesn't need below: https://github.com/vllm-project/vllm/pull/45953
        if (
            not use_v2_model_runner
            and cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and uniform_decode_query_len > 1
        ):
            self.adjust_cudagraph_sizes_for_spec_decode(
                uniform_decode_query_len,
                tensor_parallel_size,
            )

        # For Mamba models with FULL decode cudagraphs, each decode
        # sequence needs one Mamba cache block. The decode cudagraph
        # dispatcher already caps batch sizes at max_num_seqs, so we just
        # need to verify that enough blocks exist. Raising here instead of
        # silently capping cudagraph_capture_sizes avoids unintended
        # restrictions on PIECEWISE (prefill) cudagraphs.
        # See: https://github.com/vllm-project/vllm/issues/34094
        if (
            kv_cache_config is not None
            and max_num_reqs is not None
            and cudagraph_mode.has_full_cudagraphs()
            and not is_profiling
            and kv_cache_config.has_mamba_layers
            and max_num_reqs > kv_cache_config.num_blocks
        ):
            raise ValueError(
                f"max_num_seqs ({max_num_reqs}) exceeds available Mamba cache "
                f"blocks ({kv_cache_config.num_blocks}). Each decode sequence "
                "requires one Mamba cache block, so CUDA graph capture cannot "
                "proceed. Please lower max_num_seqs to at most "
                f"{kv_cache_config.num_blocks} or increase "
                "gpu_memory_utilization."
            )

        self.cudagraph_mode = cudagraph_mode
        return cudagraph_mode

    # SOURCE: vllm/config/compilation.py:L1518-L1563 adjust_cudagraph_sizes_
    #   for_spec_decode —— spec-decode 捕获尺寸对齐（uniform_decode_query_len
    #   的倍数；SP 时再对齐 tp）
    def adjust_cudagraph_sizes_for_spec_decode(  # SOURCE: vllm/config/compilation.py:L1518-L1563
        self, uniform_decode_query_len: int, tensor_parallel_size: int
    ):
        multiple_of = uniform_decode_query_len
        if tensor_parallel_size > 1 and self.pass_config.enable_sp:
            multiple_of = max(uniform_decode_query_len, tensor_parallel_size)
            if (
                multiple_of % uniform_decode_query_len != 0
                or multiple_of % tensor_parallel_size != 0
            ):
                raise ValueError(
                    f"Can't determine cudagraph shapes that are both a "
                    f"multiple of {uniform_decode_query_len} "
                    f"(num_speculative_tokens + 1) required by spec-decode "
                    f"and {tensor_parallel_size} (tensor_parallel_size) "
                    f"required by sequence parallelism please adjust "
                    f"num_speculative_tokens or disable sequence parallelism"
                )

        if not self.cudagraph_capture_sizes or multiple_of <= 1:
            return

        assert self.max_cudagraph_capture_size is not None
        rounded_sizes = sorted(
            set(
                round_up(size, multiple_of)
                for size in self.cudagraph_capture_sizes
                if round_up(size, multiple_of) <= self.max_cudagraph_capture_size
            )
        )

        if len(rounded_sizes) == 0 and multiple_of <= self.max_cudagraph_capture_size:
            # if one valid but would be round_down use that
            rounded_sizes = [multiple_of]

        if len(rounded_sizes) == 0:
            raise ValueError(
                f"No valid cudagraph sizes after rounding to multiple of {multiple_of} "
                f"(num_speculative_tokens + 1 or tp if sequence parallelism is enabled)"
                f" please adjust num_speculative_tokens ({uniform_decode_query_len - 1}"
                f") or max_cudagraph_capture_size ({self.max_cudagraph_capture_size})"
                f" or cudagraph_capture_sizes ({self.cudagraph_capture_sizes})"
            )

        self.max_cudagraph_capture_size = rounded_sizes[-1]
        self.cudagraph_capture_sizes = rounded_sizes

    # SOURCE: vllm/config/compilation.py:L1565-L1570 get_compile_ranges ——
    #   compile_ranges_endpoints → Range 区间表
    def get_compile_ranges(self) -> list[Range]:  # SOURCE: vllm/config/compilation.py:L1565-L1570
        """Get the compile ranges for the compilation config."""
        if self.compile_ranges_endpoints is None:
            return []
        endpoints = sorted(set(self.compile_ranges_endpoints))
        return [Range(s + 1, e) for s, e in zip([0] + endpoints[:-1], endpoints)]
