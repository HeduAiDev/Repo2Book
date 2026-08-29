# Subtract-only companion for v3 ch19 — vllm/config/vllm.py
# (pin v0.27.1 / 6e448d0ea). Same names, same structure, same control flow;
# only dossier-approved deletions (each marked `# SUBTRACTED:`), plus
# 章范围外域段以 SUBTRACTED+归属注记收窄（impl-notes §范围裁剪）。
#
# Kept surface: OptimizationLevel + -O0..-O3 预设表（m11 站 1 的读者入口）、
# O2 谓词函数、VllmConfig 的 ch19 切面（optimization_level 字段 + post_init
# 的编译/图档落账段 L1253-L1321 + 档位默认应用）与 config-context 三件套
# （set/get_current_vllm_config、get_cached_compilation_config——custom_op 与
# wrapper 的查询面）。VllmConfig 其余 20+ 子配置与 post_init 主体是 ch03 域，
# SUBTRACTED；测试构造面经 _for_tests HOST SEAM（见 impl-notes §Host 决策）。
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, is_dataclass
from functools import lru_cache
from enum import IntEnum
from typing import Any

from .._host_seams import (
    current_platform,
    envs,
    has_flashinfer,
    init_logger,
    rocm_aiter_ops,
)
from .compilation import CUDAGraphMode, CompilationConfig, CompilationMode

logger = init_logger(__name__)


# SOURCE: vllm/config/vllm.py:L104-L116 OptimizationLevel —— -O0..-O3 枚举
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


# SUBTRACTED: PerformanceMode（L119——性能模式档，编排域）。

# SOURCE: vllm/config/vllm.py:L121-L128 IS_QUANTIZED/IS_DENSE —— 量化/稠密
#   谓词当前恒 False（真源同此；上游 lambda 化被 issue #25689 挂起）
IS_QUANTIZED = False
IS_DENSE = False
# The optimizations that depend on these properties currently set to False
# in all cases.
# if model_config is not None:
#     IS_QUANTIZED = lambda c: c.model_config.is_quantized()
#     IS_DENSE = lambda c: not c.model_config.is_model_moe()
# See https://github.com/vllm-project/vllm/issues/25689.


# SOURCE: vllm/config/vllm.py:L131-L139 enable_norm_fusion
def enable_norm_fusion(cfg: "VllmConfig") -> bool:
    """Enable if either RMS norm or quant FP8 custom op is active;
    otherwise Inductor handles fusion."""

    return (
        cfg.compilation_config.is_custom_op_enabled("rms_norm")
        or cfg.compilation_config.is_custom_op_enabled("quant_fp8")
        or cfg.kernel_config.ir_op_priority.rms_norm[0] != "native"
    )


# SOURCE: vllm/config/vllm.py:L142-L152 enable_act_fusion
def enable_act_fusion(cfg: "VllmConfig") -> bool:
    """
    Enable if either SiLU+Mul or quant FP8 custom op is active;
    otherwise Inductor handles fusion.
    Also enable for FP4 models as FP4 quant is always custom so Inductor cannot fuse it.
    """
    return (
        cfg.compilation_config.is_custom_op_enabled("silu_and_mul")
        or cfg.compilation_config.is_custom_op_enabled("quant_fp8")
        or (cfg.model_config is not None and cfg.model_config.is_nvfp4_quantized())
    )


# SOURCE: vllm/config/vllm.py:L155-L175 enable_allreduce_rms_fusion
def enable_allreduce_rms_fusion(cfg: "VllmConfig") -> bool:
    """Enable if TP > 1 and Hopper/Blackwell and flashinfer installed."""
    if current_platform.is_rocm():
        return (
            rocm_aiter_ops.is_enabled() and cfg.parallel_config.tensor_parallel_size > 1
        )

    return (
        cfg.parallel_config.tensor_parallel_size > 1
        and current_platform.is_cuda()
        and has_flashinfer()
        and (
            current_platform.is_device_capability_family(100)
            or current_platform.is_device_capability(90)
        )
    )


# SOURCE: vllm/config/vllm.py:L178-L191 enable_rope_kvcache_fusion
def enable_rope_kvcache_fusion(cfg: "VllmConfig") -> bool:
    """Enable if rotary embedding custom op is active and
    use_inductor_graph_partition is enabled.
    """
    return (
        rocm_aiter_ops.is_enabled()
        and cfg.compilation_config.is_custom_op_enabled("rotary_embedding")
        and (
            cfg.compilation_config.use_inductor_graph_partition
            or not cfg.compilation_config.splitting_ops_contain_kv_cache_update()
        )
    )


# SOURCE: vllm/config/vllm.py:L194-L200 enable_rope_kvcache_mla_fusion
def enable_rope_kvcache_mla_fusion(cfg: "VllmConfig") -> bool:
    """Enable if use_inductor_graph_partition is enabled."""

    return (
        cfg.compilation_config.use_inductor_graph_partition
        or not cfg.compilation_config.splitting_ops_contain_kv_cache_update()
    )


# SOURCE: vllm/config/vllm.py:L203-L210 enable_norm_pad_fusion
def enable_norm_pad_fusion(cfg: "VllmConfig") -> bool:
    """Enable if using AITER RMSNorm and hidden size is 2880 i.e. gpt-oss."""

    return (
        cfg.kernel_config.ir_op_priority.fused_add_rms_norm[0] == "aiter"
        and cfg.model_config is not None
        and cfg.model_config.get_hidden_size() == 2880
    )


# SOURCE: vllm/config/vllm.py:L213-L217 enable_mla_dual_rms_norm_fusion
def enable_mla_dual_rms_norm_fusion(cfg: "VllmConfig") -> bool:
    """Enable MLA dual RMS norm fusion on ROCm with AITER."""
    return rocm_aiter_ops.is_enabled()


# SOURCE: vllm/config/vllm.py:L220-L226 enable_qk_norm_rope_kvcache
def enable_qk_norm_rope_kvcache(cfg: "VllmConfig") -> bool:
    """Enable fused QK-norm + RoPE + KV cache update on ROCm with AITER."""
    if not rocm_aiter_ops.is_enabled():
        return False
    return cfg.compilation_config.is_custom_op_enabled("rotary_embedding")


# SOURCE: vllm/config/vllm.py:L229-L251 OPTIMIZATION_LEVEL_00 —— O0 预设
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
            "fuse_act_padding": enable_norm_pad_fusion,
            "fuse_mla_dual_rms_norm": enable_mla_dual_rms_norm_fusion,
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
# SOURCE: vllm/config/vllm.py:L275-L297 OPTIMIZATION_LEVEL_02 —— 默认档
OPTIMIZATION_LEVEL_02 = {
    "compilation_config": {
        "pass_config": {
            "fuse_norm_quant": enable_norm_fusion,
            "fuse_act_quant": enable_act_fusion,
            "fuse_allreduce_rms": enable_allreduce_rms_fusion,
            "fuse_attn_quant": IS_QUANTIZED,
            "enable_sp": IS_DENSE,
            "fuse_gemm_comms": IS_DENSE,
            "fuse_act_padding": enable_norm_pad_fusion,
            "fuse_mla_dual_rms_norm": enable_mla_dual_rms_norm_fusion,
            "fuse_rope_kvcache": enable_rope_kvcache_fusion,
            "fuse_qk_norm_rope_kvcache": enable_qk_norm_rope_kvcache,
            "enable_qk_norm_rope_fusion": False,
            "fuse_rope_kvcache_cat_mla": enable_rope_kvcache_mla_fusion,
        },
        "cudagraph_mode": CUDAGraphMode.FULL_AND_PIECEWISE,
        "use_inductor_graph_partition": False,
    },
    "kernel_config": {
        "enable_flashinfer_autotune": True,
    },
}
# SOURCE: vllm/config/vllm.py:L298-L320 OPTIMIZATION_LEVEL_03（暂同 O2）
OPTIMIZATION_LEVEL_03 = {
    "compilation_config": {
        "pass_config": {
            "fuse_norm_quant": enable_norm_fusion,
            "fuse_act_quant": enable_act_fusion,
            "fuse_allreduce_rms": enable_allreduce_rms_fusion,
            "fuse_attn_quant": IS_QUANTIZED,
            "enable_sp": IS_DENSE,
            "fuse_gemm_comms": IS_DENSE,
            "fuse_act_padding": enable_norm_pad_fusion,
            "fuse_mla_dual_rms_norm": enable_mla_dual_rms_norm_fusion,
            "fuse_rope_kvcache": enable_rope_kvcache_fusion,
            "fuse_qk_norm_rope_kvcache": enable_qk_norm_rope_kvcache,
            "enable_qk_norm_rope_fusion": False,
            "fuse_rope_kvcache_cat_mla": enable_rope_kvcache_mla_fusion,
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


# SUBTRACTED: vllm/config/vllm.py VllmConfig 的 20+ 子配置字段与其
#   compute_hash/__repr__/update_from_env 等（L330-L809、L431-L808——ch03
#   配置域）；本章切面只保留 ch19 消费的载体字段。use_v2_model_runner /
#   performance_mode / weight_transfer_config / shutdown_timeout 等随其
#   消费分支删除（V2 实验态、编排域）。

# HOST SEAM: KernelConfig 载面（真源 vllm/config/kernel.py——kernel 调优/
#   ir_op_priority 域，ch12/ch27 消费）。O2 谓词只读
# enable_flashinfer_autotune 与 ir_op_priority 两个键；宿主缺省走 native/
# False 路径（非 flashinfer/非 aiter 行为）。dataclass 以便
# _apply_optimization_level_defaults 递归应用预设。
@dataclass
class KernelConfig:  # HOST SEAM
    enable_flashinfer_autotune: bool | None = None

    def __post_init__(self):  # SOURCE: vllm/config/vllm.py:L950-L1321 __post_init__（本章切面自 L1253 起）
        # HOST SEAM: ir_op_priority 载面（"native" 优先级=宿主非融合路径）
        self.ir_op_priority = _IrOpPrioritySeam()

    # SOURCE: vllm/config/kernel.py KernelConfig.set_platform_defaults
    def set_platform_defaults(self, vllm_config) -> None:  # HOST SEAM
        return None


# HOST SEAM: ir_op_priority 的最小载面（"native" 优先级=宿主非融合路径）
class _IrOpPrioritySeam:  # HOST SEAM  # SOURCE: vllm/config/kernel.py
    rms_norm = ("native", 0)
    fused_add_rms_norm = ("native", 0)


# HOST SEAM: ParallelConfig 载面（真源 vllm/config/parallel.py——ch03/ch34
#   域；本章 O2 谓词只读 tensor_parallel_size）
@dataclass
class ParallelConfig:  # HOST SEAM  # SOURCE: vllm/config/parallel.py tensor_parallel_size（HOST SEAM 载面）
    tensor_parallel_size: int = 1


# SOURCE: vllm/config/vllm.py:L330-L429 VllmConfig（ch19 切面字段集）——
#   逐字段 SUBTRACTED 注记见 __post_init__ 头；optimization_level 默认 O2
#   （真源 L409）。
@dataclass
class VllmConfig:
    """Dataclass which contains all vllm-related configuration. This
    simplifies passing around the distinct configurations in the codebase.

    （本章切面：仅 ch19 消费的子配置载体。真源 20+ 子配置字段为 ch03 域，
    SUBTRACTED；见 impl-notes §范围裁剪。）
    """

    model_config: Any = None
    """Model configuration."""
    parallel_config: Any = field(default_factory=ParallelConfig)
    """Parallel configuration (O2 预设谓词的查询面)."""
    quant_config: Any = None
    """Quantization configuration (has_blocked_weights 查询面，F10 苗)."""
    kernel_config: Any = field(default_factory=KernelConfig)
    """Kernel tuning configuration (O2 预设谓词的查询面)."""
    speculative_config: Any = None
    """Speculative decoding configuration."""
    observability_config: Any = None
    """Observability configuration."""
    compilation_config: CompilationConfig = field(default_factory=CompilationConfig)
    """Compilation configuration."""
    optimization_level: OptimizationLevel = OptimizationLevel.O2
    """The optimization level. These levels trade startup time cost for
    performance, with -O0 having the best startup time and -O3 having the best
    performance. -O2 is used by default. See OptimizationLevel for full
    description."""

    # SUBTRACTED: cache/scheduler/device/load/offload/attention/lora/
    #   kv_transfer/... 子配置字段（L340-L429——ch03 域；本章消费的
    #   scheduler_config 等面由测试与调用方以属性载体注入）。

    @classmethod
    def _for_tests(cls, **kw) -> "VllmConfig":  # SOURCE: vllm/config/vllm.py:L826-L853（HOST SEAM 测试构造面：预设应用/落账锚）
        # HOST SEAM: 测试构造面 —— 真源 VllmConfig 需全量子配置（ch03 域）；
        # 本切面收 optimization_level / quant_config / compilation_config /
        # model_config / kernel_config / speculative_config /
        # observability_config 七键，backend= 便捷键折进 compilation_config
        # （构造后 dataclass __init__ 自跑 __post_init__）。
        backend = kw.pop("backend", None)
        if backend is not None:
            cc = kw.get("compilation_config") or CompilationConfig()
            cc.backend = backend
            kw["compilation_config"] = cc
        return cls(**kw)

    # SOURCE: vllm/config/vllm.py:L811-L824 _set_config_default —— 用户未设
    #   （None）才应用预设值；callable 谓词以 self 为根求值
    def _set_config_default(self, config_obj: Any, key: str, value: Any) -> None:  # SOURCE: vllm/config/vllm.py:L811-L824
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

    # SOURCE: vllm/config/vllm.py:L826-L853 _apply_optimization_level_defaults
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
            # SOURCE: vllm/config/vllm.py:L841-L851（self 为根递归应用）
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

    # SUBTRACTED: _maybe_override_dynamic_sd_cudagraph_mode /
    #   _maybe_disable_dynamic_sd_for_data_parallel（L855-L891——动态 spec
    #   decode 域）、_post_init_kv_transfer_config/_verify_kv_transfer_compat
    #   （L893-L948——ch16 KV connector 域）及 post_init 主体其余
    #   （L950-L1240——模型/缓存/调度器装配，ch03 域）。

    def __post_init__(self) -> None:
        # SUBTRACTED: VllmConfig.__post_init__ 前段（L950-L1252——hf_config
        #   对齐、 enforce_eager/TORCH_COMPILE_DISABLE 环境降级、breakable
        #   cudagraph 自动启用、动态 SD 等模型/环境域；本章切面自 L1253 起）。

        # SOURCE: vllm/config/vllm.py:L1253-L1259 has_blocked_weights —— 块状
        #   量化权重的探测闭包
        def has_blocked_weights():
            # SOURCE: vllm/config/vllm.py:L1254-L1259
            if self.quant_config is not None:
                if hasattr(self.quant_config, "weight_block_size"):
                    return self.quant_config.weight_block_size is not None
                elif hasattr(self.quant_config, "has_blocked_weights"):
                    return self.quant_config.has_blocked_weights()
            return False

        # Enable quant_fp8 CUDA ops (TODO disable in follow up)
        # On H100 the CUDA kernel is faster than
        # native implementation
        # https://github.com/vllm-project/vllm/issues/25094
        # SOURCE: vllm/config/vllm.py:L1261-L1268（F10 苗：量化改变算子选择）
        if has_blocked_weights():
            custom_ops = self.compilation_config.custom_ops
            if "-quant_fp8" not in custom_ops:
                custom_ops.append("+quant_fp8")

        # SOURCE: vllm/config/vllm.py:L1270 平台配置默认值钩子
        current_platform.apply_config_platform_defaults(self)

        # SOURCE: vllm/config/vllm.py:L1272-L1276 档位落账：>O0 → VLLM_COMPILE
        if self.compilation_config.mode is None:
            if self.optimization_level > OptimizationLevel.O0:
                self.compilation_config.mode = CompilationMode.VLLM_COMPILE
            else:
                self.compilation_config.mode = CompilationMode.NONE

        # By default, enable torch wrapping only when using custom Inductor lowering
        # SOURCE: vllm/config/vllm.py:L1278-L1283
        if self.compilation_config.ir_enable_torch_wrap is None:
            self.compilation_config.ir_enable_torch_wrap = (
                self.compilation_config.mode == CompilationMode.VLLM_COMPILE
                and self.compilation_config.backend == "inductor"
            )

        # SOURCE: vllm/config/vllm.py:L1285-L1292 custom_ops 基础档落账：
        #   Inductor 编译时 'none'（走 forward_native 让编译器融合）、否则 'all'
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
        # SOURCE: vllm/config/vllm.py:L1294-L1297
        self.kernel_config.set_platform_defaults(self)

        # SOURCE: vllm/config/vllm.py:L1299-L1305 应用档位预设（None 字段
        #   才被覆盖——用户设定优先）
        default_config = OPTIMIZATION_LEVEL_TO_CONFIG[self.optimization_level]
        self._apply_optimization_level_defaults(default_config)
        if self.kernel_config.enable_flashinfer_autotune is None:
            raise ValueError(
                "KernelConfig.enable_flashinfer_autotune must be set after applying "
                "optimization level defaults."
            )

        # SUBTRACTED: _maybe_disable_dynamic_sd_for_data_parallel /
        #   _maybe_override_dynamic_sd_cudagraph_mode 调用行（L1307-L1308
        #   ——动态 spec decode 域）。

        # SOURCE: vllm/config/vllm.py:L1310-L1321 档位与编译模式的相容闸：
        #   PIECEWISE 需 VLLM_COMPILE，否则降 NONE
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

        # SUBTRACTED: SP/fuse_gemm_comms 尾段（L1323-L1353——SP 阈值启发，
        #   TP>1 部署域）与 LayerName/fast_moe_cold_start 尾段（L1355 起
        #   ——torch>=2.11 强制关 fast_moe 等，MoE 域）。


# SOURCE: vllm/config/vllm.py:L2369-L2370 config-context 模块级载体
_current_vllm_config: VllmConfig | None = None
_current_prefix: str | None = None


# SOURCE: vllm/config/vllm.py:L2373-L2425 set_current_vllm_config —— 模型
#   构造期注入当前配置（custom_op 的 dispatch 查询面）
@contextmanager
def set_current_vllm_config(  # SOURCE: vllm/config/vllm.py:L2373-L2425
    vllm_config: VllmConfig, check_compile=False, prefix: str | None = None
):
    """
    Temporarily set the current vLLM config.
    Used during model initialization.
    We save the current vLLM config in a global variable,
    so that all modules can access it, e.g., custom ops
    can access the vLLM config to determine how to dispatch.
    """
    global _current_vllm_config, _current_prefix
    old_vllm_config = _current_vllm_config
    old_prefix = _current_prefix
    from ..compilation.counter import compilation_counter

    num_models_seen = compilation_counter.num_models_seen
    try:
        # Clear the compilation config cache when context changes.
        # This is needed since the old config may have been accessed
        # and cached before the new config is set.
        get_cached_compilation_config.cache_clear()

        _current_vllm_config = vllm_config
        _current_prefix = prefix
        yield
    except Exception:
        raise
    else:
        if check_compile:
            vllm_config.compilation_config.custom_op_log_check()

        if (
            check_compile
            and vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
            and compilation_counter.num_models_seen == num_models_seen
        ):
            # If the model supports compilation,
            # compilation_counter.num_models_seen should be increased
            # by at least 1.
            # If it is not increased, it means the model does not support
            # compilation (does not have @support_torch_compile decorator).
            logger.warning(
                "`torch.compile` is turned on, but the model %s"
                " does not support it. Please open an issue on GitHub"
                " if you want it to be supported.",
                vllm_config.model_config.model,
            )
    finally:
        _current_vllm_config = old_vllm_config
        _current_prefix = old_prefix
        # Clear the compilation config cache when context changes
        get_cached_compilation_config.cache_clear()


# SOURCE: vllm/config/vllm.py:L2428-L2431 get_cached_compilation_config
@lru_cache(maxsize=1)
def get_cached_compilation_config():  # SOURCE: vllm/config/vllm.py:L2428-L2431
    """Cache config to avoid repeated calls to get_current_vllm_config()"""
    return get_current_vllm_config().compilation_config


# SOURCE: vllm/config/vllm.py:L2434-L2444 get_current_vllm_config
def get_current_vllm_config() -> VllmConfig:
    if _current_vllm_config is None:
        raise AssertionError(
            "Current vLLM config is not set. This typically means "
            "get_current_vllm_config() was called outside of a "
            "set_current_vllm_config() context, or a CustomOp was instantiated "
            "at module import time or model forward time when config is not set. "
            "For tests that directly test custom ops/modules, use the "
            "'default_vllm_config' pytest fixture from tests/conftest.py."
        )
    return _current_vllm_config


# SOURCE: vllm/config/vllm.py:L2447-L2448 get_current_vllm_config_or_none
def get_current_vllm_config_or_none() -> VllmConfig | None:
    return _current_vllm_config
