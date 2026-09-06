# HOST SEAM 登记处（ch21）——真实代码之外唯一允许的承载面
# 每一项都在行内标注真实源锚；本章在 CPU host（无 CUDA、无 vllm 包）跑通
# 选后端→归组→定形→每拍 metadata→读写两腿的全链测试。分五类：
#   A. logger/envs/分布式组探测：vllm 顶层设施的 host 替身；
#   B. 平台面：current_platform 惰性转出 + DeviceCapability 真身 + 算力代
#      装配位（真实 pynvml/NVML 探测不进 host）；
#   C. 配置面：VllmConfig 各 namespace 的本章消费字段子集（ch03/ch17 域
#      全量配置不进；被删支的开关取默认关的真实值、守卫位永不触发）；
#   D. 配置上下文与装饰器：set_current_vllm_config / @config（真身
#      vllm/config/vllm.py 与 vllm/config/utils.py，机制逐字）；
#   E. fa_utils 版本探测装配位：host 无 vllm_flash_attn 库，恒 FA2。
from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import contextmanager
from enum import Enum
from typing import Any

import torch
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as _pydantic_dataclass


# ── A. vllm 顶层设施的 host 替身 ───────────────────────────────────────────

# SOURCE: vllm/logger.py:L118-L143 _OnceLogger 的 info_once/debug_once/
#   warning_once —— HOST SEAM：once-守卫（scope=local 按 (msg,args) 去重）
#   退化为直接转发（once 语义只影响日志量、不影响控制流；caplog 可见）。
class _LoggerOnceSeam:
    # SOURCE: vllm/logger.py —— HOST SEAM：包一层真 logging.Logger
    def __init__(self, logger: logging.Logger):  # HOST SEAM 装配位
        self._logger = logger

    def _log(self, level, msg, *args, **kwargs):  # HOST SEAM：转发位  # SOURCE: vllm/logger.py:L109 _VllmLogger(Logger)——_log 是 stdlib logging.Logger 的转发原语，真身无 vllm 侧定义
        self._logger.log(level, msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):  # HOST SEAM  # SOURCE: vllm/logger.py
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):  # HOST SEAM  # SOURCE: vllm/logger.py
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):  # HOST SEAM  # SOURCE: vllm/logger.py
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):  # HOST SEAM  # SOURCE: vllm/logger.py
        self._logger.error(msg, *args, **kwargs)

    # SOURCE: vllm/logger.py:L118 debug_once
    def debug_once(self, msg, *args, **kwargs):  # HOST SEAM：once 守卫退化
        self._logger.debug(msg, *args, **kwargs)

    # SOURCE: vllm/logger.py:L127 info_once
    def info_once(self, msg, *args, **kwargs):  # HOST SEAM：once 守卫退化
        self._logger.info(msg, *args, **kwargs)

    # SOURCE: vllm/logger.py:L136 warning_once
    def warning_once(self, msg, *args, **kwargs):  # HOST SEAM：once 守卫退化
        self._logger.warning(msg, *args, **kwargs)


# SOURCE: vllm/logger.py init_logger —— HOST SEAM：真实 init_logger 返回
#   vllm.<module> 命名的 Logger；host 镜像把包名 implementation.* 映射回
#   vllm.* 命名空间并包 once 族。
def init_logger(name: str) -> _LoggerOnceSeam:  # SOURCE: vllm/logger.py:L204
    real_name = ("vllm" + name[len("implementation"):]) if name.startswith(
        "implementation"
    ) else name
    return _LoggerOnceSeam(logging.getLogger(real_name))


# SOURCE: vllm/envs.py —— HOST SEAM：本章消费的环境位（取真实默认值）
class envs:  # HOST SEAM
    # SOURCE: vllm/envs.py VLLM_BATCH_INVARIANT（默认 False）
    VLLM_BATCH_INVARIANT: bool = False
    # SOURCE: vllm/envs.py VLLM_USE_LAYERNAME（torch>=2.11 默认 True）
    VLLM_USE_LAYERNAME: bool = True
    # SOURCE: vllm/envs.py VLLM_KV_CACHE_LAYOUT（默认 None——用户环境变量覆写位）
    VLLM_KV_CACHE_LAYOUT: str | None = None


# SOURCE: vllm/distributed/parallel_state.py get_dcp_group/get_pcp_group 的
#   HOST SEAM：与真实「测试环境未初始化组」同型抛 AssertionError——
#   AttentionImplBase.__new__ 与 FlashAttentionMetadataBuilder.__init__ 的
#   try/except 捕获后退化单卡（源码原生路径）。
def get_dcp_group():  # SOURCE: vllm/distributed/parallel_state.py
    raise AssertionError("Default process group has not been initialized.")


def get_pcp_group():  # SOURCE: vllm/distributed/parallel_state.py
    raise AssertionError("Default process group has not been initialized.")


# SOURCE: vllm/utils/torch_utils.py record_function_or_nullcontext —— HOST
#   SEAM：无 profiler，nullcontext 直通。
@contextmanager  # type: ignore[arg-type]
def record_function_or_nullcontext(name: str):  # SOURCE: vllm/utils/torch_utils.py
    yield


# ── B. 平台面 ──────────────────────────────────────────────────────────────

# SOURCE: vllm/platforms/interface.py:L89-L119 DeviceCapability ——（逐字）
#   NamedTuple + 全序比较：validate 探针（supports_compute_capability 的
#   >= DeviceCapability(8, 0)）与优先级表分档（major 10/12）都吃它。
class DeviceCapability:
    # SOURCE: vllm/platforms/interface.py:L90-L91 字段（NamedTuple 展开）
    def __init__(self, major: int, minor: int):  # HOST SEAM: NamedTuple 面
        self.major = major
        self.minor = minor

    # SOURCE: vllm/platforms/interface.py:L93-L96 __lt__
    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, DeviceCapability):
            return NotImplemented
        return (self.major, self.minor) < (other.major, other.minor)

    # SOURCE: vllm/platforms/interface.py:L98-L101 __le__
    def __le__(self, other: Any) -> bool:
        if not isinstance(other, DeviceCapability):
            return NotImplemented
        return (self.major, self.minor) <= (other.major, other.minor)

    # SOURCE: vllm/platforms/interface.py:L103-L106 __eq__
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DeviceCapability):
            return NotImplemented
        return (self.major, self.minor) == (other.major, other.minor)

    # SOURCE: vllm/platforms/interface.py:L108-L111 __ge__
    def __ge__(self, other: Any) -> bool:
        if not isinstance(other, DeviceCapability):
            return NotImplemented
        return (self.major, self.minor) >= (other.major, other.minor)

    # SOURCE: vllm/platforms/interface.py:L113-L116 __gt__
    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, DeviceCapability):
            return NotImplemented
        return (self.major, self.minor) > (other.major, other.minor)

    # SOURCE: vllm/platforms/interface.py:L118-L119 __hash__
    def __hash__(self) -> int:
        return hash((self.major, self.minor))

    def __repr__(self) -> str:  # HOST SEAM: NamedTuple 自带 repr 的等价面  # SOURCE: vllm/platforms/interface.py:L89 DeviceCapability(NamedTuple)——__repr__ 由 typing.NamedTuple 机制自带，真实类未显式定义
        return f"DeviceCapability(major={self.major}, minor={self.minor})"


# SOURCE: vllm/platforms/__init__.py current_platform —— HOST SEAM 平台面：
#   真实经 detect_platform() 按 host 探测选平台；host 无 CUDA 时会选 CPU 平台
#   （其 get_attn_backend_cls 是 interface.py:L373 的空实现）。本章讲 CUDA
#   优先级表，故 get_attn_backend_cls / get_device_capability 惰性转出到
#   platforms/cuda.py 的 CudaPlatform 切面（真身 L208 CudaPlatformBase）。
class _PlatformSeam:
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    # SOURCE: vllm/platforms/interface.py Platform.device_name —— HOST SEAM：
    #   _cached_get_attn_backend 的 ValueError 消息读它（ch17 平台域）
    device_name = "cuda"

    # SOURCE: vllm/platforms/interface.py Platform.dispatch_key —— HOST SEAM:
    #   host 全 CPU 张量跑（direct_register_custom_op 需 CPU dispatch key）；
    #   "CUDA" 面是 vllm 容器域。
    @property
    def dispatch_key(self) -> str:  # SOURCE: vllm/platforms/interface.py
        return "CPU"

    # SOURCE: vllm/platforms/interface.py Platform.is_cuda / is_cpu / ...
    def is_cuda(self) -> bool:
        return torch.cuda.is_available()

    def is_cuda_alike(self) -> bool:  # SOURCE: vllm/platforms/interface.py
        return torch.cuda.is_available()

    def is_rocm(self) -> bool:  # SOURCE: vllm/platforms/interface.py
        return False

    def is_cpu(self) -> bool:  # SOURCE: vllm/platforms/interface.py
        return not torch.cuda.is_available()

    def is_xpu(self) -> bool:  # SOURCE: vllm/platforms/interface.py
        return False

    def is_device_capability_family(self, capability: int) -> bool:
        # SOURCE: vllm/platforms/interface.py:L481 is_device_capability_family
        #   ——HOST SEAM：惰性转出 CudaPlatform 切面
        from .platforms.cuda import CudaPlatform

        cap = CudaPlatform.get_device_capability()
        return cap is not None and cap.major == capability // 10

    # SOURCE: vllm/platforms/cuda.py:L569 CudaPlatform.opaque_attention_op
    #   （cuda 平台返回 True——attention 注册成一个大不透明算子）—— HOST
    #   SEAM：host 走 direct-call 分支；torch.ops 分支在容器内同控制流。
    def opaque_attention_op(self) -> bool:  # SOURCE: vllm/platforms/interface.py:L1116
        return False

    # SOURCE: vllm/platforms/interface.py:L373-L380 Platform.get_attn_backend_cls
    #   （基类空实现）—— HOST SEAM：惰性转出 CudaPlatform 切面（选后端主线）
    def get_attn_backend_cls(self, selected_backend, attn_selector_config, num_heads=None):  # SOURCE: vllm/platforms/interface.py:L373-L380
        from .platforms.cuda import CudaPlatform

        return CudaPlatform.get_attn_backend_cls(
            selected_backend, attn_selector_config, num_heads=num_heads
        )

    # SOURCE: vllm/platforms/interface.py:L420-L430 Platform.get_device_capability
    #   —— HOST SEAM：惰性转出 CudaPlatform 切面
    def get_device_capability(self):  # SOURCE: vllm/platforms/interface.py:L420-L430
        from .platforms.cuda import CudaPlatform

        return CudaPlatform.get_device_capability()

    # SOURCE: vllm/platforms/interface.py Platform.set_additional_forward_context
    #   —— HOST SEAM：平台无附加上下文，恒空 dict。
    def set_additional_forward_context(self, **kwargs) -> dict:  # SOURCE: vllm/platforms/interface.py:L1265
        return {}

    # SOURCE: vllm/platforms/interface.py Platform.fp8_dtype —— HOST SEAM
    #   （schedule 闭包量化分支读它；本章 kv_cache_dtype="auto" 恒不触发）
    def fp8_dtype(self):  # SOURCE: vllm/platforms/interface.py
        return torch.float8_e4m3fn


# SOURCE: vllm/compilation/breakable_cudagraph.py eager_break_during_capture —
#   HOST SEAM identity（真实：可断 cudagraph 捕获激活时在算子两侧结束当前
#   stream-capture 段；禁用路径就是恒等透传——本章钉的分支，ch19 域）
def eager_break_during_capture(fn):  # HOST SEAM  # SOURCE: vllm/compilation/breakable_cudagraph.py
    return fn


# SOURCE: vllm/model_executor/layers/attention/kv_transfer_utils.py
#   maybe_transfer_kv_layer —— HOST SEAM identity（kv-connector 传输是 ch16
#   域；无 connector 路径即恒等装饰器）
def maybe_transfer_kv_layer(fn):  # HOST SEAM  # SOURCE: vllm/model_executor/layers/attention/kv_transfer_utils.py
    return fn


current_platform = _PlatformSeam()  # HOST SEAM


# SOURCE: vllm/v1/attention/backends/fa_utils.py:L140-L282 get_flash_attn_
#   version —— HOST SEAM 装配位：真身按 vllm_flash_attn 可用性与算力代解析
#   3/4/2（容器内 SM90→FA3、SM100→FA4、否则 FA2）；host 无库——try/except
#   ImportError 的真实回退是 None，但 None 会让 FlashAttentionImpl 不可用，
#   故装配为恒 FA2（FA2 语义可由 host 镜像精确承载，ch22 同款）。
def get_flash_attn_version(  # SOURCE: vllm/v1/attention/backends/fa_utils.py:L140-L282
    requires_alibi: bool = False,
    head_size: int | None = None,
    head_size_v: int | None = None,
    has_sinks: bool = False,
    requires_local_attention: bool = False,
) -> int:
    return 2


# ── C. 配置面占位（ch03/ch17 域 namespace；取值=精简配置的真实值） ─────────

# SOURCE: vllm/config/compilation.py:L53-L104 CUDAGraphMode ——（逐字）ch19
#   编译/捕获域枚举；本章 _check_and_update_cudagraph_mode 的降级链接口侧。
class CUDAGraphMode(Enum):
    """Constants for the cudagraph mode in CompilationConfig.
    Meanwhile, the subset enum `NONE`, `PIECEWISE` and `FULL` are also
    treated as concrete runtime mode for cudagraph runtime dispatching.
    """

    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = (FULL, NONE)
    FULL_AND_PIECEWISE = (FULL, PIECEWISE)

    # SOURCE: vllm/config/compilation.py:L60 decode_mode
    def decode_mode(self) -> "CUDAGraphMode":
        return CUDAGraphMode(self.value[0]) if self.separate_routine() else self

    # SOURCE: vllm/config/compilation.py:L63 mixed_mode
    def mixed_mode(self) -> "CUDAGraphMode":
        return CUDAGraphMode(self.value[1]) if self.separate_routine() else self

    # SOURCE: vllm/config/compilation.py:L66 has_mode
    def has_mode(self, mode: "CUDAGraphMode") -> bool:
        assert not mode.separate_routine()
        if self.separate_routine():
            return mode.value in self.value
        return self == mode

    # SOURCE: vllm/config/compilation.py:L74 requires_piecewise_compilation
    def requires_piecewise_compilation(self) -> bool:
        return self.has_mode(CUDAGraphMode.PIECEWISE)

    # SOURCE: vllm/config/compilation.py:L77 max_cudagraph_mode
    def max_cudagraph_mode(self) -> "CUDAGraphMode":
        return CUDAGraphMode(max(self.value)) if self.separate_routine() else self

    # SOURCE: vllm/config/compilation.py:L80 has_full_cudagraphs
    def has_full_cudagraphs(self) -> bool:
        return self.max_cudagraph_mode() == CUDAGraphMode.FULL

    # SOURCE: vllm/config/compilation.py:L83 has_piecewise_cudagraphs
    def has_piecewise_cudagraphs(self) -> bool:
        return self.requires_piecewise_compilation()

    # SOURCE: vllm/config/compilation.py:L86 separate_routine
    def separate_routine(self) -> bool:
        return isinstance(self.value, tuple)

    @classmethod
    # SOURCE: vllm/config/compilation.py:L90 valid_runtime_modes
    def valid_runtime_modes(cls) -> frozenset["CUDAGraphMode"]:
        return frozenset({cls.NONE, cls.PIECEWISE, cls.FULL})

    # SOURCE: vllm/config/compilation.py:L96 is_valid_runtime_mode
    def is_valid_runtime_mode(self) -> bool:
        return self in CUDAGraphMode.valid_runtime_modes()

    # SOURCE: vllm/config/compilation.py:L100 __str__
    def __str__(self) -> str:
        return self.name

    # SOURCE: vllm/config/compilation.py:L103 __bool__
    def __bool__(self) -> bool:
        return self != CUDAGraphMode.NONE


# SOURCE: vllm/config/compilation.py CompilationConfig.resolve_cudagraph_mode_
#   and_sizes —— ch19 域全文（最弱链降级链：min_cg_support → mode/size 裁决）；
#   本章只取其接口侧。HOST SEAM 装配位：记录收到的 min_cg_support 供测试断言。
class _CompilationConfigSeam:
    # SOURCE: vllm/config/compilation.py CompilationConfig.__init__ —— HOST
    #   SEAM 装配位（本章消费字段：static_forward_context / cudagraph_mode /
    #   max_cudagraph_capture_size / fast_moe_cold_start）
    def __init__(self):  # SOURCE: vllm/config/compilation.py
        # static_forward_context: layer_name → Attention 层实例（真身由
        # torch.compile 包装器装配，ch19 域；Attention.__init__ 在此注册）
        self.static_forward_context: dict[str, Any] = {}
        self.cudagraph_mode = CUDAGraphMode.NONE
        self.max_cudagraph_capture_size: int | None = None
        self.fast_moe_cold_start = False
        # HOST SEAM 观测位：最弱链降级链收到的 (min_cg_support, backend 名)
        self.seam_last_min_cg_support = None
        self.seam_last_min_cg_backend = None

    # SOURCE: vllm/config/compilation.py:L1368 resolve_cudagraph_mode_and_sizes
    #   —— HOST SEAM 装配位（ch19 域真身做 FULL→PIECEWISE→NONE 降级与 size
    #   集裁剪；本章记录输入、按 FA2=UNIFORM_BATCH 的真实结论给 NONE 档）
    def resolve_cudagraph_mode_and_sizes(  # SOURCE: vllm/config/compilation.py:L1368
        self,
        min_cg_support,
        min_cg_attn_backend,
        uniform_decode_query_len,
        use_v2_model_runner=False,
        tensor_parallel_size=1,
        kv_cache_config=None,
        max_num_reqs=None,
        is_profiling=False,
    ):
        self.seam_last_min_cg_support = min_cg_support
        self.seam_last_min_cg_backend = min_cg_attn_backend
        if min_cg_support is not None and min_cg_support.value < 2:
            # HOST SEAM：UNIFORM_BATCH(2) 以下按真实降级链落 PIECEWISE（FA2
            # 的 FULL 档被压掉的语义位；size 裁剪细节 → ch19 全文）
            return CUDAGraphMode.PIECEWISE
        return self.cudagraph_mode


# SOURCE: vllm/v1/cudagraph_dispatcher.py CudagraphDispatcher.initialize_
#   cudagraph_keys —— ch19 域（降级后 keys 预生成）；HOST SEAM 观测位。
class _CudagraphDispatcherSeam:
    def __init__(self):  # HOST SEAM 装配位  # SOURCE: vllm/v1/cudagraph_dispatcher.py
        self.seam_initialized_modes = []

    # SOURCE: vllm/v1/cudagraph_dispatcher.py initialize_cudagraph_keys
    def initialize_cudagraph_keys(self, cudagraph_mode, uniform_decode_query_len=None):
        self.seam_initialized_modes.append(cudagraph_mode)


# SOURCE: vllm/config/cache.py CacheConfig 本章消费字段面 —— HOST SEAM 装配位
class _CacheConfigSeam:
    # SOURCE: vllm/config/cache.py CacheConfig.__init__ —— HOST SEAM 装配位
    def __init__(self):
        self.block_size = 16
        self.user_specified_block_size = False  # 只有用户显式 --block-size 才 True
        self.cache_dtype = "auto"
        self.calculate_kv_scales = False
        self.enable_prefix_caching = False
        self.kv_cache_dtype_skip_layers = None
        self.sliding_window = None
        self.gpu_memory_utilization = 0.9
        self.mamba_cache_mode = "none"  # mamba 缓冲域（delete[8] 守卫位）
        self.skip_page_size_padded = None  # skip-quant 共享页（ch22 域守卫位）


# SOURCE: vllm/config/model.py ModelConfig 本章消费字段面 —— HOST SEAM 装配位
class _ModelConfigSeam:
    # SOURCE: vllm/config/model.py ModelConfig.__init__ —— HOST SEAM 装配位
    def __init__(self, max_model_len: int):
        self.max_model_len = max_model_len
        self.is_mm_prefix_lm = False  # mm_prefix 域（delete[9] 删支守卫位）
        self.rswa_window = None  # R-SWA 域（delete[5]/[7] 删支守卫位）
        self.is_encoder_decoder = False
        self.dtype = torch.float16

        # SOURCE: vllm/config/model.py hf_config —— HOST SEAM（ch23 模型域
        #   装配面；bind_kv_cache 的 longcat 双 attn 模块判别读 model_type）
        class _HfConfig:  # HOST SEAM  # SOURCE: vllm/config/model.py
            model_type = "llama"

        self.hf_config = _HfConfig()

    # SOURCE: vllm/config/model.py ModelConfig.get_num_attention_heads ——
    #   HOST SEAM 装配位（Builder.__init__ L381-L384 消费；TP 分头 → ch23）
    def get_num_attention_heads(self, parallel_config) -> int:  # SOURCE: vllm/config/model.py
        return 4

    # SOURCE: vllm/config/model.py ModelConfig.get_num_kv_heads —— HOST SEAM
    def get_num_kv_heads(self, parallel_config) -> int:
        return 2

    # SOURCE: vllm/config/model.py ModelConfig.get_head_size —— HOST SEAM
    def get_head_size(self) -> int:
        return 64


# SOURCE: vllm/config/parallel.py ParallelConfig 本章消费字段面 —— HOST SEAM
class _ParallelConfigSeam:
    # SOURCE: vllm/config/parallel.py ParallelConfig.__init__ —— HOST SEAM
    def __init__(self):
        self.prefill_context_parallel_size = 1
        self.decode_context_parallel_size = 1
        self.cp_kv_cache_interleave_size = 1
        self.use_ubatching = False
        self.num_ubatches = 1
        self.tensor_parallel_size = 1
        self.data_parallel_size = 1


# SOURCE: vllm/config/scheduler.py SchedulerConfig 本章消费字段面 —— HOST SEAM
class _SchedulerConfigSeam:
    # SOURCE: vllm/config/scheduler.py SchedulerConfig.__init__ —— HOST SEAM
    def __init__(self, max_num_seqs: int, max_num_batched_tokens: int):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens


# SOURCE: vllm/config/vllm.py VllmConfig 本章消费子集 —— HOST SEAM 装配位
#   （AttentionConfig 是真身——本章主角用户面；其余 namespace 是消费面占位）
class VllmConfigSeam:
    """HOST SEAM 配置面：vllm_config 的本章消费字段子集。"""

    # SOURCE: vllm/config/vllm.py VllmConfig.__init__ —— HOST SEAM 装配位
    def __init__(self, max_num_seqs: int, max_batched_tokens: int, max_model_len: int):
        self.model_config = _ModelConfigSeam(max_model_len)
        self.cache_config = _CacheConfigSeam()
        self.parallel_config = _ParallelConfigSeam()
        self.scheduler_config = _SchedulerConfigSeam(max_num_seqs, max_batched_tokens)
        self.compilation_config = _CompilationConfigSeam()
        # attention_config 由测试按需换装真 AttentionConfig（默认即真身）
        from .config.attention import AttentionConfig

        self.attention_config = AttentionConfig()
        self.speculative_config = None  # spec 域（delete[7]/[8] 删支守卫位）
        self.kv_transfer_config = None  # KV connector 域（delete[8]）


# ── D. 配置上下文与装饰器 ──────────────────────────────────────────────────

# SOURCE: vllm/config/vllm.py:L~2400-L2432 _current_vllm_config 模块级持有 +
#   set_current_vllm_config 上下文 —— HOST SEAM：机制逐字（模块级单值 +
#   保存/恢复），供 get_current_vllm_config（L2434）取用。
_current_vllm_config = None


# SOURCE: vllm/config/vllm.py:L2434-L2445 get_current_vllm_config ——（逐字）
def get_current_vllm_config():
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


# SOURCE: vllm/config/vllm.py:L2447-L2448 get_current_vllm_config_or_none（逐字）
def get_current_vllm_config_or_none():
    return _current_vllm_config


# SOURCE: vllm/config/vllm.py:L2454-L2475 get_layers_from_vllm_config ——（逐字）
#   从 static_forward_context 按 layer_names + 类型筛层（initialize_attn_
#   backend 的归组循环消费它）
def get_layers_from_vllm_config(  # SOURCE: vllm/config/vllm.py:L2454-L2475
    vllm_config,
    layer_type: type,
    layer_names=None,
) -> dict:
    """
    Get layers from the vLLM config.

    Args:
        vllm_config: The vllm config.
        layer_type: The type of the layer to get.
        layer_names: The names of the layers to get. If None, return all layers.
    """

    forward_context = vllm_config.compilation_config.static_forward_context
    if layer_names is None:
        layer_names = forward_context.keys()

    return {
        layer_name: layer
        for layer_name in layer_names
        if isinstance(layer := forward_context.get(layer_name), layer_type)
    }


# SOURCE: vllm/config/vllm.py set_current_vllm_config —— HOST SEAM 同型
@contextmanager
def set_current_vllm_config(vllm_config):  # SOURCE: vllm/config/vllm.py
    global _current_vllm_config
    prev = _current_vllm_config
    _current_vllm_config = vllm_config
    try:
        yield vllm_config
    finally:
        _current_vllm_config = prev


# SOURCE: vllm/config/utils.py:L56-L75 config 装饰器 ——（逐字机制）pydantic
#   dataclass 包装（extra="forbid" 默认）；AttentionConfig 全部 vllm config
#   类都经它成型。
def config(cls=None, *, config: ConfigDict | None = None, **kwargs: Any):
    # SOURCE: vllm/config/utils.py:L63-L64 merged_config
    merged_config = ConfigDict(extra="forbid")
    if config is not None:
        merged_config.update(config)

    # SOURCE: vllm/config/utils.py:L66-L67 decorator
    def decorator(cls):
        return _pydantic_dataclass(cls, config=merged_config, **kwargs)

    if cls is None:
        return decorator
    return decorator(cls)


# ── E. 协议载体与杂项 host 替身 ────────────────────────────────────────────

# SOURCE: vllm/v1/worker/gpu_input_batch.py InputBatch 的块表线字段面 ——
#   HOST SEAM（ch18/ch22 域全文）：_get_block_table/_get_slot_mappings 消费
#   block_tables[gid]（真身是 BlockTable.get_device_tensor 的持有者；host 以
#   裸张量承载 get_device_tensor 恒等返回）。
# SOURCE: vllm/v1/utils.py:L110-L149 CpuGpuBuffer —— ENGINE SEAM 装配位：
#   本章只消费 .gpu/.cpu 两切片面；完整双镜像协议 → ch22 全文。
class _CpuGpuSeam:
    def __init__(self, cpu: torch.Tensor):  # SOURCE: vllm/v1/utils.py CpuGpuBuffer.__init__（ENGINE SEAM）
        self.cpu = cpu
        self.gpu = cpu.clone()

    def copy_to_gpu(self, n: int):  # SOURCE: vllm/v1/utils.py CpuGpuBuffer.copy_to_gpu（ENGINE SEAM）
        self.gpu[:n].copy_(self.cpu[:n])


class _BlockTableSeam:
    # SOURCE: vllm/v1/worker/block_table.py BlockTable.get_device_tensor ——
    #   HOST SEAM：真实对象返回其设备张量切片；host 直接持张量。slot_mapping
    #   面同理（真实由 compute_slot_mapping 算出——ch22 域全文；测试注入）。
    def __init__(self, tensor: torch.Tensor, slot_mapping: torch.Tensor | None = None):  # HOST SEAM 装配位  # SOURCE: vllm/v1/worker/block_table.py
        self._tensor = tensor
        self.slot_mapping = _CpuGpuSeam(
            slot_mapping if slot_mapping is not None else torch.zeros(0, dtype=torch.int64)
        )

    def get_device_tensor(self, num_reqs: int) -> torch.Tensor:  # SOURCE: vllm/v1/worker/block_table.py
        return self._tensor[:num_reqs]


class InputBatchSeam:
    """HOST SEAM：InputBatch 的块表线字段面（ch18/ch22 域全文）。"""

    # SOURCE: vllm/v1/worker/gpu_input_batch.py InputBatch.__init__ —— HOST
    #   SEAM 装配位：block_tables（gid→BlockTable 面）+ 列式 CPU 镜像。
    def __init__(self, max_num_reqs: int):  # SOURCE: vllm/v1/worker/gpu_input_batch.py
        self.max_num_reqs = max_num_reqs
        self.num_reqs = 0
        self.block_tables: dict[int, Any] = {}
        self.num_computed_tokens_cpu_tensor = torch.zeros(max_num_reqs, dtype=torch.int32)
        self.num_prompt_tokens_cpu_tensor = torch.full(
            (max_num_reqs,), 10_000, dtype=torch.int32
        )


# ── F. 量化域类型面（ch27）——占位载体：本章配置恒 Unquantized/None，分支
#   不触发；名字面与调用位逐字保留 ────────────────────────────────────────

# SOURCE: vllm/model_executor/layers/linear.py UnquantizedLinearMethod ——
#   HOST SEAM 类型面（should_load_quant_weights 的 isinstance 判别）
class UnquantizedLinearMethod:  # HOST SEAM  # SOURCE: vllm/model_executor/layers/linear.py
    pass


# SOURCE: vllm/model_executor/layers/quantization/base_config.py QuantizeMethodBase
#   —— HOST SEAM 类型面
class QuantizeMethodBase:  # HOST SEAM  # SOURCE: vllm/model_executor/layers/quantization/base_config.py
    pass


# SOURCE: vllm/model_executor/layers/quantization/__init__.py QuantizationConfig
#   —— HOST SEAM 类型面
class QuantizationConfig:  # HOST SEAM  # SOURCE: vllm/model_executor/layers/quantization/__init__.py
    pass


# SOURCE: vllm/model_executor/layers/quantization/kv_cache.py BaseKVCacheMethod
#   —— HOST SEAM 类型面（_init_kv_cache_quant 的 assert isinstance 位）
class BaseKVCacheMethod(QuantizeMethodBase):  # HOST SEAM  # SOURCE: vllm/model_executor/layers/quantization/kv_cache.py
    pass


# SOURCE: vllm/v1/attention/backends/fa_utils.py:L350-L378 is_flash_attn_
#   varlen_func_available —— HOST SEAM：真身在 CUDA/XPU 恒 True；host 的两个
#   CUDA op 由包内 HOST SEAM 镜像承载（flash_attn.py 尾部），故此处恒 True
#   保住真实 import 面（`if is_flash_attn_varlen_func_available(): from …`）。
def is_flash_attn_varlen_func_available() -> bool:  # SOURCE: vllm/v1/attention/backends/fa_utils.py:L350-L378
    return True


# SOURCE: vllm/v1/worker/block_table.py:L20-L40 get_block_table_width ——（逐字）
#   ch22 域全文；本章 create_metadata_builders 的 requires_block_table_width
#   分支消费它（FA builder 恒 False 不触发，守卫位保留）。
def get_block_table_width(  # SOURCE: vllm/v1/worker/block_table.py:L20-L40
    max_num_blocks: int,
    block_size: int,
    kernel_block_size: int | None = None,
    *,
    token_alignment: int | None = 128,
) -> int:
    """Return the width after optional alignment and virtual block splitting."""
    import math

    from .utils.math_utils import cdiv

    if kernel_block_size is None:
        kernel_block_size = block_size
    if block_size % kernel_block_size != 0:
        raise ValueError(
            f"kernel_block_size {kernel_block_size} must divide "
            f"block_size {block_size} evenly"
        )
    if token_alignment is not None:
        if token_alignment <= 0:
            raise ValueError("token_alignment must be positive")
        block_alignment = token_alignment // math.gcd(token_alignment, block_size)
        max_num_blocks = cdiv(max_num_blocks, block_alignment) * block_alignment
    return max_num_blocks * block_size // kernel_block_size


# SOURCE: vllm/utils/import_utils.py:L104 resolve_obj_by_qualname —— 真身在
#   utils/import_utils.py（本包逐字收录）；此处 re-export 保住 fa_utils 面。
from .utils.import_utils import resolve_obj_by_qualname  # noqa: F401,E402
