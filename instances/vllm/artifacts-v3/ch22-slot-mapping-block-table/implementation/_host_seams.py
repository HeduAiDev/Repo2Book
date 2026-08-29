# HOST SEAM 登记处（ch22）——真实代码之外唯一允许的承载面
# 每一项都在行内标注真实源锚；本章在 CPU host（无 CUDA、无 vllm 包）跑通
# 全链测试。分四类：
#   A. 分布式组探测 / logger / 观测：vllm 顶层设施的 host 替身；
#   B. 配置面：ch17/ch19 域的 config namespace 占位（精简配置的真实取值）；
#   C. 协议载体：SchedulerOutput/InputBatch 的块表线字段面（delete[14] 明示
#      「InputBatch 直接复用 ch18 精简版产物」——本章以同接口的最小承载）；
#   D. 测试装配位：make_kv_cache_config / make_attn_group。
from __future__ import annotations

import warnings
from contextlib import nullcontext
from enum import Enum
from typing import Any

import numpy as np
import torch

from .block_table import MultiGroupBlockTable
from .kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
)
from .torch_utils import PIN_MEMORY


# ── A. vllm 顶层设施的 host 替身 ───────────────────────────────────────────

# SOURCE: vllm/logger.py init_logger（HOST SEAM：no-op logger——本章文件内无
#   日志消费，仅保住调用位）
# SOURCE: vllm/logger.py —— HOST SEAM：no-op logger 载体
class _NullLogger:
    # SOURCE: vllm/logger.py —— HOST SEAM no-op 位
    def info_once(self, *a, **k):
        pass

    # SOURCE: vllm/logger.py —— HOST SEAM no-op 位
    def debug(self, *a, **k):
        pass


# SOURCE: vllm/logger.py init_logger —— HOST SEAM
def init_logger(name):
    return _NullLogger()


# SOURCE: vllm/distributed/parallel_state.py get_pcp_group/get_dcp_group 的
#   HOST SEAM 转出（单置叶模块 _dist_seams.py——避免与 block_table 循环
#   导入；此处 re-export 保住消费面）。
from ._dist_seams import get_dcp_group, get_pcp_group  # noqa: F401


# SOURCE: vllm/utils/torch_utils.py record_function_or_nullcontext 的 host
#   替身（无 profiler——nullcontext 直通）
# SOURCE: vllm/utils/torch_utils.py record_function_or_nullcontext —— HOST SEAM
def record_function_or_nullcontext(name: str):
    return nullcontext()


# SOURCE: vllm/utils/deprecation.py deprecated(msg) 的 host 镜像：包一层
#   DeprecationWarning（措辞由调用方原文传入——CommonAttentionMetadata 两个
#   deprecated 属性即用此承载，warnings 原样可见）
# SOURCE: vllm/utils/deprecation.py deprecated —— HOST SEAM 镜像
def deprecated(msg: str):
    # SOURCE: vllm/utils/deprecation.py deprecated 内层 deco —— HOST SEAM 镜像
    def deco(fn):
        # SOURCE: vllm/utils/deprecation.py deprecated 内层 wrapper —— HOST SEAM 镜像
        def wrapper(*args, **kwargs):
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return fn(*args, **kwargs)

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = (fn.__doc__ or "") + f"\nDeprecated: {msg}"
        return wrapper

    return deco


# SOURCE: vllm/config/compilation.py:L53-L104 CUDAGraphMode ——（逐字）ch19
#   编译/捕获域的枚举；本章 execute_model 的 pad_attn 判据（== FULL）用它。
# SOURCE: vllm/config/compilation.py:L53 CUDAGraphMode（逐字——ch19 域枚举）
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


# SOURCE: vllm/v1/worker/ubatch_utils.py UBatchSlices —— 类型名占位
#   （delete[5]：ubatching/DBO 不进本章，字段恒 None）
UBatchSlices = Any

# SOURCE: vllm/v1/attention/backends/fa_utils.py get_flash_attn_version 的
#   HOST SEAM 装配位：host 无 FA3/4 库，恒 FA2。
# SOURCE: vllm/v1/attention/backends/fa_utils.py get_flash_attn_version —— HOST SEAM 装配位
def get_flash_attn_version(
    requires_alibi: bool = False,
    requires_local_attention: bool = False,
    head_size: int = 0,
    has_sinks: bool = False,
) -> int:
    return 2


# SOURCE: vllm/v1/kv_cache_interface.py:L85-L86 is_quantized_kv_cache 的
#   HOST SEAM 装配位（等价实现：非 auto/float16/bfloat16 即量化——与
#   KVQuantMode.NONE 的判别在这些 dtype 上逐值一致；量化全域 → ch14/ch21）。
# SOURCE: vllm/v1/kv_cache_interface.py:L85 is_quantized_kv_cache —— HOST SEAM 等价装配
def is_quantized_kv_cache(kv_cache_dtype: str) -> bool:
    return kv_cache_dtype not in ("auto", "float16", "bfloat16")


# SOURCE: vllm/v1/outputs.py EMPTY_MODEL_RUNNER_OUTPUT 的 host 占位
#   （空批早退的返回值——真身是空 ModelRunnerOutput 单例，ch12/ch18 域）。
EMPTY_MODEL_RUNNER_OUTPUT = None

# 注解面：ForwardContext 类型名（真身在本包 forward_context.py）。
ForwardContextLike = Any


# ── B. 配置面占位（ch17/ch19 域 namespace；取值=精简配置的真实值） ─────────

# SOURCE: vllm/config/model.py ModelConfig 消费字段面 —— HOST SEAM 装配位
class _ModelConfigSeam:
    # SOURCE: vllm/config/model.py ModelConfig.__init__ —— HOST SEAM 装配位
    def __init__(self, max_model_len: int):
        self.max_model_len = max_model_len
        self.is_encoder_decoder = False
        self.rswa_window = None  # R-SWA 域（ch21）——默认关
        self.hf_text_config = None  # mm_prefix 域守卫内才读

    # SOURCE: vllm/config/model.py ModelConfig.get_vocab_size —— HOST SEAM 装配位
    def get_vocab_size(self):
        # HOST SEAM 装配位（InputBatch 构造的 vocab_size 实参——ch18 域）
        return 0


# SOURCE: vllm/config/cache.py CacheConfig 消费字段面 —— HOST SEAM 装配位
class _CacheConfigSeam:
    # SOURCE: vllm/config/cache.py CacheConfig.__init__ —— HOST SEAM 装配位
    def __init__(self):
        self.kv_sharing_fast_prefill = False  # delete[9]
        self.use_replayssm = False  # replayssm 域（delete[14]）
        self.mamba_cache_mode = "none"  # delete[3]
        self.cache_dtype = "auto"


# SOURCE: vllm/config/parallel.py ParallelConfig 消费字段面 —— HOST SEAM 装配位
class _ParallelConfigSeam:
    # SOURCE: vllm/config/parallel.py ParallelConfig.__init__ —— HOST SEAM 装配位
    def __init__(self):
        self.cp_kv_cache_interleave_size = 1
        self.data_parallel_size = 1
        self.use_ubatching = False  # delete[5]


# SOURCE: vllm/config/scheduler.py SchedulerConfig 消费字段面 —— HOST SEAM 装配位
class _SchedulerConfigSeam:
    # SOURCE: vllm/config/scheduler.py SchedulerConfig.__init__ —— HOST SEAM 装配位
    def __init__(self, max_num_seqs: int, max_num_batched_tokens: int):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens


# SOURCE: vllm/config/compilation.py CompilationConfig.static_forward_context —— HOST SEAM 装配位
class _CompilationConfigSeam:
    # SOURCE: vllm/config/compilation.py CompilationConfig.__init__ —— HOST SEAM 装配位
    def __init__(self):
        # static_forward_context: layer_name → Attention 层实例（真身由
        # torch.compile 包装器装配，ch19 域；测试在此注册层对象）
        self.static_forward_context: dict[str, Any] = {}
        self.cudagraph_mode = CUDAGraphMode.NONE


# SOURCE: vllm/config/vllm.py VllmConfig 的块表线消费子集 —— HOST SEAM 装配位
class VllmConfigSeam:
    """HOST SEAM 配置面：vllm_config 的块表线消费字段子集。"""

    # SOURCE: vllm/config/vllm.py VllmConfig.__init__ —— HOST SEAM 装配位
    def __init__(self, num_seqs: int, max_batched_tokens: int, max_model_len: int):
        self.model_config = _ModelConfigSeam(max_model_len)
        self.cache_config = _CacheConfigSeam()
        self.parallel_config = _ParallelConfigSeam()
        self.scheduler_config = _SchedulerConfigSeam(num_seqs, max_batched_tokens)
        self.compilation_config = _CompilationConfigSeam()
        self.speculative_config = None  # spec 域（delete[2]/ch12/ch33）
        self.lora_config = None  # LoRA 域（ch33）
        self.reasoning_config = None  # delete[14]：thinking 域不进


# ── C. 协议载体（SchedulerOutput / InputBatch 的块表线字段面） ─────────────

# SOURCE: vllm/v1/core/sched/output.py:L116 CachedRequestData 字段面 —— HOST SEAM（ch18 域全文）
class _CachedReqDataSeam:
    """HOST SEAM：SchedulerOutput.scheduled_cached_reqs 的字段面
    （真身 vllm/v1/core/sched/output.py:CachedRequestData，ch18 域全文）。"""

    # SOURCE: vllm/v1/core/sched/output.py:L116 CachedRequestData 字段装填 —— HOST SEAM
    def __init__(self, req_ids, new_block_ids, num_computed_tokens,
                 num_output_tokens, resumed_req_ids=()):
        self.req_ids = list(req_ids)
        self.new_block_ids = new_block_ids
        self.num_computed_tokens = list(num_computed_tokens)
        self.num_output_tokens = list(num_output_tokens)
        self.resumed_req_ids = list(resumed_req_ids)
        self.new_token_ids = [[] for _ in req_ids]
        self.all_token_ids: dict[str, list[int]] = {}


# SOURCE: vllm/v1/core/sched/output.py:L193 SchedulerOutput 字段面 —— HOST SEAM（ch18 域全文）
class SchedulerOutputSeam:
    """HOST SEAM：SchedulerOutput 的块表线消费字段
    （真身 vllm/v1/core/sched/output.py:SchedulerOutput，ch18 域全文）。"""

    # SOURCE: vllm/v1/core/sched/output.py:L193 SchedulerOutput 字段装填 —— HOST SEAM
    def __init__(self, total_num_scheduled_tokens, num_scheduled_tokens=None,
                 req_ids=(), new_block_ids=None, num_computed_tokens=None,
                 num_output_tokens=None, resumed_req_ids=()):
        self.total_num_scheduled_tokens = total_num_scheduled_tokens
        self.num_scheduled_tokens = dict(num_scheduled_tokens or {})
        self.scheduled_cached_reqs = _CachedReqDataSeam(
            req_ids, new_block_ids or [], num_computed_tokens or [],
            num_output_tokens or [], resumed_req_ids)
        self.scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        self.scheduled_encoder_inputs: list = []
        self.finished_req_ids: list[str] = []
        self.scheduled_new_reqs: list = []
        self.kv_connector_metadata = None  # delete[9]
        self.new_block_ids_to_zero: list = []  # ch13 域
        self.kv_cache_block_copies: list = []


class InputBatchSeam:
    """HOST SEAM：InputBatch 的块表线字段面 + 构造面。

    delete[14] 明示：InputBatch 的采样参数列/thinking budget/replayssm/
    reasoning 等与块表无关的装填不进本章，「InputBatch 直接复用 ch18 精简版
    产物（append_row/clear_row/move_row 接口不变）」——本载体即那一接口面：
    block_table（MultiGroupBlockTable）+ num_reqs/req_ids/req_id_to_index +
    num_computed_tokens_cpu(_tensor)/num_prompt_tokens_cpu(_tensor)。
    构造签名对齐真实 InputBatch（vllm/v1/worker/gpu_input_batch.py:L~150 起）
    的块表相关子集；块表构造逐字镜像 gpu_input_batch.py:L186-L195。
    """

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        device: torch.device,
        block_sizes: list[int],
        kernel_block_sizes: list[int],
        max_num_blocks_per_req: list[int],
        num_spec_tokens: int = 0,
        vocab_size: int = 0,
        logitsprocs: Any = None,
        logitsprocs_need_output_token_ids: bool = False,
        is_pooling_model: bool = False,
        cp_kv_cache_interleave_size: int = 1,
        reasoning_config: Any = None,
        use_replayssm: bool = False,
        slot_mapping_modes: Any = None,
    ):
        # SUBTRACTED（delete[14]）：采样参数列/thinking budget/replayssm/
        #   reasoning 等装填——ch18 域全文已立，接口不变。
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.num_reqs = 0
        self.req_ids: list[str] = []
        self.req_id_to_index: dict[str, int] = {}
        # SOURCE: vllm/v1/worker/gpu_input_batch.py:L186-L195（块表构造逐字
        #   ——PIN_MEMORY 例外：host 无 pinned，恒 False）
        self.block_table = MultiGroupBlockTable(
            max_num_reqs=max_num_reqs,
            max_num_batched_tokens=max_num_batched_tokens,
            pin_memory=PIN_MEMORY,
            device=device,
            block_sizes=block_sizes,
            kernel_block_sizes=kernel_block_sizes,
            max_num_blocks=max_num_blocks_per_req,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            slot_mapping_modes=slot_mapping_modes,
        )
        # 列式 CPU 镜像（ch18 域的字段面；本章 _update_states/_prepare_
        # inputs/_build_attention_metadata 消费其中四个）。真实 InputBatch 的
        # np 视图与 torch 张量共享同一存储（np = tensor.numpy()）——同构。
        self.num_computed_tokens_cpu_tensor = torch.zeros(max_num_reqs, dtype=torch.int32)
        self.num_computed_tokens_cpu = self.num_computed_tokens_cpu_tensor.numpy()
        self.num_prompt_tokens_cpu_tensor = torch.zeros(max_num_reqs, dtype=torch.int32)
        self.num_prompt_tokens_cpu = self.num_prompt_tokens_cpu_tensor.numpy()
        # 透传位（may_reinitialize 的构造实参读它们——真实 InputBatch 字段）
        self.logitsprocs = logitsprocs
        self.logitsprocs_need_output_token_ids = logitsprocs_need_output_token_ids


# ── D. 测试装配位 ──────────────────────────────────────────────────────────

# SOURCE: vllm/v1/kv_cache_interface.py:L973 KVCacheConfig —— 测试装配位（真实载体类）
def make_kv_cache_config(groups: list[tuple[str, int, str]]):
    """(layer_name, block_size, kind) 列表 → KVCacheConfig（真实载体类）。"""
    from .kv_cache_interface import KVCacheConfig

    group_specs = []
    for layer_name, block_size, kind in groups:
        if kind == "full":
            spec = FullAttentionSpec(block_size=block_size)
        elif kind == "mamba":
            spec = MambaSpec(block_size=block_size,
                             shapes=((8, 2),), dtypes=(torch.float32,))
        else:
            raise ValueError(kind)
        group_specs.append(KVCacheGroupSpec([layer_name], spec))
    return KVCacheConfig(num_blocks=64, kv_cache_tensors=[],
                         kv_cache_groups=group_specs)


# SOURCE: vllm/v1/worker/utils.py:L217 AttentionGroup —— 测试装配位（真实载体类）
def make_attn_group(backend_cls, layer_names, gid):
    from .worker_utils import AttentionGroup

    return AttentionGroup(backend=backend_cls, layer_names=list(layer_names),
                          kv_cache_spec=None, kv_cache_group_id=gid)
