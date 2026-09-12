# SOURCE: vllm/config/__init__.py（真实为 config/ 包各类的 re-export 门面）
# HOST SEAM：配置面的最小承载。真实 VllmConfig 是 ch03 域的大配置对象——
# 本章只消费其字段面（model/cache/scheduler/parallel/compilation/attention/
# speculative/kv_transfer），以同名字段载体镜像；use_mla 属性为真实属性
#（config/model.py:L1791-L1792）逐字；set/get_current_vllm_config 是真实
# 函数的减法子集（语义逐字，ch23 同款骨架）。
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

import vllm.envs as envs


# SOURCE: vllm/config/cache.py:L44 CacheConfig —— HOST SEAM 字段面
#   （本章消费：cache_dtype/block_size/calculate_kv_scales/
#   kv_cache_dtype_skip_layers/enable_prefix_caching/user_specified_block_size）
@dataclass
class CacheConfig:
    # SOURCE: vllm/config/cache.py cache_dtype —— 默认 "auto"
    cache_dtype: str = "auto"
    # SOURCE: vllm/config/cache.py block_size —— HOST SEAM 默认位
    block_size: int = 128
    # SOURCE: vllm/config/cache.py calculate_kv_scales —— 默认 False
    calculate_kv_scales: bool = False
    # SOURCE: vllm/config/cache.py enable_prefix_caching —— 默认 None
    enable_prefix_caching: Optional[bool] = None
    # SOURCE: vllm/config/cache.py user_specified_block_size —— 默认 False
    user_specified_block_size: bool = False
    # SOURCE: vllm/config/cache.py kv_cache_dtype_skip_layers —— 默认 None
    kv_cache_dtype_skip_layers: Optional[set[int]] = None


# SOURCE: vllm/config/scheduler.py SchedulerConfig —— HOST SEAM 字段面
#   （本章消费：max_num_seqs（workspace 定容）/max_num_batched_tokens（DSV4
#   装配）/disable_hybrid_kv_cache_manager（unify_hybrid 的开关位））
@dataclass
class SchedulerConfig:
    # SOURCE: vllm/config/scheduler.py max_num_seqs —— HOST SEAM 默认 256
    max_num_seqs: int = 256
    # SOURCE: vllm/config/scheduler.py max_num_batched_tokens —— HOST SEAM
    #   默认 8192
    max_num_batched_tokens: int = 8192
    # SOURCE: vllm/config/scheduler.py disable_hybrid_kv_cache_manager
    #   —— HOST SEAM 默认 False（--disable-hybrid-kv-cache-manager 位）
    disable_hybrid_kv_cache_manager: bool = False


# SOURCE: vllm/config/compilation.py:L753 static_forward_context —— 逐字字段
#   （本章消费：static_forward_context——插座自注册账本 + ForwardContext 快照）
@dataclass
class CompilationConfig:
    # SOURCE: vllm/config/compilation.py:L753 static_forward_context —— 逐字字段
    static_forward_context: dict[str, Any] = field(default_factory=dict, init=False)
    # SOURCE: vllm/config/compilation.py cudagraph_mode —— HOST SEAM 位：
    #   has_full_cudagraphs() 恒 False（ch19 cudagraph 域不进 host；
    #   FlashMLAMetadataBuilder 的 CG 缓冲分支因此不触发——真实默认 NONE 同型）
    cudagraph_mode: Any = None

    def has_full_cudagraphs(self):
        # SOURCE: vllm/config/compilation.py CUDAGraphMode.has_full_cudagraphs
        #   —— HOST SEAM：NONE 档恒 False
        return False


# SOURCE: vllm/config/parallel.py ParallelConfig —— HOST SEAM 字段面
#   （本章消费：tensor_parallel_size / decode_context_parallel_size /
#   prefill_context_parallel_size / cp_kv_cache_interleave_size——
#   use_pcp/dcp 的开关面（delete[0] 已删其 >1 分支，保默认 1 的字段位））
@dataclass
class ParallelConfig:
    # SOURCE: vllm/config/parallel.py tensor_parallel_size —— HOST SEAM 默认 1
    tensor_parallel_size: int = 1
    # SOURCE: vllm/config/parallel.py pipeline_parallel_size —— HOST SEAM 默认 1
    pipeline_parallel_size: int = 1
    # SOURCE: vllm/config/parallel.py prefill_context_parallel_size —— 默认 1
    prefill_context_parallel_size: int = 1
    # SOURCE: vllm/config/parallel.py decode_context_parallel_size —— 默认 1
    decode_context_parallel_size: int = 1
    # SOURCE: vllm/config/parallel.py cp_kv_cache_interleave_size —— 默认 16
    cp_kv_cache_interleave_size: int = 16


# SOURCE: vllm/config/attention.py AttentionConfig —— HOST SEAM 字段面
#   （本章消费：backend（decode 家族的显式指定轴——selector 消费）/
#   mla_prefill_backend（prefill 家族的显式指定轴——get_mla_prefill_backend
#   消费）/ use_prefill_query_quantization（FP8 prefill 查询量化开关，默认
#   False——determine_prefill_query_data_type 的 q_data_type 落 model dtype））
@dataclass
class AttentionConfig:
    # SOURCE: vllm/config/attention.py backend —— HOST SEAM 默认 None
    backend: Optional[str] = None
    # SOURCE: vllm/config/attention.py mla_prefill_backend —— HOST SEAM 默认
    #   None（显式指定时为 MLAPrefillBackendEnum 成员）
    mla_prefill_backend: Any = None
    # SOURCE: vllm/config/attention.py use_prefill_query_quantization —— 默认
    #   False
    use_prefill_query_quantization: bool = False


# SOURCE: vllm/config/model.py:L122 ModelConfig —— HOST SEAM 字段面 + 真实属性
@dataclass
class ModelConfig:
    # SOURCE: vllm/config/model.py hf_config 字段 —— HOST SEAM
    hf_config: Any = None
    # SOURCE: vllm/config/model.py dtype 字段 —— HOST SEAM
    dtype: torch.dtype = torch.float32
    # SOURCE: vllm/config/model.py max_model_len 字段 —— HOST SEAM（workspace
    #   定容公式输入）
    max_model_len: int = 8192

    @property
    def hf_text_config(self) -> Any:
        # SOURCE: vllm/config/model.py hf_text_config 属性 —— HOST SEAM 镜像
        #   （真实区分 text_config 多模态嵌套；本章测试的 hf config 即文本面）
        return self.hf_config

    @property
    def is_deepseek_mla(self) -> bool:
        # SOURCE: vllm/config/model.py:L1401-L1402 is_deepseek_mla —— 逐字
        #   （真实经 model_arch_config.is_deepseek_mla——DeepseekV2/V3/V4 架构
        #   判定；HOST SEAM 以 model_type 前缀承载同一判定面）
        return str(getattr(self.hf_text_config, "model_type", "")).startswith(
            ("deepseek_v2", "deepseek_v3", "deepseek_v4")
        )

    @property
    def use_mla(self) -> bool:
        # SOURCE: vllm/config/model.py:L1791-L1792 use_mla —— 逐字
        #   （MLA 不是运行时开关，是装配期就定死的层类型——站 1 的判定键）
        return self.is_deepseek_mla and not envs.VLLM_MLA_DISABLE

    def get_head_size(self) -> int:
        # SOURCE: vllm/config/model.py:L1412-L1413 get_head_size —— HOST SEAM
        #   镜像（真实走 model_arch_config.head_size；MLA 架构下 =
        #   kv_lora_rank + qk_rope_head_dim，DSV4 统一记法 = head_dim+rope）
        hf = self.hf_text_config
        if hasattr(hf, "compress_ratios"):  # DSV4 统一记法
            return hf.head_dim + hf.qk_rope_head_dim
        if getattr(hf, "kv_lora_rank", 0):
            return hf.kv_lora_rank + hf.qk_rope_head_dim
        return getattr(hf, "head_dim", 64)

    def get_num_attention_heads(self, parallel_config) -> int:
        # SOURCE: vllm/config/model.py:L1432-L1435 get_num_attention_heads
        #   —— 逐字语义（total // TP）
        return self.hf_text_config.num_attention_heads // max(
            1, parallel_config.tensor_parallel_size
        )


# SOURCE: vllm/config/vllm.py:L331 VllmConfig —— HOST SEAM 字段面载体
@dataclass
class VllmConfig:
    # SOURCE: vllm/config/vllm.py model_config 字段 —— HOST SEAM
    model_config: ModelConfig = field(default_factory=ModelConfig)
    # SOURCE: vllm/config/vllm.py cache_config 字段 —— HOST SEAM
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    # SOURCE: vllm/config/vllm.py scheduler_config 字段 —— HOST SEAM
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    # SOURCE: vllm/config/vllm.py compilation_config 字段 —— HOST SEAM
    compilation_config: CompilationConfig = field(default_factory=CompilationConfig)
    # SOURCE: vllm/config/vllm.py parallel_config 字段 —— HOST SEAM
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    # SOURCE: vllm/config/vllm.py attention_config 字段 —— HOST SEAM
    attention_config: AttentionConfig = field(default_factory=AttentionConfig)
    # SOURCE: vllm/config/vllm.py speculative_config 字段 —— HOST SEAM 默认
    #   None（spec-decode 归 ch33）
    speculative_config: Any = None
    # SOURCE: vllm/config/vllm.py kv_transfer_config 字段 —— HOST SEAM 默认
    #   None（KV connector 归 ch16；delete[8] 已删 maybe_transfer 钩子）
    kv_transfer_config: Any = None
    # SOURCE: vllm/config/vllm.py quant_config 字段 —— HOST SEAM 默认 None
    quant_config: Any = None


# SOURCE: vllm/config/vllm.py:L2374-L2408 set_current_vllm_config —— 减法子集
#   （ch23 同款：全局暂存/恢复主干逐字；编译计数与告警段删除——ch19 域）
_current_vllm_config: VllmConfig | None = None
_current_prefix: str | None = None


# SOURCE: vllm/config/vllm.py:L2374 set_current_vllm_config
@contextmanager
def set_current_vllm_config(
    vllm_config: VllmConfig, check_compile=False, prefix: str | None = None
):
    """
    Temporarily set the current vLLM config.
    Used during model initialization.
    We save the current vLLM config in a global variable,
    so that all modules can access it, e.g. custom ops
    can access the vLLM config to determine how to dispatch.
    """
    # SOURCE: vllm/config/vllm.py:L2374 set_current_vllm_config
    global _current_vllm_config, _current_prefix
    old_vllm_config = _current_vllm_config
    old_prefix = _current_prefix
    try:
        # SUBTRACTED: compilation_counter 计数与告警段（vllm/config/vllm.py:
        #   L2387-L2394、L2401-L2415）——ch19 编译域
        _current_vllm_config = vllm_config
        _current_prefix = prefix
        yield
    finally:
        _current_vllm_config = old_vllm_config
        _current_prefix = old_prefix


# SOURCE: vllm/config/vllm.py:L2434-L2447 get_current_vllm_config —— 逐字
def get_current_vllm_config() -> VllmConfig:
    if _current_vllm_config is None:
        raise AssertionError(
            "Current vLLM config is not set. This typically means "
            "get_current_vllm_config() was called outside of a "
            "set_current_vllm_config() context, or a CustomOp was instantiated "
            "at module import time or model forward time when config is not set. "
        )
    return _current_vllm_config


# SOURCE: vllm/config/vllm.py get_current_vllm_config_or_none —— HOST SEAM 位
def get_current_vllm_config_or_none() -> Optional[VllmConfig]:
    # SOURCE: vllm/config/vllm.py get_current_vllm_config_or_none —— HOST SEAM
    return _current_vllm_config


# SOURCE: vllm/config/vllm.py:L2454-L2475 get_layers_from_vllm_config —— 逐字
def get_layers_from_vllm_config(
    vllm_config: VllmConfig,
    layer_type: type,
    layer_names=None,
) -> dict:
    """
    Get layers from the vLLM config.

    Args:
        vllm_config: The vLLM config.
        layer_type: The type of the layer to get.
        layer_names: The names of the layers to get. If None, return all layers.
    """

    forward_context = vllm_config.compilation_config.static_forward_context
    if layer_names is None:
        layer_names = list(forward_context.keys())

    return {
        layer_name: layer
        for layer_name in layer_names
        if isinstance(layer := forward_context.get(layer_name), layer_type)
    }
