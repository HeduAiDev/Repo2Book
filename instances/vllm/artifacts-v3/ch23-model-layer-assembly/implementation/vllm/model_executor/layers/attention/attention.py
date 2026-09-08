# SOURCE: vllm/model_executor/layers/attention/attention.py
# ch23 主角文件之五（m3/m11）：Attention 插座——统一注意力封装（模型层与
# kernel 层之间）。__init__ 尾部自注册 static_forward_context（重复 layer_name
# 即 raise——模型层自己把自己登记进编译账本）；forward 签名不收
# attn_metadata/kv_cache/slot_mapping，全部经 get_attention_context 从
# ForwardContext 按 layer_name 取（unified_kv_cache_update 写 KV 并返回
# dummy 张量作 torch.compile 数据依赖、unified_attention_with_output 读算）。
# SUBTRACTED：后端自动选择/优先级表/validate 回退（L262-L435 的选择面）→
#   ch21 已立（本章插座面要求显式 attn_backend 注入——真实参数位 L251 的
#   else 支 L362-L363）；KV cache 量化初始化与 query 量化（L466-L486）→
#   ch27；get_kv_cache_spec（L621-L694）→ ch22；torch.library 算子注册面 →
#   ch19；kv_sharing 跨层共享校验（L449-L454）→ ch21/ch22 域。
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from vllm.config import CacheConfig, get_current_vllm_config
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    AttentionType,
)

logger = init_logger(__name__)

# SUBTRACTED: AttentionBackendEnum（v1/attention/backends/registry）——后端
#   注册表枚举归 ch21；self.backend 标记位随之删除


# SOURCE: vllm/model_executor/layers/attention/attention.py:L223 Attention
class Attention(nn.Module, AttentionLayerBase):
    """Attention layer.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    # SOURCE: vllm/model_executor/layers/attention/attention.py:L235-L254
    #   __init__ 签名（逐字——含真实注入位 attn_backend 参数 L251）
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        use_alibi_sqrt: bool | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        logits_soft_cap: float | None = None,
        per_layer_sliding_window: int | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        mm_prefix_clamp_sliding_window: bool = False,
        attn_backend: type[AttentionBackend] | None = None,
        head_size_v: int | None = None,
        **extra_impl_args,
    ) -> None:
        """
        The KV cache is stored inside this class and is accessed via
        `self.kv_cache`.
        """
        super().__init__()
        sliding_window: int | None
        # SOURCE: vllm/model_executor/layers/attention/attention.py:L261-L268
        #   sliding_window 三态解析（逐字——逐层 > 模型级 > None）
        if per_layer_sliding_window is not None:
            # per-layer sliding window
            sliding_window = per_layer_sliding_window
        elif cache_config is not None:
            # model-level sliding window
            sliding_window = cache_config.sliding_window
        else:
            sliding_window = None

        # SOURCE: vllm/model_executor/layers/attention/attention.py:L270-L276
        #   vllm_config/cache_config 的 dtype 与 scales 位（逐字）
        vllm_config = get_current_vllm_config()
        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            calculate_kv_scales = cache_config.calculate_kv_scales
        else:
            kv_cache_dtype = "auto"
            calculate_kv_scales = False

        # SUBTRACTED: kv_cache_scheme FP8 方案位与 per-head scale 判整
        #   （attention.py:L278-L296）——ch27；kv_cache_dtype_skip_layers
        #   （L298-L321）——量化 KV 跳层面

        # SOURCE: vllm/model_executor/layers/attention/attention.py:L323-L341
        #   属性面（逐字——kv_cache_torch_dtype/calculate_kv_scales/
        #   layer_name/头数三件/sliding_window/has_sink）
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            kv_cache_dtype, vllm_config.model_config
        )
        self.kv_cache_dtype = kv_cache_dtype
        self.calculate_kv_scales = calculate_kv_scales
        if num_kv_heads is None:
            num_kv_heads = num_heads
        assert num_heads % num_kv_heads == 0, (
            f"num_heads ({num_heads}) is not divisible by num_kv_heads ({num_kv_heads})"
        )
        self.quant_config = quant_config
        self.layer_name = prefix

        self.num_heads = num_heads
        self.head_size = head_size
        self.head_size_v = self.head_size if head_size_v is None else head_size_v
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window
        self.has_sink = extra_impl_args.get("sinks") is not None

        # SUBTRACTED: use_mm_prefix 位（attention.py:L343-L345）——多模态域

        # During model initialization, the default dtype is set as the model
        # weight and activation dtype.
        # SOURCE: vllm/model_executor/layers/attention/attention.py:L347-L349（逐字）
        dtype = torch.get_default_dtype()
        # SUBTRACTED: attn_backend=None 时 get_attn_backend 自动选择
        #   （attention.py:L350-L361）——后端优先级表归 ch21；本章插座面走真实
        #   注入位（L362-L363 的 else 支），None 时延后绑定（placeholder-then-
        #   bind——同 kv_cache L460-L463 的真实模式）
        self.attn_backend = attn_backend

        # SUBTRACTED: use_alibi_sqrt 校验与注入（L364-L373）、prefix caching +
        #   batch invariance 告警（L374-L389）、chunk_lookback 断言（L391-L395）、
        #   FLEX_ATTENTION 块尺寸（L397-L418）——ch20/ch21 域

        # SOURCE: vllm/model_executor/layers/attention/attention.py:L420-L433
        #   impl 构造（逐字——后端交出 impl 类、以此签名实例化；None 时占位）
        if attn_backend is not None:
            impl_cls = self.attn_backend.get_impl_cls()
            self.impl = impl_cls(  # type: ignore[assignment]  # impl_cls always returns an AttentionImpl subclass
                num_heads,
                head_size,
                scale,
                num_kv_heads,
                alibi_slopes,
                sliding_window,
                kv_cache_dtype,
                logits_soft_cap,
                attn_type,
                kv_sharing_target_layer_name,
                **extra_impl_args,
            )
        else:
            self.impl = None  # 占位：真实代码此处经 ch21 的选择面必得 impl
        # SUBTRACTED: self.backend = AttentionBackendEnum[...] 标记位
        #   （attention.py:L434）——后端注册表面归 ch21
        self.dtype = dtype

        # For cuda-alike (CUDA and ROCM) and cpu platforms, we control how
        # torch.compile works by registering the attention as one giant
        # opaque custom op. For other platforms, we directly call them
        # and let torch.compile handle them.
        # SOURCE: vllm/model_executor/layers/attention/attention.py:L437-L441
        #   use_direct_call（逐字——host 平台基类默认 False → 直调分支）
        self.use_direct_call = not current_platform.opaque_attention_op()

        # SOURCE: vllm/model_executor/layers/attention/attention.py:L443-L446
        #   static_forward_context 自注册（逐字——m11 核心：重复 layer_name 即
        #   raise，模型层把自己登记进编译账本）
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.attn_type = attn_type

        # SUBTRACTED: validate_kv_sharing_target 跨层共享校验
        #   （attention.py:L449-L454）——kv sharing 域（ch21/ch22）
        # SOURCE: vllm/model_executor/layers/attention/attention.py:L455（逐字）
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        # SUBTRACTED: mm_prefix_clamp_sliding_window 位（L456-L458）——多模态域

        # use a placeholder kv cache tensor during init, which will be replaced
        # by bind_kv_cache
        # this variable will not be accessed if use_direct_call is True
        # SOURCE: vllm/model_executor/layers/attention/attention.py:L460-L463（逐字）
        self.kv_cache = torch.tensor([])

        # SUBTRACTED: _init_kv_cache_quant 与 query_quant 位
        #   （attention.py:L465-L486）——KV cache 量化归 ch27

    # SOURCE: vllm/model_executor/layers/attention/attention.py:L488-L582 forward
    #   —— 减法子集（query_quant 量化块 L514-L524 删除——ch27；其余逐字：
    #   docstring 官方自述 + view 重排 + 直调/算子双通道）
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        # For some alternate attention backends like MLA the attention output
        # shape does not match the query shape, so we optionally let the model
        # definition specify the output tensor shape.
        output_shape: torch.Size | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """
        The KV cache is stored inside this class and is accessed via
        `self.kv_cache`.

        Attention metadata (`attn_metadata`) is set using a context manager in
        the model runner's `execute_model` method. It is accessed via forward
        context using
        `vllm.forward_context.get_forward_context().attn_metadata`.
        """
        # SOURCE: vllm/model_executor/layers/attention/attention.py:L488-L582 forward
        if self.calculate_kv_scales:
            torch.ops.vllm.maybe_calc_kv_scales(
                query, key, value, _encode_layer_name(self.layer_name)
            )
        if output_dtype is None:
            output_dtype = query.dtype
        # SUBTRACTED: query_quant 量化块（attention.py:L514-L524）——ch27

        if output_shape is None:
            # Handle both 2D [num_tokens, hidden] and
            # 3D [num_tokens, heads, head_dim] query
            num_tokens = query.shape[0]
            output_shape = torch.Size((num_tokens, self.num_heads * self.head_size_v))
        output = torch.empty(output_shape, dtype=output_dtype, device=query.device)
        hidden_size = output_shape[-1]
        # Reshape the query, key, and value tensors.
        # NOTE(woosuk): We do this outside the custom op to minimize the
        # CPU overheads from the non-CUDA-graph regions.
        query = query.view(-1, self.num_heads, self.head_size)
        output = output.view(-1, self.num_heads, self.head_size_v)
        if key is not None:
            key = key.view(-1, self.num_kv_heads, self.head_size)
        if value is not None:
            value = value.view(-1, self.num_kv_heads, self.head_size_v)
        kv_cache_dummy_dep = None
        if self.use_direct_call:
            # Skip this if sharing KV cache with an earlier attention layer.
            if (
                not self.attn_backend.forward_includes_kv_cache_update
                and self.kv_sharing_target_layer_name is None
                and key is not None
                and value is not None
            ):
                kv_cache_dummy_dep = unified_kv_cache_update(
                    key, value, self.layer_name
                )
            unified_attention_with_output(
                query,
                key,
                value,
                output,
                self.layer_name,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
        else:
            # Skip this if sharing KV cache with an earlier attention layer.
            encoded = _encode_layer_name(self.layer_name)
            if (
                not self.attn_backend.forward_includes_kv_cache_update
                and self.kv_sharing_target_layer_name is None
                and key is not None
                and value is not None
            ):
                kv_cache_dummy_dep = torch.ops.vllm.unified_kv_cache_update(
                    key, value, encoded
                )
            torch.ops.vllm.unified_attention_with_output(
                query,
                key,
                value,
                output,
                encoded,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
        return output.view(-1, hidden_size)

    # SUBTRACTED: calc_kv_scales 方法（attention.py:L584-L594）——KV scale
    #   计算面归 ch27（calculate_kv_scales 默认 False 不达）

    # SOURCE: vllm/model_executor/layers/attention/attention.py:L596-L602
    #   extra_repr（逐字）
    def extra_repr(self) -> str:
        # SOURCE: vllm/model_executor/layers/attention/attention.py:L596-L602
        s = f"head_size={self.impl.head_size}"  # type: ignore
        s += f", num_heads={self.impl.num_heads}"  # type: ignore
        s += f", num_kv_heads={self.impl.num_kv_heads}"  # type: ignore
        s += f", scale={self.impl.scale}"  # type: ignore
        s += f", backend={self.impl.__class__.__name__}"
        return s

    # SOURCE: vllm/model_executor/layers/attention/attention.py:L604-L616
    #   process_weights_after_loading —— 减法子集（impl 委托逐字；quant
    #   scale 默认化块 L607-L616 删除——ch27；impl 占位 None（选择面删除的
    #   延后绑定）时跳过——真实代码此处 impl 必已构造）
    def process_weights_after_loading(self, act_dtype: torch.dtype):
        # SOURCE: vllm/model_executor/layers/attention/attention.py:L604-L616
        if self.impl is not None:
            self.impl.process_weights_after_loading(act_dtype)

    # SOURCE: vllm/model_executor/layers/attention/attention.py:L618-L619
    #   get_attn_backend（逐字——AttentionLayerBase 抽象的实现位）
    def get_attn_backend(self) -> type[AttentionBackend]:
        # SOURCE: vllm/model_executor/layers/attention/attention.py:L618-L619
        return self.attn_backend

    # SUBTRACTED: get_kv_cache_spec 全方法（attention.py:L621-L694）——KV
    #   cache 规格体系归 ch22 已立（FullAttentionSpec/SlidingWindowSpec/
    #   turboquant 分支），本章插座面不构造规格


# SOURCE: vllm/model_executor/layers/attention/attention.py:L697-L712
#   maybe_calc_kv_scales（逐字——kv scale 计算算子本体；ch27 域经
#   calculate_kv_scales=False 不达）
def maybe_calc_kv_scales(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    # SOURCE: vllm/model_executor/layers/attention/attention.py:L697-L712
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]

    # Only calculate if the layer's calculate_kv_scales flag is True
    # This flag gets set to False after the first forward pass
    if not self.calculate_kv_scales:
        return

    self.calc_kv_scales(query, key, value)


# SOURCE: vllm/model_executor/layers/attention/attention.py:L715-L721
#   maybe_calc_kv_scales_fake（逐字）
def maybe_calc_kv_scales_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    # SOURCE: vllm/model_executor/layers/attention/attention.py:L715-L721
    return


# SUBTRACTED: direct_register_custom_op("maybe_calc_kv_scales") 注册块
#   （attention.py:L724-L729）——torch.library 注册面归 ch19；本章直调 Python
#   函数体（同控制流）


# SOURCE: vllm/model_executor/layers/attention/attention.py:L732-L772
#   get_attention_context（逐字——插座取上下文：attn_metadata 按 dict/list/
#   裸值三态解包、no_compile_layers[layer_name] 拿层实例、slot_mapping 表同源）
def get_attention_context(
    layer_name: str,
) -> tuple[Any, "Attention | MLAAttention", torch.Tensor, torch.Tensor]:
    """Extract attention context for a given layer.

    This helper function extracts the attention metadata, attention layer
    instance, KV cache tensor, and slot mapping for a specific layer.

    Args:
        layer_name: The name/identifier of the attention layer.

    Returns:
        A tuple containing:
        - attn_metadata: Attention metadata for this specific layer, or None if
            no metadata available
        - attn_layer: The attention layer instance (Attention or MLAAttention)
        - kv_cache: The KV cache tensor for current forward pass
        - slot_mapping: The slot mapping for this specific layer

        Note: attn_metadata may be None, but attn_layer and kv_cache are always
        extracted from the forward context.
    """
    # SOURCE: vllm/model_executor/layers/attention/attention.py:L732-L772
    forward_context: ForwardContext = get_forward_context()
    attn_metadata_raw = forward_context.attn_metadata
    attn_metadata: AttentionMetadata
    if isinstance(attn_metadata_raw, dict):
        attn_metadata = attn_metadata_raw[layer_name]
    elif isinstance(attn_metadata_raw, list):
        # list[dict[str, AttentionMetadata]]: used in speculative decoding
        # where [0] is the base-model (non-speculative) metadata dict.
        attn_metadata = attn_metadata_raw[0][layer_name]
    else:
        attn_metadata = attn_metadata_raw
    attn_layer: Attention = forward_context.no_compile_layers[layer_name]
    kv_cache = attn_layer.kv_cache
    slot_mapping = forward_context.slot_mapping
    assert isinstance(slot_mapping, dict), (
        f"Expected slot_mapping to be a dict, got {type(slot_mapping)}. "
    )
    layer_slot_mapping = slot_mapping.get(layer_name)
    return attn_metadata, attn_layer, kv_cache, layer_slot_mapping


# SOURCE: vllm/model_executor/layers/attention/attention.py:L775-L798
#   unified_kv_cache_update（逐字——KV 写算子：slot_mapping 交 impl 写、
#   返回 key.new_empty(0) 空张量 dummy 保 torch.compile 顺序）
def unified_kv_cache_update(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    """
    Returns a dummy that is passed to unified_attention to signal a side effect and
    the data dependency between them to ensure torch.compile preserves ordering.
    """
    # SOURCE: vllm/model_executor/layers/attention/attention.py:L775-L798
    layer_name = _resolve_layer_name(layer_name)
    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)
    if layer_slot_mapping is not None:
        assert hasattr(attn_layer.impl, "do_kv_cache_update"), (
            f"{attn_layer.impl.__class__.__name__} does not support kv cache update"
        )
        attn_layer.impl.do_kv_cache_update(  # type: ignore[attr-defined]
            attn_layer,
            key,
            value,
            kv_cache,
            layer_slot_mapping,
        )

    return key.new_empty(0)


# SOURCE: vllm/model_executor/layers/attention/attention.py:L801-L806
#   unified_kv_cache_update_fake（逐字）
def unified_kv_cache_update_fake(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    # SOURCE: vllm/model_executor/layers/attention/attention.py:L801-L806
    return torch.empty(0, device=key.device, dtype=key.dtype)


# SUBTRACTED: direct_register_custom_op("unified_kv_cache_update") 注册块
#   （attention.py:L809-L814）——ch19；直调 Python 函数体（同控制流）


# SOURCE: vllm/model_executor/layers/attention/attention.py:L817-L846
#   unified_attention_with_output —— 减法子集（函数体逐字——读算算子：
#   get_attention_context 取四元组后调 self.impl.forward 后端实现；
#   @eager_break_during_capture/@maybe_transfer_kv_layer 两装饰器删除——
#   ch19 捕获纪律/ch16 KV 搬运域）
def unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    # kv_cache_dummy_dep is not used but accepting it creates a data dependency
    # that ensures torch.compile preserves ordering between KV cache update and
    # attention forward.
    # SOURCE: vllm/model_executor/layers/attention/attention.py:L817-L846
    del kv_cache_dummy_dep
    layer_name = _resolve_layer_name(layer_name)
    attn_metadata, self, kv_cache, _ = get_attention_context(layer_name)

    self.impl.forward(
        self,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=output,
        output_scale=output_scale,
        output_block_scale=output_block_scale,
    )


# SOURCE: vllm/model_executor/layers/attention/attention.py:L849-L861
#   unified_attention_with_output_fake（逐字）
def unified_attention_with_output_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    # SOURCE: vllm/model_executor/layers/attention/attention.py:L849-L861
    return


# SUBTRACTED: direct_register_custom_op("unified_attention_with_output")
#   注册块（attention.py:L864-L869）——ch19；直调 Python 函数体（同控制流）
