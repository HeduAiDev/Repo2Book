# Subtract-only companion for v3 ch19 — vllm/model_executor/layers/attention/
# attention.py (pin v0.27.1 / 6e448d0ea). Same names, same structure, same
# control flow; only dossier-approved deletions (each marked `# SUBTRACTED:`),
# plus 章范围外域段以 SUBTRACTED+归属注记收窄（impl-notes §范围裁剪）。
#
# Deletions here (dossier subtraction_plan.delete #10 + 范围收窄):
#   #10 maybe_calc_kv_scales 算子及注册（L697-L729）与 forward 内调用
#      （L508-L511）、kv_sharing 校验/存储（L449-L455）与 forward 内两处
#      守卫（L544-L547/L563-L570）、mm_prefix_clamp_sliding_window 存储
#      （L456-L458）、get_kv_cache_spec（L621-L694——KV cache 域）。
#   范围收窄: attn_backend 选择与 impl 构造（L343-L434——ch21 后端域，
#      调用方/测试注入 attn.impl 与 attn.attn_backend）；量化标定装配
#      （L153-L221——ch27 域，query_quant 分支 F10 苗原文保留）。
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ...._host_seams import (
    current_platform,
    eager_break_during_capture,
    init_logger,
    maybe_transfer_kv_layer,
)
from ....config import get_current_vllm_config
from ....forward_context import ForwardContext, get_forward_context
from ....utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from ....v1.attention.backend import AttentionBackend, AttentionMetadata, AttentionType

# HOST SEAM 注记: CacheConfig 是 ch03 配置域的类型注记面（真源 from
#   vllm.config import CacheConfig）；伴读版以 Any 注记承载。
CacheConfig = Any

logger = init_logger(__name__)

# SUBTRACTED: validate_kv_sharing_target（L55-L59）、_largest_kernel_block_
#   within（L62-L121）、set_default_quant_scales（L124-L150）、_init_kv_
#   cache_quant（L153-L221）——kv_sharing 校验与量化标定装配（delete[10]
#   + ch27 域）。


# SOURCE: vllm/model_executor/layers/attention/attention.py:L223-L486 Attention
#   —— 算子化的注意力层：构造期自注册 static_forward_context、forward 只做
#   reshape + 调统一算子（KV 写拆独立算子 dummy 依赖保序）
class Attention(nn.Module):
    """Attention layer.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    # SUBTRACTED: AttentionLayerBase 基类（L223——attention_layer_base 的
    #   delayed-forward 协议，ch21 域）。

    # SOURCE: vllm/model_executor/layers/attention/attention.py:L235-L254
    #   __init__ 签名（**extra_impl_args 承接 impl 构造参数）
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        use_alibi_sqrt: bool | None = None,
        cache_config: "CacheConfig | None" = None,
        quant_config: Any | None = None,
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
        # SOURCE: vllm/model_executor/layers/attention/attention.py:L259-L268
        sliding_window: int | None
        if per_layer_sliding_window is not None:
            # per-layer sliding window
            sliding_window = per_layer_sliding_window
        elif cache_config is not None:
            # model-level sliding window
            sliding_window = cache_config.sliding_window
        else:
            sliding_window = None

        # SOURCE: vllm/model_executor/layers/attention/attention.py:L270-L276
        vllm_config = get_current_vllm_config()
        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            # SUBTRACTED: calculate_kv_scales 读取（L273——maybe_calc_kv_scales
            #   随 delete[10] 删）。
        else:
            kv_cache_dtype = "auto"

        # SUBTRACTED: llm-compressor kv_cache_scheme/逐层跳过量化（L278-L321
        #   ——ch27 量化域）。

        # SOURCE: vllm/model_executor/layers/attention/attention.py:L326-L341
        self.kv_cache_dtype = kv_cache_dtype
        # SUBTRACTED: kv_cache_torch_dtype/calculate_kv_scales（L323/L327
        #   ——get_kv_cache_spec 与 kv scales 消费侧均随 delete[10] 删）。
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

        # SUBTRACTED: attn_backend 选择（get_attn_backend L350-L363）、
        #   alibi_sqrt/flex_attn/chunk_lookback 校验（L364-L419）与 impl 构造
        #   （L420-L434）、use_mm_prefix（L343-L345）——ch21 后端域：调用方/
        #   测试注入 attn.impl 与 attn.attn_backend（ch21 接口，站 5/14）。
        self.impl = None
        self.attn_backend = None

        # For cuda-alike (CUDA and ROCM) and cpu platforms, we control how
        # torch.compile works by registering the attention as one giant
        # opaque custom op. For other platforms, we directly call them
        # and let torch.compile handle them.
        # SOURCE: vllm/model_executor/layers/attention/attention.py:L437-L441
        self.use_direct_call = not current_platform.opaque_attention_op()

        # SOURCE: vllm/model_executor/layers/attention/attention.py:L443-L447
        #   —— 构造期自注册：prefix 重名即 raise、注册进
        #   compilation_config.static_forward_context
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.attn_type = attn_type

        # SUBTRACTED: kv_sharing 校验/存储与 mm_prefix_clamp_sliding_window
        #   （L449-L458——delete[10]：层间共享特例与 Gemma4 clamp）。

        # use a placeholder kv cache tensor during init, which will be replaced
        # by bind_kv_cache
        # this variable will not be accessed if use_direct_call is True
        # SOURCE: vllm/model_executor/layers/attention/attention.py:L460-L463
        self.kv_cache = torch.tensor([])

        # SUBTRACTED: _init_kv_cache_quant 调用（L466——量化标定装配，ch27
        #   域）。

        # for attn backends supporting query quantization
        # SOURCE: vllm/model_executor/layers/attention/attention.py:L468-L469
        self.query_quant = None
        # SUBTRACTED: query_quant 构造分支（L470-L486——读 self.impl.
        #   supports_quant_query_input 与 ch27 量化器构造；impl 随 ch21 域
        #   注入后方可用。forward 内的量化分支（L514-L524，F10 苗）原文
        #   保留）。

    # SOURCE: vllm/model_executor/layers/attention/attention.py:L488-L582
    #   forward —— 预分配 output、reshape 在 op 外（NOTE(woosuk)：minimize
    #   the CPU overheads from the non-CUDA-graph regions）、CUDA 系平台走
    #   torch.ops.vllm.unified_attention_with_output（out-variant）
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
        # SUBTRACTED: calculate_kv_scales 前置分支（L508-L511——delete[10]）。
        if output_dtype is None:
            output_dtype = query.dtype
        # SOURCE: vllm/model_executor/layers/attention/attention.py:L514-L524
        #   query_quant 分支（F10 苗，原文保留：simple torch operation 让
        #   torch.compile 把量化融进前面的算子）
        if self.query_quant is not None:
            # quantizing with a simple torch operation enables
            # torch.compile to fuse this into previous ops
            # which reduces overheads during decoding.
            # Otherwise queries are quantized using custom ops
            # which causes decoding overheads
            assert self.kv_cache_dtype in {"fp8", "fp8_e4m3", "nvfp4"}

            # check if query quantization is supported
            if self.impl.supports_quant_query_input:
                query, _ = self.query_quant(query, self._q_scale)

        # SOURCE: vllm/model_executor/layers/attention/attention.py:L526-L541
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
        # SOURCE: vllm/model_executor/layers/attention/attention.py:L543-L561
        #   direct-call 支（kv_sharing 守卫随 delete[10] 删净）
        if self.use_direct_call:
            # Skip this if sharing KV cache with an earlier attention layer.
            if (
                not self.attn_backend.forward_includes_kv_cache_update
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
        # SOURCE: vllm/model_executor/layers/attention/attention.py:L562-L581
        #   torch.ops 支（不透明平台进图的形态）
        else:
            # Skip this if sharing KV cache with an earlier attention layer.
            encoded = _encode_layer_name(self.layer_name)
            if (
                not self.attn_backend.forward_includes_kv_cache_update
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

    # SUBTRACTED: calc_kv_scales（L584-L594——delete[10]）、process_weights_
    #   after_loading（L604-L616——权重加载域）、get_kv_cache_spec（L621-L694
    #   ——KV cache 域，ch13-16 已立消费侧）、maybe_calc_kv_scales(+fake/
    #   注册)（L697-L729——delete[10]）。


# SOURCE: vllm/model_executor/layers/attention/attention.py:L732-L772
#   get_attention_context —— 算子实现内按 layer_name 从 ForwardContext 取
#   attn_metadata（dict 按层 / list 按 spec decode）/ 层实例 / kv_cache /
#   slot_mapping——thread-local 执行环境的消费口
def get_attention_context(  # SOURCE: vllm/model_executor/layers/attention/attention.py:L732-L772
    layer_name: str,
) -> tuple[Any, "Attention", torch.Tensor, torch.Tensor]:
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
#   unified_kv_cache_update —— KV 写独立算子：返回空张量作 dummy 数据依赖
#   传入 attention 算子保序；实现经 get_attention_context 调
#   impl.do_kv_cache_update
def unified_kv_cache_update(  # SOURCE: vllm/model_executor/layers/attention/attention.py:L775-L798
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    """
    Returns a dummy that is passed to unified_attention to signal a side effect and
    the data dependency between them to ensure torch.compile preserves ordering.
    """
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
#   unified_kv_cache_update_fake —— fake 实现让 Dynamo 可 trace
def unified_kv_cache_update_fake(  # SOURCE: vllm/model_executor/layers/attention/attention.py:L801-L806
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    return torch.empty(0, device=key.device, dtype=key.dtype)


# SOURCE: vllm/model_executor/layers/attention/attention.py:L809-L814 注册
direct_register_custom_op(
    op_name="unified_kv_cache_update",
    op_func=unified_kv_cache_update,
    fake_impl=unified_kv_cache_update_fake,
    mutates_args=[],
)


# SOURCE: vllm/model_executor/layers/attention/attention.py:L817-L846
#   unified_attention_with_output —— out-variant 统一算子：del dummy 依赖后
#   经 get_attention_context 取上下文转调 self.impl.forward（执行环境全部
#   从 forward context 来，签名里只有一个层名）
@eager_break_during_capture
@maybe_transfer_kv_layer
def unified_attention_with_output(  # SOURCE: vllm/model_executor/layers/attention/attention.py:L817-L846
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


# SOURCE: vllm/model_executor/layers/attention/attention.py:L849-L859 fake
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
    return


# SOURCE: vllm/model_executor/layers/attention/attention.py:L862-L867 注册
direct_register_custom_op(
    op_name="unified_attention_with_output",
    op_func=unified_attention_with_output,
    mutates_args=["output", "output_block_scale"],
    fake_impl=unified_attention_with_output_fake,
)
