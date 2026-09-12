# SOURCE: vllm/model_executor/layers/attention/attention.py
# ch25 消费面：MLAAttention 共用的 quant 尺度三件 + 上下文取件口——
#   should_load_quant_weights（L88-L92 逐字）
#   set_default_quant_scales（L124-L151 逐字）
#   _init_kv_cache_quant（L153-L200 减法：compressed-tensors checkpoint
#     尺度分支归 ch27）
#   get_attention_context（L732-L772 逐字——unified 算子的三态解包）
# SUBTRACTED：Attention 通用插座类与 unified_kv_cache_update /
#   unified_attention_with_output 双算子——ch21/ch23 域（MLA 有自己的
#   unified_mla_* 两个算子，在 mla_attention.py）。
from __future__ import annotations

import torch
from torch import nn

import vllm.envs as envs
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.v1.attention.backend import AttentionMetadata


# SOURCE: vllm/model_executor/layers/attention/attention.py:L88-L92
#   should_load_quant_weights —— 逐字
def should_load_quant_weights(quant_method) -> bool:
    # SOURCE: vllm/model_executor/layers/attention/attention.py:L88-L92（锚点双置：声明上方注释同文）
    """Returns whether the quantization method should load quantized weights."""
    # HOST SEAM：UnquantizedLinearMethod 以 isinstance 判定——本包 seam
    #   Linear 的 quant_method 面（quant_config=None → None）语义同真实
    return quant_method is not None


# SOURCE: vllm/model_executor/layers/attention/attention.py:L124-L151
#   set_default_quant_scales —— 逐字
def set_default_quant_scales(layer: nn.Module, register_buffer: bool = False) -> None:
    """Sets default quantization scales for the layer."""
    # SOURCE: vllm/model_executor/layers/attention/attention.py:L124-L151
    if register_buffer:
        layer.register_buffer("_k_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_v_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_q_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_prob_scale", torch.tensor(1.0, dtype=torch.float32))
    else:
        layer._k_scale.fill_(1.0)
        layer._v_scale.fill_(1.0)
        layer._q_scale.fill_(1.0)
        layer._prob_scale.fill_(1.0)

    # We also keep q/k/v_scale on host (cpu) memory for attention
    # backends that require the scales to be on host instead of on device.
    # e.g. Flashinfer & AITER
    layer._q_scale_float = 1.0
    layer._k_scale_float = 1.0
    layer._v_scale_float = 1.0
    layer._k_scale_cpu = torch.tensor(1.0, dtype=torch.float32)
    layer._v_scale_cpu = torch.tensor(1.0, dtype=torch.float32)
    layer._prob_scale_float = 1.0

    # Initialize q/k/v range constants used by calc_kv_scales
    layer.q_range = torch.tensor(envs.Q_SCALE_CONSTANT, dtype=torch.float32)
    layer.k_range = torch.tensor(envs.K_SCALE_CONSTANT, dtype=torch.float32)
    layer.v_range = torch.tensor(envs.V_SCALE_CONSTANT, dtype=torch.float32)


# SOURCE: vllm/model_executor/layers/attention/attention.py:L153-L200
#   _init_kv_cache_quant —— 减法子集
def _init_kv_cache_quant(
    layer: nn.Module,
    quant_config,
    prefix: str,
) -> None:
    """Initializes KV cache scaling factors and quantization method.

    This helper function sets up the KV cache quantization attributes that are
    shared between Attention and MLAAttention layers. It initializes scale
    tensors for query, key, value, and probability, and configures the
    quantization method if applicable.

    Args:
        layer: The attention layer instance to initialize.
        quant_config: Optional quantization configuration.
        prefix: Layer name prefix for quantization method lookup.
    """

    # Note [Register q/k/v/prob scales in state dict]
    # When calling model.to(device), only parameters/buffers in state dict are
    # moved. If not registering q/k/v/prob scales in state dict, there would be
    # an IMA error when a cuda kernel (e.g. quant_fp8) accesses the tensor
    # on cpu.
    # Registering in state dict means it interacts with weight loading. One edge
    # case is when quant_method is None, or quant_method is UnquantizedLinearMethod
    # (i.e. should_load_quant_weights(quant_method) == False).
    # In this case, the checkpoint does not have the scales. We need to
    # initialize the scales to 1.0 and update the scales after weight loading.
    # This is espectially important when we load dummy weights first (providing
    # wrong scales) and then load real weights (which misses scales and keeps the
    # wrong scales from dummy load).
    # SOURCE: vllm/model_executor/layers/attention/attention.py:L187-L189
    set_default_quant_scales(layer, register_buffer=True)

    # The output scale on host memory. This should be the input scale of
    # the quant op after this attention layer.
    layer._o_scale_float = None

    # SOURCE: 真实此处 quant_method = quant_config.get_quant_method(...) 后按
    #   BaseKVCacheMethod 装配 checkpoint 尺度（L192-L200）——SUBTRACTED：
    #   compressed-tensors 尺度面归 ch27；quant_config=None 时语义同真实
    #   （quant_method None → should_load False → 尾段走默认 1.0）


# SOURCE: vllm/model_executor/layers/attention/attention.py:L732-L772
#   get_attention_context —— 逐字
def get_attention_context(
    layer_name: str,
):
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
    # SOURCE: vllm/model_executor/layers/attention/attention.py:L760-L772
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
    attn_layer = forward_context.no_compile_layers[layer_name]
    kv_cache = attn_layer.kv_cache
    slot_mapping = forward_context.slot_mapping
    assert isinstance(slot_mapping, dict), (
        f"Expected slot_mapping to be a dict, got {type(slot_mapping)}. "
    )
    layer_slot_mapping = slot_mapping.get(layer_name)
    return attn_metadata, attn_layer, kv_cache, layer_slot_mapping
