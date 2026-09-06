# Subtract-only companion for v3 ch21 — vllm/model_executor/layers/attention/
# attention.py (pin v0.27.1 / 6e448d0ea). Same names, same structure, same
# control flow; only dossier-approved deletions (delete[9], each marked
# `# SUBTRACTED:`), plus 章范围外域段以 SUBTRACTED+归属注记收窄。
#
# 本章主角文件之一（站 1 + 站 11-12）：模型层插座——Attention.__init__ 建层
# 即选后端（get_attn_backend → impl 实例化 → 枚举反查 → static_forward_context
# 自注册）、forward 双算子分发（写腿先行 + dummy 依赖保序）、get_kv_cache_spec
# 层自报规格、以及统一算子三件（get_attention_context / unified_kv_cache_
# update / unified_attention_with_output——真实 torch.library 注册）。
from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn

from ...._host_seams import (
    BaseKVCacheMethod,
    QuantizationConfig,
    QuantizeMethodBase,
    UnquantizedLinearMethod,
    current_platform,
    eager_break_during_capture,
    get_current_vllm_config,
    init_logger,
    maybe_transfer_kv_layer,
)
from ....forward_context import ForwardContext, get_forward_context
from ....utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from ....v1.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    AttentionType,
)
from ....v1.attention.backends.registry import AttentionBackendEnum
from ....v1.attention.selector import get_attn_backend
from ....v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
    get_kv_quant_mode,
)
from ...layers.attention_layer_base import AttentionLayerBase

logger = init_logger(__name__)

# SUBTRACTED: maybe_transfer_kv_layer / eager_break_during_capture 的真身
#   （ch16/ch19 域——HOST SEAM 恒等装饰器承载，见 _host_seams）。


# SOURCE: vllm/model_executor/layers/attention/attention.py:L55-L85 validate_
#   kv_sharing_target ——（逐字）跨层 KV 共享的目标层校验
def validate_kv_sharing_target(  # SOURCE: vllm/model_executor/layers/attention/attention.py
    current_layer_name, target_layer_name, static_forward_context
):
    error_msg = (
        f"Specified KV sharing target layer for {current_layer_name} "
        f"is not valid: target layer {target_layer_name} "
    )

    if current_layer_name == target_layer_name:
        raise ValueError(error_msg + "cannot be the same as the current layer.")

    if target_layer_name not in static_forward_context:
        from ...models.utils import extract_layer_index

        # If target layer name is not in the static fwd context, it means either
        # a) the target layer does not come BEFORE the current layer, or
        # b) the target layer is not an Attention layer that exists in the model
        current_layer_idx = extract_layer_index(current_layer_name)
        target_layer_idx = extract_layer_index(target_layer_name)
        if current_layer_idx <= target_layer_idx:
            raise ValueError(error_msg + "must come before the current layer.")
        else:
            raise ValueError(error_msg + "is not a valid Attention layer in the model.")

    # Currently KV sharing is only supported between layers of the same type
    target_layer_attn_type = static_forward_context[target_layer_name].attn_type
    expected = static_forward_context[current_layer_name].attn_type
    if target_layer_attn_type != expected:
        raise ValueError(
            error_msg + f"must be the same type as the current layer ({expected})."
        )


# SOURCE: vllm/model_executor/layers/attention/attention.py:L88-L92 should_load_
#   quant_weights ——（逐字；UnquantizedLinearMethod 是 ch27 量化域类型面）
def should_load_quant_weights(quant_method: QuantizeMethodBase | None) -> bool:  # SOURCE: vllm/model_executor/layers/attention/attention.py
    """Returns whether the quantization method should load quantized weights."""
    return quant_method is not None and not isinstance(
        quant_method, UnquantizedLinearMethod
    )


# SOURCE: vllm/model_executor/layers/attention/attention.py:L95-L121 _largest_
#   kernel_block_within ——（逐字）滑窗层选能塞进共享页的最大 kernel 块
def _largest_kernel_block_within(  # SOURCE: vllm/model_executor/layers/attention/attention.py
    attn_backend: "type[AttentionBackend]",
    per_token_bytes: int,
    page_budget: int | None,
    fallback: int,
) -> int:
    """Largest supported kernel block size whose page fits in ``page_budget``.

    A padded spec (e.g. skip-quant layer) that pads its page up to a large shared page
    wastes ``page_budget - block*per_token`` bytes per block. Picking the largest kernel
    block whose natural page still fits under ``page_budget`` minimizes that waste.
    Falls back to the smallest supported block when ``page_budget`` is None (no padding
    — the block is handled by ``unify``'s integer scaling instead) or nothing fits.
    """
    from ....v1.attention.backend import MultipleOf

    sizes = attn_backend.get_supported_kernel_block_sizes()
    candidates = [s for s in sizes if isinstance(s, int)]
    if not candidates:
        candidates = [s.base for s in sizes if isinstance(s, MultipleOf)]
    if not candidates:
        return fallback
    smallest = min(candidates)
    if not page_budget or per_token_bytes <= 0:
        return smallest
    fitting = [b for b in candidates if b * per_token_bytes <= page_budget]
    return max(fitting) if fitting else smallest


# SOURCE: vllm/model_executor/layers/attention/attention.py:L124-L150 set_
#   default_quant_scales ——（逐字；envs.Q_SCALE_CONSTANT 等 ch27 常量面以
#   HOST SEAM 等值承载，见 _host_seams）
def set_default_quant_scales(layer: nn.Module, register_buffer: bool = False) -> None:
    """Sets default quantization scales for the layer."""
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
    # SOURCE: vllm/model_executor/layers/attention/attention.py:L147-L149
    #   （Q/K/V_SCALE_CONSTANT 是 ch27 量化域常量，HOST SEAM 等值 1.0）
    layer.q_range = torch.tensor(1.0, dtype=torch.float32)
    layer.k_range = torch.tensor(1.0, dtype=torch.float32)
    layer.v_range = torch.tensor(1.0, dtype=torch.float32)


# SOURCE: vllm/model_executor/layers/attention/attention.py:L153-L220 _init_kv_
#   cache_quant ——（逐字；量化分支的 CompressedTensors 局部 import 是
#   ch27 域——本章 quant_config=None 恒不触发，调用位保留）
def _init_kv_cache_quant(  # SOURCE: vllm/model_executor/layers/attention/attention.py
    layer: nn.Module,
    quant_config: QuantizationConfig | None,
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
    # an IMA error when a cuda kernel (e.g., quant_fp8) accesses the tensor
    # on cpu.
    # Registering in state dict means it interacts with weight loading. One edge
    # case is when quant_method is None, or quant_method is UnquantizedLinearMethod
    # (i.e. should_load_quant_weights(quant_method) == False).
    # In this case, the checkpoint does not have the scales. We need to
    # initialize the scales to 1.0 and update the scales after weight loading.
    # This is espectially important when we load dummy weights first (providing
    # wrong scales) and then load real weights (which misses scales and keeps the
    # wrong scales from dummy load).
    set_default_quant_scales(layer, register_buffer=True)

    # The output scale on host memory. This should be the input scale of
    # the quant op after this attention layer.
    layer._o_scale_float = None

    quant_method = (
        quant_config.get_quant_method(layer, prefix=prefix) if quant_config else None
    )

    # See [Note: Register q/k/v/prob scales in state dict]
    if should_load_quant_weights(quant_method):
        assert isinstance(quant_method, BaseKVCacheMethod)
        # TODO (mgoin): kv cache dtype should be specified in the FP8
        # checkpoint config and become the "auto" behavior
        if layer.kv_cache_dtype == "fp8_e5m2":
            # SUBTRACTED: CompressedTensors 局部 import 块（L203-L214——
            #   ch27 量化域；本章无量化 checkpoint，分支不可达，守卫位保留）。
            pass
        # If quantization is enabled, we make "k_scale" and "v_scale"
        # parameters so that it can be loaded from the model checkpoint.
        # The k/v_scale will then be converted back to native float32
        # values after weight loading.
        layer.quant_method = quant_method
        layer.quant_method.create_weights(layer)


# SOURCE: vllm/model_executor/layers/attention/attention.py:L223-L694 Attention ——
#   模型层插座：建层即选后端 + forward 双算子分发 + 层自报 KV 规格
class Attention(nn.Module, AttentionLayerBase):
    """Attention layer.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    def __init__(  # SOURCE: vllm/model_executor/layers/attention/attention.py:L235-L486（delete[9] 删特化段）
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        use_alibi_sqrt: bool | None = None,
        cache_config: Any | None = None,
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
        if per_layer_sliding_window is not None:
            # per-layer sliding window
            sliding_window = per_layer_sliding_window
        elif cache_config is not None:
            # model-level sliding window
            sliding_window = cache_config.sliding_window
        else:
            sliding_window = None

        vllm_config = get_current_vllm_config()
        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            calculate_kv_scales = cache_config.calculate_kv_scales
        else:
            kv_cache_dtype = "auto"
            calculate_kv_scales = False

        # llm-compressor models declare an FP8 KV-cache scheme in their
        # checkpoint config. Honor it only when the user did not explicitly
        # pick a kv_cache_dtype; an explicit choice (e.g. bfloat16) must win.
        # The "auto" case is normally resolved upstream in
        # resolve_kv_cache_dtype_string, but we re-apply here defensively in
        # case anything bypassed that path.
        kv_cache_scheme = getattr(quant_config, "kv_cache_scheme", None)
        if kv_cache_scheme is not None and kv_cache_dtype == "auto":
            kv_cache_dtype = "fp8"
            calculate_kv_scales = False
            if cache_config is not None:
                cache_config.cache_dtype = "fp8"
                cache_config.calculate_kv_scales = False

        # Check if per-head quant scales are required based on kv_cache_scheme
        use_per_head_quant_scales = (
            kv_cache_scheme is not None
            and kv_cache_scheme.get("strategy") == "attn_head"
        )

        # Skip quantization for specified layers
        if cache_config is not None and cache_config.kv_cache_dtype_skip_layers:
            from ...models.utils import extract_layer_index

            skip = False
            # Check attention type
            if (
                sliding_window is not None
                and "sliding_window" in cache_config.kv_cache_dtype_skip_layers
            ):
                skip = True
            # Check layer index
            layer_idx = extract_layer_index(prefix)
            if str(layer_idx) in cache_config.kv_cache_dtype_skip_layers:
                skip = True
            if skip:
                kv_cache_dtype = "auto"
                calculate_kv_scales = False
            logger.debug(
                "Layer %s: kv_cache_dtype=%s, sliding_window=%s",
                prefix,
                kv_cache_dtype,
                sliding_window,
            )

        from ....utils.torch_utils import kv_cache_dtype_str_to_dtype

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

        # NOTE: model_config may be None during certain tests
        model_config = vllm_config.model_config
        self.use_mm_prefix = model_config is not None and model_config.is_mm_prefix_lm

        # During model initialization, the default dtype is set as the model
        # weight and activation dtype.
        dtype = torch.get_default_dtype()
        if attn_backend is None:
            # SOURCE: vllm/model_executor/layers/attention/attention.py:L350-L363
            #   ——建层即选后端（本章站 1 原文）：构造方没显式传 attn_backend
            #   就按 (head_size, dtype, kv_cache_dtype, 层信号) 自选
            self.attn_backend = get_attn_backend(
                head_size,
                dtype,
                kv_cache_dtype,
                use_mla=False,
                has_sink=self.has_sink,
                use_mm_prefix=self.use_mm_prefix,
                use_per_head_quant_scales=use_per_head_quant_scales,
                attn_type=attn_type,
                has_sliding_window=sliding_window is not None,
            )
        else:
            self.attn_backend = attn_backend

        # SUBTRACTED: alibi_sqrt / batch-invariance+prefix-caching 互斥 /
        #   chunk_lookback / FLEX_ATTENTION block_m 段（L364-L418）——
        #   delete[9]：各后端特化与批不变参数是正交特性；精简版走标准
        #   decoder 路径。

        # SOURCE: vllm/model_executor/layers/attention/attention.py:L420-L435
        #   ——选完即装：get_impl_cls() 实例化 impl + 枚举反查 + dtype 记账
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
        self.backend = AttentionBackendEnum[self.attn_backend.get_name()]
        self.dtype = dtype

        # For cuda-alike (CUDA and ROCM) and cpu platforms, we control how
        # torch.compile works by registering the attention as one giant
        # opaque custom op. For other platforms, we directly call them
        # and let torch.compile handle them.
        # SOURCE: vllm/model_executor/layers/attention/attention.py:L441 ——
        #   平台分叉旗标（host 直调；torch.ops 分支同控制流）
        self.use_direct_call = not current_platform.opaque_attention_op()

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        # SOURCE: vllm/model_executor/layers/attention/attention.py:L446 ——
        #   自注册（重名即 raise）——注册的发起方
        compilation_config.static_forward_context[prefix] = self
        self.attn_type = attn_type

        if kv_sharing_target_layer_name is not None:
            validate_kv_sharing_target(
                prefix,
                kv_sharing_target_layer_name,
                compilation_config.static_forward_context,
            )
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        # Gemma4: clamp mm_prefix bidirectional ranges by the sliding window
        # (read by the Triton backend impl). Default False for all other models.
        # SOURCE: vllm/model_executor/layers/attention/attention.py:L456-L458
        self.mm_prefix_clamp_sliding_window = mm_prefix_clamp_sliding_window

        # use a placeholder kv cache tensor during init, which will be replaced
        # by bind_kv_cache
        # this variable will not be accessed if use_direct_call is True
        # SOURCE: vllm/model_executor/layers/attention/attention.py:L460-L463
        self.kv_cache = torch.tensor([])

        # Initialize KV cache quantization attributes
        # SOURCE: vllm/model_executor/layers/attention/attention.py:L466 ——
        #   保留（attention.py 内唯一调用点；删则保留的写腿读 _k_scale/_v_scale
        #   必 AttributeError）
        _init_kv_cache_quant(self, quant_config, prefix)

        # SUBTRACTED: query_quant 装配（L467-L486）——delete[9]：量化 query
        #   输入的 FP8 装配（ch27 线）；impl.supports_quant_query_input 位
        #   恒 False 的配置下不触发。

    def forward(  # SOURCE: vllm/model_executor/layers/attention/attention.py:L488-L582（delete[9] 删量化段）
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
        # SUBTRACTED: calculate_kv_scales 段（L508-L511——maybe_calc_kv_scales
        #   算子调用）与 query_quant 段（L514-L524）——delete[9]。
        # SOURCE: vllm/model_executor/layers/attention/attention.py:L512-L513
        #   ——保留（output_dtype 默认值，下方 torch.empty 必需）
        if output_dtype is None:
            output_dtype = query.dtype
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
                # SOURCE: vllm/model_executor/layers/attention/attention.py:L551-L553
                #   ——写腿先行（forward_includes_kv_cache_update=False 时）
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

    def calc_kv_scales(self, query, key, value):  # SOURCE: vllm/model_executor/layers/attention/attention.py:L584-L594（逐字）
        self._q_scale.copy_(torch.abs(query).max() / self.q_range)
        self._k_scale.copy_(torch.abs(key).max() / self.k_range)
        self._v_scale.copy_(torch.abs(value).max() / self.v_range)
        self._q_scale_float = self._q_scale.item()
        self._k_scale_float = self._k_scale.item()
        self._v_scale_float = self._v_scale.item()
        self._k_scale_cpu.fill_(self._k_scale_float)
        self._v_scale_cpu.fill_(self._v_scale_float)
        # We only calculate the scales once
        self.calculate_kv_scales = False

    def extra_repr(self) -> str:  # SOURCE: vllm/model_executor/layers/attention/attention.py:L596-L602（逐字）
        s = f"head_size={self.impl.head_size}"  # type: ignore
        s += f", num_heads={self.impl.num_heads}"  # type: ignore
        s += f", num_kv_heads={self.impl.num_kv_heads}"  # type: ignore
        s += f", scale={self.impl.scale}"  # type: ignore
        s += f", backend={self.impl.__class__.__name__}"
        return s

    def process_weights_after_loading(self, act_dtype: torch.dtype):  # SOURCE: vllm/model_executor/layers/attention/attention.py:L604-L616
        self.impl.process_weights_after_loading(act_dtype)

        # If we should not load quant weights, we initialize the scales to 1.0
        # as the default value. See [Note: Register q/k/v/prob scales in state dict]
        # for more details.
        quant_method = (
            self.quant_config.get_quant_method(self, prefix=self.layer_name)
            if self.quant_config
            else None
        )
        if not should_load_quant_weights(quant_method):
            set_default_quant_scales(self, register_buffer=False)

    def get_attn_backend(self) -> type[AttentionBackend]:  # SOURCE: vllm/model_executor/layers/attention/attention.py:L618-L619（逐字）
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: Any) -> KVCacheSpec | None:  # SOURCE: vllm/model_executor/layers/attention/attention.py:L621-L694（delete[9] 删 turboquant 支）
        # Block size may get updated after model loading, refresh it
        block_size = vllm_config.cache_config.block_size
        # Encoder-only attention is prefill-only and keeps no autoregressive KV
        # cache. In hybrid models (e.g. Qwen3.5 / ColQwen3.5: GatedDeltaNet
        # linear_attention interleaved with full_attention) the runner iterates
        # every attention module to build the KV-cache spec, so an ENCODER_ONLY
        # full_attention layer reaches here; it contributes no KV cache group.
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return None
        # Should not be called for enc-dec attention.
        assert self.attn_type == AttentionType.DECODER
        quant_mode = get_kv_quant_mode(self.kv_cache_dtype)
        if self.sliding_window is not None:
            assert not self.attn_backend.is_mla(), (
                "MLA is not supported for sliding window"
            )
            # SW chooses its own block_size, decoupled from the user's
            # ``--block-size`` (which only constrains primary attention).
            # When this SW layer is a padded spec (skip-quant: its page is
            # padded up to ``skip_page_size_padded``), pick the largest kernel
            # block that still fits the shared page so we waste fewer padding
            # bytes per block. Otherwise (page_size_padded is None) the smallest
            # block is fine — ``unify`` scales it up by an integer ratio.
            shared_page = vllm_config.cache_config.skip_page_size_padded
            sw_per_token = SlidingWindowSpec(
                block_size=1,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                head_size_v=self.head_size_v,
                dtype=self.kv_cache_torch_dtype,
                kv_quant_mode=quant_mode,
                sliding_window=self.sliding_window,
            ).real_page_size_bytes
            sw_block_size = _largest_kernel_block_within(
                self.attn_backend, sw_per_token, shared_page, block_size
            )
            return SlidingWindowSpec(
                block_size=sw_block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                head_size_v=self.head_size_v,
                dtype=self.kv_cache_torch_dtype,
                kv_quant_mode=quant_mode,
                sliding_window=self.sliding_window,
                page_size_padded=shared_page,
            )
        # SUBTRACTED: turboquant 分支（L668-L685）——delete[9]：TQ 量化 KV
        #   的专属 spec（ch27 线）。
        else:
            return FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                head_size_v=self.head_size_v,
                dtype=self.kv_cache_torch_dtype,
                kv_quant_mode=quant_mode,
            )


# SOURCE: vllm/model_executor/layers/attention/attention.py:L697-L712 maybe_calc_
#   kv_scales ——（逐字；调用支随 delete[9] 删，算子本体与注册保留）
def maybe_calc_kv_scales(  # SOURCE: vllm/model_executor/layers/attention/attention.py
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]

    # Only calculate if the layer's calculate_kv_scales flag is True
    # This flag gets set to False after the first forward pass
    if not self.calculate_kv_scales:
        return

    self.calc_kv_scales(query, key, value)


# SOURCE: vllm/model_executor/layers/attention/attention.py:L715-L721 maybe_calc_kv_scales_fake
def maybe_calc_kv_scales_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    return


# SOURCE: vllm/model_executor/layers/attention/attention.py:L724-L729 注册块（逐字）
direct_register_custom_op(
    op_name="maybe_calc_kv_scales",
    op_func=maybe_calc_kv_scales,
    mutates_args=["query", "key", "value"],
    fake_impl=maybe_calc_kv_scales_fake,
)


# SOURCE: vllm/model_executor/layers/attention/attention.py:L732-L772 get_attention_
#   context ——（逐字）按 layer_name 取本层执行环境的核心取数函数
def get_attention_context(  # SOURCE: vllm/model_executor/layers/attention/attention.py
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


# SOURCE: vllm/model_executor/layers/attention/attention.py:L775-L798 unified_kv_
#   cache_update ——（逐字）写腿算子体：do_kv_cache_update 派发 + 空张量 dummy
def unified_kv_cache_update(  # SOURCE: vllm/model_executor/layers/attention/attention.py
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


# SOURCE: vllm/model_executor/layers/attention/attention.py:L801-L806 unified_kv_cache_update_fake
def unified_kv_cache_update_fake(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    return torch.empty(0, device=key.device, dtype=key.dtype)


# SOURCE: vllm/model_executor/layers/attention/attention.py:L809-L814 注册块（逐字）
direct_register_custom_op(
    op_name="unified_kv_cache_update",
    op_func=unified_kv_cache_update,
    fake_impl=unified_kv_cache_update_fake,
    mutates_args=[],
)


# SOURCE: vllm/model_executor/layers/attention/attention.py:L817-L846 unified_
#   attention_with_output ——（逐字）读+算算子体（装饰器为 ch19/ch16 域 seam）
@eager_break_during_capture
@maybe_transfer_kv_layer
def unified_attention_with_output(  # SOURCE: vllm/model_executor/layers/attention/attention.py
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


# SOURCE: vllm/model_executor/layers/attention/attention.py:L849-L859 unified_attention_with_output_fake
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


# SOURCE: vllm/model_executor/layers/attention/attention.py:L862-L867 注册块（逐字）
direct_register_custom_op(
    op_name="unified_attention_with_output",
    op_func=unified_attention_with_output,
    mutates_args=["output", "output_block_scale"],
    fake_impl=unified_attention_with_output_fake,
)
