"""ch24 精简版 — v1 统一注意力层 + forward_context 分发（只做减法，f18 回收）。

对应 vllm/model_executor/layers/attention/attention.py：
  - Attention.__init__：调 get_attn_backend 选后端、get_impl_cls 实例化 impl，并以 prefix
    (=layer_name) 注册进 static_forward_context（f18『后端选择』那一头）。
  - Attention.forward：reshape q/k/v → unified_kv_cache_update + unified_attention_with_output
    两个自定义算子分发。
  - get_attention_context(layer_name)：按 layer_name 从 forward_context 取本层
    kv_cache/attn_metadata/slot_mapping（f18『取数』那一头）。
  - unified_kv_cache_update（写）/ unified_attention_with_output（算+读）。

命名/控制流与真实一致。host 用模块级 forward_context 替身复现「model_runner 那头按 layer_name
装料、backend 这头按 layer_name 消费」。
"""

from __future__ import annotations

import torch

from selector import get_attn_backend


# ============================================================================
# forward_context 替身（对应 vllm.forward_context.ForwardContext + get_forward_context）
# ============================================================================
# SUBTRACTED: 真实 ForwardContext 是 vllm/forward_context.py 的 dataclass，由 model runner 在
# execute_model 里用 set_forward_context 上下文管理器设置，含 attn_metadata（dict[layer_name]）、
# no_compile_layers、slot_mapping（dict[layer_name]）等数十个字段。本章用一个最小可设置的全局
# 复现 f18 的取数语义：model_runner 那头按 layer_name 装料，backend 这头按 layer_name 取。
class ForwardContext:  # SOURCE: vllm/forward_context.py (ForwardContext)
    def __init__(self):  # SOURCE: vllm/forward_context.py (ForwardContext.__init__)
        self.attn_metadata = {}       # dict[layer_name -> AttentionMetadata]
        self.no_compile_layers = {}   # dict[layer_name -> Attention]
        self.slot_mapping = {}        # dict[layer_name -> slot_mapping tensor]


_FORWARD_CONTEXT = ForwardContext()


# SOURCE: vllm/forward_context.py (get_forward_context)
def get_forward_context() -> ForwardContext:
    return _FORWARD_CONTEXT


# SOURCE: vllm/model_executor/layers/attention/attention.py:L259 (Attention)
class Attention(torch.nn.Module):
    """v1 统一注意力层：每个 layer_name 各建一个。"""

    # SOURCE: vllm/model_executor/layers/attention/attention.py:L298 (__init__ backend 选择)
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        kv_cache_dtype: str = "auto",
        attn_type: str = "decoder",
        prefix: str = "",
        attn_backend=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_size = head_size
        self.head_size_v = head_size
        self.layer_name = prefix
        self.attn_type = attn_type

        # SUBTRACTED: model_config/use_mm_prefix/has_sink/量化初始化(_init_kv_cache_quant)、
        # alibi_sqrt 校验、batch-invariant prefix-caching 警告（attention.py:L260-L347）——
        # 都是配置探测/量化/告警旁支，删之不影响『选后端→实例化 impl→注册』主链。
        dtype = torch.get_default_dtype()
        if attn_backend is None:
            # f18『后端选择』那一头：选出后端类。
            self.attn_backend = get_attn_backend(
                head_size,
                dtype,
                kv_cache_dtype,
                use_mla=False,
                attn_type=attn_type,
            )
        else:
            self.attn_backend = attn_backend

        # backend.get_impl_cls() 实例化具体 impl（『抽象→具体』的桥）。
        impl_cls = self.attn_backend.get_impl_cls()
        self.impl = impl_cls(
            num_heads,
            head_size,
            scale,
            self.num_kv_heads,
            None,
            None,
            kv_cache_dtype,
            None,
            attn_type,
            None,
        )

        # SUBTRACTED: use_direct_call = not current_platform.opaque_attention_op()
        # （attention.py:L364）——CUDA/CPU 走不透明算子(False)、其他平台直调(True)。host 固定
        # 走直调路径（True），见 forward。
        self.use_direct_call = True

        # 以 prefix(=layer_name) 注册进 static_forward_context，并加入 no_compile_layers，
        # 让运行时能按 layer_name 取到本层。
        _FORWARD_CONTEXT.no_compile_layers[prefix] = self
        # self.kv_cache 由 bind_kv_cache 在初始化后绑定真实显存；本章测试里直接赋值。
        self.kv_cache = None

    # SOURCE: vllm/model_executor/layers/attention/attention.py:L409 (forward)
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        # SUBTRACTED: calculate_kv_scales(maybe_calc_kv_scales)、query_quant 量化分支
        # （attention.py:L428-L443）——量化旁支，非量化主路径不走。
        output_dtype = query.dtype
        if output_shape is None:
            num_tokens = query.shape[0]
            output_shape = torch.Size((num_tokens, self.num_heads * self.head_size_v))
        output = torch.empty(output_shape, dtype=output_dtype, device=query.device)

        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        output = output.view(-1, self.num_heads, self.head_size_v)
        if key is not None:
            key = key.view(-1, self.num_kv_heads, self.head_size)
        if value is not None:
            value = value.view(-1, self.num_kv_heads, self.head_size_v)

        kv_cache_dummy_dep = None
        # SUBTRACTED: use_direct_call=False 分支（torch.ops.vllm.* 那条，attention.py:L481-L500）
        # ——编译路径下经注册的不透明算子调度；host 固定走 use_direct_call=True 直调，语义等价。
        # 若 forward_includes_kv_cache_update=False（如 FA），写由独立算子先行；dummy_dep 串一条
        # 写→算的数据依赖、保住顺序。
        if (
            not self.attn_backend.forward_includes_kv_cache_update
            and key is not None
            and value is not None
        ):
            kv_cache_dummy_dep = unified_kv_cache_update(key, value, self.layer_name)
        unified_attention_with_output(
            query,
            key,
            value,
            output,
            self.layer_name,
            kv_cache_dummy_dep=kv_cache_dummy_dep,
        )
        hidden_size = output_shape[-1]
        return output.view(-1, hidden_size)


# SOURCE: vllm/model_executor/layers/attention/attention.py:L620 (get_attention_context)
def get_attention_context(layer_name: str):
    """f18『取数』那一头——按 layer_name 从 forward_context 取本层 attn_metadata /
    kv_cache / slot_mapping。"""
    forward_context: ForwardContext = get_forward_context()
    attn_metadata_raw = forward_context.attn_metadata
    if isinstance(attn_metadata_raw, dict):
        attn_metadata = attn_metadata_raw[layer_name]
    elif isinstance(attn_metadata_raw, list):
        # list[dict[str, AttentionMetadata]]: used in speculative decoding.
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


# SOURCE: vllm/model_executor/layers/attention/attention.py:L662 (unified_kv_cache_update)
def unified_kv_cache_update(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    """Returns a dummy that signals a side effect and the data dependency to
    unified_attention so torch.compile preserves ordering."""
    # SUBTRACTED: _resolve_layer_name 处理 LayerName/str 两形（torch_utils.py:L875）；host 用
    # 纯 str layer_name，无需解码。
    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)
    if layer_slot_mapping is not None:
        assert hasattr(attn_layer.impl, "do_kv_cache_update"), (
            f"{attn_layer.impl.__class__.__name__} does not support kv cache update"
        )
        attn_layer.impl.do_kv_cache_update(
            attn_layer,
            key,
            value,
            kv_cache,
            layer_slot_mapping,
        )
    return torch.empty(0, device=kv_cache.device, dtype=kv_cache.dtype)


# SUBTRACTED: direct_register_custom_op 把上面两个函数注册成 torch.ops.vllm.* 不透明算子
# （attention.py:L697-L703, L760-L766）+ 各自的 *_fake 实现 + maybe_transfer_kv_layer KV 传输
# 装饰器（PD 分离正交特性）。host 走直调路径，直接以 Python 函数调用复现行为，删注册壳不改语义。


# SOURCE: vllm/model_executor/layers/attention/attention.py:L705 (unified_attention_with_output)
def unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    # kv_cache_dummy_dep is not used but accepting it creates a data dependency
    # that ensures torch.compile preserves ordering between KV cache update and
    # attention forward.
    del kv_cache_dummy_dep
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
