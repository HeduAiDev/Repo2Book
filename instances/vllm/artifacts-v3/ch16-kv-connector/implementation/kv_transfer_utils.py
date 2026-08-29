# SOURCE: vllm/model_executor/layers/attention/kv_transfer_utils.py
# maybe_transfer_kv_layer 装饰器（m8——逐层重叠的挂点，长在模型层而非
# runner）：层执行前 wait_for_layer_load（阻塞到该层 KV 到位）、执行后
# save_kv_layer（异步存出）；无 kv_transfer_group / 无 metadata 时零开销
# 直通。签名只检一次（装饰时）。
# SUBTRACTED: 无（61 行全保——契约最深的挂点整面进本章）。
import inspect
from collections.abc import Callable
from functools import wraps

from .kv_transfer_state import (
    get_kv_transfer_group,
    has_kv_transfer_group,
    is_v1_kv_transfer_group,
)
from .torch_utils import _resolve_layer_name
from .attention import get_attention_context


# SOURCE: vllm/model_executor/layers/attention/kv_transfer_utils.py:L15
#   maybe_transfer_kv_layer
def maybe_transfer_kv_layer(func: Callable) -> Callable:
    """Decorator that handles KV layer transfer prior and after execution of
    an attention layer, if enabled. Otherwise, the wrapper is a no-op.

    On entry: waits for the KV layer from the connector.
    On exit: saves the KV layer to the connector.
    """
    # Import at runtime to avoid circular dependency
    # SOURCE: vllm/model_executor/layers/attention/kv_transfer_utils.py:L22-L23
    #   （get_attention_context 的延迟导入在本镜像内为模块级相对导入）

    # Inspect the signature ONCE when the decorator is applied.
    # SOURCE: vllm/model_executor/layers/attention/kv_transfer_utils.py:L25-L27
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())

    # Find the index of 'layer_name' parameter.
    # SOURCE: vllm/model_executor/layers/attention/kv_transfer_utils.py:L29-L35
    try:
        layer_name_index = param_names.index("layer_name")
    except ValueError as e:
        raise TypeError(
            f"Function {func.__name__} must have a 'layer_name' parameter"
        ) from e

    # SOURCE: vllm/model_executor/layers/attention/kv_transfer_utils.py:L37-L38
    @wraps(func)
    def wrapper(*args, **kwargs):
        # SOURCE: vllm/model_executor/layers/attention/kv_transfer_utils.py:L39-L40
        #   （零开销旁路之一：无 kv_transfer_group）
        if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
            return func(*args, **kwargs)

        # SOURCE: vllm/model_executor/layers/attention/kv_transfer_utils.py:L42
        layer_name = _resolve_layer_name(args[layer_name_index])

        # Extract attention context (metadata, layer, kv_cache, layer_slot_mapping)
        # SOURCE: vllm/model_executor/layers/attention/kv_transfer_utils.py:L44-L47
        attn_metadata, _, kv_cache, _ = get_attention_context(layer_name)
        connector = get_kv_transfer_group()
        if attn_metadata is None or not connector.has_connector_metadata():
            return func(*args, **kwargs)

        # Wait for KV layer on entry
        # SOURCE: vllm/model_executor/layers/attention/kv_transfer_utils.py:L50-L51
        connector.wait_for_layer_load(layer_name)

        # Execute the function
        # SOURCE: vllm/model_executor/layers/attention/kv_transfer_utils.py:L53-L54
        result = func(*args, **kwargs)

        # Save KV cache layer on exit
        # SOURCE: vllm/model_executor/layers/attention/kv_transfer_utils.py:L56-L57
        connector.save_kv_layer(layer_name, kv_cache, attn_metadata)

        return result

    # SOURCE: vllm/model_executor/layers/attention/kv_transfer_utils.py:L59
    return wrapper
