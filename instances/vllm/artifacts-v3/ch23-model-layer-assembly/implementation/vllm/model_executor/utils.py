# SOURCE: vllm/model_executor/utils.py:L13-L41 set_weight_attrs
# 权重张量属性戳：weight_loader/input_dim/output_dim 的挂载面
from __future__ import annotations

from typing import Any

import torch


# SOURCE: vllm/model_executor/utils.py:L13 set_weight_attrs
def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: dict[str, Any] | None,
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(weight, key), f"Overwriting existing tensor attribute: {key}"

        # NOTE(woosuk): During weight loading, we often do something like:
        # narrowed_tensor = param.data.narrow(0, offset, len)
        # narrowed_tensor.copy_(real_weight)
        # expecting narrowed_tensor and param.data to share the same storage.
        # However, on TPUs, narrowed_tensor will lazily propagate to the base
        # tensor, which is param.data, leading to the redundant memory usage.
        # This sometimes causes OOM errors during model loading. To avoid this,
        # we sync the param tensor after its weight loader is called.
        # TODO(woosuk): Remove this hack once we have a better solution.
        # SUBTRACTED: TPU 同步 weight_loader 包装（utils.py:L34-L38）——TPU 平台
        #   特例；host/CUDA 路径 use_sync_weight_loader()=False 原样 setattr
        setattr(weight, key, value)
