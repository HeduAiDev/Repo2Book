# SOURCE: vllm/model_executor/utils.py
# ch25 消费面：replace_parameter（process_weights_after_loading 的
# W_UV/W_UK_T 重排落点，must_keep）——逐字。
from __future__ import annotations

import torch


# SOURCE: vllm/model_executor/utils.py:L47 replace_parameter —— 逐字
def replace_parameter(
    layer: torch.nn.Module,
    param_name: str,
    new_data: torch.Tensor | None,
    prefer_copy: bool = False,
):
    """
    Replace a parameter of a layer while maintaining the ability to reload the weight.
    Called within implementations of the `process_weights_after_loading` method.

    This function should not be called on weights which are tied/shared

    Args:
        layer: Layer containing parameter to replace
        param_name: Name of parameter to replace
        new_data: New data of the new parameter, or None to set the parameter to None
        prefer_copy: If True and the existing parameter is compatible with
            ``new_data`` (same shape, dtype, and device), copy ``new_data``
            into the existing parameter in place rather than re-registering
            a new parameter. This preserves the parameter's storage address
            (``data_ptr``), which is required for captured CUDA graphs to
            remain valid across weight updates (e.g. in RL training loops).
    """
    # should not be used on a tied/shared param

    # If new_data is None, set the parameter to None
    if new_data is None:
        setattr(layer, param_name, None)
        return

    if isinstance(new_data, torch.nn.Parameter):
        new_data = new_data.data

    old_param: torch.nn.Parameter | None = getattr(layer, param_name, None)

    if (
        prefer_copy
        and old_param is not None
        and old_param.shape == new_data.shape
        and old_param.dtype == new_data.dtype
        and old_param.device == new_data.device
    ):
        # SOURCE: vllm/model_executor/utils.py prefer_copy 支 —— 逐字
        old_param.data.copy_(new_data)
        return

    new_param = torch.nn.Parameter(new_data, requires_grad=False)

    # SUBTRACTED: old_param.weight_loader 保留位（vllm/model_executor/
    #   utils.py:L88-L90）——weight_loader 面归 ch23（本包 seam Linear 无
    #   loader 属性，逐字语义不受影响）

    setattr(layer, param_name, new_param)
