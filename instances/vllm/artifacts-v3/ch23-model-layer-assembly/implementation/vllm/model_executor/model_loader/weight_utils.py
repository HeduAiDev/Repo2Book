# SOURCE: vllm/model_executor/model_loader/weight_utils.py
# ch23 消费面：default_weight_loader（AutoWeightsLoader._load_param 的兜底
# 装载器）。SUBTRACTED：checkpoint IO 族（safetensors 迭代/量化过滤器/
# row_parallel_weight_loader 等）——default_loader.get_all_weights 的域。
from __future__ import annotations

import torch


# SOURCE: vllm/model_executor/model_loader/weight_utils.py:L1231
#   default_weight_loader（逐字）
def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Default weight loader."""
    # SOURCE: vllm/model_executor/model_loader/weight_utils.py:L1231
    try:
        if param.numel() == 1 and loaded_weight.numel() == 1:
            # Sometimes scalar values aren't considered tensors with shapes
            # so if both param and loaded_weight are a scalar,
            # reshape to match before copying
            param.data.copy_(loaded_weight.view(param.shape))
        else:
            assert param.size() == loaded_weight.size(), (
                f"Attempted to load weight ({loaded_weight.size()}) "
                f"into parameter ({param.size()})"
            )

            param.data.copy_(loaded_weight)
    except Exception:
        # NOTE: This exception is added for the purpose of setting breakpoint to
        # debug weight loading issues.
        raise
