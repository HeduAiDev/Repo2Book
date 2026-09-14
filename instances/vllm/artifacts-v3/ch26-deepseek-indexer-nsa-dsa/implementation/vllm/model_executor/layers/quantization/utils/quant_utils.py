# SOURCE: vllm/model_executor/layers/quantization/utils/quant_utils.py
# ch26 消费面：get_fp8_min_max（e4m3 量程）+ GroupShape + scaled_dequantize
# （_try_load_fp8_indexer_wk 的 FP8→BF16 反量化数学——逐 token/逐块组广播）。
from __future__ import annotations

from typing import NamedTuple

import torch


# SOURCE: vllm/model_executor/layers/quantization/utils/quant_utils.py:L43-L52
#   GroupShape —— NamedTuple 承载（(group_rows, group_cols) 可下标）
# SOURCE: quant_utils.py:L43-L52（锚点双置）
class GroupShape(NamedTuple):
    """This class describes the quantization group shape.
    It includes static members for common shapes (per-tensor, per-token).
    """

    group_rows: int
    group_cols: int


# SOURCE: vllm/model_executor/layers/quantization/utils/quant_utils.py
#   get_fp8_min_max —— HOST SEAM（e4m3 量程 ±448）
def get_fp8_min_max():
    # SOURCE: vllm/model_executor/layers/quantization/utils/quant_utils.py
    return (-448.0, 448.0)


# SOURCE: vllm/model_executor/layers/quantization/utils/quant_utils.py:L415-L426
#   scaled_dequantize —— HOST SEAM 参考数学（组广播反量化）
def scaled_dequantize(
    x_q: torch.Tensor,
    x_s: torch.Tensor,
    group_shape: GroupShape | None = None,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    # SOURCE: vllm/model_executor/layers/quantization/utils/quant_utils.py
    #   :L415-L426 —— HOST SEAM（prep_scale_for_group_broadcast/group_broadcast
    #   的组广播数学承载）
    if group_shape is not None and x_s.dim() == 1:
        # prep_scale_for_group_broadcast：1-D per-channel scale → [-1, 1]
        x_s = x_s.unsqueeze(-1)
    if group_shape is not None:
        assert x_s.shape[-1] == x_q.shape[-1] // group_shape[1]
        assert x_s.shape[-2] == x_q.shape[-2] // group_shape[0]
        reps = (x_q.shape[-2] // x_s.shape[-2], x_q.shape[-1] // x_s.shape[-1])
        x_s = x_s.repeat_interleave(reps[0], dim=-2).repeat_interleave(
            reps[1], dim=-1)
    return (x_q.to(torch.float32) * x_s.to(torch.float32)).to(out_dtype)
