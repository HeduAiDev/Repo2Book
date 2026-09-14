# SOURCE: vllm/model_executor/layers/quantization/utils/fp8_utils.py
# HOST SEAM（B1 精确数学）：per_token_group_quant_fp8 的 host 参考镜像。
# 真实实现优先 CUDA kernel；host 镜像承载同一数学（Indexer.forward 的
# 「we only quant q here since k quant is fused with cache insertion」位）：
#   scale = exp2(ceil(log2(max(amax, eps) / fp8_max)))   （use_ue8m0=True，
#   与 _fused_indexer_q_rope_quant_kernel L176-L178 同式）
#   q_fp8 = clamp(x / scale, -fp8_max, fp8_max).to(float8_e4m3fn)
from __future__ import annotations

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform


# SOURCE: vllm/model_executor/layers/quantization/utils/fp8_utils.py:L533-…
#   per_token_group_quant_fp8 —— HOST SEAM 参考数学
def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype | None = None,
    column_major_scales: bool = False,
    tma_aligned_scales: bool = False,
    out_q: torch.Tensor | None = None,
    use_ue8m0: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.
    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.
    Args:
        x: The input tensor with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dtype of output tensor. Note that only `torch.float8_e4m3fn`
        is supported for now.
        column_major_scales: Outputs scales in column major.
        tma_aligned_scales: Outputs scales in TMA-aligned layout.
        out_q: Optional output tensor. If not provided, function will create.
        use_ue8m0: Whether to produce power-of-two scales (ue8m0 format).
    Returns:
        tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the
        scaling factor.
    """
    # SOURCE: vllm/model_executor/layers/quantization/utils/fp8_utils.py:L533-…
    #   —— HOST SEAM 参考数学（ue8m0 幂次 scale 与 fused 核同式）
    dtype = current_platform.fp8_dtype() if dtype is None else dtype
    assert x.shape[-1] % group_size == 0, (
        f"the last dimension of `x` {x.shape} must be divisible "
        f"by `group_size` {group_size}"
    )
    assert x.stride(-1) == 1, "`x` groups must be contiguous"

    fp8_min, fp8_max = get_fp8_min_max()
    assert out_q is None or out_q.shape == x.shape
    x_q = out_q if out_q is not None else torch.empty(
        x.shape, device=x.device, dtype=dtype)
    x_f = x.float()
    amax = x_f.abs().amax(dim=-1, keepdim=True)
    if use_ue8m0:
        # e8m0 format: scale = 2^ceil(log2(max(amax, eps) / fp8_max))
        scale = torch.exp2(
            torch.ceil(torch.log2(torch.clamp(amax, min=eps) / fp8_max)))
    else:
        scale = torch.clamp(amax, min=eps) / fp8_max
    x_q.copy_(
        torch.clamp(x_f / scale, fp8_min, fp8_max).to(dtype))
    x_s = scale.squeeze(-1)
    if column_major_scales:
        x_s = x_s.t().contiguous().t()
    return x_q, x_s
