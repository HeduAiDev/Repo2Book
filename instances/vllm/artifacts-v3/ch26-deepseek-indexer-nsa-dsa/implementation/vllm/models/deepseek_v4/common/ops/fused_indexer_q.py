# SOURCE: vllm/models/deepseek_v4/common/ops/fused_indexer_q.py
# ch26 切面（m15）：MXFP4_BLOCK_SIZE 常量（逐字——FP4 布局：32 值/块、2
# nibble/字节、ue8m0 块 scale）+ fused_indexer_q_rope_quant 的 HOST SEAM 镜像
#（真实为 @triton.jit 核：GPT-J interleaved RoPE 打每头**末** rope_dim 维 +
# 前 nope 段直通 + FP8/FP4 量化 + weights 折 scale；V4 新布局的融合路径——
# 与 V3.2 扁平版同数学不同布局）。
# SUBTRACTED：_fp32x2_to_fp4x2/_quantize_mxfp4_pair 的 inline PTX 打包——
#   delete[2]（FP4 分支；FP8 布局讲解用常量与真实源码 excerpt 承载）。
from __future__ import annotations

import torch

# MXFP4: 32 elements per block, packed 2 nibbles per byte, ue8m0 block scale.
# SOURCE: vllm/models/deepseek_v4/common/ops/fused_indexer_q.py:L10
#   MXFP4_BLOCK_SIZE —— 逐字
MXFP4_BLOCK_SIZE = 32


# SOURCE: vllm/models/deepseek_v4/common/ops/fused_indexer_q.py
#   fused_indexer_q_rope_quant —— HOST SEAM 参考数学（FP8 臂）
def fused_indexer_q_rope_quant(
    positions: torch.Tensor,
    index_q: torch.Tensor,  # [T, H, head_dim] bf16
    index_q_cos_sin_cache: torch.Tensor,  # [max_pos, rope_dim]
    index_weights: torch.Tensor,  # [T, H]
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
    use_fp4: bool = False,
    output_buffers=None,
    fp8_max: float = 448.0,
):
    # SOURCE: vllm/models/deepseek_v4/common/ops/fused_indexer_q.py
    #   _fused_indexer_q_rope_quant_kernel —— HOST SEAM 逐式：
    #   布局与 unfused 参考一致（DeepseekV4ScalingRotaryEmbedding +
    #   per_token_group_quant_fp8）：GPT-J interleaved RoPE 打每头 LAST
    #   rope_dim 维；前 [0, nope) 直通。
    assert not use_fp4, "FP4 path is subtracted (delete[2])"
    T, H, head_dim = index_q.shape
    rope_dim = index_q_cos_sin_cache.shape[-1]
    nope_dim = head_dim - rope_dim
    out_q = torch.empty_like(index_q, dtype=torch.float8_e4m3fn)
    out_w = torch.empty_like(index_weights, dtype=torch.float32)
    for t in range(T):
        pos = int(positions[t].item())
        cos = index_q_cos_sin_cache[pos, : rope_dim // 2].float()
        sin = index_q_cos_sin_cache[pos, rope_dim // 2:].float()
        for h in range(H):
            x = index_q[t, h].float()
            q_nope = x[:nope_dim]
            xr_even = x[nope_dim::2]
            xr_odd = x[nope_dim + 1 :: 2]
            r_even = xr_even * cos - xr_odd * sin
            r_odd = xr_odd * cos + xr_even * sin
            # bf16 round-trip（核内 .to(bfloat16).to(float32) 的参考路径）
            r_even = r_even.to(torch.bfloat16).float()
            r_odd = r_odd.to(torch.bfloat16).float()
            amax = max(float(q_nope.abs().max()), float(r_even.abs().max()),
                       float(r_odd.abs().max()))
            scale = 2.0 ** _ceil(_log2(max(amax, 1e-10) / fp8_max))
            row = torch.empty(head_dim)
            row[:nope_dim] = torch.clamp(q_nope / scale, -fp8_max, fp8_max)
            row[nope_dim::2] = torch.clamp(r_even / scale, -fp8_max, fp8_max)
            row[nope_dim + 1 :: 2] = torch.clamp(r_odd / scale, -fp8_max, fp8_max)
            out_q[t, h] = row.to(torch.float8_e4m3fn)
            out_w[t, h] = (float(index_weights[t, h].item()) * scale
                           * index_weights_softmax_scale
                           * index_weights_head_scale)
    return out_q, out_w


import math as _math  # noqa: E402

_ceil = _math.ceil
_log2 = _math.log2
