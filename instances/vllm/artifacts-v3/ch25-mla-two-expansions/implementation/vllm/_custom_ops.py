# SOURCE: vllm/_custom_ops.py
# ch25 消费面：MLA 前向的两个 CUDA cache 算子的 host 镜像——
#   concat_and_cache_mla（L2763-L2774 wrapper；kernel 本体
#   csrc/cache_kernels.cu 的 concat+scatter 语义）
#   gather_and_maybe_dequant_cache（L2914-L2936 wrapper；分页 gather 语义）
# 其余两个 gather 变体（cp_gather_cache / cp_gather_and_upconvert_fp8_
# kv_cache）是 fp8/DCP 专用（delete[0]/ch27 域），host 镜像如实抛错——
# bf16/auto 路径（本章数值测试）永不触达。
from __future__ import annotations

import torch


# SOURCE: vllm/_custom_ops.py:L2763-L2774 concat_and_cache_mla —— HOST SEAM：
#   非量化路径 = cat(kv_c, k_pe, -1) 后按 slot_mapping 散写进扁平 cache
#   （csrc concat_and_cache_mla kernel 的 bf16/auto 语义镜像；fp8 量化
#   分派归 ch27——本章 kv_cache_dtype 恒 auto/bf16）
def concat_and_cache_mla(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    scale: torch.Tensor,
) -> None:
    # SOURCE: vllm/_custom_ops.py:L2763-L2774 —— HOST SEAM
    flat = kv_cache.view(-1, kv_cache.shape[-1])
    fused = torch.cat([kv_c, k_pe], dim=-1)
    flat[slot_mapping.flatten()] = fused


# SOURCE: vllm/_custom_ops.py:L2914-L2936 gather_and_maybe_dequant_cache
#   —— HOST SEAM：分页 gather 镜像——按 cu_seq_lens/token_to_seq/seq_starts
#   把每请求 [seq_start, seq_start+len) 的潜向量行搬进 workspace 前缀
#   （csrc kernel 的非量化语义；dequant 分派归 ch27）
def gather_and_maybe_dequant_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    token_to_seq: torch.Tensor,
    num_tokens: int,
    kv_cache_dtype: str,
    scale: torch.Tensor,
    seq_starts: torch.Tensor | None = None,
) -> None:
    # SOURCE: vllm/_custom_ops.py:L2914-L2936 —— HOST SEAM
    flat = src_cache.view(-1, src_cache.shape[-1])
    bs = src_cache.shape[1]  # block_size
    cu = cu_seq_lens.tolist()
    starts = seq_starts.tolist() if seq_starts is not None else [0] * (
        len(cu) - 1
    )
    for r in range(len(cu) - 1):
        n = cu[r + 1] - cu[r]
        s0 = starts[r]
        for t in range(n):
            pos = s0 + t
            blk = int(block_table[r, pos // bs])
            dst[cu[r] + t] = flat[blk * bs + pos % bs]


# SOURCE: vllm/_custom_ops.py:L2951 cp_gather_and_upconvert_fp8_kv_cache
#   —— HOST SEAM：fp8_ds_mla 打包布局的 gather+上变换（CUDA-only），
#   host 镜像如实抛错（bf16/auto 路径走 gather_and_maybe_dequant_cache 支）
def cp_gather_and_upconvert_fp8_kv_cache(*args, **kwargs) -> None:
    # SOURCE: vllm/_custom_ops.py:L2951 —— HOST SEAM（CUDA-only）
    raise NotImplementedError(
        "cp_gather_and_upconvert_fp8_kv_cache is a CUDA-only op for the "
        "fp8_ds_mla packed layout (ch27 domain)."
    )


# SOURCE: vllm/_custom_ops.py:L2938 cp_gather_cache —— HOST SEAM：
#   fp8 prefill 的免反量化 gather（CUDA-only），host 镜像如实抛错
def cp_gather_cache(*args, **kwargs) -> None:
    # SOURCE: vllm/_custom_ops.py:L2938 —— HOST SEAM（CUDA-only）
    raise NotImplementedError(
        "cp_gather_cache is a CUDA-only op for the FP8 prefill path."
    )
