# Subtract-only companion for v3 ch21 — vllm/_custom_ops.py（切面）
# (pin v0.27.1 / 6e448d0ea). 本章切面：reshape_and_cache_flash thin wrapper
# （写腿算子——转发 _C_cache_ops CUDA kernel）+ 该 kernel 本体的 host 镜像
# （csrc/libtorch_stable/cache_kernels.cu:L315-L344，逐 token 语义：slot<0
# 跳过 + slot//block_size、slot%block_size 逆分解 + 行拷贝）。
# 其余 ~120 个 custom op → 各 kernel 章（ch13/ch20 等），章界收窄。
from __future__ import annotations

import torch


# SOURCE: vllm/_custom_ops.py:L2614-L2633 reshape_and_cache_flash ——（逐字
#   转发面；torch.ops._C_cache_ops 在 host 无 CUDA 扩展——转发目标由下方
#   HOST SEAM 镜像承载，同一签名与逐 token 语义）
def reshape_and_cache_flash(  # SOURCE: vllm/_custom_ops.py:L2614
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    _reshape_and_cache_flash_host_mirror(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )


# ── HOST SEAM：cache_kernels.cu kernel 本体的逐 token 镜像 ─────────────────
# SOURCE: csrc/libtorch_stable/cache_kernels.cu:L315-L344 reshape_and_cache_
#   flash kernel（每 token 一线程：slot 逆分解定位 (block_idx, block_offset)
#   后行拷贝 K/V）。host 镜像以 python 循环承载同语义；"auto" dtype 无缩放、
#   向量化拷贝退化为行赋值。
def _reshape_and_cache_flash_host_mirror(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
) -> None:
    # SOURCE: csrc/libtorch_stable/cache_kernels.cu:L329-L331 PAD 消费端
    #   （'slot_idx can be -1 if the token is padded' → 该 token 线程直接
    #   return；循环镜像里等价于 continue）
    for token_idx in range(slot_mapping.shape[0]):
        slot_idx = int(slot_mapping[token_idx])
        # NOTE: slot_idx can be -1 if the token is padded
        if slot_idx < 0:
            continue  # HOST SEAM：kernel 的 per-thread return 之循环镜像
        # SOURCE: csrc/libtorch_stable/cache_kernels.cu:L332-L333 slot 逆分解
        #   （block_idx = slot//block_size、block_offset = slot%block_size）
        block_idx = slot_idx // key_cache.shape[1]
        block_offset = slot_idx % key_cache.shape[1]
        # SOURCE: csrc/libtorch_stable/cache_kernels.cu:L336-L344 源/目的行拷贝
        key_cache[block_idx, block_offset] = key[token_idx]
        value_cache[block_idx, block_offset] = value[token_idx]


# SUBTRACTED: vllm/_custom_ops.py 其余 custom op 面（L1-L2613 / L2636 起）——
#   各 op 属 kernel/量化/MoE 域（ch13/ch20/ch27/ch33），本章零调用。
