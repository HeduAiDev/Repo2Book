# SOURCE: vllm/v1/attention/ops/common.py
# HOST SEAM（B1 Triton kernel 镜像·精确数学，ch25 同款骨架）：pack_seq_triton
# / unpack_seq_triton（decode 打分的 pack/unpack——变长 decode_lens 时 pad 到
# 统一 next_n；pad 值 -inf（fp32 pad 路）由下游 context_lens mask）。
from __future__ import annotations

import torch


# SOURCE: vllm/v1/attention/ops/common.py:L315-… pack_seq_triton —— HOST SEAM
#   参考数学（x [N,…] + lengths [B] → [B, Lmax, …]）
def pack_seq_triton(
    x: torch.Tensor,
    lengths: torch.Tensor,
    pad_value: float | int = -float("inf"),
    block_t: int = 64,
    block_d: int = 64,
) -> torch.Tensor:
    """Pack sequences of different lengths into a batched tensor.

    Supports float dtypes (any, via fp32 pad) and ``torch.uint8`` (exact-byte
    pad — e.g. MXFP4 packed nibbles or ue8m0 scale bytes). For uint8 inputs
    ``pad_value`` must be an integer in ``[0, 255]``.

    Args:
        x: [N, ...] — input tensor where N is total number of tokens.
        lengths: [B] — sequence lengths for each batch.
        pad_value: value to use for padding. Defaults to ``-inf`` which is
            only sensible for float dtypes; pass ``0`` (or any byte) for
            uint8 inputs.
        block_t: block size for time dimension.
        block_d: block size for feature dimension.

    Returns:
        packed: [B, Lmax, ...] — packed tensor.
    """
    # SOURCE: vllm/v1/attention/ops/common.py:L315-… —— HOST SEAM
    lens = lengths.tolist()
    Lmax = max(lens) if lens else 0
    packed = torch.full(
        (len(lens), Lmax, *x.shape[1:]), float(pad_value) if x.dtype.is_floating_point else int(pad_value),
        dtype=x.dtype)
    row = 0
    for b, n in enumerate(lens):
        packed[b, :n] = x[row:row + n]
        row += n
    return packed


# SOURCE: vllm/v1/attention/ops/common.py:L429-… unpack_seq_triton —— HOST
#   SEAM 参考数学
def unpack_seq_triton(
    packed_tensor: torch.Tensor,
    lengths: torch.Tensor,
    block_t: int = 64,
    block_d: int = 64,
) -> torch.Tensor:
    """
    Unpack a packed decode query tensor back to the original format.
    Efficient Triton implementation.

    Args:
        packed_tensor: [B, Lmax, ...] - packed tensor from pack_seq_triton
        lengths: [B] - sequence lengths for each batch

    Returns:
        unpacked_tensor: [N, ...] where N = sum(lengths)
    """
    # SOURCE: vllm/v1/attention/ops/common.py:L429-… —— HOST SEAM
    lens = lengths.tolist()
    total = sum(lens)
    out_shape = (total, *packed_tensor.shape[2:])
    out = torch.empty(out_shape, dtype=packed_tensor.dtype)
    row = 0
    for b, n in enumerate(lens):
        out[row:row + n] = packed_tensor[b, :n]
        row += n
    return out
