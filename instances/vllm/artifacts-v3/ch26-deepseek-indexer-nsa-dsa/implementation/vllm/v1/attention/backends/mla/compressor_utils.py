# SOURCE: vllm/v1/attention/backends/mla/compressor_utils.py
# ch26 切面（m11）：get_compressed_slot_mapping —— 压缩坐标系的 slot 换算
# （(pos+1)%compress_ratio==0 的 token 才有效；pos//ratio 经 block_table 落
# 物理压缩 slot；无效 → -1）。HOST SEAM：真实为 @triton.jit kernel
# （_compressed_slot_mapping_kernel L9-L51）+ wrapper（L53-L86）；host 以同
# 签名纯 torch 镜像承载精确数学。
from __future__ import annotations

import torch


# SOURCE: vllm/v1/attention/backends/mla/compressor_utils.py:L53-L86
#   get_compressed_slot_mapping —— HOST SEAM 参考数学
def get_compressed_slot_mapping(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    # SOURCE: vllm/v1/attention/backends/mla/compressor_utils.py:L53-L86
    #   —— HOST SEAM（kernel L9-L51 逐式）
    if out is not None:
        # Guard: for padded / invalid sequences.
        # Negative positions produce bogus block indices that lead to illegal memory
        # accesses inside the block_table load.
        # NOTE: Fill -1 to the whole tensor, not just the first `num_tokens`.
        out.fill_(-1)
        slot_mapping = out[:num_tokens]
    else:
        slot_mapping = torch.full(
            (num_tokens,), -1, dtype=torch.int64, device=query_start_loc.device
        )

    num_reqs = block_table.shape[0]
    qsl = query_start_loc.tolist()
    for batch_idx in range(num_reqs):
        query_start = int(qsl[batch_idx])
        query_end = int(qsl[batch_idx + 1])
        query_len = query_end - query_start

        seq_len = int(seq_lens[batch_idx].item())
        start_pos = seq_len - query_len

        for offset in range(query_len):
            pos = start_pos + offset
            is_valid = (pos + 1) % compress_ratio == 0
            if not is_valid:
                continue
            pos_after_compress = pos // compress_ratio
            block_id = pos_after_compress // block_size
            block_number = int(block_table[batch_idx, block_id].item())
            slot_mapping[query_start + offset] = (
                block_number * block_size + pos_after_compress % block_size)
    return slot_mapping
