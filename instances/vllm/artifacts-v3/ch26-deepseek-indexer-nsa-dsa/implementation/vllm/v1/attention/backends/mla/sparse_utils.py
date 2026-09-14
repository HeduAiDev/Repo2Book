# SOURCE: vllm/v1/attention/backends/mla/sparse_utils.py
# ch26 切面（m07）：triton_convert_req_index_to_global_index —— index 空间换算
# 的契约（out = block_table[req, idx//BLOCK]*BLOCK + idx%BLOCK；-1 直通；
# 越界块 → -1；prefill 映 workspace 偏移；return_valid_counts 同程计数）。
# HOST SEAM：真实实现为 @triton.jit kernel（_convert_req_index_to_global_
# index_kernel L11-L117）+ wrapper（L120-L236）；host 以同签名纯 torch 镜像
# 承载精确数学（kernel 体见真实源锚）。
# SUBTRACTED：triton_filter_and_convert_dcp_index（L239-L343）——DCP 路径
# （delete[0] 同族；深讲归 ch34）。
from __future__ import annotations

import torch


# SOURCE: vllm/v1/attention/backends/mla/sparse_utils.py:L120-L236
#   triton_convert_req_index_to_global_index —— HOST SEAM 参考数学
def triton_convert_req_index_to_global_index(
    req_id: torch.Tensor,  # int32 [num_tokens]
    block_table: torch.Tensor,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    BLOCK_SIZE: int = 64,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,  # tile width along columns
    HAS_PREFILL_WORKSPACE: bool = False,
    prefill_workspace_request_ids: torch.Tensor | None = None,
    prefill_workspace_starts: torch.Tensor | None = None,
    return_valid_counts: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    out[token_id, indice_id] =
        block_table[req_id[token_id],
            token_indices[token_id, indice_id] // BLOCK_SIZE] * BLOCK_SIZE
        + token_indices[token_id, indice_id] % BLOCK_SIZE

    Only when token_indices[token_id, indice_id] == -1 do we output -1.
    For safety, we also output -1 if the derived block_id would be
        out-of-bounds.

    When HAS_PREFILL_WORKSPACE is True, prefill tokens are mapped to workspace offsets
    instead of global cache slots. prefill_workspace_request_ids and
    prefill_workspace_starts must be provided.

    prefill_workspace_request_ids: int32 [num_tokens], -1 for decode else
        prefill request index (maps to prefill_workspace_starts)
    prefill_workspace_starts: int32 [num_prefills], 0-indexed workspace
        starts for each prefill request

    When return_valid_counts is True, also returns the count of valid (non -1)
    indices per row, computed during the same kernel pass (no extra overhead).
    """
    # SOURCE: vllm/v1/attention/backends/mla/sparse_utils.py:L120-L236 —— HOST
    #   SEAM（kernel L11-L117 的逐式：-1 直通/越界块守卫/prefill 偏移覆盖）
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert req_id.shape[0] == token_indices.shape[0], (
        f"req_id ({req_id.shape[0]}) and token_indices ({token_indices.shape[0]}) "
        "must cover the same tokens; the grid is sized by req_id but the output "
        "is allocated like token_indices, so a longer req_id writes out of bounds"
    )
    assert token_indices.shape[1] == NUM_TOPK_TOKENS

    if HAS_PREFILL_WORKSPACE:
        assert prefill_workspace_request_ids is not None
        assert prefill_workspace_starts is not None
        assert prefill_workspace_request_ids.dtype == torch.int32
        assert prefill_workspace_starts.dtype == torch.int32

    num_tokens = req_id.shape[0]
    max_num_blocks_per_req = block_table.shape[1]

    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()
    out = torch.empty_like(token_indices_c)

    valid_counts: torch.Tensor | None = None
    if return_valid_counts:
        valid_counts = torch.zeros(
            num_tokens, dtype=torch.int32, device=token_indices.device
        )

    for token_id in range(num_tokens):
        req = int(req_id_c[token_id].item())
        is_prefill = False
        prefill_req_id = -1
        if HAS_PREFILL_WORKSPACE:
            prefill_req_id = int(prefill_workspace_request_ids[token_id].item())
            is_prefill = prefill_req_id >= 0
        count = 0
        for col in range(NUM_TOPK_TOKENS):
            tok = int(token_indices_c[token_id, col].item())
            is_invalid = tok < 0
            block_id = 0
            inblock_off = 0
            if not is_invalid:
                block_id = tok // BLOCK_SIZE
                inblock_off = tok % BLOCK_SIZE
                if block_id >= max_num_blocks_per_req or block_id < 0:
                    is_invalid = True
            if is_invalid:
                out[token_id, col] = -1
                continue
            if is_prefill:
                workspace_start = int(prefill_workspace_starts[prefill_req_id].item())
                out[token_id, col] = workspace_start + tok
            else:
                base = int(block_table_c[req, block_id].item())
                out[token_id, col] = base * BLOCK_SIZE + inblock_off
            count += 1
        if valid_counts is not None:
            valid_counts[token_id] = count

    if return_valid_counts:
        assert valid_counts is not None
        return out, valid_counts
    return out


# SUBTRACTED: triton_filter_and_convert_dcp_index（sparse_utils.py:L239-L343）
#   ——DCP rank 局部槽过滤 + 前缀压实（compact_valid_to_front 的原子槽分配）
#   ——delete[0] DCP 族；深讲归 ch34
