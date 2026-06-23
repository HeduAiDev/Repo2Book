# SUBTRACTED: SPDX 版权头（vllm/v1/spec_decode/metadata.py:L1-L2）—— 仅许可证注释，不影响行为。
from dataclasses import dataclass

import numpy as np
import torch


# SOURCE: vllm/v1/spec_decode/metadata.py:L9-L66
@dataclass
class SpecDecodeMetadata:
    """把『全批变长草稿』摊平进同一批 logits 的契约容器。

    model_runner 在 _calc_spec_decode_metadata 中填好这些字段，rejection_sampler
    再据此用三组 index 做『间接定位』。所有字段与真实 vLLM 逐字一致。
    """

    # [num_tokens]：把全批变长草稿摊平成一维。
    draft_token_ids: torch.Tensor
    # [batch_size]：每请求草稿数（CPU 端 list，可能为 0）。
    num_draft_tokens: list[int]
    # [batch_size]：num_draft_tokens 的（含末项）累积和。triton kernel 据此反推
    #              每请求的草稿区间 [start, end)。
    cu_num_draft_tokens: torch.Tensor
    # [batch_size]：num_sampled_tokens(=num_draft_tokens+1) 的累积和。
    #              bonus_logits_indices = cu_num_sampled_tokens - 1 由它而来。
    cu_num_sampled_tokens: torch.Tensor
    # [num_tokens]：把扁平目标 logits 定位到每个草稿位置的 index。
    target_logits_indices: torch.Tensor
    # [batch_size]：定位每请求 bonus 位（每请求最后一行）的 index。
    bonus_logits_indices: torch.Tensor
    # [num_tokens + batch_size]：草稿位 + bonus 位的总 index（draft_token_ids
    #                            二次 gather 的第一跳）。
    logits_indices: torch.Tensor

    # SOURCE: vllm/v1/spec_decode/metadata.py:L26-L27
    def __post_init__(self):
        # 输出 buffer 第二维 max_spec_len+1 的来源，+1 即 bonus 槽。
        self.max_spec_len = max(self.num_draft_tokens)

    # SOURCE: vllm/v1/spec_decode/metadata.py:L29-L66
    @classmethod
    def make_dummy(
        cls,
        draft_token_ids: list[list[int]],
        device: torch.device,
    ) -> "SpecDecodeMetadata":
        # SOURCE: vllm/v1/spec_decode/metadata.py:L29-L66
        batch_size = len(draft_token_ids)
        num_draft_tokens = [len(ids) for ids in draft_token_ids]
        num_sampled_tokens = [len(ids) + 1 for ids in draft_token_ids]
        flattened_draft_token_ids = sum(draft_token_ids, [])
        num_tokens = len(flattened_draft_token_ids)

        draft_token_ids_tensor = torch.tensor(
            flattened_draft_token_ids, dtype=torch.int32, device=device
        )
        cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
        cu_num_draft_tokens_tensor = torch.from_numpy(cu_num_draft_tokens).to(device)
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens, dtype=np.int32)
        cu_num_sampled_tokens_tensor = torch.from_numpy(cu_num_sampled_tokens).to(
            device
        )

        target_logits_indices = torch.zeros(
            num_tokens, dtype=torch.int32, device=device
        )
        bonus_logits_indices = torch.zeros(batch_size, dtype=torch.int32, device=device)
        logits_indices = torch.zeros(
            num_tokens + batch_size, dtype=torch.int32, device=device
        )
        return cls(
            draft_token_ids=draft_token_ids_tensor,
            num_draft_tokens=num_draft_tokens,
            cu_num_draft_tokens=cu_num_draft_tokens_tensor,
            cu_num_sampled_tokens=cu_num_sampled_tokens_tensor,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )


# SOURCE: vllm/v1/worker/gpu_model_runner.py:L1572-L1596 (_get_cumsum_and_arange)
# SUBTRACTED: 这是 GpuModelRunner 的一个方法；精简版提为自由函数，去掉
#             self.arange_np / arange_out 预分配 scratch（纯性能复用），
#             直接用临时数组，算出的累积和与 batched arange 数值完全一致。
def _get_cumsum_and_arange(
    num_tokens: np.ndarray,
    cumsum_dtype: "np.dtype | None" = None,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (cumsum, batched_arange)。

    E.g. [2, 5, 3] -> cumsum=[2, 7, 10], arange=[0,1, 0,1,2,3,4, 0,1,2]。
    """
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1572-L1596
    # Step 1. [2, 5, 3] -> [2, 7, 10]
    cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
    total_num_tokens = int(cu_num_tokens[-1])
    # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
    cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
    # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
    arange = np.arange(total_num_tokens, dtype=np.int64) - cumsums_offsets
    return cu_num_tokens, arange


# SOURCE: vllm/v1/worker/gpu_model_runner.py:L2596-L2674 (_calc_spec_decode_metadata)
# SUBTRACTED: 提为自由函数；input_ids 由参数显式传入而非 self.input_ids.gpu；
#             CPU->GPU non_blocking 拷贝细节省略（device 由 input_ids 决定）。
#             三组 index 的构造算术与真实 vLLM 逐行一致。
def calc_spec_decode_metadata(
    num_draft_tokens: np.ndarray,
    cu_num_scheduled_tokens: np.ndarray,
    input_ids: torch.Tensor,
) -> SpecDecodeMetadata:
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2596-L2674
    device = input_ids.device
    # Inputs:
    # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
    # num_draft_tokens:         [  3,   0,   2,   0,   1]
    # Outputs:
    # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
    # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
    #                            206, 207, 208]
    # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
    # bonus_logits_indices:     [  3,   4,   7,   8,  10]

    # [4, 1, 3, 1, 2]
    num_sampled_tokens = num_draft_tokens + 1

    # Step 1. cu_num_sampled_tokens: [4, 5, 8, 9, 11]
    cu_num_sampled_tokens, arange_sampled = _get_cumsum_and_arange(
        num_sampled_tokens, cumsum_dtype=np.int32
    )
    # Step 2. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
    logits_indices = np.repeat(
        cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens
    )
    # Step 3. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
    logits_indices = logits_indices + arange_sampled

    # bonus = last sampled position of each request.
    bonus_logits_indices = cu_num_sampled_tokens - 1

    # cu_num_draft_tokens: [3, 3, 5, 5, 6]
    cu_num_draft_tokens, arange_draft = _get_cumsum_and_arange(
        num_draft_tokens, cumsum_dtype=np.int32
    )
    # [0, 0, 0, 5, 5, 9]
    target_logits_indices = np.repeat(
        cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens
    )
    # [0, 1, 2, 5, 6, 9]
    target_logits_indices = target_logits_indices + arange_draft

    cu_num_draft_tokens_t = torch.from_numpy(cu_num_draft_tokens).to(device)
    cu_num_sampled_tokens_t = torch.from_numpy(cu_num_sampled_tokens).to(device)
    logits_indices_t = torch.from_numpy(logits_indices.astype(np.int64)).to(device)
    target_logits_indices_t = torch.from_numpy(
        target_logits_indices.astype(np.int64)
    ).to(device)
    bonus_logits_indices_t = torch.from_numpy(bonus_logits_indices).to(device)

    # Compute the draft token ids: first gather sampled positions from the
    # input stream, then re-gather at target_logits_indices + 1 (the draft
    # token actually lives one slot after each target logit position).
    # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
    draft_token_ids = input_ids[logits_indices_t]
    draft_token_ids = draft_token_ids[target_logits_indices_t + 1]

    return SpecDecodeMetadata(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=num_draft_tokens.tolist(),
        cu_num_draft_tokens=cu_num_draft_tokens_t,
        cu_num_sampled_tokens=cu_num_sampled_tokens_t,
        target_logits_indices=target_logits_indices_t,
        bonus_logits_indices=bonus_logits_indices_t,
        logits_indices=logits_indices_t,
    )
