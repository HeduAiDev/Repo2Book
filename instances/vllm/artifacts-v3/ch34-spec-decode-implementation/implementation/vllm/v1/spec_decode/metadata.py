# SOURCE: vllm/v1/spec_decode/metadata.py —— 本章主角文件之一（零删减，逐字）
# SpecDecodeMetadata：摊平后的草稿与三组 index 的容器（runner↔rejection_sampler
# 的契约）。draft_token_ids 是全批变长草稿摊平成一维；cu_num_* 是含末项的累积和
# （Triton kernel 用 [start,end) 反推每请求区间）；target/bonus/logits 三组 index
# 把 target 一次前向产出的扁平 logits 重新定位到每个草稿位与每个 bonus 位。
#
# 精简版范围注记（非源码内容，正文以内嵌真源码解读、不做精简版——
# dossier subtraction_plan.delete[5]：跨进程编排+持久批依赖，单文件减法无法保持
# 可运行性）：草稿的所有权环各站在真实源码中的锚点为
#   worker 产草稿   gpu_model_runner.propose_draft_token_ids
#                  (vllm/v1/worker/gpu_model_runner.py:L5010-L5290)
#   同步回程信封   executor.take_draft_token_ids → DraftTokenIds
#                  (vllm/v1/engine/core.py:L616-L623 · vllm/v1/outputs.py:L338)
#   调度器挂账     scheduler.update_draft_token_ids
#                  (vllm/v1/core/sched/scheduler.py:L2146-L2167)
#   排批通道       SchedulerOutput.scheduled_spec_decode_tokens /
#                  num_spec_tokens_to_schedule（动态 K 查表，
#                  vllm/v1/core/sched/scheduler.py:L1192-L1197)
# 三组 index 的构造算术（_calc_spec_decode_metadata 的 np.repeat+arange 与
# 二次 gather）在 gpu_model_runner.py:L2851-L2924，同样只作正文内嵌。
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import numpy as np
import torch


# SOURCE: vllm/v1/spec_decode/metadata.py:L9-L30 SpecDecodeMetadata —— 逐字
@dataclass
class SpecDecodeMetadata:
    # [num_tokens]
    draft_token_ids: torch.Tensor
    # [batch_size]
    num_draft_tokens: list[int]
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor
    # [batch_size]
    cu_num_sampled_tokens: torch.Tensor
    # [num_tokens]
    target_logits_indices: torch.Tensor
    # [batch_size]
    bonus_logits_indices: torch.Tensor
    # [num_tokens + batch_size]
    logits_indices: torch.Tensor

    def __post_init__(self):
        # SOURCE: vllm/v1/spec_decode/metadata.py:L26-L27 __post_init__ —— 逐字
        self.max_spec_len = max(self.num_draft_tokens)

    @classmethod
    def make_dummy(
        cls,
        draft_token_ids: list[list[int]],
        device: torch.device,
    ) -> "SpecDecodeMetadata":
        # SOURCE: vllm/v1/spec_decode/metadata.py:L29-L66 make_dummy —— 逐字
        #   （warmup/cudagraph 预热：真实草稿形状、index 全零占位）
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
