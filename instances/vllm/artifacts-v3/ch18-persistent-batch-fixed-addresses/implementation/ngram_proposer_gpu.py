# SOURCE: vllm/v1/spec_decode/ngram_proposer_gpu.py
# 本章消费面（m10）：update_scheduler_for_invalid_drafts —— 可变性裁决的
# 就地裁剪实现（全代码库罕见的『worker 改写调度器输出』点）；调用位在
# gpu_model_runner._update_states L1346-L1351（ngram-GPU 守卫内）。
# SUBTRACTED: NgramProposerGPU 类与其余 drafter 面板（ch33 域——dossier.
#   delete[5] 删镜像维护、保可变性裁决协议）。
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .output import SchedulerOutput


# SOURCE: vllm/v1/spec_decode/ngram_proposer_gpu.py:L475 update_scheduler_for_
#   invalid_drafts —— 全文逐字（must_keep）
def update_scheduler_for_invalid_drafts(
    num_valid_draft_tokens_event: torch.cuda.Event,
    num_valid_draft_tokens_cpu: torch.Tensor,
    scheduler_output: "SchedulerOutput",
    req_id_to_index: dict[str, int],
) -> None:
    """Trim invalid speculative slots using per-request valid draft counts.

    Args:
        num_valid_draft_tokens_event: Event for async D2H completion.
        num_valid_draft_tokens_cpu: CPU buffer of valid draft counts.
        scheduler_output: Scheduler metadata to update in-place.
        req_id_to_index: Request-id to batch-index mapping.
    """
    # SOURCE: vllm/v1/spec_decode/ngram_proposer_gpu.py:L487-L488
    req_data = scheduler_output.scheduled_cached_reqs
    num_valid_draft_tokens_event.synchronize()

    # SOURCE: vllm/v1/spec_decode/ngram_proposer_gpu.py:L490-L515 逐请求就地
    #   裁剪 token 账
    for req_id in req_data.req_ids:
        req_index = req_id_to_index.get(req_id)
        if req_index is None:
            continue

        spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
        if spec_token_ids is None:
            continue

        scheduled_k = len(spec_token_ids)

        # SOURCE: vllm/v1/spec_decode/ngram_proposer_gpu.py:L498-L499 钳制
        valid_k = int(num_valid_draft_tokens_cpu[req_index].item())
        valid_k = max(0, min(valid_k, scheduled_k))

        # SOURCE: vllm/v1/spec_decode/ngram_proposer_gpu.py:L501-L503 token 账
        #   双回退
        tokens_to_trim = scheduled_k - valid_k
        scheduler_output.total_num_scheduled_tokens -= tokens_to_trim
        scheduler_output.num_scheduled_tokens[req_id] -= tokens_to_trim

        # SOURCE: vllm/v1/spec_decode/ngram_proposer_gpu.py:L505-L515 spec 账
        if valid_k == 0:
            scheduler_output.scheduled_spec_decode_tokens.pop(req_id, None)
        else:
            scheduler_output.scheduled_spec_decode_tokens[req_id] = spec_token_ids[
                :valid_k
            ]
