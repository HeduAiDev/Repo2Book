# SOURCE: vllm/v1/worker/gpu/spec_decode/utils.py
# v3 ch31 脊柱⑪：DraftTokensHandler（L11-L52——spec+结构化输出时草稿的 D2H
# 回传通道：has_structured_output_reqs 门控整批跳过；为真在独立 copy_stream 上
# async_copy_to_np，copy_event(blocking) 防忙轮询 CUDA 驱动锁、record_stream
# 防缓存分配器提前复用——异步内存安全两面）。get_parallel_drafting_token_id
# 原样保留（不在批准删除清单）。
# SUBTRACTED（delete[5]）：get_draft_tokens 的 async 关闭分支（[-1] 占位列表）
#   ——精简版固定演示 async 开启路径（v0.27 默认）；set_draft_tokens 的
#   record_stream 之外的注释性代码删。
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.async_utils import async_copy_to_np

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch
# SUBTRACTED: InputBatch 真实为模块级导入（L8）——纯注解消费，移入
#   TYPE_CHECKING（与 V1 utils.py 的真实写法同款）。
# SUBTRACTED: 注释性代码（delete[5]）。


# SOURCE: vllm/v1/worker/gpu/spec_decode/utils.py:L11 DraftTokensHandler —— 逐字
class DraftTokensHandler:
    # SOURCE: vllm/v1/worker/gpu/spec_decode/utils.py:L12-L20 __init__ —— 逐字
    def __init__(self, device: torch.device | None = None):
        self.device = device
        self.copy_stream = torch.cuda.Stream(device)
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = torch.cuda.Event(blocking=True)

        self.req_ids: list[str] = []
        self.draft_tokens_np: np.ndarray | None = None
        self.num_draft_tokens: int = 0

    # SOURCE: vllm/v1/worker/gpu/spec_decode/utils.py:L22-L43 set_draft_tokens —— 逐字
    def set_draft_tokens(
        self, input_batch: InputBatch, draft_tokens: torch.Tensor
    ) -> None:
        self.req_ids = input_batch.req_ids
        self.num_draft_tokens = draft_tokens.shape[1]
        if not input_batch.has_structured_output_reqs:
            # No draft token validation needs to be performed by
            # the scheduler for this batch.
            self.draft_tokens_np = None
            return

        # For spec decoding + structured outputs, we must transfer the
        # draft tokens back to the scheduler for grammar validation.
        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.copy_stream):
            self.draft_tokens_np = async_copy_to_np(draft_tokens)
            # draft_tokens is a temporary allocation on the main stream and read here on
            # copy_stream; without record_stream, the caching allocator may reuse its
            # memory before the async copy executes.
            draft_tokens.record_stream(self.copy_stream)
            self.copy_event.record()

    #   逐字（仅删 else 分支）
    # SOURCE: vllm/v1/worker/gpu/spec_decode/utils.py:L45-L52 get_draft_tokens
    def get_draft_tokens(self) -> DraftTokenIds | None:
        if self.draft_tokens_np is not None:
            self.copy_event.synchronize()
            draft_token_ids = self.draft_tokens_np.tolist()
        # SUBTRACTED: vllm/v1/worker/gpu/spec_decode/utils.py:L49-L51 else 分支
        #   （async 关闭时的 [-1] 占位列表——delete[5]，v0.27 默认 async 开）。
        return DraftTokenIds(self.req_ids, draft_token_ids)


# SOURCE: vllm/v1/worker/gpu/spec_decode/utils.py:L55-L70 get_parallel_drafting_token_id —— 逐字
def get_parallel_drafting_token_id(hf_config) -> int:
    """Resolve the mask token id used for parallel drafting slots.

    Checks (in order): `dflash_config.mask_token_id`, top-level `mask_token_id`,
    `dspark_noise_token_id`, `pard_token`, `ptd_token_id`. Raises ValueError if
    none are present.
    """
    dflash_config = getattr(hf_config, "dflash_config", None) or {}
    if "mask_token_id" in dflash_config:
        return int(dflash_config["mask_token_id"])
    if getattr(hf_config, "mask_token_id", None) is not None:
        return int(hf_config.mask_token_id)
    if hasattr(hf_config, "dspark_noise_token_id"):
        return int(hf_config.dspark_noise_token_id)
    if hasattr(hf_config, "pard_token"):
        return int(hf_config.pard_token)
    if hasattr(hf_config, "ptd_token_id"):
        return int(hf_config.ptd_token_id)
    raise ValueError(
        "Model config must specify `dflash_config.mask_token_id`,"
        " `mask_token_id`, `dspark_noise_token_id`, `pard_token`, or"
        " `ptd_token_id` for parallel drafting."
    )
