# SOURCE: vllm/v1/worker/gpu/spec_decode/utils.py
# 只做减法的忠实精简版。DraftTokensHandler 是投机解码 + 结构化输出耦合的 D2H 通道：
# 只有 has_structured_output_reqs 为真时才把本步草稿 token 传回调度器做语法校验
# （m14）——has_structured_output_reqs 就是 input_batch.py 里那个字段，本章第二处
# 门控（第一处是调度侧 has_structured_output_requests）。
#
# 文件改名 spec_decode_utils.py 以免与本章顶层的 utils.py（legacy apply_grammar_bitmask
# 所在文件）撞名——真实仓库两者分属 vllm/v1/worker/gpu/spec_decode/ 与
# vllm/v1/structured_output/ 两个不同包，无此冲突。
#
# SUBTRACTED: SPDX 版权头。
import numpy as np
import torch

from async_utils import async_copy_to_np
from input_batch import InputBatch
from output import DraftTokenIds


class DraftTokensHandler:
    def __init__(self, device: "torch.device | None" = None):
        # SOURCE: vllm/v1/worker/gpu/spec_decode/utils.py:L11-18
        self.device = device
        self.copy_stream = torch.cuda.Stream(device)
        self.copy_event = torch.cuda.Event()

        self.req_ids: "list[str]" = []
        self.draft_tokens_np: "np.ndarray | None" = None
        self.num_draft_tokens: int = 0

    def set_draft_tokens(
        self, input_batch: InputBatch, draft_tokens: torch.Tensor
    ) -> None:
        # SOURCE: vllm/v1/worker/gpu/spec_decode/utils.py:L21-38
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
            self.copy_event.record()

    def get_draft_tokens(self) -> "DraftTokenIds | None":
        # SOURCE: vllm/v1/worker/gpu/spec_decode/utils.py:L40-47
        #
        # SUBTRACTED: async scheduling 关闭时的 `[-1] * self.num_draft_tokens`
        # 占位分支（真实源码里 draft_tokens_np is None 时的 else 分支）——
        # subtraction_plan 批准项7：精简版固定演示 async scheduling 开启路径，
        # 该分支只影响草稿 id 的来源，不影响语法校验流程。
        if self.draft_tokens_np is None:
            return None
        self.copy_event.synchronize()
        draft_token_ids = self.draft_tokens_np.tolist()
        return DraftTokenIds(self.req_ids, draft_token_ids)
