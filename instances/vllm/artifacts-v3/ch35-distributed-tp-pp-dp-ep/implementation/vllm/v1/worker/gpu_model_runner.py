# SOURCE: vllm/v1/worker/gpu_model_runner.py
# ch34 切面（m12）：PP 采样 token 回传两方法（末段 GPU broadcast 直回首段 +
# 首段写 prev_sampled_token_ids/本地账 -1 占位）。GPUModelRunner 的前向/编译/
# 采样主体归 ch17/ch19/ch29 域——以最小类载体承载这两个方法。

from __future__ import annotations

import numpy as np
import torch

from vllm.distributed.parallel_state import get_pp_group


# SOURCE: vllm/v1/worker/gpu_model_runner.py GPUModelRunner —— 载体类（前向/
#   编译/采样主体 ch17/ch19 域；本章只挂 PP token 回传两方法）
class GPUModelRunner:
    # SUBTRACTED: __init__/execute_model/编译捕获/采样族（L228-L4840）——各归
    #   其域。

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4156+ _is_all_reqs_chunked_prefill
    #   —— 判据语义位（chunked prefill 的拍内判定；构建细节归 ch10 域）
    def _is_all_reqs_chunked_prefill(self) -> bool:
        # HOST/ch10 SEAM：载体默认 False（非 chunked 拍）；真实版按
        #   scheduler 配置与拍内请求构成判定。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4156-L4163（锚点双置）
        return False

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4842-L4857
    #   _pp_broadcast_prev_sampled_token_ids — 逐字（末段广播）
    def _pp_broadcast_prev_sampled_token_ids(
        self, sampled_token_ids: torch.Tensor
    ) -> None:
        """Broadcast sampled token ids (GPU) from last PP stage"""
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4842-L4857（锚点双置）
        pp = get_pp_group()
        assert pp.is_last_rank
        # `prev_sampled_token_ids` is expected to have shape [num_reqs, 1].
        assert sampled_token_ids.dim() == 2 and sampled_token_ids.shape[-1] == 1, (
            "PP+async expects sampled_token_ids to have shape [num_reqs, 1]"
        )
        # Skip for chunked prefill: sampled tokens are dummy
        # and will be discarded, no need to broadcast.
        if not self._is_all_reqs_chunked_prefill():
            torch.distributed.broadcast(
                sampled_token_ids, src=pp.rank, group=pp.device_group
            )

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4859-L4887
    #   _pp_receive_prev_sampled_token_ids_to_input_batch — 逐字（首段接收 +
    #   prev_req_id_to_index 重建 + 本地账 -1 占位）
    def _pp_receive_prev_sampled_token_ids_to_input_batch(self) -> None:
        """Receive sampled token ids broadcast from last PP stage"""
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4859-L4887（锚点双置）
        pp = get_pp_group()
        assert not pp.is_last_rank
        num_reqs = self.input_batch.num_reqs
        # `prev_sampled_token_ids` is expected to have shape [num_reqs, 1].
        recv = torch.empty((num_reqs, 1), dtype=torch.int32, device=self.device)
        # skip for chunked prefill.
        if not self._is_all_reqs_chunked_prefill():
            torch.distributed.broadcast(recv, src=pp.last_rank, group=pp.device_group)
        self.input_batch.prev_sampled_token_ids = recv

        # construct `prev_req_id_to_index` here so `_prepare_input_ids`
        # can map req_id -> previous batch row
        discard_req_indices = np.nonzero(self.discard_request_mask.np[:num_reqs])[0]
        discard_req_indices_set = set(discard_req_indices)
        prev_req_id_to_index: dict[str, int] = {}
        for i, req_id in enumerate(self.input_batch.req_ids):
            if i in discard_req_indices_set:
                continue
            prev_req_id_to_index[req_id] = i
            # PP+async scheduling: advance per-request local cached output length by
            # appending a placeholder (-1) token id.
            if (req_state := self.requests.get(req_id)) is not None:
                req_state.output_token_ids.append(-1)
            pos = self.input_batch.num_tokens_no_spec[i]
            self.input_batch.is_token_ids[i, pos] = True
            self.input_batch.num_tokens_no_spec[i] = pos + 1
        self.input_batch.prev_req_id_to_index = prev_req_id_to_index
