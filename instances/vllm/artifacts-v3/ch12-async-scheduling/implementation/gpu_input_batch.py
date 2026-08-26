# SOURCE: vllm/v1/worker/gpu_input_batch.py
# InputBatch —— 持久批次的两个影子字段（L309-L311：prev_sampled_token_ids /
# prev_req_id_to_index——『采样 token 不落 CPU』的载体）+ 写回循环消费的三件
# （token_ids_cpu / is_token_ids / num_tokens_no_spec，L134-L158）。容器内景
# （block table/condense/sampling_metadata）归 ch18；本章保消费面。
from __future__ import annotations

from typing import Any

import numpy as np
import torch

PIN_MEMORY = False  # HOST SEAM：CPU host 无 pinned memory（容器内为 True）


# SOURCE: vllm/v1/utils.py:L110 CpuGpuBuffer —— ENGINE SEAM（ch18 边界）：
# 真实是 pinned CPU 张量 + GPU 张量 + 拷贝流的通用桥；HOST 侧以
# torch CPU 张量对（.cpu numpy 视图 / .gpu 张量 / copy_to_gpu）承载同一消费面。
class CpuGpuBuffer:
    # SOURCE: vllm/v1/utils.py:L110 CpuGpuBuffer.__init__ — ENGINE SEAM
    def __init__(self, *size: int, dtype: torch.dtype):
        # SOURCE: vllm/v1/utils.py:L110 CpuGpuBuffer.__init__ — ENGINE SEAM
        self._cpu_tensor = torch.zeros(*size, dtype=dtype)
        self.cpu = self._cpu_tensor.numpy()
        self.np = self.cpu
        self.gpu = torch.zeros(*size, dtype=dtype)

    # SOURCE: vllm/v1/utils.py CpuGpuBuffer.copy_to_gpu — ENGINE SEAM
    def copy_to_gpu(self, n: int) -> None:
        # SOURCE: vllm/v1/utils.py CpuGpuBuffer.copy_to_gpu — ENGINE SEAM
        self.gpu[:n].copy_(self._cpu_tensor[:n], non_blocking=True)


# SOURCE: vllm/v1/worker/gpu_input_batch.py:L93 InputBatch（消费面子集）
class InputBatch:
    # SOURCE: vllm/v1/worker/gpu_input_batch.py:L93-L121 __init__ — ENGINE SEAM
    # （ch18 边界：真实按 vllm_config/max_model_len/device 装配几十个缓冲）
    def __init__(
        self,
        max_num_reqs: int = 256,
        max_model_len: int = 4096,
        vocab_size: int = 16,
    ):
        # SOURCE: vllm/v1/worker/gpu_input_batch.py:L121
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.vocab_size = vocab_size
        # SUBTRACTED: 采样参数行缓冲（temperature/top_p/...，L160-L260——采样域）。
        # SOURCE: vllm/v1/worker/gpu_input_batch.py:L134-L140 token_ids_cpu
        self.token_ids_cpu_tensor = torch.zeros(
            (max_num_reqs, max_model_len), dtype=torch.int32
        )
        self.token_ids_cpu = self.token_ids_cpu_tensor.numpy()
        # SOURCE: vllm/v1/worker/gpu_input_batch.py:L141-L147 is_token_ids
        self.is_token_ids_tensor = torch.zeros(
            (max_num_reqs, max_model_len), dtype=torch.bool
        )
        self.is_token_ids = self.is_token_ids_tensor.numpy()
        # SOURCE: vllm/v1/worker/gpu_input_batch.py:L152-L158 num_tokens_no_spec
        self.num_tokens_no_spec_cpu_tensor = torch.zeros((max_num_reqs,), dtype=torch.int32)
        self.num_tokens_no_spec = self.num_tokens_no_spec_cpu_tensor.numpy()
        # SOURCE: vllm/v1/worker/gpu_input_batch.py num_computed_tokens 缓冲
        # （gpu_model_runner.py:L2086 消费位——乐观 seq_lens 的加数）
        self.num_computed_tokens_cpu_tensor = torch.zeros(
            (max_num_reqs,), dtype=torch.int32
        )

        # SOURCE: vllm/v1/worker/gpu_input_batch.py:L299-L300 spec 占位列表
        # Store last speculative tokens for sampler.
        self.spec_token_ids: list[list[int]] = [[] for _ in range(max_num_reqs)]

        # SUBTRACTED: mm/pooling/logitsproc 行缓冲（L281-L307）。

        # SOURCE: vllm/v1/worker/gpu_input_batch.py:L309-L311 两个影子字段
        # （m10 载体——真 token 的 GPU 住所 + 上拍槽位表）
        # Cached reference to the GPU tensor of previously sampled tokens
        self.prev_sampled_token_ids: torch.Tensor | None = None
        self.prev_req_id_to_index: dict[str, int] | None = None
        # These are used to update output_token_ids with real sampled
        # ids from prior step, if required by current sampling params
        # (e.g. penalties).
        # SOURCE: vllm/v1/worker/gpu_input_batch.py:L312-L316
        self.sampled_token_ids_cpu: torch.Tensor | None = None
        self.async_copy_ready_event: Any | None = None

        # ENGINE SEAM（ch17/18 边界）：行态批的 req 序与记账。
        self._req_ids: list[str] = []
        self.num_computed_tokens: dict[str, int] = {}

    # SOURCE: vllm/v1/worker/gpu_input_batch.py:L318-L322 req_ids property
    @property
    def req_ids(self) -> list[str]:
        # SOURCE: vllm/v1/worker/gpu_input_batch.py:L318-L322
        return self._req_ids

    # SOURCE: vllm/v1/worker/gpu_input_batch.py:L1115 num_reqs property
    @property
    def num_reqs(self) -> int:
        # SOURCE: vllm/v1/worker/gpu_input_batch.py:L1115-L1116
        return len(self._req_ids)

    # SOURCE: vllm/v1/worker/gpu_input_batch.py:L324 _register_add_request —
    # ENGINE SEAM（ch18：真实走 batch_update_builder 的空位回收；HOST 侧追加）
    def add_request(self, req_id: str, prompt_token_ids: list[int]) -> int:
        # SOURCE: vllm/v1/worker/gpu_input_batch.py:L324-L390 add_request — SEAM
        new_req_index = self.num_reqs
        assert new_req_index < self.max_num_reqs
        self._req_ids.append(req_id)
        # SOURCE: vllm/v1/worker/gpu_input_batch.py:L376-L390 prompt 行落位
        num_prompt_tokens = len(prompt_token_ids)
        self.token_ids_cpu[new_req_index, :num_prompt_tokens] = prompt_token_ids
        self.is_token_ids[new_req_index, :num_prompt_tokens] = True
        self.num_tokens_no_spec[new_req_index] = num_prompt_tokens
        self.num_computed_tokens[req_id] = 0
        return new_req_index

    # SOURCE: vllm/v1/worker/gpu_input_batch.py condense/remove 面（L700-L836
    # 内景归 ch18）—— ENGINE SEAM：行摘除的最小动作
    def remove_request(self, req_id: str) -> None:
        # SOURCE: vllm/v1/worker/gpu_input_batch.py condense — ENGINE SEAM
        if req_id in self._req_ids:
            self._req_ids.remove(req_id)
        self.num_computed_tokens.pop(req_id, None)

    # SOURCE: vllm/v1/worker/gpu_input_batch.py:L521-L527 spec 占位写入位
    # （async 路径的 -1 语义——本章保留调用位存证，spec 深水归 ch33）
    def append_spec_tokens(self, req_index: int, spec_token_ids: list[int]) -> None:
        # SOURCE: vllm/v1/worker/gpu_input_batch.py:L521-L527
        # For async scheduling, token_ids_cpu assigned from
        # placeholder entries (-1); real ids updated in-place later.
        start_index = int(self.num_tokens_no_spec[req_index])
        end_token_index = start_index + len(spec_token_ids)
        self.token_ids_cpu[req_index, start_index:end_token_index] = spec_token_ids
        self.is_token_ids[req_index, start_index:end_token_index] = True

    # SOURCE: vllm/v1/worker/gpu_input_batch.py:L1030-L1045 set_async_sampled_
    # token_ids（penalties 消费的上拍采样快照位）
    def set_async_sampled_token_ids(
        self,
        sampled_token_ids_cpu: torch.Tensor,
        async_copy_ready_event,
    ) -> None:
        """In async scheduling case, store ref to sampled_token_ids_cpu
        and the corresponding event, for updating output_token_ids with real
        ids if needed."""
        # SOURCE: vllm/v1/worker/gpu_input_batch.py:L1037-L1045（无 penalties
        #   需求时释放引用的判定随采样域删——恒存引用）
        self.sampled_token_ids_cpu = sampled_token_ids_cpu
        self.async_copy_ready_event = async_copy_ready_event

    # SUBTRACTED: condense/swap_row/capture_scheduled_slots/sampling_metadata
    #   族（L400-L1010——容器内景归 ch18）。
