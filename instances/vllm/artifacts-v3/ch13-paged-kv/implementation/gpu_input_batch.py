# SOURCE: vllm/v1/worker/gpu_input_batch.py
# InputBatch / CachedRequestState 的**块表消费面**（m7 第 7 站的容器）：
# CachedRequestState.block_ids（每组一份块 id 列表——worker 侧块账）、
# InputBatch.req_id_to_index + num_computed_tokens_cpu + block_table
# （append_row/add_row/clear_row 的落点）。
# 容器内景（condense/采样元数据/持久批维护全貌）归 ch18；本章只持块表行面。
# SUBTRACTED: 采样/logprobs/pooling/mrope/lora/spec/replayssm 影子字段与
#   BatchUpdateBuilder 装配（ch08/12/18/30/33）；PIN_MEMORY 装配（HOST SEAM
#   恒 False——ch12 同款）；MultiGroupBlockTable 构造（L186——单组 = 一张
#   BlockTable，第 4 条）。
from dataclasses import dataclass

import numpy as np

from .block_table import BlockTable


# SOURCE: vllm/v1/worker/gpu_input_batch.py:L34 CachedRequestState
@dataclass
class CachedRequestState:
    # SOURCE: vllm/v1/worker/gpu_input_batch.py:L36-L42
    req_id: str
    prompt_token_ids: list[int] | None
    # SUBTRACTED: mm_features/sampling_params/generator/mrope/xdrope/lora/
    #   prompt_embeds/in_progress_prompt_logprobs_cpu/prompt_is_token_ids/
    #   prev_num_draft_len/pooling_states（L38-L60——各邻章消费面）
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    output_token_ids: list[int]

    # SOURCE: vllm/v1/worker/gpu_input_batch.py:L62 __post_init__
    def __post_init__(self):
        # SOURCE: vllm/v1/worker/gpu_input_batch.py:L64-L65（简化：无 embeds——
        #   length_from_prompt_token_ids_or_embeds 的 None-embeds 支）
        self.num_prompt_tokens = len(self.prompt_token_ids)

    # SOURCE: vllm/v1/worker/gpu_input_batch.py:L72 num_tokens
    @property
    def num_tokens(self) -> int:
        # SOURCE: vllm/v1/worker/gpu_input_batch.py:L74
        return self.num_prompt_tokens + len(self.output_token_ids)


# SOURCE: vllm/v1/worker/gpu_input_batch.py:L92 InputBatch（块表消费面切面）
class InputBatch:
    """Persistent batch of requests on the worker (block-table facet).

    真实 InputBatch 持续整批请求状态（ch18 全文）；本章只保留块表镜像
    消费面：req_id_to_index 行定位、num_computed_tokens_cpu、block_table 行写。
    """

    # SOURCE: vllm/v1/worker/gpu_input_batch.py:L93 __init__（切面）
    def __init__(
        self,
        block_table: BlockTable,
        max_num_reqs: int,
    ) -> None:
        # SUBTRACTED: token_ids_cpu/采样元数据/logits 处理器等整批缓冲
        #   （L95-L185——ch18）；block_table 的 MultiGroupBlockTable 构造
        #   （L186——单组直持一张 BlockTable，第 4 条）。
        # SOURCE: vllm/v1/worker/gpu_input_batch.py:L~115 num_reqs/max_num_reqs
        self.num_reqs = 0
        self.max_num_reqs = max_num_reqs
        # SOURCE: vllm/v1/worker/gpu_input_batch.py:L128 req_id_to_index
        self.req_id_to_index: dict[str, int] = {}
        # SOURCE: vllm/v1/worker/gpu_input_batch.py:L166-L172 num_computed_
        #   tokens_cpu（CPU 侧账——真实还带 torch 镜像 L166-L168，此处以
        #   numpy 数组承载同一行账）
        self.num_computed_tokens_cpu = np.zeros(max_num_reqs, dtype=np.int32)
        # SOURCE: vllm/v1/worker/gpu_input_batch.py:L186 block_table
        self.block_table = block_table

    # SOURCE: vllm/v1/worker/gpu_input_batch.py:L321 add_request（切面）
    def add_request(self, request: CachedRequestState) -> None:
        # SUBTRACTED: BatchUpdateBuilder 行号仲裁（L325-L333——ch18；切面
        #   直取尾行）与 token_ids 行铺/采样元数据（L338-L396）。
        # SOURCE: vllm/v1/worker/gpu_input_batch.py:L331-L333 新请求取行号
        new_req_index = self.num_reqs
        self.num_reqs += 1
        assert new_req_index < self.max_num_reqs, "Cannot add request to full batch."
        # SOURCE: vllm/v1/worker/gpu_input_batch.py:L366 行号登记
        self.req_id_to_index[request.req_id] = new_req_index
        # SOURCE: vllm/v1/worker/gpu_input_batch.py:L397-L398 num_computed 落行
        #   + 块表整行写（真实 add_row(request.block_ids, req_index) 扇出各组；
        #   单组下取 [0] 组直写——MultiGroup 扇出删，第 4 条）
        self.num_computed_tokens_cpu[new_req_index] = request.num_computed_tokens
        self.block_table.add_row(request.block_ids[0], new_req_index)

    # SOURCE: vllm/v1/worker/gpu_input_batch.py:L540 remove_request（切面）
    def remove_request(self, req_id: str) -> None:
        # SOURCE: vllm/v1/worker/gpu_input_batch.py:L543 摘行号
        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            return None
        # SUBTRACTED: batch_update_builder/req_ids/spec 行回收（L545-L547
        #   ——ch18）。
        self.num_reqs -= 1
        # SOURCE: vllm/v1/worker/gpu_input_batch.py:L548 块表腾行
        self.block_table.clear_row(req_index)
