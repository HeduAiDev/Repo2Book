# vllm_ascend/_310p/npu_input_batch.py —— subtract-only 精简版（ch17 主线之一：输入批子类化）
#
# "再向下子类化"最干净的样本：NPUInputBatch310(NPUInputBatch) 唯一实质改动是
# __init__ 末尾——先 super().__init__ 原样透传全部参数（行为全继承），再用 _310p 的
# MultiGroupBlockTable 覆盖父类建好的 self.block_table。差异只此一处：把块表换成走
# CPU NumPy slot_mapping 的 310 版。
#
# NPUInputBatch 自身又是 vLLM InputBatch 的子类+猴补（ch14），故 310 是"vLLM → 昇腾
# 主栈 → 310"三层继承线的底层。
import torch
from vllm.v1.kv_cache_interface import KVCacheGroupSpec
from vllm.v1.sample.logits_processor import LogitsProcessors

from vllm_ascend._310p.block_table import MultiGroupBlockTable
from vllm_ascend.worker.npu_input_batch import NPUInputBatch


# SOURCE: vllm_ascend/_310p/npu_input_batch.py:L9
class NPUInputBatch310(NPUInputBatch):
    # SOURCE: vllm_ascend/_310p/npu_input_batch.py:L10-L59
    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        device: torch.device,
        pin_memory: bool,
        vocab_size: int,
        block_sizes: list[int],
        kernel_block_sizes: list[list[int]],
        max_num_blocks_per_req: list[int] | None = None,
        logitsprocs: LogitsProcessors | None = None,
        logitsprocs_need_output_token_ids: bool = False,
        is_spec_decode: bool = False,
        is_pooling_model: bool = False,
        num_speculative_tokens: int = 0,
        cp_kv_cache_interleave_size: int = 1,
        kv_cache_groups: list[KVCacheGroupSpec] | None = None,
    ):
        # 原样透传全部参数给父类——行为全继承，父类已建好 self.block_table。
        super().__init__(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            device=device,
            pin_memory=pin_memory,
            vocab_size=vocab_size,
            block_sizes=block_sizes,
            kernel_block_sizes=kernel_block_sizes,
            max_num_blocks_per_req=max_num_blocks_per_req,
            logitsprocs=logitsprocs,
            logitsprocs_need_output_token_ids=logitsprocs_need_output_token_ids,
            is_spec_decode=is_spec_decode,
            is_pooling_model=is_pooling_model,
            num_speculative_tokens=num_speculative_tokens,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            kv_cache_groups=kv_cache_groups,
        )
        # 唯一实质改动：用 _310p 的 MultiGroupBlockTable 覆盖父类建好的 block_table。
        self.block_table = MultiGroupBlockTable(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            pin_memory=pin_memory,
            device=device,
            block_sizes=block_sizes,
            max_num_blocks=max_num_blocks_per_req,
            num_speculative_tokens=num_speculative_tokens,
            kernel_sizes=kernel_block_sizes,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            kv_cache_groups=kv_cache_groups,
        )
