# ch29 llm_base_proposer.py —— subtract-only 精简版（重量级核心，只挑骨架·重度减法）
#
# 真实文件 2043 行。本章对重量级 base「只挑骨架讲、不逐行」，按 dossier.code_spine 只保留
# 四个骨架节点：__init__（持久缓冲 + 昇腾并行组 patch + ACLGraph 开关）、load_model 尾段
# （ACLGraphWrapper 包裹 draft 前向）、_propose（draft 一步骨架）、prepare_inputs（按拒绝数
# 收缩输入）。其余方法（_get_model/_run_merged_draft/compute_draft_token_ids/set_inputs_first_pass/
# prepare_inputs_padded 等）整段属本章范围外，标 SUBTRACTED 略去。
#
# 只删 subtraction_plan.delete 批准项（见各处 SUBTRACTED 标注）。
# must_keep 全保留：AscendSpecDecodeBaseProposer / SpecDecodeBaseProposer /
#   pass_hidden_states_to_model / prepare_inputs / _propose / ACLGraphWrapper / _runnable /
#   num_speculative_tokens / patch_tensor_parallel_group。
#
# SUBTRACTED: llm_base_proposer.py:L1 SPDX 抬头。
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any

import numpy as np
import torch
from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig
from vllm.distributed.parallel_state import (
    get_world_group,
    init_model_parallel_group,
    patch_tensor_parallel_group,
)
from vllm.forward_context import BatchDescriptor
from vllm.model_executor.models.deepseek_eagle3 import Eagle3DeepseekV2ForCausalLM
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.sample.metadata import SamplingMetadata

# SOURCE: vllm/v1/spec_decode/llm_base_proposer.py（SpecDecodeBaseProposer 父类）
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.utils import enable_sp, shared_expert_dp_enabled

# SUBTRACTED: 仅被本章范围外方法体使用的 import（copy / functools.partial / torch.nn /
#   torch.nn.functional / get_layers_from_vllm_config / pcp/pp/tp group / ForwardContext /
#   get_forward_context / logger / AttentionLayerBase / get_model / supports_multimodal /
#   DeepseekV32IndexerCache / triton / cdiv / MLAAttentionSpec / SpecDecodeMetadata /
#   PADDING_SLOT_ID / compute_new_slot_mapping / extend_all_queries_by_N / CachedRequestState /
#   InputBatch / get_ascend_config / _EXTRA_CTX / set_ascend_forward_context /
#   AscendAttentionState / get_lmhead_tp_group / prepare_inputs_padded_kernel /
#   get_vectorcore_num / lmhead_tp_enable）—— 随被删方法体一并删去，原文件 L2-L55。
# SUBTRACTED: _PREPARE_INPUTS_BLOCK_SIZE 常量 + 模块级 split_inputs_tp_to_sp / greedy_sample
#   （原 L58, L73-108）—— 仅本章范围外的多步 draft 循环/分布式采样使用。


# SOURCE: vllm_ascend/spec_decode/llm_base_proposer.py:L63 (_maybe_eager_context)
@contextmanager
def _maybe_eager_context(vllm_config):
    # SOURCE: vllm_ascend/spec_decode/llm_base_proposer.py:L63
    raw_compilation_config_mode = vllm_config.compilation_config.mode
    vllm_config.compilation_config.mode = CompilationMode.NONE
    try:
        yield
    finally:
        vllm_config.compilation_config.mode = raw_compilation_config_mode


# SOURCE: vllm_ascend/spec_decode/llm_base_proposer.py:L111
class AscendSpecDecodeBaseProposer(SpecDecodeBaseProposer):
    _runnable: ACLGraphWrapper | Callable

    # SOURCE: vllm_ascend/spec_decode/llm_base_proposer.py:L114
    def __init__(self, vllm_config: VllmConfig, device: torch.device, pass_hidden_states_to_model: bool, runner=None):
        super().__init__(vllm_config, device, pass_hidden_states_to_model, runner=runner)

        # Assign runner before it's used in the methods below
        self.runner = runner

        # SUBTRACTED: logger.debug(...) 初始化日志（原 L120-131）—— 纯日志，不参与计算。

        self.use_async_scheduling = self.vllm_config.scheduler_config.async_scheduling
        self.use_compress = hasattr(self.vllm_config.model_config.hf_config, "compress_ratios")
        self.pass_hidden_states_to_model = pass_hidden_states_to_model
        self.decode_threshold = 1 + self.num_speculative_tokens
        self.query_start_loc = self.runner._make_buffer(self.runner.max_num_reqs + 2, dtype=torch.int32)
        self.arange_cpu = torch.arange(self.arange.shape[0], device="cpu", dtype=torch.int32)
        self.attn_mask_builder = AttentionMaskBuilder(self.device)

        self.enable_shared_expert_dp = shared_expert_dp_enabled()

        self.pcp_size = self.runner.pcp_size
        self.dcp_size = self.runner.dcp_size
        self.pcp_rank = self.runner.pcp_rank
        self.dcp_rank = self.runner.dcp_rank

        self.full_indices = range(
            self.runner.max_num_tokens * self.pcp_size * self.dcp_size
            + self.pcp_size * self.dcp_size * self.runner.max_num_reqs
        )

        self.use_sparse = hasattr(vllm_config.model_config.hf_text_config, "index_topk")
        # NOTE: `draft_tensor_parallel_size` does not take effect for Eagle: the draft model uses
        # the same TP size as the target model in practice, so we patch tp=1 of draft separately.
        if vllm_config.parallel_config.tensor_parallel_size != self.speculative_config.draft_tensor_parallel_size:
            tp_group = init_model_parallel_group(
                [[get_world_group().rank]],
                get_world_group().rank,
                torch.distributed.get_backend(get_world_group().device_group),
                use_message_queue_broadcaster=True,
                group_name="tp",
            )
            self.tp_group_context = patch_tensor_parallel_group(tp_group)
        else:
            self.tp_group_context = nullcontext()

        # 「cuda_graph」沿用 vLLM 命名，落地是昇腾 ACLGraph（回指 ch25）。
        self.use_cuda_graph = self.runner._use_aclgraph() and not self.speculative_config.enforce_eager

        # TODO: Remove it when the bug of fx-graph is solved
        self.maybe_eager_context: AbstractContextManager[Any] = nullcontext()
        if not self.use_cuda_graph and enable_sp(vllm_config):
            self.maybe_eager_context = _maybe_eager_context(vllm_config)

        self.token_indices_to_sample = torch.zeros(
            self.vllm_config.scheduler_config.max_num_batched_tokens, dtype=torch.int32, device=device
        )
        slot_mapping_lens = self.runner.max_num_tokens + 2 * self.pcp_size * self.runner.max_num_reqs
        self.slot_mapping_group = [
            torch.zeros(slot_mapping_lens, dtype=torch.int32, device=device, pin_memory=self.runner.pin_memory)
            for _ in range(self.num_speculative_tokens)
        ]
        # dsv32 needs seq_lens and query_start_loc persistent tensors for full graph mode
        self.seq_lens_group = [
            torch.zeros(slot_mapping_lens, dtype=torch.int32, device=device, pin_memory=self.runner.pin_memory)
            for _ in range(self.num_speculative_tokens)
        ]
        self.query_start_loc_group = [
            torch.zeros(slot_mapping_lens, dtype=torch.int32, device=device, pin_memory=self.runner.pin_memory)
            for _ in range(self.num_speculative_tokens)
        ]

        self.block_table_tensor_clone: torch.Tensor | None = None

        # 默认 _runnable 为普通函数；FULL graph 时在 load_model 尾段替换为 ACLGraphWrapper 包裹版。
        self._runnable = self._run_merged_draft
        self.is_multimodal_model = self.vllm_config.model_config.is_multimodal_model
        # SUBTRACTED: mrope/xdrope/positions 三选一分支（原 L207-218）—— 长序列/多模态 rope 持久
        #   缓冲细节，本章只讲提议骨架，不展开这些子系统。
        self.token_arange_np = np.arange(self.max_num_tokens + 1, dtype=np.int32)
        self.enable_enpu = self.runner.enable_enpu
        self.use_eagle = self.runner.use_eagle

    # SUBTRACTED: _get_model（原 L223-245）—— 建 draft vllm_config + get_model 细节，本章范围外。

    # SOURCE: vllm_ascend/spec_decode/llm_base_proposer.py:L246
    def load_model(self, model: torch.nn.Module) -> None:
        # SUBTRACTED: load_model 主体（原 L247-423）—— 建 draft 模型、可选共享 embedding/lm_head/
        #   topk_indices、mtp 特判等，重 NPU 加载细节，本章范围外。骨架只保留下方 ACLGraph 尾段。
        if self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs() and self.use_cuda_graph:
            # SUBTRACTED: logger.info(...) ACLGraph 包裹日志（原 L426-431）—— 纯日志。
            self.update_stream = torch.npu.Stream()
            self._runnable = ACLGraphWrapper(
                self._run_merged_draft,
                self.vllm_config,
                runtime_mode=CUDAGraphMode.FULL,
                use_eagle=self.use_eagle,
                enable_enpu=self.enable_enpu,
            )

    # SOURCE: vllm_ascend/spec_decode/llm_base_proposer.py:L621
    def _propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens] or [3, num_tokens] when M-RoPE is enabled
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        target_model_batch_desc: BatchDescriptor,
        sampling_metadata: SamplingMetadata,
        # SUBTRACTED: 与主线无关的可选参数 mm_embed_inputs / req_scheduled_tokens /
        #   long_seq_metadata / num_prefill_reqs / num_decode_reqs / scheduler_output /
        #   num_scheduled_tokens / num_rejected_tokens_gpu（原 L635-642），本章骨架不涉及。
        num_rejected_tokens_gpu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = common_attn_metadata.batch_size()

        if token_indices_to_sample is None:
            token_indices_to_sample = common_attn_metadata.query_start_loc[1:] - 1

        if self.method in ("eagle3", "dflash"):
            assert isinstance(
                self.get_model(), (Eagle3LlamaForCausalLM, DFlashQwen3ForCausalLM, Eagle3DeepseekV2ForCausalLM)
            )
            target_hidden_states = self.model.combine_hidden_states(target_hidden_states)
            assert target_hidden_states.shape[-1] == self.hidden_size

        num_tokens, token_indices_to_sample, common_attn_metadata, long_seq_args = self.set_inputs_first_pass(
            target_token_ids=target_token_ids,
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=token_indices_to_sample,
            cad=common_attn_metadata,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
        )
        # SUBTRACTED: _propose 主体（原 L669-953）—— cudagraph_dispatcher.dispatch、pcp/dcp 切分、
        #   多步 draft 循环、调 self._runnable(...) 跑 (ACLGraph 包裹的) draft 前向、采样出 draft
        #   token，最后 `return draft_token_ids`。骨架等价于（保留真实控制流，不在本章逐行展开）：
        #
        #     aclgraph_runtime_mode, batch_descriptor = \
        #         self.runner.cudagraph_dispatcher.dispatch(num_tokens=num_tokens, ...)
        #     run_draft = partial(self._runnable, num_input_tokens=..., target_positions=...,
        #                         token_indices_to_sample=..., multi_steps_attn_metadata=...)
        #     draft_token_ids = run_draft()      # ACLGraph 包裹的 draft 前向（含昇腾 Triton/MLA）
        #     return draft_token_ids
        #
        # 本章只讲到此骨架；多步 draft 前向需 NPU/CANN，不在精简版内真跑，故此处不产出 draft token。
        raise NotImplementedError(
            "draft 前向主体属本章范围外（重 NPU/Triton/MLA），见上方 SUBTRACTED 骨架说明"
        )

    # SUBTRACTED: compute_draft_token_ids / _run_merged_draft / set_inputs_first_pass /
    #   prepare_inputs_padded 等方法（原 L954-1700）—— draft 前向与多步循环实现，重 NPU/Triton/MLA，
    #   本章范围外（回指 ch20 MLA / ch25 ACLGraph）。

    # SOURCE: vllm_ascend/spec_decode/llm_base_proposer.py:L1701
    def prepare_inputs(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: list[list[int]],
        num_draft_tokens: list[int],
    ) -> tuple[CommonAttentionMetadata, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It updates to the common_attn_metadata to account for the rejected
        tokens (and newly sampled tokens). It also returns the token indices
        of the tokens that should be fed to the speculator.
        """
        # SUBTRACTED: docstring 内 [0,q1,q1+q2,...] 的 ASCII 算法示例（原 L1713-1727）—— 说明
        #   意图，正文改述为公式即可，不影响计算。
        num_actual_reqs = len(num_draft_tokens)
        num_rejected_tokens = [
            n + 1 - len(sampled_token_ids[i]) if n > 0 else 0 for i, n in enumerate(num_draft_tokens)
        ]
        num_rejected_tokens = torch.tensor(num_rejected_tokens, dtype=torch.int32)

        device = common_attn_metadata.query_start_loc.device
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_actual_reqs + 1]
        # Prefer the upstream-canonical ``_seq_lens_cpu``; fall back to the Ascend subclass field.
        if common_attn_metadata._seq_lens_cpu is not None:
            seq_lens_cpu = common_attn_metadata._seq_lens_cpu[:num_actual_reqs]
        else:
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu[:num_actual_reqs]
        new_seq_lens_cpu = seq_lens_cpu - num_rejected_tokens

        # [0, q1, q1 + q2, q1 + q2 + q3] -> [q1, q2, q3]
        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        # [q1, q2, q3] -> [q1 - n1, q2 - n2, q3 - n3]
        new_num_tokens_per_req = new_query_len_per_req - num_rejected_tokens
        new_num_tokens_per_req_np = new_num_tokens_per_req.numpy()

        # [q1 - n1, ...] -> [0, q1 - n1, q1 + q2 - n1 - n2, ...]
        new_query_start_loc_cpu = torch.zeros(
            query_start_loc_cpu.shape,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
        )
        new_query_start_loc_np = new_query_start_loc_cpu.numpy()
        np.cumsum(new_num_tokens_per_req_np, out=new_query_start_loc_np[1:])

        total_num_tokens = new_query_start_loc_np[-1]
        new_query_start_locs_expanded = np.repeat(new_query_start_loc_np[:-1], new_num_tokens_per_req_np)
        token_offsets = self.token_arange_np[:total_num_tokens] - new_query_start_locs_expanded

        old_query_start_locs_expanded = np.repeat(query_start_loc_cpu[:-1].numpy(), new_num_tokens_per_req_np)
        token_indices_np = token_offsets + old_query_start_locs_expanded
        token_indices = torch.from_numpy(token_indices_np).to(device, non_blocking=True)

        common_attn_metadata.slot_mapping[: token_indices.shape[0]].copy_(
            common_attn_metadata.slot_mapping[token_indices]
        )
        common_attn_metadata.slot_mapping[token_indices.shape[0] :].fill_(-1)

        spec_common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=new_query_start_loc_cpu.to(device, non_blocking=True),
            query_start_loc_cpu=new_query_start_loc_cpu,
            seq_lens=new_seq_lens_cpu.to(device, non_blocking=True),
            seq_lens_cpu=new_seq_lens_cpu,
            _seq_lens_cpu=new_seq_lens_cpu,
            num_computed_tokens_cpu=common_attn_metadata.num_computed_tokens_cpu,
            _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
            seq_lens_cpu_upper_bound=new_seq_lens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            num_input_tokens=common_attn_metadata.num_input_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            slot_mapping_cpu=common_attn_metadata.slot_mapping_cpu,
            actual_seq_lengths_q=self.runner.actual_seq_lengths_q,
            positions=common_attn_metadata.positions[token_indices],
            positions_cpu=common_attn_metadata.positions_cpu[token_indices]
            if common_attn_metadata.positions_cpu is not None
            else None,
            attn_state=self.runner.attn_state,
            decode_token_per_req=self.runner.decode_token_per_req,
            max_seq_len=0,
        )
        return spec_common_attn_metadata, token_indices
