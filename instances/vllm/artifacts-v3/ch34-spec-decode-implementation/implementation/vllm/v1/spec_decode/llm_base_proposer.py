# SOURCE: vllm/v1/spec_decode/llm_base_proposer.py —— 本章主角文件（契约骨架）
# V1 模型类 drafter 统一基类（EAGLE/EAGLE3/DFlash/MTP/draft_model/Gemma4）：
# 对外契约 = 吃目标 token/positions/hidden_states，跑草稿模型前向，吐每请求
# num_speculative_tokens 个草稿（draft 恒 greedy——分布正确性由拒绝采样兜底，
# 只影响接受率不影响输出分布）。
#
# 减法分两类（每处均有 SUBTRACTED 标记）：
# (a) dossier subtraction_plan.delete[2] 批准项：mrope/xdrope position 分支、
#     多模态（mm_embed_inputs/supports_mm_inputs）、cudagraph padding
#     （_determine_batch_execution_and_padding/input_batch_size——精简版固定
#     input_batch_size==batch_size）、DFlash 专属 extra-slots 分支
#     （needs_extra_input_slots/copy_and_expand_*）。（v0.27.1 本文件已无独立
#     tree attention 分支——该谱系位由 V2 树承担，计划所指分支现核不存在。）
# (b) 引擎栈未内嵌（沿 v2 ch34「契约骨架」先例 + delete[5] 对编排栈的处理）：
#     注意力元数据构建（build_per_group_and_layer_attn_metadata/
#     _update_positions_dependent_metadata/prepare_inputs 族）、EPLB、ROCm
#     allowed_attn_types、模型加载与 embed/lm_head 权重共享（load_model/
#     _maybe_share_*）、cudagraph dispatcher（initialize_cudagraph_keys/
#     dummy_run）、_create_draft_vllm_config/_get_model——正文以内嵌真源码
#     解读，精简版不载（依赖完整 vLLM 模型/注意力栈，无法脱离 vLLM 运行）。
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import dataclasses

import torch

from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import (
    empty_exponential_noise_like,
    sample_with_exponential_noise,
)
from vllm.v1.sample.sampler import _SAMPLING_EPS

# SUBTRACTED: vllm/v1/spec_decode/llm_base_proposer.py:L1-L65 —— 一大批引擎栈
#   import（BreakableCUDAGraphWrapper/CudagraphDispatcher/EplbState/
#   set_forward_context/各 EAGLE·EAGLE3·DFlash 模型类/MULTIMODAL_REGISTRY/
#   AttentionGroup/CpuGpuBuffer/spec_decode.utils 的融合 kernel 等）——
#   只被 (a)(b) 两类未载分支消费。


# SOURCE: vllm/v1/spec_decode/llm_base_proposer.py:L68-L318 SpecDecodeBaseProposer.__init__
#   —— 主路径字段派生逐字保留（speculative_config/draft_model_config/hidden_size/
#   parallel_drafting/采样开关/probabilistic draft_probs 两态/buffer 预分配）
# SUBTRACTED: vllm/v1/spec_decode/llm_base_proposer.py:L82/L88 —— eplb_state/
#   dp_rank 两个记账位（消费方已删）。L106-L127 的 extra_slots/constant_draft_
#   positions 之外的 DFlash 参数（_init_parallel_drafting_params 的 mask token
#   探测与 parallel_drafting_hidden_state_tensor）—— delete[2] DFlash 专属。
# SUBTRACTED: vllm/v1/spec_decode/llm_base_proposer.py:L146-L164 —— 多模态
#   registry（supports_mm_inputs）与 compilation_config/cudagraph_dispatcher、
#   draft_attn_groups/eagle3_use_aux_hidden_state（注意力栈/图栈）。
# SUBTRACTED: vllm/v1/spec_decode/llm_base_proposer.py:L170-L201 的 mrope/
#   xdrope position buffer 两分支 —— delete[2]；精简版固定 1D positions
#   （L196-L201 分支）。L207-L214 的 block_size/arange/token_arange_np（消费方
#   已删的注意力元数据维护）。
# SUBTRACTED: vllm/v1/spec_decode/llm_base_proposer.py:L216-L257 —— extra-slots
#   卫兵三连（_raise_if_padded_drafter_batch/_warn_if_multimodal/_raise_if_mrope
#   调用位）、is_rejected/is_masked mask、inputs_embeds buffer（多模态）、
#   backup_next_token_ids（CpuGpuBuffer）、_slot_mapping_buffer —— delete[2]。
# SUBTRACTED: vllm/v1/spec_decode/llm_base_proposer.py:L259-L318 —— ROCm
#   allowed_attn_types 白名单（is_rocm() 分支体内）。
class SpecDecodeBaseProposer:
    def __init__(
        self,
        vllm_config,
        device: torch.device,
        pass_hidden_states_to_model: bool,
        runner=None,
    ):
        # SOURCE: vllm/v1/spec_decode/llm_base_proposer.py:L69-L318 SpecDecodeBaseProposer.__init__（契约骨架字段派生）
        self.vllm_config = vllm_config
        assert vllm_config.speculative_config is not None
        self.speculative_config = vllm_config.speculative_config
        self.draft_model_config = self.speculative_config.draft_model_config
        self.method = self.speculative_config.method
        self.pass_hidden_states_to_model = pass_hidden_states_to_model
        self._share_mtp_indices = False

        self.device = device
        self.dtype = vllm_config.model_config.dtype
        self.max_model_len = vllm_config.model_config.max_model_len
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens

        # We need to get the hidden size from the draft model config because
        # the draft model's hidden size can be different from the target model's
        # hidden size (e.g., Llama 3.3 70B).
        self.hidden_size = self.draft_model_config.get_hidden_size()
        self.inputs_embeds_size = self.draft_model_config.get_inputs_embeds_size()

        # DeepSeek V4 MTP consumes the target's pre-hc_head residual stream,
        # shape (T, hc_mult * hidden_size). Expand the hidden_states buffer
        # so target_hidden_states fits; detect DeepseekV4 via draft hf_config.
        draft_hf_config = self.draft_model_config.hf_config
        if hasattr(draft_hf_config, "compress_ratios") and hasattr(
            draft_hf_config, "hc_mult"
        ):
            self.hidden_size = self.hidden_size * draft_hf_config.hc_mult

        # Unifying eagle, draft model, and parallel drafting support.
        # DFlash always uses parallel drafting (all tokens in one pass),
        # but has an additional slot for the next_token_id (does not shift like EAGLE)
        self.parallel_drafting: bool = self.speculative_config.parallel_drafting

        # When True, all draft steps reuse the same position as the
        # first step instead of advancing by one each iteration.
        # Used by draft models with Q-only attention that share KV
        # with the target and always predict from the same position.
        self.constant_draft_positions: bool = False

        self.use_local_argmax_reduction: bool = (
            self.speculative_config.use_local_argmax_reduction
        )
        self.use_fp64_gumbel = vllm_config.model_config.use_fp64_gumbel

        self.use_heterogeneous_vocab: bool = (
            self.speculative_config.use_heterogeneous_vocab
        )
        self.vocab_mapping = None

        self.max_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.max_positions = self.max_num_tokens

        # persistent buffers for cuda graph
        self.input_ids = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device=device
        )
        # RoPE need (max_num_tokens,)
        self.positions = torch.zeros(
            self.max_positions,
            dtype=torch.int64,
            device=device,
        )
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size), dtype=self.dtype, device=device
        )

        self._enable_probabilistic_draft_probs = (
            self.speculative_config.rejection_sample_method == "standard"
            and self.speculative_config.draft_sample_method == "probabilistic"
        )
        self._last_draft_probs: torch.Tensor | None = None

    # SUBTRACTED: vllm/v1/spec_decode/llm_base_proposer.py:L320-L345 ——
    #   _raise_if_padded_drafter_batch_disabled/_warn_if_multimodal/_raise_if_mrope
    #   三卫兵方法与 set_eplb_state —— 消费位（extra-slots 装配）已按 delete[2] 删。

    # SUBTRACTED: vllm/v1/spec_decode/llm_base_proposer.py:L347-L371
    #   _init_parallel_drafting_params —— delete[2] DFlash 专属（parallel
    #   drafting 的 mask token id 探测；精简版 parallel_drafting 只作早退判据）。

    # SOURCE: vllm/v1/spec_decode/llm_base_proposer.py:L373-L378 _get_positions
    #   —— 1D 主路径逐字（mrope/xdrope 两分支已按 delete[2] 减去）
    def _get_positions(self, num_tokens: int):
        # SOURCE: vllm/v1/spec_decode/llm_base_proposer.py:L373-L378 _get_positions（1D 主路径）
        return self.positions[:num_tokens]

    # SOURCE: vllm/v1/spec_decode/llm_base_proposer.py:L380-L391 _set_positions
    #   —— 1D 主路径逐字
    # SUBTRACTED: vllm/v1/spec_decode/llm_base_proposer.py:L380-L390 的 mrope/
    #   xdrope 写回分支与 target-M-RoPE→1D 转换（positions[0] 取首维）——
    #   delete[2]；精简版固定 1D positions 直写。
    def _set_positions(self, num_tokens: int, positions: torch.Tensor):
        # SOURCE: vllm/v1/spec_decode/llm_base_proposer.py:L380-L391 _set_positions（1D 主路径）
        self.positions[:num_tokens] = positions

    # SUBTRACTED: vllm/v1/spec_decode/llm_base_proposer.py:L393-L409 _get_slot_mapping
    #   —— slot mapping buffer 面（EAGLE 层的 KV 槽位视图，注意力栈）。
    # SUBTRACTED: vllm/v1/spec_decode/llm_base_proposer.py:L411-L426
    #   initialize_cudagraph_keys —— PIECEWISE-only cudagraph 契约（正文内嵌
    #   真源码解读；依赖 CudagraphDispatcher 图栈）。docstring 原话："Only
    #   supports PIECEWISE cudagraphs (via mixed_mode)"。

    # SOURCE: vllm/v1/spec_decode/llm_base_proposer.py:L428-L438 _greedy_sample —— 逐字
    #   （draft 恒贪心：use_local_argmax_reduction 走 get_top_tokens 免全词表
    #   logits gather；异词表经 vocab_mapping 约束+映射；主路径 compute_logits
    #   .argmax）
    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm/v1/spec_decode/llm_base_proposer.py:L428-L438 _greedy_sample（draft 恒贪心三支路）
        """Greedy-sample draft tokens from hidden states."""
        if self.use_local_argmax_reduction:
            return self.model.get_top_tokens(hidden_states)
        if self.use_heterogeneous_vocab:
            logits = self.model.compute_logits(hidden_states)
            assert self.vocab_mapping is not None
            logits = self.vocab_mapping.constrain_draft_logits(logits)
            draft_token_ids = logits.argmax(dim=-1)
            return self.vocab_mapping.map_draft_to_target_ids(draft_token_ids)
        return self.model.compute_logits(hidden_states).argmax(dim=-1)

    # SOURCE: vllm/v1/spec_decode/llm_base_proposer.py:L440-L466 _sample_from_logits
    #   —— 逐字（概率化路径；parallel drafting 的 K 行 repeat_interleave 温度对齐）
    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # SOURCE: vllm/v1/spec_decode/llm_base_proposer.py:L440-L466 _sample_from_logits（概率化分流）
        if not self._enable_probabilistic_draft_probs:
            return logits.argmax(dim=-1), None
        if sampling_metadata.all_greedy:
            return logits.argmax(dim=-1), None

        # Parallel drafting (e.g. DFlash) samples num_speculative_tokens rows
        # per request in a single pass, so logits has batch_size * K rows while
        # the sampling metadata is per-request. The rows are request-major
        # (K consecutive slots per request), so repeat_interleave the
        # per-request temperature to match before probabilistic sampling.
        temperature = sampling_metadata.temperature
        if temperature is not None and temperature.shape[0] != logits.shape[0]:
            assert logits.shape[0] % temperature.shape[0] == 0
            factor = logits.shape[0] // temperature.shape[0]
            sampling_metadata = dataclasses.replace(
                sampling_metadata,
                temperature=temperature.repeat_interleave(factor, dim=0),
            )

        return compute_probs_and_sample_next_token(
            logits, sampling_metadata, self.use_fp64_gumbel
        )

    # SOURCE: vllm/v1/spec_decode/llm_base_proposer.py:L468-L497 _sample_draft_tokens
    #   —— 逐字（默认 greedy；概率化才 compute_logits + _sample_from_logits；
    #   异词表的 draft_probs 断言）
    def _sample_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # SOURCE: vllm/v1/spec_decode/llm_base_proposer.py:L468-L497 _sample_draft_tokens（greedy/概率化总闸）
        if not self._enable_probabilistic_draft_probs or sampling_metadata.all_greedy:
            return self._greedy_sample(hidden_states), None
        logits = self.model.compute_logits(hidden_states)
        if self.use_heterogeneous_vocab:
            assert self.vocab_mapping is not None
            logits = self.vocab_mapping.constrain_draft_logits(logits)
        draft_token_ids, draft_probs = self._sample_from_logits(
            logits, sampling_metadata
        )
        if self.use_heterogeneous_vocab:
            assert self.vocab_mapping is not None
            draft_token_ids = self.vocab_mapping.map_draft_to_target_ids(
                draft_token_ids
            )
            # Config validation ensures draft_sample_method == "greedy" when
            # use_heterogeneous_vocab is True, so this branch should never be
            # reached. Kept as a safety fallback until probabilistic rejection
            # sampling with heterogeneous vocabularies is implemented.
            # TODO: remap draft_probs to target-vocab space for lossless
            # probabilistic rejection sampling with heterogeneous vocabularies.
            assert draft_probs is None, (
                "probabilistic draft sampling is not supported with "
                "use_heterogeneous_vocab"
            )
        return draft_token_ids, draft_probs

    # SOURCE: vllm/v1/spec_decode/llm_base_proposer.py:L499-L500 take_last_draft_probs
    #   —— 逐字（可选概率化 draft_probs 的缓存出口：runner 侧
    #   _get_spec_decode_draft_probs 按 req_id 对行拼装，正文内嵌解读）
    def take_last_draft_probs(self) -> torch.Tensor | None:
        # SOURCE: vllm/v1/spec_decode/llm_base_proposer.py:L499-L500 take_last_draft_probs（draft_probs 缓存出口）
        return self._last_draft_probs

    # SOURCE: vllm/v1/spec_decode/llm_base_proposer.py:L502-L767 propose —— EAGLE
    #   主路径控制流逐字保留（eagle3 combine → set_inputs_first_pass → 首前向 →
    #   k=0 空草稿 → 单步/parallel 早退 → 自回归多步链式 → stack [B,k]）
    # SUBTRACTED: vllm/v1/spec_decode/llm_base_proposer.py:L528-L538 —— eagle3/
    #   dflash 的 BreakableCUDAGraphWrapper unwrap 与五个模型类 isinstance 断言
    #   （图栈/模型类未载；combine_hidden_states 调用保留）。
    # SUBTRACTED: vllm/v1/spec_decode/llm_base_proposer.py:L561-L563/L659-L661 ——
    #   _determine_batch_execution_and_padding 调用（cudagraph padding +
    #   DP 协调）—— delete[2]；精简版 num_input_tokens==num_tokens、
    #   input_batch_size==batch_size。
    # SUBTRACTED: vllm/v1/spec_decode/llm_base_proposer.py:L568-L599 —— set_forward_context
    #   上下文管理器与 MTP indexer 钩子（_share_mtp_indices/set_skip_topk/
    #   compact_topk_indices）——前向上下文/模型内部钩子栈未载，self.model(...)
    #   裸调保留。
    # SUBTRACTED: vllm/v1/spec_decode/llm_base_proposer.py:L574-L578/L733-L737 ——
    #   eplb_state.prepare_forward —— EPLB 记账（未载）。
    # SUBTRACTED: vllm/v1/spec_decode/llm_base_proposer.py:L629-L632 的 mrope
    #   positions 分支 —— delete[2]（1D 主路径保留）。
    # SUBTRACTED: vllm/v1/spec_decode/llm_base_proposer.py:L646-L654 —— ROCm
    #   allowed_attn_types 校验块（未载，非 ROCm 时本就不触发）。
    # SUBTRACTED: vllm/v1/spec_decode/llm_base_proposer.py:L663-L681 —— decode 循环的
    #   common_attn_metadata 维护（num_actual_tokens/max_query_len/query_start_loc
    #   重写、num_rejected_tokens_gpu 的 seq_lens 回扣——extra-slots 族 delete[2]、
    #   block_size 断言）与 L693-L710 的 per-step 元数据重建/融合 kernel
    #   （_update_positions_dependent_metadata，注意力栈）；精简版契约层只展示
    #   『喂回上一草稿 → 前向 → greedy 采下一草稿』，positions 沿首拍采样位不变。
    # SUBTRACTED: vllm/v1/spec_decode/llm_base_proposer.py:L688-L691 —— 异词表
    #   target→draft id 映射的循环位与 L715-L722 多模态 inputs_embeds 分支 ——
    #   后者 delete[2]；前者保留在 _greedy_sample 出口侧（map_draft_to_target_ids）。
    def propose(
        self,
        num_speculative_tokens,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens] or [3, num_tokens] when M-RoPE is enabled
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> torch.Tensor:
        # SOURCE: vllm/v1/spec_decode/llm_base_proposer.py:L502-L767 propose（EAGLE 主路径：首前向→早退/自回归多步链式）
        self.num_speculative_tokens = num_speculative_tokens
        self._last_draft_probs = None
        batch_size = common_attn_metadata.batch_size()

        if self.method in ("eagle3", "dflash"):
            target_hidden_states = self.model.combine_hidden_states(
                target_hidden_states
            )
            assert target_hidden_states.shape[-1] == self.hidden_size

        num_tokens, token_indices_to_sample, common_attn_metadata = (
            self.set_inputs_first_pass(
                target_token_ids=target_token_ids,
                next_token_ids=next_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                token_indices_to_sample=token_indices_to_sample,
                cad=common_attn_metadata,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )
        )

        model_kwargs, slot_mapping_size = self.build_model_inputs_first_pass(
            num_tokens, num_tokens, mm_embed_inputs
        )

        ret_hidden_states = self.model(**model_kwargs)
        if not self.model_returns_tuple():
            last_hidden_states = ret_hidden_states
            hidden_states = last_hidden_states
        else:
            last_hidden_states, hidden_states = ret_hidden_states

        sample_hidden_states = last_hidden_states[token_indices_to_sample]

        # No draft tokens requested (e.g. Dynamic SD decided K=0).
        # The prefill forward pass above already ran to keep the drafter
        # KV cache in sync, so just return an empty tensor.
        if self.num_speculative_tokens == 0:
            return torch.empty(
                batch_size,
                0,
                device=sample_hidden_states.device,
                dtype=torch.int64,
            )

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1 or self.parallel_drafting:
            draft_token_ids, draft_probs = self._sample_draft_tokens(
                sample_hidden_states, sampling_metadata
            )
            if draft_probs is not None:
                self._last_draft_probs = draft_probs.view(
                    -1, self.num_speculative_tokens, draft_probs.shape[-1]
                ).contiguous()
            return draft_token_ids.view(-1, self.num_speculative_tokens)

        positions = self.positions[token_indices_to_sample]
        hidden_states = hidden_states[token_indices_to_sample]

        if self.constant_draft_positions:
            # Write the sampling positions into the front of the
            # positions buffer so that subsequent loop iterations
            # (which read via _get_positions) use the correct values.
            self.positions[:batch_size] = positions

        draft_token_ids, draft_probs = self._sample_draft_tokens(
            sample_hidden_states, sampling_metadata
        )
        draft_probs_list = None if draft_probs is None else [draft_probs]

        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]

        for token_index in range(self.num_speculative_tokens - 1):
            # Update the inputs.
            # cast to int32 is crucial when eagle model is compiled.
            # tensor.argmax() returns int64 by default.
            input_ids = draft_token_ids_list[-1].int()

            # copy inputs to buffer for cudagraph
            self.input_ids[:batch_size] = input_ids
            self.hidden_states[:batch_size] = hidden_states

            # Run the model.
            model_kwargs = {
                "input_ids": self.input_ids[:batch_size],
                "positions": self._get_positions(batch_size),
                "inputs_embeds": None,
            }
            if self.pass_hidden_states_to_model:
                model_kwargs["hidden_states"] = self.hidden_states[:batch_size]

            ret_hidden_states = self.model(**model_kwargs)
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                hidden_states = ret_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states

            hidden_states = hidden_states[:batch_size]
            draft_token_ids, draft_probs = self._sample_draft_tokens(
                last_hidden_states[:batch_size], sampling_metadata
            )
            if draft_probs is not None:
                assert draft_probs_list is not None
                draft_probs_list.append(draft_probs)
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        if draft_probs_list is not None:
            self._last_draft_probs = torch.stack(draft_probs_list, dim=1).contiguous()
        return draft_token_ids

    # SUBTRACTED: vllm/v1/spec_decode/llm_base_proposer.py:L769-L819
    #   _update_positions_dependent_metadata —— eagle_step_update_slot_mapping_
    #   and_metadata 融合 kernel + seq_len/slot 推进（注意力栈）。

    # SOURCE: vllm/v1/spec_decode/llm_base_proposer.py:L821-L860 set_inputs_first_pass
    #   —— 默认 EAGLE 分支逐字（input_ids 左移一格 + 末槽插 next_token）
    # SUBTRACTED: vllm/v1/spec_decode/llm_base_proposer.py:L831-L837 —— 异词表
    #   target→draft 映射入口（vocab_mapping 未载实例、_greedy_sample 出口侧
    #   同族逻辑保留）。
    # SUBTRACTED: vllm/v1/spec_decode/llm_base_proposer.py:L854-L855 —— xdrope
    #   positions 首维取值 —— delete[2]。
    # SUBTRACTED: vllm/v1/spec_decode/llm_base_proposer.py:L861-L960 ——
    #   needs_extra_input_slots==True 的 DFlash/draft_model 分支
    #   （copy_and_expand_eagle_inputs_kernel/compute_new_slot_mapping/
    #   extend_all_queries_by_N）—— delete[2]；默认 EAGLE 分支即完整路径。
    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
    ):
        # Default EAGLE pathway: no reshaping of input tensors needed.
        # Simply rotate the input ids and leave the positions unchanged,
        # Inserting the next token ids at the last slot in each request.
        # SOURCE: vllm/v1/spec_decode/llm_base_proposer.py:L821-L960 set_inputs_first_pass（默认 EAGLE 分支：input_ids 左移一格）
        if token_indices_to_sample is None:
            token_indices_to_sample = cad.query_start_loc[1:] - 1

        num_tokens = target_token_ids.shape[0]
        # Shift the input ids by one token.
        # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        self.input_ids[: num_tokens - 1] = target_token_ids[1:]
        # Replace the last token with the next token.
        # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
        self.input_ids[token_indices_to_sample] = next_token_ids

        self._set_positions(num_tokens, target_positions)

        self.hidden_states[:num_tokens] = target_hidden_states

        return num_tokens, token_indices_to_sample, cad

    # SOURCE: vllm/v1/spec_decode/llm_base_proposer.py:L962-L991 build_model_inputs_first_pass
    #   —— 纯 input_ids 主路径逐字（多模态分支已按 delete[2] 减去）
    # SUBTRACTED: vllm/v1/spec_decode/llm_base_proposer.py:L968-L978 ——
    #   supports_mm_inputs 的 inputs_embeds 装配分支 —— delete[2]。
    def build_model_inputs_first_pass(
        self,
        num_tokens: int,
        num_input_tokens: int,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ):
        # SOURCE: vllm/v1/spec_decode/llm_base_proposer.py:L962-L991 build_model_inputs_first_pass（纯 input_ids 主路径）
        input_ids = self.input_ids[:num_input_tokens]
        inputs_embeds = None

        model_kwargs = {
            "input_ids": input_ids,
            "positions": self._get_positions(num_input_tokens),
            "inputs_embeds": inputs_embeds,
        }
        if self.pass_hidden_states_to_model:
            model_kwargs["hidden_states"] = self.hidden_states[:num_input_tokens]

        return model_kwargs, num_input_tokens

    # SUBTRACTED: vllm/v1/spec_decode/llm_base_proposer.py:L993-L1005
    #   build_per_group_and_layer_attn_metadata —— 注意力组元数据构建（栈未载）。

    # SOURCE: vllm/v1/spec_decode/llm_base_proposer.py:L1007-L1015 model_returns_tuple
    #   —— 逐字（mtp 按架构判 / eagle 系 True / draft_model·dflash False）
    def model_returns_tuple(self) -> bool:
        # SOURCE: vllm/v1/spec_decode/llm_base_proposer.py:L1007-L1015 model_returns_tuple（mtp 按架构判）
        if self.method == "mtp":
            # These models return separate hidden states for logits and for
            # feedback into the next draft step.
            architectures = self.draft_model_config.hf_config.architectures or []
            return bool(
                {"DeepSeekMTPModel", "KimiK3MTPModel"}.intersection(architectures)
            )
        return self.method not in ("mtp", "draft_model", "dflash")

    # SUBTRACTED: vllm/v1/spec_decode/llm_base_proposer.py:L1017-L1269
    #   prepare_next_token_ids_cpu/prepare_next_token_ids_padded/prepare_inputs_
    #   padded/prepare_inputs —— 下一拍输入装配族（持久批/CPU token 侧与
    #   padded batch 的 GPU kernel 装配；依赖 gpu_input_batch 与融合 kernel）。
    # SUBTRACTED: vllm/v1/spec_decode/llm_base_proposer.py:L1271-L1625
    #   get_model_name/_create_draft_vllm_config/_get_model/load_model/
    #   _maybe_share_embeddings/_maybe_share_lm_head —— 模型加载与 embed/lm_head
    #   权重共享（显存大头省法，正文内嵌解读；依赖 model_loader/parallel_state）。
    # SUBTRACTED: vllm/v1/spec_decode/llm_base_proposer.py:L1627-L1795
    #   dummy_run/_get_eagle3_use_aux_hidden_state_from_config/
    #   validate_same_kv_cache_group/initialize_attn_backend —— 图捕获预热与
    #   注意力后端装配（栈未载）。
    # SUBTRACTED: vllm/v1/spec_decode/llm_base_proposer.py:L1797-L1839
    #   _determine_batch_execution_and_padding —— cudagraph padding 与 DP 协调
    #   （delete[2]：精简版固定 input_batch_size==batch_size）。


# SOURCE: vllm/v1/spec_decode/llm_base_proposer.py:L1842-L1886 compute_probs_and_sample_next_token
#   —— 逐字（概率化 draft_probs：温度除 → softmax → Exp(1) 噪声 Gumbel-max；
#   头部 NOTE 原话 "Currently, the below code is not used and we always use
#   argmax to sample the draft tokens"——开启 draft_sample_method="probabilistic"
#   才进此路径）
# NOTE(woosuk): Currently, the below code is not used and we always use argmax
# to sample the draft tokens. We will use this after we find a way to manage
# the draft prob tensor.
# Refer to https://github.com/vllm-project/vllm/pull/16899 for the details.
# FIXME(woosuk): The logic here is duplicated with the main sampling code.
# We should refactor this to reuse the same sampling implementation.
def compute_probs_and_sample_next_token(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    use_fp64_gumbel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # SOURCE: vllm/v1/spec_decode/llm_base_proposer.py:L1848-L1886 compute_probs_and_sample_next_token（概率化 draft_probs）
    if sampling_metadata.all_greedy:
        # For greedy requests, draft_probs is not used in rejection sampling.
        # Therefore, we can just return the logits.
        probs = logits
        next_token_ids = logits.argmax(dim=-1)
        return next_token_ids, probs

    assert sampling_metadata.temperature is not None

    # Use epsilon comparison to detect greedy sampling (temperature ~ 0.0)
    # consistent with sampler.py's _SAMPLING_EPS threshold
    temperature = sampling_metadata.temperature
    # Avoid division by zero if there are greedy requests.
    if not sampling_metadata.all_random:
        is_greedy = temperature < _SAMPLING_EPS
        temperature = torch.where(is_greedy, 1.0, temperature)
    logits.div_(temperature.view(-1, 1))
    probs = logits.softmax(dim=-1, dtype=torch.float32)

    # NOTE(woosuk): Currently, we ignore most of the sampling parameters in
    # generating the draft tokens. We only use the temperature. While this
    # could degrade the acceptance rate, it does not affect the distribution
    # of the generated tokens after rejection sampling.

    # TODO(woosuk): Consider seeds.
    q = empty_exponential_noise_like(probs, use_fp64_gumbel)
    q.exponential_()
    # NOTE(woosuk): We shouldn't use `probs.div_(q)` because the draft_probs
    # will be used later for rejection sampling.
    next_token_ids = sample_with_exponential_noise(probs.clone(), q)
    if not sampling_metadata.all_random:
        greedy_token_ids = probs.argmax(dim=-1)
        next_token_ids = torch.where(is_greedy, greedy_token_ids, next_token_ids)
    return next_token_ids, probs
