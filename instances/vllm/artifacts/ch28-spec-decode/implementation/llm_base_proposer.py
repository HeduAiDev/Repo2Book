# SUBTRACTED: SPDX 版权头与一大批 import（attention backend / cudagraph dispatcher /
#             各 EAGLE/EAGLE3/DFlash 模型类 / set_forward_context / 融合 kernel 等）。
#
# 注意：SpecDecodeBaseProposer 是『模型类 proposer』（EAGLE/EAGLE3/DFlash/MTP/
# draft_model）的统一基类，它依赖完整的 vLLM 模型与注意力栈，无法脱离 vLLM 单独运行。
# 本精简版保留它对外的**契约骨架**——吃目标 token/position/hidden_states，跑草稿模型
# 前向，吐每请求 num_speculative_tokens 个草稿 token——展示 EAGLE 主路径的两条分支
# （单步早退 / 自回归多步链式），其内部模型结构（EAGLE 头、MTP）见 ch25。
# 其余字段/方法（__init__ 的全部配置、注意力 metadata 构建、cudagraph、slot mapping）
# 不在本章范围，未内嵌。
from __future__ import annotations

import torch


# SOURCE: vllm/v1/spec_decode/llm_base_proposer.py:L60 (SpecDecodeBaseProposer)
class SpecDecodeBaseProposer:
    # 以下属性由真实 __init__ 从 SpeculativeConfig 派生（见
    # vllm/v1/spec_decode/llm_base_proposer.py:L61-L110）。精简版仅列出 propose
    # 主路径用到的几个，作为契约说明；不内嵌完整构造。
    #   self.method: str                 # "eagle"/"eagle3"/"dflash"/"mtp"/"draft_model"
    #   self.num_speculative_tokens: int # 每请求要产的草稿数 k
    #   self.parallel_drafting: bool     # DFlash 一次出全部 k 个草稿
    #   self.model: nn.Module            # 草稿模型（EAGLE 头 / MTP，见 ch25）

    # SOURCE: vllm/v1/spec_decode/llm_base_proposer.py:L407-L411
    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy-sample draft tokens from hidden states."""
        # SUBTRACTED: use_local_argmax_reduction 分支（L409-L410）—— 局部 argmax 归约
        #             优化；主路径直接对草稿模型 logits 取 argmax。
        return self.model.compute_logits(hidden_states).argmax(dim=-1)

    # SOURCE: vllm/v1/spec_decode/llm_base_proposer.py:L413-L655
    # SUBTRACTED: tree attention 分支（propose_tree / TreeAttentionMetadata，L502-L514,
    #             L518-L526）—— 高级 tree drafting 路径，subtraction_plan.delete 批准。
    # SUBTRACTED: M-RoPE / xdrope position 分支（uses_mrope / uses_xdrope_dim 三处 if/elif，
    #             L496-L500, L562-L591）—— 位置编码工程旁路；精简版固定 1D positions。
    # SUBTRACTED: 多模态（supports_mm_inputs / inputs_embeds / mm_embed_inputs）与
    #             cudagraph padding（_determine_batch_execution_and_padding / input_batch_size
    #             vs batch_size，L464-L470, L531-L533, L616-L640）—— 性能/兼容旁路；
    #             精简版令 input_batch_size==batch_size、走纯 input_ids 路径。
    # SUBTRACTED: parallel_drafting/DFlash 专属 set_inputs_first_pass 的 needs_extra_input_slots
    #             分支（copy_and_expand_eagle_inputs_kernel 等）—— 见 set_inputs_first_pass
    #             默认 EAGLE 分支即可；DFlash『一次出 k』概念由 self.parallel_drafting 早退体现。
    def propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens]
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata,
        sampling_metadata,
    ) -> torch.Tensor:
        # SOURCE: vllm/v1/spec_decode/llm_base_proposer.py:L413-L655
        batch_size = common_attn_metadata.batch_size()

        if self.method in ("eagle3", "dflash"):
            # EAGLE3/DFlash 把目标模型多层 hidden_states 融合成单个草稿输入。
            target_hidden_states = self.model.combine_hidden_states(
                target_hidden_states
            )

        # 把目标模型的 token/position/hidden_state 摆进草稿模型的输入 buffer（默认
        # EAGLE 路径：input_ids 整体左移一格、最后槽填入 next_token）。
        num_tokens, token_indices_to_sample, common_attn_metadata = (
            self.set_inputs_first_pass(
                target_token_ids=target_token_ids,
                next_token_ids=next_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                token_indices_to_sample=token_indices_to_sample,
                cad=common_attn_metadata,
            )
        )

        # 跑一遍草稿模型前向，取出每请求要预测位置的 hidden_states。
        ret_hidden_states = self.model(
            input_ids=self.input_ids[:num_tokens],
            positions=self.positions[:num_tokens],
            hidden_states=self.hidden_states[:num_tokens],
        )
        if not self.model_returns_tuple():
            last_hidden_states = ret_hidden_states
            hidden_states = last_hidden_states
        else:
            last_hidden_states, hidden_states = ret_hidden_states

        sample_hidden_states = last_hidden_states[token_indices_to_sample]

        # Early exit if there is only one draft token to be generated.
        # （DFlash parallel_drafting 也走这里：一次前向直接出全部 k 个草稿。）
        if self.num_speculative_tokens == 1 or self.parallel_drafting:
            draft_token_ids = self._greedy_sample(sample_hidden_states)
            return draft_token_ids.view(-1, self.num_speculative_tokens)

        # ---- num_speculative_tokens > 1：自回归多步链式草稿（EAGLE）----
        positions = self.positions[token_indices_to_sample]
        hidden_states = hidden_states[token_indices_to_sample]

        draft_token_ids = self._greedy_sample(sample_hidden_states)
        draft_token_ids_list = [draft_token_ids]

        for _token_index in range(self.num_speculative_tokens - 1):
            # 把上一步采的 draft_token 当作下一步输入。
            # cast to int32 is crucial when eagle model is compiled.
            input_ids = draft_token_ids_list[-1].int()

            # SUBTRACTED: eagle_step_update_slot_mapping_and_metadata 融合 kernel
            #             与 seq_len/position/slot_mapping 的增量推进（L559-L611）——
            #             单 token decode 步的注意力 metadata 维护是引擎旁路；契约层
            #             只需展示『喂回上一草稿 → 前向 → greedy 采下一草稿』。

            # copy inputs to buffer for cudagraph
            self.input_ids[:batch_size] = input_ids
            self.hidden_states[:batch_size] = hidden_states

            # Run the draft model for one decode step.
            ret_hidden_states = self.model(
                input_ids=self.input_ids[:batch_size],
                positions=self.positions[:batch_size],
                hidden_states=self.hidden_states[:batch_size],
            )
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                hidden_states = ret_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states

            hidden_states = hidden_states[:batch_size]
            draft_token_ids = self._greedy_sample(last_hidden_states[:batch_size])
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids

    # SOURCE: vllm/v1/spec_decode/llm_base_proposer.py:L657-L689 (默认 EAGLE 分支)
    # SUBTRACTED: needs_extra_input_slots==True 的 DFlash/draft_model 分支
    #             （copy_and_expand_eagle_inputs_kernel 等，L690-L756）——
    #             subtraction_plan.delete 批准；精简版仅保留 EAGLE 主路径。
    # SUBTRACTED: uses_xdrope_dim position 调整（L683-L684）—— 位置编码旁路。
    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad,
    ):
        # SOURCE: vllm/v1/spec_decode/llm_base_proposer.py:L657-L689
        # Default EAGLE pathway: rotate input ids, insert next token ids at the
        # last slot in each request.
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
