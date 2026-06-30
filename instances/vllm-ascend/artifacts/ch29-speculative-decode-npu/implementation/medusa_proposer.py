# ch29 medusa_proposer.py —— subtract-only 精简版（中等薄壳）
#
# 无 subtraction_plan.delete 批准的可删项 —— 逐字保留。
# 「中等薄壳」标本：覆写 dummy_run（把 vLLM 前向上下文换成 set_ascend_forward_context，
# is_draft_model=True）+ propose（自己只做一段从拼接的 sample_hidden_states 里按已接受 token
# 数 gather 出每请求最后一个 hidden state 的索引计算，真正的 medusa 提议仍 super().propose
# 交父类）—— 典型「只在 device/数据布局接缝处插一层，核心算法照搬」。对应 method='medusa'。
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.sample.metadata import SamplingMetadata

# SOURCE: vllm/v1/spec_decode/medusa.py（MedusaProposer 父类）
from vllm.v1.spec_decode.medusa import MedusaProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm_ascend.ascend_forward_context import set_ascend_forward_context


# SOURCE: vllm_ascend/spec_decode/medusa_proposer.py:L10
class AscendMedusaProposer(MedusaProposer):
    """
    Medusa proposer class for generating token sequences
    """

    # SOURCE: vllm_ascend/spec_decode/medusa_proposer.py:L15
    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        in_graph_capturing: bool = False,
        num_reqs: int = 0,
        num_tokens_across_dp: torch.Tensor | None = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
    ):
        # SOURCE: vllm_ascend/spec_decode/medusa_proposer.py:L15
        hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        with set_ascend_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_tokens,
            num_actual_tokens=0,
            in_profile_run=is_profile,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
        ):
            self.model(hidden_states)
            dummy_compute_logits(hidden_states)

    # SOURCE: vllm_ascend/spec_decode/medusa_proposer.py:L46
    def propose(
        self,
        valid_sampled_token_ids: list[list[int]],
        sampling_metadata: SamplingMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
        sample_hidden_states: torch.Tensor,
    ):
        if sample_hidden_states.shape[0] == len(valid_sampled_token_ids):
            # The input to the target model does not include draft tokens.
            hidden_states = sample_hidden_states
        else:
            num_accepted_tokens = torch.tensor(
                [len(t) for t in valid_sampled_token_ids], device=self.device, dtype=torch.long
            )
            num_draft_tokens = torch.tensor(spec_decode_metadata.num_draft_tokens, device=self.device, dtype=torch.long)

            offsets = torch.cumsum(num_draft_tokens + 1, dim=0) - (num_draft_tokens + 1)
            indices = offsets + num_accepted_tokens - 1
            hidden_states = sample_hidden_states[indices]

        spec_token_ids = super().propose(
            target_hidden_states=hidden_states,
            sampling_metadata=sampling_metadata,
        )
        return spec_token_ids
