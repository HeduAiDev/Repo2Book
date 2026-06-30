# ch29 ngram_proposer.py —— subtract-only 精简版（真干活的 CPU n-gram 薄壳）
#
# 无 subtraction_plan.delete 批准的可删项 —— 逐字保留。
# 与 ngram_proposer_npu 形成对照：load_model/dummy_run 为 no-op，但 propose 真干活 ——
# 把新采样 token 写回 input_batch.token_ids_cpu，再交父类 NgramProposer.batch_propose 做
# 纯 CPU 的 n-gram 匹配（device 无关，host 可跑读控制流）。对应 method='ngram'。
import torch

# SOURCE: vllm/v1/spec_decode/ngram_proposer.py（NgramProposer 父类）
from vllm.v1.spec_decode.ngram_proposer import NgramProposer


# SOURCE: vllm_ascend/spec_decode/ngram_proposer.py:L5
class AscendNgramProposer(NgramProposer):
    # SOURCE: vllm_ascend/spec_decode/ngram_proposer.py:L6
    def __init__(self, vllm_config, runner):
        self.runner = runner
        super().__init__(vllm_config)

    # SOURCE: vllm_ascend/spec_decode/ngram_proposer.py:L10
    def load_model(self, *args, **kwargs):
        # No model to load.
        pass

    # SOURCE: vllm_ascend/spec_decode/ngram_proposer.py:L14
    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens,
        with_prefill=None,
        in_graph_capturing=None,
        num_reqs=None,
        num_tokens_across_dp=None,
        aclgraph_runtime_mode=None,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
    ):
        # SOURCE: vllm_ascend/spec_decode/ngram_proposer.py:L14
        pass

    # SOURCE: vllm_ascend/spec_decode/ngram_proposer.py:L29
    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec=None,
        token_ids_cpu=None,
        slot_masks: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
    ) -> list[list[int]]:
        valid_ngram_requests = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            num_sampled_ids = len(sampled_ids)
            if not num_sampled_ids:
                continue

            req_id = self.runner.input_batch.req_ids[i]
            if req_id in self.runner.input_batch.spec_decode_unsupported_reqs:
                continue

            num_tokens = self.runner.input_batch.num_tokens_no_spec[i]
            if num_tokens >= self.runner.input_batch.max_model_len:
                # Skip requests that have already reached the max model length.
                continue

            start_idx = self.runner.input_batch.num_tokens_no_spec[i]
            end_idx = start_idx + num_sampled_ids
            self.runner.input_batch.token_ids_cpu[i, start_idx:end_idx] = sampled_ids

            valid_ngram_requests.append(i)

        draft_token_ids = self.batch_propose(
            len(sampled_token_ids),
            valid_ngram_requests,
            self.runner.input_batch.num_tokens_no_spec,
            self.runner.input_batch.token_ids_cpu,
        )

        return draft_token_ids
