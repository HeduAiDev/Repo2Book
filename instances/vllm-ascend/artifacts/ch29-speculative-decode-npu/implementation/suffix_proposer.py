# ch29 suffix_proposer.py —— subtract-only 精简版（全章最薄薄壳）
#
# 无 subtraction_plan.delete 批准的可删项 —— 逐字保留。
# propose 仅一行转发给父类（补上 self.runner.input_batch 参数），dummy_run 为 no-op，
# 新增状态只有 self.runner。对应 method='suffix'。

# SOURCE: vllm/v1/spec_decode/suffix_decoding.py（SuffixDecodingProposer 父类）
from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer


# SOURCE: vllm_ascend/spec_decode/suffix_proposer.py:L4
class AscendSuffixDecodingProposer(SuffixDecodingProposer):
    # SOURCE: vllm_ascend/spec_decode/suffix_proposer.py:L5
    def __init__(self, vllm_config, runner):
        super().__init__(vllm_config)
        self.runner = runner

    # SOURCE: vllm_ascend/spec_decode/suffix_proposer.py:L9
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
        pass

    # SOURCE: vllm_ascend/spec_decode/suffix_proposer.py:L23
    def propose(self, valid_sampled_token_ids):
        return super().propose(self.runner.input_batch, valid_sampled_token_ids)
