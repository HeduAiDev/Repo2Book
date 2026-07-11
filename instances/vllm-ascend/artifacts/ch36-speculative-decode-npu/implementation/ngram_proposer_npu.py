# ch29 ngram_proposer_npu.py —— subtract-only 精简版（薄壳之极致：no-op stub）
#
# 整文件仅 35 行，无 subtraction_plan.delete 批准的可删项 —— 逐字保留。
# 这是「薄壳」标本：直接继承 vLLM 的 NgramProposerGPU，但 load_model/dummy_run/propose
# 三个 device 相关方法全部 no-op（propose 体是裸 pass，并不复用父类那段 GPU 批量 n-gram
# kernel）；继承父类只为沿用其构造/形状约定，真正的 ngram 提议走 method='ngram' 的 CPU 路径。
import torch

# SOURCE: vllm/v1/spec_decode/ngram_proposer_gpu.py（NgramProposerGPU 父类）
from vllm.v1.spec_decode.ngram_proposer_gpu import NgramProposerGPU


# SOURCE: vllm_ascend/spec_decode/ngram_proposer_npu.py:L5
class AscendNgramProposerNPU(NgramProposerGPU):
    # SOURCE: vllm_ascend/spec_decode/ngram_proposer_npu.py:L6
    def __init__(self, vllm_config, device: torch.device, runner):
        super().__init__(vllm_config, device=device)

    # SOURCE: vllm_ascend/spec_decode/ngram_proposer_npu.py:L9
    def load_model(self, *args, **kwargs):
        # No model to load.
        pass

    # SOURCE: vllm_ascend/spec_decode/ngram_proposer_npu.py:L13
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
        # SOURCE: vllm_ascend/spec_decode/ngram_proposer_npu.py:L13
        pass

    # SOURCE: vllm_ascend/spec_decode/ngram_proposer_npu.py:L28
    def propose(
        self,
        num_tokens_no_spec: torch.Tensor,  # [batch_size]
        token_ids_gpu: torch.Tensor,  # [batch_size, max_len]
        valid_sampled_token_ids_gpu: torch.Tensor,  # [batch_size, num_spec_tokens + 1]
        valid_sampled_tokens_count: torch.Tensor,  # [batch_size]
    ):
        # 注意：此 propose 体是裸 pass，不复用父类 NgramProposerGPU 的 GPU 批量 n-gram kernel
        # （对照 vllm/v1/spec_decode/ngram_proposer_gpu.py 父类 propose 是真实 GPU 实现）。
        pass
