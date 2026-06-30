# ch29 eagle_proposer.py —— subtract-only 精简版（走重量级 base 的薄入口）
#
# 整文件 19 行，无 subtraction_plan.delete 批准的可删项 —— 逐字保留。
# 与 draft_proposer 并列对照：同样多继承 (vLLM 策略类, AscendSpecDecodeBaseProposer)，
# __init__ 仅转调昇腾 base 构造，唯一差异是 pass_hidden_states_to_model=True
# （eagle 需把 target hidden states 喂进 draft 模型）。method='eagle'/'eagle3'/'mtp' 共用。
import torch
from vllm.config import VllmConfig
from vllm.v1.spec_decode.eagle import EagleProposer

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


# SOURCE: vllm_ascend/spec_decode/eagle_proposer.py:L10
class AscendEagleProposer(EagleProposer, AscendSpecDecodeBaseProposer):
    # SOURCE: vllm_ascend/spec_decode/eagle_proposer.py:L11
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        AscendSpecDecodeBaseProposer.__init__(
            self, vllm_config, device, pass_hidden_states_to_model=True, runner=runner
        )
