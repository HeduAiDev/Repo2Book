# ch29 draft_proposer.py —— subtract-only 精简版（走重量级 base 的薄入口）
#
# 整文件 18 行，无 subtraction_plan.delete 批准的可删项 —— 逐字保留。
# 多继承顺序 (DraftModelProposer, AscendSpecDecodeBaseProposer)：拿 vLLM DraftModelProposer
# 的策略语义，但 __init__ 显式只调 AscendSpecDecodeBaseProposer.__init__
# （pass_hidden_states_to_model=False）——「行为复用基类、构造走昇腾重量级 base」。
# 对应 method='draft_model'。
import torch
from vllm.config import VllmConfig
from vllm.v1.spec_decode.draft_model import DraftModelProposer

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


# SOURCE: vllm_ascend/spec_decode/draft_proposer.py:L8
class AscendDraftModelProposer(DraftModelProposer, AscendSpecDecodeBaseProposer):
    # SOURCE: vllm_ascend/spec_decode/draft_proposer.py:L9
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        AscendSpecDecodeBaseProposer.__init__(self, vllm_config, device, False, runner=runner)
        self._raise_if_vocab_size_mismatch()
        self._raise_if_draft_tp_mismatch()
