# ch29 dflash_proposer.py —— subtract-only 精简版（继承链延伸标本）
#
# 只删 subtraction_plan.delete 批准项：
#   - 删 set_inputs_first_pass / dummy_run / build_model_inputs_first_pass / _raise_if_multimodal
#     的 DFlash 专属 Triton kernel 调用与缓冲细节（原文件 L63-L267）：DFlash 是 eagle 之上的
#     进一步特化，本章把它作为「继承链延伸」一笔带过；其 Triton kernel 需 NPU/CANN 不真跑。
# 保留：类定义 + 继承 AscendEagleProposer + __init__（建 DFlash 交叉注意力专属缓冲），
# 体现「在 eagle 重量级之上再叠一层」的二级继承。对应 method='dflash'。
import torch
from vllm.config import VllmConfig

from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer


# SOURCE: vllm_ascend/spec_decode/dflash_proposer.py:L15
class AscendDflashProposer(AscendEagleProposer):
    # SOURCE: vllm_ascend/spec_decode/dflash_proposer.py:L16
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            vllm_config,
            device,
            runner=runner,
        )

        self.max_query_tokens = self.max_batch_size * (1 + self.num_speculative_tokens)
        self.max_positions = self.max_num_tokens + self.max_query_tokens

        self._context_slot_mapping_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int32,
            device=device,
        )

        self._slot_mapping_buffer = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )

        self._context_positions_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int32,
            device=device,
        )

        self.positions = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )

        self.arange_dflash = torch.arange(self.max_positions + 1, device=device, dtype=torch.int32)

        self._dflash_hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size), dtype=self.dtype, device=self.device
        )

        self.parallel_drafting_hidden_state_tensor = None

    # SUBTRACTED: set_inputs_first_pass / dummy_run / build_model_inputs_first_pass /
    #   _raise_if_multimodal（原 dflash_proposer.py:L63-L267）—— DFlash 专属交叉注意力缓冲与
    #   Triton kernel（copy_and_expand_dflash_inputs_kernel_single_grid / precompute_and_store_context_kv），
    #   需 NPU/CANN 不真跑；本章作「继承链延伸」一笔带过，删后不影响工厂/薄壳/继承立意。
