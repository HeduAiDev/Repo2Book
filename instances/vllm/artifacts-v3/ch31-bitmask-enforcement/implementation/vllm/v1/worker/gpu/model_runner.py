# SOURCE: vllm/v1/worker/gpu/model_runner.py
# v3 ch31 脊柱⑩：V2 runner 的 sample() 调用点（L1143-L1175——与 V1 同一落点
# 次序：hidden_states[logits_indices] 切片 → compute_logits →
# grammar_output 非空先 structured_outputs_worker.apply_grammar_bitmask 原地改
# logits → 才进 sampler/rejection_sampler 分派）。
# SUBTRACTED（delete[6]）：V2 model_runner 的其余全部（注意力/slot/PP/
#   execute_model 面——归 ch12/ch18/ch19 的主线；V2 落地差异只在本调用点）。
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput
    from vllm.v1.outputs import SamplerOutput
    from vllm.v1.worker.gpu.input_batch import InputBatch


# SOURCE: vllm/v1/worker/gpu/model_runner.py:L127 GPUModelRunner（V2）—— 本章切面
class GPUModelRunner:
    # SUBTRACTED: 真实 __init__ L128-L1142（structured_outputs_worker 构造位
    #   L375、attn/cudagraph/sampler 装配——ch17-ch19）；ENGINE SEAM：
    #   model/structured_outputs_worker/sampler/rejection_sampler 由外部注入。

    # SOURCE: vllm/v1/worker/gpu/model_runner.py:L1143-L1175 sample —— 逐字
    def sample(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        grammar_output: GrammarOutput | None,
    ) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor]:
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)
        if grammar_output is not None:
            # Apply grammar bitmask to the logits in-place.
            assert self.structured_outputs_worker is not None
            self.structured_outputs_worker.apply_grammar_bitmask(
                logits,
                input_batch,
                grammar_output.structured_output_request_ids,
                grammar_output.grammar_bitmask,
            )

        if input_batch.num_draft_tokens == 0 or self.rejection_sampler is None:
            assert self.sampler is not None
            sampler_output = self.sampler(logits, input_batch)
        else:
            # Rejection sampling for spec decoding.
            assert self.rejection_sampler is not None
            assert self.speculator is not None
            sampler_output = self.rejection_sampler(
                logits,
                input_batch,
                # Draft logits are needed for probabilistic rejection sampling.
                self.speculator.draft_logits,
            )

        return sampler_output, sampler_output.num_sampled, sampler_output.num_rejected
