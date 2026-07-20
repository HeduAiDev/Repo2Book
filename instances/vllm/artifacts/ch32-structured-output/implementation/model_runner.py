# SOURCE: vllm/v1/worker/gpu/model_runner.py
# 只做减法的忠实精简版。GPUModelRunner 是新 gpu worker 路径（V2，opt-in）里的模型
# 运行器；本章只抽取 sample() 方法里「掩码落地」的落点——compute_logits 之后、
# sampler 之前，原地把 grammar bitmask 应用到 logits 上。
#
# SUBTRACTED: SPDX 版权头、GPUModelRunner 其余部分（prepare_inputs/attention 元数据
# 构建/CUDA graph 捕获等，属 ch18/ch19 与 ch30/ch34 范围）、sample() 方法后半段
# （`if input_batch.num_draft_tokens == 0:` 起的 sampler / rejection_sampler 分派，
# model_runner.py:L924+，属 ch30/ch34——subtraction_plan 批准项4）。
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from output import GrammarOutput

if TYPE_CHECKING:
    from input_batch import InputBatch
    from structured_outputs import StructuredOutputsWorker


@dataclass
class GPUModelRunner:
    # 非源码符号：本章只需要 sample() 依赖的两个协作对象，真实类里它们是
    # __init__ 挂载的实例属性（model / structured_outputs_worker），这里用
    # dataclass 字段代替完整的 __init__（后者依赖 vllm_config/device 等本章未建模
    # 的构造参数）。
    model: "object"
    structured_outputs_worker: "StructuredOutputsWorker | None"

    def sample(
        self,
        hidden_states: torch.Tensor,
        input_batch: "InputBatch",
        grammar_output: "GrammarOutput | None",
    ):
        # SOURCE: vllm/v1/worker/gpu/model_runner.py:L906-922
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
        # SUBTRACTED: 从这里开始的采样器分派（num_draft_tokens==0 的常规采样 /
        # >0 时投机解码的 rejection_sampler，model_runner.py:L924+）属 ch30/ch34。
        # 精简版直接把 (被原地改写过的) logits 返回，让本章测试能断言掩码效果——
        # 真实签名返回的是 (SamplerOutput, torch.Tensor, torch.Tensor)。
        return logits
