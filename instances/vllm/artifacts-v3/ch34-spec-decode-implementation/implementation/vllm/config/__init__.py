# SOURCE: vllm/config/__init__.py（真实为 config/ 包各类的 re-export 门面）
# HOST SEAM：配置面最小承载（ch30 同款做法，按 ch34 消费面扩展）。
# 本章消费面：
#   - NgramProposer（ngram_proposer.py:L13-L32）：speculative_config 的
#     prompt_lookup_min/max/num_speculative_tokens、model_config.max_model_len、
#     scheduler_config.max_num_seqs
#   - SpecDecodeBaseProposer（llm_base_proposer.py:L69-L131 契约骨架）：
#     draft_model_config.get_hidden_size()/get_inputs_embeds_size()/hf_config、
#     method/num_speculative_tokens/parallel_drafting/use_local_argmax_reduction/
#     use_heterogeneous_vocab/rejection_sample_method/draft_sample_method、
#     model_config.dtype/max_model_len/use_fp64_gumbel、
#     scheduler_config.max_num_seqs/max_num_batched_tokens、
#     parallel_config.data_parallel_rank
#   - build_logitsprocs（logits_processor/__init__.py:L202）：speculative_config
#     真值判断；MinP 构造器：scheduler_config.max_num_seqs
# 真实 VllmConfig/SpeculativeConfig 是 ch03 域的装配链（数千行）——此处以
# 同名字段载体镜像。
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


# SOURCE: vllm/config/scheduler.py SchedulerConfig —— HOST SEAM 字段面
#   （ch34 消费：max_num_seqs——ngram buffer 宽度/MinP 槽位预分配；
#   max_num_batched_tokens——drafter 输入 buffer 宽度）
@dataclass
class SchedulerConfig:
    # SOURCE: vllm/config/scheduler.py max_num_seqs 字段 —— HOST SEAM
    max_num_seqs: int = 8
    # SOURCE: vllm/config/scheduler.py max_num_batched_tokens 字段 —— HOST SEAM
    max_num_batched_tokens: int = 8192


# SOURCE: vllm/config/model.py ModelConfig —— HOST SEAM 字段面
#   （ch34 消费：max_model_len——ngram/llm_base_proposer；dtype/use_fp64_gumbel
#   ——drafter buffer 与组合 Sampler 的 fp64 Gumbel 开关）
@dataclass
class ModelConfig:
    # SOURCE: vllm/config/model.py max_model_len 字段 —— HOST SEAM
    max_model_len: int = 8192
    # SOURCE: vllm/config/model.py dtype 字段 —— HOST SEAM（真实为解析后的
    #   torch dtype；默认 float32 与 decode 主流一致）
    dtype: Any = torch.float32
    # SOURCE: vllm/config/model.py use_fp64_gumbel 字段 —— HOST SEAM
    use_fp64_gumbel: bool = False


# SOURCE: vllm/config/parallel.py ParallelConfig —— HOST SEAM 字段面
#   （ch34 消费：data_parallel_rank——drafter 的 DP rank 记账位）
@dataclass
class ParallelConfig:
    # SOURCE: vllm/config/parallel.py data_parallel_rank 字段 —— HOST SEAM
    data_parallel_rank: int = 0
    # SOURCE: vllm/config/parallel.py tensor_parallel_size 字段 —— HOST SEAM
    tensor_parallel_size: int = 1


# SOURCE: vllm/config/speculative.py —— HOST SEAM 字段面
#   真实 SpeculativeConfig 是投机解码全量配置（含 drafter 谱系注册校验），
#   ch34 只镜像 NgramProposer/SpecDecodeBaseProposer 读过的字段。
@dataclass
class DraftModelConfig:
    # SOURCE: vllm/config/speculative.py draft_model_config.get_hidden_size()
    #   —— HOST SEAM：方法位直存标量（草稿模型 hidden 宽度，可与 target 不同）
    hidden_size: int = 0
    # SOURCE: vllm/config/speculative.py draft_model_config.get_inputs_embeds_size()
    #   —— HOST SEAM：方法位直存标量
    inputs_embeds_size: int = 0
    # SOURCE: vllm/config/speculative.py draft_model_config.hf_config —— HOST SEAM
    hf_config: Any = None

    def get_hidden_size(self) -> int:
        return self.hidden_size

    def get_inputs_embeds_size(self) -> int:
        return self.inputs_embeds_size


# SOURCE: vllm/config/speculative.py SpeculativeConfig —— HOST SEAM 字段面
@dataclass
class SpeculativeConfig:
    # SOURCE: vllm/config/speculative.py method 字段（drafter 谱系方法名）—— HOST SEAM
    method: str = "eagle"
    # SOURCE: vllm/config/speculative.py num_speculative_tokens 字段（k）—— HOST SEAM
    num_speculative_tokens: int = 1
    # SOURCE: vllm/config/speculative.py prompt_lookup_min/max（ngram）—— HOST SEAM
    prompt_lookup_min: int | None = None
    prompt_lookup_max: int | None = None
    # SOURCE: vllm/config/speculative.py parallel_drafting 字段（DFlash 一次出 k）—— HOST SEAM
    parallel_drafting: bool = False
    # SOURCE: vllm/config/speculative.py use_local_argmax_reduction —— HOST SEAM
    use_local_argmax_reduction: bool = False
    # SOURCE: vllm/config/speculative.py use_heterogeneous_vocab —— HOST SEAM
    use_heterogeneous_vocab: bool = False
    # SOURCE: vllm/config/speculative.py rejection_sample_method —— HOST SEAM
    rejection_sample_method: str = "standard"
    # SOURCE: vllm/config/speculative.py draft_sample_method —— HOST SEAM
    draft_sample_method: str = "greedy"
    # SOURCE: vllm/config/speculative.py draft_model_config —— HOST SEAM
    draft_model_config: DraftModelConfig | None = None


# SOURCE: vllm/config/vllm.py:L331 起 VllmConfig —— HOST SEAM 字段面
@dataclass
class VllmConfig:
    # SOURCE: vllm/config/vllm.py model_config 字段 —— HOST SEAM
    model_config: ModelConfig = field(default_factory=ModelConfig)
    # SOURCE: vllm/config/vllm.py scheduler_config 字段 —— HOST SEAM
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    # SOURCE: vllm/config/vllm.py parallel_config 字段 —— HOST SEAM
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    # SOURCE: vllm/config/vllm.py speculative_config 字段 —— HOST SEAM
    speculative_config: SpeculativeConfig | None = None
