# SOURCE: vllm/config/__init__.py
# HOST SEAM：VllmConfig 对象图的最小字段面。真实 VllmConfig 是 pydantic 大
# 配置树（ch03 域）——本章只消费 model_config（get_vocab_size/
# skip_tokenizer_init/is_diffusion）、parallel_config（distributed_executor_
# backend）、scheduler_config（max_num_seqs/max_num_batched_tokens）、
# structured_outputs_config（详见 config/structured_outputs.py 近逐字镜像）、
# speculative_config（num_speculative_tokens）五个字段的读取面。
# SUBTRACTED: SPDX 版权头；真实 vllm/config/__init__.py 的数百行聚合导出。
from dataclasses import dataclass


# SOURCE: vllm/config/model.py ModelConfig —— HOST SEAM 字段面
#   （本章消费位：grammar_init 的 get_vocab_size（__init__.py:L132）、
#   skip_tokenizer_init 守卫（L70）、_validate_structured_outputs 的
#   is_diffusion 拒单（sampling_params.py:L932））
@dataclass
class ModelConfig:
    tokenizer: str = "gpt2"
    vocab_size: int = 50257
    is_diffusion: bool = False
    skip_tokenizer_init: bool = False
    max_model_len: int = 8192

    def get_vocab_size(self) -> int:
        # SOURCE: vllm/config/model.py get_vocab_size —— HOST SEAM
        #   （真实走 hf_text_config.vocab_size；HOST SEAM 直存）
        return self.vocab_size


# SOURCE: vllm/config/parallel.py ParallelConfig —— HOST SEAM 字段面
#   （本章消费位：_use_async_grammar_compilation 的 external_launcher 判定，
#   structured_output/__init__.py:L52-L55）
@dataclass
class ParallelConfig:
    distributed_executor_backend: str = "mp"


# SOURCE: vllm/config/scheduler.py SchedulerConfig —— HOST SEAM 字段面
#   （本章消费位：Scheduler 的 max_num_running_reqs/max_num_scheduled_tokens）
@dataclass
class SchedulerConfig:
    max_num_seqs: int = 16
    max_num_batched_tokens: int = 8192
    max_num_scheduled_tokens: int | None = None
    policy: str = "fcfs"
    # SUBTRACTED: 真实 SchedulerConfig 另有 chunked prefill/抢占/长 prefill
    #   阈值等数十字段（ch10/ch11/ch12 域）。


# SOURCE: vllm/config/speculative.py SpeculativeConfig —— HOST SEAM 字段面
#   （本章消费位：XgrammarBackend 的 num_speculative_tokens=
#   max_rollback_tokens，backend_xgrammar.py:L72-L76）
@dataclass
class SpeculativeConfig:
    num_speculative_tokens: int = 1


# SOURCE: vllm/config/__init__.py VllmConfig —— HOST SEAM 字段面
#   （真实 VllmConfig 字段远多于此——cache/lora/kv_events/observability 等面
#   本章不消费）
@dataclass
class VllmConfig:
    model_config: ModelConfig
    parallel_config: ParallelConfig
    scheduler_config: SchedulerConfig
    structured_outputs_config: "StructuredOutputsConfig"
    speculative_config: SpeculativeConfig | None = None


# SOURCE: vllm/config/__init__.py 的 StructuredOutputsConfig re-export
#   （真实经 vllm/config/__init__.py 从 config/structured_outputs.py 聚合）
from vllm.config.structured_outputs import StructuredOutputsConfig  # noqa: E402
