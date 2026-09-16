# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/config/__init__.py —— HOST SEAM（最小承载）：真实包是
# 数千行强校验配置族（ch3 域）。本章服务面只消费 ModelConfig 的只读属性面
# （model/hf_config.model_type/max_model_len/get_diff_sampling_param…）与
# VllmConfig 的组合配置挂点（model_config/lora_config/shutdown_timeout）。
# SUBTRACTED: 其余 ~30 个配置节（Cache/Scheduler/ParallelConfig…）——
# 各归其章（ch3/ch13/ch15）。
from typing import Any, Optional


# SOURCE: vllm/config/model.py:L1593 —— get_diff_sampling_param 的白名单
# （L1615-L1622：仅 repetition_penalty/temperature/top_k/top_p/min_p/
# max_new_tokens 六项可进 server 默认参数；无 stop_token_ids——
# dossier m3 现核结论：default_stop_ids 恒 None、合并分支不触发）
_DIFF_SAMPLING_PARAM_WHITELIST = (
    "repetition_penalty",
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "max_new_tokens",
)


# SOURCE: vllm/config/model.py —— HOST SEAM：hf_config 形状（真实为
# transformers PretrainedConfig 包装；本章只读 model_type）
class _HfConfigLike:
    # SOURCE: vllm/config/model.py —— HOST SEAM：model_type 属性位
    def __init__(self, model_type: str = "llama"):
        self.model_type = model_type


# SOURCE: vllm/config/model.py:L152 —— HOST SEAM：ModelConfig 消费面属性
class ModelConfig:
    # SOURCE: vllm/config/model.py —— HOST SEAM：最小构造位
    def __init__(
        self,
        model: str = "test-model",
        model_type: str = "llama",
        max_model_len: int = 4096,
        generation_config: str = "auto",
        override_generation_config: dict | None = None,
        diff_sampling_param: dict[str, Any] | None = None,
        enable_prompt_embeds: bool = False,
    ):
        self.model = model
        self.hf_config = _HfConfigLike(model_type)
        self.hf_text_config = None
        self.hf_overrides = None
        self.max_model_len = max_model_len
        self.served_model_name = model
        self.generation_config = generation_config
        self.override_generation_config = override_generation_config or {}
        self._diff_sampling_param = diff_sampling_param or {}
        self.enable_prompt_embeds = enable_prompt_embeds
        self.multimodal_config = None
        # SUBTRACTED: ModelConfig 其余 ~200 属性与校验器（ch3 域）。

    # SOURCE: vllm/config/model.py:L1593-L1633 —— get_diff_sampling_param
    # （HOST SEAM：真实从 generation_config 文件读六项白名单差量；host
    # 精简环境直接回放构造期给定的差量字典，白名单语义一致）
    def get_diff_sampling_param(self) -> dict[str, Any]:
        config = self._diff_sampling_param
        available_params = list(_DIFF_SAMPLING_PARAM_WHITELIST)
        if any(p in config for p in available_params):
            diff_sampling_param = {
                p: config.get(p) for p in available_params if config.get(p) is not None
            }
            # Huggingface definition of max_new_tokens is equivalent
            # to vLLM's max_tokens
            if "max_new_tokens" in diff_sampling_param:
                diff_sampling_param["max_tokens"] = diff_sampling_param.pop(
                    "max_new_tokens"
                )
        else:
            diff_sampling_param = {}
        return diff_sampling_param


# SOURCE: vllm/config/cache.py —— HOST SEAM：CacheConfig 形状位（async_llm
# add_request 的 kv_sharing_fast_prefill 校验消费；真实为完整缓存配置节）
class CacheConfig:
    # SOURCE: vllm/config/cache.py —— HOST SEAM：kv_sharing_fast_prefill 位
    def __init__(self):
        self.kv_sharing_fast_prefill = False


# SOURCE: vllm/config/lora.py —— HOST SEAM：LoRAConfig 形状位
# （init_app_state 的 default_mm_loras 消费）
class LoRAConfig:
    # SOURCE: vllm/config/lora.py —— HOST SEAM：default_mm_loras 位
    def __init__(self):
        self.default_mm_loras: dict[str, str] | None = None


# SOURCE: vllm/config/vllm.py —— HOST SEAM：VllmConfig 组合挂点
class VllmConfig:
    # SOURCE: vllm/config/vllm.py —— HOST SEAM：最小构造位
    def __init__(
        self,
        model_config: ModelConfig | None = None,
        shutdown_timeout: float | None = 300.0,
    ):
        self.model_config = model_config or ModelConfig()
        self.cache_config = CacheConfig()
        self.lora_config: LoRAConfig | None = None
        self.kv_transfer_config = None
        self.shutdown_timeout = shutdown_timeout
        # SUBTRACTED: VllmConfig 其余配置节挂点（ch3 域）。


# SOURCE: vllm/config/speculative.py —— HOST SEAM：SpeculativeConfig 占位
class SpeculativeConfig:
    pass


# SOURCE: vllm/config/structured_outputs.py —— HOST SEAM：StructuredOutputsConfig 占位
class StructuredOutputsConfig:
    # SOURCE: vllm/config/structured_outputs.py —— HOST SEAM：
    # reasoning_parser 字段位（init_generate_state 消费）
    def __init__(self, reasoning_parser: str = ""):
        self.reasoning_parser = reasoning_parser
