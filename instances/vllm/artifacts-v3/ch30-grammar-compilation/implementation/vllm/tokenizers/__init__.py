# SOURCE: vllm/tokenizers/registry.py（经 vllm/tokenizers/__init__.py 聚合导出）
# HOST SEAM：本章消费面只有 cached_tokenizer_from_config（structured_output/
# __init__.py:L78 构造引擎级 tokenizer）与 TokenizerLike 类型名（sampling_
# params.py:L21 的注解面）。真实 vllm/tokenizers 是大包（registry 的分词器
# 后端路由/deepseek/kimi/mistral 各家专版）——退化为 transformers
# AutoTokenizer 直载；语义与真实 get_tokenizer 一致（HF fast tokenizer）。
# SUBTRACTED: SPDX 版权头；registry 的 tokenizer_mode/runner_type 路由面
#   （_maybe_register_hf_config/tokenizer_cls_ 选型 L236-L258）与各家专版模块。
from functools import lru_cache
from typing import Any

from transformers import AutoTokenizer

# HOST SEAM：真实 TokenizerLike 是 vllm/tokenizers/protocol.py 的运行时检查协议
TokenizerLike = Any


# HOST SEAM（真实 registry.get_tokenizer 的最小承载——L287 的
# tokenizer_cls_.from_pretrained(tokenizer_name)）
def get_tokenizer(tokenizer_name: str, **kwargs):
    return AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)


# SOURCE: vllm/tokenizers/registry.py:L267 cached_get_tokenizer
cached_get_tokenizer = lru_cache(get_tokenizer)


# SOURCE: vllm/tokenizers/registry.py:L270-L285 cached_tokenizer_from_config
def cached_tokenizer_from_config(model_config: "ModelConfig", **kwargs):
    if model_config.skip_tokenizer_init:
        return None

    # SUBTRACTED: _maybe_register_hf_config(getattr(model_config, "hf_config",
    #   None))（L277——HF config 专版注册面）与 runner_type/tokenizer_mode/
    #   tokenizer_revision/trust_remote_code 四个路由参数（L279-L284——
    #   HOST SEAM ModelConfig 只带 tokenizer 名）。

    return cached_get_tokenizer(
        model_config.tokenizer,
        **kwargs,
    )
