"""配置与运行环境的忠实最小替身。

InputProcessor 真实地从 VllmConfig 取出一堆子配置（model_config / lora_config /
parallel_config / speculative_config / structured_outputs_config / reasoning_config 等）
并持有 renderer / tokenizer。这些都是『被读取的环境』而非本章算法主线。这里给出能让
process_inputs 主控制流真实跑起来的最小可配置替身，每处标 # SOURCE 指向真实出处。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# SOURCE: vllm/config/model.py (class ModelConfig)
@dataclass
class ModelConfig:
    max_model_len: int = 2048
    runner_type: str = "generate"  # 'generate' or 'pooling'
    vocab_size: int = 32000

    # SOURCE: vllm/config/model.py (ModelConfig.get_vocab_size)
    def get_vocab_size(self) -> int:
        return self.vocab_size

    # SOURCE: vllm/config/model.py (ModelConfig.try_get_generation_config)
    def try_get_generation_config(self) -> dict[str, Any]:
        # SUBTRACTED: 真实从 HF generation_config.json 读取 eos_token_id 等；
        #   精简版默认空 dict，测试可显式注入，原 vllm/config/model.py。
        return {}


@dataclass
class LoRAConfig:
    # SOURCE: vllm/config/lora.py (class LoRAConfig)
    enable_tower_connector_lora: bool = False


@dataclass
class ParallelConfig:
    # SOURCE: vllm/config/parallel.py (class ParallelConfig)
    data_parallel_size: int = 1
    data_parallel_size_local: int = 1
    local_engines_only: bool = False


@dataclass
class VllmConfig:
    # SOURCE: vllm/config/__init__.py (class VllmConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    lora_config: LoRAConfig | None = None
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    speculative_config: Any | None = None
    structured_outputs_config: Any | None = None
    reasoning_config: Any | None = None
    # SUBTRACTED: cache_config/scheduler_config/observability_config 等 — 本章未触及。


# SOURCE: vllm/renderers (Renderer / tokenizer)
class _FakeTokenizer:
    """tokenizer 替身：仅本章用到 encode() / max_token_id。"""

    def __init__(self, max_token_id: int = 31999):
        # SOURCE: vllm/tokenizers (TokenizerLike) — 替身
        self.max_token_id = max_token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # SOURCE: vllm/tokenizers (TokenizerLike.encode) — 替身
        # 极简确定性编码：每个字符映射到一个 id，仅供 update_from_tokenizer 演示。
        return [ord(c) % (self.max_token_id + 1) for c in text] or [0]


class Renderer:
    """Renderer 的忠实最小替身。

    真实 Renderer 承载 tokenize/多模态/embeds 渲染重活（render_cmpl/render_chat）。
    本章 InputProcessor 主路径吃**已渲染**的 EngineInput dict，因此这里只暴露
    InputProcessor 直接用到的接口：tokenizer / get_eos_token_id / get_tokenizer。
    """

    # SOURCE: vllm/renderers (BaseRenderer)
    def __init__(self, tokenizer=None, eos_token_id: int | None = None):
        self.tokenizer = tokenizer
        self._eos_token_id = eos_token_id

    def get_eos_token_id(self) -> int | None:
        # SOURCE: vllm/renderers (BaseRenderer.get_eos_token_id)
        return self._eos_token_id

    def get_tokenizer(self):
        # SOURCE: vllm/renderers (BaseRenderer.get_tokenizer)
        if self.tokenizer is None:
            raise ValueError("No tokenizer configured")
        return self.tokenizer


class _CurrentPlatform:
    @staticmethod
    def validate_request(processed_inputs, params) -> None:
        # SOURCE: vllm/platforms/interface.py:L848 (Platform.validate_request)
        """Raises if this request is unsupported on this platform.

        基类实现为 no-op（CUDA 等平台可覆盖）；精简版保留调用点即可。
        """
        return None


current_platform = _CurrentPlatform()
