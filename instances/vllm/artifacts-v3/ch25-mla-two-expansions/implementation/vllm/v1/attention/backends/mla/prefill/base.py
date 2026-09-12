# SOURCE: vllm/v1/attention/backends/mla/prefill/base.py
# ch25 切面（m10）：MLA prefill 家族抽象基类全文——MLADimensions +
# MLAPrefillBackend（validate_configuration/clone/prepare_metadata/
# run_prefill_new_tokens/run_prefill_context_chunk 双抽象）逐字；
# supports_quant_output 面按 delete[2] 删（融合输出量化归 ch27）。
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.mla.prefill.selector import (
        MLAPrefillSelectorConfig,
    )


# SOURCE: vllm/v1/attention/backends/mla/prefill/base.py:L23-L34 MLADimensions
#   —— 逐字
@dataclass(frozen=True, kw_only=True)
class MLADimensions:
    # SOURCE: vllm/v1/attention/backends/mla/prefill/base.py:L23-L34（锚点双置：声明上方注释同文）
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int

    def __str__(self) -> str:
        # SOURCE: vllm/v1/attention/backends/mla/prefill/base.py:L23-L34（锚点双置：声明上方注释同文）
        return (
            f"(qk_nope_head_dim={self.qk_nope_head_dim}, "
            f"qk_rope_head_dim={self.qk_rope_head_dim}, "
            f"v_head_dim={self.v_head_dim})"
        )


# SOURCE: vllm/v1/attention/backends/mla/prefill/base.py:L37-L183
#   MLAPrefillBackend —— 减法子集（supports_quant_output 按 delete[2] 删）
class MLAPrefillBackend(ABC):
    """Abstract base class for MLA prefill backends."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_mla_dimensions: ClassVar[list[MLADimensions]] = []

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        # SOURCE: vllm/v1/attention/backends/mla/prefill/base.py:L37-L183（锚点双置：声明上方注释同文）
        raise NotImplementedError

    @classmethod
    def supports_compute_capability(cls, device_capability: "DeviceCapability") -> bool:
        # SOURCE: vllm/v1/attention/backends/mla/prefill/base.py:L52-L53
        return True

    @classmethod
    def supports_dtype(cls, dtype: torch.dtype) -> bool:
        # SOURCE: vllm/v1/attention/backends/mla/prefill/base.py:L56-L57
        return dtype in cls.supported_dtypes

    @classmethod
    def supports_mla_dimensions(cls, mla_dimensions: MLADimensions) -> bool:
        # SOURCE: vllm/v1/attention/backends/mla/prefill/base.py:L60-L64
        return (
            not cls.supported_mla_dimensions
            or mla_dimensions in cls.supported_mla_dimensions
        )

    @classmethod
    def is_available(cls) -> bool:
        # SOURCE: vllm/v1/attention/backends/mla/prefill/base.py:L67-L68
        return True

    # SUBTRACTED: supports_quant_output（L70-L74）——delete[2]（融合输出
    #   量化门归 ch27）

    def supports_out(self) -> bool:
        """Whether `run_prefill_new_tokens` honors a caller-provided `out`
        tensor of shape `[num_tokens, num_heads, v_head_dim]`, writing the
        final result into it in place.

        When True, callers may pass `out` and skip the post-hoc
        slice/flatten/copy. False for backends that ignore `out` or emit a
        padded (`qk_head_dim`) output. Overridden by backends that support it.
        """
        # SOURCE: vllm/v1/attention/backends/mla/prefill/base.py:L76-L85
        return False

    @classmethod
    def validate_configuration(
        cls,
        device_capability: "DeviceCapability",
        selector_config: "MLAPrefillSelectorConfig",
    ) -> list[str]:
        # SOURCE: vllm/v1/attention/backends/mla/prefill/base.py:L87-L122 —— 逐字
        invalid_reasons: list[str] = []

        if not cls.supports_compute_capability(device_capability):
            invalid_reasons.append(
                f"compute capability {device_capability.major}."
                f"{device_capability.minor} not supported"
            )

        if not cls.supports_dtype(selector_config.dtype):
            invalid_reasons.append(f"dtype {selector_config.dtype} not supported")

        if not cls.is_available():
            invalid_reasons.append("required dependencies not available")

        mla_dimensions = selector_config.mla_dimensions
        if not cls.supports_mla_dimensions(mla_dimensions):
            reason = (
                f"Model does not have supported MLA dimensions (got {mla_dimensions}"
            )
            if (
                cls.supported_mla_dimensions
                and mla_dimensions not in cls.supported_mla_dimensions
            ):
                supported = ", ".join(
                    str(dims) for dims in cls.supported_mla_dimensions
                )
                reason += f"; supported: {supported}"
            invalid_reasons.append(reason + ")")

        return invalid_reasons

    def __init__(
        self,
        num_heads: int,
        scale: float,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        vllm_config: "VllmConfig",
    ) -> None:
        # SOURCE: vllm/v1/attention/backends/mla/prefill/base.py:L124-L140 —— 逐字
        self.num_heads = num_heads
        self.scale = scale
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.vllm_config = vllm_config

    def clone(self) -> "MLAPrefillBackend":
        # SOURCE: vllm/v1/attention/backends/mla/prefill/base.py:L142-L151 —— 逐字
        return self.__class__(
            num_heads=self.num_heads,
            scale=self.scale,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            vllm_config=self.vllm_config,
        )

    def prepare_metadata(  # noqa: B027
        self,
        prefill_metadata: "MLACommonPrefillMetadata",
    ) -> None:
        """Prepare backend-specific metadata before the forward pass.

        Called by the metadata builder after constructing the prefill metadata.
        """
        # SOURCE: vllm/v1/attention/backends/mla/prefill/base.py:L153-L161 —— 逐字
        self._prefill_metadata = prefill_metadata

    @abstractmethod
    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # SOURCE: vllm/v1/attention/backends/mla/prefill/base.py:L163-L173 —— 逐字
        raise NotImplementedError

    @abstractmethod
    def run_prefill_context_chunk(
        self,
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # SOURCE: vllm/v1/attention/backends/mla/prefill/base.py:L175-L183 —— 逐字
        raise NotImplementedError
