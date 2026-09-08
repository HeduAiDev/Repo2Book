# SOURCE: vllm/model_executor/models/interfaces.py
# ch23 消费面：llama.py bases 上的接口标记与 EAGLE mixin（delete[0] 明示
# 「标记类保留声明」——hasattr/isinstance 探测位）。SupportsQuant 的
# _find_quant_config 委托链保留（quant_config 从构造参数里的 VllmConfig 摘出）。
# SUBTRACTED：接口探测函数族（supports_pp/supports_multimodal 等）与其余
# 数十个 Supports* 协议——本章只保 llama.py bases 消费面。
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, runtime_checkable
from typing_extensions import Self, TypeIs

import torch

from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationConfig
    from vllm.model_executor.models.utils import WeightsMapper


# SOURCE: vllm/model_executor/models/interfaces.py:L564 SupportsLoRA
#   packed_modules_mapping 的协议位：LoRA 的融合对应关系
class SupportsLoRA(Protocol):
    """The interface required for all models that support LoRA."""

    # SOURCE: vllm/model_executor/models/interfaces.py:L564 SupportsLoRA
    supports_lora: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports LoRA.

    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """
    is_3d_moe_weight: ClassVar[bool] = False
    is_non_gated_moe: ClassVar[bool] = False
    # The `embedding_module` and `embedding_padding_modules`
    # are empty by default.
    embedding_modules: ClassVar[dict[str, str]] = {}
    packed_modules_mapping: dict[str, list[str]] = {}
    # Module prefixes to skip during LoRA loading (e.g., ["mtp."] for MTP layers)
    lora_skip_prefixes: ClassVar[list[str]] = []
    lora_manager: "LoRAModelManager | None"


# SUBTRACTED: _SupportsLoRAType/supports_lora 探测函数族
#   （interfaces.py:L581-L~640）——探测面归 registry/LoRA 域


# SOURCE: vllm/model_executor/models/interfaces.py:L643 SupportsPP
#   make_empty_intermediate_tensors 钩子的协议位
class SupportsPP(Protocol):
    """The interface required for all models that support pipeline parallel."""

    # SOURCE: vllm/model_executor/models/interfaces.py:L643 SupportsPP
    supports_pp: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports pipeline parallel.

    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "IntermediateTensors":
        """Called when PP rank > 0 for profiling purposes."""
        # SOURCE: vllm/model_executor/models/interfaces.py:L655-L661
        ...

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        *,
        intermediate_tensors: "IntermediateTensors | None",
    ) -> "IntermediateTensors | None":
        """
        Accept [`IntermediateTensors`][vllm.sequence.IntermediateTensors] when
        PP rank > 0.

        Return [`IntermediateTensors`][vllm.sequence.IntermediateTensors] only
        for the last PP rank.
        """
        # SOURCE: vllm/model_executor/models/interfaces.py:L664-L684
        ...


# SOURCE: vllm/model_executor/models/interfaces.py:L1050 SupportsQuant
#   子集（__new__ 的 quant_config 摘取主干逐字；_maybe_apply_model_mapping 的
#   模型映射面删除——ch27 量化分发域）
class SupportsQuant:
    """The interface required for all models that support quantization."""

    hf_to_vllm_mapper: ClassVar["WeightsMapper | None"] = None
    packed_modules_mapping: ClassVar[dict[str, list[str]] | None] = None
    quant_config: "QuantizationConfig | None" = None

    def __new__(cls, *args, **kwargs) -> Self:
        # SOURCE: vllm/model_executor/models/interfaces.py:L1057-L1066 SupportsQuant.__new__
        instance = super().__new__(cls)

        # find config passed in arguments and attach it to model for general use
        instance.quant_config = cls._find_quant_config(*args, **kwargs)

        # SUBTRACTED: _maybe_apply_model_mapping（interfaces.py:L1071）——
        #   量化分发 override 归 ch27

        return instance

    # SOURCE: vllm/model_executor/models/interfaces.py:L1068-L1079
    #   _find_quant_config（逐字——从构造参数里的 VllmConfig 摘 quant_config）
    @staticmethod
    def _find_quant_config(*args, **kwargs) -> "QuantizationConfig | None":
        """Find quant config passed through model constructor args"""
        # SOURCE: vllm/model_executor/models/interfaces.py:L1068-L1079
        from vllm.config import VllmConfig  # avoid circular import

        args_values = list(args) + list(kwargs.values())
        for arg in args_values:
            if isinstance(arg, VllmConfig):
                return arg.quant_config
        return None


# SUBTRACTED: SupportsQuant 的 model mapping 尾段与其余 Supports* 协议族
#   （interfaces.py:L1086-L1363）——各特性域


# SOURCE: vllm/model_executor/models/interfaces.py:L1364 LocalArgmaxMixin
#   （逐字——draft 头的 D2T 感知 argmax；ch33 消费）
class LocalArgmaxMixin:
    """Mixin for draft model heads in speculative decoding.

    Provides a D2T-aware ``get_top_tokens`` that preserves the
    local-argmax communication reduction even when the draft vocabulary
    is smaller than the target vocabulary.

    When ``draft_id_to_target_id`` is present (shape ``(draft_vocab_size,)``,
    containing per-token offset to target vocab id), the draft argmax index
    ``k`` is mapped to the target vocab id via::

        target_id = k + draft_id_to_target_id[k]

    This is mathematically equivalent to computing the full-vocab scatter
    logits and taking the global argmax, but requires only
    O(batch * 2 * tp_size) communication instead of O(batch * vocab_size).

    Requires the subclass to expose:
        ``self.logits_processor``: LogitsProcessor
        ``self.lm_head``: ParallelLMHead
        ``self.draft_id_to_target_id`` (optional): nn.Parameter
    """

    # SOURCE: vllm/model_executor/models/interfaces.py:L1387-L1397
    #   get_top_tokens（逐字——委托 LogitsProcessor.get_top_tokens（delete[6]
    #   随 ch33 回填）+ D2T 重映射）
    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Vocab-parallel argmax with optional D2T remapping."""
        # SOURCE: vllm/model_executor/models/interfaces.py:L1387-L1397
        top = self.logits_processor.get_top_tokens(
            self.lm_head,
            hidden_states,
        )
        d2t = getattr(self, "draft_id_to_target_id", None)
        if d2t is not None:
            top = top + d2t[top]
        return top


# SOURCE: vllm/model_executor/models/interfaces.py:L1399 EagleModelMixin
#   （逐字——EAGLE3 aux hidden states 钩子；llama.py L419/L426-L428 的调用点
#   随 delete[0] 删除，mixin 本体保留声明）
class EagleModelMixin:
    # SOURCE: vllm/model_executor/models/interfaces.py:L1399 EagleModelMixin
    aux_hidden_state_layers: tuple[int, ...] = ()

    def _set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        # SOURCE: vllm/model_executor/models/interfaces.py:L1402-L1403
        self.aux_hidden_state_layers = layers

    def _maybe_add_hidden_state(
        self,
        aux_hidden_states: list[torch.Tensor],
        layer_idx: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        # SOURCE: vllm/model_executor/models/interfaces.py:L1405-L1416
        if layer_idx in self.aux_hidden_state_layers:
            value = hidden_states + residual if residual is not None else hidden_states
            aux_hidden_states.append(value)
        return aux_hidden_states


# SOURCE: vllm/model_executor/models/interfaces.py:L1419 SupportsEagle（逐字）
@runtime_checkable
class SupportsEagle(Protocol):
    """The interface required for models that support
    EAGLE-1 and EAGLE-2 speculative decoding."""

    # SOURCE: vllm/model_executor/models/interfaces.py:L1419 SupportsEagle（逐字）
    supports_eagle: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports EAGLE-1 and EAGLE-2
    speculative decoding.

    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """


# SOURCE: vllm/model_executor/models/interfaces.py:L1449 SupportsEagle3
#   —— 减法子集（标记位逐字；set_aux_hidden_state_layers 方法删除——
#   runner 侧消费归 ch32/33）
@runtime_checkable
class SupportsEagle3(Protocol):
    """The interface required for models that support
    EAGLE-3 speculative decoding."""

    # SOURCE: vllm/model_executor/models/interfaces.py:L1449 SupportsEagle3
    supports_eagle3: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports EAGLE-3 speculative decoding.

    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """
