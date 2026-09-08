# SOURCE: vllm/model_executor/layers/logits_processor.py
# ch23 主角文件之三（m5/m12/m13）：LogitsProcessor——logits 物化三步
# （lm_head GEMM → TP gather/all-gather → 裁词表 padding）。
# SUBTRACTED：delete[4]（logits_as_input/soft_cap/scale≠1 三分支——MedTTS/
#   Gemma/Chameleon 类特性）、delete[5]（_apply_head 的 head_dtype≠ 分支——
#   fp32 head 属 RL 可选配置，m13 叙事保留实现删除）、delete[6]（get_top_tokens
#   全方法——spec decode drafter 专用，ch33 领地）。
from __future__ import annotations

import torch

from vllm.config import get_current_vllm_config
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_gather,
)
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.platforms import current_platform


# --8<-- [start:logits_processor]
# SOURCE: vllm/model_executor/layers/logits_processor.py:L22 @PluggableLayer.
#   register("logits_processor") —— v0.27 新面（m13）：OOT 平台可换的注册点
@PluggableLayer.register("logits_processor")
# SOURCE: vllm/model_executor/layers/logits_processor.py:L23 LogitsProcessor
class LogitsProcessor(PluggableLayer):
    """Process logits and apply logits processors from sampling metadata.

    This layer does the following:
    1. Gather logits from model hidden_states.
    2. Scale logits if needed.
    3. Apply logits processors (if any).
    """

    # --8<-- [end:logits_processor]

    # SOURCE: vllm/model_executor/layers/logits_processor.py:L34-L61 __init__
    #   （逐字——org_vocab_size/pad 裁剪边界、use_all_gather 通道选择、
    #   head_dtype 的 fp32-head 配置位）
    def __init__(
        self,
        vocab_size: int,
        org_vocab_size: int | None = None,
        scale: float = 1.0,
        logits_as_input: bool = False,
        soft_cap: float | None = None,
    ) -> None:
        """
        Args:
            scale: A scaling factor to apply to the logits.
        """
        # SOURCE: vllm/model_executor/layers/logits_processor.py:L34-L61 __init__
        super().__init__()
        self.scale = scale
        self.vocab_size = vocab_size
        # Whether the input is logits (default is hidden states).
        self.logits_as_input = logits_as_input
        # original vocabulary size (without LoRA).
        self.org_vocab_size = org_vocab_size or vocab_size
        # Soft cap the logits. Used in Gemma 2.
        self.soft_cap = soft_cap
        # Whether to use gather or all-gather to gather the logits.
        self.use_all_gather = current_platform.use_all_gather()
        # Dtype of the lm_head projection. Defaults to the model dtype; an
        # fp32 head (via `--hf-overrides '{"head_dtype": "float32"}'`) is
        # required for RL training-inference consistency.
        model_config = get_current_vllm_config().model_config
        self.head_dtype = model_config.head_dtype if model_config is not None else None

    # SOURCE: vllm/model_executor/layers/logits_processor.py:L63-L82 forward
    #   减法子集（delete[4]：logits_as_input/soft_cap/scale≠1 三分支删除；
    #   主路径 _get_logits 直通逐字）
    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        # SUBTRACTED: logits_as_input 分支（L69-L70）——delete[4]
        # Get the logits for the next tokens.
        # SOURCE: vllm/model_executor/layers/logits_processor.py:L63-L82 forward
        logits = self._get_logits(hidden_states, lm_head, embedding_bias)
        # SUBTRACTED: soft_cap 双曲正切压幅与 scale≠1 缩放分支（L74-L81）
        #   ——delete[4]（默认 soft_cap=None、scale=1.0 全不进）
        return logits

    # SOURCE: vllm/model_executor/layers/logits_processor.py:L84-L96 _gather_logits
    #   （逐字——TP 词表拼装：gather 语义 rank>0 返回 None vs all-gather 严格
    #   SPMD，woosuk NOTE 原话）
    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """gather/all-gather the logits tensor across model parallel group."""
        # SOURCE: vllm/model_executor/layers/logits_processor.py:L84-L96 _gather_logits
        if self.use_all_gather:
            # Gather is not supported for some devices such as TPUs.
            # Use all-gather instead.
            # NOTE(woosuk): Here, the outputs of every device should not be None
            # because XLA requires strict SPMD among all devices. Every device
            # should execute the same operations after gathering the logits.
            logits = tensor_model_parallel_all_gather(logits)
        else:
            # None may be returned for rank > 0
            logits = tensor_model_parallel_gather(logits)
        return logits

    # SOURCE: vllm/model_executor/layers/logits_processor.py:L98-L135 _apply_head
    #   —— 减法子集（delete[5]：head_dtype≠hidden_states.dtype 的 fp32 直累
    #   与 cast 回退分支（L110-L135）删除；默认 head_dtype=None 主路径逐字）
    def _apply_head(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Project hidden states through the lm_head, honoring head_dtype."""
        # SOURCE: vllm/model_executor/layers/logits_processor.py:L98-L135 _apply_head
        return lm_head.quant_method.apply(
            lm_head, hidden_states, bias=embedding_bias
        )

    # SOURCE: vllm/model_executor/layers/logits_processor.py:L137-L153 _get_logits
    #   （逐字——本章出口核心三步）
    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        # Get the logits for the next tokens.
        # SOURCE: vllm/model_executor/layers/logits_processor.py:L137-L153 _get_logits
        logits = self._apply_head(lm_head, hidden_states, embedding_bias)

        # Gather logits for TP
        if lm_head.tp_size > 1:
            logits = self._gather_logits(logits)

        # Remove paddings in vocab (if any).
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        return logits

    # SUBTRACTED: get_top_tokens 全方法（logits_processor.py:L155-L205）——
    #   delete[6]：只有 spec decode drafter 的 use_local_argmax_reduction 调用
    #   （vllm/v1/spec_decode/llm_base_proposer.py:L428-L431），普通采样路径
    #   零调用；词表并行局部 argmax 归约（通信 O(batch*2*tp) vs O(batch*vocab)）
    #   领地归 ch33——m13 叙事保留、实现删除。

    # SOURCE: vllm/model_executor/layers/logits_processor.py:L207-L211 extra_repr
    #   （逐字）
    def extra_repr(self) -> str:
        # SOURCE: vllm/model_executor/layers/logits_processor.py:L207-L211 extra_repr
        s = f"vocab_size={self.vocab_size}"
        s += f", org_vocab_size={self.org_vocab_size}"
        s += f", scale={self.scale}, logits_as_input={self.logits_as_input}"
        return s
