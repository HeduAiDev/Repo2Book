"""DeepSeek MTP proposer — wraps the MTP head stack as a proposer.

Note: vLLM does NOT register "mtp" as a `SpeculativeMethod` literal
(`speculative.py:L59-L70`). DeepSeek MTP is loaded via
`method="draft_model"` with `model="deepseek_mtp"`, which routes to the
DeepSeekMTP model class (`deepseek_mtp.py:L186 class DeepSeekMTP`). The
proposer machinery is the EAGLE / draft-model base class shared by all.

We mirror that flow: `DeepSeekMTPProposer` is a thin wrapper around the
`DeepSeekMultiTokenPredictor` from ``mtp_head.py``.
"""

# REFERENCE: vllm/model_executor/models/deepseek_mtp.py:L124-L488
# REFERENCE: vllm/v1/spec_decode/llm_base_proposer.py + draft_model.py

from __future__ import annotations

import torch

from ..mtp_head import DeepSeekMultiTokenPredictor
from .base import ProposerOutput, SpecDecodeBaseProposer


class DeepSeekMTPProposer(SpecDecodeBaseProposer):
    """Run K MTP heads sequentially to produce K draft tokens."""

    def __init__(
        self,
        num_speculative_tokens: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        vocab_size: int,
        num_mtp_layers: int = 1,
    ) -> None:
        # REFERENCE: vllm/v1/spec_decode/llm_base_proposer.py:L60-L106
        # MTP heads see the target's hidden state — that's the defining trait.
        super().__init__(
            num_speculative_tokens=num_speculative_tokens,
            hidden_size=hidden_size,
            pass_hidden_states_to_model=True,
        )
        self.vocab_size = vocab_size
        self.predictor = DeepSeekMultiTokenPredictor(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            vocab_size=vocab_size,
            num_mtp_layers=num_mtp_layers,
        )

    def propose(
        self,
        target_hidden_states: torch.Tensor,  # [batch, hidden]
        last_token_ids: torch.Tensor,        # [batch] — target's most recent sample
        positions: torch.Tensor | None = None,
    ) -> ProposerOutput:
        """K-step MTP: feed (target_hidden, last_token) → MTP layer → next draft → repeat."""
        if positions is None:
            positions = torch.arange(
                target_hidden_states.shape[0], device=target_hidden_states.device
            )
        # REFERENCE: vllm/model_executor/models/deepseek_mtp.py:L160-L170
        drafts = self.predictor.propose_K(
            target_last_hidden=target_hidden_states,
            target_next_token_ids=last_token_ids,
            positions=positions,
            K=self.num_speculative_tokens,
        ).to(torch.int32)
        return ProposerOutput(draft_token_ids=drafts, draft_probs=None)
