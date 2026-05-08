"""DraftModelProposer — a separate small transformer drafts for the target.

# REFERENCE: vllm/v1/spec_decode/draft_model.py:L1-L88

The "classic" speculative-decoding setup: draft with a small model (e.g.,
Llama-3.3-1B) for a large target (e.g., Llama-3.3-70B). The draft is a
completely independent model — different parameters, different layers,
different training.

Required by source:
  - Same vocab size (so target logits can verify draft tokens).
  - Same TP size (avoids torch-compile cache corruption — see source comment
    at L37-L51 for the Tomas Ruiz issue).
  - Does NOT share embeddings or lm_head with target (each model has its own).

Why this is the most flexible:
  - No architectural coupling to the target — any HF causal LM works as a draft.
  - Acceptance rate is moderate (~0.5-0.7) because the small model's distribution
    is similar but not identical to the target.
  - Draft cost is the highest of all proposers — running an entire (smaller)
    transformer per drafting step. The c ratio is typically 0.05-0.10.

Why MTP is preferred for DeepSeek-V3:
  - DeepSeek-V3 trained the MTP heads JOINTLY with the main model, so the
    draft is conditioned on the same hidden states as the target. Acceptance
    rate goes up to ~0.85 — much higher than independent-model drafting.
  - The MTP heads are smaller than a full small-model draft.
"""
# REFERENCE: vllm/v1/spec_decode/draft_model.py:L1-L14
from __future__ import annotations

from .base import SpecDecodeBaseProposer


class DraftModelProposer(SpecDecodeBaseProposer):
    """Independent-draft-model proposer.

    # REFERENCE: vllm/v1/spec_decode/draft_model.py:L17-L88
    """

    def __init__(
        self,
        num_speculative_tokens: int,
        hidden_size: int,
        target_vocab_size: int,
        draft_vocab_size: int,
        target_tp: int = 1,
        draft_tp: int = 1,
    ):
        super().__init__(
            num_speculative_tokens=num_speculative_tokens,
            hidden_size=hidden_size,
            # REFERENCE: vllm/v1/spec_decode/draft_model.py:L26-L29
            # pass_hidden_states_to_model=False — the draft model is independent.
            pass_hidden_states_to_model=False,
        )
        # REFERENCE: vllm/v1/spec_decode/draft_model.py:L33-L34
        self._raise_if_vocab_size_mismatch(target_vocab_size, draft_vocab_size)
        # REFERENCE: vllm/v1/spec_decode/draft_model.py:L36-L51
        self._raise_if_draft_tp_mismatch(target_tp, draft_tp)

    @staticmethod
    def _raise_if_vocab_size_mismatch(target: int, draft: int) -> None:
        # REFERENCE: vllm/v1/spec_decode/draft_model.py:L33-L34
        # Source delegates to `verify_equal_vocab_size_if_draft_model` on
        # the speculative_config. Vocab MUST be equal for verification to work.
        if target != draft:
            raise ValueError(
                f"Draft vocab_size={draft} must equal target vocab_size={target}; "
                "rejection sampling indexes target_logits by draft token id."
            )

    @staticmethod
    def _raise_if_draft_tp_mismatch(target_tp: int, draft_tp: int) -> None:
        # REFERENCE: vllm/v1/spec_decode/draft_model.py:L36-L51
        # Tomas Ruiz: when target TP > 1 and draft TP = 1, the torch.compile
        # cache gets corrupted across ranks because all ranks compile the
        # draft on rank 0. Issue tracked at vllm-project/vllm#5414.
        if draft_tp != target_tp:
            raise ValueError(
                f"draft_tensor_parallel_size={draft_tp} must equal "
                f"tensor_parallel_size={target_tp}. "
                f"See vllm/v1/spec_decode/draft_model.py:L37-L51 for context."
            )

    # REFERENCE: vllm/v1/spec_decode/draft_model.py:L80-L88
    # _maybe_share_embeddings and _maybe_share_lm_head both pass — draft model
    # has its own.
    def _maybe_share_embeddings(self, target_language_model) -> None:
        pass  # draft has its own embed_tokens

    def _maybe_share_lm_head(self, target_language_model) -> None:
        pass  # draft has its own lm_head
