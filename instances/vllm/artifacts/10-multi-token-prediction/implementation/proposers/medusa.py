"""MedusaProposer — K independent MLP heads, NOT inheriting from base.

# REFERENCE: vllm/v1/spec_decode/medusa.py:L1-L78

This is the most architecturally distinct proposer. Medusa does NOT chain
draft tokens autoregressively. Instead, it stacks K independent MLP heads on
the target's last hidden state and reads each head's argmax in parallel.

Why this matters:
  - Cheap draft: only K small MLPs, no transformer block, no autoregression.
  - Lower acceptance: each head sees only the SAME hidden state (no causal
    feedback from previous draft tokens), so heads further out are guessing
    blind. Acceptance rate decays sharply with position.
  - Greedy-only by construction: the source uses `argmax(dim=-1)` directly,
    and `propose()` returns argmax tokens — no probabilities for downstream
    rejection_sample. Therefore the rejection sampler runs the NO_DRAFT_PROBS
    path (greedy fast-path).

Compare against EAGLE: EAGLE has a transformer block + autoregressive chain
through the draft, so each draft sees the previous draft's hidden state.
Higher acceptance, more cost.
"""
# REFERENCE: vllm/v1/spec_decode/medusa.py:L1-L78
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MedusaHeads(nn.Module):
    """Toy K-head Medusa — each head is one Linear, parallel over K.

    Real Medusa heads are slightly bigger (typically 2-3 layer MLPs), but the
    structure (independence + greedy-only) is the same.
    """

    def __init__(self, K: int, hidden: int, vocab: int):
        super().__init__()
        # REFERENCE: vllm/v1/spec_decode/medusa.py:L48-L49
        # Source has `self.model(target_hidden_states)` returning K blocks,
        # then `self.model.compute_logits(blocks)` returning K logits.
        self.heads = nn.ModuleList([nn.Linear(hidden, vocab) for _ in range(K)])

    def __call__(self, hidden_states: torch.Tensor) -> list[torch.Tensor]:
        return [head(hidden_states) for head in self.heads]

    def compute_logits(self, blocks: list[torch.Tensor]) -> list[torch.Tensor]:
        # In source, the heads themselves are NOT the LM-head; there's a separate
        # compute_logits step. For pedagogical clarity we collapse them.
        return blocks


# REFERENCE: vllm/v1/spec_decode/medusa.py:L18-L78
class MedusaProposer:
    """Medusa proposer.

    # REFERENCE: vllm/v1/spec_decode/medusa.py:L18-L78

    Key fact: Medusa does NOT inherit from SpecDecodeBaseProposer. Its propose()
    is fundamentally different — no autoregressive loop, just parallel argmax
    over K independent heads.
    """

    def __init__(self, K: int, hidden: int, vocab: int):
        # REFERENCE: vllm/v1/spec_decode/medusa.py:L23-L37
        self.K = K
        self.hidden_size = hidden
        self.vocab_size = vocab
        self.model: Optional[MedusaHeads] = None  # set by load_model

    def load_model(self, hidden: int = None, vocab: int = None) -> None:
        # REFERENCE: vllm/v1/spec_decode/medusa.py:L57-L68 (load_model)
        # Source does `get_model(...)` with the speculative_config's draft model.
        # We just instantiate fresh heads.
        self.model = MedusaHeads(self.K, hidden or self.hidden_size,
                                 vocab or self.vocab_size)

    def propose(
        self,
        target_hidden_states: torch.Tensor,  # [batch, hidden]
    ) -> torch.Tensor:
        """Run K heads in parallel; stack argmax per head.

        # REFERENCE: vllm/v1/spec_decode/medusa.py:L39-L55

        Returns:
            [batch, K] int — the K draft tokens per request.

        NOTE: There are NO probabilities. Downstream RejectionSampler must run
        the greedy / NO_DRAFT_PROBS path. This is the same as ngram in that
        respect — but for a different reason: ngram has no model, Medusa has
        a model but uses argmax to keep latency low (each head's output is
        used directly without softmax).
        """
        assert self.model is not None
        # REFERENCE: vllm/v1/spec_decode/medusa.py:L48
        blocks = self.model(target_hidden_states)
        # REFERENCE: vllm/v1/spec_decode/medusa.py:L49
        logits = self.model.compute_logits(blocks)
        # REFERENCE: vllm/v1/spec_decode/medusa.py:L52-L53
        # Source: torch.stack([logit.argmax(dim=-1) for logit in logits], dim=1)
        return torch.stack([logit.argmax(dim=-1) for logit in logits], dim=1)


if __name__ == "__main__":
    torch.manual_seed(42)
    print("=== MedusaProposer toy demo ===")
    K, hidden, vocab, batch = 4, 64, 128, 3
    proposer = MedusaProposer(K=K, hidden=hidden, vocab=vocab)
    proposer.load_model()
    target_hidden = torch.randn(batch, hidden)
    drafts = proposer.propose(target_hidden)
    print(f"  drafts shape = {tuple(drafts.shape)}  (= [batch, K] = [3, 4])")
    print(f"  drafts =\n{drafts.tolist()}")
    print()
    print("Note: NO probabilities returned — downstream RejectionSampler runs greedy path.")
