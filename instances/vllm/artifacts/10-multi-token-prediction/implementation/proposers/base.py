"""SpecDecodeBaseProposer — the scaffolding shared by EAGLE / DraftModel / etc.

# REFERENCE: vllm/v1/spec_decode/llm_base_proposer.py:L60-L1820

The original is 1820 lines because it covers CUDA graph capture, slot mapping,
parallel drafting, attention metadata builders, draft model construction,
multi-modal embeddings, weight sharing, padded batches, and tree attention.

We retain only the algorithmic core that matters for understanding spec-decode:

  1. `propose()`: take target hidden states; run draft model K times (or once
     for parallel-drafting); return [batch, K] draft token ids.

  2. `_greedy_sample()`: argmax over draft logits — the simplest case.

  3. The control-flow split: K==1 fast-path returns immediately; K>1 runs
     a sequential autoregressive loop on the draft model.

We DO NOT implement: CUDA graph capture, slot mapping, attention metadata,
KV cache, parallel drafting state machine, weight sharing, multi-modal.
Each of those is its own chapter (Ch11+).
"""
# REFERENCE: vllm/v1/spec_decode/llm_base_proposer.py:L1-L57 (imports)
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import torch
import torch.nn as nn


@dataclass
class ProposerOutput:
    """The (drafts, draft_probs) output of a Proposer.propose() call.

    # REFERENCE: vllm/v1/spec_decode/llm_base_proposer.py:L407-L411 (return shape)
    Pedagogical container: NgramProposer / MedusaProposer return draft_probs=None
    (greedy-only path); MTPProposer / DraftModelProposer return softmax probs.
    """

    draft_token_ids: torch.Tensor
    draft_probs: Optional[torch.Tensor] = None


class _DraftModel(Protocol):
    """Minimal interface the base proposer expects from `self.model`."""

    def __call__(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor: ...

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...


class SpecDecodeBaseProposer:
    """Pedagogical mirror of vLLM's SpecDecodeBaseProposer (algorithmic core only).

    # REFERENCE: vllm/v1/spec_decode/llm_base_proposer.py:L60-L303 (__init__)

    The init does many things; we keep the load-bearing fields:
      - num_speculative_tokens: GLOBAL K for the engine. NOT per-request.
      - pass_hidden_states_to_model: True for EAGLE/MTP (target hidden states are
        an INPUT to the draft); False for DraftModelProposer (independent model).
      - parallel_drafting: when True, draft proposes ALL K in one forward.
    """

    def __init__(
        self,
        num_speculative_tokens: int,
        hidden_size: int,
        pass_hidden_states_to_model: bool = False,
        parallel_drafting: bool = False,
    ):
        # REFERENCE: vllm/v1/spec_decode/llm_base_proposer.py:L79
        self.num_speculative_tokens = num_speculative_tokens
        # REFERENCE: vllm/v1/spec_decode/llm_base_proposer.py:L84-L94
        # (The hidden_size can be different between target and draft;
        # DeepSeek-V4 multiplies by hc_mult — we omit that complication.)
        self.hidden_size = hidden_size
        # REFERENCE: vllm/v1/spec_decode/llm_base_proposer.py:L73
        # When True, `target_hidden_states` is fed into the draft model alongside
        # the input_ids. Used by EAGLE and MTP.
        self.pass_hidden_states_to_model = pass_hidden_states_to_model
        # REFERENCE: vllm/v1/spec_decode/llm_base_proposer.py:L99-L102
        # parallel_drafting: DFlash + parallel-EAGLE — all K in one forward.
        # In our pedagogical impl, we just gate the loop.
        self.parallel_drafting = parallel_drafting

        self.model: Optional[_DraftModel] = None  # set by subclass

    # REFERENCE: vllm/v1/spec_decode/llm_base_proposer.py:L407-L411 (_greedy_sample)
    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute draft logits and take argmax (the simplest case)."""
        assert self.model is not None
        return self.model.compute_logits(hidden_states).argmax(dim=-1)

    # REFERENCE: vllm/v1/spec_decode/llm_base_proposer.py:L413-L655 (propose)
    def propose(
        self,
        target_token_ids: torch.Tensor,           # [num_tokens]
        target_positions: torch.Tensor,           # [num_tokens]
        target_hidden_states: torch.Tensor,       # [num_tokens, hidden_size]
        next_token_ids: torch.Tensor,             # [batch_size]
    ) -> torch.Tensor:
        """Propose K draft tokens per request.

        # REFERENCE: vllm/v1/spec_decode/llm_base_proposer.py:L432-L494

        Two control-flow shapes the source highlights:

          Fast path: K == 1 OR parallel_drafting → run draft once, return.
            REFERENCE: vllm/v1/sample/llm_base_proposer.py:L491-L494

          Sequential path: K > 1 → run draft K times, feeding previous draft
          back as input.
            REFERENCE: vllm/v1/sample/llm_base_proposer.py:L529-L654

        We implement both shapes in plain PyTorch.
        Returns: [batch_size, num_speculative_tokens]
        """
        assert self.model is not None
        batch_size = next_token_ids.shape[0]

        # First forward: same as a regular target-step but on the draft model.
        # Source rotates input_ids by one and inserts next_token_ids at the
        # last slot (set_inputs_first_pass, L657-L725); we simplify to a fresh
        # forward using next_token_ids directly.
        # SIMPLIFIED: Source uses preallocated cudagraph buffers; we allocate.
        kwargs = {
            "input_ids": next_token_ids,
            "positions": target_positions[-batch_size:],  # last position per req
        }
        if self.pass_hidden_states_to_model:
            kwargs["hidden_states"] = target_hidden_states[-batch_size:]
        ret = self.model(**kwargs)  # [batch, hidden]
        last_hidden_states = ret if not isinstance(ret, tuple) else ret[0]
        hidden_states = ret if not isinstance(ret, tuple) else ret[1]

        # Fast path: K == 1 or parallel_drafting → return immediately.
        # REFERENCE: vllm/v1/spec_decode/llm_base_proposer.py:L491-L494
        if self.num_speculative_tokens == 1 or self.parallel_drafting:
            draft_token_ids = self._greedy_sample(last_hidden_states)
            return draft_token_ids.view(-1, self.num_speculative_tokens)

        # Sequential path: K > 1. Run draft model K-1 more times.
        # REFERENCE: vllm/v1/spec_decode/llm_base_proposer.py:L516-L654
        draft_token_ids = self._greedy_sample(last_hidden_states)
        draft_list = [draft_token_ids]
        for step in range(self.num_speculative_tokens - 1):
            input_ids = draft_list[-1].int()
            kwargs = {"input_ids": input_ids,
                      "positions": target_positions[-batch_size:] + step + 1}
            if self.pass_hidden_states_to_model:
                kwargs["hidden_states"] = hidden_states  # carry hidden state forward
            ret = self.model(**kwargs)
            last_hidden_states = ret if not isinstance(ret, tuple) else ret[0]
            hidden_states = ret if not isinstance(ret, tuple) else ret[1]
            draft_list.append(self._greedy_sample(last_hidden_states))

        # [batch_size, num_speculative_tokens]
        # REFERENCE: vllm/v1/spec_decode/llm_base_proposer.py:L653-L655
        return torch.stack(draft_list, dim=1)


# -----------------------------------------------------------------------------
# Toy draft "model" so the base proposer is runnable in a demo.
# -----------------------------------------------------------------------------
class ToyDraftModel(nn.Module):
    """A trivial draft model: embed → linear → linear. Just for demos.

    Not part of vLLM's surface — used here so propose() actually executes.
    """

    def __init__(self, vocab: int, hidden: int):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.transform = nn.Linear(hidden, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def __call__(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embed(input_ids)
        if hidden_states is not None:
            x = x + hidden_states
        return self.transform(x)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)


if __name__ == "__main__":
    torch.manual_seed(42)
    print("=== SpecDecodeBaseProposer toy demo ===")
    proposer = SpecDecodeBaseProposer(
        num_speculative_tokens=4,
        hidden_size=64,
        pass_hidden_states_to_model=False,
    )
    proposer.model = ToyDraftModel(vocab=128, hidden=64)
    batch_size, num_tokens = 3, 12
    target_token_ids = torch.randint(0, 128, (num_tokens,))
    target_positions = torch.arange(num_tokens)
    target_hidden_states = torch.randn(num_tokens, 64)
    next_token_ids = torch.randint(0, 128, (batch_size,))
    drafts = proposer.propose(
        target_token_ids, target_positions, target_hidden_states, next_token_ids
    )
    print(f"  drafts shape = {tuple(drafts.shape)}  (= [batch, K] = [3, 4])")
    print(f"  drafts =\n{drafts.tolist()}")
