"""ExtractHiddenStatesProposer — single-step MTP / KV-extraction path.

# REFERENCE: vllm/v1/spec_decode/extract_hidden_states.py:L1-L382

This is the third "non-base" proposer. Its job is unusual: it doesn't really
speculate. It uses target hidden states as both input and "draft" (returning
sampled_token_ids unchanged so they always verify), with the actual purpose
being to cache hidden states in the KV cache for downstream uses.

Critical assertion at L30:

    assert vllm_config.speculative_config.num_speculative_tokens == 1

This proposer is HARD-CODED to single-step MTP. Multi-step MTP must use the
DraftModelProposer + DeepSeekMTP model class instead. The single-step path
is for use cases where you just want to extract hidden states for KV
transfer (e.g., decoupled prefill/decode in PD architecture, Ch22+) without
actually drafting.

Why this matters for understanding MTP:
  - It illustrates that MTP-the-technique is implemented through TWO different
    paths in vLLM:
      (a) Multi-step: DraftModelProposer + DeepSeekMTP (the canonical MTP)
      (b) Single-step: ExtractHiddenStatesProposer (asserts K==1, just caches)
  - Same algorithm (rejection_sample), different proposer plumbing.
"""
# REFERENCE: vllm/v1/spec_decode/extract_hidden_states.py:L1-L25
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class ExtractHiddenStatesProposer:
    """Pedagogical mirror — single-step assertion is the load-bearing fact.

    # REFERENCE: vllm/v1/spec_decode/extract_hidden_states.py:L26-L70
    """

    def __init__(
        self,
        num_speculative_tokens: int,
        hidden_size: int,
        num_hidden_states: int = 1,
        max_num_tokens: int = 4096,
    ):
        # REFERENCE: vllm/v1/spec_decode/extract_hidden_states.py:L29-L31
        # The HARD assertion: this proposer requires K == 1.
        # If a config sets num_speculative_tokens > 1 with this proposer,
        # vLLM raises at construction.
        assert num_speculative_tokens == 1, (
            "ExtractHiddenStatesProposer requires num_speculative_tokens == 1; "
            "got {}".format(num_speculative_tokens)
        )

        self.hidden_size = hidden_size
        self.num_hidden_states = num_hidden_states
        # REFERENCE: vllm/v1/spec_decode/extract_hidden_states.py:L60-L65
        self.hidden_states_buffer = torch.zeros(
            (max_num_tokens, num_hidden_states, hidden_size),
            dtype=torch.float32,
        )
        self.model: Optional[nn.Module] = None

    # REFERENCE: vllm/v1/spec_decode/extract_hidden_states.py:L72-L130
    def propose(
        self,
        sampled_token_ids: torch.Tensor,           # [batch_size]
        target_hidden_states: List[torch.Tensor],  # one per aux layer
    ) -> torch.Tensor:
        """Return sampled_token_ids unchanged as "drafts" (always verify).

        The point isn't drafting — it's caching `target_hidden_states` in the
        engine's KV-related buffers for downstream layers.

        Returns: [batch_size, 1] — exactly one "draft" per request, equal
        to sampled_token_ids (so verification trivially accepts).
        """
        # REFERENCE: vllm/v1/spec_decode/extract_hidden_states.py:L107-L109
        # Stack hidden states from each aux layer along dim=1.
        stacked = torch.stack(target_hidden_states, dim=1)
        num_tokens = stacked.shape[0]
        # REFERENCE: vllm/v1/spec_decode/extract_hidden_states.py:L113
        self.hidden_states_buffer[:num_tokens] = stacked

        # REFERENCE: vllm/v1/spec_decode/extract_hidden_states.py:L86-L89
        # "This proposer doesn't actually perform speculation - it returns
        # the sampled tokens as 'draft' tokens, ensuring they always verify"
        return sampled_token_ids.view(-1, 1)


if __name__ == "__main__":
    torch.manual_seed(42)
    print("=== ExtractHiddenStatesProposer toy demo ===")
    proposer = ExtractHiddenStatesProposer(
        num_speculative_tokens=1, hidden_size=64, num_hidden_states=2
    )
    sampled = torch.tensor([7, 13, 21], dtype=torch.int32)  # batch=3
    target_hidden = [torch.randn(3, 64), torch.randn(3, 64)]  # 2 aux layers
    drafts = proposer.propose(sampled, target_hidden)
    print(f"  sampled_token_ids = {sampled.tolist()}")
    print(f"  drafts            = {drafts.tolist()}  (always equal to sampled — trivial verify)")
    print(f"  drafts shape      = {tuple(drafts.shape)}  (= [batch, 1] always)")
    print()
    try:
        ExtractHiddenStatesProposer(num_speculative_tokens=4, hidden_size=64)
    except AssertionError as e:
        print(f"K=4 → AssertionError as expected: {e}")
