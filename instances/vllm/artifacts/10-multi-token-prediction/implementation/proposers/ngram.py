"""NgramProposer — n-gram lookup over the prefix; no model, no probabilities.

# REFERENCE: vllm/v1/spec_decode/ngram_proposer.py:L1-L285

The simplest possible proposer. For each request, find the longest suffix of
the current context that has appeared earlier; emit the K tokens immediately
following that earlier match.

Why interesting:
  - Zero parameters, zero forward passes. Just string matching.
  - Greedy-only by definition (no probabilities — what would they even be?).
    The downstream RejectionSampler runs the NO_DRAFT_PROBS path — accept iff
    p_target(d) >= u, which for confident greedy targets is "accept iff
    target == d" (high p_target).
  - Surprisingly effective for code completion and repetitive prose: the
    context often contains the next K tokens verbatim.
  - Acceptance rate ~0.3-0.5 typical, but draft cost is ~0, so any positive
    acceptance is a net win.

Source uses a numba-compiled KMP failure-function variant (LPS array) for
O(N) suffix matching. We use the naive O(N*K) Python version — same algorithm,
~100x slower.
"""
# REFERENCE: vllm/v1/spec_decode/ngram_proposer.py:L1-L11
from __future__ import annotations

from typing import List, Optional

import numpy as np


class NgramProposer:
    """Pedagogical mirror of vLLM's NgramProposer (algorithmic core only).

    # REFERENCE: vllm/v1/spec_decode/ngram_proposer.py:L12-L62
    """

    def __init__(
        self,
        num_speculative_tokens: int,
        prompt_lookup_min: int = 1,
        prompt_lookup_max: int = 8,
        max_model_len: int = 4096,
    ):
        # REFERENCE: vllm/v1/spec_decode/ngram_proposer.py:L13-L33
        assert prompt_lookup_min is not None
        assert prompt_lookup_max is not None
        # min/max length of the n-gram to match against the suffix.
        self.min_n = prompt_lookup_min
        self.max_n = prompt_lookup_max
        # K — number of tokens to emit after each match.
        self.k = num_speculative_tokens
        self.max_model_len = max_model_len

    # REFERENCE: vllm/v1/spec_decode/ngram_proposer.py:L131-L162 (propose)
    def propose(
        self,
        sampled_token_ids: List[List[int]],   # what the target sampled this step
        token_ids_per_req: List[np.ndarray],  # full context per request
    ) -> List[List[int]]:
        """For each request, find the longest matching suffix in its context.

        Returns a list of K-length draft proposals (or shorter if context runs
        out). Empty list for requests where no suffix matches.
        """
        out: List[List[int]] = []
        for i, sampled in enumerate(sampled_token_ids):
            context = token_ids_per_req[i]
            num_tokens = len(context)
            # Skip if the request is finished or context is too short.
            # REFERENCE: vllm/v1/spec_decode/ngram_proposer.py:L143-L151
            if not sampled or num_tokens >= self.max_model_len or num_tokens < self.min_n:
                out.append([])
                continue
            drafts = self._find_and_propose(context)
            out.append(drafts)
        return out

    # REFERENCE: vllm/v1/spec_decode/ngram_proposer.py:L198-L285
    # (Source's _find_longest_matched_ngram_and_propose_tokens uses KMP/LPS;
    #  we do the naive equivalent. Same output, simpler code.)
    def _find_and_propose(self, context: np.ndarray) -> List[int]:
        """Find the longest ngram in [min_n, max_n] that suffixes the context.

        Strategy: try ngram lengths from max_n down to min_n; for each length,
        scan back through context for a match of context[-n:]; emit the K
        tokens immediately following the LATEST match.
        """
        N = len(context)
        suffix = context[-self.max_n:] if N >= self.max_n else context
        # Try lengths from max_n down to min_n.
        for n in range(min(self.max_n, len(suffix)), self.min_n - 1, -1):
            target_ngram = context[-n:]
            # Search for match in context[:-n] (cannot match itself).
            for i in range(N - n - 1, -1, -1):
                if (context[i : i + n] == target_ngram).all():
                    # Found a match; emit K tokens after it.
                    end = i + n + self.k
                    end = min(end, N)
                    return context[i + n : end].tolist()
        return []


if __name__ == "__main__":
    print("=== NgramProposer toy demo ===")
    proposer = NgramProposer(num_speculative_tokens=3, prompt_lookup_min=2, prompt_lookup_max=4)
    # Context: "the cat sat on the mat ... the cat" → next should match "sat on the"
    context = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 10, 20])  # 10,20 ≈ "the cat"
    drafts = proposer.propose([[100]], [context])
    print(f"  context  = {context.tolist()}")
    print(f"  proposed = {drafts}  (should match next 3 tokens after first '10,20': [30,40,50])")
