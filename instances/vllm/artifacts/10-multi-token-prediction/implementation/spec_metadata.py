"""SpecDecodeMetadata — the data contract between proposer and rejection sampler.

# REFERENCE: vllm/v1/spec_decode/metadata.py:L1-L66

In vLLM, the proposer (EagleProposer / MedusaProposer / NgramProposer / ...)
emits a SpecDecodeMetadata describing the K draft tokens per request, plus the
indices into the target model's logit output where bonus and target logits live.
The RejectionSampler then consumes this dataclass to drive its kernels.

The metadata is a *passive* data structure — no methods, no behavior.
Behavioral logic lives entirely in the proposer (which fills the fields) and
the sampler (which reads them). This is the cleanest backpressure-gate pattern
in vLLM: the proposer-sampler boundary is a dataclass.

Fields (verbatim names; same shapes as source):

| Field                  | Shape                  | Meaning                                   |
|------------------------|------------------------|-------------------------------------------|
| draft_token_ids        | [num_tokens]           | Flattened K-per-request draft proposals    |
| num_draft_tokens       | list[int], len=batch   | Per-request K (can vary per request)       |
| cu_num_draft_tokens    | [batch_size]           | Cumulative sum (exclusive at start). The chain-break invariant uses this. |
| cu_num_sampled_tokens  | [batch_size]           | Cumulative sum of (K_i + 1) — bonus + drafts |
| target_logits_indices  | [num_tokens]           | Where target logits for each draft live     |
| bonus_logits_indices   | [batch_size]           | Where bonus logits (sampled iff all-accept) live |
| logits_indices         | [num_tokens + batch]   | Concatenated target + bonus indices         |
"""

# REFERENCE: vllm/v1/spec_decode/metadata.py:L9-L24
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch


@dataclass
class SpecDecodeMetadata:
    """Dataclass mirror of vLLM's SpecDecodeMetadata.

    # REFERENCE: vllm/v1/spec_decode/metadata.py:L9-L24
    """

    # [num_tokens] — flattened across the batch
    draft_token_ids: torch.Tensor
    # [batch_size] — Python list (NOT a tensor) per source.
    num_draft_tokens: List[int]
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor
    # [batch_size]
    cu_num_sampled_tokens: torch.Tensor
    # [num_tokens]
    target_logits_indices: torch.Tensor
    # [batch_size]
    bonus_logits_indices: torch.Tensor
    # [num_tokens + batch_size]
    logits_indices: torch.Tensor

    # Set in __post_init__ from num_draft_tokens; used by the kernels
    # to size the output buffer as [batch, max_spec_len + 1].
    max_spec_len: int = field(init=False)

    def __post_init__(self):
        # REFERENCE: vllm/v1/spec_decode/metadata.py:L26-L27
        # The kernel output buffer is shape [batch, max_spec_len + 1].
        # The +1 slot is the bonus token if all K accept.
        self.max_spec_len = max(self.num_draft_tokens) if self.num_draft_tokens else 0

    @classmethod
    def make_dummy(
        cls,
        draft_token_ids: List[List[int]],
        device: torch.device | str = "cpu",
    ) -> "SpecDecodeMetadata":
        """Build a dummy metadata from per-request draft lists.

        # REFERENCE: vllm/v1/spec_decode/metadata.py:L29-L66

        We faithfully reproduce the source's flatten + cumsum logic; this is
        the single-call factory used by tests and demos.
        """
        device = torch.device(device)
        batch_size = len(draft_token_ids)
        num_draft_tokens = [len(ids) for ids in draft_token_ids]
        # +1 per request because the bonus token is sampled when all K accept.
        num_sampled_tokens = [len(ids) + 1 for ids in draft_token_ids]
        flat = sum(draft_token_ids, [])
        num_tokens = len(flat)

        draft_token_ids_t = torch.tensor(flat, dtype=torch.int32, device=device)
        cu_draft = np.cumsum(num_draft_tokens, dtype=np.int32)
        cu_sampled = np.cumsum(num_sampled_tokens, dtype=np.int32)

        # In the real source, target_logits_indices and bonus_logits_indices
        # are computed by the proposer based on its slot mapping. For pedagogy
        # and tests, we just set them to a canonical 0..N range — the exact
        # values do not affect the rejection-sampling algebra.
        target_logits_indices = torch.arange(num_tokens, dtype=torch.int32, device=device)
        bonus_logits_indices = torch.arange(
            num_tokens, num_tokens + batch_size, dtype=torch.int32, device=device
        )
        logits_indices = torch.arange(
            num_tokens + batch_size, dtype=torch.int32, device=device
        )

        return cls(
            draft_token_ids=draft_token_ids_t,
            num_draft_tokens=num_draft_tokens,
            cu_num_draft_tokens=torch.from_numpy(cu_draft).to(device),
            cu_num_sampled_tokens=torch.from_numpy(cu_sampled).to(device),
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )


# REFERENCE: vllm/v1/sample/rejection_sampler.py:L30
PLACEHOLDER_TOKEN_ID: int = -1
"""Marker for non-emitted positions in the rejection-sampler output buffer.

When a draft is rejected at position i, all positions i+1..K are also marked
PLACEHOLDER_TOKEN_ID — this is the chain-break invariant. parse_output filters
these out at the end. -1 is chosen because no valid token id is negative."""

GREEDY_TEMPERATURE: int = 0
"""Sentinel temperature value meaning 'argmax, no sampling'.

# REFERENCE: vllm/v1/sample/rejection_sampler.py:L31
"""

MAX_SPEC_LEN: int = 128
"""Hard limit on K per request per step (prevents pathological draft chains).

# REFERENCE: vllm/v1/sample/rejection_sampler.py:L34
"""


if __name__ == "__main__":
    # Tiny demo: build dummy metadata, print fields, show the chain-break shape.
    drafts = [[101, 102, 103], [201, 202], [301]]  # batch=3, K_i in {3, 2, 1}
    md = SpecDecodeMetadata.make_dummy(drafts, device="cpu")
    print("=== SpecDecodeMetadata dummy ===")
    print(f"batch_size       = {len(md.num_draft_tokens)}")
    print(f"num_draft_tokens = {md.num_draft_tokens}")
    print(f"max_spec_len     = {md.max_spec_len}")
    print(f"cu_num_draft     = {md.cu_num_draft_tokens.tolist()}")
    print(f"cu_num_sampled   = {md.cu_num_sampled_tokens.tolist()}")
    print(f"draft_token_ids  = {md.draft_token_ids.tolist()}")
    print()
    print("Output buffer shape would be:")
    bs = len(md.num_draft_tokens)
    print(f"  [batch, max_spec_len + 1] = [{bs}, {md.max_spec_len + 1}]")
    print(f"  Pre-filled with PLACEHOLDER_TOKEN_ID = {PLACEHOLDER_TOKEN_ID}")
    print(f"  (Rejected positions stay as PLACEHOLDER, so the chain-break is implicit.)")
