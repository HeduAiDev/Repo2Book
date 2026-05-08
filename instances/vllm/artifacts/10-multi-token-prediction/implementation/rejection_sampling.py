"""Rejection sampling for speculative decoding — provably unbiased verifier.

Implements the algorithm of Chen et al. 2023 (https://arxiv.org/abs/2211.17192)
and Leviathan et al. 2023, exactly as vLLM's `vllm/v1/sample/rejection_sampler.py`
does at commit 98661fe — but in plain PyTorch / NumPy instead of Triton.

# REFERENCE: vllm/v1/sample/rejection_sampler.py:L37-L195   class RejectionSampler
# REFERENCE: vllm/v1/sample/rejection_sampler.py:L392-L503  def rejection_sample
# REFERENCE: vllm/v1/sample/rejection_sampler.py:L659-L703  def sample_recovered_tokens
# REFERENCE: vllm/v1/sample/rejection_sampler.py:L708-L757  rejection_greedy_sample_kernel
# REFERENCE: vllm/v1/sample/rejection_sampler.py:L762-L826  rejection_random_sample_kernel
# REFERENCE: vllm/v1/sample/rejection_sampler.py:L853-L920  sample_recovered_tokens_kernel

Two algorithm paths, same theorem:

  GREEDY path  (sampling_metadata.all_greedy=True):
     accept iff draft == argmax(target_logits); on first reject, emit
     argmax(target) at that position and stop. Bonus token if all accept.

  RANDOM path  (full algorithm, supports temperature/top-k/top-p):
     accept iff u < min(1, p(d)/q(d)) where u ~ Uniform[0,1]; on reject,
     sample recovered token from (p - q)_+ residual. Bonus token if all accept.

Theorem (Chen 2023, Theorem 1): the emitted token is distributed exactly as
the target distribution p, regardless of how bad q is. Proof — for any token x:
  P(emit x) = P(accept · x ~ q) + P(reject · x ~ recover)
            = q(x) * min(1, p(x)/q(x))
              + (1 - sum_y q(y) * min(1, p(y)/q(y))) * (p(x) - q(x))_+ / Z
   where  Z = sum_y (p(y) - q(y))_+ = 1 - sum_y q(y) * min(1, p(y)/q(y))
  → algebra → P(emit x) = p(x).

The implementation also preserves the "synthetic mode" testing path that
`SpeculativeConfig.rejection_sample_method='synthetic'` triggers: each
position has a hardcoded conditional acceptance rate, useful for reproducible
test numerics without needing a real model.
"""
# REFERENCE: vllm/v1/sample/rejection_sampler.py:L1-L30 (imports + constants)
from __future__ import annotations

from typing import List, Optional

import torch

from .spec_metadata import (  # noqa: F401  (re-export PLACEHOLDER_TOKEN_ID)
    GREEDY_TEMPERATURE,
    MAX_SPEC_LEN,
    PLACEHOLDER_TOKEN_ID,
    SpecDecodeMetadata,
)


# -----------------------------------------------------------------------------
# Greedy path  — vLLM kernel mirror
# -----------------------------------------------------------------------------
# REFERENCE: vllm/v1/sample/rejection_sampler.py:L708-L757 rejection_greedy_sample_kernel
def rejection_greedy_sample_loop(
    output_token_ids: torch.Tensor,           # [batch, max_spec_len + 1] int32
    cu_num_draft_tokens: torch.Tensor,        # [batch] int32
    draft_token_ids: torch.Tensor,            # [num_tokens] int32
    target_argmax: torch.Tensor,              # [num_tokens] int32
    bonus_token_ids: torch.Tensor,            # [batch] int32
    is_greedy: Optional[torch.Tensor],        # [batch] bool, None means "all greedy"
    max_spec_len: int,
    uniform_probs: Optional[torch.Tensor] = None,           # [num_tokens], synthetic mode
    synthetic_conditional_rates: Optional[torch.Tensor] = None,  # [K]
    SYNTHETIC_MODE: bool = False,
) -> None:
    """Pythonic mirror of the Triton greedy kernel.

    The kernel walks each request linearly. As long as `not rejected`, it
    compares draft id to target argmax and either accepts (writes draft) or
    rejects (writes target argmax and sets rejected=True). After the rejection
    point, no more positions are written — they stay as PLACEHOLDER_TOKEN_ID.
    The bonus token is appended only if the entire chain accepted.

    # REFERENCE: vllm/v1/sample/rejection_sampler.py:L720-L757

    NOTE: The +1 token slot for bonus is at position `num_draft_tokens`
    (zero-indexed), NOT max_spec_len — different requests have different K.
    """
    batch_size = bonus_token_ids.shape[0]
    for req_idx in range(batch_size):
        # Per-request greedy gate. Source allows is_greedy_ptr=None to mean
        # "everyone is greedy" (early exit on the all_greedy fast path).
        # REFERENCE: vllm/v1/sample/rejection_sampler.py:L723
        if is_greedy is None:
            req_is_greedy = True
        else:
            req_is_greedy = bool(is_greedy[req_idx])
        if not req_is_greedy:
            continue

        start_idx = 0 if req_idx == 0 else int(cu_num_draft_tokens[req_idx - 1])
        end_idx = int(cu_num_draft_tokens[req_idx])
        num_draft_tokens = end_idx - start_idx

        rejected = False
        for pos in range(num_draft_tokens):
            if rejected:
                # Chain-break invariant: once we reject, all later positions
                # in this request stay PLACEHOLDER_TOKEN_ID. The Triton kernel
                # implements this by not writing — we mirror that by `break`.
                # REFERENCE: vllm/v1/sample/rejection_sampler.py:L734
                break
            draft_id = int(draft_token_ids[start_idx + pos])
            target_id = int(target_argmax[start_idx + pos])

            if SYNTHETIC_MODE:
                # SIMPLIFIED: synthetic mode uses a precomputed per-position
                # conditional rate — accept iff u < rate(pos). When rejected,
                # emit target_id (greedy fallback). No "recovered" sampling
                # in greedy path because there's no probability distribution.
                # REFERENCE: vllm/v1/sample/rejection_sampler.py:L737-L742
                u = float(uniform_probs[start_idx + pos])
                rate = float(synthetic_conditional_rates[pos])
                accepted = u < rate
                token_id = draft_id if accepted else target_id
                rejected = not accepted
            else:
                # Standard greedy: deterministic accept iff draft == argmax.
                # REFERENCE: vllm/v1/sample/rejection_sampler.py:L743-L745
                token_id = target_id  # source writes target_id always; if accepted,
                # draft_id == target_id so it doesn't matter
                rejected = draft_id != target_id

            output_token_ids[req_idx, pos] = token_id

        if not rejected:
            # All K accepted — append the bonus token at slot num_draft_tokens.
            # REFERENCE: vllm/v1/sample/rejection_sampler.py:L751-L757
            output_token_ids[req_idx, num_draft_tokens] = bonus_token_ids[req_idx]


# -----------------------------------------------------------------------------
# Random path — vLLM kernel mirror
# -----------------------------------------------------------------------------
# REFERENCE: vllm/v1/sample/rejection_sampler.py:L762-L826 rejection_random_sample_kernel
def rejection_random_sample_loop(
    output_token_ids: torch.Tensor,           # [batch, max_spec_len + 1] int32
    cu_num_draft_tokens: torch.Tensor,        # [batch] int32
    draft_token_ids: torch.Tensor,            # [num_tokens] int32
    draft_probs: Optional[torch.Tensor],      # [num_tokens, vocab] or None (ngram)
    target_probs: torch.Tensor,               # [num_tokens, vocab]
    bonus_token_ids: torch.Tensor,            # [batch] int32
    recovered_token_ids: torch.Tensor,        # [num_tokens] int32 (precomputed)
    uniform_probs: torch.Tensor,              # [num_tokens] float
    is_greedy: torch.Tensor,                  # [batch] bool — skip greedy reqs
    max_spec_len: int,
    synthetic_conditional_rates: Optional[torch.Tensor] = None,
    NO_DRAFT_PROBS: bool = False,
    SYNTHETIC_MODE: bool = False,
) -> None:
    """Pythonic mirror of the Triton random-sample kernel.

    For each non-greedy request, walk positions left-to-right:
      1. Sample u ~ Uniform[0,1] (precomputed in `uniform_probs`).
      2. Standard mode: accept iff p(d) / q(d) >= u, where p is target_probs
         and q is draft_probs. NO_DRAFT_PROBS (ngram) treats q(d) = 1, so
         the test becomes p(d) >= u — reduces to "accept if target gives
         draft any non-trivial mass".
      3. On reject, emit recovered_token_ids[pos] (precomputed via
         sample_recovered_tokens) and STOP — chain-break invariant.

    # REFERENCE: vllm/v1/sample/rejection_sampler.py:L789-L818
    """
    batch_size = bonus_token_ids.shape[0]
    vocab_size = target_probs.shape[-1]

    for req_idx in range(batch_size):
        # Skip greedy requests (they were handled by the greedy kernel above).
        # REFERENCE: vllm/v1/sample/rejection_sampler.py:L779-L782
        if bool(is_greedy[req_idx]):
            continue

        start_idx = 0 if req_idx == 0 else int(cu_num_draft_tokens[req_idx - 1])
        end_idx = int(cu_num_draft_tokens[req_idx])
        num_draft_tokens = end_idx - start_idx

        rejected = False
        for pos in range(num_draft_tokens):
            if rejected:
                break
            draft_id = int(draft_token_ids[start_idx + pos])
            u = float(uniform_probs[start_idx + pos])

            if SYNTHETIC_MODE:
                # REFERENCE: vllm/v1/sample/rejection_sampler.py:L793-L795
                rate = float(synthetic_conditional_rates[pos])
                accepted = u < rate
            else:
                # Compute accept ratio min(1, p(d)/q(d)) — but the source
                # tests `target/draft >= u` rather than `u < target/draft`,
                # which is equivalent for u in [0,1).
                # REFERENCE: vllm/v1/sample/rejection_sampler.py:L797-L810
                if NO_DRAFT_PROBS:
                    draft_prob = 1.0  # ngram: no q. Accept iff p(d) >= u.
                else:
                    draft_prob = float(draft_probs[start_idx + pos, draft_id])
                target_prob = float(target_probs[start_idx + pos, draft_id])
                # Source uses `draft_prob > 0 and target_prob / draft_prob >= u`
                # — guards against q(d) = 0 (which "should never happen" but
                # is defensively rejected).
                accepted = draft_prob > 0.0 and (target_prob / draft_prob) >= u

            if accepted:
                token_id = draft_id
            else:
                # Recovered token sampled from (p - q)_+ residual.
                # REFERENCE: vllm/v1/sample/rejection_sampler.py:L811-L815
                rejected = True
                token_id = int(recovered_token_ids[start_idx + pos])
            output_token_ids[req_idx, pos] = token_id

        if not rejected:
            # REFERENCE: vllm/v1/sample/rejection_sampler.py:L820-L826
            output_token_ids[req_idx, num_draft_tokens] = bonus_token_ids[req_idx]


# -----------------------------------------------------------------------------
# Recovered-token sampling — vLLM kernel mirror
# -----------------------------------------------------------------------------
# REFERENCE: vllm/v1/sample/rejection_sampler.py:L853-L920 sample_recovered_tokens_kernel
def sample_recovered_tokens_loop(
    cu_num_draft_tokens: torch.Tensor,        # [batch]
    draft_token_ids: torch.Tensor,            # [num_tokens]
    draft_probs: Optional[torch.Tensor],      # [num_tokens, vocab] or None
    target_probs: torch.Tensor,               # [num_tokens, vocab]
    inv_q: torch.Tensor,                      # [batch, vocab] = 1 / Exp(1) per req
    NO_DRAFT_PROBS: bool = False,
) -> torch.Tensor:
    """Sample one recovered token per draft position via Gumbel-max trick.

    The source does `q ~ Exp(1)` over vocab; then for each position, picks
    argmax over `(p - q_draft)_+ * (1/q_exp)` — which is the Gumbel-max trick
    for sampling from the residual distribution `(p - q_draft)_+`.
    For NO_DRAFT_PROBS (ngram), the residual is just `p` with the draft id
    masked out (it's already known to be rejected).

    # REFERENCE: vllm/v1/sample/rejection_sampler.py:L877-L920

    Returns:
        recovered_token_ids — [num_tokens] int. Index i is the residual sample
        for draft position i. Used by `rejection_random_sample_loop` only when
        rejection actually occurs at position i; otherwise it's wasted work
        (vLLM still computes it because GPU branch divergence costs more than
        wasted work).
    """
    batch_size = cu_num_draft_tokens.shape[0]
    vocab_size = target_probs.shape[-1]
    num_tokens = draft_token_ids.shape[0]
    recovered = torch.zeros(num_tokens, dtype=torch.long, device=target_probs.device)

    for req_idx in range(batch_size):
        start_idx = 0 if req_idx == 0 else int(cu_num_draft_tokens[req_idx - 1])
        end_idx = int(cu_num_draft_tokens[req_idx])
        num_draft = end_idx - start_idx

        for pos in range(num_draft):
            token_idx = start_idx + pos
            if NO_DRAFT_PROBS:
                # ngram path: residual = target_probs with draft_id masked.
                # REFERENCE: vllm/v1/sample/rejection_sampler.py:L877-L891
                draft_id = int(draft_token_ids[token_idx])
                p = target_probs[token_idx].clone()
                p[draft_id] = 0.0
            else:
                # Standard residual: (p - q)_+
                # REFERENCE: vllm/v1/sample/rejection_sampler.py:L892-L903
                p = torch.clamp(
                    target_probs[token_idx] - draft_probs[token_idx], min=0.0
                )
            # Gumbel-max via Exp(1) inv = 1/q : argmax of p * inv_q
            # NOTE: source skips normalization because argmax is invariant to
            # positive scaling — `prob = prob / sum(prob)` would be wasted ops.
            # REFERENCE: vllm/v1/sample/rejection_sampler.py:L904-L905
            score = p * inv_q[req_idx]
            recovered[token_idx] = int(torch.argmax(score))

    return recovered


# -----------------------------------------------------------------------------
# Top-level driver — vLLM mirror of `def rejection_sample`
# -----------------------------------------------------------------------------
# REFERENCE: vllm/v1/sample/rejection_sampler.py:L392-L503
def rejection_sample(
    metadata: SpecDecodeMetadata,
    draft_probs: Optional[torch.Tensor],   # [num_tokens, vocab] or None
    target_logits: torch.Tensor,           # [num_tokens, vocab] AFTER temperature/top-k/top-p
    bonus_token_ids: torch.Tensor,         # [batch] int32 — already sampled by `Sampler`
    all_greedy: bool,
    all_random: bool,
    is_greedy_per_req: Optional[torch.Tensor] = None,  # [batch] bool
    generator: Optional[torch.Generator] = None,
    synthetic_mode: bool = False,
    synthetic_conditional_rates: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Drive both kernels. Returns output_token_ids of shape [batch, max_spec_len + 1].

    # REFERENCE: vllm/v1/sample/rejection_sampler.py:L409-L503
    """
    assert metadata.max_spec_len <= MAX_SPEC_LEN, (
        f"max_spec_len {metadata.max_spec_len} exceeds MAX_SPEC_LEN {MAX_SPEC_LEN}"
    )

    batch_size = len(metadata.num_draft_tokens)
    num_tokens = metadata.draft_token_ids.shape[0]
    vocab_size = target_logits.shape[-1]
    device = target_logits.device

    # Output buffer pre-filled with PLACEHOLDER_TOKEN_ID.
    # REFERENCE: vllm/v1/sample/rejection_sampler.py:L425-L430
    output_token_ids = torch.full(
        (batch_size, metadata.max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )

    # Determine which requests are greedy.
    # REFERENCE: vllm/v1/sample/rejection_sampler.py:L432-L435
    if all_greedy:
        is_greedy = None  # signal "all greedy"
    elif is_greedy_per_req is not None:
        is_greedy = is_greedy_per_req
    else:
        # Default to all-random
        is_greedy = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Generate uniform probs (always for synthetic mode, also for non-all-greedy).
    # REFERENCE: vllm/v1/sample/rejection_sampler.py:L441-L448
    uniform_probs: Optional[torch.Tensor] = None
    if synthetic_mode or not all_greedy:
        if generator is None:
            uniform_probs = torch.rand(num_tokens, dtype=torch.float64, device=device)
        else:
            uniform_probs = torch.empty(num_tokens, dtype=torch.float64, device=device)
            uniform_probs.uniform_(generator=generator)

    # Greedy path (if any greedy req exists).
    # REFERENCE: vllm/v1/sample/rejection_sampler.py:L450-L466
    if not all_random:
        target_argmax = target_logits.argmax(dim=-1).to(torch.int32)
        rejection_greedy_sample_loop(
            output_token_ids,
            metadata.cu_num_draft_tokens,
            metadata.draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            metadata.max_spec_len,
            uniform_probs,
            synthetic_conditional_rates,
            SYNTHETIC_MODE=synthetic_mode,
        )
        if all_greedy:
            return output_token_ids

    # Random path: softmax target logits, sample recovered tokens, then sample.
    # REFERENCE: vllm/v1/sample/rejection_sampler.py:L468-L502
    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32).contiguous()

    # Build inv_q via Exp(1) draws — the Gumbel-max trick for residual sampling.
    # REFERENCE: vllm/v1/sample/rejection_sampler.py:L674-L688
    if generator is None:
        q = torch.empty((batch_size, vocab_size), dtype=torch.float32, device=device)
        q.exponential_()
    else:
        q = torch.empty((batch_size, vocab_size), dtype=torch.float32, device=device)
        q.exponential_(generator=generator)
    inv_q = q.reciprocal()

    recovered_token_ids = sample_recovered_tokens_loop(
        metadata.cu_num_draft_tokens,
        metadata.draft_token_ids,
        draft_probs,
        target_probs,
        inv_q,
        NO_DRAFT_PROBS=draft_probs is None,
    ).to(torch.int32)

    assert uniform_probs is not None  # invariant: random path always has uniforms
    rejection_random_sample_loop(
        output_token_ids,
        metadata.cu_num_draft_tokens,
        metadata.draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy if is_greedy is not None
            else torch.zeros(batch_size, dtype=torch.bool, device=device),
        metadata.max_spec_len,
        synthetic_conditional_rates,
        NO_DRAFT_PROBS=draft_probs is None,
        SYNTHETIC_MODE=synthetic_mode,
    )
    return output_token_ids


# -----------------------------------------------------------------------------
# Output parsing — strip PLACEHOLDER and short rows
# -----------------------------------------------------------------------------
# REFERENCE: vllm/v1/sample/rejection_sampler.py:L246-L281 RejectionSampler.parse_output
def parse_output(
    output_token_ids: torch.Tensor,    # [batch, max_spec_len + 1]
    vocab_size: int,
) -> List[List[int]]:
    """Strip PLACEHOLDER_TOKEN_ID and out-of-vocab values per request.

    The source also gates on `discard_req_indices` (e.g. requests that just
    finished); we omit that for clarity since it's orthogonal to the algorithm.
    """
    out = output_token_ids.cpu().numpy()
    valid_mask = (out != PLACEHOLDER_TOKEN_ID) & (out < vocab_size)
    return [row[valid_mask[i]].tolist() for i, row in enumerate(out)]


# -----------------------------------------------------------------------------
# RejectionSampler — pedagogical wrapper around `rejection_sample`
# -----------------------------------------------------------------------------
class RejectionSampler:
    """Pedagogical mirror of `vllm.v1.sample.rejection_sampler.RejectionSampler`.

    # REFERENCE: vllm/v1/sample/rejection_sampler.py:L37-L195

    NOTE on simplification:
      - Source extends `nn.Module` and consumes a `Sampler` to generate bonus
        tokens via top-k/top-p; we accept bonus_token_ids pre-computed.
      - Source supports `logprobs_mode='processed_logits' / 'raw_logits'` for
        downstream logprob accounting; we omit, irrelevant to algorithm.
      - Source uses CUDA Triton kernels; we use Python loops. ~100x slower,
        same semantics.
    """

    def __init__(self, synthetic_conditional_rates: Optional[torch.Tensor] = None):
        # REFERENCE: vllm/v1/sample/rejection_sampler.py:L60-L86 (synthetic mode init)
        self.synthetic_conditional_rates = synthetic_conditional_rates
        self.synthetic_mode = synthetic_conditional_rates is not None

    def __call__(
        self,
        metadata: SpecDecodeMetadata,
        draft_probs: Optional[torch.Tensor],
        target_logits: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        all_greedy: bool = False,
        all_random: bool = True,
        is_greedy_per_req: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        return rejection_sample(
            metadata=metadata,
            draft_probs=draft_probs,
            target_logits=target_logits,
            bonus_token_ids=bonus_token_ids,
            all_greedy=all_greedy,
            all_random=all_random,
            is_greedy_per_req=is_greedy_per_req,
            generator=generator,
            synthetic_mode=self.synthetic_mode,
            synthetic_conditional_rates=self.synthetic_conditional_rates,
        )


# -----------------------------------------------------------------------------
# Demo when run directly
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    print("=== Rejection sampling — toy demo (vocab=8, batch=3) ===")

    # Build a metadata for batch=3, K varying.
    drafts = [[3, 4, 5], [1, 2], [7]]
    md = SpecDecodeMetadata.make_dummy(drafts, device="cpu")
    print(f"  num_draft_tokens = {md.num_draft_tokens}")

    vocab = 8
    num_tokens = md.draft_token_ids.shape[0]
    target_logits = torch.randn(num_tokens, vocab) * 2.0
    draft_probs = torch.rand(num_tokens, vocab)
    draft_probs /= draft_probs.sum(dim=-1, keepdim=True)
    bonus_token_ids = torch.randint(0, vocab, (3,), dtype=torch.int32)

    g = torch.Generator().manual_seed(42)
    out = rejection_sample(
        md, draft_probs, target_logits, bonus_token_ids,
        all_greedy=False, all_random=True, generator=g,
    )
    print(f"  output_token_ids shape = {tuple(out.shape)}  (= [batch, max_spec_len + 1])")
    print(f"  output (raw, with placeholders -1) =\n{out.tolist()}")
    parsed = parse_output(out, vocab)
    print(f"  parsed (placeholders stripped) = {parsed}")
    print()
    print("Greedy path (target argmax = draft → accept; else reject + emit argmax):")
    g2 = torch.Generator().manual_seed(42)
    out_g = rejection_sample(
        md, None, target_logits, bonus_token_ids,
        all_greedy=True, all_random=False, generator=g2,
    )
    print(f"  parsed greedy = {parse_output(out_g, vocab)}")
