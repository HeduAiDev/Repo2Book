"""
PAPER: arXiv:2401.15077 EAGLE §3.1 drafting phase -- reproduces vLLM v1's
*chain* special case of that drafting phase (dossier paper_origin_note: the
default vllm/v1/spec_decode/eagle.py + llm_base_proposer.py path runs a
chain, not the paper's dynamic tree; the real (batched, cudagraph'd) loop
this mirrors is llm_base_proposer.py:L392-L592 propose(), with the
"token left-shift + splice sampled token" construction at L646-L678).

propose_chain() below runs that same control flow on the toy
ToyTargetLLM / AutoregressionHead from feature_autoregression.py: one first
pass over the accepted prefix using the *shifted* token input (§3.1/Fig.3-5),
greedy-sampling the first draft token, then repeatedly feeding the
just-sampled token's embedding plus the just-produced feature back in for
`num_speculative_tokens - 1` more steps (§3.1: "f_3 and t_4 are concatenated
into the input sequence to predict the next feature f_4 and sample t_5").
"""
from __future__ import annotations

import torch

from feature_autoregression import AutoregressionHead, ToyTargetLLM, build_shifted_token_input


# PAPER: §3.1 ("From p_4=LM_Head(f_3), t_4 is sampled") -- greedy argmax sampling
def greedy_sample(target_llm: ToyTargetLLM, feature: torch.Tensor) -> tuple[int, float]:
    """
    Samples the next draft token greedily (argmax) from the shared LM Head's
    distribution over `feature`. Returns (token_id, confidence) where
    confidence is the draft model's own probability mass on that token --
    this is exactly the `c_j` used later by EAGLE-2's value calculation
    (draft_tree.py::compute_value, PAPER arXiv:2406.16858 §4.1/§3.2). vLLM's
    landing code does the analogous argmax in `_greedy_sample`
    (llm_base_proposer.py:L386-L390).
    """
    probs = target_llm.next_token_distribution(feature)
    token_id = int(torch.argmax(probs).item())
    confidence = float(probs[token_id].item())
    return token_id, confidence


# PAPER: §3.1 drafting phase, chain special case (vLLM's default eagle path)
def propose_chain(
    target_llm: ToyTargetLLM,
    draft_head: AutoregressionHead,
    prefix_token_ids: torch.Tensor,
    prefix_features: torch.Tensor,
    next_token_id: int,
    num_speculative_tokens: int,
) -> tuple[list[int], list[float]]:
    """
    prefix_token_ids / prefix_features: the target LLM's already-computed
    T_{1:j} / F_{1:j} for the accepted prefix (computing them is the target
    LLM's job -- out of scope for the draft model itself, exactly as in
    vLLM where `target_hidden_states` arrives ready-made from the model
    runner, gpu_model_runner.py:L4771-L4815).
    next_token_id: t_{j+1}, the token just sampled from the target LLM
    after F_{1:j} -- this is what gets spliced into the shifted input.

    Returns (draft_token_ids, confidences), each of length
    num_speculative_tokens.
    """
    assert num_speculative_tokens >= 1
    shifted_tokens = build_shifted_token_input(prefix_token_ids, next_token_id)
    token_embeds = target_llm.embed_tokens(shifted_tokens)

    # First pass: fuse the whole shifted-token/feature sequence; only the
    # *last* position's output feature is used to draft the first token
    # (mirrors `sample_hidden_states = last_hidden_states[token_indices_to_sample]`,
    # llm_base_proposer.py:L467).
    fused_features = draft_head(token_embeds, prefix_features)
    feature = fused_features[-1]

    draft_token_ids: list[int] = []
    confidences: list[float] = []
    token_id, confidence = greedy_sample(target_llm, feature)
    draft_token_ids.append(token_id)
    confidences.append(confidence)

    # PAPER: §3.1 chain continuation -- feed (token, feature) back in.
    for _ in range(num_speculative_tokens - 1):
        token_embed = target_llm.embed_tokens(torch.tensor(draft_token_ids[-1]))
        feature = draft_head(token_embed, feature)
        token_id, confidence = greedy_sample(target_llm, feature)
        draft_token_ids.append(token_id)
        confidences.append(confidence)

    return draft_token_ids, confidences
