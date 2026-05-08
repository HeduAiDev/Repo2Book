"""MTP weight name remapping and lm_head sharing — the inference-time response
to training-time multi-step CE loss.

# OUTLINE-REFRAME: outline §3 says "Training -- multi-step CE loss weighted strategy"
# but vLLM is INFERENCE-ONLY. Reframed at chapter level to:
# "Inference-time MTP weight loading and weight sharing".
#
# Sidebar grounding (training side, NOT in vllm/):
#   L_MTP = sum_{k=0..K-1} lambda_k * CE(p_k, x_{i+k})
#   with lambda_k typically decaying 1.0 / 0.5 / 0.25 / ... per step
# (DeepSeek-V3 paper, Switch-Transformer-style multi-step supervision).
#
# Pivot: vLLM's inference response to that training is just "load the MTP
# weights and share what should be shared". This file implements the two
# load-bearing helpers:
#
#   _rewrite_spec_layer_name  — rewrite HF checkpoint names to vLLM's MTP layout
#                               (REFERENCE: vllm/model_executor/models/deepseek_mtp.py:L458-L488)
#
#   maybe_share_lm_head       — share target.lm_head with each MTP layer's
#                               shared_head.head (saves vocab*hidden params * K)
#                               (REFERENCE: vllm/v1/spec_decode/llm_base_proposer.py:L1471-L1538)
#
# Verified absence of training code in vllm/:
#   $ grep -rn 'MTPLoss\|multi_step_ce\|compute_mtp_loss\|mtp_aux_loss' instances/vllm/source/vllm/
#   (no results)
"""
# REFERENCE: vllm/model_executor/models/deepseek_mtp.py:L458-L488
# REFERENCE: vllm/v1/spec_decode/llm_base_proposer.py:L1402-L1574
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


# REFERENCE: vllm/model_executor/models/deepseek_mtp.py:L458-L488
def rewrite_spec_layer_name(spec_layer: int, name: str) -> str:
    """Mirror of `DeepSeekMTP._rewrite_spec_layer_name`.

    # REFERENCE: vllm/model_executor/models/deepseek_mtp.py:L458-L488

    HuggingFace checkpoints from DeepSeek-V3 ship MTP weights with names like

        model.layers.{spec_layer}.self_attn.q_proj.weight       (target-style)
        model.layers.{spec_layer}.embed_tokens.weight           (shared)
        model.layers.{spec_layer}.enorm.weight                  (MTP-specific)
        model.layers.{spec_layer}.eh_proj.weight                (MTP-specific)
        model.layers.{spec_layer}.shared_head.head.weight       (MTP-specific)

    The vLLM model-class layout differs slightly: it expects the transformer
    block weights nested under `mtp_block.*`, and the shared embed_tokens
    promoted to top-level. So the rewriter does THREE things:

      (1) For non-MTP-specific weights at spec_layer, insert `.mtp_block.`
          after the layer index:
            model.layers.{N}.self_attn.q_proj.weight
              → model.layers.{N}.mtp_block.self_attn.q_proj.weight

      (2) For embed_tokens (a SHARED weight), promote to top-level by
          stripping the layer prefix:
            model.layers.{N}.embed_tokens.weight
              → model.embed_tokens.weight

      (3) For other MTP-specific weights (enorm/hnorm/eh_proj/shared_head),
          leave unchanged.

    Returns: the rewritten name. Note: vLLM keeps the layer index (spec_layer)
    UNCHANGED — it does NOT reindex from 0. The MTP layers live at
    indices [num_target_layers, num_target_layers + num_mtp_layers) inside
    the ModuleDict, matching the HF layer numbering.
    """
    # REFERENCE: vllm/model_executor/models/deepseek_mtp.py:L464-L470
    spec_layer_weight_names = [
        "embed_tokens",
        "enorm",
        "hnorm",
        "eh_proj",
        "shared_head",
    ]
    # Of these, embed_tokens is also a "shared weight" (promoted to top level).
    shared_weight_names = ["embed_tokens"]
    spec_layer_weight = False
    shared_weight = False
    for w in spec_layer_weight_names:
        if w in name:
            spec_layer_weight = True
            if w in shared_weight_names:
                shared_weight = True
            break

    # REFERENCE: vllm/model_executor/models/deepseek_mtp.py:L480-L488
    if not spec_layer_weight:
        # Path (1): regular transformer block weight → wrap under .mtp_block.
        name = name.replace(
            f"model.layers.{spec_layer}.", f"model.layers.{spec_layer}.mtp_block."
        )
    elif shared_weight:
        # Path (2): promote shared weight to top level.
        name = name.replace(f"model.layers.{spec_layer}.", "model.")
    # Path (3): MTP-specific non-shared weight — leave name unchanged.
    return name


def remap_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    mtp_start_layer_idx: int,
    num_mtp_layers: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Split a HuggingFace state_dict into (target_weights, mtp_weights) with
    MTP weights renamed to vLLM's internal scheme.

    # REFERENCE: vllm/model_executor/models/deepseek_mtp.py:L271-L456 (load_weights)
    # The source iterates the HF state_dict, calls
    # `get_spec_layer_idx_from_weight_name` to detect MTP-layer weights,
    # then calls `_rewrite_spec_layer_name` to produce the vLLM-internal name.

    Returns:
        (target_state_dict, mtp_state_dict): both keyed by the rewritten
        names. MTP weights keep their layer index (matching the source's
        ModuleDict behavior).
    """
    target_sd: Dict[str, torch.Tensor] = {}
    mtp_sd: Dict[str, torch.Tensor] = {}
    spec_range = range(mtp_start_layer_idx, mtp_start_layer_idx + num_mtp_layers)

    for name, tensor in state_dict.items():
        # Determine spec_layer (returns idx if name has a layer index in the
        # MTP range, else None — matches `get_spec_layer_idx_from_weight_name`)
        spec_layer = None
        for idx in spec_range:
            if f"model.layers.{idx}." in name:
                spec_layer = idx
                break
        if spec_layer is None:
            target_sd[name] = tensor
            continue
        new_name = rewrite_spec_layer_name(spec_layer, name)
        # Path (2) outputs may collide across layers (only one
        # model.embed_tokens.weight should survive); keep last write.
        mtp_sd[new_name] = tensor
    return target_sd, mtp_sd


# REFERENCE: vllm/v1/spec_decode/llm_base_proposer.py:L1471-L1539
def maybe_share_lm_head(
    target_language_model: nn.Module,
    mtp_predictor: nn.Module,
) -> int:
    """Share the target's lm_head with each MTP layer's shared_head.head.

    # REFERENCE: vllm/v1/spec_decode/llm_base_proposer.py:L1471-L1539

    Why share: each MTP layer has its OWN ParallelLMHead instance pre-share.
    For DeepSeek-V3 with vocab=129280 and hidden=7168, that's
    129280 * 7168 ≈ 926M params per MTP layer's lm_head. With K=1 MTP layer,
    sharing saves 0.93B parameters. With K=4 it saves ~3.7B parameters.

    The source has explicit detection logic (EAGLE has has_own_lm_head flag,
    MTP doesn't), then runs over `model.model.layers.values()` and replaces
    each layer's `shared_head.head` with the target's `lm_head`.

    Returns: number of layers updated (for logging).
    """
    # REFERENCE: vllm/v1/spec_decode/llm_base_proposer.py:L1509-L1515
    # MTP path: always share (no has_own_lm_head flag).
    if not hasattr(target_language_model, "lm_head"):
        raise AttributeError(
            "target_language_model must have an `lm_head` attribute. "
            "REFERENCE: vllm/v1/spec_decode/llm_base_proposer.py:L1517"
        )
    target_lm_head = target_language_model.lm_head

    # REFERENCE: vllm/v1/spec_decode/llm_base_proposer.py:L1517-L1520
    if hasattr(mtp_predictor, "lm_head"):
        del mtp_predictor.lm_head
    mtp_predictor.lm_head = target_lm_head

    # REFERENCE: vllm/v1/spec_decode/llm_base_proposer.py:L1522-L1538
    # Inside each MTP layer's shared_head.head, also replace.
    # The source quote:
    #   "MTP models call compute_logits via shared_head.head (a ParallelLMHead
    #    inside each MTP layer), not self.model.lm_head. If the checkpoint
    #    omits a copy of the lm_head weights at the MTP layer path,
    #    shared_head.head stays uninitialised and produces NaN logits.
    #    Always share it explicitly."
    inner = getattr(mtp_predictor, "model", mtp_predictor)
    layers = getattr(inner, "layers", None)
    if layers is None:
        return 0
    items = layers.values() if isinstance(layers, nn.ModuleDict) else layers
    n = 0
    for layer in items:
        sh = getattr(layer, "shared_head", None)
        if sh is not None and hasattr(sh, "head"):
            del sh.head
            sh.head = target_lm_head
            n += 1
    return n


# REFERENCE: vllm/v1/spec_decode/llm_base_proposer.py:L1402-L1469 (_maybe_share_embeddings)
def maybe_share_embeddings(
    target_language_model: nn.Module,
    mtp_predictor: nn.Module,
) -> bool:
    """Share target's embed_tokens with the MTP predictor.

    # REFERENCE: vllm/v1/spec_decode/llm_base_proposer.py:L1453-L1465

    For MTP models (no has_own_embed_tokens flag), always share — the source
    quote: "Detected MTP model. Sharing target model embedding weights with
    the draft model."
    """
    inner_target = getattr(target_language_model, "model", target_language_model)
    if hasattr(inner_target, "embed_tokens"):
        target_embed = inner_target.embed_tokens
    elif hasattr(inner_target, "embedding"):
        target_embed = inner_target.embedding
    else:
        raise AttributeError(
            "Target model has neither 'embed_tokens' nor 'embedding'. "
            "REFERENCE: vllm/v1/spec_decode/llm_base_proposer.py:L1413-L1419"
        )

    # REFERENCE: vllm/v1/spec_decode/llm_base_proposer.py:L1461-L1464
    inner_mtp = getattr(mtp_predictor, "model", mtp_predictor)
    if hasattr(inner_mtp, "embed_tokens"):
        del inner_mtp.embed_tokens
    inner_mtp.embed_tokens = target_embed
    return True


# -----------------------------------------------------------------------------
# Synthetic acceptance-rate helpers — the testing-time contract.
# REFERENCE: vllm/config/speculative.py:L213-L227
# -----------------------------------------------------------------------------
def acceptance_length_to_rates(length: float, n: int) -> list:
    """Convert mean acceptance length to per-position UNCONDITIONAL rates.

    # REFERENCE: vllm/config/speculative.py:L213-L227 (_acceptance_length_to_rates)

    Source uses a "minimum-variance schedule": fill positions 0..floor(L-1) with
    rate=1.0, then one fractional position, then zeros. So if length=3.4, K=5:
      rates = [1.0, 1.0, 0.4, 0.0, 0.0]

    Useful for synthetic-mode rejection sampling tests where you want a precise
    expected acceptance length, not a noisy Bernoulli simulation.
    """
    num_drafts = length - 1.0
    num_full = int(num_drafts)
    fractional = num_drafts - num_full
    if num_full >= n:
        return [1.0] * n
    rates = [1.0] * num_full + [fractional] + [0.0] * (n - num_full - 1)
    return rates[:n]


def unconditional_to_conditional_rates(unconditional: list) -> list:
    """Convert UNCONDITIONAL acceptance rates to CONDITIONAL rates.

    # REFERENCE: vllm/v1/spec_decode/utils.py unconditional_to_conditional_rates
    # (Helper used by RejectionSampler synthetic mode at __init__ time.)

    Unconditional rate at position i = P(positions 0..i all accept) — a
    monotonically non-increasing sequence ending in 0.
    Conditional rate at position i = P(position i accepts | 0..i-1 all accept).
    Relation: cond[i] = uncond[i] / uncond[i-1] (with uncond[-1] = 1).

    For unconditional [0.7, 0.4, 0.1]:
        cond[0] = 0.7
        cond[1] = 0.4 / 0.7 ≈ 0.571
        cond[2] = 0.1 / 0.4 = 0.25
    """
    cond = []
    prev = 1.0
    for u in unconditional:
        if prev <= 0:
            cond.append(0.0)
        else:
            cond.append(u / prev)
        prev = u
    return cond


def loader_demo_shapes(
    target_layers: int = 61,
    mtp_layers: int = 1,
    hidden: int = 32,
    vocab: int = 128,
) -> dict:
    """Build a synthetic state_dict shaped like a DeepSeek-V3 + MTP HF
    checkpoint, run remap_checkpoint, and report counts.

    Used by demo.py to produce verbatim numerics about MTP weight loading.
    """
    state: Dict[str, torch.Tensor] = {}
    # Target layers
    for i in range(target_layers):
        state[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.zeros(hidden, hidden)
        state[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.zeros(hidden, hidden)
        state[f"model.layers.{i}.mlp.down_proj.weight"] = torch.zeros(hidden, hidden)
    # MTP layers (HF style: indices continue from target_layers)
    for j in range(mtp_layers):
        idx = target_layers + j
        # MTP-specific weights (paths 2 + 3 in rewrite_spec_layer_name)
        state[f"model.layers.{idx}.embed_tokens.weight"] = torch.zeros(vocab, hidden)  # shared
        state[f"model.layers.{idx}.enorm.weight"] = torch.zeros(hidden)
        state[f"model.layers.{idx}.hnorm.weight"] = torch.zeros(hidden)
        state[f"model.layers.{idx}.eh_proj.weight"] = torch.zeros(hidden, 2 * hidden)
        state[f"model.layers.{idx}.shared_head.head.weight"] = torch.zeros(vocab, hidden)
        # Transformer-block weights (path 1)
        state[f"model.layers.{idx}.self_attn.q_proj.weight"] = torch.zeros(hidden, hidden)
        state[f"model.layers.{idx}.self_attn.o_proj.weight"] = torch.zeros(hidden, hidden)
        state[f"model.layers.{idx}.mlp.down_proj.weight"] = torch.zeros(hidden, hidden)
    # Top-level
    state["model.embed_tokens.weight"] = torch.zeros(vocab, hidden)
    state["lm_head.weight"] = torch.zeros(vocab, hidden)

    target_sd, mtp_sd = remap_checkpoint(state, target_layers, mtp_layers)

    # Sample renames demonstrating the three paths
    sample_renames = []
    for j in range(mtp_layers):
        idx = target_layers + j
        for tail, expected_path in [
            ("self_attn.q_proj.weight", "path1"),     # gets .mtp_block. wrapped
            ("embed_tokens.weight", "path2"),         # promoted to top level
            ("eh_proj.weight", "path3"),              # unchanged
            ("shared_head.head.weight", "path3"),     # unchanged
        ]:
            old = f"model.layers.{idx}.{tail}"
            new = rewrite_spec_layer_name(idx, old)
            sample_renames.append({"path": expected_path, "old": old, "new": new})

    return {
        "input_total_keys": len(state),
        "target_keys": len(target_sd),
        "mtp_keys": len(mtp_sd),
        "sample_renames": sample_renames,
        "lm_head_present_target": "lm_head.weight" in target_sd,
    }


if __name__ == "__main__":
    print("=== Weight loading & sharing demo ===")
    print()
    print("Test 1: rewrite_spec_layer_name (matches deepseek_mtp.py:L458-L488)")
    cases = [
        (10, "model.layers.10.self_attn.q_proj.weight"),
        (10, "model.layers.10.embed_tokens.weight"),
        (10, "model.layers.10.enorm.weight"),
        (10, "model.layers.10.eh_proj.weight"),
        (10, "model.layers.10.shared_head.head.weight"),
        (10, "model.layers.10.mlp.gate_proj.weight"),
    ]
    for spec_layer, name in cases:
        rewritten = rewrite_spec_layer_name(spec_layer, name)
        if rewritten != name:
            print(f"  IN:  {name}")
            print(f"  OUT: {rewritten}")
        else:
            print(f"  UNCHANGED: {name}")
        print()
    print("Test 2: acceptance_length_to_rates")
    for length, n in [(3.4, 5), (2.0, 4), (4.7, 5), (1.0, 3)]:
        rates = acceptance_length_to_rates(length, n)
        print(f"  length={length}, n={n} → {rates}")
    print()
    print("Test 3: unconditional_to_conditional_rates")
    for uncond in [[0.7, 0.4, 0.1], [0.85, 0.7, 0.5, 0.3], [1.0, 1.0, 0.4, 0.0]]:
        cond = unconditional_to_conditional_rates(uncond)
        print(f"  uncond={uncond}  →  cond={[round(c, 4) for c in cond]}")
    print()
    print("Test 4: loader_demo_shapes (DeepSeek-V3 layout)")
    info = loader_demo_shapes(target_layers=61, mtp_layers=1, hidden=32, vocab=128)
    print(f"  total keys in checkpoint: {info['input_total_keys']}")
    print(f"  target keys after split:  {info['target_keys']}")
    print(f"  mtp keys after split:     {info['mtp_keys']}")
    for r in info["sample_renames"]:
        print(f"  [{r['path']}] {r['old']}")
        print(f"           → {r['new']}")
