"""
Multi-Token Prediction (MTP) — Our Reimplementation.

REFERENCE sources:
    DeepSeek V3 MTP:  vllm/model_executor/models/deepseek_v2.py
    Spec decode:      vllm/v1/core/sched/async_scheduler.py
    MTP acceptance:   inferred from DeepSeek technical report

Concept:
    Standard decode:   1 forward → 1 token
    MTP decode:        1 forward → M tokens (1 verified + M-1 draft)
    Speedup:           M × acceptance_rate (typically 1.5-2.5×)

    MTP adds extra heads to the final transformer layer that predict
    token_{t+1}, token_{t+2}, ..., token_{t+M-1} in one forward pass.
    These are draft tokens — they must be verified before acceptance.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


# ═══════════════════════════════════════════════════════════════════════════
# MTP Acceptance Rate Analysis
# ═══════════════════════════════════════════════════════════════════════════

def mtp_speedup_analysis(
    num_steps: int = 3,           # MTP predicts 3 future tokens
    acceptance_rates: Tuple[float, ...] = (0.85, 0.75, 0.65),
) -> dict:
    """
    Analyze MTP speedup vs. standard decode.

    Theory:
        Standard decode generates 1 token per forward pass.
        MTP generates M tokens per forward pass (1 verified + M-1 draft).
        Acceptance rate drops for later positions (harder to predict further ahead).

        Expected tokens per step = 1 + sum(acceptance_rate_i) for i in 1..M-1
        Speedup = E[tokens_per_step] / 1

    Example with M=3, rates=(0.85, 0.75, 0.65):
        E[tokens] = 1 + 0.85 + 0.75 = 2.6 tokens per step
        Speedup = 2.6× (ideal would be 3.0× with perfect prediction)
    """
    expected_tokens = 1 + sum(acceptance_rates[:num_steps-1])
    speedup = expected_tokens
    ideal_speedup = num_steps

    return {
        "num_mtp_steps": num_steps,
        "acceptance_rates": acceptance_rates[:num_steps-1],
        "expected_tokens_per_forward": expected_tokens,
        "speedup": speedup,
        "ideal_speedup": ideal_speedup,
        "efficiency": speedup / ideal_speedup,
        "note": (
            f"MTP achieves {speedup:.1f}× speedup ({(speedup/ideal_speedup)*100:.0f}% of ideal {ideal_speedup}×). "
            "The gap vs ideal is because later draft tokens have lower acceptance rates — "
            "the model is less certain about what comes 3 tokens ahead."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MTP Head Architecture
# REFERENCE: DeepSeek V3 architecture (MTP block after final transformer layer)
# ═══════════════════════════════════════════════════════════════════════════

class MTPHead(nn.Module):
    """
    One Multi-Token Prediction head.

    Takes the final hidden state and predicts the next token.
    Multiple MTP heads are stacked: head k predicts token_{t+k}.

    Architecture (from DeepSeek V3):
        MTP_k(x) = LayerNorm(x + Embedding(token_{t+k-1}))
                    → TransformerBlock (shared or independent trunk)
                    → LM Head (shared output projection)
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Lightweight projection: hidden → vocab
        # In practice, this shares the final lm_head weights
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, d_model] — final layer output

        Returns:
            logits: [batch, vocab_size] — predicted next-token distribution
        """
        return self.proj(hidden_states)


class MultiTokenPredictor(nn.Module):
    """
    Multiple MTP heads stacked: predict token_{t+1}, token_{t+2}, ...

    REFERENCE: DeepSeek V3 MTP module

    In DeepSeek V3:
        - 1 extra MTP head (predicts only token_{t+1})
        - Shared embedding output layer (not separate vocab projections)
        - MTP head receives: current hidden state + embedding of previous draft token

    The key design trade-off:
        More MTP steps → higher potential speedup
        More MTP steps → lower per-step acceptance rate (diminishing returns)
        Most models use 1-3 MTP heads (DeepSeek V3: 1, V4: 3)
    """

    def __init__(self, d_model: int, vocab_size: int, num_heads: int = 2):
        super().__init__()
        self.heads = nn.ModuleList([
            MTPHead(d_model, vocab_size) for _ in range(num_heads)
        ])

    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns list of logits: [token_{t+1}_logits, token_{t+2}_logits, ...]
        """
        return [head(hidden_states) for head in self.heads]


# ═══════════════════════════════════════════════════════════════════════════
# Acceptance Verification
# ═══════════════════════════════════════════════════════════════════════════

def verify_mtp_tokens(
    draft_tokens: List[int],    # [tok_{t+1}, tok_{t+2}, ...] from MTP
    verified_tokens: List[int], # [tok_{t+1}, tok_{t+2}, ...] from oracle model
) -> Tuple[List[int], int]:
    """
    Verify MTP draft tokens against oracle output.

    REFERENCE: This is the verification step in speculative decoding,
               adapted for MTP where the "oracle" is the same model
               running in standard decode mode.

    Algorithm:
        1. Check draft[0] == verified[0]
           - If match: accept, move to draft[1]
           - If mismatch: reject ALL subsequent drafts, accept verified[0] only
        2. Continue until first mismatch or all drafts verified

    Returns:
        accepted_tokens: verified tokens accepted this step
        num_accepted: how many were accepted (for acceptance rate tracking)
    """
    accepted = []
    for i, (draft, verified) in enumerate(zip(draft_tokens, verified_tokens)):
        if draft == verified:
            accepted.append(verified)
        else:
            # First mismatch → accept verified version of this token
            # and STOP (all subsequent drafts are now stale)
            accepted.append(verified)
            break
    return accepted, len(accepted)


def demonstrate_mtp():
    """Show MTP speedup analysis with realistic numbers."""
    print("MTP Speedup Analysis")
    print("=" * 60)
    print()

    # DeepSeek V3: 1 MTP head
    r1 = mtp_speedup_analysis(num_steps=2, acceptance_rates=(0.85,))
    print(f"DeepSeek V3 (1 MTP head):")
    print(f"  Expected tokens/step: {r1['expected_tokens_per_forward']}")
    print(f"  Speedup: {r1['speedup']:.1f}×  (ideal: {r1['ideal_speedup']}×)")
    print(f"  Efficiency: {r1['efficiency']:.1%}")
    print()

    # DeepSeek V4: 3 MTP heads
    r3 = mtp_speedup_analysis(num_steps=4, acceptance_rates=(0.85, 0.72, 0.58))
    print(f"DeepSeek V4 (3 MTP heads):")
    print(f"  Expected tokens/step: {r3['expected_tokens_per_forward']}")
    print(f"  Speedup: {r3['speedup']:.1f}×  (ideal: {r3['ideal_speedup']}×)")
    print(f"  Efficiency: {r3['efficiency']:.1%}")
    print()

    # Acceptance rate interpretation
    print("Acceptance Rate Interpretation:")
    print("  Step +1 (1 token ahead):  ~85% — model is pretty confident")
    print("  Step +2 (2 tokens ahead): ~72% — less certain")
    print("  Step +3 (3 tokens ahead): ~58% — diminishing returns")
    print("  This is why most models use 1-3 MTP heads, not 10.")
    print("  The marginal gain from each additional head decreases exponentially.")

    return r3


if __name__ == "__main__":
    demonstrate_mtp()
