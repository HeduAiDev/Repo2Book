"""
Variance Analysis — Why sqrt(d_k)?

This script PROVES why the scale factor 1/√d_k is necessary.
It's not a hyperparameter someone tuned — it's a mathematical necessity.

THE PROBLEM:
    attention = softmax(Q @ K^T / scale) @ V

    Without scaling, Q@K^T produces values with variance = d_k.
    Large d_k (64, 128) → Q@K^T values are huge → softmax saturates →
    gradient ≈ 0 → model can't learn.

THE PROOF (simplified):
    Assume q_i, k_i are independent random variables with μ=0, σ²=1.
    Then:
        Var(q·k) = Var(Σ q_i·k_i)
                 = Σ Var(q_i·k_i)         [independence of (q_i·k_i) from (q_j·k_j)]
                 = d_k                    [since Var(q_i·k_i)=1 when μ=0,σ²=1]

    Therefore: Var(Q@K^T / √d_k) = Var(Q@K^T) / d_k = 1 ⬅ stable!

This script demonstrates this experimentally with concrete numbers.
"""

import math
import torch
import torch.nn.functional as F


def analyze_variance_empirically(
    d_k: int,
    num_samples: int = 1000,
    seed: int = 42,
):
    """
    Empirically verify the variance growth and effect of scaling.

    Args:
        d_k: dimension of query/key vectors
        num_samples: number of random Q,K pairs to generate
        seed: random seed for reproducibility

    Returns:
        dict with variance before/after scaling, softmax entropy, etc.
    """
    torch.manual_seed(seed)

    # Generate random Q,K vectors with μ=0, σ²=1
    Q = torch.randn(num_samples, d_k)  # [N, d_k]
    K = torch.randn(num_samples, d_k)

    # Compute dot products: each sample gives one scalar
    dot_products = (Q * K).sum(dim=-1)  # [N]

    # Statistics WITHOUT scaling
    var_unscaled = dot_products.var().item()
    std_unscaled = dot_products.std().item()

    # Statistics WITH scaling by √d_k
    scale = math.sqrt(d_k)
    scaled = dot_products / scale
    var_scaled = scaled.var().item()
    std_scaled = scaled.std().item()

    # What happens to softmax?
    # Compare softmax on [unscaled vs scaled] for a random attention row
    scores_unscaled = torch.randn(1, d_k) @ torch.randn(d_k, d_k)  # [1, d_k]
    scores_scaled = scores_unscaled / scale

    attn_unscaled = F.softmax(scores_unscaled, dim=-1)
    attn_scaled = F.softmax(scores_scaled, dim=-1)

    # Entropy: higher = more uniform (better for gradient flow)
    def entropy(p):
        return -(p * torch.log(p + 1e-10)).sum().item()

    return {
        "d_k": d_k,
        "expected_variance": d_k,
        "empirical_variance_unscaled": var_unscaled,
        "empirical_std_unscaled": std_unscaled,
        "scale_factor": scale,
        "empirical_variance_scaled": var_scaled,
        "empirical_std_scaled": std_scaled,
        "softmax_entropy_unscaled": entropy(attn_unscaled),
        "softmax_entropy_scaled": entropy(attn_scaled),
        "softmax_max_prob_unscaled": attn_unscaled.max().item(),
        "softmax_max_prob_scaled": attn_scaled.max().item(),
    }


def demonstrate_variance_problem():
    """
    Run the analysis across different d_k values and print a clear table.

    The key insight: as d_k grows (16 → 64 → 128 → 256), the unscaled
    dot-product variance grows proportionally. After softmax, the distribution
    collapses to nearly one-hot — meaning at most one token gets gradient,
    and learning becomes extremely slow.
    """
    print("=" * 76)
    print("VARIANCE ANALYSIS: Why 1/√d_k is NOT optional")
    print("=" * 76)
    print()
    print("The theory: Var(q·k) = d_k when q,k ~ N(0,1)")
    print("  → Without scaling, softmax input variance grows with d_k")
    print("  → Large variance → softmax collapses to one-hot → gradient ≈ 0")
    print("  → Scaling by 1/√d_k brings variance back to 1 → stable softmax")
    print()

    print(f"{'d_k':>5} | {'Var(unscaled)':>15} | {'Var(scaled)':>15} | "
          f"{'Entropy(unscaled)':>18} | {'Entropy(scaled)':>18} | {'Max prob(unscaled)':>18}")
    print("-" * 110)

    results = []
    for d_k in [4, 8, 16, 32, 64, 128, 256]:
        r = analyze_variance_empirically(d_k)
        results.append(r)
        print(
            f"{r['d_k']:5} | {r['empirical_variance_unscaled']:15.2f} | "
            f"{r['empirical_variance_scaled']:15.4f} | "
            f"{r['softmax_entropy_unscaled']:18.4f} | "
            f"{r['softmax_entropy_scaled']:18.4f} | "
            f"{r['softmax_max_prob_unscaled']:18.4f}"
        )

    print()
    print("OBSERVATION:")
    print(f"  d_k=4:   variance ~ {results[0]['empirical_variance_unscaled']:.1f}  "
          f"→ softmax entropy ~ {results[0]['softmax_entropy_unscaled']:.2f} "
          f"(still okay, distribution is spread out)")
    print(f"  d_k=256: variance ~ {results[-1]['empirical_variance_unscaled']:.0f} "
          f"→ softmax entropy ~ {results[-1]['softmax_entropy_unscaled']:.2f} "
          f"(COLLAPSED — nearly one-hot!)")
    print()
    print("CONCLUSION: The 1/√d_k factor is MANDATORY for stable training.")
    print("It's derived from basic statistics, not found by hyperparameter search.")

    return results


def manual_softmax_example():
    """
    A concrete, hand-calculable example showing the softmax collapse.
    This is the exact example from the chapter narrative.
    """
    print()
    print("=" * 76)
    print("CONCRETE EXAMPLE: Hand-calculable softmax collapse")
    print("=" * 76)
    print()

    # 3 tokens, d_k=4
    d_k = 4
    scale = math.sqrt(d_k)  # 2.0

    # Simulate: token 2's query vs. all 3 token keys
    q_2 = torch.tensor([0.5, 0.1, 0.3, 0.2])
    k_0 = torch.tensor([0.2, 0.8, 0.1, 0.4])
    k_1 = torch.tensor([0.7, 0.3, 0.5, 0.1])
    k_2 = torch.tensor([0.4, 0.2, 0.9, 0.3])

    # Compute dot products
    qk_0 = (q_2 * k_0).sum().item()
    qk_1 = (q_2 * k_1).sum().item()
    qk_2 = (q_2 * k_2).sum().item()

    print(f"Token 2's Query: {q_2.tolist()}")
    print(f"Token 0's Key:   {k_0.tolist()}  → dot = {qk_0:.4f}")
    print(f"Token 1's Key:   {k_1.tolist()}  → dot = {qk_1:.4f}")
    print(f"Token 2's Key:   {k_2.tolist()}  → dot = {qk_2:.4f}")
    print()

    # Without scaling
    raw = torch.tensor([qk_0, qk_1, qk_2])
    print(f"Without scaling: scores = {raw.tolist()}")
    print(f"  softmax = {F.softmax(raw, dim=-1).tolist()}")
    print(f"  Sum = {F.softmax(raw, dim=-1).sum():.4f}")

    # With scaling
    scaled = raw / scale
    print(f"\nWith scaling (÷√{d_k}={scale}): scores = {scaled.tolist()}")
    print(f"  softmax = {F.softmax(scaled, dim=-1).tolist()}")
    print(f"  Sum = {F.softmax(scaled, dim=-1).sum():.4f}")

    # Now simulate with LARGE d_k to show the problem
    print(f"\n--- Now imagine d_k=64 (typical for Llama) ---")
    # Random Q,K pair with d_k=64
    big_q = 0.1 * torch.randn(64)
    big_k = 0.1 * torch.randn(64)
    big_dot = (big_q * big_k).sum().item()
    big_scale = math.sqrt(64)
    print(f"  Random Q·K (d_k=64) = {big_dot:.2f}")
    print(f"  With scaling (÷8):   {big_dot/big_scale:.2f}")
    print(f"  Without scaling:     softmax would be dominated by the largest value")
    print(f"  With scaling:        softmax is well-behaved")


if __name__ == "__main__":
    demonstrate_variance_problem()
    manual_softmax_example()
