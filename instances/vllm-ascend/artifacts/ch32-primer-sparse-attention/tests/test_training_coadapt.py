import numpy as np

from training_coadapt import (
    aggregate_main_attention,
    dense_warmup_kl,
    simulate_indexer_logits,
    sparse_stage_kl,
    topk_mass_recall,
)


def test_aggregate_main_attention_l1_normalizes_head_sum():
    head_scores = np.array([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]])  # (H=2, t=3), each row sums to 1
    p = aggregate_main_attention(head_scores)
    assert np.isclose(p.sum(), 1.0)
    assert np.allclose(p, head_scores.sum(axis=0) / head_scores.sum())


def test_dense_warmup_kl_is_zero_for_perfectly_aligned_indexer():
    # PAPER Eq.3: if Softmax(I_{t,:}) == p_{t,:} exactly, D_KL == 0 (perfect alignment)
    p = np.array([0.1, 0.6, 0.2, 0.1])
    indexer_scores = np.log(p + 1e-12)  # softmax(log p) == p (up to normalization)
    kl = dense_warmup_kl(p, indexer_scores)
    assert kl < 1e-6


def test_dense_warmup_kl_is_positive_for_misaligned_indexer():
    p = np.array([0.7, 0.1, 0.1, 0.1])
    indexer_scores = np.array([0.0, 5.0, 0.0, 0.0])  # indexer thinks position 1 dominates -- wrong
    kl = dense_warmup_kl(p, indexer_scores)
    assert kl > 0.5


def test_sparse_stage_kl_only_considers_selected_set():
    p = np.array([0.4, 0.3, 0.2, 0.1])
    indexer_scores = np.log(p + 1e-12)
    topk_indices = np.array([0, 1])  # restrict to a subset that is still internally aligned
    kl = sparse_stage_kl(p, indexer_scores, topk_indices)
    assert kl < 1e-6


def test_topk_mass_recall_is_maximal_when_indexer_matches_true_distribution():
    p = np.array([0.5, 0.3, 0.1, 0.1])
    indexer_scores = p.copy()  # perfectly aligned: same ranking as p
    recall = topk_mass_recall(p, indexer_scores, k=2)
    assert np.isclose(recall, 0.5 + 0.3)  # captures exactly the top-2 true attention mass


def test_topk_mass_recall_is_worse_for_misaligned_indexer():
    p = np.array([0.5, 0.3, 0.1, 0.1])
    aligned_recall = topk_mass_recall(p, p.copy(), k=2)
    misaligned_scores = np.array([0.1, 0.1, 0.5, 0.3])  # ranks the two *lowest*-mass tokens highest
    misaligned_recall = topk_mass_recall(p, misaligned_scores, k=2)
    assert misaligned_recall < aligned_recall
    assert np.isclose(misaligned_recall, 0.1 + 0.1)


def test_alignment_alpha_zero_recovers_true_distribution_ranking():
    # alpha=0 -> indexer logits are a monotone function of p (log), so top-k(indexer) == top-k(p)
    rng = np.random.default_rng(9)
    p = np.array([0.05, 0.35, 0.4, 0.1, 0.1])
    logits = simulate_indexer_logits(p, alpha=0.0, rng=rng)
    top2_true = set(np.argsort(-p)[:2].tolist())
    top2_indexer = set(np.argsort(-logits)[:2].tolist())
    assert top2_true == top2_indexer


def test_lower_kl_correlates_with_higher_topk_mass_recall_across_alignment_sweep():
    # PAPER §2.1.1's whole point: Eq.3/4's KL loss is the mechanism that makes top-k not lose
    # attention mass. Sweep the alignment knob and check the monotonic (in expectation) relationship.
    rng = np.random.default_rng(10)
    p = rng.dirichlet(np.ones(64))  # a peaky-ish true attention distribution over 64 tokens

    alphas = [0.0, 0.3, 0.6, 1.0]
    n_trials = 50
    avg_kl = []
    avg_recall = []
    for alpha in alphas:
        kls, recalls = [], []
        for trial in range(n_trials):
            trial_rng = np.random.default_rng(1000 * trial + int(alpha * 10))
            logits = simulate_indexer_logits(p, alpha, trial_rng)
            kls.append(dense_warmup_kl(p, logits))
            recalls.append(topk_mass_recall(p, logits, k=8))
        avg_kl.append(np.mean(kls))
        avg_recall.append(np.mean(recalls))

    # As alpha increases (more misaligned), average KL should increase and average top-k mass
    # recall should decrease -- this is the quantitative form of "training co-adaptation is why
    # top-k sparsity doesn't hurt". With enough trials the trend is robust end-to-end even if a
    # couple of adjacent midpoints are noisy.
    assert avg_kl[0] < avg_kl[-1]
    assert avg_recall[0] > avg_recall[-1]
    assert avg_kl[0] == min(avg_kl)
    assert avg_recall[0] == max(avg_recall)
