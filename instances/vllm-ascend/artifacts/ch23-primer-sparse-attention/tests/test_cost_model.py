import numpy as np

from cost_model import (
    decode_step_indexer_cost,
    decode_step_main_cost_dense,
    decode_step_main_cost_sparse,
    paper_training_numbers,
    prefill_total_indexer_cost,
    prefill_total_main_cost_dense,
    prefill_total_main_cost_sparse,
    speedup_accounting,
    vllm_ascend_deployment_numbers,
)


def test_decode_step_costs_match_formulas():
    assert decode_step_main_cost_dense(context_len=1000, per_kv_dim=64) == 64000
    assert decode_step_main_cost_sparse(k=10, per_kv_dim=64) == 640
    assert decode_step_indexer_cost(context_len=1000, indexer_heads=4, indexer_dim=8) == 1000 * 32


def test_main_only_speedup_equals_context_len_over_k_exactly():
    # PAPER §2.3: single decode step -- dense scans context_len keys, sparse scans only k.
    # per_kv_dim cancels out of the ratio exactly.
    acc = speedup_accounting(context_len=131072, k=512, per_kv_dim=73728, indexer_heads=64, indexer_dim=128)
    assert np.isclose(acc.main_only_speedup, 131072 / 512)
    assert np.isclose(acc.main_only_speedup, 256.0)


def test_main_only_speedup_for_k_2048_matches_paper_training_value():
    acc = speedup_accounting(context_len=131072, k=2048, per_kv_dim=73728, indexer_heads=64, indexer_dim=128)
    assert np.isclose(acc.main_only_speedup, 64.0)


def test_end_to_end_speedup_is_more_conservative_than_main_only():
    # Because the indexer itself costs something (even if cheap), including it must never make
    # the speedup estimate look better than the main-attention-only estimate.
    acc = speedup_accounting(context_len=131072, k=512, per_kv_dim=73728, indexer_heads=64, indexer_dim=128)
    assert acc.end_to_end_speedup < acc.main_only_speedup
    assert acc.end_to_end_speedup > 1.0  # still a net win, just not as dramatic


def test_vllm_ascend_deployment_numbers_uses_default_k_512():
    acc = vllm_ascend_deployment_numbers()
    assert np.isclose(acc.main_only_speedup, 256.0)


def test_paper_training_numbers_uses_k_2048():
    acc = paper_training_numbers()
    assert np.isclose(acc.main_only_speedup, 64.0)


def test_paper_training_k_gives_smaller_speedup_than_deployment_k():
    # k=2048 (paper's training value) selects more tokens than k=512 (deployment default),
    # so it must yield a smaller speedup -- more conservative, not "free".
    deploy = vllm_ascend_deployment_numbers()
    paper = paper_training_numbers()
    assert paper.main_only_speedup < deploy.main_only_speedup
    assert paper.end_to_end_speedup < deploy.end_to_end_speedup


def test_prefill_total_costs_match_triangular_accumulation():
    # Summing the per-decode-step dense cost over all t=1..L query positions gives L(L+1)/2 * d,
    # which is where the paper's stated O(L^2) indexer complexity comes from.
    seq_len, per_kv_dim = 100, 8
    total = prefill_total_main_cost_dense(seq_len, per_kv_dim)
    manual = sum(decode_step_main_cost_dense(t, per_kv_dim) for t in range(1, seq_len + 1))
    assert total == manual

    sparse_total = prefill_total_main_cost_sparse(seq_len, k=5, per_kv_dim=per_kv_dim)
    assert sparse_total == seq_len * 5 * per_kv_dim

    idx_total = prefill_total_indexer_cost(seq_len, indexer_heads=4, indexer_dim=8)
    manual_idx = sum(decode_step_indexer_cost(t, 4, 8) for t in range(1, seq_len + 1))
    assert idx_total == manual_idx
