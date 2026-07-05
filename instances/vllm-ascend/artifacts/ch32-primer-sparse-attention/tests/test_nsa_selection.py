import numpy as np

from nsa_selection import (
    block_importance_from_compression,
    compression_attention_scores,
    gather_selected_blocks,
    gqa_group_importance,
    topn_block_selection,
)


def test_compression_attention_scores_sum_to_one():
    rng = np.random.default_rng(4)
    q = rng.normal(size=(6,))
    k_cmp = rng.normal(size=(5, 6))
    p_cmp = compression_attention_scores(q, k_cmp)
    assert np.isclose(p_cmp.sum(), 1.0)
    assert np.all(p_cmp >= 0.0)


def test_block_importance_identity_when_blocking_schemes_match():
    # PAPER Eq.9: l'=l=d -> p_slc = p_cmp directly (paper's simplest case)
    p_cmp = np.array([0.1, 0.4, 0.2, 0.3])
    p_slc = block_importance_from_compression(p_cmp, l=16, d=16, l_prime=16)
    assert np.allclose(p_slc, p_cmp)
    # must be a copy, not an alias
    p_slc[0] = 999.0
    assert p_cmp[0] == 0.1


def test_block_importance_manual_mapping_when_schemes_differ():
    # l=16, d=8 (2 compression blocks overlap contribute to l'=16 selection block via Eq.9),
    # l'=16 -> l'/d=2, l/d=2. p_t^slc[j] = sum_{m=0}^{1} sum_{n=0}^{1} p_cmp[2j - m - n]
    p_cmp = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    p_slc = block_importance_from_compression(p_cmp, l=16, d=8, l_prime=16)
    # manual: j=0 -> indices {2*0-m-n for m,n in 0,1} = {0,-1,-1,-2} -> valid idx: 0 (twice? no set)
    expected = np.zeros(len(p_slc))
    for j in range(len(expected)):
        acc = 0.0
        for m in range(2):
            for n in range(2):
                idx = 2 * j - m - n
                if 0 <= idx < len(p_cmp):
                    acc += p_cmp[idx]
        expected[j] = acc
    assert np.allclose(p_slc, expected)


def test_gqa_group_importance_sums_across_heads():
    p_slc_per_head = np.array([[0.1, 0.2, 0.3], [0.4, 0.1, 0.0], [0.05, 0.05, 0.5]])
    result = gqa_group_importance(p_slc_per_head)
    assert np.allclose(result, p_slc_per_head.sum(axis=0))


def test_topn_block_selection_returns_highest_ranked_blocks():
    p_slc_prime = np.array([0.1, 0.5, 0.05, 0.3, 0.05])
    top2 = topn_block_selection(p_slc_prime, n=2)
    assert set(top2.tolist()) == {1, 3}  # 0.5 and 0.3 are the two highest scores


def test_topn_block_selection_clamps_n_to_available_blocks():
    p_slc_prime = np.array([0.5, 0.5])
    result = topn_block_selection(p_slc_prime, n=10)
    assert len(result) == 2


def test_gather_selected_blocks_concatenates_correct_slices():
    k_seq = np.arange(24).reshape(6, 4)  # 6 tokens, l_prime=2 -> 3 blocks of 2 tokens each
    selected = np.array([2, 0])  # unordered on purpose
    gathered = gather_selected_blocks(k_seq, selected, l_prime=2)
    expected = np.concatenate([k_seq[0:2], k_seq[4:6]], axis=0)  # sorted block order: 0 then 2
    assert np.array_equal(gathered, expected)


def test_gather_selected_blocks_empty_selection():
    k_seq = np.arange(24).reshape(6, 4)
    gathered = gather_selected_blocks(k_seq, np.array([], dtype=int), l_prime=2)
    assert gathered.shape == (0, 4)
