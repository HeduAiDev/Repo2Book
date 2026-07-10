from complexity import indexer_ops, main_attention_ops, speedup_ratio


def test_main_attention_ops_dense_matches_causal_l_squared_over_two():
    # PAPER: §2.3 —— 稠密因果注意力对第 t 个 query 核算 t+1 个历史 token；
    # 求和 = L(L+1)/2。
    L = 8
    ops = main_attention_ops(L)
    assert ops == sum(range(1, L + 1))


def test_main_attention_ops_sparse_caps_at_topk():
    # PAPER: §2.3 —— O(Lk)：每个 query 至多核算 k 个 token（因果掩码下早期
    # token 数不足 k 时退化为稠密，这是自然边界不是特例）。
    L, k = 10, 3
    ops = main_attention_ops(L, topk=k)
    # 前 k-1 个 query 是稠密的（history < k），之后每个都恰好核算 k 个
    expected = sum(min(t + 1, k) for t in range(L))
    assert ops == expected
    assert ops < main_attention_ops(L)  # 严格小于稠密


def test_indexer_ops_scales_with_cost_ratio():
    # PAPER: §2.3 —— indexer 本身仍是 O(L^2)，只是常数（cost_ratio）远小于主 MLA。
    L = 16
    full = indexer_ops(L, cost_ratio=1.0)
    cheap = indexer_ops(L, cost_ratio=0.1)
    assert cheap == full * 0.1
    assert cheap < full


def test_speedup_ratio_approaches_l_over_2k_for_large_l():
    # PAPER: §2.3 —— L >> k 时稠密/稀疏核算量之比趋近 L/(2k)。
    L, k = 100_000, 100
    ratio = speedup_ratio(L, k)
    assert abs(ratio - L / (2 * k)) / (L / (2 * k)) < 0.01


def test_speedup_ratio_is_one_when_topk_covers_everything():
    L = 5
    ratio = speedup_ratio(L, topk=L)
    assert ratio == 1.0
