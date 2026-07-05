"""ch31 —— KV cache 元素数对比（§2.1.4 Table 1）：小维度代数校验 + DeepSeek-V2 真实数字。"""
from kv_cache_table import (
    kv_cache_elements_mha,
    kv_cache_elements_gqa,
    kv_cache_elements_mqa,
    kv_cache_elements_mla,
    compare_kv_cache,
    deepseek_v2_numbers,
    toy_numbers,
)


def test_mqa_is_gqa_with_one_group():
    d_h, l = 8, 3
    assert kv_cache_elements_mqa(d_h, l) == kv_cache_elements_gqa(n_g=1, d_h=d_h, l=l)


def test_mha_is_gqa_with_n_h_groups():
    n_h, d_h, l = 4, 8, 3
    assert kv_cache_elements_mha(n_h, d_h, l) == kv_cache_elements_gqa(n_g=n_h, d_h=d_h, l=l)


def test_deepseek_v2_numbers_match_paper_table1():
    """论文 §3.1.2：n_h=128, d_h=128 -> 每 token 每层 2*128*128=32768 元素（Table 1 讨论段原文数字）。
    d_c=4*d_h=512, d_h^R=d_h/2=64 -> MLA=(512+64)*60=34560/60层；等效 GQA 组数=(512+64)/(2*128)=2.25
    （论文原文："its KV cache is equal to GQA with only 2.25 groups"）。
    """
    cmp = deepseek_v2_numbers()
    assert kv_cache_elements_mha(128, 128, 1) == 32768
    assert cmp.mla_equivalent_gqa_groups == 2.25
    assert cmp.mla == (512 + 64) * 60


def test_mla_cache_smaller_than_mha_for_deepseek_v2():
    cmp = deepseek_v2_numbers()
    assert cmp.mla < cmp.mha
    assert cmp.mla_compression_ratio_vs_mha > 1.0


def test_toy_numbers_are_self_consistent_and_runnable():
    cmp = toy_numbers()
    assert cmp.mla == compare_kv_cache(4, 8, 2, 12, 4, n_g=1).mla
    assert cmp.mla_equivalent_gqa_groups == (12 + 4) / (2 * 8)
