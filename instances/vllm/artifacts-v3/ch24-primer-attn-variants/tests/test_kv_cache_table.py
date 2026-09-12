"""arXiv:2405.04434 §2.1.4 Table 1 + §3.1.2 超参 + arXiv:2305.13245 §2.2
(factor-H 句)—— KV cache 元素总账:MHA 2n_h·d_h·l / MQA 2d_h·l /
GQA 2n_g·d_h·l / MLA (d_c+d_h^R)·l(按元素数、不分精度);2.25 组换算与
576/32768≈1.75% 的算术。测试先于实现书写(TDD)。"""
import numpy as np

from kv_cache_table import (
    deepseek_v2_mla_hyperparams,
    equivalent_gqa_groups,
    gqa_kv_cache_per_token,
    mha_kv_cache_per_token,
    mha_to_mqa_cache_reduction_factor,
    mla_kv_cache_per_token,
    mqa_kv_cache_per_token,
    table1_kv_cache_per_token,
)


def test_table1_four_rows_with_deepseek_hyperparams():
    # §3.1.2 超参 n_h=128、d_h=128、d_c=512、d_h^R=64 代 Table 1(l=1):
    # MHA 2·128·128=32768;MQA 2·128=256;GQA-8 2·8·128=2048;MLA 512+64=576。
    t = table1_kv_cache_per_token(
        num_heads=128, head_dim=128, num_groups=8, d_c=512, d_h_R=64
    )
    assert t["MHA"] == 2 * 128 * 128 == 32768
    assert t["MQA"] == 2 * 128 == 256
    assert t["GQA"] == 2 * 8 * 128 == 2048
    assert t["MLA"] == 512 + 64 == 576


def test_table1_scales_with_layers_l():
    # Table 1 各行均含层数因子 l(§2.1.1 "2n_h·d_h·l elements for each token"
    # /§2.1.3 "(d_c+d_h^R)l elements")。
    t60 = table1_kv_cache_per_token(
        num_heads=128, head_dim=128, num_groups=8, d_c=512, d_h_R=64,
        num_layers=60,
    )
    assert t60["MHA"] == 32768 * 60
    assert t60["MLA"] == 576 * 60
    assert mha_kv_cache_per_token(32, 128, num_layers=32) == 2 * 32 * 128 * 32


def test_spectrum_endpoints_close_in_accounting():
    # §2.2 谱系端点:GQA-H 的账 == MHA 的账;GQA-1 的账 == MQA 的账。
    n_h, d_h = 32, 128
    assert gqa_kv_cache_per_token(n_h, d_h) == mha_kv_cache_per_token(n_h, d_h)
    assert gqa_kv_cache_per_token(1, d_h) == mqa_kv_cache_per_token(d_h)


def test_mha_to_mqa_reduces_by_factor_H():
    # §2.2 原句:"Going from MHA to MQA reduces H key and value heads to a
    # single key and value head, reducing the size of the key-value cache and
    # therefore amount of data that needs to be loaded by a factor of H."
    for H in (8, 32, 128):
        assert mha_kv_cache_per_token(H, 128) == H * mqa_kv_cache_per_token(128)
        assert mha_to_mqa_cache_reduction_factor(H) == H


def test_mla_equivalent_to_gqa_with_2_25_groups():
    # §2.1.4 caption:"d_c is set to 4d_h and d_h^R is set to d_h/2. So, its
    # KV cache is equal to GQA with only 2.25 groups"
    # —— (d_c+d_h^R)/(2·d_h)=(4d_h+d_h/2)/(2d_h)=2.25。
    assert equivalent_gqa_groups(512, 64, 128) == 2.25
    for d_h in (64, 128, 256):   # 比例关系 d_c=4d_h、d_h^R=d_h/2 下恒为 2.25
        assert equivalent_gqa_groups(4 * d_h, d_h // 2, d_h) == 2.25


def test_mla_is_about_1_75_percent_of_own_mha_config():
    # 研究注记算术(非论文原句):按 §3.1.2 超参,MLA 每元素账 576 相对同配置
    # MHA 32768 ≈ 1.75%(注意:abstract 的 93.3% 是相对 DeepSeek 67B 的口径,
    # App D.2 的 14%/4% 是 MoE 整体实测口径——三个数不可混用)。
    ratio = mla_kv_cache_per_token(512, 64) / mha_kv_cache_per_token(128, 128)
    np.testing.assert_allclose(ratio, 576 / 32768)
    assert abs(ratio - 0.0175) < 0.0005   # ≈1.75%


def test_deepseek_v2_hyperparams_and_576_width():
    # §3.1.2:n_h=128、d_h=128、d_c=512、d_c'=1536、d_h^R=64(V3 §4.2 同值,
    # 576 口径跨代稳定);缓存宽度 = d_c+d_h^R = 576。
    hp = deepseek_v2_mla_hyperparams()
    assert hp["n_h"] == 128
    assert hp["d_h"] == 128
    assert hp["d_c"] == 512
    assert hp["d_c_prime"] == 1536
    assert hp["d_h_R"] == 64
    assert hp["d_c"] + hp["d_h_R"] == 576


def test_llama2_7b_kv_cache_about_half_mb_per_token():
    # dossier m01 worked example(算例非论文原句,引自 deepread/memory-kv 卡):
    # Llama-2-7B FP16 每 token KV ≈ 2×32 层×32 头×128 维×2B = 524288B = 0.5MB
    # ——Table 1 的 MHA 行 2n_h·d_h·l 乘上精度(2B/元素)就是引擎账本口径。
    per_token_bytes = mha_kv_cache_per_token(32, 128, num_layers=32) * 2
    assert per_token_bytes == 2 * 32 * 32 * 128 * 2 == 512 * 1024   # 0.5MB
