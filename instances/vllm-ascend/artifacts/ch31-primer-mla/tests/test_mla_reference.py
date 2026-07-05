"""ch31 —— 端到端旗舰测试：decode 吸收路径逐 token 增量计算，应与 prefill 物化路径
一次性算出的结果逐位置完全一致。这就是"权重吸收 + 解耦 RoPE 处处正确"的数值证明，
对应 vllm_ascend/attention/mla_v1.py 里 forward() 按 decode/prefill 两条路径分流、
但两条路径对同一 token 必须算出同一个答案这件事。
"""
import numpy as np

from mla_reference import MLAConfig, MLAReference, DecodeCache


def _toy_config():
    return MLAConfig(d=10, n_h=3, d_h=4, d_c=6, d_c_q=5, d_h_r=2)


def test_decode_step_matches_forward_full_position_by_position():
    cfg = _toy_config()
    model = MLAReference(cfg, seed=11)
    T = 5
    rng = np.random.default_rng(99)
    h_seq = rng.normal(size=(T, cfg.d))
    positions = list(range(T))

    u_seq_full = model.forward_full(h_seq, positions)

    cache = DecodeCache()
    for t in range(T):
        u_t, cache = model.decode_step(h_seq[t], positions[t], cache)
        np.testing.assert_allclose(u_t, u_seq_full[t], atol=1e-8,
                                    err_msg=f"decode/prefill 在位置 {t} 不一致")


def test_decode_cache_only_grows_by_latent_and_rope_dims():
    """缓存里从头到尾只有 (t, d_c) 和 (t, d_h_r) 两个张量——绝不出现 (t, n_h*d_h) 的物化 key/value。"""
    cfg = _toy_config()
    model = MLAReference(cfg, seed=1)
    rng = np.random.default_rng(2)
    cache = DecodeCache()
    for t in range(4):
        h_t = rng.normal(size=cfg.d)
        _, cache = model.decode_step(h_t, t, cache)
    assert cache.c_kv_history.shape == (4, cfg.d_c)
    assert cache.k_r_history.shape == (4, cfg.d_h_r)


def test_absorbed_weights_precomputed_once_are_reused_across_all_steps():
    """W~（权重吸收）是 __init__ 时算一次的静态量，decode_step 不会重新计算它——
    这正是与 decoupled_rope 里"M(delta) 每步都不同、无法复用"的对照。
    """
    cfg = _toy_config()
    model = MLAReference(cfg, seed=5)
    w_tildes_before = [w.copy() for w in model.w_tildes]
    rng = np.random.default_rng(6)
    cache = DecodeCache()
    for t in range(3):
        _, cache = model.decode_step(rng.normal(size=cfg.d), t, cache)
    for before, after in zip(w_tildes_before, model.w_tildes):
        np.testing.assert_allclose(before, after)
