"""两本账（FLOPs 账 / KV 体积账）算术测试 —— 论文锚：
arXiv:2606.19348 摘要（27% FLOPs / 10% KV，分母 = V3.2）与 §2.3.4
（~2% KV，分母 = BF16 GQA8 head_dim=128 基线）。

注意：本文件只测**算术恒等式**与账本口径；具体结论数由 explainer 跑脚本复算
（dossier m01 明记「只作口径示范，结论数须复算」）。
"""
import numpy as np

from ledger import (
    V4_FLASH_COMPRESS_RATIOS,
    attention_visible_entries,
    cache_bytes_per_token,
    classify_layers,
    compressed_entry_bytes,
    gqa8_baseline_bytes_per_token,
    indexer_cache_entry_bytes,
    kv_byte_account,
    v32_bytes_per_token,
)


def test_entry_byte_accounting():
    # fp8_ds_mla：448B NoPE + 128B RoPE + 8B fp8 scale = 584B/条（v1/kv_cache_interface.py）
    assert compressed_entry_bytes() == 448 + 128 + 8 == 584
    # IndexCache：128B fp8 小头 + 4B fp32 scale = 132B/条
    assert indexer_cache_entry_bytes() == 128 + 4 == 132
    assert v32_bytes_per_token() == 656
    # BF16 GQA8、head_dim=128 基线：8 头 × 128 维 × 2 件(K,V) × 2B = 4096B/token
    assert gqa8_baseline_bytes_per_token() == 4096


def test_v4_flash_layout_reads_as_two_zeros_then_alternating_then_mtp_slot():
    r = V4_FLASH_COMPRESS_RATIOS
    assert len(r) == 44  # 43 层主干 + 1 个 MTP 槽
    assert r[0] == 0 and r[1] == 0
    assert r[-1] == 0
    assert r[2:43] == [4 if i % 2 == 0 else 128 for i in range(41)]

    counts = classify_layers(r[:43])
    assert counts == {"swaonly": 2, "c4a": 21, "c128a": 20}


def test_cache_bytes_per_token_hand_computed():
    # 单层账：CSA 层 = (584 + 132)/4；HCA 层 = 584/128；SWA 层 = 0（压缩账上不花）
    assert np.isclose(cache_bytes_per_token(4, count_indexer=True), 179.0)
    assert np.isclose(cache_bytes_per_token(4, count_indexer=False), 146.0)
    assert np.isclose(cache_bytes_per_token(128), 4.5625)
    assert cache_bytes_per_token(1) == 0.0


def test_kv_byte_account_weighted_over_layers():
    acct = kv_byte_account(V4_FLASH_COMPRESS_RATIOS[:43])
    # 逐层型账目可手核
    assert np.isclose(acct["per_type"]["c4a"], 179.0)
    assert np.isclose(acct["per_type"]["c128a"], 4.5625)
    # 43 层加权平均 = (21·179 + 20·4.5625) / 43
    want = (21 * 179.0 + 20 * 4.5625) / 43
    assert np.isclose(acct["bytes_per_token"], want)
    assert np.isclose(acct["bytes_per_token"], 89.5, atol=0.05)

    # 两个分母都在账上：对 GQA8 基线在 2% 量级（与论文口径同量级）、对 V3.2 更大
    assert 0.01 < acct["ratio_vs_gqa8"] < 0.03
    assert acct["ratio_vs_v32"] > acct["ratio_vs_gqa8"]


def test_kv_byte_account_switches():
    ratios = V4_FLASH_COMPRESS_RATIOS[:43]
    without = kv_byte_account(ratios)
    with_swa = kv_byte_account(ratios, count_swa_window=True, context_len=1_000_000)
    # 滑窗是**加账**：计入它账更大（不是省账手段）
    assert with_swa["total_bytes_per_token"] > without["total_bytes_per_token"]
    # 常数窗：总量与上下文长度无关（n_win × 每层 × 43 层）
    assert with_swa["swa_window_total_bytes"] == 43 * 128 * 576
    # 摊薄项随上下文变长而变小（同样 43 层窗口，1M 的每 token 摊销是 100k 的 1/10）
    shorter = kv_byte_account(ratios, count_swa_window=True, context_len=100_000)
    assert np.isclose(shorter["swa_bytes_per_token"], 10 * with_swa["swa_bytes_per_token"])
    assert np.isclose(with_swa["swa_bytes_per_token"], 43 * 128 * 576 / 1_000_000)
    # 账目可加：总量 = 压缩账 + 滑窗摊销
    assert np.isclose(
        with_swa["total_bytes_per_token"],
        without["bytes_per_token"] + with_swa["swa_bytes_per_token"],
    )


def test_attention_visible_entries_per_query():
    # CSA：候选 25 万条，每 query 只挑 512 条（index_topk）
    assert attention_visible_entries(n_tokens=1_048_576, compress_ratio=4, index_topk=512) == 512
    # HCA：压到 1/128 之后全看不挑 ⇒ 每 query 8192 条
    assert attention_visible_entries(n_tokens=1_048_576, compress_ratio=128) == 8192
    # 候选不足 topk 时（短上下文）实看 = 候选数
    assert attention_visible_entries(n_tokens=1024, compress_ratio=128) == 8
    assert attention_visible_entries(n_tokens=2048, compress_ratio=4, index_topk=512) == 512
