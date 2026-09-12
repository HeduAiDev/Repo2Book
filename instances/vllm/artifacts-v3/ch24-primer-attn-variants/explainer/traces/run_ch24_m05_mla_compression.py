"""ch24-m05 驱动脚本 —— MLA 低秩联合压缩:式(9)-(11)、只缓存 c^KV。

跑法(host, 纯 CPU numpy): python run_ch24_m05_mla_compression.py
输出: ch24_m05_mla_compression.json(与本脚本同目录)

素材对应 dossier 机制 ch24-m05,论文 arXiv:2405.04434 §2.1.2 Eq.(9)-(11)
(+App C Eq.(38)/(43)/(44)、式(41) 蓝框;V3 重述 V3-1/V3-2/V3-5)。
玩具维度与 tests/test_mla.py 的 _toy_dims 同构:d=8、n_h=2、d_h=4、d_c=6、d_h_R=2。
与 GQA 的本质差异用前向实测数据说话:MLA 两个头的 k^C 是同一潜向量的不同线性
投影(逐位不同),GQA 同组头的 K 逐位相同(复制现成向量)。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
from kv_cache_table import (  # noqa: E402
    equivalent_gqa_groups,
    mha_kv_cache_per_token,
    mla_kv_cache_per_token,
)
from mha_gqa import attention_forward  # noqa: E402
from mla import make_mla_weights, mla_naive_forward  # noqa: E402


def r(v, nd=3):
    return round(float(v), nd)


def main():
    out = {}

    dims = dict(d=8, n_h=2, d_h=4, d_c=6, d_h_R=2)
    w = make_mla_weights(**dims, q_lora_rank=5, seed=29)
    rng = np.random.default_rng(30)
    T = 4
    h = rng.standard_normal((T, dims["d"]))
    positions = np.arange(T)
    trace = {}
    u, cache = mla_naive_forward(h, positions, w, trace=trace)
    c_KV, k_R = cache

    out["params"] = {
        **dims, "q_lora_rank": 5, "T": T, "seed_w": 29, "seed_h": 30,
        "eqs": [9, 10, 11, 38, 41, 43, 44],
        "eq_note": "Eq.(9) c^KV=W^DKV h;Eq.(10)/(11) k^C/v^C=W^UK/W^UV c^KV;蓝框 Eq.(41)",
    }

    # ---- 投影形状账(玩具 ↔ 真值两侧) ----
    out["shape_account"] = {
        "toy": {
            "h": [T, 8],
            "W_DKV_out_by_in": list(w.W_DKV.shape),        # (6,8) —— d_c × d
            "c_KV": list(c_KV.shape),                      # (4,6)
            "W_UK_out_by_in": list(w.W_UK.shape),          # (8,6) —— n_h·d_h × d_c
            "W_UV_out_by_in": list(w.W_UV.shape),          # (8,6)
            "k_C_per_head": list(trace["k_C"].shape),      # (4,2,4)
            "v_C_per_head": list(trace["v_C"].shape),      # (4,2,4)
            "dc_much_less_than_dh_nh": "6 ≪ 2*4=8(玩具)/512 ≪ 16384(DSV3)",
        },
        "dsv3": {
            "d": 5120, "n_h": 128, "d_h": 128, "d_c": 512, "d_h_R": 64,
            "W_DKV": [512, 5120], "W_UK": [16384, 512], "W_UV": [16384, 512],
            "kv_b_proj_output_width": 128 * (128 + 128),   # 32768 = [W^UK;W^UV] 按头拼接
            "note": "上投影把潜向量恢复回 16384 维的逐头 K/V 空间——但那是参数侧乘法,不是缓存",
        },
    }

    # ---- 同一潜向量、不同头的投影(MLA 与 GQA 共享方式的本质差异,实测) ----
    head0_k = trace["k_C"][0, 0]     # token0 head0 的 k^C
    head1_k = trace["k_C"][0, 1]     # token0 head1 的 k^C
    mla_cross = float(np.abs(head0_k - head1_k).max())
    # GQA 侧:同组两个头吃同一份 K(复制)
    rng_g = np.random.default_rng(31)
    h_g = rng_g.standard_normal((2, 6))
    g = np.random.default_rng(32)
    WQ_g = g.standard_normal((4 * 2, 6)); WK_g = g.standard_normal((2 * 2, 6))
    WV_g = g.standard_normal((2 * 2, 6)); WO_g = g.standard_normal((4 * 2, 4 * 2))
    tr_g = {}
    attention_forward(h_g, WQ_g, WK_g, WV_g, WO_g, num_heads=4, num_kv_heads=2, trace=tr_g)
    gqa_same_group = float(np.abs(tr_g["k_per_query_head"][0, 0] - tr_g["k_per_query_head"][0, 1]).max())
    out["sharing_contrast"] = {
        "mla_head0_k_token0": [r(v, 3) for v in head0_k],
        "mla_head1_k_token0": [r(v, 3) for v in head1_k],
        "mla_head0_head1_maxdiff": r(mla_cross, 3),          # >0:不同投影、逐位不同
        "mla_shared_latent_token0": [r(v, 3) for v in c_KV[0]],   # 两个头共同的上游
        "gqa_same_group_maxdiff": r(gqa_same_group, 6),      # 0.0:同组头逐位相同
        "reading": "GQA 复制现成向量(离散的头选择:同组逐位相同);MLA 共享生成基底(连续低秩子空间:同潜向量不同投影、逐位不同)",
    }

    # ---- 缓存账:蓝框两向量 ----
    out["cache_account"] = {
        "toy": {
            "c_KV_width": dims["d_c"], "k_R_width": dims["d_h_R"],
            "per_token_cache": mla_kv_cache_per_token(dims["d_c"], dims["d_h_R"]),   # 8 = 6+2
            "mha_equivalent": mha_kv_cache_per_token(dims["n_h"], dims["d_h"]),      # 16
            "expand": "6+2=8 vs 2*2*4=16(玩具上减半;真实配置 576 vs 32768 是 1.75%)",
        },
        "dsv3": {
            "per_token_cache": mla_kv_cache_per_token(512, 64),   # 576
            "mha_equivalent": mha_kv_cache_per_token(128, 128),   # 32768
            "equivalent_gqa_groups": equivalent_gqa_groups(512, 64, 128),  # 2.25
            "blue_box": "App C Eq.(41):仅 c^KV(512)与 k^R(64)需缓存——V3 重述 V3-1/V3-3 直接蓝框",
        },
        "cache_content_check": {
            "c_KV_is_rms_normed": "缓存的是 kv_a_layernorm 之后的 c^KV(§3.1.2 潜向量后 RMSNorm;vLLM 缓存 kv_c_normed)",
        },
    }

    # ---- vLLM 投影骨架锚 ----
    out["vllm_projection_skeleton"] = {
        "anchors_cited": {
            "fused": "vllm/model_executor/models/deepseek_v2.py:L1010-L1066(fused_qkv_a_proj 输出 [1536, 512+64];kv_b_proj 512→128×(128+128))",
            "forward": "vllm/model_executor/layers/mla.py:L150-L226(标准前向:split→双 RMSNorm→RoPE 只旋 rope 段→mla_attn)",
            "lines_cited": [1010, 1066, 150, 226],
        },
        "fused_qkv_a_proj_output_sizes": [1536, 576],    # [q_lora_rank, kv_lora_rank+qk_rope_head_dim]
        "kv_b_proj_in_out": [512, 32768],
        "note": "kv_b_proj 输出 32768 恰=MHA 每 token 缓存数——巧合但方向相反:这是把潜向量升回逐头空间的参数乘法宽度",
    }

    p = Path(__file__).resolve().parent / "ch24_m05_mla_compression.json"
    with open(p, "w", newline="\n", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
