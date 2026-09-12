"""ch24-m06 驱动脚本 —— query 侧低秩:式(12)-(13) c^Q 与 V2-Lite 的不压分支。

跑法(host, 纯 CPU numpy): python run_ch24_m06_query_lora.py
输出: ch24_m06_query_lora.json(与本脚本同目录)

素材对应 dossier 机制 ch24-m06,论文 arXiv:2405.04434 §2.1.2 Eq.(12)-(13)
(query 低秩压训练激活、不压 cache)+ App B.1(V2-Lite:27 层/2048 hidden/
16 头/不压 query)。同一份参考实现吃两种配置(q_lora_rank=值 vs None),
证明 query 压缩是独立旋钮、cache 恒 576。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
from mla import (  # noqa: E402
    absorb_weights,
    make_mla_weights,
    mla_absorbed_forward,
    mla_naive_forward,
)


def r(v, nd=4):
    return round(float(v), nd)


def main():
    out = {}

    out["params"] = {
        "eqs": [12, 13, 14, 37, 42],
        "eq_note": "Eq.(12) c^Q=W^DQ h;Eq.(13) q^C=W^UQ c^Q;W^QR(Eq.14)也从 c^Q 出——query 路径整条低秩",
    }

    # ---- 真值形状账:V2(压)vs V2-Lite(不压) ----
    out["real_shape_account"] = {
        "deepseek_v2": {
            "hidden_d": 5120, "n_h": 128, "d_h": 128,
            "d_c_prime": 1536, "d_c": 512, "d_h_R": 64,
            "q_path": "h(5120)→W^DQ→c^Q(1536)→[W^UQ→16384 | W^QR→8192]",
            "W_DQ_out_by_in": [1536, 5120],
            "W_UQ_out_by_in": [16384, 1536],
            "W_QR_out_by_in": [8192, 1536],
            "full_q_width_if_uncompressed": 128 * (128 + 64),  # 24576 = q^C+q^R 直投宽度
            "bottleneck_activation": 1536,
            "activation_compression_ratio": r(24576 / 1536, 2),  # 16.0
            "fused_qkv_a_proj_output_sizes": [1536, 576],
            "cache_per_token": 576,
        },
        "deepseek_v2_lite": {
            "hidden_d": 2048, "n_h": 16, "d_h": 128, "layers": 27,
            "d_c": 512, "d_h_R": 64, "q_compressed": False,
            "q_path": "h(2048)→W^QC 直投→16×(128+64)=3072(无 c^Q 瓶颈)",
            "W_QC_out_by_in": [3072, 2048],
            "q_proj_output_width": 3072,
            "kv_a_proj_with_mqa_output_width": 576,
            "cache_per_token": 576,
            "paper_quote": "App B.1 'slightly different from DeepSeek-V2, it does not compress the queries'",
            "cache_note": "d_c=512 不变 → Lite 每 token 缓存同样 576——query 旋钮不动 cache",
        },
        "reading": "c^Q∈R^{1536} 压的是训练期激活内存(query 不缓存,推理 cache 与它无关);V2-Lite 干脆不压,代码留 q_lora_rank=None 退化分支",
    }

    # ---- 玩具双分支:q_lora_rank=5 vs None,同一套下游 ----
    dims = dict(d=8, n_h=2, d_h=4, d_c=6, d_h_R=2)
    rng = np.random.default_rng(33)
    h = rng.standard_normal((4, dims["d"]))
    positions = np.arange(4)

    w_v2 = make_mla_weights(**dims, q_lora_rank=5, seed=34)
    w_lite = make_mla_weights(**dims, q_lora_rank=None, seed=35)
    tr_v2, tr_lite = {}, {}
    u_v2, cache_v2 = mla_naive_forward(h, positions, w_v2, trace=tr_v2)
    u_lite, cache_lite = mla_naive_forward(h, positions, w_lite, trace=tr_lite)

    # 两分支都能吸收,且吸收后与 naive 逐位一致
    wa_v2 = absorb_weights(w_v2)
    wa_lite = absorb_weights(w_lite)
    u_abs_v2, _ = mla_absorbed_forward(h, positions, wa_v2)
    u_abs_lite, _ = mla_absorbed_forward(h, positions, wa_lite)
    out["toy_both_branches"] = {
        "dims": dims,
        "v2_branch": {
            "q_lora_rank": 5,
            "c_Q_shape": list(tr_v2["c_Q"].shape),          # (4,5)
            "modules_present": ["W_DQ", "W_UQ", "W_QR", "W_DKV", "W_UK", "W_KR", "W_UV", "W_O"],
            "cache_shapes": [list(cache_v2[0].shape), list(cache_v2[1].shape)],  # (4,6),(4,2)
            "cache_width_per_token": 8,
            "absorb_naive_maxdiff": r(float(np.abs(u_abs_v2 - u_v2).max()), 12),
        },
        "lite_branch": {
            "q_lora_rank": None,
            "c_Q_shape": None,                               # 无 c^Q
            "modules_present": ["W_QC(直投)", "W_QR(改吃 h)", "W_DKV", "W_UK", "W_KR", "W_UV", "W_O"],
            "modules_absent": ["W_DQ", "W_UQ"],
            "cache_shapes": [list(cache_lite[0].shape), list(cache_lite[1].shape)],  # (4,6),(4,2)
            "cache_width_per_token": 8,
            "absorb_naive_maxdiff": r(float(np.abs(u_abs_lite - u_lite).max()), 12),
        },
        "cache_identical_width": "两分支 cache 同形 (4,6)+(4,2)=8 元素/token——query 旋钮不进蓝框",
    }

    # ---- vLLM 模块名锚 ----
    out["vllm_module_names"] = {
        "anchors_cited": {
            "branch": "vllm/model_executor/models/deepseek_v2.py:L1010-L1050(q_lora_rank is not None → fused_qkv_a_proj+q_a_layernorm+q_b_proj;else → kv_a_proj_with_mqa+q_proj)",
            "lines_cited": [1010, 1016, 1034, 1050],
        },
        "with_q_lora": "fused_qkv_a_proj / q_a_layernorm / q_b_proj",
        "without_q_lora": "kv_a_proj_with_mqa / q_proj",
        "name_reading": "模块名 kv_a_proj_with_mqa 里的 with_mqa——DeepSeek HF 权重名自证 decode 侧 MQA 形状",
    }

    p = Path(__file__).resolve().parent / "ch24_m06_query_lora.json"
    with open(p, "w", newline="\n", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
