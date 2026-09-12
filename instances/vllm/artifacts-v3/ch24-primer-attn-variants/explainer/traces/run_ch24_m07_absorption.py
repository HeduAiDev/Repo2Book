"""ch24-m07 驱动脚本 —— 吸收恒等式:W^UK→W^UQ、W^UV→W^O 的离线吸收(结合律)。

跑法(host, 纯 CPU numpy): python run_ch24_m07_absorption.py
输出: ch24_m07_absorption.json(与本脚本同目录)

素材对应 dossier 机制 ch24-m07,论文 arXiv:2405.04434 §2.1.2 吸收论断
('W^UK can be absorbed into W^Q, and W^UV can be absorbed into W^O …
we even do not need to compute keys and values out for attention')
+ App C('absorb W^UK into W^UQ, and W^UV into W^O … completed offline at once')。
2×2 手算与 tests/test_mla.py::test_absorption_score_three_orders_hand_example
同参:W_UQ=[[1,2],[3,4]]、W_UK=[[5,6],[7,8]]、c_Q=[0.5,-1]、c_KV=[2,3]。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
from mla import (  # noqa: E402
    absorption_score_three_orders,
    absorb_weights,
    make_mla_weights,
    mla_absorbed_forward,
    mla_naive_forward,
)


def r(v, nd=4):
    return round(float(v), nd)


def mat(m):
    return [[r(x, 2) for x in row] for row in m]


def vec(v):
    return [r(x, 2) for x in v]


def main():
    out = {}

    out["params"] = {
        "eqs": [10, 13, 43],
        "eq_note": "App C 口径:吸收进 W^UQ/W^O(V2 正文说吸进 W^Q 是简写——统一用附录 C 精确口径)",
    }

    # ---- 2×2 手算三连(tests 同参) ----
    W_UQ = np.array([[1.0, 2.0], [3.0, 4.0]])
    W_UK = np.array([[5.0, 6.0], [7.0, 8.0]])
    c_Q = np.array([0.5, -1.0])
    c_KV = np.array([2.0, 3.0])
    s1, s2, s3 = absorption_score_three_orders(W_UQ, W_UK, c_Q, c_KV)
    q = W_UQ @ c_Q
    k = W_UK @ c_KV
    q_latent = W_UK.T @ q
    B = W_UK.T @ W_UQ
    out["hand_2x2"] = {
        "W_UQ": mat(W_UQ), "W_UK": mat(W_UK),
        "c_Q": vec(c_Q), "c_KV": vec(c_KV),
        "order1_materialize": {
            "q_is_WUQ_cQ": vec(q),            # [-1.5,-2.5]
            "k_is_WUK_cKV": vec(k),           # [28,38]
            "score_q_dot_k": r(s1),           # -137
        },
        "order2_absorb_query_side": {
            "q_latent_is_WUKT_q": vec(q_latent),   # [-25,-29]
            "score_qlatent_dot_cKV": r(s2),        # -137
            "note": "k 从未物化——q 被投进潜空间直接与 c^KV 点积(decode 形态)",
        },
        "order3_offline_fold": {
            "B_is_WUKT_WUQ": mat(B),          # [[26,38],[30,44]]
            "B_cQ": vec(B @ c_Q),             # [-25,-29]
            "score_folded": r(s3),            # -137
            "note": "B 离线一次算好('related to only model parameters, it can be completed offline at once')",
        },
        "three_orders_equal": {
            "s1": r(s1), "s2": r(s2), "s3": r(s3),
            "max_abs_diff": r(max(abs(s1 - s2), abs(s2 - s3)), 12),   # 0.0
            "law": "矩阵乘法结合律:q^T(W c)=(W^T q)^T c——纯括号移动,无交换",
        },
    }

    # ---- 整段前向核验:naive vs absorbed(T=4 玩具,float64 逐位) ----
    dims = dict(d=8, n_h=2, d_h=4, d_c=6, d_h_R=2)
    w = make_mla_weights(**dims, q_lora_rank=5, seed=36)
    rng = np.random.default_rng(37)
    h = rng.standard_normal((4, dims["d"]))
    positions = np.arange(4)
    tr_n, tr_a = {}, {}
    u_naive, cache_n = mla_naive_forward(h, positions, w, trace=tr_n)
    wa = absorb_weights(w)
    u_abs, cache_a = mla_absorbed_forward(h, positions, wa, trace=tr_a)
    out["full_forward_equality"] = {
        "dims": dims, "T": 4, "compute_dtype": "numpy float64(host CPU)",
        "max_abs_diff_u": r(float(np.abs(u_naive - u_abs).max()), 12),
        "cache_identical": "两形态缓存同为蓝框 (c^KV,k_R)——吸收只动参数侧,不动缓存",
        "scale_naive": r(tr_n["scale"], 6),     # 1/sqrt(6)
        "scale_absorbed": r(tr_a["scale"], 6),  # 相等:标量缩放与结合次序交换
        "scale_real_dsv3": r(1.0 / np.sqrt(128 + 64), 6),   # 1/sqrt(192)=0.0722
        "absorbed_shapes": {
            "B_shape": list(wa.B.shape),              # (2,6,5) 逐头 d_c×d_c'
            "W_O_absorbed_shape": list(wa.W_O_absorbed.shape),  # (8,12) = d × (n_h·d_c)
        },
    }

    # ---- vLLM 脚印:process_weights_after_loading 的 split 与转置副本 ----
    out["vllm_footprint"] = {
        "note": "本机无 vLLM 包——形状账按源码公式算术镜像(mirror_of),非 import 实跑;运行时消费(bmm/forward_mqa)归 ch25",
        "anchors_cited": {
            "split": "vllm/model_executor/layers/attention/mla_attention.py:L1027-L1035(kv_b_proj 权重 view (512,128,256) 后 split 出 W_UK/W_UV)",
            "transpose_copies": "vllm/model_executor/layers/attention/mla_attention.py:L1092-L1100(W_UV→(N,L,V)、W_UK_T→(N,P,L) replace_parameter 副本)",
            "lines_cited": [1027, 1035, 1092, 1100],
        },
        "mirror_shapes_dsv3": {
            "kv_b_proj_weight_view": [512, 128, 256],
            "W_UK_after_split": [512, 128, 128],
            "W_UV_after_split": [512, 128, 128],
            "W_UK_T_permuted": [128, 128, 512],
            "W_UV_transposed": [128, 512, 128],
        },
    }

    p = Path(__file__).resolve().parent / "ch24_m07_absorption.json"
    with open(p, "w", newline="\n", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
