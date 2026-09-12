"""ch24-m08 驱动脚本 —— 解耦 RoPE:非交换性论证、共享 k^R 与 √(d_h+d_h^R) 缩放。

跑法(host, 纯 CPU numpy): python run_ch24_m08_decoupled_rope.py
输出: ch24_m08_decoupled_rope.json(与本脚本同目录)

素材对应 dossier 机制 ch24-m08,论文 arXiv:2405.04434 §2.1.3 Eq.(14)-(19)
+ §3.1.4(YaRN 只作用 k^R)。2×2 手算与 tests/test_mla.py::
test_noncommutativity_and_rope_coupling_breaks_absorption 同参:
W_UQ=[[1,2],[3,4]]、W_UK=[[5,6],[7,8]]。
三段论证:① AB≠BA(交换律失效直感);② 灾难版——RoPE 若作用于 k^C,
折叠矩阵随位置对变化、吸收失效;③ 解法核验——位置平移只动 R 段。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
from mla import make_mla_weights, mla_naive_forward, noncommutativity_and_rope_coupling, rope_rotate  # noqa: E402


def r(v, nd=4):
    return round(float(v), nd)


def mat(m):
    return [[r(x, 4) for x in row] for row in m]


def main():
    out = {}

    out["params"] = {
        "eqs": [14, 15, 16, 17, 18, 19],
        "eq_note": "Eq.(14) 逐头 q^R / Eq.(15) 共享单份 k^R / Eq.(16)(17) 拼接 [C;R] / Eq.(18) 分母 √(d_h+d_h^R)",
    }

    W_UQ = np.array([[1.0, 2.0], [3.0, 4.0]])
    W_UK = np.array([[5.0, 6.0], [7.0, 8.0]])

    # ---- ① AB ≠ BA(交换律失效直感) ----
    demo = noncommutativity_and_rope_coupling(W_UQ, W_UK, theta=1.0)
    out["noncommutativity"] = {
        "W_UQ": [[1, 2], [3, 4]], "W_UK": [[5, 6], [7, 8]],
        "AB_is_WUQ_WUK": mat(demo["AB"]),     # [[19,22],[43,50]]
        "BA_is_WUK_WUQ": mat(demo["BA"]),     # [[23,34],[31,46]]
        "AB_minus_BA": mat(demo["AB"] - demo["BA"]),   # [[-4,-12],[12,4]]
        "not_commutative": bool(demo["not_commutative"]),
    }

    # ---- ② 灾难版:RoPE 若作用于 k^C,折叠矩阵随位置对变化 ----
    theta = 1.0
    c1, s1_ = np.cos(theta), np.sin(theta)
    out["rope_coupling_breaks_absorption"] = {
        "R1_matrix": [[r(c1), r(-s1_)], [r(s1_), r(c1)]],   # 位置 1 的 2 维 RoPE 旋转
        "B_fixed_is_WUKT_WUQ": mat(demo["B_fixed"]),        # [[26,38],[30,44]] —— 无 RoPE 时的单一离线折叠
        "F00_equals_B_fixed": mat(demo["F_00"]),            # R(0)=I → 退化为固定 B
        "F01_WUKT_R1_WUQ": mat(demo["F_01"]),               # 位置对 (0,1) 的折叠矩阵——与 F_00 不同
        "fold_is_position_dependent": bool(demo["fold_is_position_dependent"]),
        "F00_minus_F01_maxabs": r(float(np.abs(demo["F_00"] - demo["F_01"]).max()), 4),
        "reading": "若 RoPE 施于 k^C,分数=(W^UQ c^Q)^T·R(m)^T R(t)·(W^UK c^KV)——R 落在 W^UQ 与 W^UK 之间,折叠矩阵 F(m,t) 随位置对变化,不存在单一离线形态(吸收失效);每 token 要为全部前缀重算 keys",
        "paper_quote": "§2.1.3 'a RoPE matrix related to the currently generating token will lie between W^Q and W^UK and matrix multiplication does not obey a commutative law'",
    }

    # ---- RoPE 的正交性(位置敏感的另一面):同位内积不变、异位改变 ----
    rng = np.random.default_rng(38)
    qv = rng.standard_normal(4)
    kv = rng.standard_normal(4)
    qr_p = rope_rotate(qv[None, :], np.array([123]))[0]
    kr_p = rope_rotate(kv[None, :], np.array([123]))[0]
    kr_other = rope_rotate(kv[None, :], np.array([456]))[0]
    out["rope_orthogonality"] = {
        "same_position_dot_before": r(qv @ kv),
        "same_position_dot_after": r(qr_p @ kr_p),          # 相等——同位同旋不改内积
        "same_pos_diff": r(abs((qv @ kv) - (qr_p @ kr_p)), 12),
        "cross_position_dot": r(qr_p @ kr_other),           # 不同——相对位置差进分数
        "norm_preserved": "RoPE 是正交旋转:范数保持(测试统计检验 512 样本锁定)",
    }

    # ---- ③ 解法核验:位置平移只动 R 段,C 段逐位不变 ----
    dims = dict(d=8, n_h=2, d_h=4, d_c=6, d_h_R=2)
    w = make_mla_weights(**dims, q_lora_rank=5, seed=39)
    rng2 = np.random.default_rng(40)
    T = 4
    h = rng2.standard_normal((T, dims["d"]))
    tr1, tr2 = {}, {}
    mla_naive_forward(h, np.arange(T), w, trace=tr1)
    mla_naive_forward(h, np.arange(T) + 100, w, trace=tr2)
    out["decoupling_check"] = {
        "positions_run1": [0, 1, 2, 3],
        "positions_run2_shift": 100,
        "C_segment_unchanged_maxdiff": {
            "q_C": r(float(np.abs(tr1["q_C"] - tr2["q_C"]).max()), 12),   # 0.0
            "k_C": r(float(np.abs(tr1["k_C"] - tr2["k_C"]).max()), 12),   # 0.0
            "c_KV_cache": r(float(np.abs(tr1["c_KV"] - tr2["c_KV"]).max()), 12),  # 0.0 —— 缓存侧不旋
        },
        "R_segment_rotated_maxdiff": {
            "q_R": r(float(np.abs(tr1["q_R"] - tr2["q_R"]).max()), 4),    # >0 才旋
            "k_R": r(float(np.abs(tr1["k_R"] - tr2["k_R"]).max()), 4),
        },
        "vllm_anchor": {
            "where": "vllm/model_executor/layers/mla.py:L200-L203(只把 q[..., qk_nope_head_dim:] 后 64 维与 k_pe 进 rotary);L192 k_pe.unsqueeze(1)=共享单份广播",
            "deepseek_v2": "vllm/model_executor/models/deepseek_v2.py:L1075-L1080(rotary 只建 qk_rope_head_dim 维)",
            "lines_cited": [192, 200, 203, 1075, 1080],
        },
    }

    # ---- 分母账:√(d_h+d_h^R) ----
    out["scale_denominator"] = {
        "eq18_quote": "Eq.(18) 分母 √(d_h+d_h^R)——拼接后 q/k 每头 192 维",
        "toy_scale": r(1.0 / np.sqrt(4 + 2), 4),        # 0.4082 = 1/√6
        "dsv3_scale": r(1.0 / np.sqrt(128 + 64), 4),    # 0.0722 = 1/√192
        "dsv3_qk_head_dim": 192,
        "code": "deepseek_v2.py:L1003 self.qk_head_dim**-0.5(含 yarn mscale 修正分支在同函数后段)",
        "yarn_note": "§3.1.4 'YaRN was specifically applied to the decoupled shared key k^R as it is responsible for carrying RoPE'——结构决定扩展点",
    }

    p = Path(__file__).resolve().parent / "ch24_m08_decoupled_rope.json"
    with open(p, "w", newline="\n", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
