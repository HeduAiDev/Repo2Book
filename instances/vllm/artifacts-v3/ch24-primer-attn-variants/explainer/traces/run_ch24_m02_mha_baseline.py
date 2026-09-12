"""ch24-m02 驱动脚本 —— MHA 基线:式(1)-(8) 与 vLLM q/k/v split 的形状同构。

跑法(host, 纯 CPU numpy): python run_ch24_m02_mha_baseline.py
输出: ch24_m02_mha_baseline.json(与本脚本同目录)

素材对应 dossier 机制 ch24-m02,论文 arXiv:2405.04434 §2.1.1 Eq.(1)-(8)
(与 tests/test_mha_gqa.py::test_trace_records_worked_example_shapes 同参数:
T=3、H=2、d_h=4、d=6——投影 (3,8)→view (3,2,4)→逐头分数 (2,3,3))。
vLLM 同构:llama.py LlamaAttention.forward 五行(split 的 kv_size 出现两次
=『K 一份 V 一份』的 2 因子;scaling=head_dim**-0.5=式(7) 的 √d_h)。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
from mha_gqa import attention_forward  # noqa: E402


def r(v, nd=3):
    return round(float(v), nd)


def main():
    out = {}

    # 与 test_trace_records_worked_example_shapes 完全同参(seed 14/15)
    T, d, n_h, d_h = 3, 6, 2, 4
    rng = np.random.default_rng(14)
    h = rng.standard_normal((T, d))

    def make_w(seed):
        g = np.random.default_rng(seed)
        W_Q = g.standard_normal((n_h * d_h, d))
        W_K = g.standard_normal((n_h * d_h, d))
        W_V = g.standard_normal((n_h * d_h, d))
        W_O = g.standard_normal((n_h * d_h, n_h * d_h))
        return W_Q, W_K, W_V, W_O

    W_Q, W_K, W_V, W_O = make_w(15)
    trace = {}
    u, (k_heads, v_heads) = attention_forward(
        h, W_Q, W_K, W_V, W_O, num_heads=n_h, trace=trace
    )

    out["params"] = {
        "T": T, "d": d, "n_h": n_h, "d_h": d_h,
        "seed_h": 14, "seed_w": 15,
        "eqs": [1, 2, 3, 4, 5, 6, 7, 8],
        "note": "T=3 token、H=2 头、d_h=4、d=6——小到形状可心算;与 ch20 站 6 varlen kernel 吃的形状同构",
    }
    out["cross_refs"] = {
        "kernel_math": "ch20(kernel 内部的 softmax/tiling 数学)",
        "backend_choice": "ch21(吃这些形状的后端家族)",
    }

    # ---- 四拍形状账 ----
    out["shape_account"] = {
        "input_h": list(h.shape),                     # (3,6)
        "after_projection_qkv": list(trace["q"].shape),   # (3,8)
        "W_shapes_out_by_in": {"W_Q": [8, 6], "W_K": [8, 6], "W_V": [8, 6], "W_O": [8, 8]},
        "after_split_heads_q": list(trace["q_heads"].shape),   # (3,2,4)
        "per_head_scores": list(trace["scores"].shape),        # (2,3,3)
        "per_head_probs": list(trace["probs"].shape),          # (2,3,3)
        "o_heads": list(trace["o_heads"].shape),               # (3,2,4)
        "output_u": list(u.shape),                             # (3,8)
        "scale_sqrt_dh": r(trace["scale"], 4),                 # 0.5 = 4**-0.5 = Eq.(7) 1/sqrt(4)
        "kv_per_token_elements": int(k_heads[0].size + v_heads[0].size),  # 16 = 2*2*4
        "kv_per_token_expand": "2(K,V 各一份)*2 头*4 维 = 16",
    }

    # ---- t=2(最后一个 token)head 0 的因果 softmax 逐值 ----
    s2 = trace["scores"][0, 2, :]      # head0, query t=2, 对 j=0..2 的分数(已乘 scale)
    p2 = trace["probs"][0, 2, :]
    out["causal_softmax_t2_head0"] = {
        "scores_j0_j1_j2": [r(v, 4) for v in s2],
        "scores_masked_check": "上三角(j>t)= -inf 掩掉,本行 t=2 是最后一行、j=0..2 全可见",
        "softmax_probs": [r(v, 4) for v in p2],
        "probs_sum": r(p2.sum(), 6),
        "masked_upper_triangle_example": {
            "t0_row_scores": [r(v, 4) if np.isfinite(v) else "-inf" for v in trace["scores"][0, 0, :]],
            "note": "t=0 行只有 j=0 可见(Eq.(7) 求和上限 j<=t),j=1/j=2 为 -inf→概率恰 0",
        },
        "o_t2_head0": [r(v, 4) for v in trace["o_heads"][2, 0, :]],
    }

    # ---- 具体中间量(供正文数值推演表引用;float64) ----
    out["concrete_values"] = {
        "h_token0": [r(v, 3) for v in h[0]],
        "q_heads_token0_head0": [r(v, 3) for v in trace["q_heads"][0, 0]],
        "q_heads_token0_head1": [r(v, 3) for v in trace["q_heads"][0, 1]],
        "k_heads_token0_head0": [r(v, 3) for v in trace["k_heads"][0, 0]],
        "k_heads_token0_head1": [r(v, 3) for v in trace["k_heads"][0, 1]],
        "heads_are_distinct": "head0 与 head1 的 q/k 向量不同(逐头切分各自算分)",
    }

    # ---- vLLM 同构锚(llama.py forward 五行) ----
    out["vllm_isomorphism"] = {
        "anchors_cited": {
            "llama_init": "vllm/model_executor/models/llama.py:L157-L159(q_size/kv_size/scaling 三行)",
            "llama_forward": "vllm/model_executor/models/llama.py:L221-L231(forward 五行)",
            "lines_cited": [157, 158, 159, 221, 231],
        },
        "q_size": n_h * d_h,        # 8
        "kv_size": n_h * d_h,       # MHA 时 = q_size = 8
        "split_sizes": [8, 8, 8],   # split([q_size, kv_size, kv_size])——kv_size 出现 2 次
        "kv_size_appears_twice": "K 一份、V 一份的 2 因子就长在 split 尺寸表里",
        "scaling_code": "self.scaling = self.head_dim**-0.5  # = 0.5(本例)= Eq.(7) 的 1/sqrt(d_h)",
    }

    p = Path(__file__).resolve().parent / "ch24_m02_mha_baseline.json"
    with open(p, "w", newline="\n", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
