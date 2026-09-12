"""ch24-m03 驱动脚本 —— MQA:H 个 KV 头减到 1(cache/带宽降 H 倍、容量同步被砍)。

跑法(host, 纯 CPU numpy): python run_ch24_m03_mqa.py
输出: ch24_m03_mqa.json(与本脚本同目录)

素材对应 dossier 机制 ch24-m03,论文出处:
- arXiv:2305.13245 §1(memory bandwidth overhead from loading keys and values)+
  §2.2 factor-H 句(Going from MHA to MQA reduces H key and value heads to a
  single key and value head, reducing the size of the key-value cache and
  therefore amount of data that needs to be loaded by a factor of H);
- vLLM 侧:QKVParallelLinear total_num_kv_heads=1 → num_kv_head_replicas=tp_size
  (linear.py:L1070-L1081;本机无 vLLM 包,以同式算术镜像并标注 mirror_of)。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
from kv_cache_table import (  # noqa: E402
    gqa_kv_cache_per_token,
    mha_to_mqa_cache_reduction_factor,
)
from mha_gqa import attention_forward, kv_head_of_query_head  # noqa: E402


def r(v, nd=4):
    return round(float(v), nd)


def mirror_vllm_head_account(total_num_kv_heads: int, tp_size: int) -> dict:
    """linear.py:L1074-L1081 头数账的算术镜像(同式,非 import)。"""
    if tp_size >= total_num_kv_heads:
        num_kv_heads = 1
        num_kv_head_replicas = tp_size // total_num_kv_heads
    else:
        num_kv_heads = total_num_kv_heads // tp_size
        num_kv_head_replicas = 1
    return {"total_num_kv_heads": total_num_kv_heads, "tp_size": tp_size,
            "num_kv_heads": num_kv_heads, "num_kv_head_replicas": num_kv_head_replicas}


def main():
    out = {}

    # ---- 主例:H=32 头、d_h=128(Llama-2-7B 头形)的元素账 ----
    H, d_h = 32, 128
    out["display_names"] = {
        "GQA-8": "8 组(2048 元素)",
        "GQA-4": "4 组(1024 元素)",
    }
    out["params"] = {
        "H": H, "d_h": d_h,
        "eqs_note": "arXiv:2305.13245 §2.2 factor-H 句 + §2.1 Shazeer 2019 谱系",
        "note": "MQA=Shazeer 2019(经 GQA 论文引用):多 query 头共享单个 K/V 头",
    }
    out["element_account_H32"] = {
        "MHA_G32": gqa_kv_cache_per_token(H, d_h),          # 8192 = 2*32*128
        "GQA8": gqa_kv_cache_per_token(8, d_h),             # 2048
        "GQA4": gqa_kv_cache_per_token(4, d_h),             # 1024
        "MQA_G1": gqa_kv_cache_per_token(1, d_h),           # 256 = 2*1*128
        "mha_to_mqa_factor": mha_to_mqa_cache_reduction_factor(H),  # 32
        "expand": {"MHA": "2*32*128", "MQA": "2*1*128"},
        "capacity_note": "K/V 线性映射的表达容量同步缩 32 倍——'bandwidth 和 capacity 双砍'(§2.2 'a more aggressive cut in both memory bandwidth and capacity')",
    }

    # ---- 玩具前向核验:H=4、d_h=2、G=1 —— 全部 query 头吃同一个 KV 头 ----
    T, d, n_h, dh_toy = 2, 6, 4, 2
    rng = np.random.default_rng(16)
    h = rng.standard_normal((T, d))
    g = np.random.default_rng(17)
    W_Q = g.standard_normal((n_h * dh_toy, d))
    W_K = g.standard_normal((1 * dh_toy, d))     # 只有 1 个 KV 头
    W_V = g.standard_normal((1 * dh_toy, d))
    W_O = g.standard_normal((n_h * dh_toy, n_h * dh_toy))
    trace = {}
    attention_forward(h, W_Q, W_K, W_V, W_O, num_heads=n_h, num_kv_heads=1, trace=trace)
    kmap = kv_head_of_query_head(n_h, 1)
    cross_head_diff = float(np.abs(trace["k_per_query_head"][:, 0] - trace["k_per_query_head"][:, 3]).max())
    out["toy_forward_G1"] = {
        "T": T, "d": d, "num_heads": n_h, "d_h": dh_toy, "num_kv_heads": 1,
        "kv_head_map": [int(x) for x in kmap],             # [0,0,0,0] 全部指向 KV 头 0
        "k_heads_shape": list(trace["k_heads"].shape),     # (2,1,2) 只有一份 K
        "k_per_query_head_shape": list(trace["k_per_query_head"].shape),  # (2,4,2) 广播给 4 个头
        "cross_head_max_diff": r(cross_head_diff, 6),      # 0.0 —— 4 个头吃的 K 逐位相同
        "kv_per_token_elements": int(trace["k_heads"][0].size + trace["v_heads"][0].size),  # 4 = 2*1*2
        "mha_equivalent_elements": 2 * n_h * dh_toy,       # 16
        "factor": n_h,                                     # 4
    }

    # ---- vLLM 头数账镜像(tp=8):MQA 时每 rank 复制同一份 KV 头 ----
    out["vllm_head_account_mirror"] = {
        "note": "本机无 vLLM 包——linear.py:L1074-L1081 同式算术镜像(mirror_of),非 import 实跑",
        "anchors_cited": {
            "linear": "vllm/model_executor/layers/linear.py:L1070-L1088(None 默认=MHA/replicas 划分/output_sizes 三段)",
            "docstring": "vllm/model_executor/layers/linear.py:L1028-L1030(KV 头少于 query 头时复制 KV 头、切分 query 头)",
            "lines_cited": [1070, 1074, 1081, 1088, 1028, 1030],
        },
        "tp_size": 8,
        "mha_32kv": mirror_vllm_head_account(32, 8),    # 4 头/rank, replicas=1
        "gqa_8kv": mirror_vllm_head_account(8, 8),      # 1 头/rank, replicas=1
        "gqa_4kv": mirror_vllm_head_account(4, 8),      # 1 头/rank, replicas=2
        "mqa_1kv": mirror_vllm_head_account(1, 8),      # 1 头/rank, replicas=8
        "mqa_reading": "total_num_kv_heads=1 时 num_kv_head_replicas=tp_size=8——每个 rank 持有同一个 KV 头的副本(权重复制)",
    }

    p = Path(__file__).resolve().parent / "ch24_m03_mqa.json"
    with open(p, "w", newline="\n", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
