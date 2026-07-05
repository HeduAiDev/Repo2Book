#!/usr/bin/env python3
"""Explainer driver — DefaultEplb.rebalance_experts 再均衡演化 worked example。

跑本章精简版 policy_default_eplb.DefaultEplb（纯 numpy，host 可跑），在两组小而具体的
参数上驱动真实控制流，逐步导出「偏斜→加冗余副本→贪心装箱→收益门槛判定」的每个关键标量。
输出 rebalance.json 供 explainer.json 的逐轮表与 figure-spec 取数（数字全部来自本次运行）。

用法: python3 run_rebalance.py   （产物写同目录 rebalance.json）
"""
import json
import sys
from pathlib import Path

import numpy as np

# 精简版实现所在目录（同章 implementation/），平铺导入。
IMPL = Path(__file__).resolve().parents[2] / "implementation"
sys.path.insert(0, str(IMPL))
from policy_default_eplb import DefaultEplb  # noqa: E402


def imbalance(loads):
    a = np.array(loads, dtype=float)
    return round(float(a.max() / a.mean()), 4)


def analyze(name, current, workload, gate=0.95):
    """驱动真实 rebalance_experts，并单独复算 add_redundant / 贪心装箱以取每步标量。"""
    policy = DefaultEplb()
    wl = np.array(workload)
    placement = np.array(current)
    num_npus = placement.shape[1]

    # ① 采集：每卡当前总热度
    heat_before = wl[0].sum(axis=1).tolist()
    peak_before = int(max(heat_before))
    mean_before = float(np.mean(heat_before))

    # ② add_redundant：把重复出现的物理 expert 折算冗余预算，按 expert 聚合负载
    row = placement[0].reshape(-1)
    expert_ids, counts = np.unique(row, return_counts=True)
    num_redundancy = int(DefaultEplb.get_redundant_num(num_npus, counts))
    per_expert = DefaultEplb.add_redundant(placement, wl, len(expert_ids))[0].tolist()

    # ③ 逐层贪心装箱：把（含冗余副本的）expert 逐个放进当前最低热度卡
    weights = np.zeros((len(per_expert),), dtype="object")
    for eid, w in enumerate(per_expert):
        weights[eid] = (eid, w)
    result, boxes = DefaultEplb.original_compute_balanced_pack_redundancy(
        weights, num_npus, num_redundancy
    )
    heat_after = [float(b["total_weight"]) for b in result]
    peak_after = float(max(heat_after))

    # ④ 收益门槛：驱动完整 rebalance_experts 取 change（真实返回值）
    change, priority, deployment = policy.rebalance_experts(current, workload)

    gate_value = round(gate * peak_before, 4)      # 0.95 * origin
    drop_pct = round((1 - peak_after / peak_before) * 100, 2)

    rec = {
        "name": name,
        "num_npus": num_npus,
        "heat_before": heat_before,
        "peak_before": peak_before,
        "mean_before": mean_before,
        "imbalance_before": imbalance(heat_before),
        "num_redundancy": num_redundancy,
        "per_expert_after_add_redundant": per_expert,
        "boxes": [list(b) for b in boxes],
        "heat_after": heat_after,
        "peak_after": peak_after,
        "imbalance_after": imbalance(heat_after),
        "gate_ratio": gate,
        "gate_value": gate_value,
        "peak_after_lt_gate": bool(peak_after < gate_value),
        "drop_pct": drop_pct,
        "change": int(change),
        "deployment": deployment,
    }
    return rec


def main():
    scenarios = [
        # ACT：单卡热门 expert 独大 [100,20,20,20]，imbalance=2.5，铺平后峰值 100→60 降 40%>5% → change=1
        ("act_skew", [[[0, 0], [1, 2], [3, 4], [5, 6]]],
         [[[100, 0], [10, 10], [10, 10], [10, 10]]]),
        # SKIP：已近均衡 [52,48,48,48]，峰值仅 52→50 降 ~4%<5% → 不值得搬 → change=0
        ("skip_mild", [[[0, 0], [1, 2], [3, 4], [5, 6]]],
         [[[52, 0], [24, 24], [24, 24], [24, 24]]]),
    ]
    out = {"gate_pct": 5, "results": [analyze(n, c, w) for n, c, w in scenarios]}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    (Path(__file__).resolve().parent / "rebalance.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
