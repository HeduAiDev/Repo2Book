"""ch20-m05 驱动脚本 —— IO 复杂度账的元素级精确计数(Theorem 2 代入具体参数)。

跑法(host, 纯 CPU): python run_ch20_m05_io_accounting.py
输出: ch20_m05_io_accounting.json

素材对应 dossier 机制 ch20-m05(IO 复杂度账: Θ(Nd+N^2) vs Θ(N^2 d^2/M) 与下界),
论文 arXiv:2205.14135 §3.2 Theorem 2 + Proposition 3。
参数 N=1024, d=64(GPT-2 头维, paper.md:L68 原例 "for GPT2, N=1024 and d=64");
Bc 扫 {64,128,256} —— 遍数 T_c=⌈N/Bc⌉ 每翻倍 Bc 减半,访问量单调下降,
Bc>256 后论文实测收益封顶(Fig.2 中图)。8K 感受数字在 m01 trace。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
from flash_attention import (  # noqa: E402
    fa_block_sizes,
    hbm_accesses_flash,
    hbm_accesses_standard,
)


def main():
    out = {}
    N, d = 1024, 64
    out["params"] = {
        "N": N, "d": 64, "note": "GPT-2 头维原例(arXiv:2205.14135 §2.2 'for GPT2, N=1024 and d=64')",
        "Bc_sweep": [64, 128, 256],
        "paper_rule_block_size": "Bc=ceil(M/4d), Br=min(ceil(M/4d),d) (Algorithm 1 line 1)",
    }

    std = hbm_accesses_standard(N, d)
    out["standard_attention"] = {
        "step1_readQK_writeS": N * d + N * d + N * N,
        "step2_readS_writeP": N * N + N * N,
        "step3_readPV_writeO": N * N + N * d + N * d,
        "total_hbm_accesses_elements": std,
        "theta": "Θ(Nd + N^2)",
        "materialized_NxN_tables": 2,
        "materialized_elements": 2 * N * N,
    }

    rows = []
    for bc in (64, 128, 256):
        acc = hbm_accesses_flash(N, d, block_size_c=bc)
        t_c = -(-N // bc)
        rows.append({
            "Bc": bc,
            "Br": min(bc, d),
            "T_c_passes": t_c,
            "outer_KV_once_elements": 2 * N * d,
            "inner_per_pass_elements": 3 * N * d + 4 * N,
            "total_hbm_accesses_elements": acc,
            "ratio_standard_over_flash": round(std / acc, 4),
            "materialized_NxN_elements": 0,
        })
    out["flash_sweep"] = rows
    out["monotone_more_Bc_less_access"] = bool(rows[0]["total_hbm_accesses_elements"] > rows[1]["total_hbm_accesses_elements"] > rows[2]["total_hbm_accesses_elements"])
    out["theta_flash"] = "Θ(N^2 d^2 / M) —— Bc=Θ(M/4d) 代入: 遍数 N/Bc=Θ(Nd/M), 每遍 Θ(Nd)"

    # 论文块尺寸规则代入 M≈100KB(元素按 fp16 计 51200): Bc=ceil(51200/256)=200
    bc_paper, br_paper = fa_block_sizes(51200, d)
    out["paper_block_size_rule_at_M_100KB_fp16"] = {
        "M_elements_fp16": 51200,
        "Bc": bc_paper,
        "Br": br_paper,
        "note": "工程上 FA-2 取 {64,128}x{64,128} 视 d 与 SMEM 而定(arXiv:2307.08691 §3.2)",
    }

    # 标准版必须物化的两张表 vs FA 的 O(N) 额外内存(Theorem 1)
    out["memory_footprint_N1024"] = {
        "standard_materialized_elements": 2 * N * N,
        "standard_materialized_fp16_bytes": 2 * N * N * 2,
        "flash_extra_stats_elements_m_and_l": 2 * N,
        "flash_extra_stats_fp16_bytes": 2 * N * 2,
    }

    p = Path(__file__).parent / "ch20_m05_io_accounting.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"OK {p}")


if __name__ == "__main__":
    main()
