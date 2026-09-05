"""ch20-m01 驱动脚本 —— 标准注意力物化两张 N×N 的感受数字 + A100 内存层级带宽差。

跑法(host, 纯 CPU): python run_ch20_m01_materialization.py
输出: ch20_m01_materialization.json

素材对应 dossier 机制 ch20-m01(标准注意力的内存带宽墙),论文 arXiv:2205.14135
§2.1(paper.md:L45 A100 内存层级原数字)与 §2.2(Alg.0 三步物化 S/P)。
A100 层级常量逐字取自论文(paper.md:L45);元素/字节计数由
flash_attention.materialized_intermediate_elements 精确计算。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
from flash_attention import materialized_intermediate_elements  # noqa: E402


def main():
    out = {}

    # ---- 8K 上下文的感受数字(一个 head) ----
    N = 8192
    one = N * N
    both = materialized_intermediate_elements(N)  # S 一张 + P 一张 = 2N^2
    out["N8192_one_head"] = {
        "N": N,
        "S_table_elements": one,
        "P_table_elements": one,
        "both_tables_elements": both,
        "one_table_fp16_bytes": one * 2,
        "one_table_fp16_MB_decimal": round(one * 2 / 1e6, 1),
        "both_tables_fp16_bytes": both * 2,
        "note": "softmax 行归一化要把一整行 K 的分数全部加起来,S/P 各 N×N 必须真实存在",
    }

    # ---- GPT-2 尺寸(paper.md:L68 原例 N=1024, d=64) ----
    N2 = 1024
    out["N1024_gpt2"] = {
        "N": N2,
        "both_tables_elements": materialized_intermediate_elements(N2),
        "both_tables_fp16_bytes": materialized_intermediate_elements(N2) * 2,
    }

    # ---- A100 内存层级(paper.md:L45 原数字) ----
    out["a100_memory_hierarchy_paper"] = {
        "HBM_capacity_GB": "40-80",
        "HBM_bandwidth_TBps": "1.5-2.0",
        "SRAM_per_SM_KB": 192,
        "SRAM_num_SMs": 108,
        "SRAM_bandwidth_TBps_est": 19,
        "bandwidth_ratio_SRAM_over_HBM_low": round(19 / 2.0, 2),
        "bandwidth_ratio_SRAM_over_HBM_high": round(19 / 1.5, 2),
        "total_SRAM_KB": 192 * 108,
        "source": "arXiv:2205.14135 §2.1 (paper.md:L45): 'an order of magnitude faster than HBM but many orders of magnitude smaller in size'",
    }

    # ---- 对照: vLLM 主路径一次调用,0 张物化 ----
    out["vllm_call_site_contrast"] = {
        "callsite": "vllm/v1/attention/backends/flash_attn.py:L1041-L1066",
        "materialized_NxN_tables": 0,
        "note": "一次 flash_attn_varlen_func 调用 = 一个融合 kernel,S/P 只在 SRAM 里以块存在",
    }

    p = Path(__file__).parent / "ch20_m01_materialization.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"OK {p}")


if __name__ == "__main__":
    main()
