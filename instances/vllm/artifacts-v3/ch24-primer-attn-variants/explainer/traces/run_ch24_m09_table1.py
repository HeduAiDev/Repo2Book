"""ch24-m09 驱动脚本 —— KV cache 总账 Table 1:四种机制对决 + 2.25 组换算 + 消融口径。

跑法(host, 纯 CPU): python run_ch24_m09_table1.py
输出: ch24_m09_table1.json(与本脚本同目录)

素材对应 dossier 机制 ch24-m09,论文 arXiv:2405.04434 §2.1.4 Table 1
(表体为包内重建,引用保留该标注)+ §3.1.2 超参 + App D.1/D.2 消融结论。
GQA 组数轴扫一遍:MLA 576 落在 G=2(512)与 G=4(1024)之间 → 等效 2.25 组。
页字节两公式(普通层带 2 因子 vs MLA 层无)以算术镜像复算并标注 mirror_of。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
from kv_cache_table import (  # noqa: E402
    deepseek_v2_mla_hyperparams,
    equivalent_gqa_groups,
    gqa_kv_cache_per_token,
    mha_kv_cache_per_token,
    mla_kv_cache_per_token,
    mqa_kv_cache_per_token,
)


def r(v, nd=6):
    return round(float(v), nd)


def main():
    out = {}

    hp = deepseek_v2_mla_hyperparams()
    n_h, d_h, d_c, d_hR = hp["n_h"], hp["d_h"], hp["d_c"], hp["d_h_R"]
    out["params"] = {
        "hyperparams": hp,
        "eqs": [41],
        "table1_note": "Table 1 表体为包内重建(paper-mla.md:L103 声明;MHA 行出自 §2.1.1 原句、MLA 行出自 §2.1.3 原句、GQA/MQA 行由 factor-H 直推)——引用时保留该标注",
    }
    out["display_names"] = {"GQA-8": "8 组(2048 元素/层)"}
    out["percent_constants"] = {
        "abstract_vs_deepseek_67b": 93.3,
        "table1_arithmetic": 1.75,
        "appD2_small_moe": 14, "appD2_large_moe": 4,
    }

    # ---- 四行总账(DSV3 超参,l=1) ----
    mha = mha_kv_cache_per_token(n_h, d_h)
    mqa = mqa_kv_cache_per_token(d_h)
    gqa8 = gqa_kv_cache_per_token(8, d_h)
    mla = mla_kv_cache_per_token(d_c, d_hR)
    out["table1_dsv3"] = {
        "MHA": mha, "MQA": mqa, "GQA_8": gqa8, "MLA": mla,
        "expand": {"MHA": "2*128*128", "MQA": "2*1*128", "GQA_8": "2*8*128", "MLA": "512+64"},
        "ratio_to_mha": {
            "MHA": 1.0,
            "MQA": mqa / mha,              # 0.0078125 = 1/128(全精度,勿舍入)
            "GQA_8": gqa8 / mha,           # 0.0625 = 1/16
            "MLA": mla / mha,              # 0.017578125 = 1.7578125%
        },
        "mla_percent_of_mha_rounded": 1.75,
        "mha_over_mla": r(mha / mla, 2),   # 56.89
        "with_layers_l60": {"MHA": mha * 60, "MQA": mqa * 60, "GQA_8": gqa8 * 60, "MLA": mla * 60},
    }

    # ---- GQA 组数轴:MLA 576 插在哪里 ----
    sweep = {f"G{g}": gqa_kv_cache_per_token(g, d_h) for g in (1, 2, 4, 8, 16, 32, 64, 128)}
    out["gqa_group_sweep"] = {
        **sweep,
        "MLA": mla,
        "reading": "576 落在 G2=512 与 G4=1024 之间——等效 2.25 组",
    }

    # ---- 2.25 组换算(Table 1 caption 原句) ----
    out["equivalent_groups"] = {
        "formula": "(d_c+d_h^R)/(2*d_h) = (4*d_h + d_h/2)/(2*d_h)",
        "value": equivalent_gqa_groups(d_c, d_hR, d_h),      # 2.25
        "d_c_is_4dh": 4 * d_h,                                # 512
        "d_hR_is_dh_half": d_h // 2,                          # 64
        "caption_quote": "Table 1 caption 'its KV cache is equal to GQA with only 2.25 groups, but its performance is stronger than MHA'",
    }

    # ---- 消融口径:D.1 分组质量罚 vs D.2 MLA 绕开 ----
    out["ablation_accounts"] = {
        "D1_dense_7b": {
            "setting": "7B dense、1.33T tokens、对齐参数量(调层数)",
            "conclusion": "MHA demonstrates significant advantages over GQA and MQA(§D.1 原句)——砍 KV 头确实掉质量,分组路线的代价上限",
        },
        "D2_moe": {
            "setting": "16B(2.4B 激活)/250B(21B 激活)两档 MoE 对齐实验",
            "conclusion": "MLA shows better performance than MHA;KV cache 仅为其 14%(小)/4%(大)(§D.2 原句)",
            "cache_percent_small": 14, "cache_percent_large": 4,
        },
        "three_percent_warning": {
            "93_3": "Abstract:相对 DeepSeek 67B(不同头配置)的口径",
            "1_75": "Table 1 公式+§3.1.2 超参的算术(576/32768),本章主用口径",
            "14_4": "App D.2 MoE 对齐实验实测口径——三个数不可混用",
        },
    }

    # ---- 页字节两公式镜像(block=16、fp16;ch14 显存账本的输入) ----
    block, dt = 16, 2
    out["page_bytes_mirror"] = {
        "note": "本机无 vLLM 包——同式算术镜像(mirror_of):普通层 2×block×num_kv_heads×head_dim×dtype vs MLA 层去 2 因子",
        "anchors_cited": {
            "normal": "vllm/v1/kv_cache_interface.py:L212-L226(AttentionSpec.real_page_size_bytes)",
            "mla": "vllm/v1/kv_cache_interface.py:L388-L426(MLAAttentionSpec.real_page_size_bytes,storage_block_size×num_kv_heads×head_dim×dtype)",
            "spec_report": "vllm/model_executor/layers/attention/mla_attention.py:L1140-L1152(get_kv_cache_spec 自报 num_kv_heads=1/head_size=576)",
            "lines_cited": [212, 226, 388, 426, 1140, 1152],
        },
        "block_size": block, "fp16_bytes": dt,
        "page_bytes": {
            "mha_if_128kvheads_layer": 2 * block * 128 * 128 * dt,   # 1048576
            "gqa8_layer": 2 * block * 8 * 128 * dt,                  # 65536
            "mqa_layer": 2 * block * 1 * 128 * dt,                   # 4096
            "mla_layer": block * 1 * 576 * dt,                       # 18432(无 2 因子)
        },
        "per_token_bytes": {
            "mha_if": 2 * 128 * 128 * dt,    # 32768
            "gqa8": 2 * 8 * 128 * dt,        # 4096
            "mqa": 2 * 1 * 128 * dt,         # 256
            "mla": 576 * dt,                 # 1152
        },
    }

    p = Path(__file__).resolve().parent / "ch24_m09_table1.json"
    with open(p, "w", newline="\n", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
