"""ch24-m01 驱动脚本 —— KV 显存账:每 token 每层缓存什么、付多少。

跑法(host, 纯 CPU): python run_ch24_m01_kv_account.py
输出: ch24_m01_kv_account.json(与本脚本同目录)

素材对应 dossier 机制 ch24-m01(KV 显存账),论文出处:
- arXiv:2405.04434 §2.1.1 尾句(MHA needs to cache 2n_h·d_h·l elements)、
  §2.1.4 Table 1(MHA/MQA/GQA/MLA 四行)、§3.1.2 超参、Abstract(93.3%)、App D.2(14%/4%);
- arXiv:2305.13245 §1(带宽瓶颈句);
- deepread/memory-kv.json why_chains[1](Llama-2-7B 0.5MB/token 算例);
- deepread/model-sample.json why_chains[10](DSV3 若 MHA 128×(192+128)=40960 口径)。

vLLM 侧页字节公式(kv_cache_interface.py:L212-L226 与 L388-L426)在本机无 vLLM 包,
以同式算术镜像复算并逐字段标注 mirror_of——数值与源码公式逐项一致(纯乘法)。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
from kv_cache_table import (  # noqa: E402
    deepseek_v2_mla_hyperparams,
    mha_kv_cache_per_token,
    mla_kv_cache_per_token,
    gqa_kv_cache_per_token,
    mqa_kv_cache_per_token,
)


def r(v, nd=4):
    return round(float(v), nd)


def main():
    out = {}

    hp = deepseek_v2_mla_hyperparams()
    out["paper_hyperparams_eq_refs"] = {"note": "arXiv:2405.04434 §3.1.2;V3 §4.2 同值(V3-1/V3-3 蓝框)", "eqs": [41, 38, 40]}
    out["dsv3_hyperparams"] = hp
    out["display_names"] = {
        "Llama-2-7B": "账例 A 的模型名(32 层/32 KV 头/128 维,FP16)",
        "GQA-8": "Table 1 GQA 行在本例的取值(8 组)",
    }

    # ---- 口径 A:Llama-2-7B FP16 整模型每 token 字节(deepread/memory-kv 卡算例) ----
    layers, kv_heads, head_dim, bytes_per_elem = 32, 32, 128, 2
    per_token_bytes = 2 * layers * kv_heads * head_dim * bytes_per_elem
    out["llama2_7b_fp16_per_token"] = {
        "layers": layers, "kv_heads": kv_heads, "head_dim": head_dim,
        "bytes_per_elem_fp16": bytes_per_elem,
        "per_token_bytes": per_token_bytes,
        "per_token_bytes_expanded": "2*32*32*128*2 = 524288",
        "per_token_mb": r(per_token_bytes / (1024 * 1024)),          # 0.5
        "per_token_mb_rounded": 0.5,
        "provenance": "deepread/memory-kv.json why_chains[1] 算例(2(K,V)×32 层×32 kv_head×128 head_dim×2B)",
        "gpu_note": "24GB 卡权重吃 14GB 后池只剩约 8GB(同卡原句)",
    }

    # ---- 口径 B:DSV3 超参下的 Table 1 四行(每 token 每层元素,l=1) ----
    n_h, d_h, d_c, d_hR = hp["n_h"], hp["d_h"], hp["d_c"], hp["d_h_R"]
    t1 = {
        "MHA": mha_kv_cache_per_token(n_h, d_h),
        "MQA": mqa_kv_cache_per_token(d_h),
        "GQA_8": gqa_kv_cache_per_token(8, d_h),
        "MLA": mla_kv_cache_per_token(d_c, d_hR),
    }
    out["table1_dsv3_per_layer"] = {
        "expand": {"MHA": "2*128*128", "MQA": "2*1*128", "GQA_8": "2*8*128", "MLA": "512+64"},
        **t1,
        "mla_percent_of_mha": t1["MLA"] / t1["MHA"] * 100,            # 1.7578125(全精度,勿舍入)
        "mla_percent_of_mha_rounded": 1.75,
        "mha_over_mla": r(t1["MHA"] / t1["MLA"], 2),                  # 56.89
    }

    # 口径 B':deepread/model-sample 卡的『若 MHA』口径(K 逐头按 MLA 的 qk 192 维计)
    out["mha_if_dsv3_deepread_account"] = {
        "per_head_k_width_assumed": 192,   # 128 nope + 64 rope(MLA 每头 qk 宽)
        "per_head_v_width_assumed": 128,
        "elements": n_h * (192 + 128),      # 40960
        "expand": "128*(192+128)",
        "provenance": "deepread/model-sample.json why_chains[10] old_design 原句",
        "vs_table1": "Table 1 公式 2*n_h*d_h=32768 假设 K/V 每头都是 d_h=128;两口径差异=K 每头按 192 还是 128 计",
    }

    # ---- 口径警示:三个百分比不可混用 ----
    out["three_percent_accounts"] = {
        "abstract_93_3_percent": {
            "value": 93.3, "base": "相对 DeepSeek 67B(不同头配置)的 KV cache 削减",
            "source": "arXiv:2405.04434 Abstract('reduces the KV cache by 93.3%')"},
        "table1_1_75_percent": {
            "value": 1.75, "base": "按 Table 1 公式+§3.1.2 超参的算术:576/32768=1.7578125%",
            "source": "arXiv:2405.04434 §2.1.4 Table 1 + §3.1.2(算术,非论文原句)"},
        "appD2_14_4_percent": {
            "value_small_moe": 14, "value_large_moe": 4,
            "base": "MoE 模型整体实测口径(16B/250B 两档 MoE 对齐实验)",
            "source": "arXiv:2405.04434 App D.2('14% for small MoE models and 4% for large MoE models')"},
        "abstract_5_76x": {
            "value": 5.76, "base": "最大生成吞吐倍数(相对 DeepSeek 67B,含 6-bit KV 量化的合并效果)",
            "source": "arXiv:2405.04434 Abstract/§3.2.3"},
    }

    # ---- 感受数字:DSV3 配置 4096-token 上下文的 KV cache 总量(fp16 元素 2B) ----
    ctx, L = 4096, 60
    mha_total = t1["MHA"] * bytes_per_elem * L * ctx
    mla_total = t1["MLA"] * bytes_per_elem * L * ctx
    out["context_4096_dsv3_total"] = {
        "context_tokens": ctx, "layers_v2": L, "bytes_per_elem": bytes_per_elem,
        "mha_if_total_bytes": mha_total, "mha_if_total_gib": r(mha_total / 2**30, 2),
        "mla_total_bytes": mla_total, "mla_total_mib": r(mla_total / 2**20, 1),
        "ratio": r(mha_total / mla_total, 2),
    }

    # ---- vLLM 页字节公式镜像(引擎账本接入点,ch14 显存账的输入) ----
    block_size, dtype_bytes = 16, 2
    out["vllm_page_bytes_mirror"] = {
        "note": "本机无 vLLM 包——按源码公式逐项乘法镜像,非 import 实跑;公式逐字对照见 anchors",
        "anchors_cited": {
            "normal_attention": "vllm/v1/kv_cache_interface.py:L212-L226(2 × block_size × num_kv_heads × head_dim × dtype_bytes)",
            "mla_attention": "vllm/v1/kv_cache_interface.py:L388-L426(去 2 因子:storage_block_size × num_kv_heads × head_dim × dtype)",
            "mla_spec": "vllm/model_executor/layers/attention/mla_attention.py:L393(head_size=kv_lora_rank+qk_rope_head_dim)、L397(num_kv_heads=1)、L1140-L1152(get_kv_cache_spec)",
            "lines_cited": [212, 226, 388, 426, 393, 397, 1140, 1152],
        },
        "block_size": block_size, "fp16_dtype_bytes": dtype_bytes,
        "page_bytes_dsv3_if_mha_128heads": 2 * block_size * 128 * 128 * dtype_bytes,   # 1048576
        "page_bytes_gqa8_layer": 2 * block_size * 8 * 128 * dtype_bytes,               # 65536
        "page_bytes_mla_layer": block_size * 1 * 576 * dtype_bytes,                    # 18432
        "per_token_bytes_gqa8_layer": 2 * 8 * 128 * dtype_bytes,                       # 4096
        "per_token_bytes_mla_layer": 576 * dtype_bytes,                                # 1152
    }

    p = Path(__file__).resolve().parent / "ch24_m01_kv_account.json"
    with open(p, "w", newline="\n", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
