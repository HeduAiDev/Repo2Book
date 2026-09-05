"""ch27-m02 驱动脚本 —— 量化粒度谱(per-tensor/per-token/per-channel)的数值轨迹。

跑法(host, 纯 CPU numpy): python run_ch27_m02_granularity.py
输出: ch27_m02_granularity.json(与本脚本同目录)

素材对应 dossier 机制 ch27-m02(figure-only),论文 arXiv:2211.10438 §2
Figure 3(粒度定义)+ §3 Table 1(per-channel 激活量化精度最好但 INT8 GEMM
不认)。矩阵 4 token × 4 通道:token 0 的激活 ~100×其他 token(模拟 SmoothQuant
Fig.4 的离群量级),固定种子可复算。per-tensor 一把尺子:小 token 的值全被
round 到 0 附近;per-token 每 token 一把:小 token 完好。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
from uniform_quant import (  # noqa: E402
    quantize_per_channel,
    quantize_per_tensor,
    quantize_per_token,
)


def r(v, nd=4):
    return round(float(v), nd)


def main():
    out = {}

    rng = np.random.default_rng(1)
    big = rng.standard_normal((1, 4)) * 100.0
    small = rng.standard_normal((3, 4)) * 1.0
    x = np.vstack([big, small])

    q_t, d_t = quantize_per_tensor(x, 8)
    q_k, d_k = quantize_per_token(x, 8)
    q_c, d_c = quantize_per_channel(x, 8)

    err_t = np.abs(x - q_t * d_t)
    err_k = np.abs(x - q_k * d_k[:, None])
    err_c = np.abs(x - q_c * d_c[None, :])

    out["params"] = {
        "shape": "4 tokens x 4 channels",
        "num_bits": 8,
        "note": "token 0 激活 ~100x 其他 token(SmoothQuant Fig.4 离群量级;固定种子 rng(1))",
        "token0_absmax": r(np.abs(big).max(), 1),
        "small_tokens_absmax": r(np.abs(small).max(), 1),
    }

    out["per_tensor"] = {
        "delta_shared_by_all": r(d_t),
        "delta_formula": "全矩阵 absmax/127",
        "small_tokens_mean_abs_err": r(err_t[1:].mean()),
        "small_tokens_unique_codes": int(np.unique(q_t[1:]).size),
        "big_token_mean_abs_err": r(err_t[0].mean()),
    }
    out["per_token"] = {
        "deltas_per_token": [r(v) for v in d_k],
        "small_tokens_mean_abs_err": r(err_k[1:].mean()),
        "small_tokens_unique_codes": int(np.unique(q_k[1:]).size),
        "big_token_mean_abs_err": r(err_k[0].mean()),
        "scale_count": 4,
        "scale_dim": "外维 T(每个 token 一把尺子)",
    }
    out["per_channel"] = {
        "deltas_per_channel": [r(v) for v in d_c],
        "small_tokens_mean_abs_err": r(err_c[1:].mean()),
        "small_tokens_unique_codes": int(np.unique(q_c[1:]).size),
        "scale_count": 4,
        "scale_dim": "内维 C_i(激活侧挂在 GEMM 缩减维——INT8 Tensor Core 不认,SmoothQuant §3 Table 1 灰色行)",
    }
    out["comparison"] = {
        "err_ratio_per_tensor_over_per_token_small_tokens": r(
            err_t[1:].mean() / err_k[1:].mean(), 1
        ),
        "err_ratio_per_tensor_over_per_channel_small_tokens": r(
            err_t[1:].mean() / err_c[1:].mean(), 1
        ),
        "extra_storage_per_token": "每 token 1 个 fp16 scale = 4x4 矩阵共 4 个(额外 4x2=8 字节,权重 16 字节的 50%)",
        "paper_table1_opt175b_avg_acc": {
            "fp16": 71.6,
            "int8_per_tensor": 32.3,
            "int8_per_token": 31.7,
            "int8_per_channel": 71.4,
            "source": "arXiv:2211.10438 §3 Table 1(paper-smoothquant.md:L39-L42,逐字转录;per-channel 精度保住但 INT8 GEMM 不可行)",
        },
        "vllm_groupshape_vocab": {
            "PER_TENSOR": "vllm/model_executor/layers/quantization/utils/quant_utils.py:L69(整张矩阵一把)",
            "PER_TOKEN": "quant_utils.py:L70(外维 token,QuantFP8 dynamic per-token 即此档)",
            "PER_CHANNEL": "quant_utils.py:L71(外维输出通道,权重侧标准做法)",
            "BLOCK_128x128": "quant_utils.py:L356 注释(DeepSeek 128x128 块)",
            "GROUP_1x128": "quant_utils.py:L357 注释(逐 token 逐组,DeepSeek 式)",
        },
    }

    p = Path(__file__).with_name("ch27_m02_granularity.json")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
