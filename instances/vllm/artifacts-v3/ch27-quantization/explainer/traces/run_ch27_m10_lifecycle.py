"""ch27-m10 驱动脚本 —— 带宽账(roofline)与量化- kernel 耦合总账的数值轨迹。

跑法(host, 纯 CPU numpy): python run_ch27_m10_lifecycle.py
输出: ch27_m10_lifecycle.json(与本脚本同目录)

素材对应 dossier 机制 ch27-m10(figure-only,F10 伏笔回收),论文侧
arXiv:2306.00978 §4.1(4090:165 TFLOPS/1TB/s,FP16 生成阶段算术强度≈1,
W4 → 4 FLOPs/Byte)+ arXiv:2210.17323 §5 Table 6/§6(提速来自带宽、
自认无计算收益)。vLLM 侧数字(算力门槛/kernel 优先级表)为源码常量,
出处以 file:L 标注,不在本 trace 重复计算。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
from roofline import (  # noqa: E402
    decode_arithmetic_intensity,
    is_memory_bound,
    matvec_flops,
    matvec_weight_bytes,
)


def r(v, nd=2):
    return round(float(v), nd)


def main():
    out = {}

    # ---- ① 算术强度 16/bits:FP16/INT8/INT4 ----
    rows = []
    for bits in (16, 8, 4):
        it = decode_arithmetic_intensity(bits)
        rows.append(
            {
                "weight_bits": bits,
                "arithmetic_intensity_flops_per_byte": r(it, 1),
                "memory_bound_on_4090": bool(is_memory_bound(it)),
                "bound_4090": 165,
            }
        )
    out["arithmetic_intensity"] = {
        "formula": "强度 = FLOPs/权重字节 = 2/(bits/8) = 16/bits(batch-1 矩阵-向量积)",
        "rows": rows,
        "roofline_4090": {"peak_flops_tflops": 165, "peak_bw_tbps": 1, "threshold": 165},
        "paper_quotes": {
            "awq": "any workload with arithmetic intensity less than 165 is memory bounded on 4090;FP16 generation stage ≈ 1;AWQ reduces the weight memory by four times(arXiv:2306.00978 §4.1, paper-awq.md:L266)",
            "gptq_kernel": "quantized-matrix full-precision-vector product kernel ... dynamically dequantizing weights when needed;almost all of the speedup is due to our kernels(arXiv:2210.17323 §5, paper.md:L483)",
            "gptq_limitation": "speedups from reduced memory movement, and does not lead to computational reductions(arXiv:2210.17323 §6, paper.md:L508)",
        },
    }

    # ---- ② 一个 4096×4096 线性层的 decode 搬运账 ----
    d = 4096
    out["layer_traffic_4096x4096"] = {
        "layer_flops_per_token": matvec_flops(d, d),
        "weight_bytes_fp16": matvec_weight_bytes(d, d, 16),
        "weight_bytes_int4": matvec_weight_bytes(d, d, 4),
        "traffic_reduction_factor": r(
            matvec_weight_bytes(d, d, 16) / matvec_weight_bytes(d, d, 4), 1
        ),
        "paper_table6": {
            "a6000": "589ms -> 130ms = 4.53x(3-bit OPT-175B, batch 1, len 128)",
            "a100": "230ms -> 71ms = 3.24x",
            "source": "arXiv:2210.17323 §5 Table 6(paper.md:L488-L489 逐字)",
        },
    }

    # ---- ③ vLLM 三重门(源码常量,出处 file:L) ----
    out["vllm_gates"] = {
        "gate_1_config_min_capability": {
            "what": "config 期算力硬门:get_min_capability 不满足直接 ValueError(docstring 自述门槛 due to the custom CUDA kernels)",
            "where": "vllm/model_executor/layers/quantization/base_config.py:L119-L126 + vllm/config/vllm.py:L706-L739",
        },
        "gate_2_constructor_choose_kernel": {
            "what": "构造期 choose_mp_linear_kernel 按平台优先级表逐个过三道闸(黑名单/算力/can_implement),第一个全过的中选,构造期一次定死",
            "cuda_priority": [
                "CutlassW4A8",
                "Machete",
                "AllSpark",
                "Marlin",
                "Conch",
                "Exllama",
                "TritonW4A16",
                "Humming",
            ],
            "where": "vllm/model_executor/kernels/linear/__init__.py:L411-L439(表)+ L747-L789(循环)",
            "same_checkpoint_different_gpu": {
                "machete_min_capability": 90,
                "marlin_min_capability": 75,
                "h100_capability": 90,
                "a100_capability": 80,
                "verdict": "H100(cap 90)过 Machete 门 --> Machete;A100(cap 80 < 90)被算力闸拦下 --> 落到 Marlin(75 <= 80)——同一 GPTQ 检查点两张卡两个 kernel",
                "min_cap_sources": "machete.py:L26-L27(90) / marlin.py:L37-L38(75)",
            },
        },
        "gate_3_loadtime_repack": {
            "what": "装载期 process_weights_after_loading:全模型遍历,把检查点格式重排成 kernel 格式(Marlin repack/FP8 转置合并/NVFP4 alpha 预计算)",
            "where": "vllm/model_executor/model_loader/utils.py:L100-L122",
        },
        "gate_4_compile_time_ops": {
            "what": "编译期算子选择:块状权重强制 +quant_fp8 手工算子;query 量化刻意用普通 torch 算子让 compile 融合;fuse_norm_quant/fuse_act_quant/fuse_attn_quant 按 -O 档位与硬件谓词启用",
            "where": "vllm/config/vllm.py:L1253-L1268 + vllm/model_executor/layers/attention/attention.py:L514-L524 + vllm/config/vllm.py:L275-L290",
            "ch19_payoff": "F10 伏笔(ch19 埋:planted=19, 本章收:paid=27)——三影子的选择标准都是『哪条路对整条编译管线更快』",
        },
    }

    p = Path(__file__).with_name("ch27_m10_lifecycle.json")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
