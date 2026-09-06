# ch29 m12 Triton pivot 截断核与批>=8 分流 — 驱动脚本（GPU 容器实跑）。
# 机制：apply_top_k_top_p 二级分流（topk_topp_sampler.py:L349-L364）——
# GPU 上 HAS_TRITON 且批>=8 走 apply_top_k_top_p_triton（Qrita pivot 核：
# 分位查表+三分搜索逼近 pivot、免排序全词表，topk_topp_triton.py:L856-L958
# wrapper 外部契约：NUM_PROGRAMS=min(SM 数,batch)、GPU BLOCK_SIZE=8192/4096、
# 按设备缓存 buffer 与两张正态分位查找表）；批<8 走 pytorch sort 教学实现。
# worked example = 同一批 6 行 vs 16 行各走哪条路 + 两路掩码语义一致。
# 行为基准：vllm/v1/sample/ops/topk_topp_sampler.py:L349-L364、
# topk_topp_triton.py:L880-L957（真实 v0.27.1 行号）。
import json
import os
import pathlib
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IMPL = pathlib.Path(__file__).resolve().parent.parent.parent / "implementation"
sys.path.insert(0, str(IMPL))
# 注意：本驱动须在 VLLM_USE_FLASHINFER_SAMPLER 默认态跑（与本机制无关，只为
# 与 m13 同环境）；分流谓词只看 HAS_TRITON 与批大小。

import torch

import vllm.v1.sample.ops.topk_topp_sampler as tts
from vllm.triton_utils import HAS_TRITON
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.sample.ops.topk_topp_sampler import (
    apply_top_k_top_p,
    apply_top_k_top_p_pytorch,
)

R = lambda x: round(float(x), 4)
dev = torch.device("cuda")
out = {
    "environment": {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "HAS_TRITON": bool(HAS_TRITON),
        "note": "GPU 容器 vllm/vllm-omni:latest 实跑（host 无 CUDA、CPU 张量无法进 triton 核）",
    }
}

# ── 间谍：记录每次分流走了哪条路（透传真实现）──────────────────────
real_triton = tts.apply_top_k_top_p_triton
real_pytorch = tts.apply_top_k_top_p_pytorch
calls = []


def spy_triton(logits, k, p):
    calls.append({"batch": logits.shape[0], "path": "triton"})
    return real_triton(logits, k, p)


def spy_pytorch(logits, k, p, allow_cpu_sync=False):
    calls.append({"batch": logits.shape[0], "path": "pytorch"})
    return real_pytorch(logits, k, p, allow_cpu_sync)


tts.apply_top_k_top_p_triton = spy_triton
tts.apply_top_k_top_p_pytorch = spy_pytorch

# ── 同一族 logits：批 6 vs 批 16 ───────────────────────────────────
torch.manual_seed(2024)
V = 64
base16 = (torch.randn(16, V, device=dev) * 3.0)  # 拉开差距、避开 pivot 近阈值歧义
k_all = torch.full((16,), 5, dtype=torch.long, device=dev)
p_all = torch.full((16,), 0.9, device=dev)

batch6_logits = base16[:6].clone()
out6 = apply_top_k_top_p(batch6_logits, k_all[:6].clone(), None)
batch16_logits = base16.clone()
out16 = apply_top_k_top_p(batch16_logits, k_all.clone(), None)
survivors_of_6 = [j for j in range(V) if out6[0, j].item() != float("-inf")]

# ── 两路一致性：批 16 同一 logits 分别走 triton 与 pytorch sort ────
py16 = apply_top_k_top_p_pytorch(base16.clone(), k_all.clone(), None)
survivors_triton = [[j for j in range(V) if out16[i, j].item() != float("-inf")] for i in range(16)]
survivors_pytorch = [[j for j in range(V) if py16[i, j].item() != float("-inf")] for i in range(16)]
identical = survivors_triton == survivors_pytorch
both_finite = (out16 > float("-inf")) & (py16 > float("-inf"))
finite_diff = (out16[both_finite] - py16[both_finite]).abs().max().item() if both_finite.any() else 0.0
argmax_kept = all(int(base16[i].argmax()) in survivors_triton[i] for i in range(16))

num_sm = num_compute_units(0)
out["dispatch_batch6_vs_16"] = {
    "vocab": V,
    "logits_scale": "randn×3.0（拉开差距，避开 pivot 近阈值歧义）",
    "batch6": {"path": calls[0]["path"], "top_k": 5, "survivors_row0_count": len(survivors_of_6)},
    "batch16": {"path": calls[1]["path"], "top_k": 5, "survivors_row0": survivors_triton[0]},
    "dispatch_rule": "HAS_TRITON and logits.shape[0] >= 8 → triton pivot 核；否则 pytorch sort（L360-L364）",
    "calls_recorded": calls,
    "claim": "同一族 logits：批 6 走 pytorch sort、批 16 走 Triton pivot 核——分流只看批大小与 Triton 可用性",
}
out["two_paths_same_mask"] = {
    "triton_survivors_row0": survivors_triton[0],
    "pytorch_survivors_row0": survivors_pytorch[0],
    "all_16_rows_survivor_sets_identical": bool(identical),
    "finite_values_max_abs_diff": R(finite_diff),
    "argmax_survives_every_row": bool(argmax_kept),
    "per_row_survivor_count_row0": len(survivors_triton[0]),
    "claim": "批 16 双路对照：Triton pivot 截断与 pytorch sort 的幸存集逐行一致（k=5 恰存 5）、有限值零差、greedy argmax 全部幸存——pivot 是掩码语义等价的免排序实现",
}
out["wrapper_contract"] = {
    "num_sm_compute_units": int(num_sm),
    "NUM_PROGRAMS_batch16": int(min(num_sm, 16)),
    "NUM_PROGRAMS_formula": "min(num_compute_units(device), batch_size)（L913-L914）",
    "gpu_block_size": 8192,
    "gpu_block_size_trunc": 4096,
    "gpu_block_size_note": "CPU 256/128、XPU 4096/2048、GPU 8192/4096 三分支（L940-L946）——本机为 GPU 位",
    "k_dummy_pointer_when_p_only": "k 为 None 时 k_ptr=logits 假指针（L896-L898 不会被读）",
    "cache": "buffer 与两张正态分位查找表按 (device,dtype,vocab)/device 缓存（L917-L935）",
}

# ── top-p 单独走 triton（批 16）：幸存集 vs pytorch 对照 ────────────
out16_p = apply_top_k_top_p(base16.clone(), None, p_all.clone())
py16_p = apply_top_k_top_p_pytorch(base16.clone(), None, p_all.clone())
surv_p_t = [[j for j in range(V) if out16_p[i, j].item() != float("-inf")] for i in range(16)]
surv_p_p = [[j for j in range(V) if py16_p[i, j].item() != float("-inf")] for i in range(16)]
out["topp_triton_batch16"] = {
    "top_p": 0.9,
    "path": calls[-1]["path"],
    "row0_survivors_triton": surv_p_t[0],
    "row0_survivors_pytorch": surv_p_p[0],
    "rows_identical": bool(surv_p_t == surv_p_p),
    "row0_count": len(surv_p_t[0]),
    "claim": "top-p 单参数同样批>=8 进 Triton 核；0.9 核大小因行而异（本例行 0 留 4 个）——pivot 近似对开分差距的分布与 sort 逐行一致",
}

out["table_rows_echo"] = [
    ["分流谓词", "HAS_TRITON=True（GPU 容器）", "批 6 < 8 → pytorch sort", "批 16 ≥ 8 → triton pivot 核", "分流只看批大小（L360-L364）"],
    ["批 6 走 sort", f"top_k=5, V={V}", f"行 0 幸存 {len(survivors_of_6)} 个", f"path={calls[0]['path']}", "教学主实现路径"],
    ["批 16 走 triton", f"top_k=5, V={V}", f"行 0 幸存 {len(survivors_triton[0])} 个", f"path={calls[1]['path']}", f"NUM_PROGRAMS=min({int(num_sm)},16)={int(min(num_sm, 16))}"],
    ["两路一致性(k=5)", f"幸存集逐行一致={bool(identical)}", f"有限值 max|差|={R(finite_diff)}", f"argmax 全行幸存={bool(argmax_kept)}", "pivot=免排序的等价掩码"],
    ["top-p 单参数批 16", "p=0.9 → triton", f"行 0 留 {len(surv_p_t[0])} 个", f"与 sort 一致={bool(surv_p_t == surv_p_p)}", "k 缺省用 dummy 指针"],
    ["wrapper 外部契约", f"GPU BLOCK_SIZE=8192/4096", f"SM 数（compute units）={int(num_sm)}", "buffer/查表按设备缓存", "断言 fp32 2D 连续"],
]

print(json.dumps(out, ensure_ascii=False, indent=1))
with open(pathlib.Path(__file__).parent / "ch29_m12_triton_dispatch.json", "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
