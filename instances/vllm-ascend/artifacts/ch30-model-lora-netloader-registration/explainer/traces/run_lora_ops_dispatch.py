#!/usr/bin/env python3
"""ch30 机制 lora-ops-dispatch 教学轨迹驱动。

忠实复刻两处控制流/算账:
  (1) vllm_ascend/lora/punica_npu.py:L31-L44 —— __init__ 里 device/rank 二选一绑算子:
      get_ascend_device_type()==310P 或 max_lora_rank>=128 -> vLLM torch_ops(通用);
      否则 -> vllm_ascend.lora.lora_ops(NPU 自定义 kernel 薄壳)。
  (2) shrink/expand 低秩 FLOPs 算账(dossier.theory / narrative L316-L342):
      shrink u=xA: 2*T*d*r; expand y+=uB*s: 2*T*r*o; 两步合计 2*T*r*(d+o);
      满秩直算 ΔW=BA 的 2*T*d*o; 省倍数 = d*o / (r*(d+o))。

真源码 import vllm/vllm_ascend, host 无 NPU 栈; 此处纯 Python 复刻布尔分支 + 算术,
控制流/公式一字对齐, host python3 直接跑。
"""
import json

_310P = "310P"      # 对位 AscendDeviceType._310P
FALLBACK_THRESHOLD = 128    # max_lora_rank >= 128 退回通用实现的阈值


def bind_ops(device_type: str, max_lora_rank: int) -> str:
    """SOURCE: vllm_ascend/lora/punica_npu.py:L31-L44 —— 二选一绑定。"""
    if device_type == _310P or max_lora_rank >= FALLBACK_THRESHOLD:
        return "vllm.lora.ops.torch_ops"          # PyTorch-native, 通用/稳
    return "vllm_ascend.lora.lora_ops"            # 昇腾 NPU C++ kernel 薄壳


def lowrank_flops(T: int, d: int, o: int, r: int) -> dict:
    """SOURCE: dossier.theory —— shrink/expand 两步 FLOPs 与省倍数。"""
    shrink = 2 * T * d * r
    expand = 2 * T * r * o
    two_step = shrink + expand           # = 2*T*r*(d+o)
    full = 2 * T * d * o                 # 满秩 ΔW=BA 直算
    saving = full // two_step
    return {
        "T": T, "d": d, "o": o, "r": r, "d_plus_o": d + o,
        "shrink_flops": shrink, "expand_flops": expand,
        "two_step_flops": two_step, "full_flops": full,
        "saving_ratio": saving,
    }


if __name__ == "__main__":
    # ---- (2) FLOPs 算账: d=o=4096, 对比 r=16(常规) 与 r=128(阈值) ----
    T, d, o = 1, 4096, 4096
    flops_r16 = lowrank_flops(T, d, o, 16)
    flops_r128 = lowrank_flops(T, d, o, 128)

    # ---- (1) 三个 (device, rank) 场景走绑定分支, 与 FLOPs 挂钩 ----
    scenarios = [
        {"name": "910B 常规 rank", "device_type": "910B", "max_lora_rank": 16,
         "flops": flops_r16},
        {"name": "910B 大 rank",  "device_type": "910B", "max_lora_rank": 128,
         "flops": flops_r128},
        {"name": "310P 推理卡",    "device_type": "310P", "max_lora_rank": 16,
         "flops": flops_r16},
    ]
    for s in scenarios:
        s["bound_ops"] = bind_ops(s["device_type"], s["max_lora_rank"])

    trace = {
        "fallback_threshold": FALLBACK_THRESHOLD,
        "flops_r16": flops_r16,
        "flops_r128": flops_r128,
        "scenarios": scenarios,
    }
    print(json.dumps(trace, ensure_ascii=False, indent=2))
