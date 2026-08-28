# run_m2.py — m2 memory_profiling 三类显存与峰值账 驱动脚本
# 参数 = mem_utils.py docstring 的量化例（官方 oracle）：他进程 1GiB、权重 2GiB、
# 激活峰 2GiB（gc 后 1GiB 常驻）、非 torch（NCCL+后端缓冲）1GiB → non_kv = 5GiB。
# 走真 memory_profiling 上下文（before/after 快照差 + total_consumed +
# transient_peak_headroom 两式）；设备读数按 HOST SEAM 注入。
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from implementation.mem_utils import MemorySnapshot, memory_profiling

GiB = 1 << 30
TOTAL = 10 * GiB

OUT = {"docstring_oracle": "vllm/utils/mem_utils.py memory_profiling docstring 量化例"}


def snap(free, peak=0, allocated=0, reserved=0, non_torch=0):
    s = MemorySnapshot(device=torch.device("cpu"), auto_measure=False)
    s.free_memory = free
    s.total_memory = TOTAL
    s.torch_peak = peak
    s.torch_allocated = allocated
    s.torch_memory = reserved
    s.cuda_memory = TOTAL - free
    s.non_torch_memory = non_torch
    return s


before_create = snap(free=9 * GiB)                                  # cat1 他进程 1GiB
before_profile = snap(free=6 * GiB,                                 # 权重 2 + NCCL 1
                      peak=2 * GiB, allocated=2 * GiB,
                      reserved=2 * GiB, non_torch=int(1.0 * GiB))
after_profile = snap(free=5 * GiB,                                  # 峰后 gc：常驻 +1、cat3 = 1
                     peak=4 * GiB, allocated=3 * GiB,
                     reserved=3 * GiB, non_torch=int(1.0 * GiB))

SEQ = iter([before_profile, after_profile])


def fake_measure(self):
    src = next(SEQ)
    for f in ("free_memory", "total_memory", "torch_peak", "torch_allocated",
              "torch_memory", "cuda_memory", "non_torch_memory"):
        setattr(self, f, getattr(src, f))


class FakeAccel:
    def empty_cache(self):
        pass

    def reset_peak_memory_stats(self, device):
        pass


MemorySnapshot.measure = fake_measure
torch.accelerator = FakeAccel()

with memory_profiling(before_create, weights_memory=2 * GiB) as result:
    pass  # profile_run 的位置（真实为 dummy 前向；本驱动只关心记账算术）

OUT["timeline"] = {
    "before_create": {"cat1_non_vllm_gib": 1, "cat2_torch_gib": 0, "cat3_non_torch_gib": 0,
                      "free_gib": 9},
    "before_profile": {"cat1_non_vllm_gib": 1, "cat2_torch_gib": 2, "cat3_non_torch_gib": 1,
                       "free_gib": 6},
    "during_peak": {"cat1_non_vllm_gib": 1, "cat2_torch_gib": 4, "cat3_non_torch_gib": 1,
                    "peak_activation_gib": 2},
    "after_profile": {"cat1_non_vllm_gib": 1, "cat2_torch_gib": 3, "cat3_non_torch_gib": 1,
                      "free_gib": 5},
}
OUT["result"] = {
    "weights_memory_bytes": result.weights_memory,
    "weights_memory_gib": round(result.weights_memory / GiB, 2),
    "total_consumed_bytes": result.total_consumed,
    "total_consumed_gib": round(result.total_consumed / GiB, 2),
    "torch_peak_increase_bytes": result.torch_peak_increase,
    "torch_peak_increase_gib": round(result.torch_peak_increase / GiB, 2),
    "non_torch_increase_bytes": result.non_torch_increase,
    "non_torch_increase_gib": round(result.non_torch_increase / GiB, 2),
    "transient_peak_headroom_bytes": result.transient_peak_headroom,
    "transient_peak_headroom_gib": round(result.transient_peak_headroom / GiB, 2),
    "non_kv_cache_memory_bytes": result.non_kv_cache_memory,
    "non_kv_cache_memory_gib": round(result.non_kv_cache_memory / GiB, 2),
    "composition": {
        "a_weights_gib": 2,
        "b_peak_activation_gib": 2,
        "c_non_torch_gib": 1,
        "sum_gib": 5,
    },
    "identity": "non_kv_cache_memory == total_consumed + transient_peak_headroom",
    "identity_holds": result.non_kv_cache_memory == (
        result.total_consumed + result.transient_peak_headroom
    ),
    "transient_formula": "transient = after.torch_peak - after.torch_allocated "
                         "(gc 后仍要为峰值留的 headroom)",
    "total_consumed_formula": "total_consumed = before_create.free - after.free "
                              "(mem_get_info 口径, 可穿 pluggable allocator)",
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m2.json"), "w",
          encoding="utf-8", newline="\n") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)
print(json.dumps(OUT, ensure_ascii=False, indent=1))
