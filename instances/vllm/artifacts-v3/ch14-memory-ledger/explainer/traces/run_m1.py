# run_m1.py — m1 启动三步定账 驱动脚本（explainer 用）
# 驱动精简版跑通三步：request_memory（预算）→ determine_available_memory
# （memory_profiling 峰值账 + cudagraph 估计 + 一行减法）→ get_num_blocks /
# get_kv_cache_configs（字节换块数 + 护栏 + 容量/并发）。
# 参数刻意选成：12.5GiB 假想卡 × util 0.8 = 10GiB 预算；non_kv 1.5GiB；
# cudagraph 估计 0.5GiB → available_kv 恰好 8GiB —— 落到 dossier theory 锚例
# （Llama-2-7B、page 256KiB、32 层 → 1024 块、容量 16384 token、并发 4×）。
# 设备读数（MemorySnapshot.measure / torch.accelerator）按 HOST SEAM 注入，
# 走的算术全部是精简版真源码路径。
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from implementation.cache import CacheConfig
from implementation.config import (
    CUDAGraphMode,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
)
from implementation import gpu_worker as gw
from implementation.gpu_worker import Worker
from implementation.kv_cache_interface import FullAttentionSpec
from implementation.kv_cache_utils import (
    get_kv_cache_capacity,
    get_kv_cache_configs,
    get_max_concurrency_for_kv_cache_config,
    get_num_blocks,
)
from implementation.mem_utils import MemorySnapshot, format_gib
from implementation.worker_utils import request_memory

GiB = 1 << 30

TOTAL = int(12.5 * GiB)      # 假想 12.5GiB 卡
UTIL = 0.8                   # requested = 10GiB（整）
OUT = {}


def snap(free, total, peak=0, allocated=0, reserved=0, non_torch=0):
    s = MemorySnapshot(device=torch.device("cpu"), auto_measure=False)
    s.free_memory = free
    s.total_memory = total
    s.torch_peak = peak
    s.torch_allocated = allocated
    s.torch_memory = reserved
    s.cuda_memory = total - free
    s.non_torch_memory = non_torch
    return s


# ---------------------------------------------------------------- 第 1 步：预算
cache_cfg = CacheConfig(gpu_memory_utilization=UTIL, enable_prefix_caching=False)
before_create = snap(free=12 * GiB, total=TOTAL, non_torch=int(0.5 * GiB))  # 他进程 0.5GiB
requested = request_memory(before_create, cache_cfg)
OUT["step1_request_memory"] = {
    "total_memory_bytes": TOTAL,
    "total_memory_gib": 12.5,
    "gpu_memory_utilization": UTIL,
    "requested_memory_bytes": requested,
    "requested_memory_gib": round(requested / GiB, 2),
    "formula": "ceil(total_memory x gpu_memory_utilization)",
}

# free 不足 → 直接 raise（预算先于一切）
try:
    request_memory(snap(free=5 * GiB, total=TOTAL), cache_cfg)
    OUT["step1_free_too_low"] = {"raised": False}
except ValueError as e:
    OUT["step1_free_too_low"] = {
        "free_memory_gib": 5,
        "raised": True,
        "error": "Free memory on device ... is less than desired GPU memory "
                 f"memory utilization ({UTIL}, {format_gib(requested)} GiB). "
                 "Decrease GPU memory utilization or reduce GPU memory used "
                 "by other processes.",
    }

# --------------------------------------------- 第 2 步：profile 差价（真 memory_profiling）
# 注入快照序列：before_profile（权重+NCCL 落地后）→ after_profile（dummy 前向后）
FREE_AT_PROFILE = int(10.75 * GiB)  # 11542724608（12 − 1.25GiB，见下）
SEQ = iter([
    snap(free=FREE_AT_PROFILE, total=TOTAL,
         peak=int(0.75 * GiB), allocated=int(0.75 * GiB),
         reserved=int(0.75 * GiB), non_torch=int(1.0 * GiB)),
    snap(free=FREE_AT_PROFILE, total=TOTAL,
         peak=int(1.0 * GiB), allocated=int(0.75 * GiB),
         reserved=int(0.75 * GiB), non_torch=int(1.0 * GiB)),
])


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


class FakeRunner:
    """GPUModelRunner 的 ENGINE SEAM：cudagraph 估计 0.5GiB、权重 0.75GiB。"""

    model_memory_usage = int(0.75 * GiB)

    def profile_run(self):
        pass

    def profile_cudagraph_memory(self):
        return int(0.5 * GiB)


class FakeCudaPlatform:
    def is_xpu(self):
        return False

    def is_cpu(self):
        return False

    def is_cuda_alike(self):
        return True


gw.current_platform = FakeCudaPlatform()
gw.envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS = True

vllm_config = VllmConfig(
    model_config=ModelConfig(max_model_len=4096, original_max_model_len=4096),
    cache_config=cache_cfg,
    parallel_config=ParallelConfig(),
)
vllm_config.compilation_config = type("CC", (), {"cudagraph_mode": CUDAGraphMode.FULL})()

worker = Worker(vllm_config=vllm_config, model_runner=FakeRunner())
worker.init_snapshot = before_create
worker.requested_memory = requested
available_kv = worker.determine_available_memory()

non_kv_bytes = (
    worker.requested_memory - available_kv - worker.cudagraph_memory_estimate
)
transient_bytes = worker.peak_activation_memory - worker.cudagraph_memory_estimate
OUT["step2_determine_available_memory"] = {
    "weights_memory_gib": 0.75,
    "non_torch_increase_gib": round((worker.total_consumed - int(0.75 * GiB)) / GiB, 2),
    "total_consumed_bytes": worker.total_consumed,
    "total_consumed_gib": round(worker.total_consumed / GiB, 2),
    "transient_peak_headroom_bytes": transient_bytes,
    "transient_peak_headroom_gib": round(transient_bytes / GiB, 2),
    "non_kv_cache_memory_bytes": non_kv_bytes,
    "non_kv_cache_memory_gib": round(non_kv_bytes / GiB, 2),
    "peak_account": "non_kv_cache_memory = total_consumed + transient_peak_headroom",
    "cudagraph_memory_estimate_bytes": worker.cudagraph_memory_estimate,
    "cudagraph_memory_estimate_gib": round(worker.cudagraph_memory_estimate / GiB, 2),
    "subtraction": "available_kv = requested - non_kv_cache_memory - cudagraph_estimate",
}

# --------------------------------------- 第 3 步：字节换块数（Llama-2-7B 锚例）
spec = FullAttentionSpec(
    block_size=16, num_kv_heads=32, head_size=128, dtype=torch.float16
)
page = spec.page_size_bytes
specs = {f"model.layers.{i}.self_attn.attn": spec for i in range(32)}
cfg2 = VllmConfig(
    model_config=ModelConfig(max_model_len=4096, original_max_model_len=4096),
    cache_config=CacheConfig(enable_prefix_caching=False),
    parallel_config=ParallelConfig(),
)
n_blocks_direct = get_num_blocks(cfg2, 32, available_kv, page)
configs = get_kv_cache_configs(cfg2, [specs], [available_kv])
config = configs[0]
concurrency = get_max_concurrency_for_kv_cache_config(cfg2, config)
num_tokens, max_concurrency = get_kv_cache_capacity(cfg2, config)
needed_one_req = math.ceil(4096 / 16) * page * 32

OUT["step3_get_num_blocks"] = {
    "model": "Llama-2-7B FP16 (32 layers, 32 kv_heads, head_dim 128)",
    "block_size": 16,
    "page_size_bytes_per_layer": page,
    "page_size_kib": page // 1024,
    "num_layers_group_size": 32,
    "formula": "num_blocks = available // page_size // group_size",
    "available_kv_bytes": available_kv,
    "available_kv_gib": round(available_kv / GiB, 2),
    "num_blocks_direct": n_blocks_direct,
    "num_blocks_via_get_kv_cache_configs": config.num_blocks,
    "guardrail_needed_one_max_len_req_bytes": needed_one_req,
    "guardrail_needed_gib": round(needed_one_req / GiB, 2),
    "guardrail_pass": needed_one_req <= available_kv,
    "per_token_kv_bytes": 2 * 32 * 32 * 128 * 2,
    "per_token_kv_mib": round(2 * 32 * 32 * 128 * 2 / (1 << 20), 2),
}
OUT["step4_capacity"] = {
    "blocks_per_req_at_4096": math.ceil(4096 / 16),
    "pool_capacity_tokens": num_tokens,
    "max_concurrency": max_concurrency,
    "capacity_formula": "tokens = num_blocks x block_size = concurrency x max_model_len",
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m1.json"), "w",
          encoding="utf-8", newline="\n") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)
print(json.dumps(OUT, ensure_ascii=False, indent=1))
