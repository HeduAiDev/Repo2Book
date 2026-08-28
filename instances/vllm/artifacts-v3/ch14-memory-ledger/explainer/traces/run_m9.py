# run_m9.py — m9 一份账喂两侧 驱动脚本（figure-only 机制的数字出处）
# EngineCore._initialize_kv_caches 全编排：收 spec → determine_available_memory
# → get_kv_cache_configs → 写回 cache_config 四件（num_gpu_blocks/block_size/
# kv_cache_size_tokens/kv_cache_max_concurrency）→ initialize_from_config。
# 玩具：2 层 full、page 65536、available 40MiB → 320 块、容量 5120、并发 1.25。
import json
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
from implementation.engine_core import EngineCore
from implementation.gpu_worker import Worker
from implementation.kv_cache_interface import FullAttentionSpec

GiB = 1 << 30
OUT = {}


def full_spec():
    return FullAttentionSpec(
        block_size=16, num_kv_heads=8, head_size=128, dtype=torch.float16
    )


def uniform_full_layers(num_layers):
    return {f"model.layers.{i}.self_attn.attn": full_spec() for i in range(num_layers)}


class FakeRunner:
    model_memory_usage = 1 * GiB

    def profile_run(self):
        pass

    def profile_cudagraph_memory(self):
        return 0

    def get_kv_cache_spec(self):
        return uniform_full_layers(2)

    def initialize_kv_cache(self, kv_cache_config):
        self.initialized_config = kv_cache_config

    def update_max_model_len(self, max_model_len):
        self.updated_max_model_len = max_model_len


class FakeExecutor:
    def __init__(self, worker):
        self.worker = worker
        self.initialized = None
        self.rpc_calls = []

    def get_kv_cache_specs(self):
        return [self.worker.model_runner.get_kv_cache_spec()]

    def determine_available_memory(self):
        return [40 * 2**20]  # 40MiB = 320 x 131072

    def initialize_from_config(self, kv_cache_configs):
        self.initialized = kv_cache_configs
        self.worker.initialize_from_config(kv_cache_configs[0])

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.rpc_calls.append((method, args))


cache = CacheConfig(enable_prefix_caching=False)
vllm_config = VllmConfig(
    model_config=ModelConfig(max_model_len=4096, original_max_model_len=4096),
    cache_config=cache,
    parallel_config=ParallelConfig(),
)
vllm_config.compilation_config = type("CC", (), {"cudagraph_mode": CUDAGraphMode.NONE})()
runner = FakeRunner()
worker = Worker(vllm_config=vllm_config, model_runner=runner)
executor = FakeExecutor(worker)
core = EngineCore(vllm_config, model_executor=executor)

OUT["boot"] = {
    "available_gpu_memory_bytes": 40 * 2**20,
    "available_mib": 40,
    "page_per_layer": 65536,
    "group_size_layers": 2,
    "bytes_per_block": 131072,
    "max_model_len": vllm_config.model_config.max_model_len,
    "blocks_per_request_at_max_len": cache.num_gpu_blocks
                                     / cache.kv_cache_max_concurrency,
    "cache_config_num_gpu_blocks": cache.num_gpu_blocks,
    "cache_config_block_size": cache.block_size,
    "cache_config_kv_cache_size_tokens": cache.kv_cache_size_tokens,
    "cache_config_kv_cache_max_concurrency": cache.kv_cache_max_concurrency,
    "worker_initialized_num_blocks": runner.initialized_config.num_blocks,
    "executor_got_same_config": executor.initialized[0].num_blocks == cache.num_gpu_blocks,
    "scheduler_flattened_group_spec": type(
        core.scheduler_kv_cache_config.kv_cache_groups[0].kv_cache_spec
    ).__name__,
    "scheduler_kv_cache_manager_ready": core.scheduler.kv_cache_manager is not None,
    "rpc_calls": executor.rpc_calls,
    "single_source_rule": "get_kv_cache_configs 是 KVCacheConfig 唯一产出点，"
                          "调度器（拍平版）与 worker（张量布局版）拿同一份账",
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m9.json"), "w",
          encoding="utf-8", newline="\n") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)
print("ok")
