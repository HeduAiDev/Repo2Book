# SOURCE: vllm/v1/engine/core.py
# EngineCore 的启动装配总编排（m1/m9 站 7）：__init__ 尾段——先 _initialize_
# kv_caches 出账、再 resolve_kv_cache_block_sizes、再 Scheduler(...)（内部
# 建 KVCacheManager——账本就位的顺序）；_initialize_kv_caches——register
# all kvcache specs → 收 spec → determine_available_memory → get_kv_cache_
# configs → auto-fit 同步（collective_rpc update_max_model_len）→ 拍平喂
# 调度器 + num_gpu_blocks/block_size/容量写回 cache_config → initialize_
# from_config（worker 真分配）。
# ENGINE SEAM：model_executor 由装配方注入（真实 executor_class(vllm_config)
#   自建 worker 池，L132——ch05/17；切面直供同契约位：get_kv_cache_specs/
#   determine_available_memory/initialize_from_config/collective_rpc）。
# SUBTRACTED（dossier.delete 批准项的落点 + 邻章边界）：
#   第 10 条 non_causal 检查段（L259-L279——Prefix LM 专用）与弹性 EP
#     分支（L139-L140、L283-L289、L331）；
#   第 9 条编译耗时观测日志（L334-L358——保留计时骨架不打印）；
#   structured_output_manager（L144——ch30）；encoder 模型的 chunked
#     prefill 关闭段（L149-L154——mm）；Scheduler 装配的 connector/
#     batch_queue/mm_registry/handshake（L160-L228——ch16/17/05）；
#   compile_or_warm_up_model（L332——ch19）；request_block_hasher 装配
#     （L220-L229——ch15）。
import time

from .kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_capacity,
    get_kv_cache_configs,
    resolve_kv_cache_block_sizes,
)
from .scheduler import Scheduler
from .single_type_kv_cache_manager import register_all_kvcache_specs


# SOURCE: vllm/v1/engine/core.py:L~100 EngineCore（账本切面——真实类为
#   完整引擎核心，ch05/09 全文）
class EngineCore:
    # SOURCE: vllm/v1/engine/core.py:L~110 __init__（切面装配）
    def __init__(
        self,
        vllm_config,
        model_executor,
    ) -> None:
        # SUBTRACTED: executor_class 自建/fail_callback/ELASTIC_EP（L125-L140
        #   ——ch05/17 + dossier.delete 第 10 条弹性 EP）。
        # SOURCE: vllm/v1/engine/core.py:L132
        self.model_executor = model_executor
        # SOURCE: vllm/v1/engine/core.py:L137
        self.available_gpu_memory_for_kv_cache = -1

        # Setup KV Caches and update CacheConfig after profiling.
        # SOURCE: vllm/v1/engine/core.py:L143
        kv_cache_config = self._initialize_kv_caches(vllm_config)
        # SUBTRACTED: structured_output_manager（L144——ch30）。

        # Setup scheduler.
        # SOURCE: vllm/v1/engine/core.py:L147（get_scheduler_cls 的动态装配
        #   ——ch02；切面直用 Scheduler）
        SchedulerCls = Scheduler

        # SUBTRACTED: encoder 无 KV 时关 chunked prefill（L149-L154——mm）。
        # SOURCE: vllm/v1/engine/core.py:L156-L158 启动装配序（resolve 两个
        #   对齐粒度 → Scheduler）
        scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )

        # SOURCE: vllm/v1/engine/core.py:L160-L168（Scheduler 内部建
        #   KVCacheManager——账本就位）
        self.scheduler: Scheduler = SchedulerCls(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            log_stats=False,
            block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
        )
        # SUBTRACTED: use_spec_decode/check_for_draft_tokens/connector 握手/
        #   batch_queue/mm_registry/request_block_hasher（L169-L229——
        #   ch16/17/05/33/ch15）。

    # SOURCE: vllm/v1/engine/core.py:L250 _initialize_kv_caches
    def _initialize_kv_caches(self, vllm_config) -> object:
        # SOURCE: vllm/v1/engine/core.py:L251
        start = time.time()

        # register all kvcache specs in enginecore process.
        # SOURCE: vllm/v1/engine/core.py:L253-L254
        register_all_kvcache_specs(vllm_config)

        # Get all kv cache needed by the model
        # SOURCE: vllm/v1/engine/core.py:L256-L257
        kv_cache_specs = self.model_executor.get_kv_cache_specs()

        # SUBTRACTED: non_causal 检查段（L259-L279——dossier.delete 第 10 条
        #   Prefix LM：把层级 non_causal 翻译成调度策略的 multiproc-safe 位）。

        # SOURCE: vllm/v1/engine/core.py:L281-L297（有 KV 才 profile——
        #   attention-free 直接 0）
        has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)
        if has_kv_cache:
            # SUBTRACTED: ELASTIC_EP 分支（L283-L289——第 10 条弹性 EP）。
            # Profiles the peak memory usage of the model to determine how
            # much memory can be allocated for kv cache.
            # SOURCE: vllm/v1/engine/core.py:L293-L294
            available_gpu_memory = self.model_executor.determine_available_memory()
            self.available_gpu_memory_for_kv_cache = available_gpu_memory[0]
        else:
            # Attention free models don't need memory for kv cache
            # SOURCE: vllm/v1/engine/core.py:L296-L297
            available_gpu_memory = [0] * len(kv_cache_specs)

        # SOURCE: vllm/v1/engine/core.py:L299
        assert len(kv_cache_specs) == len(available_gpu_memory)

        # Track max_model_len before KV cache config to detect auto-fit changes
        # SOURCE: vllm/v1/engine/core.py:L301-L302
        max_model_len_before = vllm_config.model_config.max_model_len

        # SOURCE: vllm/v1/engine/core.py:L304-L306（定账总控单点调用——
        #   同一份 config 是唯一真相）
        kv_cache_configs = get_kv_cache_configs(
            vllm_config, kv_cache_specs, available_gpu_memory
        )

        # If auto-fit reduced max_model_len, sync the new value to workers.
        # This is needed because workers were spawned before memory profiling
        # and have the original (larger) max_model_len cached.
        # SOURCE: vllm/v1/engine/core.py:L308-L313（auto-fit 缩了长度要
        #   collective_rpc 同步 worker）
        max_model_len_after = vllm_config.model_config.max_model_len
        if max_model_len_after != max_model_len_before:
            self.collective_rpc("update_max_model_len", args=(max_model_len_after,))

        # SOURCE: vllm/v1/engine/core.py:L315-L326（拍平喂调度器 + 账本写回
        #   cache_config——前端日志与 API 看到的就是这些值）
        scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
        self.scheduler_kv_cache_config = scheduler_kv_cache_config
        vllm_config.cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks
        kv_cache_groups = scheduler_kv_cache_config.kv_cache_groups
        if kv_cache_groups:
            vllm_config.cache_config.block_size = min(
                g.kv_cache_spec.block_size for g in kv_cache_groups
            )
            num_tokens, max_concurrency = get_kv_cache_capacity(
                vllm_config, scheduler_kv_cache_config
            )
            vllm_config.cache_config.kv_cache_size_tokens = num_tokens
            vllm_config.cache_config.kv_cache_max_concurrency = max_concurrency

        # SUBTRACTED: validate_block_size（L328——config 装配校验，ch03）。

        # SOURCE: vllm/v1/engine/core.py:L330（worker 侧真分配——同一份
        #   config 的另一半）
        self.model_executor.initialize_from_config(kv_cache_configs)
        # SUBTRACTED: compile_or_warm_up_model（L331-L332——ch19）与编译
        #   耗时日志（L334-L358——第 9 条观测；计时骨架保留不打印）。

        # SOURCE: vllm/v1/engine/core.py:L359
        _ = time.time() - start
        return scheduler_kv_cache_config

    # SOURCE: vllm/v1/engine/core.py:L953 collective_rpc
    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ):
        # SOURCE: vllm/v1/engine/core.py:L960（转发给 executor 的 worker 池）
        return self.model_executor.collective_rpc(method, timeout, args, kwargs)
