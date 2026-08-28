# ch15 精简版包：只做减法的前缀缓存（链式哈希/平面哈希表/命中查找/touch 救回/
# LRU 双不变量/惰性驱逐/F2 抢占恢复/块内 CoW/混合不动点/Marconi 钉住），
# 对应 vllm/v1/core/{kv_cache_utils,block_pool,single_type_kv_cache_manager,
# kv_cache_coordinator,kv_cache_manager,sched/scheduler} + vllm/v1/{request,engine/core}
# + vllm/v1/worker/{gpu_model_runner,utils} + vllm/{config/cache,utils/hashing}。
# 运行（host，纯 CPU 单元/契约测试）：cd 本章目录 && python -m pytest tests/ -q
