# ch13 精简版包：只做减法的分页 KV（BlockPool/FreeKVCacheBlockQueue/
# KVCacheManager 三段式 + block_id 跨进程契约 + worker 块表镜像 + 槽位恒等式），
# 对应 vllm/config/cache.py + vllm/v1/{core,worker,kv_cache_interface,request}。
# 运行（host，纯 CPU 单元/契约测试）：cd 本章目录 && python -m pytest tests/ -q
