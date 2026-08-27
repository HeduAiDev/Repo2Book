# ch14 精简版包：只做减法的显存账本（启动三步定账 + 准入两道门 +
# 混合组化 + SWA 窗外回收 + kernel 块细分），对应 vllm/{utils,config} +
# vllm/v1/{core,worker,kv_cache_interface,engine}。
# 运行（host，纯 CPU 单元/契约测试）：cd 本章目录 && python -m pytest tests/ -q
