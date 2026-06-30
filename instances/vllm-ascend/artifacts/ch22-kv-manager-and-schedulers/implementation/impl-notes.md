# ch22 实现笔记 —— KV 管理与调度器的 NPU 特化（subtract-only）

本章解读/对照的规范源码落点（≥3）：
- `vllm_ascend/core/single_type_kv_cache_manager.py`（CompressAttentionManager + get_manager_for_kv_cache_spec）
- `vllm_ascend/core/scheduler_dynamic_batch.py`（BudgetRefiner + SchedulerDynamicBatch）
- `vllm_ascend/core/recompute_scheduler.py`（RecomputeScheduler + Config/Output 子类）
- `vllm_ascend/core/scheduler_profiling_chunk.py`（ProfilingChunkScheduler）
- 对照基座：`vllm/v1/core/single_type_kv_cache_manager.py`（FullAttentionManager）、`vllm/v1/core/block_pool.py`（BlockPool 原样复用）、`vllm/v1/core/sched/scheduler.py`（Scheduler 基类）

立意：成熟插件对宿主「核心循环」（KV 分配 + 请求调度）尽量少碰。昇腾 90% 原样继承 vLLM，
只在 (a) 压缩 KV spec 的 block 管理、(b) 三种特殊调度策略上开子类。本精简版的减法**只删
dossier `subtraction_plan.delete` 批准项**：三个 schedule()/update_from_output 中与 vLLM 同名
方法逐字相同的 RUNNING/WAITING 循环主体与输出组装、profiling 启动采样细节、history-aware 旁路、
BudgetRefiner 的 CSV 解析。所有改动点（特化注入）原样保留。

## 1:1 Source Map（精简版 ↔ 真实源码 ↔ 改动 ↔ 原因）

| 精简版 | 真实源码 | 改动 | 原因 |
|---|---|---|---|
| single_type_kv_cache_manager.py `CompressAttentionManager` | vllm_ascend/core/single_type_kv_cache_manager.py:L28-L233 | 逐字保留（仅删 L201-205 注释 dead assert） | 昇腾对 KV 管理唯一新增 manager，全章核心，无可安全删的复刻段 |
| single_type_kv_cache_manager.py `get_manager_for_kv_cache_spec` | …:L236-L282 | 逐字保留 | 重映射工厂：复用 vLLM `spec_manager_map`，仅对压缩 MLA / SWA / ChunkedLocal override |
| scheduler_dynamic_batch.py `BudgetRefiner` | vllm_ascend/core/scheduler_dynamic_batch.py:L35-L119 | `_read_lookup_table` 删 pandas 分组 IO（L68-81） | CSV→lookup dict 的纯加载细节；查表控制流不依赖它，测试注入 lookup 即可 |
| scheduler_dynamic_batch.py `SchedulerDynamicBatch.schedule` | …:L151-L574 | 保留两处 ASCEND CHANGE（refine_budget + decode-first 重排）+ 签名/骨架；SUBTRACT L181-574 | 其余逐字复刻 vLLM `Scheduler.schedule`，非特化；Python 无法热补父方法内片段，只能整段 override |
| recompute_scheduler.py `register_ascend_mla_spec_in_manager` / `RecomputeSchedulerConfig` / `RecomputeReqInfo` / `RecomputeSchedulerOutput` | recompute_scheduler.py:L65-L107 | 逐字保留 | import 时序补丁 + 三个 dataclass 子类，体量小、全是特化 |
| recompute_scheduler.py `RecomputeScheduler.schedule` | …:L214-L793 | 保留 recompute 分叉（allocate None→kv_consumer pop+free+recomputed_reqs）+ 签名/骨架；SUBTRACT 逐字循环 | 唯一特化是丢弃重算分叉；其余同 vLLM `Scheduler.schedule` |
| recompute_scheduler.py `RecomputeScheduler.update_from_output` | …:L795-L1085 | 保留 recomputed_reqs 回吐（L831-841）+ 签名；SUBTRACT 主循环/routing/收尾 | 唯一新增是 stop_reason='recomputed' 回吐；其余逐字同 vLLM |
| recompute_scheduler.py `AsyncRecomputeScheduler` | …:L1088-L1093 | 逐字保留 | 多继承组合 AsyncScheduler + RecomputeScheduler |
| profiling_chunk_predictor.py `ChunkSizePredictor` | profiling_chunk_predictor.py:L36-L258 | 删 history 旁路（fit_chunk/predict_with_history/get_time_with_history） | 默认 with_history_ready=False，调度路径走基础 predict/get_time |
| profiling_chunk_predictor.py `ProfilingChunkManager` | …:L304-L385 | 删 history_ready 派发 + record_batch_execution_time | 同上；保留 is_ready/predict_chunk_size/predict_time |
| scheduler_profiling_chunk.py `ProfilingChunkScheduler` | scheduler_profiling_chunk.py:L46-L728 | 保留 3 处 PROFILING CHUNK 改动 + run_profiling 骨架；SUBTRACT 采样循环 + 逐字 schedule 主体 | 启动采样与调度控制流解耦；schedule 主体逐字同 vLLM |

## 测试（host，无 NPU/vLLM）

`tests/conftest.py` 在 `sys.modules` 桩掉 vllm.* / vllm_ascend.* / pandas，把精简版按规范模块名
载入，验证可观察控制流：manager 重映射 + admission cap、压缩缩放（//compress_ratio /
logical_block_size 命中）、BudgetRefiner 查表与恒等、二次模型解 chunk size、RecomputeSchedulerConfig
选类、register 补登记、recomputed_reqs 回吐、AsyncRecompute MRO。`python3 -m pytest` → 19 passed。

真实 NPU 物理 KV 分配由 vLLM 基类替身承接——只验「复用 vs 特化」边界的纯 Python 控制流。
