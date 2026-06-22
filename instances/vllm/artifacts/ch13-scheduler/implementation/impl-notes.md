# ch13 精简版实现笔记 —— Token 为中心的连续批处理

只做减法的忠实精简版：与真实 `vllm/v1/core/sched/` 同名同结构同控制流，
仅删除 `dossier.subtraction_plan.delete` 批准的子系统。删除点全部 `# SUBTRACTED:` 标注。

## 文件构成

| 精简版文件 | 对应真实 vLLM | 角色 |
|---|---|---|
| `scheduler.py` | `vllm/v1/core/sched/scheduler.py` | 同步连续批处理调度器主角（schedule / update_from_output） |
| `async_scheduler.py` | `vllm/v1/core/sched/async_scheduler.py` | AsyncScheduler 占位机制（f6 回收），与真实 1:1 无删除 |
| `output.py` | `vllm/v1/core/sched/output.py` | NewRequestData / CachedRequestData / SchedulerOutput |
| `interface.py` | `vllm/v1/core/sched/interface.py` | SchedulerInterface 契约 + PauseState |
| `request.py` | `vllm/v1/request.py` + `vllm/v1/core/sched/utils.py` | Request / RequestStatus / SamplingParams / check_stop |
| `request_queue.py` | `vllm/v1/core/sched/request_queue.py` | FCFSRequestQueue + create_request_queue |
| `kv_cache_manager.py` | `vllm/v1/core/kv_cache_manager.py` | 调度器所需的 KV 块分配**接口契约**桩（KV cache 章主角） |

## 1:1 Source Map（精简版 ↔ 真实 vllm ↔ 改动 ↔ 原因）

| 精简版符号 | 真实位置 | 改动 | 原因 |
|---|---|---|---|
| `Scheduler.schedule` | `scheduler.py:L352-L945` | 删 encoder/mm、structured、mamba、LoRA、KVConnector、PRIORITY、PP、streaming 分支 | 全部 dossier.delete 批准；删后单机文本调度的两阶段控制流完整自洽 |
| RUNNING 阶段追赶公式 | `scheduler.py:L408-L421` | 保留 `num_tokens_with_spec + num_output_placeholders - num_computed_tokens` 及三重 min | 这是「不分 prefill/decode 相」的核心；must_keep |
| async 提前剪枝 | `scheduler.py:L392-L406` | 保留 | 与 num_output_placeholders 配平相关，解释 AsyncScheduler |
| 抢占回环 | `scheduler.py:L464-L514` | 删 PRIORITY 分支，保留 FCFS `self.running.pop()` | dossier.delete 批准；FCFS 抢队尾已足以讲清 allocate_slots→None→抢占 |
| WAITING 阶段守卫 | `scheduler.py:L568` | 保留 `if not preempted_reqs and UNPAUSED` | 抢占后不调度 WAITING；must_keep design decision |
| WAITING num_new_tokens | `scheduler.py:L672-L693` | 删 load_kv_async/encoder，保留 `num_tokens - num_computed_tokens` 与 chunked prefill break | 前缀缓存命中数由 get_computed_blocks 提供（桩返 0） |
| 约束断言 | `scheduler.py:L848-L859` | 原样保留 | token 预算守恒 + running 上界 |
| new vs cached 二分 | `scheduler.py:L871-L902` | 删 use_v2_model_runner 分支 | dossier.delete；保留 from_request(全量) vs _make_cached_request_data(增量) + prev_step 刷新 |
| `_update_after_schedule` | `scheduler.py:L974-L998` | 删 structured 累计 | 保留乐观推进 num_computed_tokens + is_prefill_chunk |
| `_make_cached_request_data` | `scheduler.py:L1043-L1101` | 删 PP new_token_ids 回传分支 | 非 PP 时恒空；保留 prev_step 判定全量/增量 + resumed_req_ids |
| `update_from_output` | `scheduler.py:L1290-L1551` | 删 KVConnector/encoder/structured/pooling/logprobs/stats | 保留 spec 拒绝回退、追加 token、check_stop、free、移除停止请求 |
| `_update_request_with_output` | `scheduler.py:L1622-L1638` | 原样保留 | append_output_token_ids + check_stop |
| `AsyncScheduler.*` | `async_scheduler.py:L12-L60` | 仅删 discard_latest_async_tokens 分支 | 占位 +1+spec / 兑现 -len / cache_blocks 全保留 |

## 测试覆盖（tests/test_scheduler.py，13 项，纯单元无 import vllm）

复现的真实可观测行为：首次→NewRequestData / 二拍→CachedRequestData / prefill+decode
同拍混批 / token_budget chunk 长 prefill / chunked 关闭则整块 break / KV 块耗尽抢队尾 /
抢占后不调度 WAITING / 恢复请求进 resumed_req_ids / PAUSED_ALL 不调度 /
达 max_tokens 停止并 free / max_num_running_reqs 上界 / async 占位 +1 与兑现配平。

## 桩说明

`kv_cache_manager.py` 用按块数计数的最小实现替代真实分页器，**保留与真实一致的方法签名
与「显存满则 allocate_slots 返回 None」语义**，使 schedule() 的抢占分支被真实驱动；
前缀缓存哈希匹配、块池引用计数等属 KV cache 章，不在本章展开。
