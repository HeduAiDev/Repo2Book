# ch14 精简版实现说明（只做减法）

聚焦两条 ch13 未展开的分支：(1) RUNNING 阶段 `allocate_slots` 失败时的 FCFS LIFO
抢占循环 + 回流，(2) `update_from_output` 的请求生命周期回流（追加 token / 停止检测 /
完成态迁移 / spec 回退）。以及支撑两者的 waiting/skipped_waiting 双队列。

精简版纯单元、不 `import vllm`，host `python3 -m pytest tests/` 即可跑。
KV 缓存子系统（前缀缓存命中、块哈希、滑窗、mamba 对齐）、encoder、connector/远程 KV、
约束解码、streaming-input、pooling、统计/日志、PRIORITY 策略、PAUSED 控制、logprobs/
EngineCoreOutput 装配等均按 `subtraction_plan.delete` 删除，与抢占·回流主线正交。

## 文件
- `request.py` — `RequestStatus`(IntEnum 状态机, `is_finished = status > PREEMPTED`)、`FinishReason`、最小 `Request`。
- `request_queue.py` — `FCFSRequestQueue`（`prepend_request=appendleft` 是抢占回队头/skipped 重排的底层语义）。
- `utils.py` — `check_stop`（token 级停止：min_tokens→EOS→stop_token_ids→length）、`remove_all`。
- `kv_cache_manager.py` — 精简到块计数语义的 `KVCacheManager`：只保留「分配失败返回 None / free 归还块」这一对本章因果可观察的契约。
- `scheduler.py` — `Scheduler`：抢占循环 + 双队列 WAITING + `update_from_output`。
- `async_scheduler.py` — `AsyncScheduler`：覆写 `_update_request_with_output`（占位回扣 + discard_latest_async_tokens）。

## 1:1 Source Map

| 精简版符号 | 真实 vLLM `:Lxxx` | 改动 | 原因 |
|---|---|---|---|
| `Scheduler._allocate_with_preemption` (scheduler.py) | `vllm/v1/core/sched/scheduler.py:L464-L514` | 把 `schedule()` 内联的 `while True` 抢占块抽成方法；删 PRIORITY 整块、encoder 预算归还、record_function | 便于单测/讲解，控制流与原内联块一一对应；PRIORITY 按 delete 删除，只演示 FCFS `self.running.pop()` LIFO |
| `Scheduler._preempt_request` | `vllm/v1/core/sched/scheduler.py:L952-L972` | 删 `encoder_cache_manager.free` 与 `log_stats` 的 `record_event` | 编码器/观测正交；保留 free KV / status→PREEMPTED / num_computed_tokens=0 / 清 spec / num_preemptions++ / `waiting.prepend_request` 回队头 |
| `Scheduler.schedule` (WAITING 段) | `vllm/v1/core/sched/scheduler.py:L567-L846` | 删 `UNPAUSED` 守卫、LoRA、前缀缓存、connector load、encoder；保留 `not preempted_reqs` 守卫与双队列遍历 | PAUSED/LoRA/前缀缓存/远程 KV 按 delete 删除；`not preempted_reqs` 跳过 WAITING、`step_skipped_waiting` 重排是本章核心 |
| `Scheduler._select_waiting_queue_for_scheduling` / `_enqueue_waiting_request` / `_is_blocked_waiting_status` | `vllm/v1/core/sched/scheduler.py:L1561-L1577,L1554` | 删 PRIORITY 队头比较分支 | 双队列防队头阻塞的归类/选择核心保留；PRIORITY 删除 |
| `Scheduler._try_promote_blocked_waiting_request` | `vllm/v1/core/sched/scheduler.py:L2061-L2092` | 退化为恒 `return False` | 远程 KV/grammar/streaming 提升条件随对应特性删除；保留调用点体现「阻塞态本拍跳过」 |
| `Scheduler.update_from_output` | `vllm/v1/core/sched/scheduler.py:L1331-L1481` | 入参精简为 `req_id -> 采样 token 列表`；删 logprobs/EngineCoreOutput 装配、pooling、grammar、stats | 输出装配是 Part II 主题；保留 spec 回退、停止分流（running/preempted）、批量摘除 |
| `Scheduler._update_request_with_output` | `vllm/v1/core/sched/scheduler.py:L1622-L1638` | 无删减 | 逐 token `append_output_token_ids` + `check_stop` + 命中后 `del` 截断 |
| `Scheduler._handle_stopped_request` | `vllm/v1/core/sched/scheduler.py:L1579-L1595` | resumable 恒 False → 直接 `return True`，删 streaming 续接 | streaming-input 多轮按 delete 删除；停止即真完成的闭环完整 |
| `Scheduler._free_request` / `_free_blocks` | `vllm/v1/core/sched/scheduler.py:L1813-L1834` | 删 connector 收尾、encoder free、多客户端 dict、delay_free_blocks 主路径 | 保留 finished_req_ids 登记 + KV free + 从 `self.requests` 删除的终点闭环 |
| `RequestStatus` / `is_finished` / `get_finished_reason` / `_FINISHED_REASON_MAP` | `vllm/v1/request.py:L310-L352` | 无删减（除 `__str__`） | 状态机真相源；`is_finished = status > PREEMPTED` |
| `check_stop` | `vllm/v1/core/sched/utils.py:L94-L130` | 删 repetition_detection 分支 | 重复检测模式匹配属停止细节；EOS/stop_token/length 主线完整。注意：这是 **token 级** stop_token_ids，非 detokenizer 的 stop **string** |
| `AsyncScheduler._update_request_with_output` | `vllm/v1/core/sched/async_scheduler.py:L37-L60` | 删 `__init__`/`_update_after_schedule` 的占位预增 | 保留 num_output_placeholders 回扣 + discard_latest_async_tokens + 仅 RUNNING cache_blocks |
| `FCFSRequestQueue` | `vllm/v1/core/sched/request_queue.py:L75-L128` | 删抽象基类与 PriorityRequestQueue | `prepend_request=appendleft` 语义保留 |
| `KVCacheManager.allocate_slots` / `free` | `vllm/v1/core/kv_cache_manager.py` | 精简到块计数；删前缀缓存/块哈希/滑窗/lookahead 预留 | 只需「耗尽→None 触发抢占 / free 归还」的可观察契约驱动抢占·回流 |

## 关键忠实点
- **FCFS LIFO**：抢 `self.running.pop()`（RUNNING 末尾），最小化对老请求公平性破坏。
- **丢弃重算非换出**：抢占即 `free` KV + `num_computed_tokens=0`，回 waiting 队头从 0 重 prefill。
- **抢占后跳过 WAITING**：`not preempted_reqs` 守卫——内存已不足，不再放新请求避免抖动。
- **双队列防队头阻塞**：阻塞态进 `skipped_waiting`，遍历 `_try_promote` 失败则 pop 出来 prepend 进 `step_skipped_waiting`，step 末整体 prepend 回 `skipped_waiting`。
- **finish_reason 先抓后改**：`update_from_output` 在 `_handle_stopped_request`（可能把 status 改回 WAITING）之前调用 `get_finished_reason`。
- **停止分流**：按 `status_before_stop` 分 `stopped_running_reqs`（从 running `remove_all`）/`stopped_preempted_reqs`（从 waiting `remove_requests`）。
