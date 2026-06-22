# ch12 implementation notes —— step_with_batch_queue 精简版

「只做减法」的可运行精简版。把真实 vLLM（pin `f3fef123`）删掉所有 `# SUBTRACTED:`
分支后应当 ≈ 本目录文件。测试替身（FakeExecutor/FakeScheduler/同步 Future）替换的是
dossier `subtraction_plan.delete[4]` 批准移除的 IPC/CUDA/PP 进程编排（ch07/ch11 内容），
不是对主线方法 `step_with_batch_queue` 本身的改写——该方法逐行保留真实控制流。

## 文件

- `engine_core_batch_queue.py` —— `_MinimalEngineCore`（含 `step_with_batch_queue`、
  `step`、`has_work`、`_process_aborts_queue`、batch queue 字段落地）+ `SchedulerOutput`
  标志 + 替身。
- `executor_max_concurrent_batches.py` —— `max_concurrent_batches` 三处真实定义
  （Abstract / Multiproc / UniProc），即 `batch_queue_size` 的来源。

## 1:1 Source Map

| 精简版符号 | 真实源码 `vllm/...:Lxxx` | 改动 | 原因 |
|---|---|---|---|
| `_MinimalEngineCore.__post_init__`（batch_queue 字段） | `vllm/v1/engine/core.py:L184-L216` | 删 is_ec_consumer / is_pooling_model / request_block_hasher 初始化；async_scheduling 由构造参数注入而非读 vllm_config | dossier delete[1][2][4]：与 batch queue 主线无关的 __init__ 尾段字段 |
| `_MinimalEngineCore.step_with_batch_queue` | `vllm/v1/engine/core.py:L443-L559` | 删 is_ec_consumer 分支、is_pooling_model 快路、log_error_detail/log_iteration_details with 包裹 | delete[1][2][3]：分布式 encoder-cache 小众分支 / pooling 正交快路 / 纯日志上下文。控制流逐行保留 |
| `_MinimalEngineCore.step` | `vllm/v1/engine/core.py:L402-L432` | 删 log_*  with 包裹 | delete[3]，非队列变体仅作对照 |
| `_MinimalEngineCore._process_aborts_queue` | `vllm/v1/engine/core.py:L561-L568` | 替身化（清空 self.aborts）替代 aborts_queue 消费 | delete[4]：IPC abort 队列编排不在本章范围，保留调用点 |
| `_MinimalEngineCore.has_work` | `vllm/v1/engine/core.py:L1152-L1158` | 删 `self.engines_running`（DP 协调） | delete[4]：保留 `bool(self.batch_queue)` 保活语义 |
| `SchedulerOutput.has_structured_output_requests / pending_structured_output_tokens` | `vllm/v1/core/sched/output.py:L221-L227` | 仅保留这两个标志 + total_num_scheduled_tokens | 本章只用这几个字段；其余 SchedulerOutput 字段与 deque 机制正交 |
| `FakeScheduler.get_grammar_bitmask` | `vllm/v1/core/sched/scheduler.py:L1266-L1288` | 替身化（记录调用）替代真实 bitmask 计算 | delete[4]：bitmask 内部由正文内嵌真实源码解读；此处只需调用点观察 deferred 时序 |
| `AbstractExecutor.max_concurrent_batches` | `vllm/v1/executor/abstract.py:L256-L258` | 仅保留该属性 | 基类默认 1 |
| `MultiprocExecutor.max_concurrent_batches` | `vllm/v1/executor/multiproc_executor.py:L474-L478` | 仅保留该属性 + 所需 config 字段 | delete[4]：进程池/RPC 不在本章 |
| `UniProcExecutor.max_concurrent_batches` | `vllm/v1/executor/uniproc_executor.py:L63-L65` | 仅保留该属性 | delete[4] |

## must_keep 自检（dossier subtraction_plan.must_keep 全部在场）

step_with_batch_queue · batch_queue · batch_queue_size · appendleft · pop ·
max_concurrent_batches · step_fn · async_scheduling · deferred_scheduler_output ·
pending_structured_output_tokens · sample_tokens · execute_model · get_grammar_bitmask ·
update_from_output · has_work · model_executed · take_draft_token_ids —— 均原样保留。

## 跑测试

```
python3 -m pytest instances/vllm/artifacts/ch12-engine-core/tests/ -q
```

纯单元测试，不 import vllm，host 可跑。19 passed。
