# ch11 精简版实现笔记 — EngineCore 与忙循环

源码 pin `f3fef123`。精简版 = 只做减法的忠实子集：把真实 vLLM 的 `# SUBTRACTED` 分支删回去
≈ 得到真实单引擎 `EngineCore` / `EngineCoreProc`。两条主线逐行保留：

1. `EngineCore.step()` / `step_with_batch_queue()` —— 一次推理迭代的编排
   （schedule → execute_model(non_block) → grammar bitmask → future.result → sample_tokens →
   _process_aborts_queue → update_from_output）。
2. `EngineCoreProc.run_busy_loop()` + 生命周期（pause/resume/sleep/wake_up）。

唯一的"替换"是把 `model_executor` / `scheduler` / `structured_output_manager` 的**构造**
（属其它章节）换成由测试注入的真实协作者对象——它们的方法调用**原样保留**，由测试以 spy 记录
调用序列。删去所有 `# SUBTRACTED` 分支后，本文件 ≈ 真实 `vllm/v1/engine/core.py` 的单引擎子集。
ZMQ IO 线程主体（`process_input_sockets` / `process_output_sockets`）按减法计划删除，方法签名保留，
其核心控制流（请求如何进 `input_queue`、输出如何出 `output_queue`，含 ABORT 双投）由不含 ZMQ/序列化
的 `simulate_recv` / `simulate_send` 演示。

## 文件
- `interfaces.py` — `PauseMode` / `FinishReason` / `EngineCoreRequestType` / `PauseState` /
  `RequestStatus` / `UtilityResult` / `UtilityOutput` / `EngineCoreOutput` / `EngineCoreOutputs`
  （msgspec.Struct → dataclass，只影响序列化布局，字段语义不变）。
- `core.py` — `EngineCore`（in-proc 内循环）/ `EngineShutdownState` / `EngineCoreProc`（ZMQ 包装、忙循环）。
- `core_client.py` — `InprocClient`（无忙循环对照）/ `AsyncMPClient` 子集（output_queue 另一端，连回 ch04）。

## 1:1 Source Map

| 精简版符号 | 真实 vLLM 位置 | 改动 | 原因 |
|---|---|---|---|
| `EngineCore.__init__` | `core.py:L94-L229` | 删 executor/KV-cache/structured-output/mm/prefix-hash 装配、EEP、KV-connector 握手、GC/env 调优；改为注入 executor+scheduler | 协作者构造属其它章节；保留 batch_queue / step_fn 绑定 / aborts_queue / _idle_state_callbacks |
| `EngineCore.step` | `core.py:L402-L431` | 仅把 log_error_detail/log_iteration_details 换为透传 nullcontext | 全章第一主线，schedule→exec(non_block)→grammar→result→sample→aborts→update 顺序逐行保留 |
| `EngineCore.step_with_batch_queue` | `core.py:L443-L559` | 无删减（保留 deferred / is_ec_consumer / is_pooling_model 分支，给单引擎默认值） | batch queue 接入点，PP 异步流水线核心，章节明确要求讲清 |
| `EngineCore.post_step` | `core.py:L433-L441` | 无删减 | spec-decode draft token 钩子 |
| `EngineCore._process_aborts_queue` | `core.py:L561-L569` | 无删减 | 执行期 abort 在出输出前批量落地，正确性相关 |
| `EngineCore.{add_request,abort_requests}` | `core.py:L315-L354` | 删 pooling/kv_transfer 校验告警 | 保留"校验后转交 scheduler"主线 |
| `EngineCore.{pause_scheduler,resume_scheduler,is_scheduler_paused}` | `core.py:L634-L671` | 无删减（in-proc 同步版） | 三模式 pause → set_pause_state(PauseState) |
| `EngineCore.{sleep,wake_up,is_sleeping}` | `core.py:L673-L729` | 无删减 | 三级休眠主线；level<1 仅暂停，level≥1 委托 executor |
| `EngineCore.{reset_prefix_cache,_reset_caches}` | `core.py:L602-L632` | _reset_caches 删 mm/encoder 缓存清理 | sleep 依赖 reset_prefix_cache；mm/encoder 与本章无关 |
| `EngineCore.preprocess_add_request` | `core.py:L765-L787` | 删 mm 特征接收 / Request 转换 / grammar_init | 保留 input 线程并行预处理骨架与 (req, wave) 契约 |
| `EngineShutdownState` | `core.py:L800-L803` | 无删减 | RUNNING/REQUESTED/SHUTTING_DOWN |
| `EngineCoreProc.__init__` | `core.py:L812-L919` | 删 tensor IPC / 握手 / DP coordinator / IO 线程 spawn | 保留 input_queue/output_queue/executor_fail_callback/shutdown_state |
| `EngineCoreProc.run_engine_core` | `core.py:L1063-L1147` | 删进程命名/tracing/NUMA/DP 分支；改为接受已建 EngineCoreProc | 保留 WAKEUP 哨兵 + 信号处理器 + run_busy_loop 调用 + 关停 |
| `EngineCoreProc.{has_work,is_running,run_busy_loop}` | `core.py:L1152-L1172` | 无删减 | 忙循环骨架与谓词 |
| `EngineCoreProc._process_input_queue` | `core.py:L1174-L1203` | 删 DEBUG 日志分支 | 空闲阻塞 + idle 回调 + ADD/ABORT/UTILITY/WAKEUP dispatch 逐行保留 |
| `EngineCoreProc._process_engine_step` | `core.py:L1205-L1223` | 无删减（含 WAITING_FOR_REMOTE_KVS 的 1ms sleep） | step_fn → output_queue → post_step |
| `EngineCoreProc._handle_shutdown` | `core.py:L1230-L1264` | 删日志 | 三态关停机 |
| `EngineCoreProc._handle_client_request` | `core.py:L1266-L1299` | 删 _convert_msgspec_args（直接透传 args）、未知类型日志 | (type,payload) → add/abort/utility 分派逐行保留 |
| `EngineCoreProc.{_reject_add_in_shutdown,_reject_utility_in_shutdown,_invoke_utility_method}` | `core.py:L1301-L1339` | 删日志 | 关停期拒绝 + utility 调用/Future 延迟回填 |
| `EngineCoreProc.{process_input_sockets,process_output_sockets}` | `core.py:L1372-L1531` | 方法体删（ZMQ/poller/msgpack/零拷贝）；保留签名 + simulate_recv/simulate_send 演示核心控制流 | ZMQ/握手属部署编排；ABORT 双投 input_queue+aborts_queue 控制流由 simulate_recv 复现 |
| `EngineCoreProc.pause_scheduler` | `core.py:L1542-L1582` | 无删减（multi-proc 异步版） | 在途工作未排空时挂 idle 回调返回 Future |
| `EngineCoreProc._pause_complete` | `core.py:L1584-L1589` | 无删减 | not has_work() |
| `EngineCoreProc._send_*outputs*` | `core.py:L1591-L1619` | 无删减 | abort 输出经 output_queue 发客户端 |
| `InprocClient.{get_output,add_request,abort_requests,sleep,wake_up,is_sleeping,shutdown}` | `vllm/v1/engine/core_client.py:L284-L332` | 删 profile/reset_*/lora/collective_rpc 等管理转发 | 进程内对照：get_output 直接 step_fn() + post_step，无忙循环无 ZMQ |
| `AsyncMPClient.{process_outputs_socket,get_output_async,_send_input,add_request_async,abort_requests_async}` | `vllm/v1/engine/core_client.py:L942-L1065` | 删 socket/encoder/utility_results/EEP 装配，最小注入；process_outputs_socket 删 utility/EEP 分派 | output_queue 另一端 → asyncio outputs_queue（连回 ch04 三段式）|
| `EngineCoreRequestType` / `PauseState` / `RequestStatus` / `FinishReason` / `UtilityOutput`(s) / `EngineCoreOutput`(s) | `vllm/v1/engine/__init__.py`, `vllm/v1/serial_utils.py`, `vllm/v1/request.py`, `vllm/v1/core/sched/interface.py` | msgspec.Struct → dataclass，删次要字段，RequestStatus 仅留 FINISHED_ABORTED | 纯内存路径不发 socket；保留本章用到的字段/枚举值语义 |

## must_keep 自检
dossier 列的 41 个 must_keep 符号全部保留（`step`/`step_with_batch_queue`/`execute_model`/
`get_grammar_bitmask`/`sample_tokens`/`update_from_output`/`non_block`/`post_step`/
`_process_aborts_queue`/`aborts_queue`/`run_busy_loop`/`_process_input_queue`/`_process_engine_step`/
`_handle_client_request`/`_handle_shutdown`/`has_work`/`is_running`/`input_queue`/`output_queue`/
`step_fn`/`batch_queue`/`add_request`/`abort_requests`/`preprocess_add_request`/`pause_scheduler`/
`resume_scheduler`/`sleep`/`wake_up`/`is_sleeping`/`set_pause_state`/`PauseState`/`EngineCoreRequestType`/
`EngineShutdownState`/`WAKEUP`/`run_engine_core`/`process_input_sockets`/`process_output_sockets`/
`InprocClient`/`get_output`/`get_output_async`/`add_request_async`）。`lint_fidelity.py` 无 BLOCKING，
35 个纯单元测试全过。

## 关于 simulate_recv / simulate_send 是否违反"不加玩具模拟"
角色铁律禁止"加 vLLM 没有的抽象/伪逻辑"。`simulate_recv` / `simulate_send` **不是新增抽象**，而是
dossier `subtraction_plan.delete` 明确批准删除 `process_input_sockets` / `process_output_sockets`
方法体（ZMQ/握手/序列化）后，把其中**真实存在的核心控制流**（ADD 先 preprocess、ABORT 双投
input_queue+aborts_queue、输出从 output_queue 取出并打 engine_index）以不含 ZMQ 的形式留下供测试驱动。
它顶替的是真实方法的 socket IO 外壳，承载的语义是真实代码原样，不杜撰任何 vLLM 没有的行为。
