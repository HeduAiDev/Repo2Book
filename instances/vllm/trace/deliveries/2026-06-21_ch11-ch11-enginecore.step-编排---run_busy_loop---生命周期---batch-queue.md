# ch11 EngineCore.step 编排 / run_busy_loop / 生命周期 / batch queue 接入点 交付

- **Type**: delivery
- **Chapter**: 11
- **Date**: 2026-06-21
- **Timestamp**: 2026-06-21T16:49:30Z
- **Agents involved**: analyst,  implementer,  tester,  writer,  reviewer,  archivist
- **User present**: False
- **Tags**: delivery,  engine-core,  busy-loop,  lifecycle,  batch-queue,  APPROVED

## What happened

ch11 落地 EngineCore 进程内主线：step() 七步编排（has_requests 守门→schedule→execute_model non_block=True→get_grammar_bitmask→future.result→None 时 sample_tokens→_process_aborts_queue→update_from_output，CPU 掩码与 GPU 前向重叠）、run_busy_loop 心跳（_process_input_queue 阻塞睡/零 CPU + _process_engine_step 塞 output_queue + 1ms 让 GIL）、_handle_client_request 五分派（WAKEUP/ADD/ABORT/UTILITY RPC/EXECUTOR_FAILED）、关停三态机（WAKEUP 哨兵叫醒阻塞 get + _handle_shutdown RUNNING/REQUESTED/SHUTTING_DOWN）、生命周期（pause 三模式 abort/keep/wait + PauseState、sleep 三级 level0/1/2 + inproc 同步 vs proc 异步 idle-callback Future）、batch queue 接入点 step_fn 绑定、InprocClient 手动驱动。四 linter 全 PASS、host 35/35 passed（纯单元桩，不 import vllm/无需容器）。bible 新增 11 条 ch11 接口。

## Why it matters

结清 ch04 三段式『EngineCore 段是黑盒』与 ch07『进程边界那头的引擎本体留待后续』两笔账——本章把那台一拍接一拍的节拍器现了真身，是 Part III 引擎内核的开篇。

## What to remember

ch11 伏笔账目：正确埋下 f9(batch queue→ch12)、f10(schedule→ch13)，arc-map 维持 open；自然回收 ch04 三段式 EngineCore 段承诺（§11.7 output_queue 连回第4章）。bible due ch11 无应回收悬挂。两条 nit ACCEPT_AS_IS（PauseState docstring PAUSE_* 原文保真；T_step max 近似已注明前提）。下一章 ch12 须把 step_with_batch_queue 完整展开（deque appendleft/pop、填管道优先、deferred_scheduler_output 投机解码×结构化输出分支）。
