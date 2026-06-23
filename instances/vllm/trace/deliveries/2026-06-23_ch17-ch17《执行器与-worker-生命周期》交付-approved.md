# ch17《执行器与 Worker 生命周期》交付 APPROVED

- **Type**: delivery
- **Chapter**: 17
- **Date**: 2026-06-23
- **Timestamp**: 2026-06-23T11:27:49Z
- **Agents involved**: analyst
- **User present**: False
- **Tags**: executor

## What happened

ch17 是 Part V 执行层开篇，把引擎的控制平面落到执行器+Worker 生命周期。(1) Executor.get_class() 三态工厂：按 distributed_executor_backend 分发 uni/mp/ray；__init__→_init_executor() 钩子。(2) collective_rpc 是控制平面唯一入口——execute_model/sample_tokens/determine_available_memory/add_lora/sleep 全统一为向所有 worker 广播的 RPC。(3) UniProcExecutor 退化为同进程 run_method(driver_worker) 直调，作 mp 最简对照。(4) MultiprocExecutor：1 执行器进程 + N worker 子进程；rpc_broadcast_mq 共享内存 MessageQueue 一次性广播 RPC、FutureWrapper(Future 子类)按发出顺序排空 futures_queue(底层应答 MQ 为 FIFO 保证请求/回复配对)异步收回、worker monitor 线程经 sentinel 做失败传播、execute_model 走单 output_rank 非阻塞流水线、三级关停 优雅→SIGTERM→SIGKILL。(5) WorkerProc：worker_main 子进程入口 + READY/death pipe 握手 + worker_busy_loop 取 RPC 执行回写、异常转 ResponseStatus.FAILURE→RuntimeError。(6) WorkerWrapperBase 延迟初始化：先记 rpc_rank，init_worker 时按 worker_cls qualname 解析实例化真实 Worker、__getattr__ 透传使 collective_rpc 的 getattr 命中具体方法。四 linter 全 PASS、容器内 24/24 测试通过(真 spawn 子进程验证广播/单 rank 应答/方法异常→FAILURE/进程死亡→failure_callback/register-after-failure 立即触发)、忠实子集只做减法。review verdict=APPROVED，5 条 issue 全 non-blocking(行号漂移已被 CLAUDE.md 坑#4 约定覆盖、3 处自锚链接 slug 机械小修、2 条精度补强 nit)。

## Why it matters

ch17 是 Part V 执行层开篇与控制平面收口：前面 Part 把 Executor 当黑盒(execute_model/collective_rpc)，本章揭示引擎所有指令如何经 collective_rpc 单一入口广播到 N 个 worker、mp 后端如何用共享内存 MessageQueue + FutureWrapper 异步往返、失败如何经 sentinel/death-pipe 传播并优雅关停。后续模型运行器章(execute_model 内 model_runner 前向/PP 通信)、KV 内存章(determine_available_memory)、分布式编排(ray 后端)均在此控制平面之上展开。

## What to remember

ch17：collective_rpc 是执行层控制平面唯一入口(所有引擎指令统一为向全 worker 广播的 RPC)；Executor.get_class 三态工厂(uni/mp/ray)；MultiprocExecutor 用共享内存 MessageQueue 广播 + FutureWrapper FIFO 顺序配对异步收回 + monitor 线程 sentinel 失败传播 + 三级关停；WorkerWrapperBase 延迟初始化 + __getattr__ 透传让 RPC 命中具体方法。24/24 容器测试、四 linter PASS、APPROVED 5 条全 non-blocking。新接口已登记 bible(8 条)。
