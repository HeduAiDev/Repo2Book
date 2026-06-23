# ch17 执行器与 Worker 生命周期 精简版 — 实现笔记

只做减法的忠实精简版，source pin `f3fef123`。与 vLLM 同名、同结构、同控制流；只删不增。
控制平面主干（工厂三态分发 / collective_rpc 广播 + 单 rank 应答 / FutureWrapper FIFO 配对 /
WorkerProc 子进程生命周期 / WorkerWrapperBase 延迟初始化）逐字保留，仅删去 CUDA/torch/分布式/
NUMA/tracing/多模态/ray/多节点等与本章主线正交的实体（均带 `# SUBTRACTED:` 注释）。

精简版**真正 spawn worker 子进程**，可跑通『拉起 worker → 广播 RPC → worker 执行 → 收应答 →
失败传播 → 三级关停』全闭环，可数值追踪、可打断点——不杜撰任何前向计算（桩 Worker.execute_model
是确定性回声）。

## 文件布局

| 精简版文件 | 镜像的真实 vLLM 文件 |
| --- | --- |
| `abstract.py` | `vllm/v1/executor/abstract.py`（Executor 基类 + get_class 工厂） |
| `uniproc_executor.py` | `vllm/v1/executor/uniproc_executor.py`（UniProcExecutor 最简对照） |
| `multiproc_executor.py` | `vllm/v1/executor/multiproc_executor.py`（主角：FutureWrapper / MultiprocExecutor / WorkerProc） |
| `worker_base.py` | `vllm/v1/worker/worker_base.py`（WorkerBase 接口 + WorkerWrapperBase 延迟初始化） |
| `gpu_worker.py` | `vllm/v1/worker/gpu_worker.py`（Worker 生命周期锚点骨架） |
| `serial_utils.py` | `vllm/v1/serial_utils.py`（run_method）+ `vllm/utils/import_utils.py`（resolve_obj_by_qualname） |
| `shm_broadcast.py` | `vllm/distributed/device_communicators/shm_broadcast.py`（MessageQueue 的 FIFO 语义镜像） |

## 1:1 Source Map

| 精简版符号 | vllm/...:Lxxx | 改动 | 原因 |
| --- | --- | --- | --- |
| `Executor.get_class` | `vllm/v1/executor/abstract.py:L47-L92` | 删 ray / external_launcher 两分支 | 工厂三态分发主线；保留 type/'mp'/'uni'/qualname/非法 五路 |
| `Executor.__init__` / `_init_executor` | `vllm/v1/executor/abstract.py:L94-L116` | 摊平 config 样板赋值缩为注释 | 抽象钩子，子类拉起 worker 入口 |
| `Executor.collective_rpc`(抽象) / `execute_model` / `sample_tokens` / `determine_available_memory` / `shutdown` | `vllm/v1/executor/abstract.py:L198-L282` | 删 ~20 个同模式 collective_rpc 薄封装与 @overload | 示范『一切引擎指令 = collective_rpc 薄封装』 |
| `UniProcExecutor` | `vllm/v1/executor/uniproc_executor.py:L26-L141` | 删 async_output_thread / 真实 IP 端口 / single_value 重写 | 同进程直调最简对照；non_block→已完成 Future 保留 |
| `FutureWrapper` | `vllm/v1/executor/multiproc_executor.py:L69-L99` | 字面照搬 | 异步结果 + `appendleft`/`pop` 的 FIFO 顺序排空（本章核心巧思） |
| `MultiprocExecutor._init_executor` | `vllm/v1/executor/multiproc_executor.py:L109-L246` | 删多节点/inherited_fds/OMP/numa/日志 | 建广播 MQ→逐 rank 拉起 WorkerProc→wait_for_ready→起监控→装配 response_mqs |
| `MultiprocExecutor.collective_rpc` | `vllm/v1/executor/multiproc_executor.py:L339-L403` | 删 kv_output_aggregator 分支 | 广播 enqueue + 构造 FutureWrapper；output_rank 单 rank 应答；callable→cloudpickle |
| `start_worker_monitor` / `register_failure_callback` | `vllm/v1/executor/multiproc_executor.py:L267-L304` | 删 logger | sentinel 监控进程死亡 + 失败回调传播 |
| `_ensure_worker_termination` / `shutdown` | `vllm/v1/executor/multiproc_executor.py:L405-L468` | 删 logger | graceful→SIGTERM→SIGKILL 三级关停 + 关 MQ |
| `_get_output_rank` | `vllm/v1/executor/multiproc_executor.py:L480-L494` | 字面照搬 | TP rank0 + 最后 PP 段；execute_model 只收一个 rank |
| `WorkerProc.__init__` / `_init_message_queues` | `vllm/v1/executor/multiproc_executor.py:L546-L641` | 删多节点 MQ / 标题 / EP / 平台调整；response MQ 上移父进程预创建 | 子进程内 init_worker→init_device→load_model→建 response MQ |
| `make_worker_process` | `vllm/v1/executor/multiproc_executor.py:L643-L694` | 删 inherited_fds/numa；response MQ 父进程预创建经 kwargs 继承 | 起子进程 + ready/death pipe |
| `wait_for_ready` / `wait_for_response_handle_ready` | `vllm/v1/executor/multiproc_executor.py:L696-L753` | response handle 改从父留 UnreadyHandle 取 | READY 握手，父进程等 worker 就绪 |
| `worker_main` | `vllm/v1/executor/multiproc_executor.py:L791-L895` | 删 tracer/numa/logger；READY 不回传 queue handle | 子进程入口：信号→构造→READY→busy_loop；异常分支结构保留 |
| `worker_busy_loop` / `enqueue_output` / `handle_output` / `ResponseStatus` | `vllm/v1/executor/multiproc_executor.py:L897-L970` | 删 async 输出/AsyncModelRunnerOutput/logger | dequeue RPC→getattr/cloudpickle→执行→回写；异常转 FAILURE；单 rank 回写 |
| `WorkerProc.shutdown` / `monitor_death_pipe` | `vllm/v1/executor/multiproc_executor.py:L755-L789` | 删 destroy_*分布式组 | 关 MQ + death pipe EOF 自清理 |
| `set_multiprocessing_worker_envs` / `get_mp_context` | `vllm/v1/executor/multiproc_executor.py:L1009`、`vllm/utils/system_utils.py` | 删 OMP 调优；固定 spawn | 拉起 worker 前环境钩子；spawn 与 vllm 默认一致 |
| `WorkerWrapperBase` (`__init__`/`init_worker`/`init_device`/`__getattr__`/`execute_model`) | `vllm/v1/worker/worker_base.py:L179-L345` | 删 plugins/extension_cls/mm_cache/config 上下文 | 延迟初始化：按 worker_cls qualname 解析真实 Worker；__getattr__ 透传 |
| `WorkerBase` | `vllm/v1/worker/worker_base.py:L38-L177` | 删纯虚接口清单大部 | 硬件无关接口；保留 init_device/load_model/execute_model/check_health/shutdown 纯虚 |
| `Worker` | `vllm/v1/worker/gpu_worker.py:L106-L861` | __init__/init_device/load_model/execute_model 骨架，体作锚点 | 生命周期锚点；前向/KV/PP 通信留 ch18+ |
| `run_method` | `vllm/v1/serial_utils.py:L486-L510` | 字面照搬 | str→getattr / bytes→cloudpickle / callable→直接调，与 worker_busy_loop 派发同构 |
| `resolve_obj_by_qualname` | `vllm/utils/import_utils.py` | 保留 rsplit→import→getattr 核心路径 | 『字符串类名→类对象』，init_worker 延迟实例化关键 |
| `MessageQueue` | `vllm/distributed/device_communicators/shm_broadcast.py` | shm 环形缓冲→multiprocessing.Queue FIFO 镜像 | 保留 FIFO + 一次 enqueue 全 reader 可见 + handle 复用的可观察语义 |

## 唯一结构性减法（已在源码内逐处注明）

`MessageQueue` 的 shm 零拷贝环形缓冲（数百行 + ZMQ 远程 fallback）被减成 multiprocessing(spawn
上下文) Queue 背书的等价 FIFO：广播 MQ 内部每 reader 一个 FIFO、`enqueue` 向全部 put（复现『一次
广播被 N 个 reader 各看到一份』），应答 MQ 为单 FIFO。dossier theory 的 FIFO 配对正确性论证只依赖
FIFO 序，不依赖 shm 字节布局，故语义等价。

附带一处微调（已注明）：因 spawn 下底层 Queue 只能经『进程参数继承』传递、不能再经 Pipe 二次
pickle，worker 的 response MQ 改由父进程在 `make_worker_process` 预创建、作为 kwargs 继承给子进程
（真实 vLLM 由子进程自建再经 ready pipe 发回 shm handle）。READY 握手协议、`wait_until_ready` 顺序
约束、控制流全部不变，仅『response MQ 在谁那里 new 出来』从子进程上移到父进程。
