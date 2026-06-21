# ch04 精简版实现笔记 — AsyncLLM 三段式异步解耦

源码 pin `f3fef123`。精简版 = 只做减法的忠实子集：把真实 vLLM 的 `# SUBTRACTED` 分支删回去
≈ 得到真实 AsyncLLM。唯一的"替换"是按 dossier `subtraction_plan._principle` 把 Stage2 的
**独立进程 EngineCore + ZMQ IPC** 换成 **in-process stub**（同名 async 接口），让 AsyncLLM 的
三段式编排骨架几乎一字不改、可在 host 上跑通、可打断点观察生产者-消费者关系。IPC 物理机制留 ch07。

## 文件
- `messages.py` — `EngineCoreRequest` / `EngineCoreOutput` / `EngineCoreOutputs`（跨进程边界消息）+ 占位 `SamplingParams`
- `input_processor.py` — `InputProcessor`（Stage1 黑盒，prompt -> EngineCoreRequest）
- `output_processor.py` — `RequestOutputCollector`(f1) / `OutputProcessor`(登记+解多路复用) / `RequestOutput`
- `engine_core_stub.py` — `InProcessEngineCore`（Stage2 IPC 接缝的 in-process 替身，f2）
- `async_llm.py` — `AsyncLLM`（三段 facade，主角）

## 1:1 Source Map

| 精简版符号 | 真实 vLLM 位置 | 改动 | 原因 |
|---|---|---|---|
| `AsyncLLM.__init__` | `vllm/v1/engine/async_llm.py:L82-L200` | 删观测/profiler/tracing/DP；EngineCore 换 stub | 三段构造骨架保留；旁路特性与 IPC 留 ch07/ch21 |
| `AsyncLLM.add_request` | `async_llm.py:L280-L398` | 删流式输入/deprecated/n>1/校验分支，收敛到常规 prompt + n==1 | 主路径已含完整三段+per-request 队列；正交特性删 |
| `AsyncLLM._add_request` | `async_llm.py:L400-L415` | 仅删 log_requests 日志 | 全章最关键 16 行（双登记扇出点 f2），逐行保留 |
| `AsyncLLM.generate` | `async_llm.py:L524-L635` | 保留 CancelledError->abort 一条 except，删其余错误分类分支 | 消费者侧骨架 `get_nowait() or await get()` + finished 判停逐行保留（f3）|
| `AsyncLLM._run_output_handler` / `output_handler` | `async_llm.py:L637-L707` | 删 log_stats/logger_ref/scheduler_stats | 生产者侧 while True 拉取->分块 process_outputs->队列 + 块间 sleep(0) 逐行保留（f3）|
| `AsyncLLM.abort` | `async_llm.py:L709-L721` | 删 log + as_list 工具 | OutputProcessor+EngineCore 双向清理语义保留 |
| `RequestOutputCollector`(+put/get/get_nowait/close) | `output_processor.py:L45-L106` | 删 PoolingRequestOutput 分支与 _input_stream_task | 单槽+Event+merge 逐行保留（f1）；close 留空骨架 |
| `OutputProcessor.add_request` | `output_processor.py:L508-L537` | 删流式更新/parent/external_req_ids；RequestState 只存 req_id+queue | 登记 req_id->queue 查找表骨架保留（解多路复用前提）|
| `OutputProcessor.process_outputs` | `output_processor.py:L572-L687` | 删去 tokenize/logprobs/stats/make_request_output 字段装配 | 按 req_id 解多路复用 `queue.put` 与 `queue is not None` 分流逐行保留 |
| `RequestOutput.add` | `vllm/outputs.py:L145-L173` | n==1 收敛：DELTA 累加 token_ids / FINAL 覆盖 | merge 语义（背压替代）保留，CompletionOutput.index 配对删 |
| `EngineCoreRequest` / `EngineCoreOutput` | `vllm/v1/engine/__init__.py:L80,L161` | msgspec.Struct -> dataclass，删次要字段 | in-process 传对象无需序列化；`finished` 属性保留（判停依据）|
| `InProcessEngineCore.{add_request_async,get_output_async,abort_requests_async}` | `core_client.py:L1058,L990,L1063` | ZMQ/进程 -> 两条 in-process asyncio.Queue + 模拟引擎背景协程 | **同名同签名**，AsyncLLM 看到的接口语义不变；IPC 留 ch07 |
| `InputProcessor.{process_inputs,assign_request_id}` | `input_processor.py:L234,L215` | tokenize/校验全删，仅产结构正确的 EngineCoreRequest | Stage1 黑盒，内部留 ch05 |
| `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE` | `async_llm.py:L654 (envs)` | 从 env 读，默认 128 | 分块+sleep(0) 让步常量，writer 要讲为何分块 |

## must_keep 自检
dossier 列的 23 个 must_keep 符号全部保留（含 `AsyncLLM`/`generate`/`_add_request`/`output_handler`/
`RequestOutputCollector.{put,get,get_nowait}`/`process_outputs`/`add_request_async`/`get_output_async`/
`EngineCoreRequest`/`EngineCoreOutput`/`abort`/`VLLM_V1_OUTPUT_PROC_CHUNK_SIZE`/`asyncio.sleep(0)`）。
`lint_fidelity.py` 无 BLOCKING。

## 关于 Stage2 stub 是否违反"不加玩具模拟"
角色铁律禁止"加 vLLM 没有的抽象/伪 forward"。这里的 `InProcessEngineCore` **不是新增 AsyncLLM 层
抽象**，而是 dossier `subtraction_plan._principle` **明确批准**的「用 in-process stub 替 IPC」——它顶替
的是真实存在的 `AsyncMPClient`（同名 async 接口），其"模拟产 token"站位的是真实 EngineCore 的
schedule+execute（ch03/ch07 主题），本章 scope 明确不展开。这是减法计划授权的边界替换，不是杜撰。
