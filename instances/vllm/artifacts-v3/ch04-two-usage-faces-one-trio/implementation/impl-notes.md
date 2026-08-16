# ch04 精简版 impl-notes — 两个使用面，一套三件套（API 进程：双登记 + client_index）

- **Pin**：vLLM v0.27.1（`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`）。全部 `# SOURCE:` 行号
  已对当前 pin 逐行现核（2026-08-16，非 v2 资产的 v0.21.0 旧行号）。
- **产物**：`implementation/engine_faces.py`（单模块，host 可跑、无 torch/vllm/zmq/msgspec 依赖）。
- **验收判据**：把真实源码删掉所有 `# SUBTRACTED:` 分支 ≈ 得到本文件（EngineCore 与 MP client 的
  IPC 物理层按 dossier delete 项 1 替换为 in-process stub + 进程内队列，见下）。
- **跑法**：`cd instances/vllm/artifacts-v3/ch04-two-usage-faces-one-trio && python -m pytest tests/ -q`
  → 39 passed。`python scripts/lint_fidelity.py <本章目录>` → 无 BLOCKING。
- **源文件登记**（正文与精简版引用的规范路径全集，供 lint 对照）：`vllm/engine/protocol.py`、
  `vllm/engine/async_llm_engine.py`、`vllm/entrypoints/llm.py`、`vllm/entrypoints/offline_utils.py`、
  `vllm/entrypoints/openai/chat_completion/serving.py`、`vllm/v1/engine/__init__.py`、
  `vllm/v1/engine/llm_engine.py`、`vllm/v1/engine/async_llm.py`、`vllm/v1/engine/core_client.py`、
  `vllm/v1/engine/core.py`、`vllm/v1/engine/input_processor.py`、`vllm/v1/engine/output_processor.py`、
  `vllm/v1/engine/detokenizer.py`、`vllm/v1/core/sched/scheduler.py`、`vllm/envs.py`、`vllm/utils/counter.py`。

## 骨架总览（对齐 dossier subtraction_plan._principle）

保留「两个使用面 + 一套三件套 + 双登记 + client_index 盖章」完整骨架：
- **在线面** `AsyncLLM`（async_llm.py:L72）：`__init__` 三件套构造（L135-156 逐字）、
  `add_request`（快路径 + 懒启动 + 信箱）、`_add_request`（L420-435 双登记**逐字**）、
  `generate`（L596-616 拉信箱循环 + 断连 abort 逐字）、`_run_output_handler`（分块
  process_outputs + stop-string abort）。
- **离线面** `LLMEngine`（llm_engine.py:L48）：`__init__` 三件套构造（L91-111 逐字，
  唯一分叉 `make_client(asyncio_mode=False)`）、`from_engine_args`（L170-186 逐字含
  VLLM_ENABLE_V1_MULTIPROCESSING 翻转）、`add_request`（n==1 双登记 L272-277 逐字）、
  `step()` 四步；皮肤 `OfflineInferenceMixin`（FINAL_ONLY + 自增 id + 事务性批量 +
  `while has_unfinished: step()` 裸循环 + 末尾 `sorted(int(request_id))` 还原）与
  `LLM`（L339-341 引擎接线逐字）。
- **IPC 替换**（delete 项 1 授权）：`EngineCore`＝in-process stub（引擎侧请求表＝双登记的
  「对面半边」；`emit_step_outputs`＝忙碌循环出队 + 输出 IO 线程 `sockets[client_index]`
  路由的融合，F3 兑现端原样可见）；`SyncMPClient`/`AsyncMPClient` 保留 `outputs_queue`、
  阻塞 `get_output()`/`get_output_async()` 与 `add_request_async` 三行逐字（盖章行在内），
  `_send_input` 的 ZMQ/msgpack 体删去、消息 (request_type, request) 经引擎的 in-process
  input queue 过界（镜像 core.py:L1741）。
- **测试扮演调度器**：companion 不伪造 forward（契约明令禁止）；tests/explainer 经
  `emit_step_outputs([(client_index, EngineCoreOutputs)])` 供给一步的产出——真实系统里这
  些元组来自 step_fn（ch9）。

## 1:1 Source Map（精简版 ↔ 真实源码 ↔ 改动 ↔ 原因）

| 精简版符号 | 真实源码锚点（v0.27.1 现核） | 改动 | 原因 |
|---|---|---|---|
| `EngineClient` | vllm/engine/protocol.py:L41 | 保留类骨架 + is_running/is_stopped/errored/dead_error + generate/abort/shutdown/get_supported_tasks；encode/pooling、RL/运维控制面、profiling/sleep/lora 全删 | delete 项 7/8/9；协议面=「serving 只面向抽象」的证据 |
| `AsyncLLMEngine` 别名 | vllm/engine/async_llm_engine.py:L1-L7 | **逐字**（import 改本模块） | must_keep：v0 双引擎遗迹（WC1 old_design 实物） |
| `AsyncLLM.__init__` | async_llm.py:L75-L179 | 三件套构造 L135-156 **逐字**；stat-logger/profiler/tracing/elastic-ep 块删 | delete 项 5/9；m1 主证据 |
| `AsyncLLM.add_request` | async_llm.py:L283-L418 | 保留 errored 预检 + 已渲染 dict 同步快路径 + assign_request_id + 懒启动注释与调用 + 信箱构造 + n==1 派发；流式输入/EngineCoreRequest 直传/raw-prompt 线程池/n>1/reasoning 透传删 | delete 项 3/4/6/7；站 5-7 |
| `AsyncLLM._add_request` | async_llm.py:L420-L435 | **逐字**（parent_req 参数留位不注类型） | must_keep：双登记扇出点、本章最关键 16 行 |
| `AsyncLLM.generate` | async_llm.py:L544-L656 | docstring + 消费循环 L595-606 + CancelledError/GeneratorExit→abort(internal=True) + EngineDeadError + 泛化 Exception→EngineGenerateError + finally close **逐字**；STREAM_FINISHED 检查/VLLMClientError/InputStreamError 臂删 | delete 项 3；站 12 在线端 |
| `AsyncLLM._run_output_handler` | async_llm.py:L657-L727 | 拉 get_output_async→按 VLLM_V1_OUTPUT_PROC_CHUNK_SIZE 分片 process_outputs→片间 sleep(0)→stop-string abort 逐字；iteration_stats/scheduler_stats/logging 删 | delete 项 5/10；m8 心脏 |
| `AsyncLLM.abort` / `LLMEngine.abort_request` | async_llm.py:L729-L741 / llm_engine.py:L212-L216 | **逐字** | must_keep abort：双向清理 |
| `LLMEngine.__init__` | llm_engine.py:L51-L141 | 三件套构造 L91-111 **逐字**（`make_client(asyncio_mode=False)` 在内）；dp_group/StatLoggerManager/v0 model_executor 暴露/mm cache 删 | delete 项 2/5；与 AsyncLLM 并排 diff 的同构证据 |
| `LLMEngine.from_engine_args` | llm_engine.py:L160-L186 | **逐字**（envs 翻转 L174-176 在内） | must_keep VLLM_ENABLE_V1_MULTIPROCESSING；WC4 |
| `LLMEngine.add_request` | llm_engine.py:L218-L294 | request_id 类型校验 + process_inputs 同步调用 + assign_request_id + params 克隆注释 + n==1 双登记 L272-277 **逐字**；直传分支/n>1 删 | delete 项 4/6；站 8 离线同构 |
| `LLMEngine.step` | llm_engine.py:L296-L334 | 四步注释与调用序保留（get_output→process_outputs→abort 停止串→返回 list）；dummy-batch 分支/profiler 包裹/record_stats 删 | delete 项 2/5；站 12 离线端 |
| `EngineCoreClient.make_client` | core_client.py:L89-L112 | **逐字**（2×2 表 + asyncio∧¬mp→NotImplementedError） | must_keep：工厂二轴 |
| `EngineCoreClient.make_async_mp_client` | core_client.py:L114-L139 | 签名与 client_args 逐字；DP>1 分流（DPAsyncMPClient/DPLBAsyncMPClient）删→直落 AsyncMPClient | delete 项 2；client_count/client_index 出生参数保留 |
| `InprocClient` | core_client.py:L306-L402 | docstring（V0-style/no busy loop）+ `__init__=EngineCore(*args)` + add_request（preprocess→add_request 两行）+ abort/shutdown/dp_engines_running 逐字；get_output 的 step_fn 体换 stub socket 拉取 | must_keep：逃生舱对照 |
| `MPClient` | core_client.py:L503-L706 | 类 docstring 逐字；ZMQ ctx/BackgroundResources/启动握手/monitor 线程/engine-rank 簿记全删→构造 stub 引擎 + engine_dead 旗标（monitor 线程的原职责）；shutdown/_format_exception/ensure_alive/dp_engines_running 骨架保留 | delete 项 1/2 |
| `SyncMPClient` | core_client.py:L802-L971 | `outputs_queue = queue.Queue()` **逐字**；EngineCoreOutputQueueThread 守护线程 + shutdown PAIR socket 删→stub 直投 outputs_queue；get_output L872-882 阻塞取逐字（wave_complete 删）；add_request/abort_requests 形逐字（ZMQ 体换 input_queue.put）；call_utility 族删 | delete 项 1/2/8；must_keep outputs_queue/get_output |
| `AsyncMPClient` | core_client.py:L974-L1246 | `client_count/client_index/outputs_queue=asyncio.Queue()` 逐字；`_ensure_output_queue_task` 删成 no-op（add_request_async 第三行原样调用）；get_output_async L1093-1102 逐字；**add_request_async L1145-L1148 三行逐字（client_index 盖章）**；abort_requests_async 形逐字；utility/DP/FT/lora 族删 | delete 项 1/2/8；站 9 主锚点 |
| `EngineCore`（stub） | core.py:L103/L361/L439/L485/L751/L969/L1436-L1444/L1745-L1810 | 引擎侧半边：请求表（双登记账本②）+ input_queue（core.py:L1741 镜像）+ sockets（每前端一个输出槽，L1761-L1766 镜像）+ `emit_step_outputs`（先排空 input 再按 `sockets[client_index]` 路由 + 引擎侧 finished 清账）；调度器/执行器/忙碌循环/IO 线程体删 | delete 项 1 授权的 in-process stub；ch9 边界 |
| `InputProcessor` | input_processor.py:L38-L394 | `assign_request_id` L231-L249 **逐字**；process_inputs 保留已渲染 dict 快路径（arrival_time 缺省+params.clone+EngineCoreRequest 构造点）；校验/raw-prompt 预处理/mm 装配/embeds 提取删 | delete 项 6/7；ch6 边界；站 5-6 |
| `OutputProcessor` | output_processor.py:L429-L727 | `add_request` L525-L554（流式续跑分支删）+ `process_outputs` L589-L711（docstring 的 NOTE FOR DEVELOPERS 逐字；查不到=已 abort 跳过、detokenize、make_request_output、queue 有无分流 L679-L684、finished 清账、stop-string→reqs_to_abort）+ `abort_requests`（外→内展开 + abort 输出）+ `_finish_request` + `propagate_error`；stats/tracing/scheduler_stats/流式删 | delete 项 3/5/9；must_keep ×5 |
| `RequestState` | output_processor.py:L129-L426 | 字段子集（external_req_id/queue/detokenizer/logprobs_processor/output_kind 等）；from_new_request 采样臂保留；make_request_output 的 FINAL_ONLY 早退 L288-290 **逐字**；stream_interval>1 切片/parent 聚合/pooling 臂删 | delete 项 4/7；m7 锚点 |
| `RequestOutputCollector` | output_processor.py:L45-L106 | 单槽信箱 put/get/get_nowait/close 逐字（**DELTA 合并臂保留**——n=1 生产快于消费同样需要合并，删它会丢增量、改变可观察行为，故未按 delete 项 4 的「n>1 防覆盖分支」字面执行，见 §已知偏差 3）；pooling 臂/_input_stream_task/__del__ 删 | must_keep；ch7 深挖 |
| `EngineCoreRequest`/`EngineCoreOutput`/`EngineCoreOutputs` | v1/engine/__init__.py:L97/L184/L230 | 字段子集：client_index（L120-122 注释**逐字**）+ external_req_id（L133-138 注释**逐字**）两枚章在内；msgspec.Struct→dataclass（host 无 msgspec 的机械替换）；current_wave/priority/reasoning_*/pooling/mm/embeds/logprob 载荷字段删 | delete 项 2/3/7；站 9 线格式 |
| `FinishReason`/`EngineCoreRequestType` | v1/engine/__init__.py:L43/L261 | 枚举逐字（ADD/ABORT 两成员保留，DP/UTILITY 哨兵删） | 回程 finish_reason 语义 + 过界消息类型标记 |
| `IncrementalDetokenizer`（基类） | detokenizer.py:L31-L66 | **逐字**（tokenizer=None 分支即真实 skip 路径）；Fast/Slow 子类删 | ch7 边界；host 无 tokenizers 库 |
| `LogprobsProcessor` | logprobs.py:L29-L67, L189, L348 | dataclass 字段 + from_new_request 的 None 路径 + pop_prompt_logprobs 副作用保留；logprob 装配体删 | ch7 边界 |
| `CompletionOutput`/`RequestOutput` | outputs.py:L21/L85 | 字段子集 + `add(aggregate=)` 合并**逐字**；routed_experts/pooling 族/STREAM_FINISHED 删 | collector 合并的载体 |
| `RequestOutputKind`/`SamplingParams` | sampling_params.py:L182/L199 | 三态枚举**逐字**；params 只留 kept 路径读到的字段（n/detokenize/max_tokens/top_p/temperature/num_logprobs/prompt_logprobs/output_kind/stream_interval/skip_clone + clone） | must_keep ×2；ch6 verify 删 |
| `OfflineInferenceMixin` | offline_utils.py:L49-L627 | `_render_and_add_requests`（事务性 try/except abort **逐字**）+ `_add_request`（FINAL_ONLY 注释逐字 + `str(next(self.request_counter))` 逐字）+ `_run_engine`（裸 while 循环 + `sorted(key=int(request_id))` **逐字**）+ `_run_completion` 骨架；tqdm/吞吐统计/render 预处理/priority 透传删 | delete 项 5/7；m9 全部四特征 |
| `LLM` | llm.py:L67, L295-L341, L414-L477 | `LLMEngine.from_engine_args` 接线 L339-L341 **逐字** + `request_counter = Counter()`；EngineArgs 装配（~100 kwargs）删（ch03 章产物）；generate 的 runner_type 校验逐字 | 站 2；ch03/ch06 边界 |

## 删除台账

### dossier subtraction_plan 十项 delete（全部执行，落点见上表）
1. ZMQ/msgpack/多进程物理层 ✓ —— `_send_input` ZMQ 体、BackgroundResources、握手、monitor 线程、
   EngineCoreProc socket 线程全删；stub 保留 `request.client_index = self.client_index` 盖章行（F3 锚点）
2. DP/数据并行协调 ✓ —— DPAsyncMPClient/DPLBAsyncMPClient、current_wave、client_addresses 外部管理、
   `-1` 哨兵 coordinator 分支、engines_running 族
3. 流式输入 ✓ —— `_add_streaming_input_request`、AsyncGenerator 分支、StreamingUpdate/STREAM_FINISHED/
   streaming_input/input_chunk_queue
4. n>1 并行采样扇出 ✓ —— ParentRequest 循环（两面）；**例外**：collector.put 的合并臂保留（见 §已知偏差 3）
5. log_stats/StatLoggerManager/IterationStats/SchedulerStats/update_scheduler_stats/tqdm ✓ ——
   两面构造处的 `log_stats=` 参数穿线保留（签名保真），消费端全删、调用点传 None
6. EngineCoreRequest 直传 deprecated 分支 + raw-prompt 线程池分支 ✓ —— 只留「已渲染 EngineInput → 同步快路径」
7. reasoning 透传/kv_sharing 校验/pooling 任务族/priority+data_parallel_rank 透传 ✓ —— 生成主路径不受影响
8. RL/运维控制面 ✓ —— pause/resume、weight-transfer、scale_elastic_ep、notify_kv_transfer、collective_rpc、
   call_utility*、_logger_ref
9. torch profiler 配置/tracing/otlp ✓ —— errored 语义保留一行骨架（引擎死→add_request 即拒）
10. output_handler 的 scheduler_stats 搬运与错误传播简化 ✓ —— except → propagate_error(e) 一行骨架保留

### 机械删除（不在 delete 单、为可跑性/章节边界所必需——**请 reviewer 逐条过目**）
| 位置 | 内容 | 理由 |
|---|---|---|
| protocol.py L29-L38 | StreamingInput dataclass | delete 项 3 的直接伴生物（generate 联合类型的组成） |
| output_processor.py L498/L516-L522 | lora_states.request_finished / parent 分支 | lora_states=metrics.stats（项 5 同族）；parent=项 4 的参数面 |
| output_processor.py L115-L126, L192-L209, L556-L587 | StreamingUpdate / apply_streaming_update / _update_streaming_request_state | 项 3 的三处定义体 |
| async_llm.py L437-L537 | _add_streaming_input_request + _validate_streaming_input_sampling_params | 项 3 的两面 |
| core_client.py L780-L800 | _process_utility_output | 项 8（utility 结果分发） |
| RequestState 字段 | prompt_embeds/stats/routed_experts_chunks/is_prefilling/num_cached_tokens | embeds=ch6；stats/metrics=项 5；MoE=metrics 轴；is_prefilling 只喂已删的 prefill_stats 搬运 |
| EngineCoreOutput 字段 | events/trace_headers/prefill_stats/routed_experts/num_nans_in_logits/new_logprobs 载荷 | 同上分组；logprob 载荷=ch7 |
| make_request_output 签名 | pooling_output 参数删除 | 项 7（pooling 族）的参数面；调用点同步收窄 |
| InputProcessor.__init__ | 子 config 解包 + InputPreprocessor + mm budget + process_inputs_async 线程池包装 | ch6 黑盒；项 6（async 路径的构造端） |
| LLM.__init__ | EngineArgs 装配 + warmup + default_sampling_params | ch03（装配线）/ch06（渲染）的章节边界 |
| LLM.generate 等 | enqueue/chat/tokenize/pooling 变体 | ch6 变体 + 项 7 |
| SamplingParams 字段 | ~80 字段子集化 | kept 路径只读上表所列；verify()=ch6 |

### seam 清单（host 可跑的最小接缝，均为 vLLM 自身经过的决策口）
`init_logger`（info_once/warning_once）· `envs`（L149/L160/L210 三旗标按 envs.py 现值）·
`random_uuid`（utils L11 逐字）· `Counter`（utils/counter L6）· `as_list`（collection_utils L49 逐字）·
`UsageContext` · `LoRARequest`（仅 lora_name）· `EngineDeadError/EngineGenerateError`
（exceptions.py 逐字，基类 Exception 替 VLLMServerError）· `VllmConfig/ModelConfig/SchedulerConfig/
ParallelConfig/ObservabilityConfig`（字段 seam——完整装配线是 ch03 章产物）· `Executor.get_class`
（ch03 工厂①的标记 seam）· `BaseRenderer/renderer_from_config`（ch6 黑盒；tokenizer=None ⇒
detokenizer 走真实 no-tokenizer 路径 detokenizer.py:L57-L59）· `EngineInput = dict`
（TokensPrompt 的渲染产物）· msgspec.Struct→dataclass（三个线格式结构体，字段序与默认值保持）。

## 测试矩阵（tests/test_engine_faces.py，39 用例）

- **m1 同构**：两面各装出 renderer/InputProcessor/OutputProcessor + 分叉点 client 类型；别名垫片
  `AsyncLLMEngine is AsyncLLM` 且可被子类化（v0 老代码姿势）。
- **m2 工厂**：make_client 2×2 表 + asyncio∧¬mp NotImplementedError；make_async_mp_client 携
  client_count/client_index 出生；from_engine_args 的 envs 翻转（True→SyncMP / False→Inproc 逃生舱）。
- **m5 双轨 id**：外部 id 原样入 external_req_id、内部 id 加 8 位随机 hex；预置 external_req_id→ValueError；
  随机化逃生舱恒等；同一外部 id 重试 8 次内部 id 零碰撞。
- **m3 双登记（在线）**：request_states/external_req_ids（账本①）+ 引擎 input_queue 里带两枚章的
  EngineCoreRequest（账本②过界物）；mp 路径「已过界未入引擎」的中间态可见；inproc 无过界直入。
- **m4/F3**：`emit_step_outputs([(0,…),(1,…)])`→sockets[client_index] 各回各家；client_index=2 的
  add_request_async 盖章 2。
- **m6 分流**：同一 process_outputs——有 queue 投信箱 / 无 queue 收 list；查不到的 req_id 静默跳过
  （防御分支）；finished 后两本账清空。
- **m7 三态**：FINAL_ONLY 中间输出不构造（信箱恒空）终帧才出；DELTA 生产快于消费→collector 合并
  [1]+[2]→[1,2]。
- **abort**：外部 id 展开全部内部 id + 双信箱收到 finished 解锁输出；在线 abort 双侧清账；
  断连（aclose→GeneratorExit）→abort(internal=True)→ABORT 过界消息可检。
- **m8 驱动**：在线 DELTA 两拍产出 [[101],[102]]、外 id 还原、引擎侧账本同步清空；output_handler
  圈外构造不启动/首请求懒启动（OpenAI 优雅启动的 why）；离线裸 while 循环 + 乱序完成（"1" 先完）→
  sorted 还原 [0,1] + FINAL_ONLY 只见终帧；自增 id 前缀 "0-"/"1-"；批量事务性（第二个 add 失败→
  已加的 0 号双侧 abort + ADD/ABORT 消息序可检）。
- **边界**：engine_dead→add_request 即拒 EngineDeadError；SyncMPClient.get_output 的异常格式化；
  InprocClient 直连无过界；process_inputs 快路径（克隆 params/arrival 时钟/非 dict 拒收）。

## 已知偏差（writer/reviewer 需知）

1. **EngineCore＝in-process stub**（delete 项 1 明文授权）：调度/执行为 ch9，本版保留引擎侧请求表 +
   input_queue + per-client 输出槽 + `sockets[client_index]` 路由。引擎侧 finished 清账在
   `emit_step_outputs` 里完成（真实系统在 scheduler，ch9）。
2. **emit_step_outputs 的调用约束**：Async 路径下须在事件循环线程内调用（asyncio.Queue 直投）；
   真实 ZMQ 无此约束——这是删掉 socket task 的代价，已在代码注释标注。
3. **collector.put 合并臂保留**：delete 项 4 列了「collector.put 的 n>1 防覆盖注释分支」，但该臂同时
   承担 DELTA 在 n=1 时「生产快于消费」的合并（删掉=丢增量、可观察行为改变）。按「宁可多留」保留整臂，
   n>1 的扇出端（ParentRequest）仍删。请 analyst 确认此判读。
4. **log_stats/tracing_enabled 参数穿线保留**（两面构造处与 OutputProcessor 签名逐字需要），但其
   消费端（IterationStats/StatLoggerManager/tracing）全删——调用点一律传 None/False。
5. **process_inputs 拒收非 dict prompt**（TypeError，注释标明是删掉 raw-prompt 预处理的守卫）：
   真实代码走 InputPreprocessor（ch6），companion 只收已渲染 EngineInput。
6. **stamp 对齐**：queue=None 走 list（离线）的 `_run_engine` assert isinstance(output, output_type)
   保留；`total_in_toks/out_toks` 与 tqdm 一起删（项 5）。
7. detokenize=False / tokenizer=None 时 IncrementalDetokenizer 基类 update 恒返回 None ⇒
   stop-string 检测路径在 host 上不可触发（真实检测在 BaseIncrementalDetokenizer，ch7）；
   process_outputs 的 stop_string→reqs_to_abort 管线代码保留、由 m7/防御分支测试覆盖其无害性。
