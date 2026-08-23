# ch09 精简版 impl-notes — EngineCore 的逐拍循环（Part III：引擎的心跳：调度循环）

- **Pin**：vLLM v0.27.1（`6e448d0ea`）。全部 `# SOURCE:` 行号对当前 pin 现核（正文反复引用的
  17 个机制锚——step/run_busy_loop/_process_input_queue/_process_engine_step/has_work/
  _handle_shutdown/process_input_sockets/process_output_sockets/_send_msg_tracking_payload/
  startup_handshake/_perform_handshake/execute_model/sample_tokens/ExecuteModelState/
  AsyncOutputFuture/get_grammar_bitmask/update_from_output——逐行比对过真实源码；
  未照抄 v2 资产的 v0.21.0 旧行号）。
- **产物**：`implementation/engine_loop.py`（单模块精简版 ~4365 行，386 个 `# SOURCE` 锚 + 108 个
  `# SUBTRACTED` 标记）+ `implementation/_msgspec_seam.py`（msgspec 宿主替身，ch05 同款 seam
  原样复制、行尾归一为 LF）+ `tests/test_engine_loop.py`。
- **跑法**：`cd instances/vllm/artifacts-v3/ch09-engine-core-step-loop && python -m pytest tests/ -q`
  → **48 passed**（~14s：41 个进程内单元测试 + 7 个真 ZMQ 端到端——真 mp spawn 子进程引擎经
  真实 `launch_core_engines`/`CoreEngineProcManager` 出生、真 ROUTER/PULL 前端、两层握手、
  UTILITY 薄 RPC 注入）。host 可跑：真 pyzmq / 真 torch / 真 msgpack / 真 GC freeze；无 vllm 包、
  无 msgspec 包（seam 代行）、无 xgrammar（CPU 内核 seam 代行——容器内真内核优先）。
- **验收判据**：把真实源码删掉所有 `# SUBTRACTED:` 分支 ≈ 得到本模块（HOST/ENGINE SEAM 例外见
  §Seam 清单——每个 seam 行内标注并在此登记）。
- **lint**：`python scripts/lint_fidelity.py <本章目录>` → **全部通过（无 BLOCKING、无警告）**；
  must_keep 49 个符号经 linter `over_subtraction` 项全数核在。

## 本章主题 = 全真部分（与 ch05 的分工：ch05 持有 ZMQ 物理层与前端客户端，本章持有引擎内部）

- **五拍编排（m1/F1 回收）**：`EngineCore.step()` L584-L614 **逐字**（①schedule→②execute_model
  (non_block=True)→③get_grammar_bitmask→④future.result+条件 sample_tokens→aborts→⑤
  update_from_output；仅删 capture_iteration_details 观测附件）。测试以时间戳断言 ③ 严格落在
  ② 之后、④ 之前——bitmask 窗口（F6 埋点）的可观察证据。
- **忙循环骨架（m2）**：`run_busy_loop` L1377-L1389（四拍 minus DP publish 双调用）、
  `has_work` L1365-L1371 三来源判据**完整保留**（engines_running/batch_queue 在 DP/重叠版删除后
  恒 False/None，判据原样）、`_process_input_queue` L1404-L1433 **逐字**（idle 阻塞
  `input_queue.get(block=True)` 不空转 + 空闲清 aborts_queue + block=False 单趟模式）、
  `_process_engine_step` L1435-L1452 **逐字含 1ms GIL 让渡**（dossier 删除项 3 明示保留）、
  `raise SystemExit` 唯一正常出口。
- **两段式契约（m3）**：`ExecuteModelState` 十字段 NamedTuple **逐字**（L437-L450）；worker
  `execute_model` 入口 State error 防御**逐字**（L4171-L4175）+ 暂存/`return None`（L4516-L4535）；
  `sample_tokens` 解包→清态→`apply_grammar_bitmask`→`_sample` 调用位**逐字**（L4553-L4589）；
  `AsyncOutputFuture` **逐字**（L26-L42——result() 只等 `async_output.get_output()` 即 D2H 事件）；
  executor `collective_rpc/execute_model/sample_tokens` **逐字**（L79-L131——non_block=True 完成
  即抛的早失败面 L117-L120）。
- **bitmask 窗口的 worker 半边（m4）**：`apply_grammar_bitmask` **真码**（utils.py:L86-L175 minus
  GPU async H2D 分支）：请求序重排（batch 序 vs bitmask 序）、`torch.full(..., -1)` 排序张量、
  CPU 路径 dtype 转换——在真 numpy 位掩码 + 真 torch logits 上运行；scheduler 侧
  `get_grammar_bitmask` **逐字**（L1646-L1668，含非末块 prefill 排除与 None 快速路径）。
- **启动握手两层（m8）**：`_perform_handshake`（HELLO→yield 窗口内引擎全量构造→READY，
  L1194-L1231）、`startup_handshake`（L1233-L1269 逐字）、数据 DEALER 首条消息
  EngineCoreReadyResponse（L1684-L1693 逐字 + `_make_ready_response` post-init 配置回传）、
  `EngineCoreReadyResponse` dataclass **逐字**（engine/__init__.py:L68-L94）；前端半边
  `launch_core_engines`/`wait_for_engine_startup`/`CoreEngineProcManager` 持有（utils.py 锚）。
- **abort 双通道（m7）**：IO 线程双投递注释**逐字**（L1733-L1738）、`_process_aborts_queue` **逐字**
  （L741-L749）、`finish_requests` 幂等判据**逐字**（L2264-L2268）、`update_from_output` 对已
  abort 请求的跳过分支连注释**逐字**（L1747-L1755——『aborted while the model is executing it』
  的书面案例，e2e 测试直接断言此行为：aborted 请求从后续输出中消失且引擎静默停车）。
- **退出与死讯（m9）**：`_handle_shutdown` abort/drain 两模式**逐字**（L1459-L1505）、
  `_reject_add_in_shutdown`/`_reject_utility_in_shutdown` **保留**（ch05 曾删、本章无对应批准项）、
  信号路径（REQUESTED+WAKEUP 哨兵+『信号处理器不能碰非重入队列锁』注释逐字，L1322-L1340）、
  `ENGINE_CORE_DEAD` 单帧死讯+`_send_engine_dead` join(5s)（L1605-L1617）+ 输出 PUSH
  `linger=4000`——e2e：前向枯竭→忙循环 raise→死讯帧→进程非零退出全链真跑。
- **InprocClient 对照（m10）**：四方法面 + docstring（"no busy loop"）**逐字**
  （core_client.py:L306-L336 minus 管理面）。
- **step_fn 静态绑定（m11）**：L231-L233 **逐字**——`batch_queue_size>1` 时绑定引用
  `step_with_batch_queue`（ch12 的产物，本精简版未携带 → 构造即 AttributeError，测试断言之，
  结构洞见 §已知偏差 5）。
- **慢操作出循环的锚**：`preprocess_add_request` **逐字**（L969-L991，docstring 原话 + 两段线程
  安全注释；mm_receiver_cache 块删）、双 IO 线程装配注释**逐字**（L1092-L1119）、
  `freeze_gc_heap` **逐字**（gc_utils.py:L96-L108）、`enable_envs_cache` 调用位保留。

## 1:1 Source Map（精简版 ↔ 真实源码 ↔ 改动 ↔ 原因；核心行）

| 精简版符号 | 真实源码锚点（v0.27.1 现核） | 改动 | 原因 |
|---|---|---|---|
| `EngineCore.step` | vllm/v1/engine/core.py:L584-L614 | **逐字** minus capture_iteration_details/`_attach_iteration_details`（项 9） | must_keep；m1 五拍本体 |
| `EngineCore.__init__` | core.py:L106-L247 | 插件/日志/executor/freeze/envs 逐字；删 KV 剖析（ch13/17）、EEP（项 2）、spec 旗标（项 7）、mm（项 4）、connector 握手（项 3）、ec/pooling（项 3/8）、前缀哈希器装配（ch13，字段保留 None）、idle 回调表（项 5） | must_keep×4（step_fn/batch_queue_size/freeze_gc_heap/…） |
| `post_step`/`_process_aborts_queue`/`shutdown` | core.py:L616-L623/L741-L749/L751-L767 | post_step 删 draft 分支成直通；其余逐字（cleanup_dist_env_and_memory 为 host seam no-op） | 项 7 + must_keep×2 |
| `add_request`/`abort_requests`/`log_error_detail`/`_should_throttle_prefills` | core.py:L439-L483/L485-L491/L493-L507/L579-L582 | add_request 删 pooling/kv/ec/abort_immediately 校验；其余逐字 | 项 3/8 + must_keep×3 |
| `preprocess_add_request` + SEAM 观测/注入钩 | core.py:L969-L991 | 逐字 minus mm_receiver_cache；钩子（enqueue_forward_logits/enqueue_grammar_bitmask/get_step_count/get_request_info/boom_method/clear_forward_scripts）全走真实 UTILITY 反射 RPC | must_keep + m5 + 测试扮演模型/语法编译器（ch17/ch30 边界） |
| `EngineShutdownState`/`EngineCoreProc` 全类 | core.py:L1002-L1915 | 双队列/identity/EXECUTOR_FAILED 哨兵/握手窗口/双 IO 线程/busy loop 全家/IO 双线程主循环/输出复用池逐字（ch05 已验证的同款渲染再校）；删 DP 统计（项 1）、tensor IPC（项 4）、FT（项 3）、pause 族（项 5）、XSUB/coord 分支（项 1）、DP config-hash（项 1） | must_keep×22 |
| `DPEngineCoreProc`/`EngineCoreActor*` | core.py:L1918-L2488 | **整类删** | delete 项 1 → ch34/39 |
| `SchedulerInterface`（schedule 契约等） | vllm/v1/core/sched/interface.py:L22-L253 | schedule docstring（『busy loop 反复调用』书面契约）+ ①③⑤拍/finish/has_* 契约面逐字；删 draft/pause/reset/counts 抽象面 | m13 + must_keep(schedule/has_requests) |
| `Scheduler`（seam） | vllm/v1/core/sched/scheduler.py:L439-L450、L1325-L1341、L1646-L1668、L1670-L2033、L2213-L2298、L2058-L2111、L2383-L2421 | schedule 头（woosuk 注释）+ `_update_after_schedule` 记账尾 + get_grammar_bitmask + update_from_output 热循环骨架/判停/分桶/finished 簿记 + finish_requests 幂等主线逐字；循环体与抢占（ch10/ch11）、spec（项 7）、stale（项 11）、kv connector（项 3）、routed/stats/events（项 9/30）删；**delete 项 11 批准的同签名最小桩**：waiting 全量 prompt（受 token 预算截为 chunk）、running 补齐/逐 token | ①拍黑盒边界 + m12/m13 |
| `check_stop` | vllm/v1/core/sched/utils.py:L94-L136 | 逐字 minus repetition 检测分支 | 判停主线（LENGTH/STOP/stop_reason） |
| `RequestStatus`/`Request` | vllm/v1/request.py:L348-L390、L60-L307 | RequestStatus 全枚举逐字（is_finished 的 `> PREEMPTED` 序数判据需要完整状态表）；Request 字段 seam（token 记账/append/判停面全真） | 精简调度循环的消费面 |
| `SchedulerOutput`/`GrammarOutput`/`NewRequestData`/`CachedRequestData` | vllm/v1/core/sched/output.py:L33-L291 | 五拍消费字段+注释逐字；ch18 增量协议尾段删 | 线载体 |
| `ExecuteModelState`/`GPUModelRunner` | vllm/v1/worker/gpu_model_runner.py:L437-L450、L4165-L4233、L4516-L4535、L4552-L4589 | NamedTuple 逐字；execute_model 入口/空批早退/暂存逐字、前向深水删（ch17，seam 站位）；sample_tokens 解包→掩码→_sample 逐字、PP/记账深水删 | must_keep×4 + m3 |
| `AsyncGPUModelRunnerOutput` | gpu_model_runner.py:L259-L334 | CUDA copy stream→CPU `threading.Event` seam；get_output 语义（阻塞至拷贝完成/invalid 行清空/回填）逐字 | m3 的 D2H 半边 |
| `apply_grammar_bitmask` + `xgr` seam | vllm/v1/structured_output/utils.py:L86-L175 | 排序/重排/CPU dtype 路径逐字；GPU async H2D 分支删；`xgrammar.apply_token_bitmask_inplace` 为 host seam（位清零→-inf 的内核语义；容器内真内核优先 import） | must_keep + m4 worker 半边 |
| `Sampler.greedy_sample` | vllm/v1/sample/sampler.py:L239-L241 | **逐字**（真采样栈删——sampler 域；测试全 temperature=0，greedy 即该栈的 argmax 分支） | ④拍可观察 |
| `AsyncOutputFuture`/`UniProcExecutor` | vllm/v1/executor/uniproc_executor.py:L26-L147 | AsyncOutputFuture 逐字；executor collective_rpc/execute_model/sample_tokens 逐字、worker 生命周期（ch17）seam | must_keep×3 |
| 线协议载体五件 | vllm/v1/engine/__init__.py:L23-L274 | FinishReason/EngineCoreReadyResponse/EngineCoreRequestType 逐字（minus START_DP_WAVE）；Request/Output(s) 字段全保留、他章域类型放宽 Any | must_keep×3 + 线格式 schema |
| `MsgpackEncoder`/`MsgpackDecoder`/`run_method` | vllm/v1/serial_utils.py:L54-L512 | 单帧子集：多帧零拷贝/tensor/OOB（ch05 持有）与 pickle 回退删；encode/encode_into/dec_hook/ext 拒收逐字 | IO 线程消费面 |
| 前端半边（launcher） | vllm/v1/engine/utils.py:L40-L1346 + network_utils 子集 | ch05 已验证的同款 subtract-only 渲染再校（win32 哨兵/ipc→tcp HOST SEAM 同款）；DP coordinator/ray/external-LB/elastic-EP/GPU 绑定删（项 1/2/9） | m8 前半 + 端到端测试载体 |
| `EngineCoreClient`/`InprocClient` | vllm/v1/engine/core_client.py:L78-L336 | ABC 壳 + InprocClient 四方法逐字；SyncMP/AsyncMP/DP 系不入 | must_keep×2 + delete 项 12 |

## 删除台账 — dossier subtraction_plan 12 项 delete（全部执行）

1. **DP 全家** ✓ — DPEngineCoreProc/EngineCoreActorMixin/DPMoEEngineCoreActor/EngineCoreActor 整类；
   wave/START_DP_WAVE（枚举位注释保留）/`_maybe_publish_request_counts` 双调用（run_busy_loop 内
   SUBTRACTED 行标注）/publish_dp_lb_stats/last_counts/coordinator XSUB+b"READY"/has_coordinator/
   config-hash/external-LB 双握手/DP 标题与 kv 改名。engines_running 字段与 has_work 三来源判据
   **保留**（DP=1 下恒 False，判据原样）。
2. **EEP/FT** ✓ — `_eep_scale_up_before_kv_init`/`_eep_send_engine_core_notification`/
   VLLM_ELASTIC_EP_SCALE_UP_LAUNCH 分支/`fault_tolerant_wrapper` 装饰器/ft_sentinel/
   FT_UTILITY_METHOD 拦截/EEP 通知类型。
3. **KV/EC connector** ✓ — xfer_handshake_metadata 收集/is_ec_consumer/kv_connector_output 透传
   （sample_tokens 的 None 特例分支与 update 的 kv 分支）/get_kv_connector 族。**保留**
   `_process_engine_step` 的 1ms GIL 让渡（dossier 明示）。
4. **tensor IPC** ✓ — TensorIpcReceiver/tensor_queue 全链（含 CoreEngineProcManager 的 kwarg 与
   launch 的队列创建）；encoder/decoder 的张量/OOB 面。
5. **pause/sleep 全族** ✓ — pause/resume/sleep/wake_up/reset_caches/_reset_caches/
   `_idle_state_callbacks`/`_notify_idle_state_callbacks`（`_process_input_queue` 内调用点
   SUBTRACTED 标注）/EngineCoreProc.pause_scheduler/_pause_complete/PauseState 消费面（枚举本体
   保留在 interface 契约里）。
6. **UTILITY 转发方法群** ✓ — LoRA 四方法/save_sharded_state/profile/reset_*/collective_rpc 等
   具体方法体；`_handle_client_request` 的 UTILITY 反射分派骨架逐字保留（getattr + Future 回送）。
7. **spec decode** ✓ — check_for_draft_tokens/take_draft_token_ids/post_step draft 分支/
   update_draft_token_ids*/ExecuteModelState 的 spec 消费分支/sample_tokens 的 drafter 段。
8. **step_with_batch_queue** ✓ — 整方法删（→ ch12 的精简版）；精简版固定
   `max_concurrent_batches=1` 走同步 step()——**教学选择非源码事实**，binding 表达式逐字保留
   （见 §已知偏差 5 的结构洞）。
9. **观测旁路** ✓ — instrument/tracer/numa/decorate_logs/set_process_title/
   capture_iteration_details/_make_iteration_details_stats/_attach_iteration_details/
   `_maybe_publish_request_counts`（项 1 重叠）/perf_stats/routed_experts/KV events/GC debug 钩。
   **保留** log_error_detail→dump_engine_exception（两段式暂存的故障兜底主线）与 freeze_gc_heap。
10. **gpu_model_runner 深水枝** ✓ — ngram copy（L4180-L4195）、ExecuteModelState 非采样字段的
    装配值（字段名全保留，值为 None）、AsyncGPUModelRunnerOutput 的 D2H 深水（CPU 事件 seam）、
    `_update_states_after_model_execute`/bookkeeping 内部（ch17/18）。
11. **scheduler 深水区** ✓ — RUNNING/WAITING 循环体与抢占（→ ch10/ch11 持有）、update 的 stale
    drain/spec 回扣/failed_kv_load/finished 事件簿记外深水。**同签名最小桩**获 dossier 明示批准
    （『两段式契约在桩 executor 上同样可观察』）。
12. **core_client 侧** ✓ — InprocClient 只保留 get_output/add_request/abort_requests/shutdown
    四方法；SyncMPClient/AsyncMPClient/DP 系/BackgroundResources 不入（前端客户端是 ch05 的
    精简版）。

## Seam 清单（行内标注 HOST SEAM / ENGINE SEAM）

| Seam | 位置 | 是什么 | 为什么仍忠实 |
|---|---|---|---|
| `_msgspec_seam` | 独立文件（ch05 同款） | msgspec API 子集的 msgpack-backed 实现 | 线上字节是真 msgpack |
| `GPUModelRunner._seam_logits`/`_pop_scripted_rows` | §worker | 脚本化前向：每步 `{req_id: logits 行}`，5ms 等待建模一步前向 | ch17 边界：环内不伪造 forward——测试经**真实 UTILITY 反射 RPC** 喂行，环只消费脚本；脚本枯竭走真实错误路径（忙循环 raise→死讯）。行按 req_id 键控 → 批组成竞态无关 |
| `xgr`（_XgrammarSeam） | §utils | `apply_token_bitmask_inplace` 的 CPU 内核替身 | 文档语义（位清零→-inf）一致；容器内真 xgrammar 优先 import 生效 |
| `Sampler`（只含 greedy_sample） | §worker | 采样栈 argmax 分支 | greedy_sample 本体逐字（sampler.py:L239-L241）；测试全 temperature=0 |
| `InputBatch` | §worker | 持久批消费面（req_ids 序 + 新落批/完成清退/末块判定） | apply_grammar_bitmask 的真实重排基准；批同步语义镜像 `_update_states`（空拍也同步——真实顺序在 0-token 早退之前） |
| `StructuredOutputManager` | §sched | grammar_init no-op + grammar_bitmask 脚本队列 | ch30 边界；③拍调用位与请求序真实 |
| `Scheduler` 循环体 | §sched | 最小 token 账（waiting 全量 prompt 受预算、running 补齐/逐 1） | delete 项 11 批准；头/尾记账逐字 |
| `UniProcExecutor._init_executor` | §executor | driver_worker=GPUModelRunner + num_gpu_blocks 给定值 128 | worker 生命周期/determine_available_memory 是 ch17 域 |
| `AsyncGPUModelRunnerOutput` D2H | §worker | CUDA copy stream→`threading.Event` + 预备 CPU 缓冲 | get_output 语义（阻塞至拷贝完成）逐字 |
| config/params/request 字段 seam | §config | VllmConfig 族/SamplingParams/Request 字段子集 | 装配线是 ch03 产品；`structured_output_request` 为 ch30 注入位（过线真值标记） |
| `logger`/`envs`/`kill_process_tree`/`get_mp_context`(win32 spawn)/`ipc→tcp`/win32 哨兵轮询 | §host | stdlib 替身/平台回退 | ch04/ch05 同款先例 |
| MsgpackEncoder/Decoder 单帧化 | §serial | 多帧零拷贝/张量/OOB 面 ch05 持有 | 本章线载荷全是 msgpack 原生类型；encode_into 复用 bytearray 的面保留（m6） |

## 已知偏差（reviewer 重点）

1. **忙循环的单机节拍**：真前向 ~几毫秒；seam 前向 `time.sleep(0.005)` 建模一步——这让 e2e 的
   abort 时序（wire 往返 vs 步进速度）与真实引擎同数量级，避免『环跑得比网快』的伪竞态。
2. **max_concurrent_batches>1 是结构洞**：binding 表达式逐字保留，但 `step_with_batch_queue`
   方法本体属 ch12——构造即 `AttributeError: step_with_batch_queue`（测试断言之）。这是批准删除
   （项 8）的机械后果，与『删掉批准分支后的真码』严格一致。
3. **`sample_tokens` 的 None-state 结构洞**：execute_model_state 为 None 的 PP/kv-conn 特例分支
   删（项 3）——两段式下测试不触达；直接调用会解包 TypeError（同 ch05/ch06/ch07 的结构洞惯例）。
4. **中止通知的口径**：v0.27.1 非 DP 下，客户端发起的 abort **不**产生引擎→客户端输出
  （`_send_abort_outputs` 只在 shutdown abort 模式/预处理错误/关停拒收时发）——前端本地已知道。
   本章测试断言的正是这个真实行为（请求从后续输出中消失）。ch05 的 seam 曾把 abort 输出挂在下一步
   消息上（其 impl-notes 已注明是镜像近似）；本章按 pin 真码收窄。
5. **同步 step() 是教学选择**：v0.27.1 服务默认心跳是重叠版（async_scheduling 默认 True、
   `Enable async scheduling unless there is an incompatible option`，config/vllm.py:L1095-L1143）。
   精简版 config seam 的 `async_scheduling` 默认 **False**（m11 教学顺序：同步四段骨架是理解重叠
   版的唯一地基）；binding 与异步包裹骨架（`if not self.use_async_scheduling: return output`）
   真码保留，AsyncOutputFuture 的 D2H 等待有专测（executor.sample_tokens(non_block=True) 路径）。
6. **grammar_bitmask 需脚本**：结构化输出请求的位掩码行由测试经 UTILITY 注入（ch30 的 FSM 编译
   不在本章）；脚本枯竭即引擎死——错误路径真实。
7. **`get_supported_tasks` 返回 tuple 过线回 list**（msgpack Any 解码语义）；utility 失败路径
   `Call to X method failed: …` 逐字。

## 测试面（48 passed；断言 pin 可观察行为，非自洽）

- **m1**：五拍调用序（scheduler/runner 双 trace + perf_counter 时间戳交叉断言 ②→③→④ 顺序）、
  `non_block=True` 经 executor spy、空转守卫({}, False)、0-token 拍不过采样（EMPTY 判据）、
  throttle 钩直通 False。
- **m2**：idle 停靠（400ms 内 step 计数冻结→ADD 唤醒）、block=False 单趟、
  run_busy_loop 退出即 SystemExit、e2e 静默期 step 计数冻结。
- **m3**：State error 原文逐字（runner 直驱 + executor in-line 早抛双面）、暂存/解包清态、空批
  EMPTY、十字段 NamedTuple 形状、AsyncOutputFuture 只等 D2H（事件未置 →result() 阻塞、置位即返、
  二次 result 瞬时）、timeout 未实现、异常传播、异步调度路径经 executor 拿 AsyncOutputFuture。
- **m4**：无结构化请求 ③ 全程运行但 manager 零调用、预算截断的非末块 prefill 被排除且无部分
  prefill 输出（不变量）、位掩码压过 argmax（真 apply_grammar_bitmask+真 argmax：禁 5 选 4）、
  ②→③→apply→sample 时间戳窗口。
- **m7/m9**：aborts_queue 批合并（单次调用）、scheduler 幂等（二次返回 []）、WAKEUP 哨兵 no-op、
  abort 模式（in-flight 全 ABORTED+按 client 路由+flush 后退出）、drain 模式（有活继续步进）、
  关停拒收 ADD、EXECUTOR_FAILED 哨兵 raise、死讯单帧+join、e2e abort 静默+幂等回声+info None。
- **m13/m12**：token 账（全量 prompt→逐 1；num_computed 追赶 num_tokens）、预算截 chunk 与
  is_prefill_chunk 真语义、混相批 {req_id:num_tokens}、append+判停（LENGTH/STOP+stop_reason）、
  abort 中执行的跳过（真注释案例）、双 client 分桶。
- **m8/m6 e2e**：两层握手（HELLO→地址集→READY）+ 每数据 DEALER 首条 EngineCoreReadyResponse
  （post-init 配置回传字段逐一断言）、engine_index 盖章、两前端按 client_index 路由。
- **m10**：InprocClient docstring 'no busy loop' + get_output 直驱 step_fn 两拍到完成。
- **m11**：默认绑定 step_fn==step/batch_queue None；max_concurrent_batches=2 → 结构洞 AttributeError。

## 收工审计（2026-08-22）

- `python -m pytest tests/ -q` → **48 passed**（两次连跑稳定，~14s）。
- `python scripts/lint_fidelity.py 本章目录` → **全部通过**（must_keep 49 符号 over_subtraction
  空账；无发明标记；SUBTRACTED 108 处）。
- 行尾：三份产物均 LF-only（`data.count(b'\r\n') == 0`）。
- 全部 `# SOURCE:` 锚经脚本核过「def/class 跨度内含锚」（linter 同款判据）。
