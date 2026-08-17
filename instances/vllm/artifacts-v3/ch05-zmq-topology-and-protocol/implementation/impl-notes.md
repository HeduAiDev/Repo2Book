# ch05 精简版 impl-notes — ZMQ 拓扑与消息协议（Part II：L0 紫色边界带放大）

- **Pin**：vLLM v0.27.1（`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`）。全部 `# SOURCE:` 行号已对
  当前 pin 现核（2026-08-16 脚本机械复核 313 个带符号锚点：路径存在、区间不越界、符号确实落在
  区间内——零失配；未照抄 v2 资产的 v0.21.0 旧行号）。
- **产物**：`implementation/zmq_ipc.py`（单模块精简版，~3850 行）+ `implementation/_msgspec_seam.py`
  （msgspec 宿主替身，见 §已知偏差）。host 可跑（真 pyzmq / 真 torch / 真 msgpack / 真 mp spawn
  子进程引擎；仅 msgspec 无包——host 不允许 pip 安装，见 CLAUDE.md 硬规则 6）。
- **跑法**：`cd instances/vllm/artifacts-v3/ch05-zmq-topology-and-protocol && python -m pytest tests/ -q`
  → 47 passed（~15s）。`python scripts/lint_fidelity.py <本章目录>` → 无 BLOCKING。
- **验收判据**：把真实源码删掉所有 `# SUBTRACTED:` 分支 ≈ 得到本模块（HOST/ENGINE SEAM 例外
  见 §Seam 清单——每个 seam 都在行内标注并在此登记）。

## 本章主题 = 全真部分（这是与 ch04 的分工：ch04 删掉 IPC 物理层讲前端，本章把物理层放大全真）

- **socket 拓扑**：client ROUTER(bind)/PULL(bind) ↔ engine DEALER(connect×每前端一条,
  identity=engine_index.to_bytes(2,'little'))/PUSH(connect×每前端一条)，全对 HWM=0
  （`make_zmq_socket` 逐字，RCVHWM/SNDHWM/0.5GB 缓冲/bind 默认规则全保真）。
- **两层启动握手**：专用握手 ROUTER 收 HELLO → 回 `EngineHandshakeMetadata`(地址集) → READY；
  `tcp://host:0` 占位 bind 后 `LAST_ENDPOINT` 回填（#42585 的 port-0 解法原样）。
- **DEALER 先发言**：每条数据 socket 首条消息 = `EngineCoreReadyResponse`（含 post-init 配置回传，
  `_apply_ready_response` 消费）。
- **字节标签线格式**：`EngineCoreRequestType` 单字节 enum 作首帧，消息布局
  (Identity, Type, *Payload 帧)，引擎侧按型选 decoder（ADD 专用 + generic）。
- **msgpack 多帧零拷贝**：`MsgpackEncoder.encode/encode_into`（主帧 + aux_buffers）、
  `_encode_tensor/_decode_tensor` 三分支（内联 RAW_VIEW / OOB 句柄 / aux 索引）逐字；
  256B 阈值（`VLLM_MSGPACK_ZERO_COPY_THRESHOLD`）；客户端 `send_multipart(copy=False)` 由
  zmq 引用链保活（#50053 后无显式 tracker）；引擎输出侧 `encode_into` 复用 bytearray +
  `_send_msg_tracking_payload` 首帧 tracker + `reuse_buffers/pending/max_reuse_bufs` 回收池逐字。
- **OOB 旁路**：`TensorIpcSender`（share_memory_ + mp.Queue + 句柄 dict）与
  `TensorIpcReceiver`（drain-and-buffer 乱序重组 + 过期清理）逐字（minus debug log）。
- **引擎进程编排**：`launch_core_engines`/`CoreEngineProcManager`/`wait_for_engine_startup`
  （mp spawn 子进程、握手 HELLO→READY、哨兵监控）；`EngineCoreProc` 双 IO 线程 + 双
  `queue.Queue` + busy loop（`_process_input_queue` 逐字、`_process_engine_step`、
  `_handle_client_request` 反射 UTILITY 分派逐字）。
- **客户端三实现**：`InprocClient`（进程内对照）/`SyncMPClient`（守护线程输出循环 + PAIR
  shutdown socket）/`AsyncMPClient`（asyncio 输出任务、`add_request_async` 盖 client_index
  三行逐字、`_send_input/_send_input_message` 拼帧 copy=False 逐字）。
- **死讯契约**：`ENGINE_CORE_DEAD` 单帧哨兵 + 输出 PUSH `linger=4000` + `_send_engine_dead`
  join 输出线程 + 客户端 `validate_alive`/`ensure_alive` + monitor 线程兜底。
- **测试扮演调度器**：经**真实** UTILITY 薄 RPC（getattr 反射、call_id 配对 Future 全真）注入
  一步的脚本化产出（`enqueue_step_outputs`）与读取请求簿记（`get_request_info`）——与 ch04 的
  `emit_step_outputs` 同款契约，**不伪造 forward**（契约明令禁止）。

## 1:1 Source Map（精简版 ↔ 真实源码 ↔ 改动 ↔ 原因）

| 精简版符号 | 真实源码锚点（v0.27.1 现核） | 改动 | 原因 |
|---|---|---|---|
| `EngineCoreRequestType` | vllm/v1/engine/__init__.py:L261-L274 | **逐字**（六成员全保留） | must_keep×6；m3 字节标签 |
| `EngineCoreRequest` | vllm/v1/engine/__init__.py:L97-L154 | 字段全保留（msgspec.Struct → seam Struct 同参）；`params` property 逐字 | must_keep；线格式载体（client_index/current_wave 注释逐字） |
| `EngineCoreOutput(s)` | vllm/v1/engine/__init__.py:L184-L258 | 字段全保留；NOTE(Nick) 注释逐字 | must_keep×2；m8 按步聚合 |
| `EngineCoreReadyResponse` | vllm/v1/engine/__init__.py:L68-L94 | **逐字**（dataclass，全字段） | must_keep；m2 认亲+配置回传 |
| `FinishReason` | vllm/v1/engine/__init__.py:L43-L65 | **逐字**（IntEnum+__str__） | must_keep；紧凑序列化例子 |
| `UtilityOutput` | vllm/v1/engine/__init__.py:L218-L227 | **逐字** | must_keep；RPC 回程信封 |
| `make_zmq_socket` | vllm/utils/network_utils.py:L283-L341 | 逐字 minus `router_handover` 参数与 ROUTER_HANDOVER 段（L328-L329 XPUB、L331-L335 IPv6 尾段一并 SUBTRACTED） | delete 项 2（elastic EP→ch39）+ 项 1（XPUB=coordinator）+ IPv6 尾段机械删（依赖 urllib3 的 split_zmq_path，本章拓扑不走 IPv6） |
| `get_engine_zmq_addresses` | vllm/v1/engine/utils.py:L1005-L1048 | 逐字 minus `defer_api_server_ports` kwarg 与 elastic_ep 翻转（L1037-L1040） | delete 项 2；Rust 前端口即回填场景不在本章 |
| `EngineZmqAddresses`/`EngineHandshakeMetadata` | vllm/v1/engine/utils.py:L61-L85 | **逐字** | must_keep×2；m14 握手载荷 |
| `launch_core_engines` | vllm/v1/engine/utils.py:L1053-L1203 | DPCoordinator 运行段/ray 后端/external-LB 双握手链 SUBTRACTED；yield 收窄三元组 | delete 项 1；DP=1 自管路径完整 |
| `wait_for_engine_startup` | vllm/v1/engine/utils.py:L1206-L1346 | remote headless 校验/MoE config-hash 校验/coordinator sentinel SUBTRACTED；**win32 哨兵 HOST SEAM**（见 §Seam） | delete 项 1；HELLO→地址集→READY 主线逐字 |
| `CoreEngineProcManager`/`shutdown(procs)`/`SignalCallback` | vllm/v1/engine/utils.py:L120-L250, L590-L645 | SUBTRACTED：单模块 import 注释；kill_process_tree 为 psutil seam | must_keep；进程编排逐字 |
| `MsgpackEncoder` | vllm/v1/serial_utils.py:L136-L311 | `encode/encode_into/_encode_tensor/_encode_ndarray/enc_hook` 逐字；pickle 回退（L205-L235）与 `_encode_mm_*`（L275-L310）SUBTRACTED | delete 项 7；m4/m5 主线 |
| `MsgpackDecoder` | vllm/v1/serial_utils.py:L313-L483 | `decode/dec_hook/_decode_tensor/_decode_ndarray/ext_hook/_decode_utility_result` 逐字；`_decode_type_info_recursive`/`_decode_mm_*` SUBTRACTED | delete 项 7 |
| `OOBTensorConsumer` | vllm/v1/serial_utils.py:L57-L71 | **逐字**（ABC+双 abstractmethod） | must_keep；OOB 协议抽象 |
| `TensorIpcData`/`TensorIpcSender` | vllm/v1/engine/tensor_ipc.py:L30-L105 | 逐字 minus debug log 块（L87-L96）与失败回退注释保留 | delete 项 6 邻域；m6 |
| `TensorIpcReceiver`/`_Sender` | vllm/v1/engine/tensor_ipc.py:L108-L178 | 逐字 minus debug log | m6 drain-and-buffer |
| `tensor_data` | vllm/v1/utils.py:L777-L787 | **逐字**（uint8 memoryview） | must_keep；零拷贝视图 |
| `EngineCore`（内部=seam） | vllm/v1/engine/core.py:L98-L991 | 构造/调度器/执行器内部=ch06/ch09 章域：`SchedulerSeam`/`UniprocExecutor` 代行；`add_request/abort_requests/step 骨架/_process_aborts_queue/preprocess_add_request/get_supported_tasks` 结构逐字 | delete 项 5/6 + 章边界；busy loop 到 queue 为止（ch9 展开） |
| `EngineCoreProc` | vllm/v1/engine/core.py:L1008-L1042, L1092-L1119, L1129-L1269, L1645-L1827 | `__init__`（双队列/identity/哨兵注入/TensorIpcReceiver）逐字；IO 双线程注释逐字；`_perform_handshakes/startup_handshake` minus remote 分支；`process_input_sockets/process_output_sockets` minus XSUB/coord/-1 哨兵/FT/b"READY" 分支；`_send_msg_tracking_payload` **逐字** | delete 项 1/3；m1/m2/m5/m10/m13 主锚点 |
| `run_busy_loop`/`_process_input_queue`/`_process_engine_step`/`_handle_client_request`/`_invoke_utility_method`/`_convert_msgspec_args` | vllm/v1/engine/core.py:L1377-L1603 | `_process_input_queue` **逐字**；`_handle_client_request` minus `_reject_*_in_shutdown`（项 4）；`_invoke_utility_method/_convert_msgspec_args` **逐字** | must_keep×6；m10/m11/m12 |
| `_send_engine_dead`/`_make_ready_response`/`_handle_request_preproc_error`/`_send_*_outputs` | vllm/v1/engine/core.py:L1605-L1617, L1619-L1643, L1829-L1915 | **逐字**（minus coordinator/mp_queue 伴生） | must_keep；m13 死讯 + m2 认亲帧 |
| `EngineCoreClient`/`make_client`/`make_async_mp_client` | vllm/v1/engine/core_client.py:L78-L139 | make_client 2×2 表 **逐字**（含 asyncio∧¬mp→NotImplementedError）；make_async_mp_client minus DP>1 分流 | must_keep×3；m15 |
| `BackgroundResources` | vllm/v1/engine/core_client.py:L406-L501 | **逐字**（weakref.finalize 无环清理；validate_alive L490-L493 逐字） | must_keep×2 |
| `MPClient` | vllm/v1/engine/core_client.py:L503-L806 | `__init__` minus elastic handover/stats_update_address/coordinator（项 1/2）；ready 收集循环 L645-L671 **逐字**；`_apply_ready_response` minus dp_stats；monitor 线程逐字；utility 面 minus `_process_utility_output` 位置调整（同码） | must_keep×7；站 1-3 |
| `SyncMPClient` | vllm/v1/engine/core_client.py:L802-L972 | `process_outputs_socket` 线程逐字（PAIR shutdown socket + poller）；`get_output/call_utility/add_request/abort_requests` 逐字；具体 utility 方法面删（项 5，get_supported_tasks 留） | must_keep×5 |
| `AsyncMPClient` | vllm/v1/engine/core_client.py:L974-L1247 | `_ensure_output_queue_task/process_outputs_socket` minus EEP/FT 分支（项 2/3）；`get_output_async/_send_input/_send_input_message/call_utility_async/_call_utility_async/add_request_async` **逐字** | must_keep×6；站 5/10 |
| `InprocClient` | vllm/v1/engine/core_client.py:L306-L404 | docstring（"no busy loop"）+ 方法面逐字（get_output 的 step_fn 体走 seam） | must_keep；进程内对照 |
| `_process_utility_output` | vllm/v1/engine/core_client.py:L780-L799 | **逐字**（module 级函数） | must_keep；call_id 配对 |
| `_FINISHED_REASON_MAP` | vllm/v1/request.py:L373-L390 | 映射体逐字（v0.27.1 真名；**非** v2 资产里的 `REQUEST_STATUS_TO_FINISH_REASON`——该名在本 pin 不存在，已纠正） | SchedulerSeam 的 finish→reason 映射 |
| `Request`/`RequestStatus` | vllm/v1/request.py:L60-L82, L349-L371, L223-L247 | 字段 seam（client_index 在内）；`from_engine_core_request/use_structured_output` 逐字（结构） | ch9 章域；m9 路由键载体 |
| `EngineCore` ENGINE SEAM 观测/注入钩 | （无真实对应——ch9 章域注入位 core.py:L1436-L1442） | `enqueue_step_outputs/get_request_info/get_request_embeds_head/boom_method/fail_executor` | 测试扮演调度器/触发 executor 失败的注入口；全走真实 UTILITY 反射 RPC 过线 |

## 删除台账

### dossier subtraction_plan 八项 delete（全部执行）
1. **DP 子类与控制面** ✓ —— DPAsyncMPClient/DPLBAsyncMPClient/DPEngineCoreProc/DPMoEEngineCoreActor/
   coordinator.py 整文件/输入线程 XSUB 分支（core.py:L1669-L1698、L1695-L1698）/输出线程 coord_socket
   与 client_index==-1 哨兵（L1767-L1775、L1788-L1793）/b"READY" 忽略分支（L1706-L1710）/
   wait_for_engine_startup 的 coordinator sentinel+remote headless+MoE hash 段。
2. **弹性 EP** ✓ —— scale_elastic_ep 族/ElasticScalingCache/ReconfigureDistributedRequest/
   `enable_input_socket_handover` 参数链/router_handover 参数（make_zmq_socket、zmq_socket_ctx、
   两处 ROUTER 构造）/get_engine_zmq_addresses 的 elastic 翻转。
3. **fault tolerance** ✓ —— FT_UTILITY_METHOD 拦截（core.py:L1725-L1729）/ft_sentinel/
   FT_STATUS_CALL_ID 消费分支（core_client.py:L1050-L1059 区域）。
4. **shutdown 排空状态机细节** ✓ —— `_reject_add_in_shutdown/_reject_utility_in_shutdown`
   （core.py:L1542-L1567）+ `_handle_shutdown` 的 draining/timeout/pause wait/keep 模式差异；
   WAKEUP 枚举与 no-op 分派保留（must_keep），唤醒编排删。
5. **具体 utility 方法体** ✓ —— sleep/wake_up/collective_rpc/profile/reset_*_cache/add_lora/
   pause_scheduler 等双面（core_client 调用面 + core 实现面 L868-L966、L993-L999）；
   get_supported_tasks 保留作 RPC 示例。
6. **运维旁支** ✓ —— instrument/tracing 装饰器、decorate_logs/set_process_title/numa_bind、
   maybe_init_worker_tracer、`_log_pooler_config`。
7. **serial_utils 不安全回退与多模态工厂** ✓ —— pickle/cloudpickle 回退（L205-L235、L360-L387）、
   `_encode/_decode_type_info_recursive`、`_encode/_decode_mm_*`（L275-L310、L427-L454）；
   enc_hook/dec_hook 分派骨架与 isinstance 行保留。
8. **Rust 前端服务字段消费** ✓ —— `defer_api_server_ports` kwarg 与 #40848 扩展字段的消费链；
   Struct 字段本身未动（两端 schema 契约）。

### 机械删除/替换（不在 delete 单——为可跑性与章边界所必需，**请 reviewer 逐条过目**）
| 位置 | 内容 | 理由 |
|---|---|---|
| 全模块 | msgspec → `_msgspec_seam`（真 msgpack 字节） | host 无 msgspec 包且禁 pip 安装；见 §已知偏差 1 |
| core_client.py L90-L112 周边 | SupportedTask 类型（= str Literal 族）→ tuple[str,...] 简化注解 | 类型面非运行时行为；wire 上就是字符串数组 |
| utils.py L154 | `from vllm.v1.engine.core import EngineCoreProc` | 单模块化（进程 spawn 拾取本模块符号） |
| core.py step() 内部 | scheduler.schedule/execute_model/sample_tokens → `SchedulerSeam.take_scheduled_batch` | ch9 五拍章域；早退条件与 `_process_aborts_queue` 调用位保留 |
| scheduler.py 内部 | 真实 Scheduler → `SchedulerSeam`（requests 簿记 + scripted steps + client_index 分桶镜像 L1924/L2015-L2016 + `_pending_finishes` 行为镜像 has_finished_requests） | ch9 章域；分桶/聚合的**出口结构**与真实一致（dict[client_index, EngineCoreOutputs] + finished_requests 集） |
| multiproc_executor | 真实 executor → `UniprocExecutor` seam（supported_tasks/num_gpu_blocks=128 代行 determine_available_memory/register_failure_callback 真布线） | ch03 工厂①/ch09 章域；executor_fail_callback→EXECUTOR_FAILED 全真 |
| config 族 | VllmConfig/ParallelConfig/CacheConfig/... 字段 seam（仅本章读到的字段） | ch03 装配线产物 |
| logger/envs/version | stdlib seam（NullHandler/默认值常量） | ch04 同款 |

## Seam 清单（HOST SEAM / ENGINE SEAM，全在行内标注）

| Seam | 位置 | 是什么 | 为什么仍忠实 |
|---|---|---|---|
| `_msgspec_seam` | 独立文件 | msgspec API 子集的 msgpack-backed 实现 | 线上字节是真 msgpack；偏差见下节 |
| win32 spawn | `get_mp_context` | 真实默认 fork；win32 只有 spawn | 真实代码在 Linux 跑 fork；spawn 是 mp 标准路径，EngineCoreProc 逻辑不变 |
| win32 ipc:// | `get_open_zmq_ipc_path` | win32 无 ipc:// transport → 回环 tcp | bind-then-connect 流程与 LAST_ENDPOINT 回填不变；Linux 路径逐字 |
| win32 哨兵 | `wait_for_engine_startup` | mp spawn 哨兵是裸 pipe HANDLE，zmq.Poller 会把活子进程误报为立即可读（POLLERR，实测复现）→ win32 不入 poller，改每轮 `finished_procs()` 轮询 | 同一可观察契约：启动期引擎死亡 → RuntimeError("Engine core initialization failed", 失败名单)；Linux 路径逐字 |
| `SchedulerSeam` | EngineCore 内 | 脚本化调度器（测试经真 UTILITY RPC 注入步产出） | 不伪造 forward（契约禁令）；分桶出口结构镜像真实 scheduler.py L1924/L2015-L2028；`has_requests` 含 pending-finish（镜像真实 has_finished_requests——abort 输出必须 riding 下一步的 update_from_output，本节修复见下） |
| `UniprocExecutor` | EngineCore.model_executor | supported_tasks=(...)/num_gpu_blocks=128/失败回调布线 | executor_fail_callback→EXECUTOR_FAILED→死讯全真；真实 executor=ch03/ch09 章域 |
| config/logger/envs/version | 模块头 | 字段/日志/环境旗标 seam | ch04 同款 |

## 已知偏差（reviewer 重点）

1. **msgspec seam**（`_msgspec_seam.py` 头部也自述）：① 内联 Ext 载荷 `bytes(...)` 拷一次
   （真 msgspec 零拷贝传 memoryview）——只影响 <256B 小张量内联路径；② seam decoder 比 msgspec
   宽松（bool/int、未知 map 键、短数组补尾默认值），仅覆盖 pin 上 vLLM 用到的构造；③ Struct
   可变默认值逐实例拷贝（真 msgspec 共享；vLLM 从不改写默认值，行为等价）；④ map-like Struct
   编码**不**省略默认键（真 msgspec omit_defaults=True 会省）——**潜在且未激活**：pin 的四个线载体
   Struct（EngineCoreRequest/EngineCoreOutput/EngineCoreOutputs/UtilityOutput）全是 array_like，
   此章线上无 map-like Struct（dataclass 是另一分支、字节已对验）。**array_like×omit_defaults 语义
   与真 msgspec 一致**：位置数组编码**全部字段**、omit_defaults 只对 map-like 生效（真 msgspec
   0.19.0/0.20.0/0.21.1 容器实测；round-2 修正，见回修记录）。线上格式=标准 msgpack（含
   Ext/RAW_VIEW 自定义类型 3）；**多帧布局（主帧+aux_buffers）与零拷贝语义不受 seam 影响**
   ——那是 vLLM 自己的编解码器逻辑，逐字保留。若需对真 msgspec 验证：在有 msgspec 的环境
   （如 vllm 容器）`import msgspec` 替换 seam 即可跑同一测试面。
2. **UTILITY 回程 tuple→list**：executor 端 `supported_tasks` 是 tuple，过线 msgpack 数组解码回
   **list**——真 vLLM 同此（`-> tuple` 注解只在 InprocClient 路径成立）；测试按 wire 真值断言。
3. **win32 三处**（spawn/ipc/哨兵）如上表；Linux 路径全部逐字，容器内（Linux）三者 seam 均不激活。
4. **utility 回复先于死讯**：`fail_executor` 经 UTILITY 过线时，引擎先回 utility 应答、下一个
   loop 迭代才处理 EXECUTOR_FAILED——与真实控制流一致（回答回程在 dispatch 内同步入 output_queue，
   异常在其后展开）；测试因此必须**注册** call_id（未注册 id 的回复会在输出线程 KeyError——
   同真实 vLLM 行为）。

## 本轮（2026-08-16 续跑）修复记录

前一轮实现被中断，遗留 13 个 E2E 失败/错误，逐一根因为三类并修复（测试 47 全绿、lint 无 BLOCKING）：
1. **win32 哨兵误报**（9 error 的根因）：见 Seam 表第 4 行——zmq.Poller 裸 HANDLE 误报可读 →
   「引擎初始化失败」假阳性。修复：win32 下哨兵不入 poller + 每轮 finished_procs() 轮询。
2. **abort 输出悬挂**（race）：`SchedulerSeam.finish_requests` 立即 pop 请求使 `has_requests()`
   变 False → step() 早退 → pending abort 永不随步发出 → 客户端 get_output 挂死（并连锁让
   many_steps 测试丢首条消息——挂死的僵尸 scenario 从共享 asyncio.Queue 抢走了第一条消息）。
   修复：`has_requests` 含 `_pending_finishes`，镜像真实 scheduler.py:L2406-L2419 的
   `has_unfinished_requests() or has_finished_requests()` 语义（abort riding 下一步输出）。
3. **fail_executor 不可达**：seam 钩只装在 executor 上，UTILITY 反射分派 getattr(EngineCoreProc)
   找不到 → 回 failure_message 而非死亡。修复：EngineCore 加一行转发 `fail_executor`
   （ENGINE SEAM 标注；EXECUTOR_FAILED→死讯全真路径）。
4. **v0.21.0 名残留**：`REQUEST_STATUS_TO_FINISH_REASON` 在本 pin 不存在 → 改回真名
   `_FINISHED_REASON_MAP`（request.py:L382-L390，映射体本就逐字）。
5. **锚点修正**：envs L26→L27、exceptions L10-L22→L12-L21（文件仅 21 行）、close_sockets
   L28→L27、inproc L149-L151→L146-L148、zmq_socket_ctx L345-L373→L346-L370、
   STARTUP_POLL L43→L42、CoreEngineState L46→L45-L49、UniProcExecutor 锚到类头（真实无
   __init__）+ ~94 个 def-span SOURCE 标记补位（lint 契约：SOURCE 须在 def 上一行或 span 内）。
6. **测试侧三处**（断言真值/协议姿势）：get_supported_tasks 按 wire 断言 list；死亡测试注册
   call_id（偏差 4）；`session.run` 传协程对象 → 改传函数+参数。

## 回修记录（round 2 · 2026-08-16 闸门反馈：复现 pin 真实行为，非自洽）

round-1 host 47 全绿 + lint 无 BLOCKING，但按「精简版须复现 pin 真实行为」原则验出 2 项 BLOCKING，定点修复：

1. **F1（BLOCKING，envs seam 默认值错 → Linux 真平台 e2e 全灭）**：
   `VLLM_RPC_BASE_PATH` 曾写 `"/tmp/vllm_rpc"`（v0.21.0 时代旧默认）；pin v0.27.1 真值 =
   `tempfile.gettempdir()`（`vllm/envs.py:L17` TYPE_CHECKING 注解 + `L702-L704` 运行时
   lambda，两处同源 `os.getenv("VLLM_RPC_BASE_PATH", tempfile.gettempdir())`）。Linux 上
   `/tmp/vllm_rpc` 目录不存在 → 每次构造 MPClient 第一条 ROUTER bind 即
   `ZMQError: No such file or directory for ipc path`（win32 ipc→tcp seam 在 host 恰好
   掩盖；容器 Linux 实跑 13 个 e2e 全 ERROR、退出期还因 bind 失败路径泄漏 socket 挂死）。
   修复=一行改 `tempfile.gettempdir()` + `import tempfile` + `# SOURCE` 锚改指
   envs.py:L17+L702-L704（原锚 network_utils.py:L142 只对消费处、默认值出处错了）。
   诊断佐证：容器内仅内存 patch 该值一字 → Linux 47 全绿（fork 真路径 + ipc:// 真传输 +
   poller 哨兵注册真分支），证明其为 Linux 唯一阻断点。
2. **F2（BLOCKING，seam 杜撰 array_like 尾随裁剪线格式）**：seam 曾对 array_like Struct
   做尾随默认值裁剪并在 docstring/测试宣称与 msgspec 同款线格式。真 msgspec（0.19.0/
   0.20.0/0.21.1 容器实测三版一致）对 `array_like=True` 编码**全部字段**——omit_defaults
   只省 map-like 键、对位置数组是 no-op（API 文档未记载该交互，issue #723 佐证省略语义
   不适用 array_like）。pin 的四个线载体 Struct 全是 array_like，故真实 vLLM 线上
   `EngineCoreOutput("r",[1])` 过线为 `["r",[1],None,None,None,0]` 而非 `["r",[1]]`。
   互通性双向实测本就安全（真 msgspec 接受短数组补默认值；seam 接受全字段数组），纯事实
   错误、writer 消费 m4 前必须改。修复：`_encode_struct_values` 不再裁剪（全字段编码，
   `__struct_omit_defaults__` 仅作 kwarg 保真保留）+ seam docstring 更正 + 测试断言改
   `raw == ["r",[1],None,None,None,0]`（并补 EngineCoreOutputs 六字段全过线 + 短数组
   解码宽容两条对照）+ 本文件已知偏差 1 登记（含 map-like 不省默认键的潜在偏差④）。
3. **次要（测试卫生）**：`AsyncEngineSession.shutdown` 补 `client.shutdown(timeout=30)`
   （须在事件循环仍活时调——BackgroundResources 经 call_soon_threadsafe 把 async socket
   关闭排回循环——再停循环），镜像真实 AsyncLLM 收尾顺序；round-1 只停循环靠退出期 GC
   兜底，容器 Linux 下曾致退出需强杀。
4. **免重跑声明（后被 round-3 实跑取代）**：F1 修复即容器诊断 patch 的持久化（该 patch 下
   Linux 47 全绿已预验）；F2 改动仅编码器一侧且双向解码兼容已实测。

**round-3 收口验证（round-2 中断后独立复跑，全部实测）**：
- host：`python -m pytest tests/ -q` → 47 passed in 14.83s；
- 容器 Linux（repo2book/vllm-test:latest，真 fork + ipc:// + poller 哨兵注册路径，
  无任何内存补丁）：`47 passed in 4.84s`，退出干净无需强杀（次要问题③的
  `AsyncEngineSession.shutdown` 补 client.shutdown() 一并生效）——F1 持久化修复在
  pin 真平台全绿；
- `python scripts/lint_fidelity.py 本章目录` → 无 BLOCKING；
- F1 真值对 pin 现核：`vllm/envs.py:L17`（TYPE_CHECKING
  `VLLM_RPC_BASE_PATH: str = tempfile.gettempdir()`）与 `L702-L704`
  （`lambda: os.getenv("VLLM_RPC_BASE_PATH", tempfile.gettempdir())`）两处同源，
  精简版 `VLLM_RPC_BASE_PATH: str = tempfile.gettempdir()` 与之一致。
