# ch07 精简版实现笔记 — IPC 边界（EngineCoreClient 层级 / ZMQ / 字节标签 / msgpack 多帧 / TensorIpc）

只做减法的忠实子集：与 vLLM v1 同名/同结构/同控制流。源码 pin `f3fef123`。
读者可在 host 直接 `python3 -m pytest tests/` 跑通：ready 握手、字节标签多帧请求、
UTILITY RPC（call_id+Future）、ABORT 双队列、msgpack 多帧零拷贝、TensorIpc 共享内存旁路+乱序重组。

依赖：msgspec / numpy / torch / pyzmq（均 host 可装；无需 import vllm、无需 CUDA）。

## 模块构成

- `engine_init.py` — 支撑结构体的减法子集：`EngineCoreRequestType`（6 字节标签）、
  `EngineCoreReadyResponse`、`EngineCoreRequest`、`EngineCoreOutput(s)`、`UtilityOutput`。
- `serial_utils.py` — **多帧零拷贝主线**：`MsgpackEncoder/Decoder` + `aux_buffers` +
  `_encode_tensor/_decode_tensor`（内联 / aux_buffers / OOB 三分支）+ `OOBTensorConsumer` + `UtilityResult`。
- `tensor_ipc.py` — `TensorIpcSender/Receiver/Data` + `_Sender`（drain-and-buffer 乱序重组）。
- `core.py` — `EngineCore`（最小接缝）+ `EngineCoreProc`：两个 IO 线程
  （`process_input_sockets` DEALER 收 / `process_output_sockets` PUSH 发）+ `input_queue/output_queue`
  + `run_busy_loop` + `_handle_client_request` 字节标签分派 + `ENGINE_CORE_DEAD` 哨兵。
- `core_client.py` — 三层 client：`make_client` 工厂 / `InprocClient` / `MPClient`（ROUTER bind + PULL）
  / `SyncMPClient`（线程版输出处理）/ `AsyncMPClient`（协程版）+ `BackgroundResources`（含 `validate_alive`）
  + `_send_input`（字节标签多帧 send_multipart）+ `call_utility(_async)`（call_id+Future RPC）。

## 1:1 Source Map（精简版 ↔ 真实 vllm ↔ 改动 ↔ 原因）

| 精简版符号 | 真实 vLLM 源 | 改动 | 原因 |
|---|---|---|---|
| `EngineCoreRequestType` | `vllm/v1/engine/__init__.py:L237` | 1:1（6 个 hex 字节标签） | 字节标签协议全章核心，原样保留 |
| `MsgpackEncoder.encode/encode_into/_encode_tensor` | `vllm/v1/serial_utils.py:L166,L180,L257` | 删 mm/slice/pickle 分支；保留 inline/aux_buffers/OOB 三路 | 多帧零拷贝主线（delete 第7项） |
| `MsgpackDecoder.decode/_decode_tensor` | `serial_utils.py:L340,L399` | 删 mm/pickle 解码；保留 aux 索引零拷贝 + OOB dict 旁路 | 对应解码（delete 第7项） |
| `TensorIpcSender/Receiver` | `vllm/v1/engine/tensor_ipc.py:L45,L114` | `torch.mp.Queue`→`queue.Queue`（同 put/get 语义）；删 set_target_engine | 单进程跑通 drain-and-buffer 乱序重组 |
| `EngineCoreProc.process_input_sockets` | `vllm/v1/engine/core.py:L1372` | 删 coordinator XSUB / DP-wave / READY 订阅 | 单 engine：DEALER 发 ready / 收字节标签 / 选 decoder / 投 input_queue（delete 第3项） |
| `EngineCoreProc.process_output_sockets` | `core.py:L1466` | 删 coordinator PUSH（client_index==-1 路径） | PUSH 发 / encode_into 复用 buffer / track 零拷贝 / ENGINE_CORE_DEAD 哨兵（delete 第3项） |
| `EngineCore.run_busy_loop/_process_input_queue/_process_engine_step` | `core.py:L1164,L1174,L1205` | 删 WAITING_FOR_REMOTE_KVS 让步 / draining 状态机 | input_queue→step→output_queue 闭环（delete 第4项） |
| `EngineCore._handle_client_request` | `core.py:L1266` | 删 _reject_*_in_shutdown 排空分支 | 字节标签分派 ADD/ABORT/UTILITY/WAKEUP/EXECUTOR_FAILED（delete 第4项） |
| `EngineCore.step` | `core.py:L402` | 删 scheduler.schedule/executor.execute_model/update_from_output 实体 | 保留"无请求即空返回"短路；不杜撰伪 forward（scope_note：内部属另章） |
| `MPClient.__init__` | `core_client.py:L460` | 子进程→同进程 daemon 线程承载 EngineCoreProc；删 DP 多 rank / monitor 线程 | ROUTER(bind)+PULL / 等 ready 握手 / 装 Encoder+Decoder（delete 第1/3项） |
| `SyncMPClient._send_input` / `AsyncMPClient._send_input(_message)` | `core_client.py:L798,L1001,L1013` | 1:1 | (identity,type,*encoded) 多帧 send_multipart；含张量 track=True + pending_messages |
| `SyncMPClient.call_utility` / `AsyncMPClient.call_utility_async` | `core_client.py:L812,L1038` | 1:1 | call_id(uuid1>>64)+Future 在单向 ZMQ 流上配对（correlation-id） |
| `BackgroundResources.validate_alive` | `core_client.py:L447` | 1:1 | 见 ENGINE_CORE_DEAD 单帧抛 EngineDeadError |

## 偏离真实代码处（已注明，非"减法"而是"承载方式替换"）

1. **进程→线程**：真实 MPClient 经 `launch_core_engines` fork 子进程跑 `EngineCoreProc`；
   精简版在同进程用 daemon 线程承载。ZMQ socket 仍是真实跨 context（tcp://127.0.0.1 临时端口）通信，
   ROUTER↔DEALER / PUSH↔PULL / 字节标签 / 多帧编解码全链路 1:1。读者无需 fork 即可断点追踪。
2. **`torch.mp.Queue`→`queue.Queue`**（TensorIpc）：单进程内 put/get 语义一致，drain-and-buffer 逻辑 1:1。
3. **`EngineCore.step` 不接调度/执行**：scope_note 明确 scheduler/executor 属另章，且铁律禁止伪 forward；
   保留 step 的"无请求即 ({},False)"真实短路，IPC 闭环靠 UTILITY/ADD/ABORT 路径演示，不产假 token。

## 验收判据

把真实 vLLM 删掉所有 `# SUBTRACTED:` 分支（并把 mp.Queue/子进程视作承载等价物），≈ 得到本精简版。
must_keep 全部 34 个符号保留（`lint_fidelity` 校验通过）。tests/ 28 项全绿。
