# ch21 精简版实现笔记（implementer）

只做减法的忠实子集：与真实 vLLM 同名、同结构、同控制流，只删不增。三大互相
独立的机制——惰性 PP 同步、DP wave 共识、DP 负载均衡——的控制流均为纯 Python，
但其依赖（torch.distributed P2P 句柄、DP-group all-reduce、ZMQ XPUB/XSUB/PULL/PAIR
socket、msgpack 编解码）在 host 上不可用。本章用 `_bridge.py` 提供**语义等价的最小
替身**（可 `.wait()` 的句柄、整数 SUM/MAX all-reduce、socket 帧队列），从而在 host
上**驱动并数值追踪真实控制流**，不引入任何 vLLM 没有的抽象。

> 任何触真实 NCCL / ZMQ / msgspec 的对照运行进 `vllm/vllm-openai` 容器；这里只保
> 留足以跑通逻辑的部分。

## 文件构成

| 精简版文件 | 对应真实 vLLM | 角色 |
| --- | --- | --- |
| `pp_lazy_sync.py` | `vllm/v1/worker/gpu_worker.py` + `vllm/distributed/parallel_state.py` + `vllm/sequence.py` | 惰性 PP 同步：AsyncIntermediateTensors、irecv/isend_tensor_dict、Worker.execute_model |
| `dp_wave.py` | `vllm/v1/engine/core.py` + `vllm/config/parallel.py` | DP wave 状态机：DPEngineCoreProc.run_busy_loop / _has_global_unfinished_reqs / _handle_client_request + sync_dp_state |
| `dp_coordinator.py` | `vllm/v1/engine/coordinator.py` | 协调进程：process_input_socket 三 socket 轮询 + wave 状态机 + START_DP_WAVE 广播 |
| `dp_lb_client.py` | `vllm/v1/engine/core_client.py` | 负载均衡客户端：make_async_mp_client 工厂、DPAsyncMPClient、DPLBAsyncMPClient.get_core_engine_for_request |
| `_bridge.py` | torch.distributed / zmq / msgspec | 句柄/进程组/socket 的 host 替身（**非 vLLM 新抽象**） |

## 1:1 Source Map（精简版符号 ↔ vllm/...:Lxxx ↔ 改动 ↔ 原因）

| 精简版符号 | 真实 vLLM 位置 | 改动 | 原因 |
| --- | --- | --- | --- |
| `AsyncIntermediateTensors.{__init__,wait_for_comm,__getattribute__}` | gpu_worker.py:L74-103 | 原样保留 | 惰性 PP 同步核心：__getattribute__ 拦截 `.tensors` 首访触发 wait_for_comm → handle.wait()+postprocess |
| `PPGroupCoordinator.irecv_tensor_dict` | parallel_state.py:L954-1038 | 删 use_cpu_custom_send_recv 快路径、numel()==0 边界、SP 的 _should_use_all_gather + all_gather postprocess 闭包 | 保留「收元数据→预分配→发 torch.distributed.irecv→返回未 wait 的 handles」非阻塞主线；SP off 时即纯 irecv |
| `PPGroupCoordinator.isend_tensor_dict` | parallel_state.py:L859-914 | 删 CPU custom 路径、numel()==0 skip、SP reshape、cuda record_stream | 保留「发元数据→逐张量 isend→返回 handles」非阻塞发送主线 |
| `Worker.execute_model` | gpu_worker.py:L772-861 | 删 SP 的 all_gather_tensors 计算（L789-816）、pooling pool() 旁支（L836-841）、external_launcher/is_last_rank 断言（L848-852） | 保留 PP 一格全链路：wait 上轮 isend→非首 rank irecv 包成 AsyncIntermediateTensors→forward→非末 rank isend 存 _pp_send_work |
| `ParallelConfig.sync_dp_state` | parallel.py:L666-691 | 原样保留（all_reduce 走 bridge） | 单次 2 元素 SUM all-reduce：[0] SUM>0≡OR 得全局未完成、[1] SUM==dp_size 得 pause 共识 |
| `ParallelConfig.has_unfinished_dp` | parallel.py:L655-664 | 原样保留 | 单元素 MAX all-reduce≡OR，resume_scheduler 屏障对照用 |
| `DPEngineCoreProc.run_busy_loop` | core.py:L1790-1844 | 删外层 `while _handle_shutdown()`+`_process_input_queue`（由测试逐迭代驱动）、eep_scaling_state 弹性 EP 分支 | 保留 step→发布 counts→无 ready 则 dummy batch→32 步 all-reduce 共识→全体空闲 rank0 报 wave_complete、wave++ |
| `DPEngineCoreProc._has_global_unfinished_reqs` | core.py:L1846-1863 | 原样保留 | 每 32 步一次 sync_dp_state；pause 共识达成置 ignore_start_dp_wave、清 pending_pause |
| `DPEngineCoreProc._handle_client_request` | core.py:L1757-1775 | 删 else→super() 基类分发 | 保留 START_DP_WAVE：ignore 则丢弃；非 exclude 且 new_wave>=current 则唤醒 |
| `DPEngineCoreProc.add_request` | core.py:L1710-1724 | 删 super().add_request 基类入队 | 保留 stale-wave 请求→报 start_wave 唤醒 |
| `DPCoordinatorProc.process_input_socket` / `process_events` | coordinator.py:L194-447 | 删 socket 创建/订阅握手/poller 周期发布、stats 乱序检测、SCALE_ELASTIC_EP 分支 | 保留三 socket 轮询：前端新请求唤醒、engine wave_complete→wave++/pause、engine start_wave→推进+广播、stats 更新 counts |
| `DPCoordinatorProc._send_start_wave` / `_get_engine_counts` | coordinator.py:L449-465 | 原样保留 | 广播 START_DP_WAVE（含 exclude 去重）、组装各 engine [waiting,running] |
| `make_async_mp_client` | core_client.py:L107-130 | 删 vllm_config/executor 等构造参 + client_args 打包 | 分支逻辑（dp_size>1 + external_lb 三选一）原样保留 |
| `DPAsyncMPClient.add_request_async` | core_client.py:L1296-1311 | 删 msgpack/ZMQ 实发 | 保留盖 current_wave/client_index→选 engine→engines 暂停时发 FIRST_REQ |
| `DPAsyncMPClient.consume_stats_frame` | core_client.py:L1266-1290 | 删外层 while poller + FIRST_REQ/SCALE 上行 + NOBLOCK drain | 保留消费 (counts,wave,running) 广播→更新 current_wave/engines_running/切片 lb_engines |
| `DPLBAsyncMPClient.get_core_engine_for_request` | core_client.py:L1350-1378 | 删 late-interaction 真实 hash 路由（恒返回 None） | 保留 score=waiting*4+running 选最空、从 eng_start_index 起轮询、本地预增 waiting、记 reqs_in_flight；显式 data_parallel_rank 跳过 LB |

## 关键删减判据（均为 dossier subtraction_plan.delete 批准项）

- 弹性 EP 扩缩容（SCALE_ELASTIC_EP / eep_scaling_state / reinitialize_distributed）全部路径；
- coordinator stats 乱序检测（last_stats_step/wave、last_step_counts）；
- SP（sequence-parallel）的 all_gather_tensors 计算与 irecv/isend 内 all_gather 重组；
- use_cpu_custom_send_recv 同步快路径、numel()==0 空张量分支；
- 弹性 cache、late-interaction pooling hash 路由。

删后三主线（惰性 PP 收发、DP wave 共识状态机、负载均衡评分路由）控制流均完整正确。

## 环境桥接说明（不杜撰 vLLM 没有的东西）

- `_bridge.py` 的 `Handle`（torch.distributed Work 替身）、`FakeDPGroup`（barrier 同步的
  整数 SUM/MAX all-reduce）、`FakeSocket`（XPUB/PULL/PAIR 帧队列）只是把 vLLM 依赖的外部
  原语在 host 上补成**语义等价**最小实现，保留 `.wait()` / all-reduce 数值 / 帧内容与顺序。
- `_Sent` 让 `socket.send(...)` 既可同步（coordinator）又可 `await`（client 的 zmq.asyncio），
  与真实两侧调用形态一致。

## 测试（tests/，纯 host）

- `test_pp_lazy_sync.py`：irecv 返回未 wait 的句柄；首访 `.tensors` 才触发 wait；wait 幂等；
  world_size==1 短路；PP 中间 rank 返回 None 并把 isend 存 _pp_send_work；下一轮开头才 wait
  上轮 isend；首 rank 跳过 irecv；末 rank 返回 model output。
- `test_dp_wave.py`：sync_dp_state 的 OR / pause 共识 / 半数暂停仍 running；MAX≡OR；每 32 步
  才 all-reduce；全体空闲→rank0 报 wave_complete+wave++；running 但无 ready→dummy batch；
  pause 共识置 ignore_start_dp_wave；START_DP_WAVE 唤醒/exclude/ignore；counts 仅变化时发布。
- `test_dp_coordinator.py`：wave_complete→wave++/pause/广播；stale wave_complete 忽略；前端暂停
  时请求→START_DP_WAVE；stale wave→exclude=None 全播；engine start_wave→推进+exclude 报告者；
  stats 更新 counts；_get_engine_counts copy 隔离。
- `test_dp_lb_client.py`：工厂三选一；score 选最空 + waiting 4:1 权重 + 预增打散突发；显式
  dp_rank 跳过 LB；add_request 盖 wave/暂停时发 FIRST_REQ、running 时不发；stats 帧更新状态 + 切片。

运行：

```
python3 -m pytest tests/ -p no:cacheprovider -q     # 41 passed
python3 scripts/lint_fidelity.py instances/vllm/artifacts/ch21-async-engine
```
