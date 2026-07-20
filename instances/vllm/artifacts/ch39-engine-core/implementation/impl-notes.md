# ch33 精简版实现笔记 — 弹性 EP 扩缩状态机 + Responses API 多轮

源码 pin：`f3fef123`。本精简版"只做减法"：与真实 vLLM 同名、同结构、同控制流，
所有删除处标 `# SUBTRACTED:`，所有 def/class 标 `# SOURCE: vllm/...:Lxxx`。

精简版可纯单元测试（不 `import vllm`）：被真实代码委派给 worker 侧
（`model_executor.collective_rpc("elastic_ep_execute", ...)`）与 torch.distributed
集合通信（all_reduce / barrier / 进程组销毁）的部分，本就经 `model_executor` /
`dp_group` 这些**外部协作对象**完成——精简版照样经这些对象调用，由测试注入可
观察替身（FakeExecutor / FakeDPGroup / FakeStore / FakeParser）驱动状态机推进。
没有杜撰任何 vLLM 没有的抽象。

## Source Map（精简版 ↔ 真实源码 ↔ 改动 ↔ 原因）

| 精简版符号 | 真实源码 | 改动 | 原因 |
|---|---|---|---|
| `elastic_state.py::ElasticEPScalingState` + 四个 `_progress_*` + `_staged_barrier` + `_switch_and_prepare` + `handle_notification` + `is_complete` | `vllm/distributed/elastic_ep/elastic_state.py:L82-L588` | 控制流逐字保留；委派给 worker 的 `collective_rpc(...)` 与 torch all_reduce/barrier/destroy 改为经注入对象调用 | 这些在真实代码里也是经外部对象完成；保留调用语义即保真，单进程可测 |
| `ScaleUp{Existing,New}EngineState` / `ScaleDown{Remaining,Removing}EngineState` | `elastic_state.py:L33-L62` | 原样保留四个 IntEnum（9/4/4/3 态） | 状态机骨架，`must_keep` |
| `elastic_state.py::EEPNotificationType` / `ReconfigureRankType` / `ReconfigureDistributedRequest` | `vllm/v1/engine/__init__.py` | 以同名同取值的轻量枚举/数据类复刻（只保留状态机读到的字段） | 真实定义在引擎 `__init__`，import 会触发 vllm；复刻避免 import vllm 而值一致 |
| `elastic_state.py::sched_yield` / `stateless_destroy_torch_distributed_process_group` | `vllm/distributed` | 以等价让步 / 在 dp_group 上调 `.destroy()` 钩子复刻 | 保留"让出调度""销毁旧组"控制流步骤，不引入 torch.distributed |
| `engine_core_eep.py::EngineCore.__init__` / `_initialize_kv_caches` | `vllm/v1/engine/core.py:L80-L255` | 仅保留 `VLLM_ELASTIC_EP_SCALE_UP_LAUNCH` 的 KV-init-前扩分支；其余构造（executor/scheduler/batch_queue/profiling）整体 SUBTRACTED | 本章只读"扩在 KV init 之前"因果，其余构造与本章无关 |
| `engine_core_eep.py::DPEngineCoreProc.run_busy_loop` | `core.py:L1790-L1844` | 只保留 busy loop 的 eep 钩子段（progress()/is_complete()/removing→SystemExit）；ch21 已讲的 DP wave 段 SUBTRACTED | 本章只截到 eep 接缝，wave 交叉引用 ch21 |
| `engine_core_eep.py::reinitialize_distributed` / `_eep_scale_up_before_kv_init` / `eep_handle_engine_core_notification` / `_eep_send_engine_core_notification` | `core.py:L1865-1978` | 逐字保留控制流；通知发送（ZMQ/output_queue）改为注入 sink 回调 | 触发扩缩/转非阻塞/握手是本章主线；ZMQ 传输是正交细节 |
| `responses_multiturn.py::construct_input_messages` | `vllm/entrypoints/openai/responses/utils.py:L79-L121` | 逐字保留；`construct_chat_messages_with_tool_call(list input)` 改为等价逐项透传 | 非 harmony 多轮拼接主线（滤上轮 system、上轮 output→assistant、append 本轮 input）完整保真 |
| `responses_multiturn.py::HarmonyContext.__init__` / `append_output` / `messages` / `render_for_completion` | `vllm/entrypoints/openai/responses/context.py:L523-L713` | 保留 `_messages` 共享 + `append_output` 逐 token process→取 parser.messages→extend；token-usage 统计字段/方法 SUBTRACTED（dossier 批准），parser/render 注入替身 | 主线是"本轮 output extend 进与 msg_store 共享的同一 list"，token 计数是旁路 |
| `responses_multiturn.py::OpenAIServingResponses.create_responses` / `_make_request` / `_construct_input_messages_with_harmony` / `responses_full_generator` | `vllm/entrypoints/openai/responses/serving.py:L318-L893, L1133-L1210` | 保留 prev_response 取回→拼历史→`msg_store[id]=messages`（与 context 共享）→生成→`response_store[id]=response` 主线；background/streaming/event_store/tool_server 探测 SUBTRACTED（dossier 批准） | 有状态多轮闭环主线；正交特性删后非流式同步路径自洽 |

## 关键保真点（测试断言覆盖）

- **不停机推进**：`progress()` 每轮非阻塞推进一步，`WAIT_*` 态返回 `False` 空转，靠
  `handle_notification` 跨进程握手推进（`test_wait_state_is_passive_spin` /
  `test_handle_notification_init_ready_advances_and_counts`）。
- **KV-init 前扩拿统一显存**：新引擎 `PRE_KV_INIT` 经 `sync_kv_cache_memory_size` 得到统一
  额度并写入 `available_gpu_memory_for_kv_cache`，`_initialize_kv_caches` 据此分配而非各自
  profiling（`test_kv_init_with_eep_uses_synced_memory` /
  `test_new_engine_pre_kv_init_syncs_memory_and_sends_weights_ready`）。
- **DP wave 一致性**：`PREPARE` / `_switch_and_prepare` 用 all_reduce MAX 在
  `[engines_running, current_wave, step_counter]` 上对齐（与 ch21 接缝）
  （`test_new_engine_prepare_pulls_wave_state_via_all_reduce_max`）。
- **移除引擎下线**：`_progress_removing_engine` 走 `_switch_and_remove` 后发
  `SHUTDOWN_COMPLETE`，busy loop 见 `worker_type=="removing"` + `is_complete()` → `SystemExit`
  （`test_removing_engine_emits_shutdown_complete` /
  `test_busy_loop_removing_engine_raises_systemexit`）。
- **两阶段 barrier 竞态容忍**：首次 5s 超时 → `compare_set(sync_key)` 返回 `False`，下次用
  `timeout=None` 无超时 barrier 汇合（`test_staged_barrier_first_timeout_sets_sync_key_returns_false`）。
- **多轮共享 list 隐式留存**：`msg_store[id]` 与 `HarmonyContext._messages` 同一对象，
  `append_output` extend 后 msg_store 自动含本轮 output，续轮取回即完整历史
  （`test_append_output_extends_shared_list` / `test_two_turn_conversation_threads_history`）。
- **harmony 续轮 slice-delete-reappend 是 no-op**：如实复刻源码 FIXME 标注的"删了又原样加回"
  （`test_harmony_slice_reappend_is_noop`）——不美化为"清洗 analysis 消息"。

## 测试设置说明（非实现改动）

elastic_state 的若干新组/移除引擎路径在 EPLB barrier 处等待 `dp_group.size()` 个 rank 到齐
且断言 `rank>0`，**单进程测试不可能自满足**。测试用两种忠实手段让其在单进程下可推进，
均与真实多进程行为一致，不改实现：

1. 把"本引擎在新 DP 组中"建模为新组里唯一在跑的 rank0（`new_dp_group` size=1），SWITCH 后
   的新组 barrier 由自身满足。
2. 移除引擎（rank>0）测试预置"其它 rank 已抵达"的 TCPStore arrival key
   （`_seed_other_ranks_arrived`）+ `eep_barrier_engine_count`，模拟真实多进程下其余
   EngineCore 各自 set arrival key 的行为。

存在引擎是"已在运行"的引擎，其 `available_gpu_memory_for_kv_cache` 在原始启动
`_initialize_kv_caches`（`core.py:L251`）时已为正值；`SYNC_KV_CACHE_MEMORY_SIZE` 阶段的
`assert >0`（`elastic_state.py:L496`）正依赖这一点把额度同步给新引擎，故测试在
`make_existing_scale_up` 中将其设为正值。
