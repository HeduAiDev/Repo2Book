# ch29 精简版实现笔记 — PD 分离的抽象与调度器集成

只做减法：与 vLLM 同名/同结构/同控制流，只删不增。source pin `f3fef123`。
纯单元测试（不 import vllm），host `python3 -m pytest tests/` 即可，18 passed。

## 1:1 Source Map

| 精简版符号 | 真实 vllm 源码 | 改动 | 原因 |
|---|---|---|---|
| `base.KVConnectorRole` | `vllm/distributed/kv_transfer/kv_connector/v1/base.py:L123` | 原样 | role-split 契约的根：SCHEDULER=0 / WORKER=1 |
| `base.KVConnectorMetadata` | base.py:L140 | 原样 | 决策侧→搬运侧单向不透明信使 |
| `base.KVConnectorBase_V1` | base.py:L170 | 保留六个 abstractmethod（start_load_kv/wait_for_layer_load/save_kv_layer/wait_for_save/get_num_new_matched_tokens/update_state_after_alloc/build_connector_meta）+ get_finished/request_finished/update_connector_output/bind_connector_metadata；删 register_*/handshake/stats/cudagraph/HMA 等可选钩子 | 本章讲 role-split 骨架，可选扩展点基类皆 no-op |
| `factory.KVConnectorFactory` | `vllm/distributed/kv_transfer/kv_connector/factory.py:L27` | 保留懒加载注册表 + 按 role 单独构造；删 HMA 校验、compat 旧两参签名、外部 module_path 路径 | role 分流是本章主线，兼容/HMA 正交 |
| `factory.create_connector` | factory.py:L42 | 删 compat_sig 分支，只走新三参签名 | 演示 role 在此分流即可 |
| `example_connector.ExampleConnector` | `vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L85` | 决策侧方法原样；worker 侧 inject/extract 删 MLA/Triton 后端分支留默认一支；safetensors+.cuda() 换 torch.save/load host 占位；删 mm_hashes / scheduled_cached_reqs(resumed) 路径 | host 无 CUDA；后端分支/mm/resumed 与契约正交 |
| `example_connector.get_num_new_matched_tokens` | example_connector.py:L262 | 原样（含 block 粒度对齐 align_to_block_size((n-1))） | 决策侧查远程命中入口 |
| `example_connector.build_connector_meta` | example_connector.py:L311 | 删 scheduled_cached_reqs(resumed) 段 | new_reqs 的 load/store 分流已足以演示 |
| `request.RequestStatus` | `vllm/v1/request.py:L310` | 保留 WAITING/WAITING_FOR_*/RUNNING/PREEMPTED/FINISHED_STOPPED + is_finished；删其余 FINISHED_* 细分与 get_finished_reason | WAITING_FOR_REMOTE_KVS 是 f12 核心阻塞态 |
| `request_queue.{SchedulingPolicy,FCFSRequestQueue,create_request_queue}` | `vllm/v1/core/sched/request_queue.py:L13/L75/L201` | 保留 FCFS deque 操作；删 PriorityRequestQueue/抽象基类完整清单 | 双队列选取只需 FCFS 一支演示 |
| `scheduler.Scheduler.__init__` | `vllm/v1/core/sched/scheduler.py:L118/L167/L183` | 仅留 connector 构造 + waiting/skipped_waiting 双队列 + finished_recving/failed_recving 状态集 | KV-connector 集成的状态根基 |
| `scheduler.Scheduler.schedule` | scheduler.py:L568-L934 (WAITING 循环) | 保留查命中→隔离→提升主控制流；删 lora/encoder/chunked-prefill/mamba/stats/SchedulerOutput 大装配 | f12 闭环主线，其余调度约束正交 |
| `scheduler._try_promote_blocked_waiting_request` | scheduler.py:L2061 | 只留 WAITING_FOR_REMOTE_KVS 分支 | 提升回 WAITING/PREEMPTED 的判定点 |
| `scheduler._update_waiting_for_remote_kv` | scheduler.py:L2027 | 原样（含整 prompt 命中回退一 token、failed 路径） | KV 到位缓存 block 的实际副作用 |
| `scheduler._update_from_kv_xfer_finished` | scheduler.py:L2094 | 原样 | 消化 worker 回传 finished_recving/finished_sending |
| `scheduler._connector_finished` | scheduler.py:L1996 | 删 SupportsHMA 分支，只走单 group request_finished | 请求结束 connector 接管异步释放 |
| `scheduler._select_waiting_queue_for_scheduling` / `_is_blocked_waiting_status` | scheduler.py:L1567 / L1553 | 原样 | 双队列选取 + 阻塞态判定 |

## 验收判据
把真实 vLLM scheduler 的 WAITING 循环删掉所有 SUBTRACTED 分支（lora/encoder/chunked-prefill/mamba/stats/ec_connector/HMA），≈ 得到本精简版的 `schedule()`。控制流（查命中→ext_tokens None 隔离→load_kv_async num_new_tokens=0→allocate_slots(delay_cache_blocks)→update_state_after_alloc→置 WAITING_FOR_REMOTE_KVS 隔离→末尾 build_connector_meta）逐行对应。

## host 占位说明（唯一非「纯删除」之处）
- `ExampleConnector._save_file/_load_file`：safetensors.torch.{save,load}_file 在无 safetensors/CUDA 的 host 不可用，用 torch.save/load 等价占位（语义一致：存/取一个 `kv_cache` 张量）。inject/extract 的 `.cuda()` 一并去掉。这不改变 connector 契约与调度控制流，仅让精简版可在 host 跑通。
