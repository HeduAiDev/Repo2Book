# ch20 精简版实现笔记（implementer）

只做减法的忠实子集：与真实 vLLM 同名、同结构、同控制流，只删不增。可在
**host（CPU/gloo，无 CUDA）** 上跑通三大集合原语、PP 的 P2P send/recv、双群组
分流的 broadcast_tensor_dict，以及 MultiprocExecutor 的 collective_rpc 控制面。

## 文件构成

| 精简版文件 | 对应真实 vLLM | 角色 |
| --- | --- | --- |
| `parallel_state.py` | `vllm/distributed/parallel_state.py` | 集合算子 + custom-op 注册 + GroupCoordinator + 单例/访问器 + init 入口 |
| `base_device_communicator.py` | `vllm/distributed/device_communicators/base_device_communicator.py` | device 后端默认实现（torch.distributed on device_group） |
| `communication_op.py` | `vllm/distributed/communication_op.py` | 模型 forward 调用的公共 API（get_tp_group().*） |
| `multiproc_executor.py` | `vllm/v1/executor/multiproc_executor.py` | 多进程编排骨架（collective_rpc 广播/回收、worker_busy_loop） |
| `_env_bridge.py` | `vllm/utils/*`、`vllm/platforms/interface.py` | 环境桥接：补齐 direct_register_custom_op / current_platform / resolve_obj_by_qualname（**非 vLLM 新抽象**） |
| `_mq_bridge.py` | `vllm/distributed/device_communicators/shm_broadcast.py` | MessageQueue 的进程内（线程）替身，保留 enqueue/dequeue/广播语义 |

## 1:1 Source Map（精简版 ↔ vllm/...:Lxxx ↔ 改动 ↔ 原因）

| 精简版符号 | 真实 vLLM 位置 | 改动 | 原因 |
| --- | --- | --- | --- |
| `all_reduce/all_gather/reduce_scatter` + `*_fake` | parallel_state.py:L130-L175 | 原样保留 | 三大集合算子的模块级实现与 meta/fake 形状推断，custom-op 故事的核心 |
| `direct_register_custom_op(...)` ×3 | parallel_state.py:L262-L278 | 删第 4 个注册 | fp8 融合算子 `patched_fused_scaled_matmul_reduce_scatter`（L280-L287）是边缘特性 |
| `GroupCoordinator.__init__` | parallel_state.py:L309-L401 | 删 xpu/out_of_tree 设备分支、mq_broadcaster 实体、use_cpu_custom_send_recv | host 无这些依赖；保留 cuda+cpu 两支与双群组（cpu_group/device_group）构造，CPU 分支使其可在 host 跑 |
| `GroupCoordinator.all_reduce` / `_all_reduce_out_place` | parallel_state.py:L502-L529 | 原样保留 | `world_size==1 短路 → use_custom_op_call ? torch.ops.vllm.* : device_communicator` 二选一是本章主线 |
| `GroupCoordinator.broadcast_tensor_dict` | parallel_state.py:L725-L805 | 原样保留 | 双群组分流最完整示例：metadata 走 cpu_group、tensor 按 is_cpu 选群组 |
| `GroupCoordinator.send/recv/barrier` | parallel_state.py:L1040-L1063 | 删 send_tensor_dict/recv_tensor_dict（字典批量/异步版） | 本章用单张量 P2P 主线讲 PP；barrier 刻意走 cpu_group 的语义保留 |
| `initialize_model_parallel` | parallel_state.py:L1494-L1696 | 删 DCP/PCP/EPLB 维度、enable_elastic_ep、vllm.config 取参 | 保留 5 维 rank 张量 transpose/reshape/unbind 切出 TP/PP/DP/EP 四主维度的套路 |
| `DeviceCommunicatorBase.{all_reduce,all_gather,reduce_scatter,gather,send,recv,broadcast}` | base_device_communicator.py:L180-L310 | 删 stateless-group 分支、all2all/MoE 缓冲 | 默认 torch.distributed 实现（作用在 device_group），CudaCommunicator 覆写但语义一致 |
| `tensor_model_parallel_all_reduce/all_gather/reduce_scatter` | communication_op.py:L14-L31 | 删 `_gather`（同构） | 模型 forward 实际调用的接缝（薄到只是 get_tp_group().*） |
| `MultiprocExecutor.__init__/collective_rpc/_get_output_rank` | multiproc_executor.py:L109-L494 | 删多节点 follower、monitor、cpu-omp、kv_output_aggregator、进程 spawn | 保留 leader 建 rpc_broadcast_mq → spawn worker → 收 response_mqs → enqueue 广播 → 按 output_rank 取结果 → FutureWrapper 有序排空 |
| `WorkerProc.{__init__,worker_busy_loop,handle_output,enqueue_output,ResponseStatus}` | multiproc_executor.py:L539-L970 | worker 用线程承载；删 init_device/load_model/async 输出线程 | 聚焦 RPC 控制流：busy loop dequeue → 反射调用 → 仅 output_rank 回写 response_mq |

## 关键删减判据（均为 dossier subtraction_plan.delete 批准项）

- fp8 融合算子、graph_capture/MoE dispatch-combine、变长 all_gatherv/reduce_scatterv、
  send/recv_tensor_dict 字典版、弹性 EP（StatelessGroupCoordinator）、DCP/PCP/EPLB 维度、
  多节点/monitor/async-scheduling/kv-aggregator。删后四主维度切分、三大集合原语、
  双群组分流、PP P2P、collective_rpc 广播/回收的控制流均完整正确。

## 环境桥接说明（不杜撰 vLLM 没有的东西）

- `_env_bridge.py` / `_mq_bridge.py` 只是把 vLLM 自己的工具（`direct_register_custom_op`、
  `current_platform`、`resolve_obj_by_qualname`、`shm_broadcast.MessageQueue`）在无 vLLM
  的 host 上补齐为**语义等价**的最小实现，不引入任何 vLLM 没有的抽象/数据结构。
- `current_platform.is_cuda_alike()` 默认返回 `False`（host-CPU companion，走 GroupCoordinator
  的 CPU 设备分支，使多 rank 集合原语在单机无多 GPU 时也能跑）。设 `VLLM_CH20_CUDA=1`
  可切回 cuda 分支（需真实多 GPU，应在 `vllm/vllm-openai` 容器内）。

## 测试（tests/）

- `test_fakes_and_custom_op.py`（纯 host）：fake 形状推断、三算子注册为 `torch.ops.vllm.*`、
  meta 设备走 fake、按 group_name 查组。
- `test_group_coordinator_dist.py`（host gloo，spawn 多 rank）：all_reduce 求和 / all_gather
  拼接 / reduce_scatter 切片的数值、broadcast(_tensor_dict) 双群组分流、PP send/recv、
  initialize_model_parallel 的 TP/PP/DP group_ranks 切分、world_size==1 短路。
- `test_multiproc_executor_rpc.py`（进程内线程）：collective_rpc 收齐/按 output_rank 取、
  args 传递、non_block future、worker 异常 → RuntimeError、output_rank 公式、bytes(method)
  路径、futures 有序排空。

运行：

```
python3 -m pytest tests/ -p no:cacheprovider     # 19 passed
python3 scripts/lint_fidelity.py instances/vllm/artifacts/ch20-distributed-parallelism  # 全部通过
```
