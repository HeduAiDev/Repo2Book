# ch09 精简版实现笔记 — EPLB 在线热迁移流水线（subtract-only）

本章解读的真实源码（规范路径，**无 vLLM 基座对位文件**——vLLM 尚未合入此特性，多文件顶部
TODO 写明「待 PR/issue 22246/24069 合入即删除本套实现」）：
- `vllm_ascend/eplb/eplb_updator.py`（主线① 节拍状态机）
- `vllm_ascend/eplb/core/eplb_worker.py`（主线② 子进程 + 跨进程队列 + 规划）
- `vllm_ascend/eplb/core/eplb_device_transfer_loader.py`（主线③ 异步 P2P 三态机）
- `vllm_ascend/eplb/core/policy/{policy_factory,policy_abstract,policy_default_eplb}.py`（主线④ 策略多态）
- `vllm_ascend/eplb/adaptor/vllm_adaptor.py`（loader/updator 的取数契约）

## 总则
- **只做减法**：各 `*.py` 与真实源码同名/同结构/同控制流，主线四块逻辑逐字保留；删除项全部来自
  dossier `subtraction_plan.delete` 批准，且每处标 `# SUBTRACTED:`。
- **可跑边界**：host 无 NPU/CANN/分布式后端。纯 Python 控制流——**节拍状态机**（`cur_iterations` +
  三 flag 整数运算）、**队列解耦**（`multiprocessing.Queue` put/get + 背压）、**策略多态**
  （`PolicyFactory` + `EplbPolicy` 子类 + `DefaultEplb.rebalance_experts` 纯 numpy）、**三态机**
  （`WAITING→READY→TRANSFERRING→WAITING`）——**真跑可断言**。真实 P2P 权重搬运 / 子进程绑 NPU /
  `all_gather` 全局负载 **不真跑**，由 `eplb_runtime_stub.py` 的 record-only 替身接住。
- **仅借用 vLLM 通信原语**：`compute_and_set_moe_load` 走 `comm_group.all_gather`，loader 用
  `dist.P2POp`/`batch_isend_irecv`；`comm_group = get_dynamic_eplb_group()` 即 ch08 建的
  `_DYNAMIC_EPLB`（f6 回收）。这些原语在 host 由桩记录，正文引真实源码讲解。

## 文件
| 文件 | 角色 |
|---|---|
| `eplb_updator.py` | 主线① subtract-only of `vllm_ascend/eplb/eplb_updator.py` |
| `eplb_worker.py` | 主线② subtract-only of `vllm_ascend/eplb/core/eplb_worker.py`（EplbWorker + EplbProcess） |
| `eplb_device_transfer_loader.py` | 主线③ subtract-only of `…/core/eplb_device_transfer_loader.py` |
| `policy_factory.py` / `policy_abstract.py` / `policy_default_eplb.py` | 主线④ subtract-only of `…/core/policy/*` |
| `policy_other.py` | Random/Swift/FlashLB 占位（点到为止：留契约删本体） |
| `vllm_adaptor.py` | subtract-only of `vllm_ascend/eplb/adaptor/vllm_adaptor.py`（仅留非量化代表权重名） |
| `eplb_runtime_stub.py` | 测试接缝桩（NOT subtract-only）：logger / record_function / record-only dist / 借用的 eplb group |

## 1:1 Source Map
| 精简版符号 | 真实源码 `<repo>:Lxxx` | 改动 | 原因 |
|---|---|---|---|
| `EplbUpdator`（cur_iterations + 三 flag + update_iteration） | `eplb_updator.py:L31-L99` | 逐字保留 | 节拍状态机核心，全章主线① |
| `forward_before` / `forward_end` / `compute_and_set_moe_load` | `eplb_updator.py:L104-L148` | 删 record_path 守卫 + multi_stage permute | 落盘/FlashLB 旁支（delete 批准） |
| `warm_up_eplb` | `eplb_updator.py:L150-L174` | 删 dummy P2P 预热环，留 shared_dict 初始化 | 预热通信链路非主线（delete 批准） |
| `EplbWorker.do_update` | `eplb_worker.py:L39-L107` | 删 rank0 hotness/imbalance 监控块 | ms-service-metric，不参与放置计算（delete 批准） |
| `compose_expert_update_info_greedy` / `pack_update_info` | `eplb_worker.py:L148-L288` | 逐字保留 | expert_map -1 差集算 send/recv + 跨进程打包 |
| `EplbProcess`（planner_q/block_update_q/worker_process/_launch_process） | `eplb_worker.py:L325-L388` | 删 ms_metric try/except + flashlb warm_up | 指标/FlashLB 预热非主线（delete 批准） |
| `D2DExpertWeightLoader` 三态机 | `eplb_device_transfer_loader.py:L26-L130` | 逐字保留 | 主线③；`dist` 改 record-only 桩 |
| `PolicyFactory.generate_policy` | `policy_factory.py:L12-L41` | 删 `if policy_type==3: warm_up()` | FlashLB 专属预热（delete 批准） |
| `EplbPolicy.rebalance_experts` | `policy_abstract.py:L6-L30` | 逐字保留 | 唯一接口契约 |
| `DefaultEplb.rebalance_experts` + 五个辅助 | `policy_default_eplb.py:L27-L350` | 删未引用的 `compute_balanced_pack_redundancy`/`compute_balanced_pack` | 替代/历史装箱实现，无引用（delete 批准） |
| `VllmEplbAdaptor` | `vllm_adaptor.py:L29-L169` | 删 W8A8/W4A8/MXFP4/MXFP8 量化分支，留 `['w13_weight','w2_weight']` | 量化只改权重名，不改搬运控制流（delete 批准） |

## 已删除项（均属 dossier `subtraction_plan.delete` 批准）
1. `worker_process` 顶部 ms_service_metric try/except 初始化块。
2. `do_update` 内 rank0 hotness/imbalance 监控块 + `_compute_imbalance`/`_calculate_hotness` 两个 staticmethod。
3. `DefaultEplb` 未引用的 `compute_balanced_pack_redundancy` / `compute_balanced_pack`。
4. multi_stage / policy_type==3(FlashLB) 专属分支：`compute_and_set_moe_load` permute、工厂与子进程的 flashlb `warm_up()`。
5. `VllmEplbAdaptor.init_expert_param_per_layer` 的 W8A8/W4A8/MXFP4/MXFP8 量化分支。
6. `expert_map_record_path` 落盘分支（update_iteration 的 export、forward_end 的 record_path 守卫）。
7. `warm_up_eplb` 的 dummy P2P 预热环。

## must_keep 核对
33 个 must_keep 符号全部保留（`EplbUpdator`/`cur_iterations`/`expert_heat_collection_interval`/
`algorithm_execution_interval`/`num_moe_layers`/`update_iteration`/`get_update_info_flag`/
`wakeup_eplb_worker_flag`/`update_expert_weight_flag`/`wakeup_eplb_worker`/`forward_before`/
`forward_end`/`compute_and_set_moe_load`/`EplbProcess`/`planner_q`/`block_update_q`/`worker_process`/
`_launch_process`/`EplbWorker`/`do_update`/`compose_expert_update_info_greedy`/`pack_update_info`/
`D2DExpertWeightLoader`/`ExpertWeightUpdateState`/`generate_expert_d2d_transfer_task`/
`asyn_expert_weight_transfer`/`update_expert_map_and_weight`/`PolicyFactory`/`generate_policy`/
`EplbPolicy`/`rebalance_experts`/`DefaultEplb`/`VllmEplbAdaptor`）。
`lint_fidelity` 无 BLOCKING；`tests/test_eplb.py` 11 passed。
