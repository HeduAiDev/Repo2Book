# ch08 精简版实现笔记 — init_ascend_model_parallel 的加法式扩展

本章解读的真实源码（规范路径）：
- `vllm_ascend/distributed/parallel_state.py`（主角：init_ascend_model_parallel + MC2/细粒度 TP/flashcomm2/CP 各组排布代数）
- `vllm/distributed/parallel_state.py`（对照基座：被复用的 GroupCoordinator / init_model_parallel_group / initialize_model_parallel）
- `vllm_ascend/worker/worker.py`（调用方旁证：何时调 init_ascend_model_parallel 建组）

## 总则
- **只做减法**：`ascend_parallel_state.py` 与 `vllm_ascend/distributed/parallel_state.py` 同名/同结构/同控制流，MC2 / 细粒度 TP / flashcomm2 三段排布代数**逐字保留**。
- **可跑**：host 无 NPU/CANN，真实 hccl 进程组创建（`GroupCoordinator.__init__` 的 `new_group` + device_communicator）被 SUBTRACTED，换成记录 `group_ranks` 的桩（`vllm_distributed_base.py`）。reshape/transpose/slice 是纯 torch CPU 运算，**真跑**——测的就是切出的 `group_ranks` 与 dossier 手推一致。
- **复用而非替换**：昇腾 `import` 的 `GroupCoordinator / init_model_parallel_group / get_world_group / get_tp_group` 全来自基座桩；昇腾各组都是基座 `GroupCoordinator` 的实例。

## 文件
| 文件 | 角色 |
|---|---|
| `ascend_parallel_state.py` | 本章主角，subtract-only of `vllm_ascend/distributed/parallel_state.py` |
| `vllm_distributed_base.py` | 基座接缝桩：`GroupCoordinator`/`init_model_parallel_group`/world·tp·pcp·dcp getter + 基座 `initialize_model_parallel`(TP/DCP/PCP) |
| `ascend_runtime_stub.py` | `get_ascend_config`/`flashcomm2_enable`/`ParallelConfig` 桩（注入并行度/门控） |

## 1:1 Source Map
| 精简版符号 | 真实源码 `<repo>:Lxxx` | 改动 | 原因 |
|---|---|---|---|
| `init_ascend_model_parallel` | `vllm_ascend/distributed/parallel_state.py:L30-L52` | 删 PD 分离 `_P_TP`(L54-L82) + shard_weight(L191-L226)；`world_size`/`backend` 改取 world group 桩 | PD/shard_weight 与四主题正交（delete 批准）；真实 hccl 运行期不可跑 |
| MC2 块（`_MC2`/`_DYNAMIC_EPLB`/`_FC3_QUANT_X`） | `…parallel_state.py:L84-L108` | 逐字保留 | 排布代数核心 + `_DYNAMIC_EPLB` 复用 group_ranks 样例 |
| `_create_or_get_group` / `rank_grid` / `num_chunks` | `…parallel_state.py:L117-L149` | 逐字保留 | 细粒度 TP 沿 DP 切块取列的算法本体 |
| flashcomm2 双层循环 | `…parallel_state.py:L151-L189` | 逐字保留 | strided otp + odp 排布 |
| `get_*_group` / `model_parallel_initialized` / `destroy_*` | `…parallel_state.py:L229-L341` | 删 `get_shard_weight_group`/`get_p_tp_group` 及对应销毁块 | 对应已删的组 |
| `GroupCoordinator` | `vllm/distributed/parallel_state.py:L290-L317` | `__init__` 主体 SUBTRACTED 为记录桩 | host 无 hccl，真实进程组不可建（delete 批准） |
| `init_model_parallel_group` | `vllm/distributed/parallel_state.py:L1159-L1174` | 逐字保留 | 「复用」工厂接缝，每组都调它 |
| `initialize_model_parallel`(基座 CP) | `vllm/distributed/parallel_state.py:L1494-L1633` | 仅留 TP/DCP/PCP，删 `_PP`/`_DP`/elastic_ep | CP 归口只需 PCP/DCP 排布 |

## 已删除项（均属 dossier `subtraction_plan.delete` 批准）
1. PD 分离 `_P_TP` / alltoall 头复制分支（L54-L82）。
2. `shard_weight` 组 + `create_shard_weight_group`（L191-L226）。
3. `FinegrainedTPConfig` 的 `olora_tensor_parallel_size`（ascend_config，不建组）。
4. `init_model_parallel_group`/`GroupCoordinator` 的真实进程组创建（new_group/device_communicator）。

## must_keep 核对
27 个 must_keep 符号全部保留（`init_ascend_model_parallel`/`model_parallel_initialized`/`all_ranks`/`global_pcp_size`/`transpose`/`reshape`/`unbind`/`_MC2`/`get_mc2_group`/`_create_or_get_group`/`rank_grid`/`num_chunks`/`_OTP`/`_LMTP`/`_EMBED_TP`/`_MLP_TP`/`get_mlp_tp_group`/`get_lmhead_tp_group`/`_DYNAMIC_EPLB`/`get_dynamic_eplb_group`/`_FLASHCOMM2_OTP`/`_FLASHCOMM2_ODP`/`flashcomm2_enable`/`init_model_parallel_group`/`GroupCoordinator`/`get_world_group`/`get_tp_group`）。
`lint_fidelity` 无 BLOCKING。
