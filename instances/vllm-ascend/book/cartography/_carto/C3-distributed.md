# C3 — 设备与分布式（device & distributed）测绘 digest

源码 pin: vllm-ascend v0.21.0rc1，前缀 `vllm_ascend/…`。对照基座 vLLM v0.21.0（`vllm/…`）。
本子系统体量 ≈14k+，但**分布很不均匀**：真正"接口对位 vLLM"的核心代码很薄（通信器+并行≈1k 行、device_allocator≈0.3k），而 KV transfer 的 mooncake 三连接器（≈6.6k）和 eplb 的多策略（≈3.3k，含 flashlb 1k/swift 0.75k）占了绝大多数体量、属于"昇腾自带的重型可选特性"。出大纲时应**按解读价值而非行数**排布。

---

## 一、心智模型：OOT 插件如何"顶替 CUDA 版通信/并行"

vllm-ascend 不改 vLLM 源码，而是通过三条路接管分布式路径：

1. **Platform 回调注册通信器**（最干净的对位点）
   `vllm_ascend/platform.py:804` `get_device_communicator_cls()` → 返回字符串
   `"vllm_ascend.distributed.device_communicators.npu_communicator.NPUCommunicator"`。
   vLLM 的 `GroupCoordinator` 在建组时按 platform 给的类名实例化设备通信器——这就是 CUDA 版 `CudaCommunicator` 被 `NPUCommunicator` 顶替的全部机制。**这是讲"OOT 怎么换通信器"的最佳锚点**。

2. **子类化 vLLM 抽象基类**
   `NPUCommunicator(DeviceCommunicatorBase)`（`npu_communicator.py:23`）继承 vLLM 的
   `vllm/distributed/device_communicators/base_device_communicator.py:118 DeviceCommunicatorBase`。
   基类已用 `torch.distributed` 实现了大部分集合通信（all_reduce/all_gather/reduce_scatter，走 `device_group`），昇腾**只需重写 `all_to_all`**（`npu_communicator.py:40`，给 MoE EP 用），其余继承。注意：基类的 all2all_manager / dispatch / combine（EP 专家分发）NPU 暂未接管自定义实现（`ca_comm=None`，TODO 注释 line 32 明说"待集成 PyHcclCommunicator"）。

3. **Patch 改写 torch.distributed 行为**（仅老芯片）
   `vllm_ascend/patch/platform/patch_distributed.py:33 communication_adaptation_310p()` — 仅 310P 芯片，把 `torch.distributed.broadcast/all_reduce` monkey-patch 成 all_gather 模拟（因 310P 不支持某些原生集合算子 / int64 reduce）。A2/A3 不触发。是"OOT 用 patch 补硬件能力缺口"的小而典型样本。

## 二、两套通信路径并存：torch.distributed(HCCL backend) vs 手写 PyHCCL ctypes

- **主路径**：`NPUCommunicator` 全程走 `torch.distributed` + HCCL backend（PyTorch 已识别的昇腾通信后端），与 CUDA 版走 NCCL backend 完全同构。
- **第二路径（手写）**：`device_communicators/pyhccl.py:37 PyHcclCommunicator` + `pyhccl_wrapper.py:113 HCCLLibrary`（ctypes 直接 dlopen libhccl，`Function` 表绑定 `HcclGetRootInfo/HcclCommInitRank/HcclAllReduce/HcclBroadcast/HcclCommDestroy`）。
  这是 vLLM `pynccl.py`/`pynccl_wrapper.py` 的**逐行对位移植**：同样的 unique_id 广播建组、warmup all_reduce、disabled 降级逻辑（`pyhccl.py:71-83`）。**目前 NPUCommunicator 还没把它接进来**（line 32 TODO），但它是讲"为什么 vLLM 需要绕过 torch.distributed 自管通信器（细粒度 stream/graph capture 控制）"的最佳并列对照。教学上 pyhccl↔pynccl 几乎可以并排逐段讲。

## 三、并行状态：在 vLLM 的 GroupCoordinator 之上叠昇腾专属组

`distributed/parallel_state.py:30 init_ascend_model_parallel()`（被 `worker/worker.py` 调用）。
它**不替换** vLLM 的 TP/PP/DP/EP 组，而是**复用** vLLM 的 `init_model_parallel_group` / `GroupCoordinator` 额外建一批昇腾特有组（全局变量 `_MC2/_MLP_TP/_OTP/_LMTP/_EMBED_TP/_FLASHCOMM2_OTP/_FLASHCOMM2_ODP/_FC3_QUANT_X/_SHARD_WEIGHT/_P_TP/_DYNAMIC_EPLB`）：
- `_MC2`（line 95）：昇腾 MC2 融合算子（matmul+通信融合）专属组，EP-like 排布。
- **细粒度 TP**（line 117 `_create_or_get_group`）：把 LM-Head / O-Proj / Embedding / MLP 各自切成独立 TP size（`finegrained_tp_config`），允许"不同模块用不同 TP 宽度"——CUDA 版没有的昇腾优化。
- `_FLASHCOMM2_*` / `_FC3_QUANT_X` / `_SHARD_WEIGHT`：flashcomm（通信-计算重叠）与量化通信专属组。
- `_DYNAMIC_EPLB`（line 98）：给 eplb 做 moe_load all_gather 用。
消费方遍布 `ops/linear_op.py`、`ops/fused_moe/*`、`ops/vocab_parallel_embedding.py`、`quantization/*`。这章讲清"all_ranks 张量 reshape→各组 rank 切分"的排布代数即可，不必逐组深挖。

## 四、device_allocator/camem.py — sleep mode 显存分配器（对位 vLLM cumem）

`device_allocator/camem.py`（273 行）是 vLLM `cumem_allocator`/`CuMemAllocator` 的昇腾移植：基于 CANN 虚拟内存（`vllm_ascend_C.python_create_and_map/unmap_and_release`）实现可插拔 pluggable allocator，支持 sleep mode（offload 权重/丢弃 KV 后释放物理页、唤醒再 map）。`find_loaded_library`/`AllocationData`/`create_and_map` 与 vLLM 版几乎逐行同构，仅把 cudart→acl.rt、libcudart→vllm_ascend_C。还有 `patch/platform/patch_camem_allocator.py` 把它挂进 vLLM。可与 vLLM cumem 并排讲"sleep mode 的虚拟内存机制如何换底座"。

## 五、eplb（Expert 负载均衡）— 昇腾自带的重型可选特性

vLLM 上游尚未合入（源码多处 `Todo: Once vllm PR #xxxxx merged, remove this`，如 `eplb_updator.py:17`、`policy_abstract.py:2`、`policy_factory.py:2`），所以 eplb **整套是 vllm-ascend 自带**、与 vLLM 无直接对位接口（只通过 `GroupCoordinator.all_gather` 和 `dist.P2POp` 借用 vLLM 通信原语）。
- **是什么**：MoE 推理中各 NPU 上的 expert 收到的 token 负载不均；eplb 周期性收集每 expert 的热度（workload），用策略重排 expert 在物理 rank 上的放置（含冗余副本），再用 D2D（device-to-device）异步搬运 expert 权重，达到负载均衡。
- **运行骨架**（讲价值高，控制流清晰）：
  - `eplb_updator.py:31 EplbUpdator` 是节拍器，挂在 model_runner 的 forward 前后：`forward_before()`（line 104，生成并启动 expert 权重 D2D 搬运任务）/ `forward_end()`（line 127，gather moe_load、唤醒规划子进程）。靠 `cur_iterations` 与三个间隔常量（`expert_heat_collection_interval`/`algorithm_execution_interval`/`num_moe_layers`）拼出一个流水线状态机（line 89-99 的 flag 函数）。
  - `eplb/core/eplb_worker.py:? EplbProcess`：独立**子进程**跑重排算法（`planner_q`/`block_update_q` 跨进程队列），避免阻塞主推理流。
  - `eplb/core/eplb_device_transfer_loader.py D2DExpertWeightLoader`：把"哪些 expert 从哪发到哪"翻成异步 P2P 权重搬运。
  - `eplb/adaptor/vllm_adaptor.py VllmEplbAdaptor`：从 vLLM 模型对象取 expert_map / workload，是 eplb↔模型的桥。
- **策略多态**：`policy/policy_factory.py PolicyFactory.generate_policy(type)` → `0 RandomLoadBalance / 1 DefaultEplb / 2 SwiftBalanceEplb / 3 FlashLB`，全实现 `policy_abstract.py EplbPolicy.rebalance_experts(current_expert_table, expert_workload)`。flashlb(1026 行)/swift(751 行)是重型算法，建议只讲接口契约 + DefaultEplb(350 行) 一个代表，其余点到为止。

## 六、KV transfer — 昇腾 PD 分离 / KV 池的连接器实现

`distributed/kv_transfer/` 是 vLLM KV connector v1 框架的昇腾实现层，**对位明确**：
- `ascend_multi_connector.py:20 AscendMultiConnector(MultiConnector, SupportsHMA)` 子类化 vLLM 的 `MultiConnector`，加 HMA（hybrid memory，混合 KV cache 管理器）支持。
- `kv_p2p/mooncake_*.py`（三个，共 6.6k）= `MooncakeConnector/MooncakeHybridConnector/MooncakeLayerwiseConnector`，均子类化 vLLM `KVConnectorBase_V1`（`mooncake_connector.py:31`），用 Mooncake transfer engine 做跨实例 KV 搬运（PD 分离）。体量大但高度同构、模板化，**不建议整章逐行**，挑 layerwise 一个讲"connector 角色(scheduler/worker)+ metadata + 异步 save/load 回调"如何嵌进 vLLM 调度循环即可。
- `kv_pool/`（ascend_store / cpu_offload / lmcache / ucm）= 各种 KV 卸载/池化后端，可选特性，仅需在分布式总览里点名。

## 七、cpu_binding.py — NPU↔CPU 亲和绑定（NUMA/IRQ）

`cpu_binding.py`（544 行）按昇腾芯片型号（A2 topo_affinity / A3 global_slice / 310P）把每个 NPU worker 绑到拓扑就近的 CPU 核（解析 `/proc/self/status` allowed cpus + npu-smi 拓扑），是纯昇腾运维/性能特性，vLLM 无对位。总览里作为"昇腾特有的部署细节"一段即可。

---

## 关键 vLLM 对照表（pairs_with）

| vllm-ascend | 对位 vLLM |
|---|---|
| `NPUCommunicator` | `device_communicators/base_device_communicator.py DeviceCommunicatorBase` / `cuda_communicator.py CudaCommunicator` |
| `pyhccl.py PyHcclCommunicator` / `pyhccl_wrapper.py HCCLLibrary` | `device_communicators/pynccl.py PyNcclCommunicator` / `pynccl_wrapper.py NCCLLibrary` |
| `init_ascend_model_parallel` / `GroupCoordinator` 自定义组 | `distributed/parallel_state.py init_model_parallel_group` / `GroupCoordinator`（复用，非替换） |
| `device_allocator/camem.py CaMemAllocator` | `device_communicators/.. cumem` / `CuMemAllocator`（sleep mode） |
| `kv_transfer/ascend_multi_connector.py` | `kv_transfer/kv_connector/v1/multi_connector.py MultiConnector` |
| `kv_transfer/kv_p2p/mooncake_*.py` | `kv_transfer/kv_connector/v1/base.py KVConnectorBase_V1` |
| `eplb/*` | 无（vLLM 未合入，仅借用 `GroupCoordinator.all_gather` / `dist.P2POp`） |
| `patch/platform/patch_distributed.py`（310P） | monkey-patch `torch.distributed.broadcast/all_reduce` |

## 建议章节骨架（4 章，中等深度；细化交 outline）

1. **「换底座的通信器：从 CudaCommunicator 到 NPUCommunicator」**
   focus: platform 回调注册 → 子类化 DeviceCommunicatorBase → 只重写 all_to_all、其余继承 torch.distributed/HCCL；并排讲手写 pyhccl↔pynccl ctypes 通信器（warmup/disabled 降级/unique_id 建组）；310P patch 补能力缺口。
   key_source_paths: `npu_communicator.py`, `pyhccl.py`, `pyhccl_wrapper.py`, `patch/platform/patch_distributed.py`, `platform.py:804`
   pairs_with: `base_device_communicator.py`, `cuda_communicator.py`, `pynccl.py`, `pynccl_wrapper.py`
   teach_value: 高（OOT 换通信器机制的最干净样本）  est_size: 中  deps: 平台/patch 概览章

2. **「在 vLLM 并行组之上：MC2 / 细粒度 TP / flashcomm 专属通信组」**
   focus: init_ascend_model_parallel 复用 init_model_parallel_group 叠昇腾专属组；all_ranks reshape→各组 rank 切分代数；细粒度 TP（lmhead/oproj/embed/mlp 各自宽度）的动机与消费方。
   key_source_paths: `distributed/parallel_state.py`, `distributed/utils.py`
   pairs_with: `vllm/distributed/parallel_state.py`（GroupCoordinator/init_model_parallel_group）
   teach_value: 中高  est_size: 中  deps: ch1

3. **「Expert 负载均衡（eplb）：子进程规划 + D2D 权重热迁移」**
   focus: EplbUpdator 节拍状态机（forward_before/forward_end + iteration 间隔）→ EplbProcess 子进程跑策略 → D2DExpertWeightLoader 异步 P2P 搬权重 → PolicyFactory 策略多态（讲 DefaultEplb 一个 + 接口契约）。强调这是 vLLM 未合入的昇腾自带特性。
   key_source_paths: `eplb/eplb_updator.py`, `eplb/core/eplb_worker.py`, `eplb/core/eplb_device_transfer_loader.py`, `eplb/adaptor/vllm_adaptor.py`, `eplb/core/policy/{policy_abstract,policy_factory,policy_default_eplb}.py`
   pairs_with: 无直接对位（借 `GroupCoordinator.all_gather` / `dist.P2POp` / `dist.batch_isend_irecv`）
   teach_value: 高（独特、控制流完整、跨进程+异步搬运有看点）  est_size: 中-大  deps: ch2（用 _DYNAMIC_EPLB 组）

4. **「KV 搬运与 sleep mode：连接器、KV 池与 CANN 显存分配器」（分布式杂项总览）**
   focus: AscendMultiConnector/Mooncake 连接器如何子类化 vLLM KVConnectorBase_V1 嵌进调度循环（挑 layerwise 一个讲角色/metadata/异步 save-load）；camem 对位 cumem 的 sleep mode；cpu_binding 亲和绑定点名。
   key_source_paths: `kv_transfer/ascend_multi_connector.py`, `kv_transfer/kv_p2p/mooncake_layerwise_connector.py`, `device_allocator/camem.py`, `cpu_binding.py`
   pairs_with: `multi_connector.py`, `kv_connector/v1/base.py KVConnectorBase_V1`, cumem/`CuMemAllocator`
   teach_value: 中（体量大但模板化，取代表+对位即可，防过度展开）  est_size: 中  deps: ch1

---

## 子系统总结（这子系统怎么把昇腾通信/并行接进 vLLM 主线）

vllm-ascend 几乎不动 vLLM 的分布式骨架：它在 **platform 回调** 处把设备通信器类名换成 `NPUCommunicator`（子类化 `DeviceCommunicatorBase`，绝大多数集合通信直接继承 torch.distributed+HCCL backend，仅重写 MoE 用的 all_to_all），并另备一套逐行移植自 pynccl 的手写 `PyHcclCommunicator`（ctypes 直绑 libhccl）待接入——所以"换底座"在主路径上几乎透明。并行侧则**复用** vLLM 的 `GroupCoordinator`/`init_model_parallel_group`，只是在其上**额外叠** MC2、细粒度 TP、flashcomm、dynamic-eplb 等昇腾专属通信组以喂给昇腾融合算子。真正"加料"的是 vLLM 尚未合入的 **eplb**（子进程规划 + D2D 异步 expert 权重热迁移，仅借用 vLLM 通信原语）和体量庞大但高度模板化的 **Mooncake KV connector / KV 池**（子类化 vLLM KVConnector v1 框架做 PD 分离）。出大纲应按解读价值而非行数排布：通信器/并行组对位最值得精读，eplb 控制流独特值得整章，KV transfer 取代表对位即可。
