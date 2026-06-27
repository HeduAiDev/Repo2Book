# C4 — Worker / 执行主线（vllm-ascend v0.21.0rc1）

> 子系统规模 ~12.6k 行（worker.py 41KB + model_runner_v1.py 244KB + ascend_forward_context 15KB + 周边）。
> 本书的**执行脊柱**：昇腾版如何顶替 vLLM 的 CUDA 执行栈。对照基座 = vLLM v0.21.0 `vllm/v1/worker/`。
> 源码 pin：vllm-ascend `instances/vllm-ascend/source/`（前缀 `vllm_ascend/…`）；vLLM `instances/vllm/source/`（前缀 `vllm/…`）。

---

## 主线一句话

vllm-ascend 用**两种顶替策略**接管执行栈：
- **NPUWorker(WorkerBase)** —— **从 WorkerBase 重写**（不继承 vLLM `Worker`）。逐方法对位 `vllm/v1/worker/gpu_worker.py:Worker`：`init_device`/`determine_available_memory`/`execute_model`/`compile_or_warm_up_model`，把 `torch.cuda`→`torch.npu`、CUDAGraph→ACLGraph、新增 ATB 预热/静态 kernel/torch_npu profiler。
- **NPUModelRunner(GPUModelRunner)** —— **直接继承** vLLM `gpu_model_runner.py:GPUModelRunner`（244KB 巨类），只 override NPU 必须改的方法（`_init_device_properties`/`_use_aclgraph`/`initialize_kv_cache_tensors`/`_reshape_kv_cache_tensors`/`capture_model`/`_check_and_update_cudagraph_mode`/`get_kv_cache_spec`/`execute_model`），其余复用父类。复用父类的关键魔法 = **运行时把 CUDA API 猴补成 NPU**（见下"对照点 ★"）。

这是 OOT 插件"最大化复用 vLLM 上游 + 局部顶替设备相关层"的教科书样例：Worker 层重写（设备生命周期差异大），ModelRunner 层继承 + 猴补（控制流可复用，仅设备原语顶替）。

---

## 关键文件 / 类

| 文件 | 类/函数 | 职责 |
|---|---|---|
| `vllm_ascend/worker/worker.py` (41KB) | `NPUWorker(WorkerBase)` L81 | 进程级执行主控：设备/内存/分布式/编译预热/execute 派发 |
| `vllm_ascend/worker/model_runner_v1.py` (244KB) | `NPUModelRunner(GPUModelRunner)` L255 | 单步前向主控：input batch→attn metadata→forward→sample；继承父类 override 设备相关 |
| 同上 | `graph_capture()` L203 / `GraphCaptureContext` L198 | NPU 版图捕获上下文（`torch.npu.Stream`），顶替 vLLM 同名 |
| 同上 | `_torch_cuda_wrapper()` L4891 / `_replace_gpu_model_runner_function_wrapper()` L4936 | ★ **核心猴补**：临时把 `torch.cuda.*`→`torch.npu.*`、模块级 `graph_capture`/`CUDAGraphWrapper`→NPU/ACLGraph 版，使父类 `capture_model`/`profile_cudagraph_memory` 原样跑在 NPU 上 |
| `vllm_ascend/ascend_forward_context.py` (15KB) | `set_ascend_forward_context()` L57 / `MoECommType` L26 / `select_moe_comm_method()` L233 | 顶替 vLLM `set_forward_context`：包一层再注入 MoE 通信方式/flashcomm/SP/mc2 mask 等昇腾专属上下文 |
| `vllm_ascend/flash_common3_context.py` (1.3KB) | `FlashCommon3Context` | MoE FlashComm3 的 per-forward 旁路上下文（gate/topk/shared_experts），纯昇腾，无 vLLM 对应 |

---

## 关键 vLLM 对照点（implementer/writer 的减法基准）

★ = 最值得在正文展开的对位。

1. **★ init_device** — `NPUWorker._init_device` L260 / `init_device` L317 ↔ `gpu_worker.py:Worker.init_device` L239。
   - vLLM：`torch.device("cuda:…")` + `torch.accelerator.set_device_index` + `init_worker_distributed_environment`。
   - 昇腾：`torch.device("npu:…")` + `torch.npu.set_device`；额外 lazy import `torch_npu._inductor`(triton)、A5 设备特判 `setup_ascend_local_comm_res`、`init_device_properties_triton()`、`init_workspace_manager`、然后在 `init_device` 里 new `NPUModelRunner`（或 v2）。
2. **★ determine_available_memory** — `NPUWorker.determine_available_memory` L335 ↔ `gpu_worker.py:Worker.determine_available_memory` L354。
   - 结构同：`memory_profiling` 包 `profile_run()`，但用 `torch.npu.memory_stats`；新增 **ACLGraph 显存估算** `profile_cudagraph_memory()`（DeepSeek-V4 DSA 压缩注意力特判跳过），并据此回退建议 `--gpu-memory-utilization`。
3. **★ CUDAGraph → ACLGraph** — `capture_model` L4820 / `profile_cudagraph_memory` L4798 / `_check_and_update_cudagraph_mode` L4768 ↔ vLLM `gpu_model_runner.py:capture_model` L6150 / `profile_cudagraph_memory` L6049 / `_check_and_update_cudagraph_mode` L6456。
   - **复用父类实现**，靠 `_torch_cuda_wrapper()`+`_replace_gpu_model_runner_function_wrapper(parent_module)` 在调用周围把 cuda 原语和 `graph_capture`/`CUDAGraphWrapper` 替成 NPU/`ACLGraphWrapper`。
   - `_use_aclgraph()` L620：`cudagraph_mode!=NONE and mode==VLLM_COMPILE and not enforce_eager`。
   - `set_graph_params(capture_sizes)`/`set_draft_graph_params` 在 mode 决定后注入 NPU 侧图参数。
4. **execute_model 派发** — `NPUWorker.execute_model` L474 ↔ `gpu_worker.py:Worker.execute_model` L783；真正前向在 `NPUModelRunner.execute_model` L1904（override 父类 L? ，加 routed-experts capturer/profiling timing/PCP+MM deepcopy 等昇腾分支）。PP 收发用 `irecv/isend_tensor_dict` + `AsyncIntermediateTensors`（从 vLLM `gpu_worker` 导入复用）。`enable_sp()` 影响 all_gather_group。
5. **forward context** — `set_ascend_forward_context` L57 ↔ vLLM `set_forward_context`。昇腾**包一层**：进父类 ctx 后注入 `moe_comm_type/moe_comm_method`(`select_moe_comm_method` L233)、`flash_comm_v1/v2_enabled`、`mmrs_fusion`、`sinks`、`input_ids`、mc2 mask（`set_mc2_mask`/`get_mc2_mask`）。DP 同步 `_sync_metadata_across_dp` L627 把 num_tokens+cudagraph_mode 打包 all_reduce（昇腾要额外同步 2 个标志，且可走 NPU device group 防 CPU all_reduce 脏数据）。
6. **KV cache 初始化** — `initialize_kv_cache_tensors` L3764 / `_allocate_kv_cache_tensors` L3929 / `_reshape_kv_cache_tensors` L4144 / `get_kv_cache_spec` L4657 ↔ vLLM 同名 L6839/L6678/L7058。昇腾新增：int8/sparse-c8 cache、内存对齐 `_align_memory`/`_align_up`、DeepSeek-V4 层序绑定、hamming sparse、longcat 双 attn module。

---

## 建议章节（4 章，中等深度）

### 章 W1 — NPUWorker：从 WorkerBase 重写执行主控（设备 / 内存 / 编译预热）
- **focus**：进程级执行生命周期。`init_device`(npu set_device + workspace + triton inductor) → `determine_available_memory`(npu memory_profiling + ACLGraph 显存估算) → `compile_or_warm_up_model`(warmup sizes + ATB 预热 `_warm_up_atb`) → `execute_model` 派发到 ModelRunner。讲清"为何 Worker 选择重写而非继承"。
- **key_source_paths**：`vllm_ascend/worker/worker.py`（L81 `NPUWorker.__init__`, L260 `_init_device`, L317 `init_device`, L335 `determine_available_memory`, L474 `execute_model`, L557 `compile_or_warm_up_model`, L661 `_warm_up_atb`）。
- **pairs_with**：`vllm/v1/worker/gpu_worker.py:Worker.init_device`(L239)/`determine_available_memory`(L354)/`execute_model`(L783)/`compile_or_warm_up_model`(L574)；对应 **vLLM 书 ch17（GPU Worker）**。
- **teach_value**：高。OOT 插件如何对位重写设备生命周期；ACLGraph 显存预算是昇腾独有工程细节。
- **est_size**：~850 行。
- **deps**：C 平台/distributed 章（init_ascend_model_parallel、device type）；前置 vLLM 书 ch17 心智模型。

### 章 W2 — NPUModelRunner：继承 GPUModelRunner + 运行时 CUDA→NPU 猴补
- **focus**：本书最具"OOT 插件味"的一章。讲 `NPUModelRunner(GPUModelRunner)` 如何**继承 244KB 父类**，只 override 设备相关方法；核心揭秘 `_torch_cuda_wrapper()` + `_replace_gpu_model_runner_function_wrapper()` —— 临时把 `torch.cuda.Event/Stream/synchronize/mem_get_info` 和模块级 `graph_capture`/`CUDAGraphWrapper` 替成 NPU/ACLGraph 版，让父类 `capture_model`/`profile_cudagraph_memory` 原样跑在昇腾上。__init__ 的 PCP padding hack、`use_compress` 早设、AscendSampler/AscendAttentionState 替换。
- **key_source_paths**：`vllm_ascend/worker/model_runner_v1.py`（L255 `__init__`, L198 `GraphCaptureContext`/L203 `graph_capture`, L4891 `_torch_cuda_wrapper`, L4936 `_replace_gpu_model_runner_function_wrapper`, L4820 `capture_model`, L4768 `_check_and_update_cudagraph_mode`, L620 `_use_aclgraph`, L580 `_init_device_properties`）。
- **pairs_with**：`vllm/v1/worker/gpu_model_runner.py:GPUModelRunner`(L399)、`capture_model`(L6150)、`profile_cudagraph_memory`(L6049)、`_check_and_update_cudagraph_mode`(L6456)；对应 **vLLM 书 ch18（GPU ModelRunner）+ ch19（CUDAGraph 捕获）**。
- **teach_value**：极高（**全书最值得讲的两章之一**）。继承+猴补这条路线是 vllm-ascend 区别于"全量 fork"的精髓。
- **est_size**：~900 行。
- **deps**：W1（Worker 创建它）；C ACLGraph/attention backend 章。

### 章 W3 — 单步前向：execute_model / forward context / DP 同步
- **focus**：一次 `execute_model` 的真实数据流。`NPUModelRunner.execute_model` L1904（routed-experts capturer、profiling timing、PCP+MM deepcopy 分支）→ `_prepare_inputs` L748 → `_build_attention_metadata` L2942 → `set_ascend_forward_context` L57（注入 MoE 通信方式/flashcomm/SP/mc2 mask）→ `_model_forward` L2756 → `_sample` L2553/`sample_tokens` L2338。DP 跨卡 `_sync_metadata_across_dp` L627 打包 all_reduce(num_tokens+cudagraph_mode)。
- **key_source_paths**：`vllm_ascend/worker/model_runner_v1.py`（L1904 execute_model, L748 _prepare_inputs, L2942 _build_attention_metadata, L2756 _model_forward, L627 _sync_metadata_across_dp）；`vllm_ascend/ascend_forward_context.py`（L57 set_ascend_forward_context, L233 select_moe_comm_method）。
- **pairs_with**：`vllm/v1/worker/gpu_model_runner.py:GPUModelRunner.execute_model` + vLLM `set_forward_context`；对应 **vLLM 书 ch18**（前向主线）。
- **teach_value**：高。展示昇腾在复用父类前向骨架上"注入了哪些昇腾专属上下文"。
- **est_size**：~800 行。
- **deps**：W2；C MoE 通信/attention 章（moe_comm_method/AscendAttentionState 落地在那）。

### 章 W4 — KV cache 在昇腾上的落地（分配 / reshape / 绑定）
- **focus**：内存几何差异。`initialize_kv_cache_tensors` L3764 → `_allocate_kv_cache_tensors` L3929（int8 cache `_allocate_int8_cache_tensor` L3851、sparse-c8 `_allocate_sparse_c8_indexer_tensors` L3874、内存对齐 `_align_memory`/`_align_up`）→ `_reshape_kv_cache_tensors` L4144（NPU 布局 `_adjust_kv_layout` L4111）→ bind（DeepSeek-V4 层序 / longcat 双 module / hamming sparse）。`get_kv_cache_spec` L4657 与 `may_reinitialize_input_batch` L4464。
- **key_source_paths**：`vllm_ascend/worker/model_runner_v1.py`（L3764, L3929, L4111, L4144, L4657, L4464）。
- **pairs_with**：`vllm/v1/worker/gpu_model_runner.py:initialize_kv_cache_tensors`(L6839)/`_reshape_kv_cache_tensors`(L6678)/`get_kv_cache_spec`(L7058)；对应 **vLLM 书 ch18（KV cache 初始化）**。
- **teach_value**：中。昇腾量化 cache/对齐/特殊 layout 是工程性顶替，价值不如 W2/W3 但补全执行脊柱。
- **est_size**：~700 行。
- **deps**：W2；C KV cache 管理 / attention backend 章。

---

## 减法 / 边界提示（给 outline 与后续 analyst）

- **不要全量讲 244KB 的 NPUModelRunner**：它继承自父类，绝大多数控制流复用 vLLM。正文只讲 **override 的方法 + 猴补机制**，未 override 的方法直接指向 vLLM 书对应章。
- **大量 NPU 专属 feature 是旁支**（spec decode `propose_draft_token_ids` L1642、PCP `pcp_utils.py`、kvcomp `kvcomp_utils.py`、v2 model runner `worker/v2/`、dump/msprobe、weight prefetch）——本子系统**主线只取执行脊柱**，这些下沉到各自专题章或一笔带过，避免执行主线被淹没。
- `flash_common3_context.py` 极小（1.3KB），纯 MoE 旁路，归入 MoE 专题，不单独成章。
- pin 行号以 `instances/vllm-ascend/source/` 为准；运行验证须进容器（host 无 NPU/CANN）。
