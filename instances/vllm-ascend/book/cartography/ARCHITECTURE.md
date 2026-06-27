# vLLM-Ascend 架构地图与完整大纲

> 锁定版本：vllm-ascend **v0.21.0rc1**（commit `80610e44`，源码工作树，规范前缀 `vllm_ascend/…`）。
> 配套基座：vLLM **v0.21.0**（前缀 `vllm/…`）。
> 本书是 vLLM 书（`instances/vllm`，ch01–ch33）的**姊妹篇**：vLLM 书讲引擎本体（CUDA 线），本书讲「同一个 v0.21.0 引擎如何被搬到昇腾 NPU 上」。
> **7 Part / 28 章**（1 个 meta 概览章 skip_impl + 27 个源码解读章）。

---

## 一、心智模型：一个「不改 vLLM、却接管整条执行路径」的 OOT 插件

vLLM v0.21.0 是引擎本体——一套设备无关的骨架 + 一套 CUDA 实现。vllm-ascend **不 fork、不改 vLLM 一行源码**，而是在 **安装期** 经 setuptools entry points 把自己挂进去，在 **运行期** 经平台抽象 + monkey-patch 把昇腾 NPU 的实现替换进每一层。

读这本书 = **沿 vLLM 的执行主线，逐站看「昇腾版」如何顶替 CUDA 版**。每章主线是 vllm-ascend 源码，对照基座 vLLM v0.21.0，说明它顶替/扩展了哪一站，并尽量配对 vLLM 书的对应章作为「昇腾对位篇」。

读者画像：已读 vLLM 书或熟悉 vLLM v1 引擎的高级读者。本书不重讲 vLLM 主循环，只讲「替换面」。

---

## 二、三个接入支柱（本书的脊柱）

全书所有章节都回收到这三根支柱里的某一根。开篇 Part I 立柱，后续每章注明它落在哪根支柱上。

**支柱 1 — Entry points（安装期注册）**
**仓库根 `setup.py`** 注册两个 entry-point 组：`vllm.platform_plugins`（`ascend = vllm_ascend:register` → 返回字符串 `"vllm_ascend.platform.NPUPlatform"`，故意不 import 以免在平台选择阶段触发 torch_npu 重初始化）和 `vllm.general_plugins`（connector/model_loader/service_profiling/model 4 个回调，先 `_ensure_global_patch()` 再各自注册）。vLLM 经 `importlib.metadata.entry_points` 发现并激活——OOT 插件优先于 builtin（cuda/rocm/tpu/xpu/cpu）。

**支柱 2 — Platform 抽象（运行期分发）**
`NPUPlatform(Platform, PlatformEnum.OOT)`（`vllm_ascend/platform.py`，1188 行）是「问平台要类名/改 config」的总台。vLLM 引擎内部所有钩子——`get_attn_backend_cls` / `get_device_communicator_cls` / `worker_cls` / `get_compile_backend` / `get_static_graph_wrapper_cls` / `check_and_update_config` / `set_additional_forward_context`——全打到这里，逐一把 CUDA 类名换成昇腾类名。`check_and_update_config` 是其中最重的方法：把 GPU/ROCm 假设系统性中和成 NPU 安全值。

**支柱 3 — Patch（运行期改写 vLLM 内部）**
两段式 monkey-patch（`vllm_ascend/patch/`，platform 段 25 文件 + worker 段 29 文件），由 `utils.py adapt_patch(is_global_patch)` 单一入口分发：**platform 段**（构图前、进程级，改 scheduler/engine-core/config 校验/KV-cache 协调器/分布式原语）vs **worker 段**（worker 进程内，改 model_runner/模型 forward/算子/采样/权重）。靠 **import 副作用** 执行，5 种重绑定技法（整类替换 / 工厂替换 / 方法替换 / 库函数 wrapper / from-import 缓存陷阱修复）弥补 Platform 钩子覆盖不到、又不能 fork vLLM 之处。

---

## 三、子系统地形（按解读价值排布，非按行数）

| 子系统 | 路径 | 量级 | 解读要点 | 所属 Part |
|---|---|---|---|---|
| 接入/平台 | `__init__.py` · 仓库根 `setup.py` · `platform.py` · `ascend_config.py` | 1188+819 | 三支柱总台、entry points、配置改写器 | I |
| Patch 机制 | `patch/`（platform 25 + worker 29） | 919 manifest | 两段式、5 技法、引擎核心改写 | I |
| 设备/显存 | `distributed/device_communicators/` · `device_allocator/camem.py` | ~1.3k | NPUCommunicator/pyhccl、camem sleep-mode | II |
| 并行+KV 解耦 | `distributed/parallel_state.py` · `eplb/` · `distributed/kv_transfer/` | ~13k | 专属并行组+CP、eplb 热迁移、PD 分离、KV 池化 | III |
| Worker | `worker/`（worker.py 41KB + model_runner_v1.py 244KB） | ~12.6k | NPUWorker 重写、NPUModelRunner 继承+猴补、forward context、KV 落地 | IV |
| 注意力 | `attention/` + `core/` | ~12.8k+3k | 后端选择/MHA/MLA/SFA-DSA、KV 管理、调度器子类 | V |
| 算子+编译 | `ops/` + 仓库根 `csrc/` + `compilation/` | ~19k+2.5k | CustomOp.register_oot、torch.library+meta、AscendCompiler/ACLGraph、FusedMoE | VI |
| 量化 | `quantization/` | ~6k | registry+adapter+ABC scheme 三层接入 | VII |
| 采样 | `sample/` | ~1.9k | 规避 CPU-NPU 同步、Triton 回退 | VII |
| 投机解码 | `spec_decode/` | ~2.7k | 工厂分发 + 薄壳继承 | VII |
| 模型/LoRA | `models/` + `lora/` | ~2.2k+0.6k | DeepSeek-V4 特化、全局类替换 trick | VII |

**横切线索（不单独成章，各章点出）**：设备分代 `AscendDeviceType{A2,A3,A5,_310P}` 与 `_310p/` 子类目录贯穿全书；`enable_custom_op()` 真融合算子 vs `torch_npu.*` 回退的二分贯穿算子层；context-parallel（PCP/DCP）归口 ch08，注意力篇只引用 `enable_cp()` 分流。

**路径提示（防脚手架泄漏）**：`setup.py` 与 `csrc/` 在**仓库根**，规范引用写 `setup.py`、`csrc/torch_binding.cpp`（**不带** `vllm_ascend/` 前缀）；其余 Python 子系统才带 `vllm_ascend/` 前缀。

**减法边界（不深讲/略）**：model_loader 的 netloader/rfork（部署优化）、profiler 薄封装、xlite 轻量执行路径、KV 池其余后端（cpu_offload/lmcache/ucm）、flashlb/swift eplb 策略、profiling_chunk_predictor（pandas）——只点名或取一代表。

---

## 四、逐 Part 大纲

### Part I — 接入机制：插件如何挂进 vLLM（三支柱）
立全书地基与三支柱。从 entry points 被发现/激活，到 NPUPlatform 顶替总台，到两段式 monkey-patch 招式总纲，再到 config 配置面。

- **ch01 鸟瞰：一个不改 vLLM 却接管整条执行路径的 OOT 插件**（meta，skip_impl）/ 配 vLLM ch01–ch02 / `vllm_ascend/__init__.py`,`setup.py`,`platform.py`,`patch/__init__.py`
- **ch02 插件如何被 vLLM 发现并顶替：entry points 与 NPUPlatform** / 配 vLLM ch03 / `setup.py`,`__init__.py`,`platform.py`,`utils.py`（↔ `vllm/platforms/__init__.py`,`interface.py`,`vllm/plugins/__init__.py`）
- **ch03【旗舰】两段式 monkey-patch：不改 vLLM 就整体接管的招式（招式总纲）** / 配 vLLM ch17 / `utils.py(adapt_patch)`,`patch/__init__.py`,`patch/platform|worker/__init__.py`,`patch_distributed.py`,`patch_multiproc_executor.py`,`patch_scheduler.py`
- **ch04 顶替引擎核心：KV-cache 协调器、调度与 spec 的昇腾化 patch** / 配 vLLM ch16 / `patch_kv_cache_interface.py`,`patch_kv_cache_coordinator.py`,`patch_kv_cache_utils.py`,`patch_mamba_config.py`,`patch_qwen3_next_mtp.py`
- **ch05 check_and_update_config：插件改写 VllmConfig 的总闸 + AscendConfig 配置面** / 配 vLLM ch03 / `platform.py`,`ascend_config.py`,`envs.py`（↔ `vllm/platforms/interface.py`,`vllm/config`）

### Part II — 设备与显存：通信器与 sleep-mode 分配器
昇腾如何顶替 CUDA 版的设备通信与显存底座。

- **ch06 换底座的通信器：从 CudaCommunicator 到 NPUCommunicator** / 配 vLLM ch20 / `distributed/device_communicators/{npu_communicator,pyhccl,pyhccl_wrapper}.py`,`patch_distributed.py`（↔ `base_device_communicator.py`,`cuda_communicator.py`,`pynccl*.py`）
- **ch07 显存底座：sleep-mode 与 CANN 虚拟内存分配器（camem）** / 配 vLLM ch17,ch33 / `device_allocator/camem.py`,`patch/platform/patch_camem_allocator.py`（↔ `vllm/device_allocator/cumem.py`）

### Part III — 并行、eplb 与 KV 解耦：组、热迁移与 PD 分离
在 vLLM GroupCoordinator 之上叠昇腾专属组，叠 eplb 热迁移，并把 KV 解耦的两条路（PD 分离 / 池化）讲透。

- **ch08 在 vLLM 并行组之上：MC2 / 细粒度 TP / flashcomm 与上下文并行** / 配 vLLM ch20 / `distributed/parallel_state.py`,`distributed/utils.py`（↔ `vllm/distributed/parallel_state.py`）
- **ch09 Expert 负载均衡（eplb）：子进程规划 + D2D 权重热迁移** / 松配 vLLM ch33 / `eplb/eplb_updator.py`,`core/eplb_worker.py`,`core/eplb_device_transfer_loader.py`,`adaptor/vllm_adaptor.py`,`core/policy/*`
- **ch10 PD 分离：连接器分发与 mooncake P2P KV 传输** / 配 vLLM ch29–ch30 / `distributed/kv_transfer/ascend_multi_connector.py`,`kv_p2p/{mooncake_connector,mooncake_hybrid_connector,mooncake_layerwise_connector}.py`,`utils/mooncake_transfer_engine.py`（↔ `multi_connector.py`,`kv_connector/v1/base.py`）
- **ch11 KV 池化与 ascend_store：外存储层与池调度** / 配 vLLM ch30 / `distributed/kv_transfer/kv_pool/ascend_store/{ascend_store_connector,pool_scheduler,pool_worker,kv_transfer}.py`,`backend/mooncake_backend.py`（↔ `kv_connector/v1/base.py`,`offloading_connector.py`）

### Part IV — 执行主线：NPUWorker 与 NPUModelRunner
全书执行脊柱。两种顶替策略：Worker 重写、ModelRunner 继承+猴补。

- **ch12 NPUWorker：从 WorkerBase 重写执行主控（设备 / 内存 / 编译预热）** / 配 vLLM ch17 / `worker/worker.py`（↔ `vllm/v1/worker/gpu_worker.py`,`worker_base.py`）
- **ch13【旗舰】NPUModelRunner：继承 GPUModelRunner + 运行时 CUDA→NPU 猴补** / 配 vLLM ch18–ch19 / `worker/model_runner_v1.py`（↔ `vllm/v1/worker/gpu_model_runner.py`）
- **ch14 单步前向：execute_model、昇腾 forward context 与 DP 同步** / 配 vLLM ch18,ch21 / `model_runner_v1.py`,`ascend_forward_context.py`（↔ `gpu_model_runner.py`,`vllm/forward_context.py`）
- **ch15 KV cache 在昇腾上的落地：分配、reshape 与绑定** / 配 vLLM ch18,ch16 / `model_runner_v1.py`（↔ `gpu_model_runner.py`）

### Part V — 注意力与 KV：NPU 后端特化
vLLM 注意力后端契约的昇腾对位篇。

- **ch16 昇腾如何接进 vLLM 的注意力后端选择** / 配 vLLM ch24 / `platform.py`,`attention/attention_v1.py`,`attention/abstract.py`（↔ `vllm/v1/attention/backends/registry.py`,`flash_attn.py`）
- **ch17 AscendAttentionBackendImpl：标准 MHA 的 NPU 内核与状态机** / 配 vLLM ch24 / `attention/attention_v1.py`,`attention_mask.py`,`utils.py`（↔ `flash_attn.py`）
- **ch18【旗舰】MLA 在 NPU 上：prefill/decode 拆分与权重吸收** / 配 vLLM ch24 / `attention/mla_v1.py`（↔ `vllm/v1/attention/backends/mla/common.py`）
- **ch19 稀疏注意力：SFA 与 DSA（Lightning Indexer）** / 松配 vLLM ch24 / `attention/sfa_v1.py`,`dsa_v1.py`,`device/device_op.py`（vLLM 主干无对位）
- **ch20 KV cache 管理与调度器的 NPU 特化** / 配 vLLM ch15,ch13,ch14 / `core/single_type_kv_cache_manager.py`,`core/scheduler_dynamic_batch.py`,`recompute_scheduler.py`,`scheduler_profiling_chunk.py`

### Part VI — 自定义算子与编译：换头不换身
三套相互独立的算子/编译顶替机制。

- **ch21 CustomOp 的 OOT 顶替：昇腾算子如何替换 vLLM 算子** / 配 vLLM ch23 / `utils.py(register_ascend_customop)`,`ops/activation.py`,`ops/layernorm.py`,`ops/__init__.py`（↔ `vllm/model_executor/custom_op.py`）
- **ch22 torch.library 算子注册与 meta 实现：AscendC kernel 怎么进图** / 配 vLLM ch23 / `csrc/torch_binding.cpp`,`csrc/torch_binding_meta.cpp`,`meta_registration.py`,`ops/register_custom_ops.py`（↔ `vllm/_custom_ops.py`）
- **ch23 AscendCompiler 与 ACLGraph：torch.compile + cudagraph 栈的整体顶替** / 配 vLLM ch23 / `compilation/compiler_interface.py`,`acl_graph.py`,`graph_fusion_pass_manager.py`,`passes/`（↔ `vllm/compilation/compiler_interface.py`,`backends.py`,`cuda_graph.py`）
- **ch24 FusedMoE 算子与 batch-invariant 一致性** / 配 vLLM ch23,ch25 / `ops/fused_moe/`,`batch_invariant.py`,`ops/triton/batch_invariant/`（↔ `vllm/model_executor/layers/fused_moe/`）

### Part VII — 量化、采样、投机与模型：扩展点的注册范式
收束「找到 vLLM 扩展点 → 注册/子类化/换实现」四种范式（注册表+工厂 / 子类化覆写 / 工厂分发 / 运行时类替换）。

- **ch25 昇腾量化框架：把 NPU 量化方案接进 vLLM** / 配 vLLM ch22,ch23 / `quantization/{modelslim_config,method_adapters,quant_parser}.py`,`methods/{registry,base,w8a8_dynamic}.py`（↔ `vllm/model_executor/layers/quantization/`）
- **ch26 采样的 NPU 对位：规避 CPU-NPU 同步与 Triton 回退** / 配 vLLM ch27 / `sample/{sampler,rejection_sampler,penalties}.py`（↔ `vllm/v1/sample/{sampler,rejection_sampler}.py`）
- **ch27 投机解码的 NPU 对位：proposer 工厂与薄壳继承** / 配 vLLM ch28 / `spec_decode/{__init__,llm_base_proposer,ngram_proposer_npu}.py`（↔ `vllm/v1/spec_decode/`）
- **ch28 模型与 LoRA 的昇腾接入：注册、全局类替换与自定义算子** / 配 vLLM ch22,ch25 / `models/{__init__,deepseek_v4}.py`,`lora/{punica_npu,lora_ops,utils}.py`（↔ `vllm/model_executor/models/deepseek_v2.py`,`vllm/lora/punica_wrapper/`）

---

## 五、vLLM 配对脊柱（伏笔设计）

本书以「三支柱」开篇立伏笔，后续每章回收到具体子系统，并配对 vLLM 书的对应章：

1. **三支柱在 ch01 立、ch02–ch05 落地**：entry points（ch02）→ NPUPlatform 工厂钩子（ch02，后续 ch06/ch12/ch16/ch21/ch23 各自回收一个钩子）→ patch 招式（ch03，后续 ch04/ch06-310P/ch07 回收具体 patch）→ config 改写器（ch05，ch12/ch15 回收 worker_cls/cudagraph 决策）。
2. **每章「昇腾对位篇」配对**：见 `outline-final.json` 的 `vllm_pairing_map`。读者读本书某章前，应先具备对应 vLLM 章的心智模型；本书只讲替换面，不重述 vLLM 主循环。
3. **三个旗舰章**承载最高解读价值，是减法与精读的重点：
   - **ch03 两段式 monkey-patch**——全书地基招式，读懂后每章「昇腾顶替 CUDA」皆可秒懂。
   - **ch13 NPUModelRunner 继承+猴补**——最具 OOT 插件味，区别于「全量 fork」的精髓。
   - **ch18 MLA on NPU**——torch_npu 融合算子最密集，DeepSeek 类模型核心，昇腾改造最深。
4. **KV 解耦三章（ch04 patch / ch10 PD 分离 / ch11 池化）的分工**：ch04 讲为硬件约束改写 KV-cache 协调器/spec 的 patch；ch10 讲 prefill↔decode 节点间 mooncake P2P 直传；ch11 讲经外存储层中转/复用的 KV 池化与 pool scheduler/worker 节拍。三者互不重复，camem sleep-mode 归 ch07 不混入。
5. **跨 Part 伏笔点**：ch05 的 `check_and_update_config` 与 ch14 的 forward context 被 compilation/attention/MoE 多章引用；ch04 的 KV patch 与 ch15/ch20 的 KV 管理交叉；ch08 的 MC2/细粒度 TP 组被 ch24 FusedMoE、ch18 MLA、ch25 量化的并行特性消费——是天然的「应埋/应回收」登记点。
