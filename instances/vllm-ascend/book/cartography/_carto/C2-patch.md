# C2 — Patch 机制：两段式 monkey-patch（本书最核心招式）

子系统根：`vllm_ascend/patch/`（platform/ 25 文件 + worker/ 含 patch_v2/ 共 29 文件；`__init__.py` 是一份自带文档的 patch 清单 manifest）。

---

## 0. 两段式机制主线（写正文先讲透这 3 句）

vllm-ascend 作为 vLLM 的 OOT 插件，**完全不改 vLLM 源码**，靠在两个精确时机用 monkey-patch 把 CUDA 版实现整体顶替成昇腾版：

1. **何时打**——两个阶段，由 `vllm_ascend/utils.py:511 adapt_patch(is_global_patch)` 单一入口分发：
   - **Platform 阶段（进程级，worker 起来之前）**：`is_global_patch=True`，`import vllm_ascend.patch.platform`。触发点 `vllm_ascend/platform.py:182 NPUPlatform.pre_register_and_update()`（vLLM 平台注册钩子）+ `vllm_ascend/__init__.py:36 _ensure_global_patch()`（engine-core 子进程入口，spawn 出来的子进程没跑 conftest，必须重打）。这一阶段改的是 **scheduler / engine-core / config 校验 / KV-cache 协调器 / 分布式原语 / API server**——必须在引擎构图前生效。
   - **Worker 阶段（worker 进程内）**：`is_global_patch=False`，`import vllm_ascend.patch.worker`。触发点 `vllm_ascend/worker/worker.py:102 NPUWorker.__init__()`。这一阶段改的是 **model_runner / 模型 forward / 算子 / 采样 / 权重加载**——只需在每个 worker 内生效。
2. **怎么打**——靠 **import 副作用**：`adapt_patch` 只做 `import vllm_ascend.patch.{platform,worker}`，真正的 patch 在每个 patch 模块的**模块级语句**里执行（一次 import 即生效，天然幂等/单次）。`platform/__init__.py` 与 `worker/__init__.py` 用 `import ... # noqa` 列出全部 patch 模块，并用 `is_310p()` / `HAS_TRITON` / `vllm_version_is("0.21.0")`（v2 model-runner 仅 main 分支）/ `DYNAMIC_EPLB` 等开关条件加载。
3. **为何能不改 vLLM 就接管**——5 种重绑定技法（按攻击面从粗到细）：
   - **(a) 模块属性整类替换**：`vllm.<mod>.<Class> = AscendXxx`（子类继承原类、override 选定方法）。例：`patch_multiproc_executor.py:211`、`patch_kv_cache_interface.py` 末 3 行。
   - **(b) 工厂函数替换**：`vllm.<mod>.<factory> = lambda/new_fn`，内部按条件 dispatch 回原实现或昇腾实现。例：`patch_kv_cache_coordinator.py:360 get_kv_cache_coordinator`、`patch_mla_prefill_backend.py 末 get_mla_prefill_backend = lambda ...`。
   - **(c) 类方法/未绑定函数替换**：`Cls.method = new_method`、`module.func = new_func`。例：`patch_scheduler.py 末 Scheduler._mamba_block_aligned_split = ...`、`patch_qwen3_next_mtp.py:50 utils.bind_kv_cache = bind_kv_cache`、`patch_deepseek_mtp.py:72-76`。
   - **(d) 库函数包装（wrapper 闭包保留原函数）**：`torch.distributed.all_reduce = wrapper(torch.distributed.all_reduce)`。例：`patch_distributed.py:53/82/83`。也用于 `patch_camem_allocator`（保留原 CuMem 检查作 fallback）。
   - **(e) 消费方缓存绑定修复（关键陷阱）**：`from X import Y` 会把 Y 拷进消费模块命名空间，仅改定义模块不够。须额外重绑 **消费模块** 的符号或经 `sys.modules` 修补。例：`patch_kv_cache_coordinator.py:366-368`（修 `vllm.v1.core.kv_cache_manager` 已缓存的绑定）、`patch_v2/patch_input_batch.py 末`（同时改 `cudagraph_utils.InputBatch` 与 `model_runner.InputBatch`）。**这是 monkey-patch 体系最反直觉、最值得讲的一课。**

> 教学主线一句话：**"在正确的时机（platform/worker 两段）、用 import 副作用、把 vLLM 模块命名空间里的符号替换成昇腾子类/包装函数，并记得同时修掉所有已 `from-import` 的消费方缓存。"**

---

## 48 patch 的归类（按"patch 了 vLLM 哪一层 / 解决什么问题"）

聚成 4 组。前两组是本书核心（机制 + 引擎层接管），后两组按需略讲（模型/算子适配、服务层兼容多为机械补丁，可选段降密度）。

| 组 | patch 数 | 改的层 | 解决什么 |
|---|---|---|---|
| **G1 机制与引擎/分布式基础设施层** | ~10 | scheduler / engine-core / executor / 分布式原语 / 内存分配器 | 进程模型、调度、通信、sleep 内存——昇腾运行时与 CUDA 运行时的硬约束差异 |
| **G2 KV-cache / 内存形态层** | ~6 | KV-cache 协调器 / spec / manager / block_table / bind / mamba | block_size 16→128、DSA/MLA spec 扩展、CP+hybrid 前缀缓存、int32 slot_mapping |
| **G3 模型 forward / 算子 / 采样 / 权重层（worker）** | ~22 | 具体模型 forward、triton/融合算子、rejection sampler、load_weights | NPU 无对应算子/UVA、triton 性能差且无 dispatch、C8/fp8 量化权重重映射 |
| **G4 API/服务/解析兼容层（platform）** | ~10 | OpenAI/Anthropic serving、各家 tool-call/reasoning parser | backport 上游修复、流式增量、role 校验——纯协议兼容，与昇腾硬件无关 |

---

## 建议章节（2 章核心 + 1 章可选概览，共 2–3 章）

### 章 A（核心·必讲）— 两段式 monkey-patch：vLLM-Ascend 不改源码就接管的招式
- **title**: 两段式 monkey-patch——OOT 插件如何不改 vLLM 就整体接管
- **focus**: 讲透机制本身：`adapt_patch` 单一入口 / platform vs worker 两阶段的**时机选择依据**（构图前 vs worker 内）/ import 副作用即 patch / 5 种重绑定技法 / `from-import` 缓存陷阱与 `sys.modules` 修复。用 G1 里最干净的 2–3 个 patch 当解剖样本（distributed wrapper、multiproc_executor 子类替换、scheduler 方法替换）。
- **key_source_paths**:
  - `vllm_ascend/patch/__init__.py`（manifest 总览，节选）
  - `vllm_ascend/utils.py:511`（`adapt_patch`）
  - `vllm_ascend/platform.py:182` + `vllm_ascend/__init__.py:23-37`（platform 触发点）
  - `vllm_ascend/worker/worker.py:100-102`（worker 触发点）
  - `vllm_ascend/patch/platform/__init__.py`、`vllm_ascend/patch/worker/__init__.py`（条件加载清单）
  - `vllm_ascend/patch/platform/patch_distributed.py`（技法 d：wrapper 闭包）
  - `vllm_ascend/patch/platform/patch_multiproc_executor.py`（技法 a：子类+整类替换，daemon=False）
  - `vllm_ascend/patch/platform/patch_scheduler.py`（技法 c：方法替换/删 assert）
  - `vllm_ascend/patch/platform/patch_kv_cache_coordinator.py:360-368`（技法 b+e：工厂替换 + 缓存绑定修复）
- **pairs_with（vLLM 原文件/函数）**:
  - `vllm/v1/executor/multiproc_executor.py: MultiprocExecutor / WorkerProc.make_worker_process`（原 daemon=True）
  - `vllm/distributed/parallel_state.py`（GroupCoordinator）+ torch.distributed.all_reduce/broadcast
  - `vllm/v1/core/sched/scheduler.py: Scheduler._mamba_block_aligned_split`
  - `vllm/v1/core/kv_cache_coordinator.py: get_kv_cache_coordinator / HybridKVCacheCoordinator`
  - `vllm/platforms` 平台注册钩子（pre_register_and_update 对接处）
- **teach_value**: 整本书的"招式总纲"。读者一旦理解 import 副作用 + 两阶段时机 + 5 技法 + 缓存陷阱，后续每一章的"昇腾版顶替 CUDA 版"都能秒懂。
- **est_size**: L（最长，本书地基章）
- **deps**: 需先有"vllm-ascend 是 OOT 插件 / entry-points 注册 / NPUPlatform"概览章在前（C1 平台层）。

### 章 B（核心·必讲）— patch 引擎核心：KV-cache 协调与进程/调度的昇腾改造
- **title**: 顶替引擎核心——KV-cache 协调器、调度与 spec 的昇腾化 patch
- **focus**: 以 G2（+G1 的 KV 相关）为主线，展示"为正确性/硬件约束而 patch"的真实案例群：block_size 16→128（mamba 在 NPU 不支持 16）、MLAAttentionSpec 子类化扩展 DSA/Sparse-C8、CP+hybrid 前缀缓存（AscendHybridKVCacheCoordinator 的 `find_longest_cache_hit`）、bind_kv_cache 跳过 NPU raise、int32 slot_mapping。每个都对照 vLLM 原实现讲"原来怎么算、为什么 NPU 不行、patch 怎么改"。
- **key_source_paths**:
  - `vllm_ascend/patch/platform/patch_kv_cache_interface.py`（MLAAttentionSpec 子类 + 末 3 行重绑，含消费模块 `mla_attention`）
  - `vllm_ascend/patch/platform/patch_kv_cache_coordinator.py`（AscendHybridKVCacheCoordinator 全貌）
  - `vllm_ascend/patch/platform/patch_kv_cache_utils.py`（resolve_kv_cache_block_sizes：CP 下返回 lcm 而非 raise）
  - `vllm_ascend/patch/platform/patch_mamba_config.py`（block_size→128）+ `patch_mamba_manager.py`（AscendMambaManager 前缀缓存）
  - `vllm_ascend/patch/worker/patch_qwen3_next_mtp.py`（bind_kv_cache 跳 raise）
  - `vllm_ascend/patch/worker/patch_v2/patch_block_table.py`（int32 slot_mapping）+ `patch_input_batch.py`（消费方双重绑定示范）
- **pairs_with（vLLM 原文件/函数）**:
  - `vllm/v1/kv_cache_interface.py: MLAAttentionSpec`
  - `vllm/v1/core/kv_cache_coordinator.py: HybridKVCacheCoordinator.find_longest_cache_hit`
  - `vllm/v1/core/kv_cache_utils.py: resolve_kv_cache_block_sizes`（PR #40860 加的 CP 限制）
  - `vllm/model_executor/models/config.py: HybridAttentionMambaModelConfig.verify_and_update_config`
  - `vllm/v1/core/single_type_kv_cache_manager.py: MambaManager`
  - `vllm/v1/worker/utils.py: bind_kv_cache`；`vllm/v1/worker/gpu/block_table.py: BlockTables`
- **teach_value**: 展示 patch 不只是"换算子"，而是**为硬件约束重写引擎核心数据结构与调度算法**——最能体现"昇腾版顶替 CUDA 版"深度的一组。
- **est_size**: L
- **deps**: 章 A（机制）；与 C-KV-cache 子系统章交叉引用（spec/manager 的完整语义在那边讲，这里只讲 patch 差异）。

### 章 C（可选·概览，建议并入章 A 末尾或单独短章）— 模型/算子/服务层的 patch 全景与"未来会被上游吸收"的临时补丁
- **title**: patch 全景图——模型 forward、算子替换、API 兼容与上游回流
- **focus**: 快速扫过 G3+G4：triton 算子无 dispatch 故整体 override（patch_triton / patch_rejection_sampler / patch_v2/patch_triton）、模型 forward 注入昇腾融合算子（qwen3_next / minimax_m2 / qwen3_dflash）、C8/fp8 量化权重重映射（patch_gqa_c8 / patch_weight_utils）、UVA dummy 化、以及 G4 一大批"backport 上游 PR / 协议兼容"的临时补丁。重点讲**这类 patch 的共性**（多为 setattr 方法替换或子类）与 manifest 里每条都带的 `Related PR / Future Plan`——patch 是"临时垫片，等上游合并即删"的工程哲学。
- **key_source_paths（代表性，非全列）**:
  - `vllm_ascend/patch/worker/patch_triton.py`（triton ops 整体 override + `triton.next_power_of_2` 注入）
  - `vllm_ascend/patch/worker/patch_rejection_sampler.py`
  - `vllm_ascend/patch/worker/patch_gqa_c8.py` / `patch_weight_utils.py`（C8 KV scale 重映射）
  - `vllm_ascend/patch/worker/patch_v2/patch_uva.py`（dummy 化）
  - `vllm_ascend/patch/platform/patch_camem_allocator.py`（sleep mode allocator 兼容，技法 d 的 fallback 写法）
  - `vllm_ascend/patch/platform/patch_minimax_m2_config.py`（多符号 + Pydantic `rebuild_dataclass` 的特殊重绑——可作"高级技法补遗"）
- **pairs_with**:
  - `vllm/v1/sample/rejection_sampler.py`；`vllm/model_executor/layers/mamba|fla/ops`
  - `vllm/model_executor/models/{qwen3_next,minimax_m2,deepseek_v2,deepseek_mtp}.py`
  - `vllm/config/model.py: is_cumem_allocator_available / _verify_quantization`
  - `vllm/v1/worker/gpu/states.py: UvaBuffer`
- **teach_value**: 让读者看清 patch 体系的"长尾"形态与可维护性策略（PR/Future Plan 登记、版本/设备开关）。但单个 patch 教学价值低，**务必聚类略讲、不要逐 patch 展开**。
- **est_size**: M（或并入 A）
- **deps**: 章 A。

---

## 给 writer/implementer 的注意点
- manifest（`patch/__init__.py`）是"昇腾团队自己写的 patch 说明书"，每条带 Why/How/Related PR/Future Plan——正文可直接引用其 Why 作为"为什么 NPU 需要这么改"的权威依据，但行文须自包含、不要让读者去翻。
- **2 段式 ≠ 2 个开关函数**：强调是"同一 `adapt_patch` 入口、两个调用时机、靠 import 副作用执行"，别误写成两套机制。
- 注意条件加载会让某些 patch 在 host/CPU-only 或 v0.21.0 上**不生效**（`_V2_MODEL_RUNNER_SUPPORTED = not vllm_version_is("0.21.0")`，故 patch_v2/* 在 pin 的 0.21.0 上不加载）——写"昇腾如何接管"时要标清适用版本/设备。
