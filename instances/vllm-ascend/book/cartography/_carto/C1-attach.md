# C1 — 接入机制：插件如何挂进 vLLM

子系统范围：vllm-ascend 作为 vLLM 的 out-of-tree 平台插件，如何被 vLLM 发现、激活、并把昇腾特化的实现「顶替」进 vLLM 引擎主线。
对象版本：vllm-ascend v0.21.0rc1（`instances/vllm-ascend/source/`，规范前缀 `vllm_ascend/…`），基座 vLLM v0.21.0（`instances/vllm/source/`，前缀 `vllm/…`）。

---

## 主线一句话

vLLM 在 `vllm/platforms/__init__.py` 留了两个 setuptools entry-point 组（`vllm.platform_plugins` 和 `vllm.general_plugins`）；vllm-ascend 在 `setup.py` 注册这两组的 5 个回调（`vllm_ascend/__init__.py` 里的 `register*`），于是「`NPUPlatform` 被选为 `current_platform`」+「一批 monkey-patch 被注入」。此后 vLLM 引擎内部所有「问平台要类名」的钩子（`get_attn_backend_cls`、`get_device_communicator_cls`、`worker_cls`、`check_and_update_config`…）都打到 `NPUPlatform`，从而把 CUDA 版整套替换成昇腾版——**插件不 fork vLLM，而是从 vLLM 预留的扩展点反向钩进去**。

---

## 关键事实（已逐文件核对源码）

### 1. 注册：两层 entry points

`vllm_ascend/setup.py:540-547`：
```
entry_points={
    "vllm.platform_plugins": ["ascend = vllm_ascend:register"],
    "vllm.general_plugins": [
        "ascend_kv_connector = vllm_ascend:register_connector",
        "ascend_model_loader = vllm_ascend:register_model_loader",
        "ascend_service_profiling = vllm_ascend:register_service_profiling",
        "ascend_model = vllm_ascend:register_model",
    ],
}
```
- `vllm.platform_plugins` 的回调 `register()`（`vllm_ascend/__init__.py:40-43`）只返回**字符串** `"vllm_ascend.platform.NPUPlatform"`——故意不 import，避免在平台选择阶段触发 torch_npu 重初始化。
- 4 个 `vllm.general_plugins` 回调（connector / model_loader / service_profiling / model）都先调 `_ensure_global_patch()`（`__init__.py:23-37`）再做各自注册。`_ensure_global_patch` 用进程级 `_GLOBAL_PATCH_APPLIED` 幂等保护，调 `adapt_patch(is_global_patch=True)`。
- 注释点出关键约束（`__init__.py:24-29`）：vLLM 在 engine-core 子进程里加载 general plugins，conftest 钩子不在子进程跑，所以影响 scheduler/engine 的全局 patch 必须经 entry point 在每个进程重新落地。

### 2. vLLM 侧的接收点（对照基座）

- `vllm/plugins/__init__.py:28-66` `load_plugins_by_group(group)`：用 `importlib.metadata.entry_points(group=...)` 发现插件，受 `VLLM_PLUGINS` 白名单过滤，逐个 `plugin.load()`。
- `vllm/plugins/__init__.py:69-82` `load_general_plugins()`：`plugins_loaded` 进程级幂等，对每个 general plugin **直接执行回调**（`func()`）——这正是 ascend 4 个 register 回调被触发的地方。它在 vLLM 多处被调：`vllm/engine/arg_utils.py:731/2465`、`vllm/v1/engine/core.py:105`、`vllm/v1/worker/worker_base.py:239`、`vllm/model_executor/models/registry.py:1366`（每个新进程都会重跑→呼应 ascend 的幂等设计）。
- `vllm/platforms/__init__.py:212-252` `resolve_current_platform_cls_qualname()`：`builtin_platform_plugins`（cuda/rocm/tpu/xpu/cpu）+ OOT 插件一起 probe，每个回调返回非 None 即「激活」。OOT 激活 ≥2 报错；恰好 1 个 OOT 激活则用它，且 **OOT 优先于 builtin**（`activated_oot_plugins` 先判）。ascend 的 `register()` 永远返回字符串→恒激活。
- `vllm/platforms/__init__.py:262-281` `current_platform` 懒加载：首次访问才 resolve + `resolve_obj_by_qualname()` 实例化。注释明确解释为何懒加载——OOT 平台要 `from vllm.platforms import Platform` 继承，故 `current_platform` 不能在 import `vllm.platforms` 时就 resolve。

### 3. NPUPlatform —— 顶替的主体（`vllm_ascend/platform.py`，1188 行）

`class NPUPlatform(Platform)`（`platform.py:134`），继承 `vllm/platforms/interface.py:105` 的 `Platform`，`_enum = PlatformEnum.OOT`（`vllm/platforms/interface.py:46`）。
类属性即「身份替换」：`device_name/device_type="npu"`、`dispatch_key="PrivateUse1"`、`device_control_env_var="ASCEND_RT_VISIBLE_DEVICES"`、`ray_device_key="NPU"`、`simple_compile_backend="eager"`、`supported_quantization`（`platform.py:135-151`）。

vLLM 的钩子方法被 NPUPlatform 覆写，逐一把 CUDA 实现换成昇腾类名（全是返回 qualname 字符串的工厂方法）：
- `get_attn_backend_cls()`（`platform.py:738-765`）：按 `(use_mla, use_sparse)` 等键路由到 `AscendMLABackend / AscendAttentionBackend / AscendSFABackend / AscendDSABackend`，310P 走 `backend_map_310`。
- `get_device_communicator_cls()`→`NPUCommunicator`（804-805）；`get_punica_wrapper()`→`PunicaWrapperNPU`（795-796）；`get_compile_backend()`→`AscendCompiler`（173-179）；`get_pass_manager_cls()`/`get_static_graph_wrapper_cls()`（165-171, 815-820）。
- 设备能力类钩子用 `torch.npu.*` / `npu-smi` 实现：`get_device_name/uuid/total_memory`（262-285）、`num_compute_units` 用 `cube_core_num` 对标 CUDA SM 数（287-309）、`set_device`→`torch.npu.set_device`（337-339）。
- `pre_register_and_update()`（181-202）：平台被选定后的第一个回调，再次 `adapt_patch(is_global_patch=True)`，并把 `"ascend"` 量化方法塞进 CLI `--quantization` choices。
- `check_and_update_config()`（413-720）：本子系统最重的方法——巨型「配置改写」入口。`init_ascend_config()` 解析 `additional_config`；`_fix_incompatible_config()`（979-1180）逐个把 GPU/ROCm 专属参数（cascade_attn / cudnn_prefill / trtllm / numa_bind / nvtx…）reset 成昇腾安全值；按 cudagraph_mode 改写 compilation/splitting_ops；选 `worker_cls`（602-612：`NPUWorker` / `NPUWorker310` / `XliteWorker`）；设 `PYTORCH_NPU_ALLOC_CONF` 等。
- `set_additional_forward_context()`（840-977）：平台级钩子，注入昇腾 forward-context 额外字段（moe_comm_type、mc2_mask、flashcomm 标记…）。

### 4. monkey-patch 机制（`vllm_ascend/patch/` + `adapt_patch`）

- `utils.py:511-516` `adapt_patch(is_global_patch)`：`True`→`import vllm_ascend.patch.platform`；`False`→`import vllm_ascend.patch.worker`。**靠 import 副作用**触发 patch（被 import 的 `__init__` 顺序 import 每个 patch 模块）。
- `patch/platform/__init__.py:19-51`：固定顺序 import ~25 个 platform patch；其中按 `is_310p()` 二选一（`patch_mamba_config` vs `patch_mamba_config_310`），按 `DYNAMIC_EPLB`/`EXPERT_MAP_RECORD` 条件 import `patch_multiproc_executor`。
- `patch/__init__.py:1-919` 是一份巨型「补丁台账」注释：platform patch（worker 启动前，由 `pre_register_and_update` 调）vs worker patch（worker `__init__` 里调）两类，每条记录 patch 的 vLLM 目标符号 / Why / How / Related PR / Future Plan。这是「为什么需要 monkey-patch 而非干净继承」的一手教学材料。

### 5. 配置桥 `AscendConfig`（`vllm_ascend/ascend_config.py`，819 行）

- `init_ascend_config(vllm_config)` / `get_ascend_config()` / `clear_ascend_config()`（788-819）：进程级单例（`_ASCEND_CONFIG`），从 `vllm_config.additional_config` 这个 dict 解析出全部昇腾开关。`_is_ascend_config_initialized` 防单测 monkeypatch 污染单例。
- `AscendConfig.__init__`（27-283）把 `additional_config` 的子 dict 解析成强类型子配置对象：`XliteGraphConfig / AscendCompilationConfig / AscendFusionConfig / FinegrainedTPConfig / EplbConfig / WeightPrefetchConfig / ProfilingChunkConfig / RejectionSamplerConfig`，并做大量交叉校验。
- `_get_config_value`（284-296）：三级取值——`additional_config` 优先，回退环境变量（带 deprecation 警告），最后默认值。这是 ascend 自己 `envs.py`（21 个 lambda 环境变量，`envs.py:29+`）与 `additional_config` 的桥接点。

### 6. 设备分代抽象 `AscendDeviceType` + 310P 分叉

- `utils.py:768-816` `AscendDeviceType{A2,A3,_310P,A5}` + `get_ascend_device_type()`：从打包信息 `_build_info` 取，`check_ascend_device_type` 再用 `torch_npu.npu.get_soc_version()`（220-225→A2,250-255→A3,200-205→310P,260→A5）运行时校验。
- 全子系统按设备分代分叉：`get_attn_backend_cls` 的 `backend_map_310`、`worker_cls` 选 `NPUWorker310`、`select_moe_comm_method`（`ascend_forward_context.py:233-319` 按 soc_version 选 MC2/AllToAll/AllGather）。`_310p/` 整目录是 310P 专属子类（worker_310p、model_runner_310p、attention…）。

### 7. 自定义算子接入：meta 注册 + 环境引导

- `meta_registration.py:1-94`：为昇腾 C++ 自定义算子（`_C_ascend` 命名空间）注册 Python **meta 实现**（返回空形状张量），让 `torch.compile`/aclgraph 能 trace/shape-infer。`register_meta_if_necessary`（47-54）查重后 `lib.impl(..., "Meta")`。
- `platform.py:722-736` `import_kernels()` → `utils.py:267-277` `bootstrap_custom_op_env()`：把昇腾自定义算子 vendor 路径 prepend 进 `ASCEND_CUSTOM_OPP_PATH`。懒初始化（注释解释：直接 import `vllm_ascend_C` 会让 `ASCEND_RT_VISIBLE_DEVICES` 失效，影响 RL 场景）。

### 8. forward context 扩展（`ascend_forward_context.py`，381 行）

- `set_ascend_forward_context`（56-191）：包住 vLLM 的 `set_forward_context`，再往 forward_context 上挂昇腾额外字段（moe_comm_type/method、flashcomm v1/v2、mc2_mask、pad_size、layer_idx…）。
- `_ExtraForwardContextProxy` / `_EXTRA_CTX`（322-381）：统一 v1/v2 model runner 的额外字段读写（v2 走 `additional_kwargs`，v1 走属性）。这是「平台把自己的运行期状态塞进 vLLM 通用 forward context」的范例，与 `NPUPlatform.set_additional_forward_context` 配对。

---

## 建议「可成章单元」

> 设备分代/310P 不单独成章——它横切整个接入层，作为各章的统一线索讲（在 ch-attach-01/05 里点出）。

### ch-attach-01 — 插件如何被 vLLM 发现并顶替：entry points 与 NPUPlatform
- **focus**：从 `setup.py` 的两个 entry-point 组讲到 vLLM 的 `resolve_current_platform_cls_qualname` / `current_platform` 懒加载，说清「OOT 平台为何能优先于 builtin、为何只返回类名字符串而不 import」。
- **key_source_paths**：`vllm_ascend/setup.py`(entry_points), `vllm_ascend/__init__.py`(register/register_*), `vllm_ascend/platform.py`(NPUPlatform 类属性与工厂钩子), `vllm_ascend/utils.py`(AscendDeviceType/get_ascend_device_type)
- **pairs_with**：`vllm/platforms/__init__.py`(builtin_platform_plugins, resolve_current_platform_cls_qualname, current_platform `__getattr__`), `vllm/platforms/interface.py`(Platform 抽象, PlatformEnum.OOT), `vllm/plugins/__init__.py`(load_plugins_by_group)
- **teach_value**：全书的「地基」——读者理解整个插件如何挂进引擎、后续每个昇腾子系统是怎么被 vLLM「问到」的。
- **est_size**：L
- **deps**：无（开篇章）

### ch-attach-02 — general_plugins 与 monkey-patch：当继承不够时怎么改写 vLLM 内部
- **focus**：`vllm.general_plugins` 4 个回调 + `_ensure_global_patch`/`adapt_patch` 的 import-副作用机制；platform-patch vs worker-patch 两阶段；以 patch 台账里 2-3 个代表性 patch（如 multiproc_executor daemon、kv_cache_interface 子类化、scheduler assert 移除）说明「为何 Platform 钩子覆盖不到、只能 monkey-patch」。
- **key_source_paths**：`vllm_ascend/__init__.py`(_ensure_global_patch/register_connector 等), `vllm_ascend/utils.py`(adapt_patch), `vllm_ascend/patch/__init__.py`(补丁台账), `vllm_ascend/patch/platform/__init__.py`(加载顺序与条件 import)
- **pairs_with**：`vllm/plugins/__init__.py`(load_general_plugins, plugins_loaded 幂等), `vllm/engine/arg_utils.py` 与 `vllm/v1/engine/core.py`(多进程调用点)
- **teach_value**：讲清 out-of-tree 插件的「灰色手段」——继承 + 钩子之外，monkey-patch 如何弥补 vLLM 未预留扩展点处，及其幂等/多进程约束。
- **est_size**：L
- **deps**：ch-attach-01

### ch-attach-03 — check_and_update_config：插件改写 VllmConfig 的总闸
- **focus**：`NPUPlatform.check_and_update_config` 全流程——`init_ascend_config` → `_fix_incompatible_config`（剔除 GPU/ROCm 专属参数）→ cudagraph/compilation 改写 → 选 `worker_cls` → 设环境变量；说明 vLLM 如何把「最后改 config 的权力」交给平台。
- **key_source_paths**：`vllm_ascend/platform.py`(check_and_update_config, _fix_incompatible_config, apply_config_platform_defaults), `vllm_ascend/ascend_config.py`(init_ascend_config/get_ascend_config 单例)
- **pairs_with**：`vllm/platforms/interface.py`(Platform.check_and_update_config / apply_config_platform_defaults 默认空实现), `vllm/config`(VllmConfig)
- **teach_value**：展示「平台 = 配置改写器」这一 vLLM 设计；读者看到 GPU 假设是怎样被系统性地中和成 NPU 安全值。
- **est_size**：M
- **deps**：ch-attach-01

### ch-attach-04 — AscendConfig 与 envs：additional_config 如何成为昇腾特性总开关
- **focus**：`AscendConfig` 单例如何把 `vllm_config.additional_config` 这个开放 dict 解析成强类型子配置 + 校验；`_get_config_value` 的 additional_config→env→default 三级取值与 deprecation 策略；ascend 自有 `envs.py` 与 vLLM `envs.py` 的关系。
- **key_source_paths**：`vllm_ascend/ascend_config.py`(AscendConfig 及各子 Config 类, _get_config_value, init/get/clear), `vllm_ascend/envs.py`(env_variables 表)
- **pairs_with**：`vllm/config`(VllmConfig.additional_config 这个扩展点), `vllm/envs.py`(VLLM_PLUGINS 等)
- **teach_value**：解释 vLLM 留给插件的「无 schema 配置后门」`additional_config` 如何被插件用成结构化特性开关——接入层的配置面。
- **est_size**：M
- **deps**：ch-attach-03

### ch-attach-05 —（可选/进阶）自定义算子与 forward context 接入：meta 注册、算子环境与运行期状态注入
- **focus**：`import_kernels`/`bootstrap_custom_op_env` 如何把昇腾算子 vendor 路径接进 torch；`meta_registration` 为何要为自定义算子注册 meta 实现以支持 torch.compile/aclgraph；`set_ascend_forward_context` + `_EXTRA_CTX` 如何把昇腾运行期状态塞进 vLLM 通用 forward context。
- **key_source_paths**：`vllm_ascend/meta_registration.py`, `vllm_ascend/utils.py`(bootstrap_custom_op_env), `vllm_ascend/platform.py`(import_kernels, set_additional_forward_context), `vllm_ascend/ascend_forward_context.py`(set_ascend_forward_context, _ExtraForwardContextProxy)
- **pairs_with**：`vllm/forward_context.py`(set_forward_context/get_forward_context), `vllm/platforms/interface.py`(Platform.import_kernels)
- **teach_value**：补全接入层「算子级」与「运行期状态级」两个常被忽略的挂载面，呼应后续 attention/MoE 章。
- **est_size**：M
- **deps**：ch-attach-01；与后续 attention/MoE 子系统章有前置关系（可作其引子）

---

## 给后续 outline 编排者的提示
- 强依赖链：01 →(02, 03) → 04 → 05；01 是全书地基，建议作为 Part 开篇。
- 设备分代（A2/A3/A5/310P）与 `_310p/` 不单独成章，作为横切线索在 01/03/05 点出即可。
- 03 的 `check_and_update_config` 与 05 的 forward context 会被后续 compilation/attention/MoE 子系统反复引用，是天然伏笔点。
- 真实源码体量：platform.py 1188 行、ascend_config.py 819 行、utils.py 1567 行、patch/__init__.py 919 行（台账）——成章时 analyst 需做大量减法，但 entry-point 与 NPUPlatform 类属性/工厂钩子是 must_keep。
