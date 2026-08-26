# ch03 精简版 impl-notes — 从 EngineArgs 到 VllmConfig（装配流）

- **Pin**：vLLM v0.27.1（`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`）。全部 `# SOURCE:` 行号
  已对当前 pin 逐行现核（2026-08-16 首核；2026-08-23 复核——73 处单行锚点 + 44 处区间锚点 +
  76 处符号锚点 + 114 处字段行号注全部机检对齐真源，非 v2 资产的 v0.21.0 旧行号。
  08-23 复核修正 5 处：logger once-mechanism（真源无 LoggingContext，实为 lru_cache'd
  `_print_*_once` helpers L76-L94 + `_VllmLogger.*_once` 方法 L118-L145）、
  `Platform.device_count` 真身 cuda.py:L608（interface.py 无基类声明）、
  `get_cpu_architecture` 真身 interface.py:L972、`resolve_kv_cache_block_sizes` 真身
  vllm/v1/core/kv_cache_utils.py:L626（非 kv_cache_interface.py）、`LLM.__init__` 区间端点
  L345→L341。同轮把 6 处弱化注解恢复逐字：`type[Executor]`×3、
  `dict[UsageContext | None, int]`、`UsageContext | None`×2、`dict[str, Any] | None`×2、
  `spec_*` 三字段、`get_scheduler_cls -> type["SchedulerInterface"]`、
  EngineCore 局部 `Scheduler` 名与 `: SchedulerInterface` 注解、`executor_fail_callback:
  Callable | None`、make_async_mp_client 的 `@instrument` SUBTRACTED 注记）。
- **产物**：`implementation/config_wiring.py`（单模块，host 可跑、无 torch/vllm/CUDA 依赖）。
- **验收判据**：把真实源码删掉所有 `# SUBTRACTED:` 分支 ≈ 得到本文件。
- **跑法**：`cd instances/vllm/artifacts-v3/ch03-engineargs-to-vllmconfig && python -m pytest tests/ -q`
  → 32 passed。`python scripts/lint_fidelity.py <本章目录>` → 全绿。

## 1:1 Source Map（精简版 ↔ 真实源码 ↔ 改动 ↔ 原因）

| 精简版符号 | 真实源码锚点（v0.27.1 现核） | 改动 | 原因 |
|---|---|---|---|
| `EngineArgs`（字段子集） | vllm/engine/arg_utils.py:L421-L753 | 保留 ~100 个本章路径读到的字段；数百个多模态/LoRA/EP/NUMA/观测字段 SUBTRACTED | dossier embed elide 同款裁剪；保留字段默认值逐一生借子 Config 类属性（单一真相源） |
| `EngineArgs.__post_init__` | arg_utils.py:L755-L795 | dict→Config 升格链全保留；HF offline 路径替换（L796-L820）SUBTRACTED | delete 项 4 |
| `EngineArgs.create_model_config` | arg_utils.py:L1676-L1752 | 字段透传保留（~20 kwargs）；HF 读取在 ModelConfig 内部 SUBTRACTED；~50 个 mm/pooler kwargs SUBTRACTED | 最重子构造器；派生标志以声明默认值呈现，测试可覆写翻分支 |
| `EngineArgs.create_engine_config` | arg_utils.py:L1896-L2493 | 开场（平台预注册/DeviceConfig/env 校验/speculator 覆盖删）→ CacheConfig 全 21 kwargs（L1954-L1976 逐字）→ DP 推导大段（L2016-L2189）删、DP=1 值内联 → ParallelConfig（embed kwargs+distributed_executor_backend）→ 批默认 → SchedulerConfig 全 kwargs（L2270-L2289 逐字）→ LoRA/attention/mamba/kernel/offload/observability 样板删 → CompilationConfig 深拷贝+覆盖（L2427-L2443 保留）→ VllmConfig 聚合（保留 ~22 kwargs） | delete 项 1/2/3；code_spine 全部站点保留 |
| `EngineArgs.get_batch_defaults` | arg_utils.py:L2515-L2596 | GPU 主线逐字保留（H100 16384/8192、A100 反例 #17885）；TPU/CPU 平台分支（L2565-L2594）删 | delete 项 8 |
| `EngineArgs._set_default_max_num_seqs_and_batched_tokens_args` | arg_utils.py:L2712-L2801 | throughput 翻倍、非 chunked 抬底、min 封顶逐字保留；mm prefix-LM 抬底（L2762-L2777）删 | must_keep；mm 分支不在删除单但依赖 MULTIMODAL_REGISTRY（本章边界外，见 §机械删除） |
| `AsyncEngineArgs` | arg_utils.py:L2804-L2808 | 子类 + `enable_log_requests`；async CLI 扩展删 | must_keep：vllm serve 入口同一条装配线的证据 |
| `VllmConfig`（字段子集） | vllm/config/vllm.py:L331-L429 | 保留 25 字段；offload/observability/reasoning/ec-manager 字段随其透传删除而删 | delete 项 3 的下游 |
| `VllmConfig.__post_init__` | vllm/config/vllm.py:L972-L1600 | 主线保留：instance_id → try_verify → model↔parallel 互验（含 heads%TP）→ routed_experts/mamba 校验 → 量化缺省 → **async 三态全保留（L1052-L1143 逐字）** → disable_nccl 推导 → enforce_eager/TORCH_COMPILE_DISABLE 覆盖 → mode 按 O 级 → custom_ops all/none → 预设应用 → autotune 断言 → mode/cudagraph 一致性守卫（L1310-L1321）→ enforce_eager 分支（L1424-L1430）；deep_gemm/Turing/breakable/blocked-weights/SP/fast_moe/pooler-encoder-kvconnector 降级删（delete 项 5） | 章节主线＝交叉校验+async 三态+O0-O3 落地 |
| `VllmConfig.try_verify_and_update_config` | vllm/config/vllm.py:L2055-L2115 | 骨架保留（None 短路/config_updated 守卫/查表分发）；MODELS_CONFIG_MAP 空 seam；hybrid/classify/runai 钩子删 | must_keep；架构改写钩子不在删除单（见 §机械删除） |
| `VllmConfig.compute_hash` | vllm/config/vllm.py:L431-L537 | 保留 version/model(+mm 条件)/cache/parallel/scheduler/compilation/kernel/additional 七类因子；device/load/offload/attention/lora/spec/structured/profiler/observability/kv_transfer/ec_transfer 同构追加删 | delete 项 6 |
| `_set_config_default` + `_apply_optimization_level_defaults` | vllm/config/vllm.py:L811-L853 | 逐字（含 apply_recursive 闭包） | must_keep：「只填 None」的优先级保证 |
| `OptimizationLevel` + 4 个预设 + 查表 | vllm/config/vllm.py:L104-L116, L229-L327 | 枚举/O0/O2 逐字；O1/O3 补齐（v0.27.1 各 12 个 fusion 旗标）；谓词函数体（L131-L227）降为通用 CUDA 路径默认值（见 §机械删除） | must_keep；函数值默认机制原样保留 |
| `ParallelConfig.__post_init__`（backend 推导） | vllm/config/parallel.py:L831-L989 | world_size/external_launcher/env 回退（默认路径）/uni-mp-ray 推导（L911-L956 逐字）/GPU 不足报错全保留；elastic-EP/EPLB communicator 删 | mechanism ch03-backend-derivation |
| `ParallelConfig.compute_hash` | vllm/config/parallel.py:L774-L829 | ignored_factors 集合逐字（backend/rank/地址等拓扑全忽略）；仅 `_data_parallel_master_port_list`/`_coord_store_port` 两行随其本身已删的 init=False 字段一并删 | 「改 backend 不变哈希」测试的法律依据 |
| `SchedulerConfig` | vllm/config/scheduler.py:L26-L285 | 字段/默认值/get_scheduler_cls（工厂②）/compute_hash（只收 token 预算）/__post_init__/verify_max_model_len 全保留；qualname 解析删 | must_keep ×2 |
| `Executor.get_class` + `supports_async_scheduling` | vllm/v1/executor/abstract.py:L47-L92, L363-L368 | 查表逐字（type/ray/mp/uni/external_launcher）；ray-v2 子开关（L61-L64）与 qualname（L81-L87）删；Uni/Multiproc 覆写 True（uniproc:L146 / multiproc:L526 现核） | delete 项 7；工厂① |
| `EngineCoreClient.make_client` / `make_async_mp_client` | vllm/v1/engine/core_client.py:L89-L139 | mp×asyncio 二维表逐字（async+非mp→NotImplementedError）；make_async_mp_client 的 DP 分流（L133-L138）删→直接 AsyncMPClient | delete 项 9；工厂③ |
| `InprocClient` | core_client.py:L306-L317 | `__init__ = EngineCore(*args)` 逐字；请求面删 | must_keep：逃生舱格子 |
| `MPClient` / `SyncMPClient` / `AsyncMPClient` | core_client.py:L503 / L802 / L974 | 层级保留（Sync/Async 继承 MPClient，与真实树一致，docstring 逐字）；MPClient 的 ZMQ/spawn 机身（L503-L801）与两客户端请求面删，stub 只记录入参（签名逐字对齐 L806-L808 / L978-L986） | 工厂③产物类；mp 拓扑属 ch5 |
| `EngineCore.__init__` | vllm/v1/engine/core.py:L106-L248 | 装配主干保留：executor_class 实例化（L132）→ `_initialize_kv_caches`（L143，GPU profiling 删→stand-in seam）→ StructuredOutputManager → get_scheduler_cls（L147）→ 无 KV 守卫 → resolve_kv_cache_block_sizes → Scheduler(...) 装配（L160-L168）；尾段 L169-L248（spec/批队列/握手/step_fn 绑定/idle 回调+freeze_gc_heap）删 | 章终点＝装配完成；运行循环属 ch9/ch12 |
| `LLMEngine.from_engine_args` / `from_vllm_config` / `__init__` | vllm/v1/engine/llm_engine.py:L51-L186 | from_engine_args 逐字（含 VLLM_ENABLE_V1_MULTIPROCESSING 翻转 L174-L176）；__init__ 只保留 make_client(asyncio_mode=False) 调用，三件套删 | code_spine 起点；WC2 |
| `AsyncLLM.from_engine_args` / `from_vllm_config` | vllm/v1/engine/async_llm.py:L205-L257 | 两个 classmethod 逐字；__init__ 只保留 make_async_mp_client（L149-L156） | 双使用面同一装配线证据 |
| `LLM.__init__`（入口子集） | vllm/entrypoints/llm.py:L295-L341 | kwargs→EngineArgs→from_engine_args 主干保留（~25 kwargs）；~90 个其余 kwargs 与 mm/profiler 预构删 | code_spine 首站 |

## 删除台账

### dossier subtraction_plan 九项 delete（全部执行，位置见上表）
1. Ray runtime env / placement group（arg_utils L1989-L2014）✓
2. DP 负载均衡推导大段（L2016-L2189）✓ — DP=1 默认值按原 else 分支内联
3. TurboQuant/LoRA/attention-mamba-kernel-ir_op_priority 覆盖样板/offload-reasoning-observability 透传 ✓ — CompilationConfig 深拷贝+覆盖（L2427-L2443，code_spine 站点）显式保留
4. speculator 覆盖段 + HF offline 路径替换 ✓
5. vllm.py 硬件/特性边角校验七段 ✓ — enforce_eager 分支（L1424-L1430）按计划保留
6. compute_hash 同构追加（保留七类代表因子）✓
7. Executor.get_class ray-v2 子开关 + qualname 分支 ✓
8. get_batch_defaults TPU/CPU 分支 ✓
9. make_async_mp_client DP 分流 ✓

### 机械删除（不在 delete 单、为可跑性/章节边界所必需——**请 reviewer 逐条过目**）
| 位置 | 内容 | 理由 |
|---|---|---|
| arg_utils L822-L1669 | add_cli_args/from_cli 等 argparse 装配 | 依赖 FlexibleArgumentParser，host 不可跑；装配线研究起点是已构造的 EngineArgs（v2 同章同款处理） |
| vllm.py L131-L227 | O1/O2/O3 谓词**函数体** | ROCm/AITER/flashinfer 探测 host 不可跑；函数值默认机制原样保留，体降为通用 CUDA 路径值（False / TP>1）——与真实通用路径求值结果一致（custom_ops=none 时 is_custom_op_enabled→False） |
| vllm.py L1434 | `_set_cudagraph_sizes()` 调用 | 捕获尺寸计算＝ch19 机制（章节边界明示只指门牌） |
| vllm.py L1439-L1600 | __post_init__ 尾段（kv_sharing/Whisper/kv_events/v2 runner/compile ranges/splitting ops/SP-PP/cascade/ubatching） | 编译内部（ch19）+特性边角；主线（L1300 预设应用+一致性守卫）已完整 |
| vllm.py L2086-L2115 | is_hybrid/classify/runai 三个查表后钩子 | 查表分发骨架保留；钩子触发旗标默认关，依赖 ModelRegistry/HookConfig |
| arg_utils L2679-L2710 | `_get_min_mm_batched_tokens` | 依赖 MULTIMODAL_REGISTRY；调用点守卫 `is_multimodal_model and is_mm_prefix_lm` 默认 False |
| ModelConfig 等 14 个子 Config | 字段子集化 + HF 读取删 | 与 CacheConfig/SchedulerConfig 构造同构的重复字段；每个保留字段与真实属性同名同默认（见源内逐字段行号） |
| core.py `_initialize_kv_caches` 体 | GPU profiling → stand-in | 显存账本三步定账＝ch14，需 GPU；stand-in 保证无 KV 守卫不误触发 |
| `resolve_kv_cache_block_sizes` 体 | 后端算术 → block_size 直通（真身 vllm/v1/core/kv_cache_utils.py:L626） | hybrid 注意力细节＝ch14 |

### seam 清单（host 可跑的最小接缝，均为 vLLM 自身经过的决策口）
`Platform`（current_platform 接口子集：device_type/is_*/device_count/显存/设备名——测试注入翻分支）·
`envs`（VLLM_ENABLE_V1_MULTIPROCESSING=True 按 envs.py:L149 等六个旗标）·
`logger`（info_once/warning_once）· `load_general_plugins` no-op ·
`safe_hash`（md5，FIPS 回退删）· `hash_factors`（JSON 序列化加 `default=str` 以容纳 IntEnum——真实 normalize_value 的等价简化）·
`MODELS_CONFIG_MAP`（空表 seam）· `resolve_kv_cache_dtype_string`（"auto" 直通，HF 量化探测删）。

## 测试矩阵（tests/test_config_wiring.py，32 用例）

- 站 1-2：字段默认借子 Config（含 v0.27.1 的 gpu_memory_utilization=0.92）/ dict 升格 / FT 自动开启。
- 站 7：get_batch_defaults H100/A100/小卡/探测失败四象限；usage_context+throughput 翻倍（用户值不翻）；非 chunked 抬到 max_model_len；min 封顶；seqs 落到 token 预算。
- 站 7：uni/mp 推导 + GPU 不足 ValueError。
- 工厂①：查表五分支 + type 校验 + 未知串报错；supports_async_scheduling 反向查询（base False / uni-mp True / external 继承）。
- 站 10-11：heads%TP 互验；async 三态（None→True 默认心跳 / pooling→False / 显式 True×不支持执行器→raise / ×medusa→raise / ×eagle→过 / 显式 False 跳检查）；disable_nccl 默认推导。
- 站 12：O2 默认（VLLM_COMPILE+FULL_AND_PIECEWISE+autotune+custom_ops none）；O0 纯 eager；O1 PIECEWISE；用户显式胜预设；enforce_eager 双 NONE+capture 清零；TORCH_COMPILE_DISABLE=1 经一致性守卫连 cudagraph 也 NONE。
- 站 17：compute_hash 十位 hex/确定性；max_num_seqs 不入、max_num_batched_tokens 入、backend 不入、TP 入。
- 工厂③：2×2 表 + async+非 mp NotImplementedError；工厂②：async→AsyncScheduler/False→Scheduler/自定义类直通；verify_max_model_len 双守卫。
- 端到端：LLM→InprocClient→EngineCore→UniProcExecutor+AsyncScheduler+StructuredOutputManager+10 位指纹；默认 mp（SyncMPClient）；AsyncEngineArgs→OPENAI_API_SERVER 8192→AsyncMPClient。

## 已知偏差（writer/reviewer 需知）

1. 谓词函数体与若干平台分支降为「通用 CUDA 路径值」——机制（函数值默认/只填 None）不变，具体谓词在 ROCm/AITER 环境下的取值不在本章范围。
2. `_initialize_kv_caches`/`resolve_kv_cache_block_sizes` 为 stand-in——真值属 ch14 显存账本。
3. `hash_factors` 的 `default=str` 与真实 `normalize_value` 在枚举等类型上等价序列化，哈希值本身与真实 vLLM 不逐位对齐（真实哈希含 __version__/全字段集）——本章只消费「哪些因子入哈希」的作用域语义。
4. v0.27.1 的 `LLM.__init__` seed 参数为 `int = 0`（非 Optional）——精简版与真实一致（曾误写 Optional，已改）。
5. `LLM.__init__` 入口子集中 9 个 kwarg（max_model_len / max_num_batched_tokens /
   max_num_seqs / enable_chunked_prefill / async_scheduling / optimization_level /
   performance_mode / distributed_executor_backend / speculative_config）在真实入口走
   `**kwargs` 透传（llm.py:L177-L294 不显式声明）——精简版为展示本章旋钮显式化，行为等价
   （同样进 EngineArgs）；已在代码 SUBTRACTED 注记中说明。compilation_config=None 的路由
   两版等价：真实 `_make_config(None)→CompilationConfig()` 新鲜默认 vs 精简版省略 kwarg→
   EngineArgs default_factory 同样新鲜默认。
6. `EngineCore.__init__` 里真源以局部变量 `Scheduler` 遮蔽同名类再实例化（core.py:L147/L160），
   精简版按真源逐字保留该遮蔽写法。
