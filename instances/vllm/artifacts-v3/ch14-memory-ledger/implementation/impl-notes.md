# ch14 显存账本 — impl-notes（只做减法精简版）

对应真实源码 pin **vLLM v0.27.1 (6e448d0ea)**，行号全部本日现核（2026-08-27，
`instances/vllm/source`，HEAD == tag v0.27.1），**不是** v2 资产的 v0.21.0 旧行号。
运行：`cd instances/vllm/artifacts-v3/ch14-memory-ledger && python -m pytest tests/ -q`
（81 passed；纯 host 单元/契约测试，不 import vllm——设备读数/前向机器以
ENGINE SEAM/HOST SEAM 承载，见下）。

本章精简版跑 **enable_prefix_caching=False** 支（cache_config 的正交开关）：
`get_kv_cache_coordinator` 的 `if not enable_caching: return KVCacheCoordinator
NoPrefixCache(...)` 是源码原生路径（kv_cache_coordinator.py:L864-L876），且
**支持任意组数（含 0 组）**——正是本章混合组化的主路径。控制流闭合点：
find_longest_cache_hit 恒空命中、add_local_computed_blocks 走 not enable_caching
断言支、cache_blocks 由 allocate_slots 的 False 早退短路（L551-L552）。

## 骨架（dossier.delete 第 13 条的指示：「以门段+组化+定账为骨架」）

三幕：**定账**（request_memory → determine_available_memory → get_kv_cache_
configs → 一份账喂两侧）+ **门**（full-ISL 准入门 + 回收感知准入上限 + 水位）
+ **组化**（页统一 → 等量化组 → 张量共享布局 → LCM/GCD 对齐 → SWA 窗外回收
→ kernel 块细分）。挂块/分块的 allocate_slots 门内尾部（L529-L547）与 ch13
同源逐字保留（ch13 已建精简版；本章复用同一份源码行，保证门通过后可真实分配）。

## 包结构（与真实树同名同构；v1/worker/utils.py → worker_utils.py 防重名）

| 精简版文件 | 真实文件 | 本章切面 |
|---|---|---|
| `mem_utils.py` | `vllm/utils/mem_utils.py` | MemorySnapshot/MemoryProfilingResult/memory_profiling 三件套（三类显存 + 峰值账） |
| `worker_utils.py` | `vllm/v1/worker/utils.py`（+attention/backend.py:L49 MultipleOf 折入） | request_memory + select_common_block_size/prepare_kernel_block_sizes |
| `gpu_worker.py` | `vllm/v1/worker/gpu_worker.py` | Worker：init_device 快照尾段 / determine_available_memory / get_kv_cache_spec / update_max_model_len / initialize_from_config / _maybe_get_memory_pool_context |
| `gpu_model_runner.py` | `vllm/v1/worker/gpu_model_runner.py` | profile_run / _init_minimal_kv_cache_for_profiling / profile_cudagraph_memory（估计算术）/ get_kv_cache_spec / initialize_kv_cache(_tensors) / _allocate+_reshape |
| `kv_cache_spec_registry.py` | `vllm/v1/kv_cache_spec_registry.py` | KVCacheSpecRegistry（spec→manager/分组基类注册表） |
| `kv_cache_interface.py` | `vllm/v1/kv_cache_interface.py` | KVQuantMode + spec 家族（Full/SWA/ChunkedLocal/Mamba/UniformType）+ 准入上限两法 + KVCacheTensor/GroupSpec/Config |
| `kv_cache_utils.py` | `vllm/v1/core/kv_cache_utils.py` | KVCacheBlock + FreeKVCacheBlockQueue（ch13 同款原语）+ 定账/护栏/组化/布局/对齐/容量全套 |
| `block_pool.py` | `vllm/v1/core/block_pool.py` | 池构造（null 占 0）+ get_new_blocks/touch/free_blocks + get_num_free_blocks/get_usage |
| `single_type_kv_cache_manager.py` | `vllm/v1/core/single_type_kv_cache_manager.py` | 基类（含 admission cap 夹取/remove_skipped/_remove_blocks_in_range）+ Full/SWA/Chunked managers + get_manager_for_kv_cache_spec + register_all_kvcache_specs |
| `kv_cache_coordinator.py` | `vllm/v1/core/kv_cache_coordinator.py` | 基类扇出（一个池 + 每组一 manager）+ NoPrefixCache + get_kv_cache_coordinator |
| `kv_cache_manager.py` | `vllm/v1/core/kv_cache_manager.py` | KVCacheBlocks 包装 + KVCacheManager（watermark 装配 + allocate_slots 全三段：full-ISL 门/窗外回收/稳态门） |
| `block_table.py` | `vllm/v1/worker/block_table.py` | get_block_table_width + BlockTable（kernel 细分）+ map_to_kernel_blocks + MultiGroupBlockTable |
| `v1_utils.py` | `vllm/v1/utils.py` | CpuGpuBuffer（CPU/GPU 双镜像） |
| `engine_core.py` | `vllm/v1/engine/core.py` | EngineCore 装配序（_initialize_kv_caches 全编排 + collective_rpc） |
| `scheduler.py` | `vllm/v1/core/sched/scheduler.py` | 站点抽块：KVCacheManager 构造（L276-L290）+ 入场调用点（L965-L985） |
| `cache.py` | `vllm/config/cache.py` | CacheConfig 账本位（util/override/prefix_match_unit/写回账） |
| `scheduler_config.py` | `vllm/config/scheduler.py` | SchedulerConfig 两道门 + 组化回退开关 |
| `config.py` | `vllm/config/vllm.py`（+model/parallel/compilation 折入） | VllmConfig seam + max_in_flight_tokens + CUDAGraphMode |
| `request.py` | `vllm/v1/request.py` | Request 两道门消费面 |
| `envs.py` / `math_utils.py` / `torch_utils.py` | 对应真实文件 | 估计开关 / cdiv / get_dtype_size |

## 1:1 Source Map（关键段；改动=减法或 seam，原因=批准条/章节边界）

| 精简版 | 真实源 | 改动 | 原因 |
|---|---|---|---|
| `worker_utils request_memory` | `worker/utils.py:L409-L429` | 逐字（ceil(total×util) + free 不足 raise） | m1 must_keep「预算先于一切」 |
| `mem_utils memory_profiling` | `mem_utils.py:L233-L326` | docstring 量化例逐字；measure() 的设备读数为 HOST SEAM（host 测试 monkeypatch 注入，容器真跑）；UMA/platform 分支删 | m2；三类显存账 non_kv = total_consumed + transient |
| `gpu_worker init_device_snapshot_tail` | `gpu_worker.py:L372-L396` | 站点抽块：NCCL 先行注释逐字保留（station 1 why 锚），分布式装配体删（→ch05） | m1 站 1 |
| `gpu_worker determine_available_memory` | `gpu_worker.py:L459-L611` | L498-L548 核心段逐字（memory_profiling+profile_run+cudagraph 门+开关+减法+快照 assert）；startup_plan/kv_cache_memory_bytes 早退删（第 2 条）、mm IPC 包装删（第 3 条）、debug/等价 util 日志删（第 11 条） | m1/m3/m16 |
| `runner profile_run` | `gpu_model_runner.py:L6433-L6506` | mm 段删（第 1 条）；dummy 前向+采样器+sync+gc 逐字；PP/pooling 分支删 | m2 站 2 |
| `runner _init_minimal_kv_cache_for_profiling` | `gpu_model_runner.py:L6508-L6534` | 逐字——临时 num_gpu_blocks_override=min_blocks 再还原（账本机器复用，dossier m3 锚点） | m3 |
| `runner profile_cudagraph_memory` | `gpu_model_runner.py:L6645-L6811` | 头（图数清点+0 早退）+采样循环骨架+估计算术（first_capture+max(1MiB,per-graph)×(n−1)、跨 mode 取 max）逐字；捕获机器内景/graph_pool 调换/encoder 管理器删（→ch19/第 1 条） | m3 |
| `runner get_kv_cache_spec` | `gpu_model_runner.py:L7800-L7837` | 遍历+backend indexes 判定逐字；ec_transfer 早退（→ch16）与 kv 共享跳过（ch13 边界）删；attn_layers 为 ENGINE SEAM 注入面 | 站 4 |
| `runner initialize_kv_cache(_tensors)` | `gpu_model_runner.py:L7541-L7594、L7624-L7661` | general 分支逐字：_allocate 每层 int8 原始缓冲（packed 别名删，第 4 条）+ _reshape 的 kernel 细分乘子（256→4×64 注释逐字）；backend 形状仲裁 → ch21（说明性布局 [kernel_num_blocks, 每块字节]）；metadata builders/input batch（→ch21/18）删 | m14 站 12 |
| `interface AttentionSpec.real_page_size_bytes` | `kv_cache_interface.py:L211-L226` | 2×bs×heads×head_dim×dtype 逐字；nvfp4/int4 特判删（→ch27） | m1 页公式 |
| `interface FullAttentionSpec` | `kv_cache_interface.py:L234-L350` | max_memory_usage/merge/merge_window_sizes/real_page(head+head_v 版) 逐字；MLA 排除桩删（第 4 条） | m5 |
| `interface ChunkedLocal/SlidingWindow max_admission_blocks_per_request` | `kv_cache_interface.py:L519-L546、L587-L618` | 逐字（含 docstring「Single source of truth」与 +1 的 [XXCD][EF] 例）；max_memory_usage_bytes = cap×page 逐字 | m11 must_keep 单源铁律 |
| `interface MambaSpec` | `kv_cache_interface.py:L709-L758` | page_size_bytes/max_memory_usage_bytes 三 mode 逐字；max_num_blocks_per_req 行宽推导删（→邻章） | m6 pad 原料 |
| `utils resolve_kv_cache_block_sizes` | `kv_cache_utils.py:L626-L688` | 逐字（LCM/GCD/prefix_match_unit 整除校验/无缓存回退/mamba 非 align 回退）；dcp 乘位保留值恒 1 | m8 |
| `utils _check_enough/estimate/check_enough` | `kv_cache_utils.py:L751-L879` | 逐字（报错文案含二分估长提示；try/finally 恢复） | m4 护栏 1+2 |
| `utils get_num_blocks/may_override/_pool_bytes_per_block` | `kv_cache_utils.py:L962-L1010、L972-L990` | 逐字（//page//group_size + max(0) + override 凌驾）；packed 分支删（第 4 条） | m1 总算术 + m4 override 折算的每块字节口径 |
| `utils unify_kv_cache_spec_page_size` | `kv_cache_utils.py:L1070-L1132` | 逐字（调大 block_size / Mamba pad / stride pad / NotImplementedError） | m6 |
| `utils _get_kv_cache_groups_uniform_page_size` | `kv_cache_utils.py:L1140-L1280` | 六条假设 docstring + 分桶合并 + 1.5 启发式 + padding warning + layers[i::n] 交错逐字 | m5 |
| `utils get_kv_cache_config_from_groups` | `kv_cache_utils.py:L1361-L1443` | 三型保留两型：空/单组异宽（逐层张量）/通用 group_size 池（每池每组出一层）；packed 分支删（第 4 条，正文 why 注点名） | m7 |
| `utils unify_hybrid_kv_cache_specs(+_promote)` | `kv_cache_utils.py:L1446-L1589` | 逐字（SWA/chunked→Full promote + warning 原话「we do not enable any optimizations」）；MLA/HiddenState/RSWA promote 分支删（第 4/5 条） | m5 回退 |
| `utils get_kv_cache_groups` | `kv_cache_utils.py:L1781-L1852` | 三路（disable 回退/uniform/uniform-type）+页统一 try-fallback 逐字；DSV4 分支（第 4 条）与 HiddenState 抽离（第 5 条）删 | m5 总入口 |
| `utils generate_scheduler_kv_cache_config/get_kv_cache_capacity/get_max_concurrency` | `kv_cache_utils.py:L1855-L1887、L937-L959` | 逐字（num_blocks 一致断言 + 拍平 + 容量=并发×max_model_len 按组求和） | m9/m15 |
| `utils _max_memory_usage_bytes_from_groups/_estimate_from_groups/_auto_fit` | `kv_cache_utils.py:L1890-L2049` | 单组异宽支 + 通用支（group_size×page×Σcdiv）逐字；全 UniformType DSV4 支删（第 4 条） | m4 护栏 3 |
| `utils _project_kv_cache_groups_to_worker` | `kv_cache_utils.py:L2052-L2091` | 过滤逐字；UniformType 逐层重建内景删（第 12 条「保留函数与过滤逻辑即可」）；is_eagle_group 透传删（第 6 条） | m4 PP 投影 |
| `utils get_kv_cache_configs` | `kv_cache_utils.py:L2094-L2242` | 五步全逐字（合并断言/注册表检查/分组/PP 投影/override 折算/auto-fit/逐 worker 护栏/出 config/PP 取最小缩张量/容量日志） | m1/m4 定账总控 |
| `pool BlockPool` | `block_pool.py:L162-L196、L647-L742、L799-L818` | 构造（null 占 0）+get_new_blocks/touch/free_blocks/get_num_free_blocks/get_usage 逐字；哈希表/驱逐/CoW 删（→ch15），free_blocks 哈希劈分两行**原样保留**（False 支全走 append_n）；events/metrics 删（第 9 条） | ch13 同款 + m13 回收归池 |
| `single_type get_num_blocks_to_allocate` | `single_type:L144-L230` | cdiv+cap 夹取（#39734 注释原文逐字）+fast-path+skipped 推导+evictable 逐字；partial-hit +1 删（→ch15） | m11 预测器 |
| `single_type remove_skipped/_remove_blocks_in_range/get_num_skipped_tokens` | `single_type:L595-L672` | 逐字（逆序 null 换位遇 null 早停） | m13 must_keep |
| `single_type SlidingWindowManager.get_num_skipped_tokens` | `single_type:L1057-L1083` | 逐字（含 8-token 窗 4 ASCII 图 docstring：max(0, computed−window+1)） | m13 |
| `single_type ChunkedLocalAttentionManager.get_num_skipped_tokens` | `single_type:L1200-L1244` | 逐字（三个 chunk 对齐例 docstring：//chunk×chunk） | m13 |
| `single_type get_manager_for_kv_cache_spec/register_all_kvcache_specs` | `single_type:L1836-L1878、L1881-L1942` | 查建+SWA/chunked cap 注入（single source 注释逐字）逐字；注册缩到 Full/SWA/Chunked（TQ/MLA/RSWA/Cross/Hidden/SinkFull/Mamba 注册随对应条删——mamba manager → 邻章） | m11 装配点 |
| `coordinator` | `kv_cache_coordinator.py:L60-L128、L130-L190、L336-L357、L385-L432、L851-L903` | 一个 BlockPool + 每组一 manager（max_in_flight 由此进）+ 扇出（apply_admission_cap 只由门传 True）逐字；retention（第 7 条）/eagle（第 6 条）/events+metrics（第 9 条）/DCP（第 8 条）/external 扇出（第 13 条）/Cross 静态分配（第 5 条）删 | m11+m13 装配 |
| `manager allocate_slots` | `kv_cache_manager.py:L344-L565` | 全三段逐字：水位条件（L463-L470 只对 WAITING/PREEMPTED+has_scheduled）→full-ISL 门（L472-L488 含 apply_admission_cap=True）→remove_skipped 先回收（L495-L508）→稳态门 required≤free−reserved（L510-L527）→挂命中+分块+写回；lookahead/external/encoder/delay 参数分支删（第 13 条）；watermark_blocks 装配（L168-L171）逐字 | m10/m12 绝对主角 |
| `scheduler` | `scheduler.py:L272-L307、L965-L985` | 站点抽块逐字：KVCacheManager(watermark=..., max_in_flight_tokens=...) 构造 + scheduler_reserve_full_isl 绑定 + allocate_slots 三预算调用点；async load 预约计算删（→ch16，reserved_blocks=0 参数位保留） | m10 站 8/9 |
| `engine_core _initialize_kv_caches` | `core.py:L250-L359、L142-L168` | 编排逐字：register → 收 spec → determine_available_memory → get_kv_cache_configs → auto-fit 同步 collective_rpc → 拍平喂调度器+写回 cache_config 四件 → initialize_from_config；non_causal（第 10 条）/弹性 EP（第 10 条）/warmup（→ch19）/编译观测（第 9 条）删 | m9 站 7 |
| `block_table BlockTable/MultiGroupBlockTable` | `block_table.py:L20-L40、L48-L154、L220-L248、L270-L376` | 细分判定（use_hybrid_blocks/blocks_per_kv_block）+append_row(map_to_kernel_blocks 展开)+map_to_kernel_blocks 纯 numpy 算术逐字；slot_mapping kernel（→ch13/ch22）、CP（第 8 条）、NONE mode（→邻章）删 | m14 |
| `worker_utils prepare_kernel_block_sizes/select_common_block_size` | `worker/utils.py:L266-L376` | 逐字（后端协商的纯算术；协商内景深讲 → ch21） | m14 |

## HOST SEAM / ENGINE SEAM 登记（跨章边界，非减法）

- **MemorySnapshot.measure / torch.accelerator**（mem_utils.py）：CUDA 设备读数面。
  host 测试 monkeypatch 注入（tests 的 docstring 量化例 oracle：non_kv=5GiB）；
  容器内真跑。empty_cache/reset_peak 是 CUDA 缓存语义，host no-op。
- **GPUModelRunner 的执行机器**（gpu_model_runner.py）：`_dummy_run`/
  `_dummy_sampler_run`/`_sync_device`（→ch09/17）、`_warmup_and_capture`+
  `cudagraph_dispatcher`（→ch19）、`attn_layers`（真实 get_layers_from_
  vllm_config）、`attn_groups`（→ch21 后端分组）、`compilation_config`/
  `max_num_reqs` 账位直供。切面 no-op 本体 + 同契约位注入——采样循环/估计
  算术/调用序逐字（tests 以 fake desc + get_memory_info 注入验证估计器）。
- **Worker.model_runner 注入**（gpu_worker.py）：真实 Worker 自建 runner
  （L408-L423 → ch17）；切面构造期直供。`current_platform` 为 HOST SEAM 平台位
  （host=CPU：池上下文恒 nullcontext、cudagraph 门由测试以 CUDA 替身打开）。
- **_maybe_get_memory_pool_context**（gpu_worker.py:L256-L276）：三分支
  nullcontext 判定逐字；allocator.use_memory_pool(tag=...) 的设备分配器接线
  （device_allocator/cumem.py:L313）删——sleep/offload 域（→ch16/19），本章
  只消费 tag="kv_cache" 的调用位（must_keep 符号在 SUBTRACTED 标注内点名）。
- **_reshape_kv_cache_tensors 的形状仲裁**（gpu_model_runner.py）：真实
  attn_backend.get_kv_cache_shape（主流后端内容维打包）→ ch21；切面用块最外层
  说明性布局 [kernel_num_blocks, 每块字节]（ch13 同款），页字节数与 kernel
  细分乘子逐字。
- **LOGGER SEAM**：vllm.logger.init_logger → stdlib logging.getLogger——
  warning/info 账目同构（padding 浪费上界、disable 回退警告、容量/并发日志、
  auto-fit 决策），tests 以 caplog 断言原话。
- **VllmConfig seam**（config.py）：max_in_flight_tokens 属性（vllm.py:L553-
  L561）逐字；compilation_config/kv_transfer_config/speculative_config 以账位
  直供（各邻章装配域）。

## 减法批准对照（dossier.subtraction_plan.delete 13 条 → 落点）

1. profile_run 的 mm encoder profiling 段（L6434-L6490）→ 删（gpu_model_
   runner.py 头注 + 站内标注）；encoder_cudagraph_manager 同族删。
2. kv_cache_memory_bytes 手动凌驾早退（L474-L496）+ maybe_apply_startup_plan
   （L472）→ 删；CacheConfig 字段账位保留。
3. reserve_mm_ipc_gpu_memory 包装（L492-L496/L607-L611）→ 删，返回 int。
4. DSV4/SlidingWindowMLA 特路 + packed 布局三件 + group_and_unify + 全
   UniformType DSV4 支 → 删（kv_cache_interface.py/kv_cache_utils.py 各标注）。
5. R-SWA/SinkFull/Cross/HiddenState 族（spec+manager+分组分支）→ 删。
6. eagle（use_eagle/eagle_group_ids/is_eagle_group/_annotate）→ 删。
7. retention_interval 校验与 VLLM_PREFIX_CACHE_RETENTION_INTERVAL → 删（→ch15）。
8. DCP/PCP 乘子与 CP 交错 → 删（乘位保留、值恒 1；resolve 的 ×dcp 逐字位保留）。
9. metrics_collector/KVCacheMetricsCollector/enable_kv_cache_events 全部调用
   → 删（观测面；vllm.logger → stdlib 为 LOGGER SEAM 非观测数据面）。
10. non_causal 检查段（L259-L279）与弹性 EP（L139-L140、L283-L289、L331）→ 删。
11. weight_transfer_config（L444-L450）与 debug/等价 util 日志块（L550-L605）
    → 删（教学金句正文引用）。
12. _project 中 UniformType 逐层重建内景（L2076-L2083）→ 删（函数+过滤保留）。
13. allocate_slots 的 lookahead/encoder/external(delay) 参数分支 → 删
    （L490-L493 封顶/external 合账/挂块半边；参数面以 0/缺省账位保留）。

## 与 must_keep 的对账

dossier.subtraction_plan.must_keep 全部 74 项逐一在精简版中以**实现体**存在
（lint_fidelity over_subtraction 0 报告）；唯二说明：`use_memory_pool`——本章
只消费其调用位（initialize_from_config 的 `with self._maybe_get_memory_
pool_context(tag="kv_cache")` 逐字），分配器接线本体属 sleep/offload 域
（→ch16/19），在 SUBTRACTED 标注内点名；`_max_in_flight_tokens /
max_in_flight_tokens`——leaf 为复合词条，实际以 VllmConfig.max_in_flight_tokens
属性 + KVCacheManager 参数 + coordinator 透传三处落地。

## 行号勘误（相对 dossier 锚点）

- dossier m3 第 2 锚 `gpu_model_runner.py:L6508-L6532` 实为
  `_init_min_kv_cache_for_profiling`（已按真身实现并标注）；真正的
  `profile_cudagraph_memory` 在 **L6645-L6811**（本 impl 按真身行号标注）。
- dossier 站 6 引 `_check_enough L854-L879` 实为 check_enough_kv_cache_
  memory 外包装（L854-L879 ✓）；本体 _check_enough_kv_cache_memory 在
  L751-L788（两者都已实现）。其余抽查锚点（request_memory/L409-L429、
  memory_profiling/L233-L326、get_num_blocks/L993-L1010、resolve/L626-L688、
  组化/L1781-L1852、定账/L2094-L2242、准入上限/L519-L546+L587-L618、
  夹取段/L178-L192、SWA 回收/L622-L659+L1057-L1083、kernel 细分/L220-L248、
  装配序/L250-L359）全部与 v0.27.1 现核一致。

## 测试（tests/test_memory_ledger.py，81 passed）

纯 host 单元/契约测试（不 import vllm）。oracle 全部取自真实源的可观测行为：
docstring 量化例（mem_utils 5GiB 三类显存账 / SWA 窗 4·computed 7→4 / chunked
8:13→8,8→8,7→0 / map_to_kernel_blocks [0,1,2]→[0..5]）、源码公式（cap =
cdiv(window−1+in_flight, bs)+1、num_blocks = avail//page//group_size、
LCM/GCD、并发 = num_blocks/Σcdiv）、warning 原话（padding 上界 / disable 回退）。
覆盖 16 个 mechanisms：m1 三步定账与 worked example（Llama-7B 8GiB→1024 块→
4×并发）、m2 峰值账、m3 估计器（first+per-graph×(n−1)、跨 mode max、开关与
NONE 门）、m4 四道护栏（报错文案/二分边界 1600/override 折算不漂/PP 取最小
缩张量/auto-fit 触发 RPC）、m5 组化（5:1 拆组交错、12+13→13/13 的 1.5 启发式、
disable 回退）、m6 页统一（调大/pad/raise/stride）、m7 两型布局、m8 LCM/GCD
五路、m9 写回与拍平、m10 full-ISL（#39734 超收对照：门开 None/门关放进）、
m11 cap（公式/夹取只由门传/注入装配/混合求和）、m12 水位（精修版三条件）、
m13 SWA/chunked 回收与 null 占位、m14 kernel 细分与多组表、m15 容量核算、
m16 util 语义与快照 assert。容器内差分电池（钉版真源码 vs 精简版逐行 diff）
归 workflow 的 Test 站（ch13 同款纪律），本 impl-notes 不代claim。
