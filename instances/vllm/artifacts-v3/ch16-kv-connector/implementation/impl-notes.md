# ch16 KVConnector — impl-notes（只做减法精简版）

对应真实源码 pin **vLLM v0.27.1 (6e448d0ea)**，行号全部本日现核（2026-08-29，
`instances/vllm/source`，`git log -1` == 6e448d0ea == tag v0.27.1），**不是** v2
资产的 v0.21.0 旧行号。
运行：`cd instances/vllm/artifacts-v3/ch16-kv-connector && python -m pytest tests -q`
（60 passed；纯 host 单元/契约测试，不 import vllm——分布式/GPU 面以 HOST SEAM
承载，见下）。

本章精简版跑 **enable_prefix_caching=True**（默认开——本地前缀缓存与外部缓存
**双查**的主线）；`get_kv_cache_coordinator` 三态分派全保（NoPrefixCache=
『外部缓存是唯一缓存』对照支 / 单组 Unitary / 混合 Hybrid）。子块尾仲裁场景用
ch15 m15/m16 已立的 **partial-hit 粒度配置**（full(16)+mamba-align(16)、hash 8）
驱动——块内边界命中（`partial_tail≠0`）在真实源码里只在此形态出生（Unitary
断言 hash==block）。

## 主角文件（本章第一主角，近全文）

| 精简版文件 | 真实文件 | 本章切面 |
|---|---|---|
| `base.py` | `vllm/distributed/kv_transfer/kv_connector/v1/base.py` | **双面契约本体**：模块 docstring（契约正文）、KVConnectorRole、SupportsHMA+supports_hma、两份不透明信封（Metadata/WorkerMetadata）、KVConnectorBase_V1 全方法面（调度器侧五原语+worker 侧六原语+requires_kv_delivery+has_pending_push_work）。删：握手族/cross-layer 族/cudagraph 族/stats-events 族/reset_cache 等 optional 钩子的非默认分支（第 3/4/5/6/7 条） |
| `factory.py` | `.../kv_connector/factory.py` | 懒加载注册表 + create_connector（NOTE 原话逐字）+ HMA 门 + 外部模块路径优先。删：13 个后端注册行（留 ExampleConnector 一条示范）、MultiConnector 特例、旧 2 参构造告警 |
| `kv_transfer_state.py` | `.../kv_transfer_state.py` | worker 侧装配：全局 agent、三取用谓词、ensure_kv_transfer_initialized/shutdown。HOST SEAM：_sync_engine_id_across_tp 的 broadcast（单进程恒等，多机 → ch36） |
| `kv_transfer.py` | `vllm/config/kv_transfer.py` | KVTransferConfig：kv_role 三态+校验、failure policy、三谓词、get_from_extra_config。删：P/D 拓扑/握手参数（→ch36）、permute 布局、compute_hash |
| `kv_connector_model_runner_mixin.py` | `vllm/v1/worker/kv_connector_model_runner_mixin.py` | worker 一拍生命周期（_get_kv_connector_output 逐字）+ no_forward + maybe_get + finalize。删：uniform 布局两函数（第 12 条）、stats/events 两行（第 3 条） |
| `kv_transfer_utils.py` | `vllm/model_executor/layers/attention/kv_transfer_utils.py` | **61 行全保**——逐层钩子装饰器（契约最深的挂点） |
| `example_connector.py` | `.../v1/example_connector.py` | 官方 debug 参考实现全流程：ReqMeta slot 寻址、inject/extract、调度器侧两原语、build_connector_meta（调用即重置）。删：MLA reshape 两分支（第 8 条）、resumed-from-preemption 支 |
| `gpu_model_runner.py` | `vllm/v1/worker/gpu_model_runner.py`（+gpu_worker.py:L655-L679 折入） | worker 三挂点：execute_model 入口 handle_preemptions、无 token 步 no_forward、_model_forward 的 connector 包裹（EPLB 段删——dossier elide）；register_kv_caches 装配点；Worker.initialize_from_config 的装配序 |
| `scheduler.py` | `vllm/v1/core/sched/scheduler.py` | **connector 面全落点**（见 1:1 Map）：四旗标装配、双查+仲裁、护轨分配、WAITING 态、元数据过线、回收/失败/终局/边界三例外 |
| `kv_cache_manager.py` | `vllm/v1/core/kv_cache_manager.py` | **池侧 connector 面**：get_computed_blocks_for_connector、allocate_slots 的 ext_comp 段（五段布局注释图逐字）、truncate/zeroing/partial-tail 三开口 |

## 共享底座（ch13/14/15 已立、本章只消费的池侧栈；同源同减法）

| 精简版文件 | 真实文件 | 本章切面 |
|---|---|---|
| `single_type_kv_cache_manager.py` | `vllm/v1/core/single_type_kv_cache_manager.py` | 基类（块表/需块预测/挂块）+ **allocate_external_computed_blocks（本章核心：ext_comp 段分配半边）** + 零清账（records_new_block_ids/take_new_block_ids）+ partial-tail 队列 + FullAttentionManager.find（phase1/2）+ MambaManager 最小面（find 的 fine 分支——enable_partial_hash_hits 的装配前提）。删：CoW 三件套（第 10 条，与 +1 预留成对删除保分配算术一致）、SWA/Chunked/RSWA/Cross/SinkFull 管理器族、mamba align 分配内部 |
| `kv_cache_coordinator.py` | `vllm/v1/core/kv_cache_coordinator.py` | 三态分派全保 + **外部块第二相**（两阶段分配 #33775）+ **find_longest_cache_hit_per_group**（m3 发散判定半边）+ Hybrid 不动点 |
| `block_pool.py` | `vllm/v1/core/block_pool.py` | BlockHashToBlockMap + 池原语 + cache_full/partial_blocks（本地命中链路）+ evict_blocks（失败块逐出）+ touch（钉住原语）。删：事件族/metrics/reset_prefix_cache/move_block_hashes |
| `kv_cache_utils.py` | `vllm/v1/core/kv_cache_utils.py` | KVCacheBlock+FreeKVCacheBlockQueue+链式哈希族+resolve（ch15 同款切面） |
| `kv_cache_interface.py` | `vllm/v1/kv_cache_interface.py` | spec 最小面 + **needs_kv_cache_zeroing 派生谓词**（站 6/10 的开关：mamba 或混合精度） |
| `request.py` | `vllm/v1/request.py` | WAITING_FOR_REMOTE_KVS + 先行记账账位 + drop_stale/last_sched_seq/num_in_flight + 哈希账本 |
| `request_queue.py` | `vllm/v1/core/sched/request_queue.py` | FCFS 队列全原语（含 remove_requests）+ create_request_queue |
| `output.py` | `vllm/v1/core/sched/output.py` | SchedulerOutput 的 connector 面（kv_connector_metadata/partial_tail_offloads/new_block_ids_to_zero/finished_req_ids）+ New/CachedRequestData 最小面 |
| `outputs.py` | `vllm/v1/outputs.py` | KVConnectorOutput（finished_sending/recving/invalid_block_ids）+ ModelRunnerOutput connector 面 + with_kv_conn_output_only |
| `stats.py` | `vllm/v1/metrics/stats.py` | PrefixCacheStats（record 的 preempted 分账——外部命中率官方口径） |
| `forward_context.py` | `vllm/forward_context.py` | ForwardContext 最小镜像 + get/set/override（HOST SEAM：DP/cudagraph 装配删） |
| `attention.py` | `vllm/model_executor/layers/attention/attention.py` | get_attention_context（装饰器的层上下文访问器；DBO list 形态删） |
| `config.py`/`cache.py`/`scheduler_config.py` | `vllm/config/vllm.py`/`cache.py`/`scheduler.py` | VllmConfig seam（max_concurrent_batches/kv_transfer_config/static_forward_context）+ watermark/disable_hma/enable_prefix_caching 账位 |
| `hashing.py`/`torch_utils.py`/`math_utils.py`/`envs.py` | 对应真实文件 | sha256+safe_hash / _resolve_layer_name / cdiv / retention 环境位 |

## 1:1 Source Map（关键段；改动=减法或 seam，原因=批准条/章节边界）

| 精简版 | 真实源 | 改动 | 原因 |
|---|---|---|---|
| `base.py 模块 docstring` | `base.py:L3-L41` | 逐字（契约正文——两半原语清单） | m1 must_keep 第一条 |
| `base.py KVConnectorBase_V1.__init__` | `base.py:L196-L217` | 逐字（实验性警告+role 注入+必设校验） | m1 |
| `base.py requires_kv_delivery` | `base.py:L184-L194` | 逐字 | m13 must_keep |
| `base.py worker 侧四抽象` | `base.py:L304-L367` | 逐字（wait_for_save docstring 原话=正确性论据） | m8 must_keep |
| `base.py get_finished/get_block_ids_with_load_errors` | `base.py:L369-L405` | 逐字 | m9/m10 must_keep |
| `base.py 调度器侧三抽象` | `base.py:L465-L539` | 逐字 | m2/m6 must_keep |
| `base.py request_finished 族` | `base.py:L541-L578` | 逐字（True=接管） | m11 must_keep |
| `base.py SupportsHMA/supports_hma` | `base.py:L85-L121` | 逐字 | m11 must_keep |
| `base.py has_pending_push_work` | `base.py:L589-L599` | 逐字（含 TODO 注释） | m17 must_keep |
| `base.py` 其余钩子 | `base.py:L407-L720` | shutdown/reset_cache/get_finished_count 默认体保留；stats/events/握手/布局/cudagraph 删 | 第 3/4/5/6/7 条 |
| `factory.py create_connector` | `factory.py:L43-L75` | 逐字（NOTE『We build separately…』原话） | m1 must_keep |
| `factory.py register_connector/loader` | `factory.py:L30-L40` | 逐字（懒加载机制） | must_keep |
| `factory.py 注册行` | `factory.py:L152-L156` | 只留 ExampleConnector（module_path 改 `implementation.example_connector`——包重定位 seam） | 第 2 条『一条示范』 |
| `kv_transfer_state.py ensure_kv_transfer_initialized` | `kv_transfer_state.py:L72-L94` | 逐字；_sync_engine_id 广播体删（HOST SEAM 恒等） | 第 4 条 → ch36 |
| `kv_transfer.py 三谓词/__post_init__` | `config/kv_transfer.py:L92-L121` | 逐字（校验+谓词） | m16 must_keep |
| `mixin _get_kv_connector_output` | `mixin:L76-L112` | 逐字（bind→start→yield→finally 收尾）；stats/events 两行删 | m7 must_keep / 第 3 条 |
| `mixin no_forward/maybe_get/finalize` | `mixin:L36-L72` | 逐字 | must_keep |
| `kv_transfer_utils.py` | `kv_transfer_utils.py:L15-L61` | **全保逐字**（运行时延迟导入折为模块级相对导入——同一函数） | m8 must_keep |
| `example_connector.py inject/extract` | `example_connector.py:L122-L149/L221-L235` | MLA 分支删，非 MLA 主路径逐字 | 第 8 条 / m14 must_keep |
| `example_connector.py 调度器侧两原语` | `example_connector.py:L251-L298` | 逐字 | m2/F must_keep |
| `example_connector.py build_connector_meta` | `example_connector.py:L300-L374` | resumed-from-preemption 支（L341-L370）删 | 第 8 条可省段 |
| `gpu_model_runner execute_model 三挂点` | `gpu_model_runner.py:L4197-L4200/L4231-L4234/L4420-L4456` | EPLB 段删（dossier elide）；执行主体 ENGINE SEAM 抽出 | 第 11 条 |
| `scheduler.py 构造四旗标` | `scheduler.py:L125-L158` | 逐字 | 站 1 |
| `scheduler.py bind_gpu_block_pool` | `scheduler.py:L289-L294` | 逐字 | 站 1 尾 |
| `scheduler.py waiting 双查+仲裁` | `scheduler.py:L744-L832` | **逐字**（connector 分支+None→skipped+truncate+保尾+混合回退） | 站 3/4 —— m2/m3 核心 |
| `scheduler.py 护轨分配段` | `scheduler.py:L934-L985` | mamba 切分调用/encoder 分支/长 prefill/chunked 段删；reserved_blocks+allocate_slots(ext,delay) 逐字 | 站 5 —— m4/m5 |
| `scheduler.py update_state_after_alloc+stats` | `scheduler.py:L996-L1014` | 逐字 | 站 5 尾 / m17 |
| `scheduler.py WAITING_FOR_REMOTE_KVS 段` | `scheduler.py:L1023-L1053` | 逐字（先行记账+_skip_zero 登记） | 站 6 —— m4 |
| `scheduler.py producer partial-tail` | `scheduler.py:L1165-L1179` | 逐字；CoW 打包段（L1181-L1190）删 | 站 12③ / 第 10 条 |
| `scheduler.py build_connector_meta 过线` | `scheduler.py:L1231-L1258` | 逐字（EC 段删） | 站 7 —— m6 |
| `scheduler.py update_from_output` | `scheduler.py:L1670-L1976` | 栅栏推进/invalid 消化/主循环骨架（在途排水/失败跳过/stale 丢弃/停判/_free_request）+error 收尾+xfer 消化逐字；观测/输出装配删 | 站 8/9/10 侧 |
| `scheduler.py _preempt_request` | `scheduler.py:L1274-L1315` | drop_stale_output 记账逐字；spec/encoder/事件删 | m13 |
| `scheduler.py finish_requests` | `scheduler.py:L2237-L2298` | 逐字（WAITING_FOR_REMOTE_KVS 延迟释放分支含） | m11 邻接 |
| `scheduler.py _free_request/_free_blocks` | `scheduler.py:L2300-L2332` | EC 镜像钩子删，主干逐字 | m11 must_keep |
| `scheduler.py _free_request_blocks/_drain` | `scheduler.py:L2341-L2380` | 逐字；_free_cow_retained_blocks 删 | m12 must_keep / 第 10 条 |
| `scheduler.py has_finished_requests/has_requests` | `scheduler.py:L2394-L2421` | 逐字（EC 支删） | m11/m17 |
| `scheduler.py _connector_finished` | `scheduler.py:L2577-L2612` | 逐字（窗外回收+整块表+HMA/单组双路） | m11 must_keep |
| `scheduler.py _request_remaining_blocks/_inflight_prefill_reserved_blocks` | `scheduler.py:L2614-L2633` | 逐字 | m5 must_keep |
| `scheduler.py _update_waiting_for_remote_kv` | `scheduler.py:L2635-L2676` | 逐字（成功补缓存+全命中退一/失败截断补缓存+补登记清零/无有效 free） | m9/m10 must_keep |
| `scheduler.py _try_promote` | `scheduler.py:L2678-L2712` | WAITING_FOR_REMOTE_KVS 支逐字；grammar/streaming 支删 | m9 must_keep |
| `scheduler.py _update_from_kv_xfer_finished` | `scheduler.py:L2714-L2741` | 逐字 | m9/m11 must_keep |
| `scheduler.py _update_requests_with_invalid_blocks` | `scheduler.py:L2743-L2844` | 逐字（第一个坏块截断+共享去重） | m10 must_keep |
| `scheduler.py _handle_invalid_blocks` | `scheduler.py:L2846-L2915` | 逐字（async/sync 分扫+fail/recompute 双策） | m10 must_keep |
| `kv_cache_manager get_computed_blocks` | `kv_cache_manager.py:L229-L295` | 'full' 事件段删（L266-L284） | 第 3 条 |
| `kv_cache_manager get_computed_blocks_for_connector` | `kv_cache_manager.py:L297-L342` | **逐字**（混合感知命中） | m3 must_keep |
| `kv_cache_manager allocate_slots` | `kv_cache_manager.py:L344-L565` | **五段布局注释图逐字**；ext 计账/挂块条件/delay 早退逐字（ch14 删掉的 external 半边本章全通电） | m4/m5 must_keep |
| `kv_cache_manager truncate/zeroing/partial-tail` | `kv_cache_manager.py:L777-L794/L796-L829/L848-L874` | 逐字 | m3/m10/m15 must_keep |
| `kv_cache_manager pop_blocks_for_free/free` | `kv_cache_manager.py:L567-L617` | 逐字（pins 前置随行） | m12/m15 |
| `single_type allocate_external_computed_blocks` | `single_type:L291-L328` | 逐字（两阶段第二相的实施端） | m4 核心 |
| `coordinator allocate_new_computed_blocks` | `coordinator:L192-L236` | 逐字（外部第二相 L230-L236 复电） | #33775 两阶段 |
| `coordinator find_longest_cache_hit_per_group` | `coordinator:L819-L848` | 逐字（eagle 位实参 False） | m3 must_keep |

## HOST SEAM / SEAM 清单（不 import vllm 的等价复现点）

1. **LOGGER SEAM**：`vllm.logger.init_logger` → stdlib `logging.getLogger`
   （实验性警告/完成回传 debug 账目同构）。
2. **分布式 SEAM**：`_sync_engine_id_across_tp` 的 `get_tp_group().broadcast_
   object` → 单进程恒等（TP=1 时 rank0 值即全组值；多机对齐 → ch36）。
3. **forward_context SEAM**：create_forward_context 的 compilation_config/DP
   装配链 → 最小直构（attn_metadata/slot_mapping/no_compile_layers 三件）；
   set_forward_context 的 DP 协调删。
4. **attention SEAM**：get_attention_context 的 DBO 双微批 list 形态删（spec
   decode → ch33）；层实例取 `no_compile_layers[layer_name].kv_cache` 逐字。
5. **torch_utils SEAM**：LayerName opaque 包装删（str 直通——语义等价）。
6. **safetensors SEAM**：显式 `import safetensors.torch` 绑定子模块（真实
   环境由包惰性属性提供；host 版需显式）。
7. **factory 包重定位**：注册行的 module_path 由
   `vllm.distributed.kv_transfer...` 改为 `implementation.example_connector`
   （本精简版的包结构；懒加载机制本体逐字）。
8. **ENGINE SEAM**：scheduler 的 waiting 循环 connector 段保留在 schedule()
   原位（真实控制流位置）；gpu_model_runner 的执行主体以
   _forward_with_connector/_model_forward 抽出为方法（控制流=挂点调用序，
   ch17-20 主体删）。

## 与 ch13/14/15 精简版的关系（读者跑四本对照）

- **ch13**（False 支+槽位恒等式）：本章 slot 寻址（m14 的 block_id×bs+offset）
  消费 ch13 的块表间接寻址。
- **ch14**（账本/准入门/水位）：allocate_slots 的 full-ISL 门与 watermark 面
  逐字复用；本章在其上通电 **external 半边**（ch14 曾删：ext 计账/挂块条件/
  delay 早退/reserved 门）。
- **ch15**（链式哈希/CoW/LRU）：本地命中链路（哈希链/两相 find/惰性驱逐）
  整栈复用同一切面；本章第 4 站消费 `truncate_computed_blocks` 与子块尾语义
  ——ch15 第 9 站留下的『→ ch16』路标在此接上。CoW 打包段按第 10 条删
  （本章 m3 仲裁正是『免 CoW』）；与 CoW 成对的 +1 预留/换尾重定向在
  manager 侧对称删除，分配算术自洽。
- **impl≠pin 边界（仅 mamba-align 组的分配内部）**：MambaManager 只保 find
  的 fine 分支（align 分配内部/状态块滚动按 ch15 同款删法走基类回退）——
  align 配置下 mamba 组逐块登记而非『NULL×n+唯一状态块』。与 ch15
  impl-notes 已记录的同名边界一致：m3/m15 的场景在 partial-hit 粒度配置
  驱动时，full 组行为与钉版逐字节一致（本版 m3 两测试实测）；写作时勿把
  mamba 组的逐块命中讲成真实行为。

## 验收

- `python -m pytest tests -q`：**60 passed**（0 failed/skipped）
- `python scripts/lint_fidelity.py <本章>`：见收尾自检输出（要求 0 BLOCKING）
- must_keep 60+ 符号全在（lint over_subtraction 零命中）
