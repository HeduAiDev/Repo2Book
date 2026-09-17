# ch38《KV 池化》impl-notes — 只做减法的精简版

- **pin**：vLLM v0.27.1（`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`，source 树 HEAD 现核）
- **验收判据**：把真实源码删掉所有 `# SUBTRACTED:` 标记的分支 ≈ 本精简版；每 def/class 带
  `# SOURCE: vllm/...:Lxxx`（行号对 v0.27.1 现核，全部锚点由 AST 从钉版源码映射）。
- **运行**：`cd instances/vllm/artifacts-v3/ch38-kv-pooling && python -m pytest tests/ -q`
  （host 纯控制流：69 passed；不 import 真 vllm、不需 CUDA/NIXL/Mooncake）。
- **lint**：`python scripts/lint_fidelity.py instances/vllm/artifacts-v3/ch38-kv-pooling` → 0 BLOCKING。

## 1. 文件总览（implementation/vllm/...，与目标仓同树同名）

| 精简版文件 | 真源锚 | 状态 |
|---|---|---|
| `distributed/kv_transfer/kv_connector/v1/offloading_connector.py` | L49-L229 | facade：best-effort 三件逐字；删 register_cross_layers（删除项 5）+ 遥测方法体（删除项 7） |
| `.../v1/offloading/scheduler.py` | L1-L1470 | 调度器半边：删 eagle/mamba 深分支（删除项 4）+ 全部指标记录点（删除项 7） |
| `.../v1/offloading/worker.py` | L1-L397 | worker 半边：删 packed/cross-layer 布局族（删除项 5）；`_is_store_writer` 保字段恒 True（删除项 1） |
| `.../v1/offloading/common.py` | L1-L104 | 过线三件逐字（TransferJob/meta/completed_jobs 聚合） |
| `.../v1/offloading/config.py` | L26-L182 | chunk 粒度解析逐字；删 replicated_layout 推导（删除项 1） |
| `.../v1/offloading/canonical_mapping.py` | L387-L444 | 入口保留、恒返 {}（uncertified 直通）；推导体全删（删除项 1） |
| `.../v1/offloading/events.py` | L1-L326 | 逐字保留（m15 全部面） |
| `.../v1/offloading/metrics.py` | 类位 | 契约签名 + 默认空返回（删除项 7） |
| `v1/kv_offload/base.py` | L1-L625 | 第二份双面契约逐字（OffloadKey/四态/六原语/OffloadingWorker/OffloadingSpec） |
| `v1/kv_offload/config.py` / `factory.py` | L1-L76 / L14-L66 | 逐字（spec 注册表两条） |
| `v1/kv_offload/cpu/spec.py` | L27-L186 | 定价公式逐字；删 rank=0 单拷贝分支（删除项 1） |
| `v1/kv_offload/cpu/manager.py` | L30-L327 | 池账本逐字；get_stats 体删（删除项 7） |
| `v1/kv_offload/cpu/gpu_worker.py` | L35-L552 | DMA 引擎：删平台择路/Triton/XPU uint64（删除项 2），三戒时序注释原样保留 |
| `v1/kv_offload/cpu/shared_offload_region.py` | L15-L212 | /dev/shm 共享池逐字 |
| `v1/kv_offload/cpu/policies/{base,lru,arc,factory}.py` | 各文件 | 驱逐策略逐字 |
| `v1/kv_offload/cpu/common.py` | L1-L17 | 逐字 |
| `v1/kv_offload/tiering/manager.py` | L1-L852 | 五原则逐字；删双指标计时（删除项 7） |
| `v1/kv_offload/tiering/{base,async_lookup,spec,factory}.py` | 各文件 | 副层契约/异步查询/分层 spec 逐字（spec 删指标定义）；factory 删 obj/example 注册行（删除项 3） |
| `v1/kv_offload/tiering/fs/{manager,io,thread_pool}.py` | 各文件 | fs 副层逐字（io.py 带 host SEAM，见 §3） |
| `v1/kv_offload/tiering/p2p/manager.py` | L1-L837 | 主干（三角色键/PYTHONHASHSEED 硬门/停靠账本）保留；session 机器/收割/排水删（删除项 9） |
| `v1/kv_offload/tiering/p2p/session/protocol.py` | L1-L341 | 线协议消息类逐字（m13） |
| `v1/kv_offload/tiering/p2p/{control,data}/base.py` | 各文件 | 传输 ABC 逐字（ZmqTransport/NixlTransport 实现体不进，删除项 9） |
| `v1/kv_offload/file_mapper.py` | L1-L134 | 三级分桶命名逐字 |
| `.../v1/mooncake/store/{connector,scheduler,worker,protocol,data}.py` | 各类位 | 生态骨架：查询路径（LookupKeyClient 逐字）+ 双 no-op 契约面；实现体删（删除项 6） |
| `.../v1/lmcache_connector.py` | L1-L354 | 薄壳逐字（双路 lazy import 本体） |
| `.../v1/multi_connector.py` | L1-L671 | 组合器逐字（首命中判据 L399-L404 是 m14 锚点）；删 set_host_xfer_buffer_ops |
| `distributed/kv_transfer/kv_connector/factory.py` | L27-L227 | 注册机制逐字；六条生态注册行保留、十条删（删除项 6） |
| 基建 stub（logger/envs/platforms/_custom_ops/utils/config/v1/...） | 各消费面 | 只含本章消费的最小面，ch37 同款删法 |

## 2. 1:1 Source Map（改动点明细，摘承重项）

| 精简版 | 真源 | 改动 | 原因（减法计划项） |
|---|---|---|---|
| scheduler.py `storable_chunks` | L338-L364 | 删 eagle decode 期尾块剔除（`num_chunks-1` 分支） | 删除项 4 |
| scheduler.py `_lookup` | L631-L802 | 删 `eagle_verified`/`query_max` 扩张/尾块弹出/`_mamba_align_size` 对齐；收敛循环与 SWA 尾扫逐字保留 | 删除项 4 |
| scheduler.py `SchedulerOffloadConfig.from_spec` | L169-L258 | 删 `full_attn_tokens_per_chunk` 对齐段推导与 `eagle_groups` 推导（字段位保留恒默认值） | 删除项 4 |
| scheduler.py `is_store_reachable_swa_chunk` | L122-L139 | 只剩 `alignment_chunk_count is None → True` 快路（SWA 窗口语义以函数骨架保留） | 删除项 4 |
| scheduler.py `get_num_new_matched_tokens` | L816-L871 | 删 LOOKUP_SYNC/ASYNC 双指标计时与 `deferred_lookup_start_time` 簿记；None『稍后再问』语义逐字 | 删除项 7 |
| scheduler.py `update_connector_output` | L1280-L1358 | 删 transfer_stats→四对指标翻译链；completed_jobs 聚满结算逐字 | 删除项 7 |
| worker.py `register_kv_caches` | L69-L243 | 删 `UniformTypeKVCacheSpecs` 逐层分支、`layer_is_packed` 表、packed 单张量装配分支；通用逐层路径逐字 | 删除项 5 |
| worker.py `__init__` | L55-L57 | `_is_store_writer` 判定化简为 `True`（字段位与非 writer ack 分支保留） | 删除项 1 |
| cpu/spec.py `__init__`/`create_worker` | L90/L149-L173 | 删 `replicated_layout` 的 rank=0 单拷贝槽位；`num_copies=world_size` 主线逐字 | 删除项 1 |
| offloading/config.py `build_offloading_config` | L113-L136 | 删 MLA 复制页判定（`replicated_layout` 字位保留恒 False） | 删除项 1 |
| canonical_mapping.py | L31-L444 | 推导机器全删；`derive_canonical_mappings` 入口保留恒返 {}（worker 调用点不动 → `mapping=None` uncertified 直通） | 删除项 1 |
| gpu_worker.py 模块头 | L35-L58 | 删 `_select_swap_blocks_fn` 择路（固定 `ops.swap_blocks_batch`）与 XPU uint64 dtype 分支 | 删除项 2 |
| tiering/factory.py | L57-L79 | 删 `example`/`obj` 两条注册行（fs/p2p 两条示范保留） | 删除项 3 |
| mooncake/store/* | 各类 | worker 双传输线程/master 协调/token DB/LookupKeyServer 删；`LookupKeyClient`（ZMQ REQ + non_block=None）与三方法对照面保留 | 删除项 6 |
| p2p/manager.py | L188-L837 | 传输构造/`_get_or_create_session`/`_accept_new_peers`/双收割/排水删；三角色键解析、PYTHONHASHSEED 门、无对端退化分支逐字 | 删除项 9 |
| kv_connector/factory.py | L152-L242 | 十条注册行删（Example/LMCacheMP/NixlPull/Push/MoRIIO/DecodeBench/FlexKV/SimpleCPUOffload/HF3FS/ExampleHiddenStates） | 删除项 6 |
| facade `build_*`/`get_kv_connector_stats` | L204-L229 | 遥测体删、签名保留默认 None/空返回 | 删除项 7 |

## 3. SEAM 清单（host 替身；真部署不触）

| SEAM | 真源 | 替身 | 保留的可观察语义 |
|---|---|---|---|
| `vllm/platforms/__init__.py` Stream 三件套 | `Platform.__getattr__` 委托 `torch.cuda`（interface.py:L1131-L1152） | `HostStream`：wait_stream/wait_event no-op、`stream()` 空 ctx；`torch.Event` 在 host 即记即完 | 控制流与字节流不变；**DMA 三戒的时序**（wait_stream 计算流/同向保序/SRC_ACCESS_ORDER_ANY）host 不可观察，正文以 gpu_worker.py:L376-L389 注释原文为准 |
| `vllm/_custom_ops.py swap_blocks_batch` | C++ 扩展（csrc/cache_kernels.cu 批量拷贝 kernel） | ctypes.memmove 逐描述符搬运 | 描述符三缓冲（src 指针/dst 指针/size）语义逐条等价 |
| `vllm/v1/kv_offload/tiering/fs/io.py` ②处 | POSIX `os.readv`；`os.open` 无 O_BINARY（Linux 不需要） | `_host_readv`（os.read+拷贝）；Windows 强制 `|O_BINARY`（实测 CRT 文本模式会静默 `\n`→`\r\n` 加字节） | 读到什么、写到哪逐字节相同 |
| 测试 harness `HostOffloadingWorker`（tests/_kv_harness.py） | `CPUOffloadingWorker`+`SingleDirectionOffloadingHandler`（host 无 CUDA，`assert gpu_tensor.is_cuda` 不可越） | 同一套 `compute_sub_block_ptrs` 描述符数学同步搬字节 | 提交/围栏/完成全链路真身；只有 stream/event 异步时序不可观察 |

## 4. 减法偏离表（⚠️ 供 writer/Lead 裁量）

| 位置 | 减法计划原文 | 偏离 | 判定依据 |
|---|---|---|---|
| `tiering/fs/thread_pool.py` 交叉互捞 | 删除项 9：「fs 的线程池优先级互捞（thread_pool.py 交叉取队）」 | **保留原样（该文件零删减）** | 互捞是**唤醒传播承重结构**而非优先级微优化：两组线程共享一个 condition，enqueue 只 `notify(n_tasks)`——被唤醒的可能是另一优先级线程，若无『回退取对方队列』，任务滞留到下一次 notify 凑巧落到本组（实测：删互捞后 fs store 卡死、promotion 永不完成）。建议 analyst 复核该项措辞。 |

## 5. host 不可覆盖面（正文锚定源码直读，非测试缺口）

- **DMA 时序**（m6 三戒）：stream/event 交错须 CUDA（见 §3 第 1 行）；描述符数学与字节搬运已 host 全覆盖。
- **/dev/shm 真共享 mmap**（m2）：Windows 无 `/dev/shm`/unix mmap 参数；`SharedOffloadRegion` 源码逐字保留（layout 注释/creator 协议/MADV_POPULATE_WRITE/cleanup），host 测试走 spec 的 no-mmap 回退分支（源码自带）。tiering 测试以 fake region（numpy 池 + create_kv_memoryview）装配调度器侧 mmap 位。
- **Mooncake 双传输线程**（m14）：`get_finished` 的 I/O 下发须 MooncakeDistributedStore（外部包）；三方法契约面（双 no-op + docstring 原话）与查询通道（LookupKeyClient 走真 ZMQ）保留。
- **P2P session 机器**（m13）：ZMQ 控制面 + NIXL 数据面实现体不进（删除项 9）；协议消息类/三角色键/PYTHONHASHSEED 硬门/无对端退化分支 host 全覆盖。

## 6. 与 must_keep 的对账

dossier.subtraction_plan.must_keep 共 96 项符号，`lint_fidelity` over_subtraction 检查 0 缺失。
特殊保位：`alignment_chunk_count`/`is_eagle_group`（GroupOffloadConfig 字段位保留恒默认值——推导删但
`is_store_reachable_swa_chunk` 骨架的签名消费它们）；`derive_canonical_mappings`（入口保留恒空映射）；
`MooncakeStore{Connector,Scheduler,Worker}`（骨架 + 查询/对照面）；`LookupMsg/FetchMsg/TransferDoneMsg`
（protocol.py 逐字）。
