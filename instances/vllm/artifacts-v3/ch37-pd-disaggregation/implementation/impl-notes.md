# ch37《P/D 分离》impl-notes —— 只做减法精简版

对应真实源码 pin **vLLM v0.27.1（6e448d0ea）**，行号全部现核（2026-09-17，
`instances/vllm/source`）。**不是** v2 资产的 v0.21.0 旧行号（v2 的 NIXL 连接器
是 v0.21 老版，缺租约心跳 / 推模式 / 拒绝回执整条链）。

运行：`cd instances/vllm/artifacts-v3/ch37-pd-disaggregation && python -m pytest tests/ -q`
→ **36 passed**（全 host：单进程装配「两台引擎」，带外握手走**真 ZMQ + 真 msgspec**；
无 vllm 包安装、无 CUDA、无 RDMA——NIXL 数据面以进程内 seam 承载）。

**验收判据**：把真实源码删掉所有 `# SUBTRACTED:` 分支 ≈ 得到本包
（SEAM 例外见 §Seam 清单——每处行内标注真实源锚与偏离声明）。

**lint**：`python3 scripts/lint_fidelity.py <本章目录>` → **全部通过（无 BLOCKING）**；
must_keep 89 符号经 linter `over_subtraction` 项全数核在。锚点另经自建 AST 校验器
对 pin 现核（每个 def/class 的 `# SOURCE:` 行号与真实文件 AST 行号对表）。

**锚点双置惯例**（ch22/ch25/ch35 同款）：每个 def/class 的 `# SOURCE:` 锚点在
声明上方（供读者）与 def/class 体内复置一行——后者是 `lint_fidelity.
_spans_missing_source` 的跨度判据（span=def 行上一行到函数尾）。

## 包结构（与真实树同名同构）

| 精简版文件 | 真实文件 | 本章切面 |
|---|---|---|
| `vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py` | 同名 | **主文件 1**（m1/站 1）：NixlBaseConnector facade（按 role 只建半边）+ NixlPull/PushConnector + `NixlConnector` 别名 |
| `…/nixl/base_scheduler.py` | 同名 | **主文件 2**（m8/站 2）：side channel 配置、四账本、`kv_lease_duration`/心跳间隔、SWA 尾剪裁、`build_connector_meta`（时钟基准）、ZMQ ROUTER 监听线程、握手元数据注入 |
| `…/nixl/pull_scheduler.py` | 同名 | **主文件 3**（m3/m4/m10）：D 查命中（全 prompt 异步拉 + `kv_recompute_threshold` 骨架）、D 登记未哈希块（一次性翻转）、P 终局（双条件 + delay_free + 租约 + 回执字典）、拒绝回执的空 recv 分支 |
| `…/nixl/push_scheduler.py` | 同名 | **主文件 4**（m9）：D 侧 PUSH_REG 注册暂存 + watchdog、P 侧完成块暂存、`has_pending_push_work`、`build_connector_meta` 打包两表 |
| `…/nixl/base_worker.py` | 同名 | **主文件 5**（m5/m7/m8/站 7/9/10）：NIXL 区域登记与描述符几何、带外握手（compat hash 门 + RTT 时钟偏移）、`_ensure_handshake` 单飞、`get_finished` 双源 + 租约到期收割、心跳双端、失败处理、TTL 驱逐、shutdown |
| `…/nixl/pull_worker.py` | 同名 | **主文件 6**（m6/站 8）：`start_load_kv`（握手或直发 READ + 租约重基准 + 心跳）、`_read_blocks`（READ 本体 + 全命中 notif-only）、`_get_new_notifs`（消费者计数齐才放块） |
| `…/nixl/push_worker.py` | 同名 | **主文件 7**（m9）：`nixl-push-writer` 线程主循环、配对机器（随机后缀剥离）、WRITE 本体、D 侧完成物化、`get_finished` 的记账 |
| `…/nixl/metadata.py` | 同名 | 协议面（m5）：`NixlHandshakePayload`（两段解码）、`NixlAgentMetadata`、`NIXL_CONNECTOR_VERSION=5` 版本史、`NixlConnectorMetadata` 四账本 + 推模式两表 |
| `…/nixl/tp_mapping.py` | 同名 | 对称 TP 的映射（每 rank 对一个远端 rank）+ `ReadSpec` |
| `…/nixl/utils.py` | 同名 | `zmq_ctx`、`get_base_request_id`（随机后缀归一）、`get_representative_spec_type` |
| `vllm/distributed/kv_transfer/kv_connector/utils.py` | 同名 | m7：`KVOutputAggregator`（TP 各 worker 计数归零才上报）+ `TransferTopology`（tp_ratio/block_size_ratio/握手目标 rank）+ `EngineTransferInfo` |
| `vllm/distributed/kv_transfer/kv_connector/factory.py` | 同名 | m1：注册表 + 三条 NIXL 注册（NixlConnector / NixlPullConnector / NixlPushConnector） |
| `…/kv_connector/v1/base.py` | 同名 | ch16 立的双面契约（调度器五原语 / worker 收发族 / `SupportsHMA` / role 枚举） |
| `vllm/config/kv_transfer.py` | 同名 | m1：`kv_role` 三态部署门、`engine_id`（默认 uuid4）、extra config 取值 |
| `vllm/v1/request.py` | 同名 | m2：`kv_transfer_params` 从 `extra_args` 摘出挂上请求；`RequestStatus`（含 `WAITING_FOR_REMOTE_KVS`） |
| `vllm/v1/core/sched/scheduler.py` | 同名 | m3/m7：四处 connector 挂钩（装配/`_connector_finished`/`update_connector_output`/`_update_from_kv_xfer_finished`）+ `has_requests` 判活 |
| `vllm/v1/engine/core.py` | 同名 | 站 2 + m10：worker 握手元数据聚合注入 + `abort_immediately` 立即 abort |
| `vllm/v1/engine/async_llm.py` | 同名 | m10：`notify_kv_transfer_request_rejected`（占位 EngineCoreRequest） |
| `vllm/entrypoints/generate/base/serving.py` | 同名 | m10：`_with_kv_transfer_rejection_cleanup`（拒收检测包装） |
| `vllm/v1/executor/abstract.py` | 同名 | m7：`init_kv_output_aggregator` 装配点 |
| `tests/v1/kv_connector/nixl_integration/toy_proxy_server.py` | 同名 | m2：disaggregator 参考实现（发单 P / 转交 D / round-robin） |
| `vllm/v1/{outputs,kv_cache_interface,core/kv_cache_manager,core/sched/output,attention/backend,worker/utils}.py`、`vllm/{logger,envs,sampling_params}.py`、`vllm/utils/{math_utils,network_utils,torch_utils,hashing}.py`、`vllm/platforms/__init__.py`、`vllm/distributed/parallel_state.py` | 各自真实路径 | 载体/接缝（见 §Seam 清单） |

## SEAM 清单（真源码锚 + 偏离声明，全部行内标注）

| seam | 真实源锚 | 偏离 |
|---|---|---|
| **`NixlWrapper`（本文件 `vllm/distributed/nixl_utils.py`）** | `vllm/distributed/nixl_utils.py:L13-L18 + L57-L76`（惰性导入外部 `nixl` 包的 `nixl_agent`） | **本章唯一的数据面替换**：真 RDMA 单边 READ/WRITE → 进程内 memcpy，但**保留了单边语义**（只按描述符里的裸地址取数，不经对端 Python 代码路径）、描述符按块号索引、handle 轮询（`DONE`）、notif 随传输完成送达对端。偏离面：真传输是异步的、`check_xfer_state` 会先返回若干次 `PROC`；替身当场搬完，故 PROC 分支在本章多步轮询里基本观察不到（代码路径逐字保留）。 |
| `HostAttentionBackend`（`vllm/v1/attention/backend.py`） | `vllm/v1/attention/selector.py:L101-L160`（从平台插件挑 FlashAttention/FlashInfer/Triton） | 宿主无加速器：替身后端不产生任何 attention 计算，但**名字（`FLASH_ATTN`）与 HND KV 形状**与真后端一致——connector 只读这两样（`attn_backend_name` 进兼容 hash、KV 形状进区域几何）。`get_supported_kernel_block_sizes` 返回 `[MultipleOf(1)]`：真 FlashAttention 只收 16 的倍数（会触发 `_physical_blocks_per_logical_kv_block>1` 的重映射路径），宿主恒 ratio=1 = 对称正典路径。 |
| `_HostPlatform`（`vllm/platforms/__init__.py`） | `vllm/platforms/interface.py` Platform 谓词族 | 宿主无加速器：`device_type="cpu"`、`get_nixl_memory_type()=None`（走 DRAM 分支）、`discover_numa_topology()=[]`（跳过 CPU 绑核）。 |
| `_StubBlockPool`（`vllm/v1/core/sched/scheduler.py`） | `vllm/v1/core/kv_cache_manager.py:L117-L900` | 真 scheduler 持一个 KVCacheManager 产出块表；本章的块是测试注入的（P 的块号靠回执搬运而非本地分配），故只留「按请求登记块表 / 归还」两个面。 |
| `Engine` / `StubModelExecutor`（测试脚手架 `tests/_pd_harness.py`） | `vllm/v1/engine/core.py` EngineCore + `vllm/v1/executor/*` | 真实部署里 P/D 是两台独立进程、scheduler 与 worker 各自成进程；脚手架把「一台引擎」装配进同一进程，但**只经 `KVConnectorMetadata` 交换**（与真引擎一致），带外握手仍走真 ZMQ 套接字。`StubModelExecutor` 提供 `get_kv_connector_handshake_metadata` 与 `init_kv_output_aggregator` 两个真接口。 |
| `envs` / `logger` / `parallel_state` / `config` 载体 | `vllm/envs.py`、`vllm/logger.py`、`vllm/distributed/parallel_state.py`、`vllm/config/*.py` | 接口位 stand-in：字段集收敛到本章消费面，默认值对 pin（如 side channel 默认 `localhost:5600`、租约默认 30s、`kv_recompute_threshold` 默认 64）。 |

## 1:1 Source Map（核心行；改动 = 减法或 seam，原因 = 批准条/章节边界）

| 精简版 | 真实源码 | 改动 | 原因 |
|---|---|---|---|
| `nixl/connector.py:__init__` | `nixl/connector.py:L112-L135` | 逐字 | — |
| `nixl/base_scheduler.py:__init__` 的租约段 | `base_scheduler.py:L64-L76` | 逐字 | `kv_lease_duration`=30、心跳=租约/6 |
| `nixl/base_scheduler.py:build_connector_meta` | `base_scheduler.py:L437-L476` | 逐字 | 时钟基准 + 四账本清空 + 心跳节流 |
| `nixl/base_scheduler.py:_nixl_handshake_listener` | `base_scheduler.py:L324-L365` | 删 PP 维（loop 仍按目标 rank 查询） | 删除项 8 |
| `nixl/base_scheduler.py:get_exchange_clipped_blocks` | `base_scheduler.py:L235-L279` | 删 SSM 分支（保 SWA 尾剪裁） | 删除项 2 |
| `nixl/base_scheduler.py:_get_remote_prefill_token_count` | `base_scheduler.py:L367-L398` | 删 Mamba N−1 | 删除项 2 |
| `nixl/pull_scheduler.py:get_num_new_matched_tokens` | `pull_scheduler.py:L34-L110` | 删 Mamba 截尾臂；**保 do_remote_decode 骨架**（含 `kv_recompute_threshold`） | 删除项 2；骨架为减法计划明示保留项 |
| `nixl/pull_scheduler.py:update_state_after_alloc` | `pull_scheduler.py:L112-L179` | 删双向回拉臂 | 删除项 5 |
| `nixl/pull_scheduler.py:request_finished` | `pull_scheduler.py:L181-L280` | 删 D 侧 TTL 切换与 expiry 生产；**保空 recv 分支**（拒绝回执）与 v5 字典键 | 删除项 5；m10 必须留 |
| `nixl/push_scheduler.py:*` | `push_scheduler.py:L54-L356` | 删双向回拉守卫 + Mamba 截尾；watchdog 默认值改显式常量 480 | 删除项 5/2 |
| `nixl/base_worker.py:_compute_desc_ids` | `base_worker.py:L93-L161` | 删 SSM 描述符族与块尺寸扩张；保 all-attention 快路径 | 删除项 2/1 |
| `nixl/base_worker.py:_nixl_handshake` | `base_worker.py:L577-L724` | 删 PP product / 保 compat hash 门与 RTT 时钟偏移 | 删除项 8 |
| `nixl/base_worker.py:register_kv_caches` | `base_worker.py:L1037-L1277` | 删 packed 探测 / Mamba 页长 / PP 区域偏移 / MLA 区域识别 | 删除项 4/2/8/1 |
| `nixl/base_worker.py:add_remote_agent` | `base_worker.py:L1490-L1696` | 删 split handles / PP 窗口切片 / Mamba 远端区域 | 删除项 1/8/2 |
| `nixl/base_worker.py:_validate_remote_agent_handshake` | `base_worker.py:L1698-L1867` | 删异构断言链，保「层数/块长/块数逐项相等」主断言 | 删除项 1 |
| `nixl/base_worker.py:get_finished` | `base_worker.py:L2044-L2164` | 删接收后处理与 Mamba 同步；**保租约到期收割** | 删除项 0/2 |
| `nixl/base_worker.py:_apply_prefix_caching` | `base_worker.py:L2398-L2486` | 删 Mamba front-trim；**保非 Mamba 尾裁剪**（must_keep） | 删除项 2 |
| `nixl/pull_worker.py:start_load_kv` | `pull_worker.py:L42-L113` | 逐字（含租约重基准与心跳） | — |
| `nixl/pull_worker.py:_read_blocks` | `pull_worker.py:L225-L344` | 删块尺寸映射；保全命中 notif-only 与 `notif_id=req:tp_size` | 删除项 0 |
| `nixl/push_worker.py:_push_writer_loop` | `push_worker.py:L210-L272` | 逐字（四收件箱 + notif 路由 + 自轮询条件） | — |
| `nixl/push_worker.py:_pop_matching_*` | `push_worker.py:L369-L399` | 逐字（精确键 + `get_base_request_id` 剥后缀） | — |
| `nixl/push_worker.py:_xfer_blocks` | `push_worker.py:L593-L686` | 删块尺寸映射；保 WRITE + 句柄回收语义 | 删除项 0 |
| `kv_connector/utils.py:KVOutputAggregator` | `kv_connector/utils.py:L53-L173` | 删 stats/worker_meta/events 聚合 | 删除项 6 |
| `v1/core/sched/scheduler.py:_update_from_kv_xfer_finished` | `scheduler.py:L2714-L2741` | 逐字 | m7 的调度器侧落点 |
| `v1/core/sched/scheduler.py:has_requests` | `scheduler.py:L2406-L2420` | 逐字（含 `has_pending_push_work`） | 推模式保活 |
| `v1/engine/core.py:__init__` 装配段 | `core.py:L172-L200` | 逐字 | 聚合器 + 握手元数据注入 |
| `v1/engine/core.py:add_request` | `core.py:L438-L483` | 删池化白名单/EC 告警 | 本章不涉 |
| `v1/engine/async_llm.py:notify_kv_transfer_request_rejected` | `async_llm.py:L743-L769` | 逐字 | m10 |
| `entrypoints/…/serving.py:_with_kv_transfer_rejection_cleanup` | `serving.py:L218-L252` | 逐字 | m10 |
| `nixl/metadata.py:compute_nixl_compatibility_hash` | `metadata.py:L81-L141` | 逐字（因子集与日志） | m5 |
