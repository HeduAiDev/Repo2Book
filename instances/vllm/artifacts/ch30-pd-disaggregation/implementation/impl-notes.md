# ch30 精简版实现笔记（subtract-only）

源码 pin `f3fef123`。精简版与 vLLM **同名/同结构/同控制流，只删不增**。
不 `import vllm`、不触 CUDA，可在 host 直接跑（`python3 -m pytest tests/`，21 passed）。
对外部依赖（forward context / 全局 connector 注册 / 回传载体 / 底层引擎与 nixl_wrapper /
offloading worker）用形状一致的 loopback stub 顶替，剥离网络/RDMA/磁盘细节但保留接口
与控制流，使三类后端能真跑出『发→收→完成』闭环。

## 1:1 Source Map

| 精简版符号 | 真实 vllm 源码 | 改动 | 原因 |
|---|---|---|---|
| `mixin.py::_get_kv_connector_output` | `vllm/v1/worker/kv_connector_model_runner_mixin.py:L81-L119` | 1:1 保留 bind→start_load_kv→yield→wait_for_save→get_finished→build_worker_meta→clear；删 stats/events 两行回填 | dossier delete：可观测性扩展点，基类默认 None |
| `mixin.py::maybe_get_kv_connector_output` | 同文件 L57-L68 | 原样 | has_kv_transfer_group 才进 context，无 connector → nullcontext |
| `mixin.py::kv_connector_no_forward` | 同文件 L37-L55 | 原样 | 无 token 也走收发，wait_for_save=False |
| `mixin.py::finalize_kv_connector` | 同文件 L70-L79 | 原样 | spec-decode defer：draft forward 后补 wait_for_save+clear |
| `mixin.py::maybe_transfer_kv_layer` | `vllm/model_executor/layers/attention/kv_transfer_utils.py:L15-L61` | 删 inspect.signature 取 layer_name 索引样板，用固定签名 | dossier delete：工程样板，与逐层 hook 语义无关 |
| (删) `use_uniform_kv_cache` / `allocate_uniform_kv_caches` | mixin L121-L283 | 整删 | dossier delete：跨层统一 layout，与生命周期主线正交 |
| `base.py::KVConnectorBase_V1` 五契约方法 | `vllm/distributed/kv_transfer/kv_connector/v1/base.py:L298-L399` + L217-L255 | 保留契约签名/docstring 要点 + bind/clear/has metadata | 本章三后端都填这套 worker 契约 |
| `p2p_connector.py::P2pNcclConnector.start_load_kv` | `vllm/.../p2p/p2p_nccl_connector.py:L111-L229` | 保留 consumer 逐请求逐层 recv_tensor+inject；inject 删 MLA 支留 FlashAttention 支 | dossier delete：layout 二选一 |
| `p2p_connector.py::...save_kv_layer` | 同文件 L242-L307 | 保留 producer extract+send_tensor；extract 删 MLA 支留 FlashAttention 支 | dossier delete：layout 二选一 |
| `p2p_connector.py::...wait_for_save` / `get_finished` | 同文件 L309-L331 | 原样转发引擎 | wait_for_sent 等队空；get_finished 委托引擎 |
| `p2p_connector.py::P2pNcclEngine.send_tensor`/`wait_for_sent`/`recv_tensor`/`get_finished` | `p2p_nccl_engine.py:L235-L258, L486-L498, L308-L335, L540` | 保留 PUT/PUT_ASYNC 两模式 + 后台线程 + 队空 fence；删 GET 模式 + LRU + ZMQ/pynccl，换 loopback 投递 | dossier delete：GET 第三策略与控制面握手 |
| `nixl_connector.py::NixlConnector.__init__` + 五转发方法 | `vllm/.../nixl/connector.py:L87-L264, L204-L212` | facade 按 role 建半边 + worker 契约转发；wait_for_layer_load/save_kv_layer no-op | NIXL 不逐层、不显式 save |
| `nixl_connector.py::NixlConnectorWorker.start_load_kv`/`_read_blocks`/`get_finished`/`_pop_done_transfers`/`_get_new_notifs` | `vllm/.../nixl/worker.py:L1840-L1895, L1980-L2109, L1651-L1730, L1777-L1822, L1732-L1775` | 保留握手→READ→轮询 DONE/通知主干；删异构 TP/MLA/Mamba/host buffer/block_size_ratio 分支 | dossier delete：对称 TP 之外的进阶路径 |
| `offloading_connector.py::OffloadingConnector` + 转发 | `vllm/.../offloading_connector.py:L46-L119` | facade + 五契约转发；删 handle_preemptions/register/take_events 等 | dossier delete：可选扩展点 |
| `offloading_connector.py::OffloadingConnectorWorker.start_kv_transfers`/`prepare_store_kv`/`get_finished`/`build_connector_worker_meta` | `vllm/.../offloading/worker.py:L295-L352` | 1:1 保留『store 推迟到下一步』『load 报 recving、sending 恒空』『completed_jobs 围栏』；删 stats 记账 | dossier delete：可观测性 |

## 关键不变式（测试覆盖）
- 生命周期时序：start_load_kv 在 forward 前、wait_for_save 在 forward 后（`test_lifecycle_order_brackets_forward`）。
- defer_finalize 推迟 wait_for_save（`test_defer_finalize_*`）。
- 逐层 hook：进层 wait_for_layer_load、出层 save_kv_layer（`test_maybe_transfer_kv_layer_*`）。
- P2P：producer save→consumer load 数值往返、PUT_ASYNC 后台发 + wait_for_sent fence（`test_producer_save_consumer_load_roundtrip`）。
- NIXL：首遇握手再 READ、handle DONE 才报收完成、对端通知报发完成（不对称）（`test_first_contact_*`, `test_get_finished_*`）。
- Offloading：store 推迟、load 报 recving、sending 恒空走 completed_jobs 围栏（`test_store_is_deferred_*`, `test_store_completion_*`）。
