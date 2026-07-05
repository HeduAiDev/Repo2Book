# ch10 实现笔记 — PD 分离三层 + KV 亲和调度（subtract-only）

## 边界（dossier 明示）
昇腾 PD 代码 host 无 NPU/CANN/mooncake 不可整跑。精简版只验**可读控制流**：连接器分发选举 /
layerwise 角色分发 / 亲和路由决策 / proxy 最少负载分发 / 连续块合并——这些都是纯 Python，可跑可断言。
真实 mooncake P2P 跨节点 KV 搬运、NPU 重排/量化、worker 收发后台线程绑卡**不真跑**，由
`runtime_stub.py`（record-only 接缝，NOT subtract-only，但每个替身仍 `# SOURCE` 指向被顶替的真实符号）
接住，按 `subtraction_plan.delete` 删除。

## 文件
- `runtime_stub.py` — vllm / torch_npu / mooncake 运行期符号的 host 接缝（测试种子）。
- `multi_connector.py` ← `vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py` — 对照基座 vLLM `MultiConnector`（fan-out 选举 + save-to-all）。
- `ascend_multi_connector.py` ← `vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py` — `register_connector` 工厂覆写 + `AscendMultiConnector` 子类。
- `mooncake_layerwise_connector.py` ← `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py` — layerwise 连接器（facade / Scheduler / Worker / 地址算术 / 块合并）。
- `mooncake_transfer_engine.py` ← `vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py` — `GlobalTE` 进程级单例引擎。
- `pool_scheduler.py` ← `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py` — ★ KV 亲和：`KVPoolScheduler.get_num_new_matched_tokens` + `LookupKeyClient`。
- `load_balance_proxy_server_example.py` ← `examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py` — proxy 最少负载分发（`SharedProxyScheduler` + `assign_instances`）。

## 1:1 Source Map（精简版 ↔ 真实源码 ↔ 改动 ↔ 原因）

| 精简版符号 | 真实源码 :Lxxx | 改动 | 原因 |
|---|---|---|---|
| `register_connector` | `vllm_ascend/distributed/kv_transfer/__init__.py:L21` | 删 UCM/LMCache/SimpleCPUOffload 注册（L57-L81） | 同 register 套路，不在 PD/亲和路径 |
| `AscendMultiConnector.update_state_after_alloc` | `…/ascend_multi_connector.py:L32` | 逐字保留 | layerwise 永远拿真 blocks——昇腾对基座的关键分歧 |
| `AscendMultiConnector.request_finished_all_groups` | `…/ascend_multi_connector.py:L43` | 逐字保留 | HMA 逐组聚合，证明子类化 SupportsHMA |
| `MultiConnector.get_num_new_matched_tokens` | `vllm/…/v1/multi_connector.py:L358` | 逐字保留 | 「首个命中子连接器赢 load」选举 |
| `MultiConnector.update_state_after_alloc` | `vllm/…/v1/multi_connector.py:L379` | 逐字保留 | 对照：base 只给 chosen 真 blocks（无 layerwise 豁免） |
| `MultiConnector`（其余） | `vllm/…/v1/multi_connector.py:L128` | 删 stats/prom/worker-meta/工厂构造/事件流（L45-L125,L504-L629 等） | 与连接器分发主线无关 |
| `MooncakeLayerwiseConnector`（facade） | `…/mooncake_layerwise_connector.py:L690` | 逐字保留全部 role 转发 | vLLM v1 连接器 scheduler/worker 角色分发契约 |
| `…Scheduler.get_num_new_matched_tokens` | `…/mooncake_layerwise_connector.py:L827` | 删 debug log | do_remote_prefill → 拉整段 prompt（异步） |
| `…Scheduler.update_state_after_alloc` | `…/mooncake_layerwise_connector.py:L860` | 删 debug log | do_remote_prefill/do_remote_decode 分 recv/send + 握手 POST |
| `…Scheduler.build_connector_meta` | `…/mooncake_layerwise_connector.py:L931` | 删 debug log | 待 recv/send → ReqMeta，scheduler→worker 桥 |
| `…Worker.start_load_kv` | `…/mooncake_layerwise_connector.py:L1534` | 删 producer 块映射解析/重排准备（L1547-L1619） | NPU 拓扑算术；host 无 NPU |
| `…Worker.save_kv_layer` | `…/mooncake_layerwise_connector.py:L1621` | 删 reshape_event/量化/重排（L1634-L1733），保留默认分支 | 逐层 push 控制流不变；NPU 路径不可跑 |
| `…Worker.get_finished` | `…/mooncake_layerwise_connector.py:L1331` | 删失败块回收边角 | 异步闭环：回收 done-recv → 释放 blocks |
| `get_transfer_meta` | `…/mooncake_layerwise_connector.py:L285` | 删 Mamba/量化/pd_head_ratio>1 分支，保留 pd_head_ratio==1 默认 | base_addr+block_len 地址算术核心 |
| `group_concurrent_contiguous` | `…/mooncake_layerwise_connector.py:L1922` | 逐字保留 | local&remote 都连续才合批的块合并 |
| `GlobalTE.get_transfer_engine` | `…/utils/mooncake_transfer_engine.py:L11` | 替 `from mooncake.engine import TransferEngine` 为 record-only 替身 | host 无 mooncake；P2PHANDSHAKE/'ascend' 初始化语义保留 |
| `KVPoolScheduler.get_num_new_matched_tokens` | `…/kv_pool/ascend_store/pool_scheduler.py:L224` | 删 debug/info log | ★ lookup → need_to_allocate → LoadSpec 亲和路由 |
| `KVPoolScheduler.__init__` | `…/pool_scheduler.py:L39` | 删 hybrid/mamba/swa/block-size 推断与 store 状态（留 ch11） | ch10 只用命中查询 |
| `LookupKeyClient.lookup` | `…/pool_scheduler.py:L643` | 逐字保留 | (token_len,block_hashes,group_ids) → 命中 token 数的 zmq RPC |
| `SharedProxyScheduler`（核心） | `…/load_balance_proxy_server_example.py:L237` | 删 snapshot/log/实例增删/drain/taint（L321-L500 多处） | 部署弹性脚手架，非分发主线 |
| `SharedProxyScheduler._priority/_pick_server` | `…/load_balance_proxy_server_example.py:L276/L352` | 逐字保留 | prefill=tokens+0.3·kv，decode=tokens 的最少负载分发 |
| `build_prefill_request` | `…/load_balance_proxy_server_example.py:L790` | 逐字保留 | 给请求盖 P 角色章：do_remote_decode、max_tokens=1 |
| `send_request_to_service` | `…/load_balance_proxy_server_example.py:L809` | 逐字保留 | build → POST 链 |
| `assign_instances` | `…/load_balance_proxy_server_example.py:L896` | 删 try/except 选型回滚 + client 取用日志 | prefill→握手→decoder 端到端编排骨架 |

## 测试
`tests/test_pd_disaggregation.py`（host `python3 -m pytest`，20 passed）：
- 第 1 层：选举挑首个 advertiser / None 短路 / layerwise 永远拿真 blocks（对照 base 只给 chosen）/
  register_connector 覆写 / layerwise scheduler do_remote_prefill 拉整段。
- 第 2 层：`group_concurrent_contiguous` 连续合批 + remote 跳号分批 + 空输入。
- 第 3 层：prefill/decode 打分公式 / build_prefill_request 盖章 / `SharedProxyScheduler` 最少负载轮转。
- ★ 亲和：need_to_allocate = 命中−已算 / 全命中砍 1 / 命中不超已算则不加载 / consumer 短路 /
  layerwise 关异步加载 / LoadSpec 字段。

## 前向引用（ch11）
`KVPoolScheduler` 的 store/save 侧与池调度节拍（sending_events / build_connector_meta save 分支 /
池 worker）已整体删除并标注，留 ch11——ch10 只借其 cache-hit lookup 做亲和路由。
