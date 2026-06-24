# ch29《PD 分离的抽象与调度器集成》交付 APPROVED

- **Type**: delivery
- **Chapter**: 29
- **Date**: 2026-06-24
- **Timestamp**: 2026-06-24T12:12:14Z
- **Agents involved**: analyst, implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch29, pd-disaggregation, kv-connector, scheduler, foreshadow-payoff, f12

## What happened

解读真实 vLLM v1 PD 分离：为何把 prefill(O(L^2 d) 算力密集)与 decode(O(Ld) 访存密集)拆到不同 engine。KVConnectorBase_V1 的 role-split 契约——KVConnectorRole.SCHEDULER 决策侧(get_num_new_matched_tokens 查远程命中/update_state_after_alloc/build_connector_meta) vs WORKER 搬运侧(start_load_kv/wait_for_save)。调度器集成：scheduler 调 connector 查远程命中→请求进 WAITING_FOR_REMOTE_KVS→隔离到 skipped_waiting 队列避队头阻塞→KV 到位 _try_promote_blocked_waiting_request 提升回 WAITING。KVConnectorFactory 懒加载注册。回收伏笔 f12(WAITING_FOR_REMOTE_KVS 完整远程 KV 加载与提升路径)。4/4 linter 过(formulas 4 处 inline 密度非阻断告警)、18/18 pytest 过、reviewer APPROVED(8 条 non-blocking+negotiable)。登记 5 类核心接口。

## Why it matters

Part VIII PD 分离首章；把 ch14 埋的 skipped_waiting/WAITING_FOR_REMOTE_KVS 阻塞态在调度器侧闭环；建立 KV Connector 契约词汇，为后续 connector 实现章铺垫。

## What to remember

解读真实 vLLM v1 PD 分离：为何把 prefill(O(L^2 d) 算力密集)与 decode(O(Ld) 访存密集)拆到不同 engine。KVConnectorBase_V1 的 role-split 契约——KVConnectorRole.SCHEDULER 决策侧(get_num_new_matched_tokens 查远程命中/update_state_after_alloc...
