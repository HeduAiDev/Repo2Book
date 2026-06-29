# ch10《PD 分离：连接器分发、mooncake P2P 传输与 KV 亲和调度》交付 (APPROVED)

- **Type**: delivery
- **Chapter**: 10
- **Date**: 2026-06-29
- **Timestamp**: 2026-06-29T12:13:40Z
- **Agents involved**: analyst, implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: pd-disaggregation, mooncake, kv-connector, affinity-routing, proxy

## What happened

三层全貌讲透：(1) 连接器分发层 AscendMultiConnector(MultiConnector, SupportsHMA) fan-out 路由到三个 mooncake 连接器，挑 MooncakeLayerwiseConnector 讲透 facade(role 二选一 SCHEDULER/WORKER)+metadata+逐层 fire-and-forget save/load 回调如何嵌进 vLLM 调度循环；(2) P2P 传输层 GlobalTE 进程级单例握 mooncake TransferEngine(P2PHANDSHAKE/ascend 后端 RDMA 直传)，group_concurrent_contiguous 要求 src/dst 双连续以减 P2P 操作数；(3) proxy/router 层 SharedProxyScheduler 堆+heap_seq 惰性删除在 P/D 实例间分发。高潮小节 KV 亲和(cache-hit-aware)调度：KVPoolScheduler.get_num_new_matched_tokens 经 LookupKeyClient.lookup(zmq REQ/REP) 查命中→亲和路由最小化跨节点传输，三轮数值追踪走清(命中查询；store/池调度留 ch11 前向引用)。Reviewer APPROVED，全部 issue 均 negotiable/non-blocking(1 处量化归因臆测中性化 + 多处 reader-comprehension 名词补释 + 呈现顺序微调)。

## Why it matters

PD 分离是昇腾跨实例省传输的关键路径；本章打通「子类化 vLLM KVConnectorBase_V1/MultiConnector 加了什么(HMA/亲和)」与亲和路由的命中→路由数值追踪，为 ch11 KV 池 store/池调度节拍铺垫。

## What to remember

ch10 APPROVED 交付。已注册 13 个精简版接口入 bible(三层+亲和)。无 ch10 应埋/应回收伏笔(arc-map 确认)。review-report.json 28 issues 全 non-blocking。下一章 ch11 接 KV 池 store/池调度(本章只引命中查询，前向引用未展开)。
