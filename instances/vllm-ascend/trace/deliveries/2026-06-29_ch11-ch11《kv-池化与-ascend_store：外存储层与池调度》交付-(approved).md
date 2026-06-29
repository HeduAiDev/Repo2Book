# ch11《KV 池化与 ascend_store：外存储层与池调度》交付 (APPROVED)

- **Type**: delivery
- **Chapter**: 11
- **Date**: 2026-06-29
- **Timestamp**: 2026-06-29T13:46:59Z
- **Agents involved**: analyst, implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: kv-pooling, ascend-store, pool-scheduler, pool-worker, mooncake-backend, kv-connector, async-offload

## What happened

四线讲透昇腾 KV 池化(经外存储层中转-复用，区别于 ch10 PD 分离的节点间 P2P 直传)：(1) 入口 AscendStoreConnector(子类化 vLLM KVConnectorBase_V1+SupportsHMA) 薄分发层，按 kv_role 把调度侧钩子转 KVPoolScheduler、worker 侧转 KVPoolWorker，rank0 起 LookupKeyServer；(2) 两端协作 = 调度器决定搬什么(KVPoolScheduler.get_num_new_matched_tokens 经 LookupKeyClient zmq REQ/REP 问池命中→build_connector_meta 打包 AscendConnectorMetadata 下发) + worker 异步搬(KVPoolWorker register_kv_caches 注册显存 base_addr/stride 并起收/发后台线程 KVTransferThread，request_queue 单消费者循环把生产者(主循环 add_request)与消费者(后台线程)解耦)，背压靠 wait_for_save 的 join 屏障(异步 put 全 task_done 才解除)；(3) 数据通路 kv_transfer.py 收/发线程：发端 process_tokens 生成内容寻址 PoolKey→lookup 去重(跨请求只存一次)→仅 missing 块 prepare_value 取(addr,size)→m_store.put；收端组 key/addr/size→m_store.get→失败块 record_failed_blocks→set_finished_request；(4) 可插拔后端 Backend(ABC) 6 抽象方法契约，MooncakeBackend 讲透(put/get 经 batch_*_into_multi_buffers，异常 try/except 吞为记日志)，cpu_offload/lmcache/ucm 点名。Reviewer APPROVED，15 issues 全 negotiable/non-blocking(2 处 fidelity 小修：get/put 的 try 包裹呈现不一致、torch.npu.Event 长句拆短；1 处 algorithm 精度打磨：join 终止性证明把异常逃逸后果点破；多处 reader-comprehension 名词补释 kv_role/use_layerwise/zmq/npu.Event/block_stride 等；formulas 5 处启发式警告非阻断退出码 0)。

## Why it matters

KV 池化是昇腾跨请求-跨实例复用 KV、省重算的关键路径，与 ch10 PD 分离(P2P 直传)拓扑-节拍不同(写进池、别的请求-实例再捞)。本章回收 ch10 留的 store-pool 存取与池调度节拍前向引用(arc-map f8 resolved)，并姊妹对照基座 vLLM v0.21.0 的 KVConnectorBase_V1 抽象与 offloading_connector，确立「两端解耦+join 背压+内容寻址+可插拔后端契约」这套池化范式。

## What to remember

ch11 APPROVED 交付。已注册 18 个精简版接口入 bible(连接器/调度器/worker/搬运线程/后端契约)。ch10→ch11 前向引用(store-pool 存取+池调度节拍)经 arc-map f8 resolved 回收。无本章新埋伏笔。review-report.json 15 issues 全 non-blocking。下一章接 ch12。
