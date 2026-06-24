# ch30《PD 分离的 worker 侧执行与可插拔传输后端》交付 APPROVED

- **Type**: delivery
- **Chapter**: 30
- **Date**: 2026-06-24
- **Timestamp**: 2026-06-24T12:58:26Z
- **Agents involved**: analyst
- **User present**: False
- **Tags**: PD分离

## What happened

Part VIII PD 分离第二章，承接 ch29 决策侧契约，落到 worker 侧执行。核心: KVConnectorModelRunnerMixin._get_kv_connector_output 用一个上下文管理器把 WORKER-role connector 整条生命周期夹在 model forward 两侧——bind_connector_metadata 绑定搬运计划 → start_load_kv 异步发起 load → forward 内逐层 wait_for_layer_load / save_kv_layer 交织 → wait_for_save 围栏收齐异步发出的 save → get_finished 上报，使 KV load 与 compute 重叠。同一套 KVConnectorBase_V1 worker 契约被三类传输各自填实: P2P NCCL 点对点 send/recv、NIXL 高性能 RDMA READ、Offloading CPU/磁盘 transfer_async 后台线程卸载——facade 模式把 scheduler/worker 两半拆进各自子对象。§30.3 给出重叠模型 T_overlap≈max(T_xfer, N·t_c) 的量化（32 层×1ms vs 20ms → 52ms vs 32ms）；§30.4 happens-before 正确性论证、§30.7 send_queue 单调递减→0 的终止性论证。4/4 linter 全过，21/21 pytest 过，reviewer APPROVED（7 条 issues 全 non-blocking+negotiable，集中在 §30.3 单符号 T_xfer 宜内联、max 近似可补残差说明、§30.7 fence/围栏 用词统一、可选补一组随步数值表）。无伏笔应埋/应回收。bible 登记 6 个 worker 侧精简版接口。

## Why it matters

本章是 ch29 决策侧契约的 worker 侧落地，完成 PD 分离的「决策—执行」闭环；确立 KVConnectorBase_V1 worker 契约（start_load_kv/wait_for_layer_load/save_kv_layer/wait_for_save/get_finished）为三类传输后端的统一抽象边界，后续任何新传输后端都按这套契约填实。围栏（wait_for_save）作为异步发-收齐不变量的命名锚点跨节贯穿。

## What to remember

ch30 worker 侧契约: mixin._get_kv_connector_output 上下文管理器编排整条生命周期（bind→start_load_kv 异步→逐层 wait_for_layer_load/save_kv_layer→wait_for_save 围栏→get_finished）使 load 与 compute 重叠；三后端 P2P-NCCL/NIXL-RDMA/Offloading 各自填实同一 KVConnectorBase_V1。mixin 纯静态无状态，真 connector 在进程级全局 get_kv_transfer_group()。无伏笔。bible 登记 6 接口。
