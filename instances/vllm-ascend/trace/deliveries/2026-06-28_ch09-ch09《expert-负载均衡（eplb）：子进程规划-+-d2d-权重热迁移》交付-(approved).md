# ch09《Expert 负载均衡（eplb）：子进程规划 + D2D 权重热迁移》交付 (APPROVED)

- **Type**: delivery
- **Chapter**: 09
- **Date**: 2026-06-28
- **Timestamp**: 2026-06-28T21:48:42Z
- **Agents involved**: analyst,  implementer,  tester,  writer,  reviewer,  archivist
- **User present**: False
- **Tags**: delivery,  ch09,  eplb,  expert-load-balancing,  subprocess-planning,  d2d-weight-transfer,  p2p,  policy-polymorphism,  APPROVED

## What happened

reviewer 终判 APPROVED（8 个 source-grounding/算法/readability 建议 + 7 个 reader-comprehension 建议，全部 blocking=false negotiable=true，属定点小修不退章）。本章无基座对位文件——eplb 是 vLLM 尚未合入、昇腾自带的重型特性，只点明它仅借用 vLLM 通信原语(GroupCoordinator.all_gather / dist.P2POp)。主线四块串成「在线热迁移专家权重」流水线：①EplbUpdator 节拍状态机(forward_before 启 D2D 搬运/forward_end gather moe_load 唤醒规划，cur_iterations+三间隔常量拼节拍)；②EplbProcess 独立子进程跑均衡策略(planner_q/block_update_q 两条跨进程队列解耦计算与规划)；③D2DExpertWeightLoader 异步 P2P 三态机(dist.P2POp 批量 isend/irecv 搬 expert 权重)；④PolicyFactory 策略多态(policy_abstract 抽象基类 + DefaultEplb 走完，flashlb/swift 点到为止)。讲透「为什么子进程+队列+异步搬运」：规划是重 CPU 计算，不能卡在推理主循环。f6 伏笔(ch08 _DYNAMIC_EPLB 组)在 compute_and_set_moe_load 处回收。

## Why it matters

昇腾「在基座之上自带重型特性、仅借通信原语」的样板章；把跨进程解耦+异步热迁移这类生产级工程范式讲透，且把 ch08 埋下的 _DYNAMIC_EPLB 组(f6)落地回收，闭合 Part 内伏笔。

## What to remember

ch09 已交付 APPROVED。Bible 已登 13 个精简版接口(EplbUpdator/forward_before/forward_end/compute_and_set_moe_load/EplbWorker/compose_expert_update_info_greedy/EplbProcess/D2DExpertWeightLoader/ExpertWeightUpdateState/EplbPolicy.rebalance_experts/PolicyFactory.generate_policy/DefaultEplb.rebalance_experts/VllmEplbAdaptor)。f6(_DYNAMIC_EPLB→ch09)已 status=resolved/resolved_in=ch09。精简版只验可读控制流(节拍状态机/队列解耦/策略多态纯 Python，test_eplb.py 11 passed)，真实 P2P 搬运+子进程绑 NPU+all_gather 不真跑(host 无 NPU/CANN，eplb_runtime_stub.py 接住)。lint_fidelity 无 BLOCKING。遗留非阻断小修：DefaultEplb 内嵌片段 expert_num 省略标注补来源、_launch_process spawn 措辞贴 docstring、do_update 解读长段拆四元组→五元组、big-O 的 N 明确为 num_expert、pipeline_overview.png 残留键 expert_map 加未用标记——均定点小修不退章。
