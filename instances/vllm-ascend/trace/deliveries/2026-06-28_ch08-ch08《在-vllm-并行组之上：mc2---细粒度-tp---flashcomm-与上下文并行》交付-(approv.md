# ch08《在 vLLM 并行组之上：MC2 / 细粒度 TP / flashcomm 与上下文并行》交付 (APPROVED)

- **Type**: delivery
- **Chapter**: 08
- **Date**: 2026-06-28
- **Timestamp**: 2026-06-28T20:23:21Z
- **Agents involved**: analyst, implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: delivery, ch08, parallel-state, MC2, fine-grained-tp, flashcomm, context-parallel, APPROVED

## What happened

reviewer 终判 APPROVED（含 5 个 source-grounding/算法维度建议 + 13 个 reader-comprehension 建议，全部 blocking=false negotiable=true，属定点小修不退章）。主线讲「加法式扩展」：init_ascend_model_parallel 复用基座 init_model_parallel_group/GroupCoordinator，在其上叠加昇腾专属组——MC2(matmul+通信融合)、细粒度TP(mlp/o-proj/lm-head/embedding 各自独立 TP 宽度)、flashcomm(2/3)、_DYNAMIC_EPLB(前向引用 ch09)。讲透 all_ranks reshape(5D)→各组 rank 切分的排布代数，用样例 A/B/C 做数值排布追踪。本章是 context-parallel(PCP/DCP) 排布的归口（attention 篇 ch19 enable_cp 运行期分流引用本章 CP 组排布）。

## Why it matters

昇腾在基座并行抽象上「加法式扩展而非 patch 重绑」的样板章；CP 组排布在此一次讲清供后续 attention 篇复用，避免排布代数在多章重复/漂移。

## What to remember

ch08 已交付 APPROVED。Bible 已登 12 个精简版接口(init_ascend_model_parallel/model_parallel_initialized/get_*_group/destroy_*)。新埋 2 伏笔：f6 _DYNAMIC_EPLB→ch09(EPLB 机制)、f7 CP 组排布→ch19(enable_cp 运行期分流)。归档时遗留非阻断机械项：lint_source_grounding 扫内部 impl-notes.md 仅列 2 源文件(<3 门禁)，补登 worker.py/ascend_config.py/消费方即清；正文源码根基合规。另有 3 处行号标签微偏(细粒度TP L107→L110、worker.py L837→L836)属定点小修。
