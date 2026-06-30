# ch22《KV cache 管理与调度器的 NPU 特化》交付 (APPROVED) — Part V 收官

- **Type**: delivery
- **Chapter**: 22
- **Date**: 2026-06-30
- **Timestamp**: 2026-06-30T06:07:37Z
- **Agents involved**: analyst, implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: delivery, part-v, kv-manager, scheduler, subtract-only, approved

## What happened

reviewer 判 APPROVED。本章讲昇腾对 KV 管理/调度的「少碰核心循环」哲学：90% 原样继承 vLLM，只在两处开子类——(1) KV manager 复用 FullAttentionManager/BlockPool，新增 CompressAttentionManager 处理压缩 MLA spec 的 block 分配与前缀命中，get_manager_for_kv_cache_spec 按 spec 类型重映射；(2) 三个 Scheduler 子类 SchedulerDynamicBatch(BudgetRefiner 查表动态 token 预算)/RecomputeScheduler(PD 分离丢弃重算+异步)/ProfilingChunkScheduler(profiling 分块)。精简版只删 dossier 批准项，host 19 passed。8 个新接口已登记 bible。

## Why it matters

收束 Part V「注意力子系统」，与 ch20/21 重度特化形成对照，示范「该复用就复用、别为改而改」的插件复用哲学。是 ch13/14/15 worker/runner/单步前向的上游调度与 KV 分配层。

## What to remember

review-report.json: APPROVED，全部 issue negotiable=true/blocking=false。两类非阻断点：(a) lint_source_grounding 对内部 impl-notes.md 报源码文件 <3（正文 lint_chapter_structure 已过，零脚手架泄漏，纯机械补登记可转绿）；(b) §22.5 RecomputeScheduler 终止性两拍表记号与 pop 取队尾的代码语义/设定方向自相矛盾（散文论证与代码均无误，纯表述层，让 request 不在队尾即自洽）。其余为 reader-comprehension 维度的术语/动机补注（block/prefix cache 定义、decode-first 与 PD recompute 动机等）。无伏笔到期（埋/回收均空）。
