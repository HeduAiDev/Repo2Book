# v3 ch18《持久批次与固定地址》交付归档（APPROVED）

- **Type**: delivery（v3 Archive 站，chapter-pipeline-v3 第十八章、Part V「GPU 不等 Python：执行管线」首章）
- **Chapter**: v3 ch18 · Part V · kind=code（L0 缩放：执行臂中层——worker 进程里 GPUModelRunner 一拍之内的全部内务）
- **Pin**: vLLM v0.27.1（6e448d0ea）；行号基线即此版
- **Date**: 2026-08-28 · **Agents**: pipeline 各站（analyst→researcher→implementer→tester→explainer→illustrator→writer→reviewer）+ archivist（本记录）
- **Verdict**: APPROVED，16 条 issue（0 blocking、16 negotiable，其中 7 条 reader-comprehension 维），全文见 `artifacts-v3/ch18-persistent-batch-fixed-addresses/reviews/review-report.json`

## What happened

- **回环**（`reviews/run-ledger.json`）：impl↔test 1 轮（44 测全绿 ~1.7s、host 直跑：真 torch/numpy/triton kernel 定义逐字、无 vllm 包无 CUDA——CUDA 面 HOST SEAM 承载；run-ledger 的 impl_test_ledger 数组为空，实施事实出自 impl-notes）；write↔review 2 轮；L2 1 轮；盲审 1 轮零失败（8 图全 PASS）。foreshadow_due=[]、escalated=null。
- **归档时抽查（issue 兑现状态）**：抽查 6 处标记（① L387 之后静默跳过 L388 `is_token_ids` 写行无省略标记；② L65 注释引文丢所有格 's；③ L596「第 12 章立过三件套基础」；④ L550 CpuGpuBuffer 开场缺 L2 定位句；⑤ L69 图注「1 个块号 + 2 个整数」丢「至多」；⑥ L157 第二个语义坑未显式回扣）**全部未在稿**——以 APPROVED 归档（ch02/ch07/ch08/ch09/ch10/ch12 同先例），writer 定点小修清单留 review-report.json。最优先：issue-1 是全书「只删必标注」纪律的唯一破口，修复时**连 dossier embed_excerpts 的 elide 登记一起核**（ch12 issue-1 同族教训：dossier 是漂移源，suggested_fix 已给出「L388 原样纳入摘录不越权」的省事路径）；issue-9 是跨章术语漂移（「三件套」已注册为 ch4 章标题术语，ch12 原文用的是「三个 CUDA 原语」）。
- **bible 登记（v3 侧车）**：glossary-v3 +11 首现章 ch18（InputBatch/slot 三段式/BatchUpdateBuilder/CpuGpuBuffer/固定地址地基/先等后录/可变性裁决/ngram / Prompt Lookup 投机解码/logitsprocs/CU 偏移/BatchDescriptor——BatchDescriptor 首现章如实记 ch18，ch03 优化级一节仅名单式预告无展开）；concepts-v3 +7（批次不回家——InputBatch 差量调和/SchedulerOutput 差量协议收件侧/CpuGpuBuffer 固定双端缓冲/slot 三段式/固定地址地基/收集 O(total) 向量算子链/持久批次自产自销闭环，对齐 pedagogy-plan introduces 三项+拆细）；interfaces-v3 +26（SchedulerOutput/CachedRequestData/NewRequestData 三载体 + execute_model/_update_states 四段调和 + InputBatch 五方法 + BatchUpdateBuilder 全类 + _prepare_inputs/_prepare_input_ids/_get_cumsum_and_arange + CpuGpuBuffer 全类 + __init__ 持久缓冲块 + _bookkeeping_sync/sample_tokens/synchronize_input_prep/_may_reorder_batch + BlockTable 六方法 + update_scheduler_for_invalid_drafts + LateInteractionRunner 调用位 + ExecuteModelState，出自 impl-notes 1:1 Source Map）；figures.json 追加 8 张（L2-ch18(l2) + diff-protocol(m01)/reconcile-five-beats(m02)/slot-lifecycle(m03)/inputbatch-layout(m04)/cpugpubuffer(m05)/gather-pipeline(m06)/fixed-address-replay(m14)，book:v3、mechanism_id 对齐 dossier ch18-mNN 账本）。
- **伏笔对账**：本章应埋无、应收无（pedagogy-plan F1-F10 的 planted/paid 集合均不含 18，与 run-ledger foreshadow_due:[] 一致；正文 grep 零 F 标记零「伏笔」字样）——foreshadow-v3.json 零改动。F7（block_table 已寻址 planted 13→paid 22）本章重度使用 append_row/commit_block_table 属信息性回指、slot_mapping 深水区收款仍 ch22；F10 planted 19 非 18；章尾「下一章《编译与捕获》」是信息性指路、非登记伏笔。
- **图登记门禁**：`python scripts/lint_figures_registered.py <章目录>` 显式传参 exit 0——active_instance=triton-ascend 下显式传参即可绿（ch01 已记的坑：无参/--all 模式经 instance.py 只扫 artifacts/ 照不到 artifacts-v3）；manifest 8 图与 bible v3-ch18 条目逐 id 集合相等（本记录内程序核对）。

## Why it matters

Part V 首章：执行臂中层点亮。前半「批次不回家」（差量协议收件侧 + slot 三段式 + 四段调和，赌注批次间高重叠：O(R·L) 全量重建 vs O(ΔN·行长) 增量）与后半「地址不搬家」（CpuGpuBuffer 三视图 + runner 持久缓冲块 + 先等后录防踩，六喂图缓冲 data_ptr 五拍实证）合起来，才接得住 ch19 CUDA graph 的回放语义——回放命中 = BatchDescriptor 全等 AND 地址不变，本章交付的是后半条件的全部供给。四立桩：① resumed 同字段两种语义（常规 append vs 整体替换+全量重算）与 ngram-GPU 可变性裁决（全库唯一「worker 改写调度器输出」豁口）；② slot 三段式不变量（[0,num_reqs) 恒连续、add 永不覆盖活行——GPU 侧按行号的一切算术的前提）；③ 收集 O(total) 向量算子链无 prefill/decode 分支（「调度只认 token 数」在 worker 侧的镜像）；④ 持久批次自产自销闭环（调度器不回传 token，PP 唯一例外）。PyTorch 官方文档点名 vLLM 是「变长批次共享静态缓冲池」的 notable example——本书正在读的就是官方正面教材。

## What to remember

1. **【writer 定点小修清单待用】** 16 条全部未兑现即归档（APPROVED 不阻断合规）。最优先三条见 What happened 抽查段；注意 issue-1 修复要连 dossier 一起核（embed_excerpts elide 把 L377-L388 登记为一整段、writer 剜出 L387 后产生的第二个缺口未补标记——dossier 是漂移源，ch12 issue-1 同族）。
2. **【账本缺口如实上报（2026-08-28）】** bible 侧车与 state.v3.chapters 目前只覆盖 ch01-ch12 + ch18：**ch13/ch14/ch17 已有 narrative+reviews/review-report 但未经 archivist 归档**（ch13 有 delivery 文件、ch14/ch17 连 delivery 也缺），ch15 无 reviews/、ch16 未发车。本章未抢注 ch13/ch14 已立概念（「行主序摊平」「KV cache group」「query_start_loc CU 偏移的注意力侧」等），留给其补归档——**ch13-17 补归档前，gap 审计对「前章已立」的判定会失真**，需 Lead 排期（review issue-16 Triton 括注是否回指 ch13 也悬在此）。
3. **【跨章图系分叉】** ch12-fig-token-two-paths（GPU 侧 worker 泳道头绿）与 ch18-fig-diff-protocol（worker 框橙）对同一对进程端点画法分叉（review issue-6，negotiable）——倾向 worker 框改绿对齐 L0 角色色，属图系级决策归 Lead，非本章单图回修。
4. **【python3 坏桩复现】** 本机 python3 仍是 WindowsApps stub（exit 49 无输出），本次全部脚本走 Miniconda python——ch09/ch10/ch12 delivery 已记，续跑者直接用 python。
