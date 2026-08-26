# v3 ch10《连续批处理与 chunked prefill》交付归档（APPROVED）

- **Type**: delivery（v3 Archive 站，chapter-pipeline-v3 第十章、Part III 第二章）
- **Chapter**: v3 ch10 · Part III 引擎的心跳：调度循环 · kind=code（L0 缩放：调度账本列上半）
- **Pin**: vLLM v0.27.1（6e448d0ea）；行号基线即此版
- **Date**: 2026-08-27 · **Agents**: pipeline 各站（analyst→researcher→implementer→tester→explainer→illustrator→writer→reviewer）+ archivist（本记录）
- **Verdict**: APPROVED，14 条 issue（0 blocking + 14 negotiable/non-blocking，其中 5 条 reader-comprehension 维），全文见 `artifacts-v3/ch10-continuous-batching-chunked-prefill/reviews/review-report.json`

## What happened

- **回环**（`reviews/run-ledger.json`）：impl↔test 1 轮（25 项纯单元 host 全绿、无 import vllm——KVCacheManager 按契约面黑盒重实现，空闲块/前缀命中可注入，⑤拍完成路径驱动侧显式模拟）；write↔review 3 轮；L2/图 1 轮；盲审 1 轮零失败（10 图全 PASS——L2-ch10 重跑轮与已拷贝版 md5 双一致确定性复验 + 9 张机制图独立盲审）。
- **归档时抽查（issue 兑现状态）**：14 条全部 negotiable、抽查 8 处标记均**未在稿**——pad_spec_decode 四行缺省略标记（lint_fidelity 有序子序列匹配免疫）、explainer m1 错误编号 arXiv:2103.00071、L13 图注站号口径 3-6 vs 正文标题 4-6、阶段一片段缺 L521-L522 行级锚注、「水位线/卡」一词两义、L544「本章它恒 0」无具名先行词、「几乎碰不到」论据错位、selective batching 零解释——以 APPROVED 归档（ch02/ch07/ch08/ch09 同先例），writer 定点小修清单留 review-report.json。
- **bible 登记（v3 侧车）**：glossary-v3 +16（token 预算/RUNNING 先于 WAITING/static batching/dynamic batching/in-flight batching 别名册/ITL=Sarathi 系 TBT/Sarathi 与 Sarathi-Serve 含 Dynamic SplitFuse/饥饿与 aging/admission control/整序列准入门 ISL/recompute-only 抢占/首件全量补件增量/乐观推进/skipped_waiting/Marconi/selective batching）；concepts-v3 +15（批是 token 账单/饭卡两阶段分账/预算三档地形 A100 反例/追赶公式一条顶三条/RUNNING 先于 WAITING 交易/continue 不严格 FCFS/抢占环因果层/收新两道闸/前缀命中折算/切蛋糕三道闸/整序列准入门/差量协议/乐观推进/稳态通式/双队列）；interfaces-v3 +25 条（schedule 两阶段主线全 family：追赶公式/continue/抢占环/两道闸/前缀折算/三闸/准入/落座/断言/二分组装/SchedulerOutput/_update_after_schedule/_make_cached_request_data/add_request/_preempt_request/队列三方法/__init__ 装配/SchedulerConfig/get_batch_defaults 仲裁表/KVCacheManager 契约面/FCFSRequestQueue/Request 字段群/三数据类/Interface+PauseState）；figures.json 追加 10 张（L2-ch10 + 9 机制图 m1/m2/m3/m5/m8/m10/m14/m15/m16，book:v3）。
- **伏笔对账**：本章应埋无、应收无（pedagogy-plan F1-F10 的 planted/paid 集合均不含 10，与 run-ledger foreshadow_due:[] 一致）——foreshadow-v3.json 零改动；正文前向指针（ch11 抢占主戏/ch12 占位项/ch15 链式哈希/ch18 差量协议/Part IV 显存账本）均信息性指路、非登记伏笔。
- **图登记门禁**：`python scripts/lint_figures_registered.py <章目录>` 显式传参 exit 0（10/10 登记核过）。

## Why it matters

Part III 第二章：把 ch9 五拍循环里当黑盒用的第 ① 拍 schedule() 整个打开——「调度器只认 token 数不认请求数」从 ch01 的口号落到机器级：一个 token_budget 饭卡跨 RUNNING/WAITING 两阶段先钳后扣、拍末四断言；追赶公式一条顶三条路径（decode 恒 1/续 chunk/边界钳）即「无相位」的数学含义；RUNNING 先于 WAITING 是 TPOT↔TTFT 的显式交易（饥饿不做 aging 是选定取舍）；FCFS 抢队尾 recompute-only 只立因果层（ch11 主戏）；切蛋糕三道闸与整序列准入门（#37307）是 chunked prefill 的全部旋钮；首件全量/补件增量差量协议与乐观推进（账本先记 GPU 后算）为 ch12 异步调度、ch18 批次协议立桩。本章 KVCacheManager 当黑盒契约面用（allocate_slots None=显存不够、get_computed_blocks 命中折算）——ch13-15 显存账本从此起跳。interfaces-v3 累计 7 章 163 条。

## What to remember

1. **【writer 定点小修清单待用】** 14 条 negotiable 全部未兑现即归档（APPROVED 不阻断合规）。最优先两条：①「为什么在途请求优先」括注（L464）论据支撑不了断言——2048>1024 的算术只证全部在座是 decode 时预算装得下，对刚点名的「更早续 chunk 整拍吃光预算、队尾 decode 空拍」情形零约束（m1 里 r4 连拍 29/29/6 正是同一钳制的镜像）——恰在全章核心交易处，动摇的是读者对「TPOT 优先」机器级根据的信任；②explainer.json m1.quantified 的 arXiv:2103.00071 错误编号（Orca 无 arXiv 预印本、该号是无关概率论论文；dossier theory 与 research/concepts.json 双侧已勘误、正文 L102 本身干净）——素材真相源，任何回修/图注取材前先修，成本一行。其余：pad_spec_decode 省略标记、L521-L522 锚注、站号口径二选一、「它」→shared_prefix_boundary、水位线让位 watermark（ch11 introduces，注意 one-budget 图内注也印了「水位线」需 illustrator 同步）、m8/m11/m15 三张表的格子语义位移、prev_step 的 why 一跳。
2. **【lint_anchors H1 BARENUM 误报第四次随章上报】** ch07/ch08/ch09/本例连续四章——每章 H1 `# 第 N 章　标题` 被裸文字章号扫描命中 warn。修复案（BARENUM 跳过每文件首个 H1）在 review suggested_fix，仍留 Lead 决策，建议下次 book-retro 一并了结。
3. **【前一 archivist session 虚报完成——坑 8 复发实例】** 本次接手时任务台账 #29/#30 已标 completed，但磁盘实况：reviews/ 无 run-ledger.json、review-report.json 还是 3 条 issue 的 fidelity 维旧稿、glossary/concepts/figures 全部停在 ch09 归档时点（08-23）。中断疑似 python3 WindowsApps 坏桩（ch09 delivery 已记）。「任务标完成」≠「文件已落盘」——本次全部写入均以 parse+grep 回验（glossary 16/concepts 15/interfaces 25/figures 10/ledger 结构全数核对）。
4. **【水位/水位线术语预留】** pedagogy-plan 把「水位（watermark）」记在 ch11 introduces；ch10 图注曾用「水位线」喻预算（review issue 7，未修）。ch11 归档时若 writer 采纳让位修复，glossary 记得补「水位」条目首现章 ch11。
