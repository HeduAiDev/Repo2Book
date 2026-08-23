# v3 ch09《EngineCore 的逐拍循环》交付归档（APPROVED）

- **Type**: delivery（v3 Archive 站，chapter-pipeline-v3 第九章、Part III 第一章）
- **Chapter**: v3 ch09 · Part III 引擎的心跳：调度循环 · kind=code（L0 缩放：循环框放大）
- **Pin**: vLLM v0.27.1（6e448d0ea）；行号基线即此版
- **Date**: 2026-08-23 · **Agents**: pipeline 各站（analyst→researcher→implementer→tester→explainer→illustrator→writer→reviewer）+ archivist（本记录）
- **Verdict**: APPROVED，11 条 issue（0 blocking + 11 negotiable/non-blocking），全文见 `artifacts-v3/ch09-engine-core-step-loop/reviews/review-report.json`

## What happened

- **回环**（`reviews/run-ledger.json`）：impl↔test 1 轮（48 用例 host 全绿 13.94s——**4 次连续全过**：原始门禁 2 跑 + 独立二次 tester 复门 2 跑；41 进程内单元 + 7 真 ZMQ 端到端，真 mp spawn 子进程引擎经真实 `launch_core_engines`/`CoreEngineProcManager` 出生、真 ROUTER/PULL 前端、两层握手、UTILITY 薄 RPC 注入）；write↔review 3 轮；L2/图 1 轮；盲审 1 轮零失败（4 图全 PASS：L1-partIII 导览/L2-ch9/五拍时间轴/bitmask 窗口）。
- **归档时抽查（issue 兑现状态）**：11 条全部 non-blocking/negotiable（含 1 条 negotiable:false 的 L2 锚点区间 cosmetic），**抽查均未在稿**——省略标记两处补全（loop active 日志行 / failed_kv_load 跳过分支）未见、空批早退「无 KV 传输组时」括注未见、「双队列」→「input_queue/output_queue 这对交接队列」限定未见、GC「老年代收集的暂停面」人话化未见、吞吐账 20%/17% 基准统一未见、图注 0.29 与 0.34/0.35 口径两侧均未动、llm_engine.step() 的 dummy 批三分支仍为裸代码块（L880-L884）——以 APPROVED 归档（ch02/ch07/ch08 同先例），writer 定点小修清单留 review-report.json。台账/工具侧 1 条：lint_anchors H1 标题 BARENUM 系统性误报（见 What to remember 3）。
- **bible 登记（v3 侧车）**：glossary-v3 +14（忙循环/热循环/两段式契约/ExecuteModelState 暂存态/AsyncOutputFuture/Future 提货单/语法位掩码 grammar bitmask/前向窗口/混相批/self-pipe trick/WAKEUP 哨兵/busy-waiting 忙等/迭代级调度/连续批处理/重叠版 step_with_batch_queue）；concepts-v3 +14（忙循环不空转/慢操作出循环清单/两段式契约/暂存态不变式/只等搬运不等计算/掩码窗口/拍序不变式/0-token flush 拍/迭代级调度契约/热循环性能账/撤单急件通道/握手两层时序/关停三态仲裁/心脏与外壳分离）；interfaces-v3 +33 条（step/__init__/run_busy_loop/_process_input_queue/_process_engine_step/has_work/_handle_shutdown/run_engine_core 信号路径等 EngineCore 全家桶 + 两段式 worker 面 + 握手面 + InprocClient 四方法面）；figures.json 追加 4 张（L1-partIII=l1-partiii-guide Part III 导览 + L2-ch9=l2-engine-core-step-loop + m1-five-beats-timeline 五拍时间轴 + m4-bitmask-window，book:v3）。
- **伏笔对账**：本章应埋 F6「grammar bitmask 窗口」（pedagogy-plan planted=9/paid=30）planted done:true——正文三处钩子（L610/L668/L910「允许表怎么算出来，Part VII 约束解码章回收」）+ L2 站 4 注记「F6 埋点：掩码怎么算归 ch30」，收款 ch30 未到期；本章应收 F1「EngineCore 五拍」（planted=1/paid=9）paid done:true——ch01 开篇承诺（「Part III 会把这个循环框逐拍拆开」）在 L5 兑现 + 「一拍五段」整节逐拍拆完 + 总结 L907 逐条销账 + m1/m3 实测佐证。与 run-ledger foreshadow_due 完全一致，foreshadow-v3.json 已带 evidence 对账条目。
- **图登记门禁**：`python scripts/lint_figures_registered.py <章目录>` 显式传参 exit 0（4/4 登记核过）。

## Why it matters

Part III（引擎的心跳：调度循环 ch9-12）首章 + 全书第一颗大伏笔 F1 兑现章：ch01 在 L0 图循环框里点名的五拍至此逐段拆完——①迭代级调度契约（每 forward 重组批，Orca 首创，sched/interface.py docstring 书面契约）、②④两段式契约（execute_model(non_block=True) 发起即返回拿 Future、worker 前向算完暂存十字段返回 None、sample_tokens 解包清态掩码采样——「只等搬运、不等计算」，AsyncOutputFuture.result() 等 D2H 事件而非前向计算）、③掩码窗口（被 ②④ 夹缝钉死、与前向窗口重合、收益=min(掩码耗时,前向耗时)——F6 埋点供 ch30 收）、⑤逐请求记账；框外忙循环骨架（idle 睡 input_queue.get(block=True) 零 CPU 不空转、1ms GIL 让渡、raise SystemExit 唯一正常出口）、0-token flush 拍、aborts_queue 急件通道、握手两层（HELLO→READY）、关停三态仲裁与 ENGINE_CORE_DEAD 死讯、InprocClient 心脏与外壳分离（同一 step 三种驱动）。ch10（连续批处理与 chunked prefill）从此章的 scheduler.schedule 一拍进入调度器内部；ch30 收 F6；ch13/17 的 KV 剖析/EEP 删除项已在此章 impl-notes 立账。interfaces-v3 累计 6 章 138 条。

## What to remember

1. **【writer 定点小修清单待用】** 11 条 negotiable 全部未兑现即归档（APPROVED 不阻断合规，ch02/ch07/ch08 有先例）——其中「④拍两半」一条（L602 future.result() 与 L604 sample_tokens 各等什么、`if model_output is None` 两分支语义、L424/L481 与 L359/L524 两条线索相斥）是 reviewer 自述「读了三遍仍拼不拢」的承重段，建议下轮 writer 小修最优先吃掉；吞吐账 20%/17% 基准不一与「两倍的损失」两种读法均验算不出、图注 0.29 图上无此数、0.34/0.35 口径微差三条是可核性硬伤类，同批修。
2. **【F6 收款到期提醒】** ch30（Part VII 约束解码章）应收「grammar bitmask 窗口」——本章已立术语「语法位掩码」、三步工作流（allocate→fill→apply，默认后端 xgrammar）、掩码窗口不变式（时序半边+数学半边）与 m4 实测（同行 logits 掩面之变/排除探针/窗口时序），ch30 从「允许表怎么算出来」直接起跳。
3. **【lint_anchors H1 BARENUM 误报第三次随章上报】** `# 第 N 章　标题` 全书统一 H1 格式被裸文字章号扫描命中（warn 级不计退出码；ch07 实测同报）——ch01/ch02 delivery 已记两次、本次又占 review issue 一条名额，修复案（排除各章自身 H1 行）在 review-report.json suggested_fix，仍留 Lead 决策。
4. **【工具卫生延续】** `python3` WindowsApps 坏桩（exit 49 零输出）本章归档再次复发（上轮 archivist 中断疑似同因），INSTANCE.md 已有记载、改用 `python` 即可；lint_dossier/lint_fidelity 对 artifacts-v3 静默跳过锚点核验的缺陷（ch04-ch08 已记）仍未修，本章由 impl-notes 收工审计（17 个机制锚逐行比对 v0.27.1 现核、未照抄 v2 旧行号）兜底。
