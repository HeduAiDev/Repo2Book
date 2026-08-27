# v3 ch12《异步调度》交付归档（APPROVED）

- **Type**: delivery（v3 Archive 站，chapter-pipeline-v3 第十二章、Part III 收官章）
- **Chapter**: v3 ch12 · Part III 引擎的心跳：调度循环 · kind=code（L0 缩放：循环框+执行臂接缝）
- **Pin**: vLLM v0.27.1（6e448d0ea）；行号基线即此版
- **Date**: 2026-08-27 · **Agents**: pipeline 各站（analyst→researcher→implementer→tester→explainer→illustrator→writer→reviewer）+ archivist（本记录）
- **Verdict**: APPROVED，14 条 issue（0 blocking；13 条 negotiable + 1 条 negotiable:false「五类计数口径三处不齐」；其中 5 条 reader-comprehension 维），全文见 `artifacts-v3/ch12-async-scheduling/reviews/review-report.json`

## What happened

- **回环**（`reviews/run-ledger.json`）：impl↔test 1 轮（39 项纯单元 host 全绿、复跑 1.79s 再绿、无 import vllm——CUDA 面以 HOST SEAM 承载：HostEvent/HostCopyStream 站 `torch.cuda.Event` 契约位、脚本化 logits 前向、StructuredOutputManager 恒全 1 掩码）；write↔review 2 轮；L2 2 轮（盲审②回修走廊双标签两处——墨顶标签进北行带底 + 车道横线穿字避让，改 gen_L2.py 布局逻辑 spec 零改动；Review Repair 又补一次章内副本同步——正文内嵌副本曾停在旧渲缺第 14 条流标签）；盲审 1 轮零失败（9 图全 PASS）。
- **归档时抽查（issue 兑现状态）**：抽查 8 处标记（_prepare_input_ids elide 行号仍写 L1807-L1856/正文缺「显式 True 硬失败」句/#27614「一律开启」/「第三件事」缺前置枚举/「两种人生」措辞/「bitmask 窗口」回指错名/「谁家孩子」比喻叠加/m1 实测表无 ROCm DeepEP 行）**全部未在稿**——以 APPROVED 归档（ch02/ch07/ch08/ch09/ch10 同先例），writer 定点小修清单留 review-report.json。最优先三条：① elide 行号范围失准（dossier embed_excerpts 同源漂移，suggested_fix 给了 L1813-L1857 连续省略合并案，**需同步订正 dossier 防 retrofit 复发**）；② early-stop「确定已算」未定义、与刚立的 computed−ph=真实已算口径差 1 未打通（全章最烧脑式子上读者第一次拿公式对数字就失败）；③ num_in_flight_tokens 与 ph 两个「在飞」计数器全章无对照（m8 第三列数字无解说、抢占节 stale 账失支点）。另两条零成本：显式 True 硬失败一句话（explainer m1 invariant 已主张、图注 L178 已有、正文缺）与 #27614「一律」改「默认开启（仅五类例外）」。
- **bible 登记（v3 侧车）**：glossary-v3 +22 首现章 ch12（两态心跳/双缓冲/盲调度/填管道优先/三元组/AsyncScheduler/占位账本/early-stop 剪枝/spec 拒绝双回退/deferred sampling/采样 token 不落 CPU/影子状态/CUDA 流/CUDA 事件/H2D/同步禁区/乐观纠错群/uniform decode/气泡/bonus token/NanoFlow/AsyncGPUModelRunnerOutput）**另补登 3 条前章漏登术语、首现章如实记 ch03**（异步调度/max_concurrent_batches/DeepEP 高吞吐 DBO——ch12 是主场但术语 ch03 已立面，防抢注）；concepts-v3 +15；interfaces-v3 +26（EngineCore 两态心跳主线/批队列装配段/post_step/has_work/step 同步对照/AsyncScheduler 两覆写/early-stop 剪枝/追赶公式占位项/乐观推进/preempt async 账单/update_from_output 热循环/get_grammar_bitmask/max_concurrent_batches/默认仲裁/get_scheduler_cls/AsyncOutputFuture/AsyncGPUModelRunnerOutput/_bookkeeping_sync async 分支/_prepare_input_ids 三岔口/_compute_prev_positions/乐观纠错群/影子字段/tripwire 位/Request 四计数器/SchedulerOutput async 标志对，累计 9 章 213 条）；figures.json 追加 9 张（L2-ch12 + 机制图 step-binding-chain(m2)/queue-triple(m5)/two-state-queue(m4)/placeholder-ledger(m6)/block-convert(m7)/token-two-paths(m10)/gpu-backfill(m11)/deferred-sampling(m14)，book:v3）。
- **伏笔对账**：本章应埋无、应收无（pedagogy-plan F1-F10 的 planted/paid 集合均不含 12，与 run-ledger foreshadow_due:[] 一致）——foreshadow-v3.json 零改动；F6（grammar bitmask 窗口）在本章「第二次路过」属信息性回指（只讲时序半边：缺 token 必须推迟），收款仍 ch30 未到期；其余前向指针（Part IV 块池内景/Part V 执行管线与 CUDA graph/ch17 执行三层与 tripwire 反面案例/ch18 持久批次内景/Part VII 投机解码与掩码本体）均信息性指路、非登记伏笔。
- **图登记门禁**：`python scripts/lint_figures_registered.py <章目录>` 显式传参 exit 0——active_instance=triton-ascend 下裸跑与 REPO2BOOK_INSTANCE=vllm 双跑均绿；manifest 9 图与 bible v3-ch12 条目逐 id 集合相等（本记录内程序核对）。

## Why it matters

Part III 收官：ch09 立心跳骨架、ch10 打开 token 账本、ch11 拆抢占与请求的一生、本章把心跳换成重叠版并打通缝两侧的记账——整个 Part III 调度器的活收束成一句「只认 token 数、只借还块 id」。三件立桩：① 占位账本不变式 computed−ph=真实已算（cache_blocks 消费面即 ch13-15 块池/显存账本的入口）；② worker 影子状态 + GPU 直拷回填三岔口（持久批次内景 ch18、CUDA graph 回放 Part V、同步禁区纪律先于 slot_mapping 反面案例立下）；③ v0.27.1 默认心跳=重叠版（None 仲裁 True、#27614 翻转默认）——「不是深水区选项、是出厂默认」的叙事角度全文贯彻。async+spec+structured 三特性两两组合的状态机复杂度（本 pin 前三个月三个占位 underflow 修复 #42117/#46066/#48245）是这台精密账机的维护成本，全部欠账 Part VII 投机解码落地章回收。

## What to remember

1. **【writer 定点小修清单待用】** 14 条全部未兑现即归档（APPROVED 不阻断合规）。最优先三条见 What happened；注意 issue-1 的修复要**连 dossier 一起订正**（elide 注行号漂移是 v2→v3 迁移没核到底的同一处，正文与 dossier 双落点）。
2. **【DBO 展开口径冲突：ch12 与 ch03 不一致，源码裁决】** ch03 立「DBO（Dual-Batch Overlap，双批重叠）」与源码 `vllm/config/vllm.py:L2222` "dual batch overlap" 一致；ch12 正文 L95 写「高吞吐双缓冲重叠」、review issue-13 的 suggested_fix 又写「double-buffered overlap」——两者均与源码/ch03 口径不符。glossary-v3「DeepEP 高吞吐 DBO」条已按源码口径落（dual batch overlap，双批重叠）；writer 小修 issue-13 时以「双批重叠」收口，勿照抄 suggested_fix 的英文展开。
3. **【ch11 出序在后：主场术语不抢注】** ch11（抢占与请求的一生）narrative 已在稿、未归档（无 reviews/、state 无条目）——`num_stale_output_tokens`/`drop_stale_output`/`num_in_flight_tokens` 三枚计数器与 stale 协议是 ch11 主场，本章 glossary/concepts 均未抢注、待 ch11 归档时登记。另发现同机制两名：ch12「锁步排空」vs ch11「锁步冲销」（stale drain），ch11 归档时统一（倾向随 ch11 主场词）。
4. **【补登 3 条 ch03 术语】** 异步调度/max_concurrent_batches/DeepEP 高吞吐 DBO 在 ch03 正文已立而 glossary 侧车漏登，本次补登、首现章如实记 ch03（与 ch03 delivery「3 条首现章勘正」同族问题）；后续 gap 审计按此口径。
5. **【python3 坏桩复现】** 本机 python3 仍是 WindowsApps stub（exit 49 无输出），本次全部脚本走 Miniconda python——ch09/ch10 delivery 已记，续跑者直接用 python。
