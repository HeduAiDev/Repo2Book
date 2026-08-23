# v3 ch05《ZMQ 拓扑与消息协议》交付归档（APPROVED）

- **Type**: delivery（v3 Archive 站，chapter-pipeline-v3 第五章、Part II 第二章）
- **Chapter**: v3 ch05 · Part II 分而治之：进程边界与消息 · kind=code（L0 缩放：紫色 ZMQ 边界带放大）
- **Pin**: vLLM v0.27.1（6e448d0ea）；行号基线即此版
- **Date**: 2026-08-17 · **Agents**: pipeline 各站（analyst→researcher→implementer→tester→explainer→illustrator→writer→reviewer）+ archivist（本记录）
- **Verdict**: APPROVED，18 条 issue（1 blocking fidelity + 17 non-blocking），全文见 `artifacts-v3/ch05-zmq-topology-and-protocol/reviews/review-report.json`

## What happened

- **回环**（`reviews/run-ledger.json`）：impl↔test 3 轮（47 用例 host + 容器 Linux 双平台全绿；round 1 验出 2 项 BLOCKING——F1 envs seam `VLLM_RPC_BASE_PATH` 旧默认致 Linux 真平台 e2e 全灭【已修 tempfile.gettempdir()】、F2 msgspec seam 杜撰 array_like 尾随裁剪线格式【已改全字段编码，真 msgspec 0.19/0.20/0.21.1 容器实测对照】；round 2 implementer 限流崩溃；round 3 独立复跑收口双平台 47 全绿）；write↔review 3 轮、L2/图 2 轮、盲审 2 轮（轮 1 失败 2 处——图 1 zmq-topology 悬空箭头+箭头尖压带标题、图 3 zero-copy 泳道说明文字被内框遮挡，均为 linter rect-rect/端点盲区，重排框边到框边+内框下移后二渲独立复审 PASS；轮 2 零失败）。
- **归档时抽查（issue 兑现状态，blocking 已在稿）**：blocking 1 条（linger「ZMQ 默认 0」→ 实为默认 -1 无限滞留、0 才是立刻丢；linger=4000 的动机重写为「有界冲刷 vs 无限滞留」的权衡）已兑现（chapter.md L844 现稿）。non-blocking 抽查 15 条内容类全部兑现：L765「两行之前」→「update_from_output 的逐请求循环里」；L588 pin/clone 大小分岔 → 真实开关 share_mem/pin_tensors（PIN_MEMORY 平台旗标）+ 大小取舍归因源码注释；L464 memoryview「只读」删去、点明活视图与 #50053 风险面；L657 图注补右上面板方位；L653 rank 0 补白话（DP 组里 0 号引擎取货）；L163 engine_ranks_managed/rank≈engine_index 行话打通；DP coordinator 首现处补身份 gloss；EEP 展开为弹性专家并行；水位线补定义（已成功取走的最新 message_id）；握手「这包回音」点名=前端对 HELLO 的地址集回音（READY 后引擎不等回执）；m5 轮 5 补 ZMQ 多帧原子投递机制；保活「不许改动」半边补调用纪律桥 + m5 轮 2 物证定性（编码侧验证视图、非在途安全）。#12287 溯源 issue 走选项 (b)：researcher 一手核实 PR body（sharegpt/6000 prompts/--max-concurrency=400 逐项吻合）回写 research/concepts.json provenance（2026-08-17 REVISE 补核），正文保留。
- **bible 登记（v3 侧车）**：glossary-v3 +12（HWM/linger/认亲帧/identity/many-to-many/OOB 旁路/反压/死讯/call_id/array_like/pinned memory/aux 零拷贝帧）；concepts-v3 +13（进出不对称、DEALER 先发言、两层握手、三段式线格式、array_like 契约、256B 阈值、零拷贝两答案、复用池纪律、OOB 旁路、HWM=0、两层队列解耦、路由纯函数、控制面折进同一条线）；interfaces-v3 +21 条（make_zmq_socket/线载体 Struct 族/MsgpackEncoder 三分支/TensorIpc 收发对/EngineCoreProc 双 IO 线程与复用循环/认亲收齐循环/validate_alive/UTILITY 薄 RPC/BackgroundResources 等）；figures.json 追加 5 张（L2-ch5=l2 章图 / ch05-fig-zmq-topology=m1 / ch05-fig-wire-format=m3 / ch05-fig-zero-copy-two-sides=m5 / ch05-fig-oob-bypass=m6，book:v3）。
- **伏笔对账**：本章应埋 F4「ROUTER 寻址能力」——正文「进出为什么不对称」代价条（单引擎为用不上的寻址能力付信封开销）+ 节末钩子（Part VIII 分布式章回来回答）+ 图 1 why 注框（#15906→revert→#17546 标 ch34 回收）+ 总结第 1 条均在稿，planted done:true（foreshadow-v3.json 已登记，收款按 pedagogy-plan 记 ch34 未到期）；本章无应收，对账零出入。
- **图登记门禁**：`REPO2BOOK_INSTANCE=vllm PYTHONIOENCODING=utf-8 python scripts/lint_figures_registered.py <章目录>` 显式传参通过（active_instance=triton-ascend，无参模式照不到 vllm）。

## Why it matters

Part II 的物理层章：ch04 把「过线」留成一根线头，本章把它展开成完整协议事实——四扇门拓扑与认亲时序、三段式线格式、msgspec array_like 线上契约、256B/64KiB 两层阈值、零拷贝两侧两答案（zmq 引用链 vs 首帧 tracker）、HWM=0 取舍、按步聚合按章路由、控制面折进同一条线。此后「引擎永不被慢前端阻塞」「路由无共享可变表」「没有心跳但每个万一都有具名机制值班」可被 ch06-08（上下行章）与 Part III/Part VIII 按「前章已立」直接引用；interfaces-v3 账本自 ch04 开张后首次登记 IPC 物理层接口（21 条）。

## What to remember

1. **【Lead 决策项·图定点小修】** review issue 6（非阻断）：图 1 ch05-fig-zmq-topology 底部 socket 记账行只数 5 只（漏引擎侧一次性握手 DEALER），与正文 L169「6 只」及图面自绘 6 框不一致。REVISE 轮已在 `diagrams/figure-requests.json` 留 replace 请求（done:[]）但**未执行**——gen 脚本 L214 仍是旧文案、PNG 未重渲（requests 01:51 > PNG 22:46）。修法一行：末句改「另有 1 对一次性握手 ROUTER/DEALER（startup 期即弃）」→ 重渲 → lint 双绿 → Read PNG 亲看 → 更新 manifest selfcheck。不需要动正文。
2. **【Lead 决策项·linter 缺陷仍未修（ch04 记录 #4 的跟进）】** lint_dossier/lint_fidelity 只认 `artifacts` 目录名、对 `artifacts-v3` 静默跳过锚点核验——ch04 交付记录问「ch05 收口时值得确认是否已修」：**未修**。本章靠人工兜底（impl-notes：313 个带符号锚点对 v0.27.1 机械复核零失配），但该缺陷对后续 v3 章仍是系统性风险。
3. **【环境·与 ch04 记录同款第二次】** review issue 8：本机 `python3` 是 WindowsApps 坏桩（exit 49 静默）+ GBK 控制台打印 ✓ 即 UnicodeEncodeError 崩——两章连发，INSTANCE.md「用 python + PYTHONIOENCODING=utf-8」的一行提示仍缺。
4. **【自愈确认】** ch04 记录的跨章链接待自愈项已闭环：ch05 narrative 已落盘（2026-08-17 01:51），ch04→ch05 相对链接现可达。
