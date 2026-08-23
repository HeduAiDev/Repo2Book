# v3 ch04《两个使用面，一套三件套》交付归档（APPROVED）

- **Type**: delivery（v3 Archive 站，chapter-pipeline-v3 第四章、Part II 首章）
- **Chapter**: v3 ch04 · Part II 分而治之：进程边界与消息 · kind=code（L0 缩放：API 进程左上＝使用面+双登记）
- **Pin**: vLLM v0.27.1（6e448d0ea）；行号基线即此版
- **Date**: 2026-08-16 · **Agents**: pipeline 各站（analyst→researcher→implementer→tester→explainer→illustrator→writer→reviewer）+ archivist（本记录）
- **Verdict**: APPROVED，16 条 issue（2 blocking 同源失配锚点 + 14 non-blocking），全文见 `artifacts-v3/ch04-two-usage-faces-one-trio/reviews/review-report.json`

## What happened

- **回环**（`reviews/run-ledger.json`）：write↔review 3 轮、L2/图 1 轮、盲审 2 轮（轮 1 失败 1 处——双登记图回指小片 ↔ 字形缺字渲染成豆腐块，换 ASCII 短横二渲后独立复审 PASS；轮 2 零失败）；impl↔test 1 轮（39 用例全绿）。
- **归档时抽查（issue 兑现状态，blocking 两条均已在稿）**：两条 blocking 实为同一失配（正文 L482 称「L0 图上 API 进程带底部那对『驱动』框」而 L0 图无此要素）——现稿已改为锚定本章 L2 站 12「两种驱动 · 拉取端」+ L0 上「output_handler（单任务）」单框、离线裸循环不单独成框的如实口径。non-blocking 抽查全部兑现：llm.py 引文补全 `[vllm.AsyncLLMEngine]`、asyncio.sleep(0) 官方链接、第三件取法分岔口径全文统一（在线直呼 make_async_mp_client 旁路）、EngineCoreOutput/EngineCoreOutputs 信封-载荷绑定、PR #11074「按环境尽力选择」展开（fork/spawn/CUDA 检测）、generate 四臂联合类型就地绑定、在线侧 `__init__` 并排再嵌、core.py 两段未讲解代码各补就地括注（reuse_buffers 缓冲回收池→ch5 承接、wave_complete→Part VIII）、demux→「按内部 id 拆开分发」、「死路」→「来路」、m8 表前 caveat 瘦身+表后记两笔。impl-notes 源文件登记机械告警（issue 2）已补登，lint_source_grounding 现跑全绿。
- **bible 登记（v3 侧车）**：glossary-v3 +12（门面/协议面/client 工厂/前端与调用方/VLLM_ENABLE_V1_MULTIPROCESSING/fork 与 spawn/pickle/EngineCoreOutput 与 EngineCoreOutputs/RequestState/收数线程与收数任务/MPClientEngineMonitor/CancelledError 与 GeneratorExit）；concepts-v3 +11（三件套可 diff、工厂二轴、同步 LLM≠进程内引擎、client_index 随请求过线、回程对账一个分支吃两种面、信封-载荷、双登记不变量、两种驱动拉取端、asyncio 取消语义、自增 id 按序还原、门面 vs 协议面）；interfaces-v3 新建（v3 首个有精简版的章，17 条接口签名）；figures.json 追加 4 张（L2-ch4 / L1-partII / ch04-fig-client-factory=m2 / ch04-fig-double-registration=m3，book:v3）。
- **伏笔对账**：本章应埋 F3「client_index 随请求过线」——正文「盖章过线」节 + 总结第 3 条均在稿，planted done:true（foreshadow-v3.json 已登记，收款按 pedagogy-plan 记 ch34/Part VIII 未到期）；本章无应收，对账零出入。
- **图登记门禁**：`REPO2BOOK_INSTANCE=vllm python scripts/lint_figures_registered.py <章目录>` 显式传参通过（active_instance=triton-ascend，无参模式照不到 vllm）。

## Why it matters

Part II 首章把「v1 凭什么一套通吃」落成可 diff 的事实：门面 vs 协议面、工厂二轴、双登记不变量、client_index 路由键、两种驱动拉取端自此可被后章按「前章已立」引用；interfaces-v3 账本自此开张（ch05 的 _msgspec_seam/zmq_ipc 接口将接续登记）。

## What to remember

1. **【Lead 决策项·环境】** review issue 5：本 Windows 宿主 `python3` 是 WindowsApps 静默坏桩（exit 49 零输出）——CLAUDE.md/RUNBOOK 全写 `python3 scripts/…`，照抄即假阴性；建议 INSTANCE.md 运行环境段加一行「一切 scripts/ 用 python（Miniconda）跑」或删别名桩。本记录与门禁均已用 Miniconda python 执行。
2. **【在制时序·自愈】** review issue 6：ch04 末行跨章链接指向 ch05 narrative/chapter.md（尚未落盘，ch05 在产线）——lint_anchors 目录级检查不验文件属已知盲区；ch05 Write 站落盘即自愈，发布顺序保证 ch05 先于/同时于 ch04 上线即可。
3. **【L1 导览图盲审附注】** L1-partII 的 selfcheck_note 写「L0 ×0.98」、图面徽标实为「L0 ×0.72」——note 笔误（盲审已提示），真相源侧车小注待订正，不影响图面。
4. **【对账惯例】** ch02 交付记录提的 lint_dossier/lint_fidelity 只认 `artifacts` 目录名的 Lead 决策项，本章归档未复跑验证——ch05 收口时值得确认是否已修。
