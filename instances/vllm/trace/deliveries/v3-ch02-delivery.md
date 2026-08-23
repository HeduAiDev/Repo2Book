# v3 ch02《跟一个请求走完全程》交付归档（APPROVED）

- **Type**: delivery（v3 Archive 站，chapter-pipeline-v3 第二章）
- **Chapter**: v3 ch02 · Part I 全景与读法 · kind=meta（L0 整图动态版，无精简版）
- **Pin**: vLLM v0.27.1（6e448d0ea）；行号基线即此版
- **Date**: 2026-08-16 · **Agents**: pipeline 各站（analyst→…→reviewer）+ archivist（本记录）
- **Verdict**: APPROVED，24 条 issue 全部 non-blocking/negotiable，全文见 `artifacts-v3/ch02-request-lifecycle/reviews/review-report.json`

## What happened

- **回环**：write↔review 2 轮、L2/图 3 轮、盲审 1 轮零失败（`reviews/run-ledger.json`）；meta 章无精简版（impl_test_rounds=0）。
- **归档时抽查（issue 兑现状态）**：m8 行号勘误（sched/utils.py L101-L107→L104-L111）已在正文兑现（chapter.md L320）；其余词级建议未逐条兑现——「十来个」（L92）、「右列其余」（L11）、「一次性开关」（L564）等仍在稿。素材侧 explainer m8 表仍写 L101-L107/L109-L115（issue 3 的 explainer 同步建议未落，素材-正文存在一处行号不一致回流风险）。APPROVED 不阻断，是否发起 writer 定点小修由 Lead 定。
- **bible 登记（v3 侧车，不动 v2 封版账）**：glossary-v3 +14＝ch02 新术语 9（FastAPI/pydantic/msgpack/双轨 id/client_index/ROUTER→DEALER 与 PUSH→PULL/EOS/stop_token/WHATWG）+ ch01 归档漏登补账 5（msgspec/logits/hidden_states/stop-string/pooling，首现章如实标 ch01、释义尾注标补账）；concepts-v3 +14（含 pedagogy-plan introduces 两条：主线 16 站走读、SSE；其余为六变两过线定位问句/双登记/双轨 id/单槽信箱合并语义/output_kind 三态/事件循环单核/128 分片理货/回程班车/拍口径/判停两地协作/断连反向 abort 三层接力/异步生成器拉动语义）；figures.json 同文件追加 L2-ch2（book:v3）；无精简版故无 interfaces。
- **伏笔对账**：ch02 无应埋无应收——pedagogy-plan 的 planted 集合 {1,4,5,7,9,11,13,16,19} 与 paid 集合 {9,15,22,27,30,34,36,37,38,40} 均不含 2，与 run-ledger `foreshadow_due:[]` 一致；foreshadow-v3.json 零改动，无出入。
- **图登记门禁**：`REPO2BOOK_INSTANCE=vllm python scripts/lint_figures_registered.py <章目录>` 显式传参通过。

## Why it matters

Part I 的动态主线索落定：十六站坐标系与「谁在哪个进程把数据变成了什么」定位问句、判停两地协作、单槽信箱合并语义、事件循环单核直觉、拍口径换算自此可被后 38 章按「前章已立」引用；ch01 漏登的 5 条基础术语（logits/hidden_states 等 Part V-VII 重度依赖项）补进账本，gap 审计不再误判。

## What to remember

1. **【Lead 决策项·linter 修复】** review issue 5：lint_dossier 的 `_source_root` 与 lint_fidelity 的 `_cite_source_root` 只认父目录名恰为 `artifacts`，对全部 artifacts-v3 章静默跳过锚点/逐字核验——建议放宽为 `startswith("artifacts")` 并对 ch02 补跑机械复核（本章 ~60 锚点人工核仅 1 失准且已修）。
2. **【Lead 决策项·机械误报】** review issue 9：lint_anchors 对 H1 标题「第 2 章」的 BARENUM warn 与 ch01 同型（系统性误报第二次出现；ch01 交付记录已登记，按 reviewer 契约全书保留一条标注即可）。
3. **【待办·图同步】** review issue 6：章节图副本 `artifacts-v3/ch02-request-lifecycle/diagrams/L2-ch2.{svg,png}`（07:53 渲染）滞后于真相源 `book/cartography/L2-ch2.{svg,png}`（08:59 重渲，minimap 引线层级/端点+一处标签基线微调，语义无差）——archivist 不动章产物，需 Lead/illustrator 执行同步覆盖 + manifest selfcheck_note 追记 + Read PNG 复核 minimap 区；FIGURE-SYSTEM 一张图原则要求两副本逐字节一致。
4. **【待办·可选批量小修】** 其余 negotiable 词级修正（十来个→七个、最右列、放行开关、demux 括注、stop_token/pooling 括注、SSE「你好」巧合设定、站 1/2 补 ch38 指针、「三班」计数收口、心跳口径统一、ZMQ 重述压缩、streaming_input 半句注、账本表 msgspec/msgpack 表注、model_output None 守卫半句、partial prefill 解码、徽标↔站号桥接、图注枚举补第四项）——均单行定点、不退整章，是否批量发起 writer 小修由 Lead 定；若修，explainer m8 表行号须同步勘误。
5. **补账惯例（新增）**：归档时发现前章 glossary 漏登，随本章登记补回——首现章如实标前章、释义尾注标「ch0N 归档漏登，ch0M 归档时补账」。
