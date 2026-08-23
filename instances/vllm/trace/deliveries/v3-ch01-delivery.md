# v3 ch01《一张图看懂 vLLM v1》交付归档（APPROVED）

- **Type**: delivery（v3 Archive 站，chapter-pipeline-v3 首章）
- **Chapter**: v3 ch01 · Part I 全景与读法 · kind=meta（panorama_l0_no_companion，无精简版）
- **Pin**: vLLM v0.27.1（6e448d0ea）；行号基线即此版
- **Date**: 2026-08-16 · **Agents**: pipeline 各站（analyst→…→reviewer）+ archivist（本记录）
- **Verdict**: APPROVED，11 条 issue（2 blocking / 9 non-blocking），全文见 `artifacts-v3/ch01-vllm-v1-in-one-map/reviews/review-report.json`

## What happened

- **回环**：write↔review 3 轮、L2/图 3 轮、盲审 1 轮零失败（`reviews/run-ledger.json`）；meta 章无精简版（impl_test_rounds=0）。
- **blocking 修复核实在稿**（archivist 逐条对照正文）：L39 进程数归属拆开（tp=4=6 进程挂官方算例、单卡=2 进程挂源码 UniProcExecutor）；L151 构造段「前两条注释一字不差、底部两处换说法」如实化；L181 make_client 头部就地内嵌 raise NotImplementedError；L292 woosuk 注释「沿用至今、中途两笔扩写」如实化；L21 L1 图注补站号读法；L355 按前端分装的理由补齐；L373 P/D 字序对齐（P=消化、D=生成）。
- **bible 登记（v3 侧车，不动 v2 封版账）**：glossary-v3.json +28 术语；concepts-v3.json +11 概念（pedagogy-plan introduces 四条 + 正文实际建立的七条）；foreshadow-v3.json 埋 F1「EngineCore 五拍」/F9「L0 图逐章点亮」均 done:true，收款 F1→ch09、F9→ch40 未到期；figures.json 同文件追加 L0-architecture / L1-partI（带 book:v3 标记）；无精简版故无 interfaces 登记。
- **图登记门禁**：`REPO2BOOK_INSTANCE=vllm python scripts/lint_figures_registered.py <章目录>` 显式传参通过（review issue 6 收口）。

## Why it matters

v3 首章落定全书骨架账本：L0 全图 + 逐 Part 行程表 + 五拍/token 记账/三件套等基础概念自此可被后 39 章按「前章已立」引用；伏笔 F1/F9 的埋点状态进入显式对账文件，后续章 gap 审计有据可查。

## What to remember

1. **active_instance=triton-ascend**：凡 bible/trace 相关脚本（lint_figures_registered、archivist.py…）必须 `REPO2BOOK_INSTANCE=vllm` 覆盖，否则路径解析到错实例；`--all`/无参模式对 vllm v3 章静默漏检。
2. **v3 侧车账本制**：glossary/concepts/foreshadow 写 `*-v3.json`，仅 figures.json 同文件追加（条目带 `book:"v3"`）。
3. **遗留 Lead 决策项**（review issue 7/11、8）：lint_chapter_map `--require` 对 v3 章误报（v2 chapter-map 已退役属 v3 设计，需豁免）；lint_anchors 对 H1 标题「第 N 章」的 BARENUM warn 属机械误报。
4. **工具坑**：archivist.py 与 lint_figures_registered.py 在 GBK 控制台因未指定 encoding/输出 ✓ 崩溃——跑时加 `PYTHONIOENCODING=utf-8`；`python3` 是 WindowsApps 桩（exit 49），用 `python`。
