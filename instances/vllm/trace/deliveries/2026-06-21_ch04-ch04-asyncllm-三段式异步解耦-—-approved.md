# ch04 AsyncLLM 三段式异步解耦 — APPROVED

- **Type**: delivery
- **Chapter**: 04
- **Date**: 2026-06-21
- **Timestamp**: 2026-06-21T07:01:20Z
- **Agents involved**: archivist
- **User present**: False
- **Tags**: ch04, delivery, async-llm, pilot, approved

## What happened

首章试点交付并归档。4 linter 全过（fidelity/structure/grounding 全绿；formulas 仅 1 条非阻塞风格提示 chapter.md:410-411，3 inline 公式同段，已判 WONTFIX）；19/19 测试全绿（host 纯 in-process，pytest-asyncio，无需容器）；行为对照真实 pin f3fef123 逐处验证。叙事 757 行内嵌真源码 + Roadmap。Stage2 IPC -> in-process stub 为 dossier 批准的边界替换，IPC 留 ch07。已登记 11 条精简版接口入 Bible interfaces.json；伏笔 f1/f2/f3 已埋并在 arc-map 登记（payoff ch08/ch07/ch08，status=open）。本章为早章无应回收伏笔。

## Why it matters

验证 v2 源码解读体系是否根除旧'writer 讲杜撰代码'脱节——结果：档案即真相源 + 只做减法 + 自包含内嵌真源码三支柱跑通，闸门有效。

## What to remember

首章试点交付并归档。4 linter 全过（fidelity/structure/grounding 全绿；formulas 仅 1 条非阻塞风格提示 chapter.md:410-411，3 inline 公式同段，已判 WONTFIX）；19/19 测试全绿（host 纯 in-process，pytest-asyncio，无需容器）；行为对照真实 pin f3fef123 逐处验证。叙事 7...
