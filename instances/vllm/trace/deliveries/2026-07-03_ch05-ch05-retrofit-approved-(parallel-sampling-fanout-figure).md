# ch05 retrofit APPROVED (parallel-sampling-fanout figure)

- **Type**: delivery
- **Chapter**: 05
- **Date**: 2026-07-03
- **Timestamp**: 2026-07-03T12:45:37Z
- **Agents involved**: archivist
- **User present**: False
- **Tags**: retrofit, figure, parallel-sampling-fanout

## What happened

存量外科回修完成：ch05-input-processing 针对 parallel-sampling-fanout 机制补图 02-request-id-and-fanout.png（重绘旧版重叠问题，行距>=14px，盲审 PASS，与 explainer.json claim/numbers 完全相符）。retrofit-review.json 判定 APPROVED，仅 1 条 non-blocking/negotiable 建议（可选补充 Big-O 记法与量化描述对齐，不影响事实准确性）。

## Why it matters

巩固 fan-out 机制的可视化真相源，为后续涉及并行采样(n>1)的章节提供可复用/链接的插图，避免重复画图；figures.json 登记后建立机制→图→章的跨章可追溯性。

## What to remember

ch05 retrofit APPROVED；新图 02-request-id-and-fanout 已登记入 bible/figures.json（mechanism_id=parallel-sampling-fanout）；review 唯一 issue 是 non-blocking 的 Big-O 记法建议。
