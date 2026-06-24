# ch02《请求的一生(鸟瞰追踪, meta 无精简版)》交付 APPROVED

- **Type**: delivery
- **Chapter**: 02
- **Date**: 2026-06-24
- **Timestamp**: 2026-06-24T18:45:04Z
- **Agents involved**: archivist
- **User present**: False
- **Tags**: 

## What happened

ch02 端到端低分辨率追踪章交付并归档。一条真实源码主线串起 generate()/HTTP 入口→渲染(chat template/tokenize)→InputProcessor(Stage1 出 EngineCoreRequest)→EngineCore(schedule+execute+sample+update 一拍)→OutputProcessor(Stage3 增量去 token 化+组装 RequestOutput)→流式/批量返回；每站点到为止并带 markdown 跳转链接指向放大它的专章。meta 章无精简版。四 linter 全过(structure/formulas/source_grounding/fidelity)。review APPROVED——5 条 issue 全 negotiable + non-blocking(grammar_output 悬空引用注释、persistent batch 译名、四步链接密度、玩具一token一字示意免责声明、render 站缺出口链接)。

## Why it matters

ch02 是全书地图章，为后续 31 章埋下指针式伏笔(f19-f26 已在 arc-map 登记 plant=ch02)；读者先有完整端到端时序图再钻细节。三段式异步(input→out-of-proc EngineCore→background output handler)如何把链串起来在此首次完整点名。

## What to remember

ch02 端到端低分辨率追踪章交付并归档。一条真实源码主线串起 generate()/HTTP 入口→渲染(chat template/tokenize)→InputProcessor(Stage1 出 EngineCoreRequest)→EngineCore(schedule+execute+sample+update 一拍)→OutputProcessor(Stage3 增量去 token ...
