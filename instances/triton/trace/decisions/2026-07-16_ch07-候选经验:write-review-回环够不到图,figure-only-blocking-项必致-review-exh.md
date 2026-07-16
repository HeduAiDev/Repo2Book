# 候选经验:write-review 回环够不到图,figure-only blocking 项必致 review-exhausted

- **Type**: decision
- **Chapter**: 07
- **Date**: 2026-07-16
- **Timestamp**: 2026-07-16T15:43:10Z
- **Agents involved**: archivist
- **User present**: False
- **Tags**: experience-candidate, pipeline, figure-integration, review-exhausted, book-retro

## What happened

ch07 与更早的 ch05 各出现一次 review 阶段升级 Lead 的逃生舱,表面原因不同(ch05=chapter-map 站点被跳过导致的缺图;ch07=fig-ch07-block-ptr-pack 画法与 make_block_ptr 真实源码签名/图自身 alt 文本矛盾,3 轮 write↔review 回环耗尽仍未收敛),但共同根因相同:write↔review 有界回环的『writer 收 issue→改 narrative→reviewer 再核』这条链路里没有 illustrator 角色,一旦 reviewer 判定的 blocking 项落在图本身(而非正文文字),writer 无论怎么改 narrative 都无法让图变化,回环必然在轮数上限耗尽后判 review-exhausted、升级 Lead。ch07 这次 Lead 核实后手工派 illustrator 定点重画 2 图 + 补本章地图,过盲审后手工归档收口。

## Why it matters

目前只有 2 次样本(ch05/ch07),按 CLAUDE.md 的经验回流规则『≥2 章重复才算』,这已经达到候选经验升级的门槛,应提交 book-retro 复盘评估是否要把它落成正式经验条目——候选落点:(a) reviewer.md 的 figure-integration 维度评审契约里加一条『若 blocking 问题的根因在图(非正文可修),直接标记 figure_only:true,workflow 应路由给 illustrator 而非空转 writer 回环』;(b) chapter-pipeline.js 的 Review 阶段加一条子分支:figure-integration 维度出现 figure_only blocking 时不消耗 write↔review 轮数预算,直接触发一次 illustrator 修图+重盲审,再回 Review;(c) 或至少在 Review 阶段升级判定里区分『figure-only 阻断』与『真正 write↔review 都救不了的路线错』两种 review-exhausted,给 Lead 更精确的诊断起点(本条已在两次实践里由 Lead 人工诊断出这个区分,值得写进流程使其自动化)。

## What to remember

候选经验(2/2 samples,达到升级门槛):figure↔narrative/图缺陷类 blocking 问题,write↔review 回环触不到图,必然耗尽升级。下次(第3次)复现时,应由 Lead 直接批准把此模式落成正式经验条目——落点候选是 reviewer.md 契约新增 figure_only 标记 + chapter-pipeline.js 加 illustrator 子回环路由,而不是每次都靠 Lead 人工诊断。
