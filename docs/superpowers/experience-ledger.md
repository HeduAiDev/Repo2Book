# 经验台账(experience ledger)

> 经验回流系统的生效验证账本:每条 = 一次"发现→批准→落笔"。retro 复盘时对照本表——
> pattern 复发 = 沉淀无效 → 升级落点(契约→linter);连续两次复盘未复发 → 标 proven。
> 详见 docs/superpowers/specs/2026-07-04-experience-backflow-design.md。

| id | 日期 | pattern | 落点(文件) | 针对指标 | 状态 |
|---|---|---|---|---|---|
| exp-0705-1 | 2026-07-05 | primer 章含 engineering-only 机制时 lint_dossier 强制 paper_origin 误伤(ch32 实例) | scripts/lint_dossier.py(paper_origin_note 豁免) | lint_dossier 对 primer 章误报数 | active |
| exp-0705-2 | 2026-07-05 | lint_paper_grounding 只读 paper.md,多论文包(paper-mtp.md 等)小节核对误报(ch33 评审发现) | 待落:linter 多论文支持(retro 候选) | paper_ref 误报数 | pending |
