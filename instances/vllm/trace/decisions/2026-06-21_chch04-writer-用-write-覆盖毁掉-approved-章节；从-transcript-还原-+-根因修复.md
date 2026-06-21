# writer 用 Write 覆盖毁掉 APPROVED 章节；从 transcript 还原 + 根因修复

- **Type**: bug
- **Chapter**: ch04
- **Date**: 2026-06-21
- **Timestamp**: 2026-06-21T08:00:39Z
- **Agents involved**: archivist
- **User present**: False
- **Tags**: 

## What happened

ch04 定点小修时 writer 误用 Write 整文件覆盖，毁掉 758 行 APPROVED chapter.md（仅剩一段）。文件未提交无 git 兜底。从 workflow transcript 重放 Write+11 Edit 精确还原。根因：writer.md tools 只有 Write 没有 Edit，被迫覆盖。

## Why it matters

修复：①6 角色 tools 全加 Edit ②writer.md 立铁律 Edit-only 改已有章节 ③通过即提交兜底 ④逃生舱按设计生效(writer 诚实升级未伪造)。教训：通过的成果立即 commit；改文件优先 Edit。

## What to remember

ch04 定点小修时 writer 误用 Write 整文件覆盖，毁掉 758 行 APPROVED chapter.md（仅剩一段）。文件未提交无 git 兜底。从 workflow transcript 重放 Write+11 Edit 精确还原。根因：writer.md tools 只有 Write 没有 Edit，被迫覆盖。...
