# ch37 图面脚手架泄漏 3 处 —— 待回修(2026-07-21 立)

发现经过:写 triton-ascend ch09 时，独立盲审抓出 ch09 图上一句作图者自证话术
(`halo 代价(与正文/explainer m15 逐字一致)`)泄漏内部产物名。据此加固
`scripts/lint_diagram_scaffolding.py`(补「内部产物名的裸用法」)，全语料 881 张 SVG
oracle 复扫时**顺带照出了本章**。

## 三处泄漏(均为 BLOCKING，`lint_diagram_scaffolding` 现已阻断)

| 图 | 图面渲染的原文 | 泄漏物 |
|---|---|---|
| `fig-m6-loop-statetable.svg` | `锚点 A 出发；base_logits 只算 1 次，每步用 prev token 生成偏置逐位修正（traces/m6_sequential.out 复现）` | 内部 trace 文件路径 |
| `fig-m8-greedy-earlystop.svg` | `玩具 c=[.9,.8,.5,.4]（traces/m8_scheduler.out 复现）——本 PR #46995 快照无此调度器代码` | 内部 trace 文件路径 |
| `fig-m8-granularity-landed.svg` | `dossier honest_gaps 第 2 条` | 内部产物名 + 内部字段名 |

违反 **HARD RULE 3(零脚手架泄漏)**:正文是正式出版物，不得出现内部文件(dossier/traces/
impl-notes)与内部编号。读者手里没有这些文件，"（traces/xxx.out 复现）"对读者是无意义的。

## 为什么一直没被发现(两层原因，都已处置)

1. **`lint_diagram_scaffolding` 不在 CLAUDE.md 的例行 `--all` 清单里**——例行只跑
   `lint_punct` / `lint_anchors` / `lint_diagram_geometry` / `lint_chapter_map` 四个 `--all`。
   ⇒ 已把它补进 CLAUDE.md 的质量闸门清单。
2. **`--all` 只扫「活动实例」**(当前是 triton-ascend)，而本章在 vllm-ascend
   ⇒ 即使有人跑了 `--all` 也照不到。这是 exp-2026-07-20-03 同一病灶的残留面
   (那次修的是「显式传路径时按路径定实例」，`--all` 的口径没动)。
   ⇒ 已记 exp-2026-07-21-14；`--all` 是否应扫全部实例，留作后续决策(改动面较大，
   会把其它实例的存量问题一次性变红，需先评估存量)。

**注意**:前两处(`traces/…`)**改前的旧正则就能抓到**，也就是说本章在定稿时
这条门禁要么没跑、要么跑了没处置——不是 linter 漏检。第三处(`dossier`)才是本次加固新增的。

## 处置建议

派 illustrator 定点修三张图的 **gen 脚本**(不要手改 SVG)再重渲染:

- 两处 `（traces/xxx.out 复现）` → 删掉括号，或改成读者可理解的出处措辞
  (如「本章参考实现实测」——ch09 的页脚就是这个写法)。
- `dossier honest_gaps 第 2 条` → 改成该条 gap 的**内容本身**(读者读得懂的一句话)，
  不要指向内部文件的第几条。

⚠️ 改完须 **独立盲审**(非作图者自审)——ch09 这次正是作者自审记了 PASS、
把自己写的自证话术漏了过去，才由独立盲审抓出来的。

⚠️ 本章属 **vllm-ascend** 实例(非当前 active)。跑 linter 时显式传章节路径即可
(exp-2026-07-20-03 已修:按路径定实例，不必设 `REPO2BOOK_INSTANCE`)。
