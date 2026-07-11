# 经验回流系统(experience backflow)设计——替代 wisdom/knowledge

日期:2026-07-04
状态:待用户评审
动机:2026-07-04 实证审计——wisdom/knowledge 是半摆设:knowledge 只写不读(learn.py 写进被 .gitignore 的仓库根目录、workflow 零处读取指令、两实例事实混居);wisdom 只读不长(5/4 后基本冻结,promote 零次)。而真正有效的经验沉淀(视觉自查进 skill、wrapper 检查法进 RUNBOOK、终审修复进契约)全部绕开了这套系统。
治理模式(用户已定):**半自动·Lead 批准制**——自动发现,人把方向,agent 落笔。

---

## 1. 核心原则

- **P1 经验的家 = 下次必然被读的地方**,按强度分级:① 可机检 → linter(最强);② 角色行为 → 契约条款(spawn 必读);③ 操作方法 → skill/RUNBOOK;④ 仓库事实 → INSTANCE.md/trace。"文集"(wisdom/knowledge)不再是归宿。
- **P2 信号驱动,不例行抽取**:经验来自有信号的时刻——评审 issue、回环 ≥2 轮、盲审 FAIL、逃生舱、gap-audit cliff——不是每章收工的例行总结(噪音源)。
- **P3 重复才是经验**:同类问题 **≥2 章出现**才成候选;单发是章级问题,已被该章评审消化,不回流。
- **P4 生效可验证**:每条落地经验登记台账,下次复盘对比复发率;复发 = 沉淀无效 → 升级落点(契约→linter)。
- **P5 契约是敏感资产**:curator 只许改 Lead 批准清单内的目标文件与内容;禁动 workflow 编排逻辑;linter 类改动走 TDD。

## 2. 部件一:信号持久化(run-ledger)

现状:评审 issue 已持久(reviews/review-report.json,32 章 × ~19 条,含 problem/suggested_fix/rationale/dimension);但回环轮数、盲审失败史、逃生舱原因只活在 workflow 返回值里。

改法:`chapter-pipeline.js` 与 `chapter-retrofit.js` 的 Archive 阶段,把 workflow 已持有的过程变量注入归档 agent,原样写 `{chapter_dir}/reviews/run-ledger.json`:

```
{"chapter_id", "kind": "code|primer|retrofit",
 "impl_test_rounds": <int>, "impl_test_ledger": [<每轮失败摘要>],
 "write_review_rounds": <int>,
 "blind_rounds": <int>, "blind_failures": [{figure_id, problem, suggested_fix}(历轮累计)],
 "escalated": null | {stage, reason}}
```

逃生舱路径(早退 return)不经过 Archive——由 Lead 在处理升级时补记该章 run-ledger(RUNBOOK 逃生舱表加一行提醒);不苛求全自动。

## 3. 部件二:复盘 workflow(book-retro.js)

发车:批次/Part 收尾时 Lead 手动 `Workflow({name:"book-retro", args:{instance, chapters:[slug]|null, date}})`。

- **挖掘(并行,sonnet)**:每章一个 agent,只读该章 reviews/(review-report + run-ledger + retrofit-review 若有)与 book/audits/ 相关条目,输出该章信号摘要:`{issues_by_dimension, rounds, blind_failures, escalation, notable:[{signal, evidence(引用原文), candidate_rule(一句话)}]}`。
- **聚类(单 agent,opus)**:合并全批信号,按 root cause 聚类;**≥2 章重复才成候选**;reader 顾问 issue 降权(仅当 ≥4 章重复才候选)。对照 `docs/superpowers/experience-ledger.md`(部件四):已落地 pattern 再现 → 标 `recurrence:true` 并建议升级落点。
- **产物**:`instances/<x>/book/retro/retro-<date>.json`:

```
{"candidates": [{
  "id": "exp-<date>-<n>",
  "pattern": "一句话:什么问题反复出现",
  "occurrences": [{"chapter", "evidence": "引用 issue/ledger 原文"}],
  "root_cause": "为什么会反复",
  "target": "linter:<脚本名>|contract:<role>|skill:<name>|runbook|instance",
  "draft_patch": "具体条款文字 / linter 规则描述(可直接落笔的草案)",
  "expected_effect": "下批次哪个可数指标应下降(如:盲审 FAIL 率/figure 维 issue 数)",
  "recurrence": false | {"ledger_id": "exp-…", "现落点": "…", "建议升级": "…"}
}], "stats": {"chapters", "issues_total", "rounds_avg", "blind_fail_total", "escalations"}}
```

- workflow 返回候选摘要给 Lead;**不做任何修改动作**。

## 4. 部件三:批准与落笔(curator)

1. Lead 逐条批(批准/改落点/改措辞/驳回),把批准清单(含最终 target 与 patch 文字)交 curator。
2. **curator = 一次性 agent(sonnet)**,契约要点(新文件 `.claude/agents/curator.md`,≤40 行):只许 Edit 批准清单列出的目标文件;契约/skill/RUNBOOK 类 → 定点插入批准文字(全角标点,融入原文格式);linter 类 → **不直接改**,产出 SDD 任务简报交 Lead 走 TDD 小任务;INSTANCE/trace 类 → 追加条目。落笔后逐条回报 diff 位置。
3. 每条落地即写台账(部件四);同一 commit 内完成。

## 5. 部件四:经验台账(生效验证闭环)

`docs/superpowers/experience-ledger.md`,一条一行:

```
| id | 日期 | pattern | 落点(文件) | 针对指标 | 状态 |
| exp-0704-1 | 2026-07-04 | 图注描述画面不给结论 | .claude/skills/svg-diagram/SKILL.md | figure 维 issue 数 | active |
```

- retro 聚类 agent 每次对照台账:active 条目的 pattern 再现 → `recurrence` 标记 + 升级建议(契约→linter);连续两次复盘未再现 → Lead 可标 `proven`。
- 台账是人读的(Lead 复盘时的 meta 视图),不做机器强校验。

## 6. 退役清单(与部件一同批实施)

- 8 个角色契约删"收工后 `learn.py extract`"行(每章省 1-2 次 agent 调用);explainer/illustrator 契约同。
- `scripts/learn.py` 移入 `scripts/attic/`(保留代码考古,不再被引用)。
- 仓库根 `knowledge/`(untracked):十余文件由一次性 agent 按内容分拣——仓库事实并入对应 `instances/<x>/INSTANCE.md` 或 trace,过程性内容丢弃;分拣报告给 Lead 后删除目录,.gitignore 该行移除;`instances/*/knowledge/` 目录退役删除。
- `wisdom/`:一次性 agent 逐条盘点四个文件,仍有效且未被契约覆盖的条款 → 作为首批"经验候选"走部件三流程(Lead 批准落进对应契约/skill);目录随后移入 `docs/attic/wisdom/` 冻结。
- 文档同步:CLAUDE.md「记忆体系」段改写(wisdom/knowledge → 经验回流:run-ledger/retro/台账);README 目录结构与协作节对应行;RUNBOOK 加「复盘发车」节 + 逃生舱表补 run-ledger 提醒;`repo2book.json` 若引用 knowledge 路径则清理。

## 7. 验收

- 机械:run-ledger 在新完成章出现且 schema 合法(新增 `lint_run_ledger` 不必要——由 retro 挖掘 agent 容错读取;不加 linter,YAGNI);retro workflow 语法过 async-wrapper 检查;契约中 learn.py 引用为 0。
- 功能闭环(首跑即验):对已完成的 Part VIII + 回修批次跑一次 book-retro → 产出候选清单 → Lead 批 ≥1 条落笔 → 台账建档。预期它至少能自动再发现我们已手工沉淀过的模式(如公式锚缺失/图注写法),作为挖掘质量的基准对照。

## 8. 风险与对策

- **聚类产出噪音多** → ≥2 章门槛 + reader 降权 + Lead 批准制兜底;首跑用已知模式作基准校验挖掘质量。
- **curator 写坏契约** → 只许动批准清单文件;落笔后 Lead 抽查 diff;契约在 git 里,可回滚。
- **run-ledger 增加归档负担** → 数据是 workflow 现成变量,注入提示词原样落盘,无新计算。
- **台账膨胀** → 一行一条;proven/retired 条目每年归档一次(人工,不建机制)。
