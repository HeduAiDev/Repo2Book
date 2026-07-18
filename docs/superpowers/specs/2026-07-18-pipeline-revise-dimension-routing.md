# SDD 简报：chapter-pipeline revise 步按 issue.dimension 分流（figure→illustrator 子回环）

> 来源：exp-0716-1（watching→本次 book-retro 优先），样本 5+1。
> 性质：任务简报——**不直接改 `.claude/workflows/chapter-pipeline.js`**，由 Lead 另走 TDD
> （node --test 先行）落地。落地后发车**必须用 scriptPath**（named 发车吃旧缓存快照，
> 见 memory: workflow-byname-stale-snapshot）。

## 1. 问题

`chapter-pipeline.js` Phase E 的 revise 步（现 L362-370）把**全部** issues（含
figure-integration 维 blocking）只派 writer，而 writer 只有权改 `narrative/chapter.md`——
图侧矛盾永远修不掉，3 轮耗尽必 `review-exhausted` 升级 Lead，每次烧一整轮 pipeline + Lead
全程介入。

**样本 5+1**：
- ch05（本章地图缺失）、ch07（fig-block-ptr-pack 把 boundary_check/padding 错画进
  make_block_ptr 六参签名）、ch27（fig-ab-operand-structure 的 REVISE 方案写进
  figure-requests.json 却无人执行，blind_review 卡 PENDING→lint_diagrams BLOCKING）、
  ch29、ch33——figure-only 阻断项在回环内无解。
- **ch29 新变体（时序竞态）**：图 off-by-one 实际已被 pipeline 内建按需补图站在 loop 内修好
  （gen+PNG+盲审 PASS+lint_diagrams green），但 review-exhausted 仍触发——reviewer 记账
  滞后于补图站（verdict 基于旧渲染快照）。
- **ch39 纯文字环变体**：reviewer 快照滞后于补图站/writer 末轮，问题修好了也照样逃逸。
  ⇒ 分流设计**必须含「复验读最新稿」**，不只是「把图派给对的人」。

## 2. 设计要求

### 2.1 分流

revise 步把 blocking issues 按 `issue.dimension` 切两组：
- `figure-integration` 组 → **illustrator 子回环**（动 `diagrams/`）：修图 → 重渲 →
  Read PNG 自查 → **被修各图的盲审再验证** → manifest blind_review 回填 PASS。
  提示词形状可复用现有按需补图站（L261-279 fig-request/fig-blind 对）——把 issues 转成
  与 figure-requests done 条目同构的 {figure_id, problem, suggested_fix} 清单喂给它。
- 其余维 → **writer**（动 `narrative/`，现有 revise 提示词不变形）。
- 两组动不同文件，**可 `parallel([...])` 并行**；任一方 BLOCKED 沿用现有逃生舱语义。
- 图修完若涉及正文引用/图注变化，接一个 writer 微任务收尾（形状同 L280-285 fig-insert）。

### 2.2 难点一：blind_review=PENDING 时序

illustrator 修图后 manifest 该图 `blind_review` 必须回 PENDING 重盲审；若下一轮
figure-integration 维先跑 `lint_diagrams` 会被 PENDING 拦死。**「补图→盲审再验证」必须在
revise 步内闭合**：进入下一轮 review 前，被修各图已盲审 PASS、manifest 已回填。盲审员
prompt 沿用「只看 PNG+对应 spec/issue，禁看 gen 代码与正文」纪律。

### 2.3 难点二：保 resume 缓存约束

文件内有显式约束（L332-333 注释）：`DIMS/dimThunks/readerPrompt` **逐字不动**保 resume 缓存。
设计只允许**在 revise 段（L362 之后）新增代码**：新增 thunk/agent 调用用新 label
（如 `'revise-fig r'+r`、`'revise-fig-blind r'+r`），不改既有 label 与既有提示词模板字符串。
若评估后认定无法完全绕开（如需要改 revise 提示词骨架），**必须在实现 PR 里显式声明缓存失效
代价**：在途 run 不可 resumeFromRunId 跨过该步，需批次间隙上线。

### 2.4 难点三：复验读最新稿（ch29/ch39 变体）

现状缺陷是「终局判定基于修复前快照」。要求：
- 每轮 review 的 verdict 聚合必须发生在**该轮全部修复动作（writer+illustrator 子回环+盲审
  回填）完成之后**；下一轮各维 agent 天然重读最新文件，此性质要有测试锁住（见 §3 用例 4）。
- 第 3 轮修复完成后**不再有下一轮复核**就 return review-exhausted——这是 ch29/ch39 逃逸的
  直接机制。方案：3 轮耗尽前加一次**轻量终局复验**（只对上轮 blocking 项逐条核对最新稿/最新
  PNG 是否已解决，不开新维度全审；全清 → APPROVED，未清 → 才 review-exhausted）。复验 agent
  用新 label，不动 DIMS。

### 2.5 边界

- 只分流 `blocking` 项；non-blocking 照旧全量附给 writer 参考。
- `dimension` 字段缺失的 issue 一律按 writer 组处理（现 DIM_SCHEMA 的 issue 无 dimension
  字段——聚合处已有 reader/derivation 打标先例 L353-355；需在聚合时给 4 个真维度的 issues
  也打上 dimension 标，**在 `ok.flatMap` 之后新增映射**，不改 DIMS 数组本身）。
- 罕见图文耦合 issue（图和正文各改一半）：按 figure 组走，illustrator 修完由 §2.1 的 writer
  收尾微任务处理正文侧；不引入第三种路由。

## 3. node --test 用例草案（`.claude/workflows/lib/` 下抽纯函数 + 测试，参照 resolve-cfg.js 先例）

把可测逻辑抽成纯函数（如 `routeIssues(issues)` → {figIssues, textIssues}、
`aggregateVerdict(dims, fixups)`），workflow 里只留接线：

1. **路由**：混合 issues（fidelity+figure-integration+reader-comprehension，blocking 混
   non-blocking）→ figure-integration blocking 进 fig 组，其余 blocking 进 writer 组，
   non-blocking 不路由、随附 writer。
2. **纯图环**：全部 blocking 均 figure-integration → writer 组为空（不该派 writer 白跑）。
3. **dimension 缺失**：无 dimension 字段的 blocking → writer 组（不崩、不丢）。
4. **PENDING 时序**：模拟 illustrator 修图后 manifest=PENDING → 断言「盲审回填 PASS」步
   在「下一轮 lint_diagrams 判定」之前被调用（用调用序列 stub 断言顺序）。
5. **ch39 变体**：第 3 轮修复动作全部成功 → 断言终局复验被调用且以复验结果为准，而非
   直接 review-exhausted；复验未清 → 才 exhausted 且 issues 为**最新**未清项。
6. **缓存约束回归**：快照断言 DIMS 数组/readerPrompt 字符串与改前逐字节一致
   （防止实现顺手「优化」既有提示词）。

## 4. 验收指标（台账 exp-0716-1）

- review-exhausted 中 figure-only 占比 → 0；实现后新章零「图致逃逸」。
- ch29/ch39 型「修好了仍逃逸」→ 0（终局复验兜底）。
- 现 6 样本模式（ch05/ch07/ch27/ch29/ch33/ch39）在后续批次零复发。
