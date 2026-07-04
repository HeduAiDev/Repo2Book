export const meta = {
  name: 'chapter-retrofit',
  description: '存量章节外科回修：逐机制体检→增量素材→补图/换错图→定点改写算法段→缩编评审→归档（禁整章重写）',
  phases: [
    { title: 'Diagnose', detail: '读章+Read 全部 PNG，逐机制体检；免修即终止' },
    { title: 'Explain', detail: '只对 flagged 机制产经验证素材' },
    { title: 'Illustrate', detail: '补缺图/重绘错图：视觉自查+盲审' },
    { title: 'PatchWrite', detail: 'writer 只许定点 Edit 算法段与图引用' },
    { title: 'Review', detail: 'algorithm-pedagogy + figure-integration 两维门控' },
    { title: 'Archive', detail: 'trace 记 retrofit + bible figures.json 登记' },
  ],
}

// args 注入不可靠时的兜底配置（与 chapter-pipeline 同款约定）
const CFG = {
  chapter_id: 'ch16',
  slug: 'ch16-kv-cache-manager',
  instance: 'vllm',
  highlight: 'kv-cache',
  repo_root: '/mnt/e/Laboratory/Repo2Book',
}
const A = (typeof args !== 'undefined' && args && args.chapter_id) ? args : CFG
// 模型分配（spec §7：全流水线 opus/sonnet，不继承主会话模型；args.models 可覆盖）
const MODELS = Object.assign(
  { diagnose: 'opus', explain: 'opus', illustrate: 'sonnet', blind: 'sonnet', patch: 'opus', review: 'sonnet', archive: 'sonnet' },
  A.models || {})
const REPO = A.repo_root || '/mnt/e/Laboratory/Repo2Book'
const INST = A.instance || 'vllm'
const CH = REPO + '/instances/' + INST + '/artifacts/' + A.slug
const HL = A.highlight || ''

const ESC = '\n\n**逃生舱（重要）**：发现体检单/素材/路线是错的——不要硬着头皮做。立即返回 status="BLOCKED"，blocker_reason 写清「哪里错 + 建议怎么改」，workflow 中止升级 Team Lead。'

function head(role) {
  return [
    '你的角色契约在 ' + REPO + '/.claude/agents/' + role + '.md —— **先读它**，严格遵守其中所有铁律。',
    '本章目录（绝对路径）：' + CH,
    '本章：' + A.chapter_id + '（存量外科回修——只动图和算法段，不重写章节主体）',
    '',
  ].join('\n')
}

const STATUS_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['status', 'note'],
  properties: { status: { type: 'string', enum: ['OK', 'BLOCKED'] }, note: { type: 'string' }, blocker_reason: { type: 'string' } },
}
const DIAG_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['flagged_count', 'summary'],
  properties: { flagged_count: { type: 'number' }, summary: { type: 'string' } },
}
const BLIND_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['all_pass', 'failures'],
  properties: {
    all_pass: { type: 'boolean' },
    failures: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['figure_id', 'problem', 'suggested_fix'],
      properties: { figure_id: { type: 'string' }, problem: { type: 'string' }, suggested_fix: { type: 'string' } } } },
  },
}
const DIM_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['pass', 'issues'],
  properties: {
    pass: { type: 'boolean' },
    issues: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['problem', 'suggested_fix', 'rationale', 'negotiable', 'blocking'],
      properties: { problem: { type: 'string' }, suggested_fix: { type: 'string' }, rationale: { type: 'string' }, negotiable: { type: 'boolean' }, blocking: { type: 'boolean' } } } },
  },
}

// ---------- Phase 1: Diagnose（体检不动刀；免修即终止） ----------
phase('Diagnose')
const diag = await agent(
  head('reviewer') +
  '任务：**体检本章，不做任何修改**（除按下述补 dossier 的 mechanisms 字段外）。\n' +
  '① 读 ' + CH + '/narrative/chapter.md 与 ' + CH + '/dossier/dossier.json。若 dossier 无 mechanisms 字段：**只登记需动工的机制**（体检判定 depth=shallow 或 figure!=ok 的），Edit 加 mechanisms 字段、不动其他字段；已合格的机制本次不登记（外科范围外，避免账本要求全书追溯）。kind 如实填，**kind=algorithm 的登记项一律 needs_worked_example=true**（lint_dossier 规则——重绘算法图也需要经运行/推演验证的数字）；needs_figure 按体检结果填。\n' +
  '② 逐机制评深度：三层递进（直觉/机制含数值推演/源码）齐吗？不变量有论证吗？→ depth: ok|shallow。\n' +
  '③ 用 Read 逐张打开 ' + CH + '/diagrams/ 的内容 PNG（roadmap 除外）**亲眼看**：该机制有图吗？图与正文数字/源码一致吗？可读吗？→ figure: ok|missing|wrong。diagrams/ 若有 svg/png 而无对应 gen_*.py，记 action="rebuild-gen"。\n' +
  '④ 写 ' + CH + '/retrofit/retrofit-plan.json：{mechanisms:[{id,name,depth,figure,evidence,actions:[]}]}——每条判定必须带 evidence（引用正文行/图名）。\n' +
  '返回 flagged_count（depth=shallow 或 figure!=ok 的机制数）与 summary（一句话体检结论）。' + ESC,
  { schema: DIAG_SCHEMA, label: 'diagnose', phase: 'Diagnose', agentType: 'general-purpose', model: MODELS.diagnose }
)
if (!diag) return { chapter: A.chapter_id, escalated: 'diagnose-failed', stage: 'Diagnose' }
if (diag.flagged_count === 0) { log('体检通过，本章免修'); return { chapter: A.chapter_id, verdict: 'CLEAN', summary: diag.summary } }
log('体检：' + diag.flagged_count + ' 个机制需动工 —— ' + diag.summary)

// ---------- Phase 2: Explain（只对 flagged 机制产素材） ----------
phase('Explain')
const expl = await agent(
  head('explainer') +
  '任务：读 ' + CH + '/retrofit/retrofit-plan.json，**只**对 flagged 机制（depth=shallow 或 figure!=ok）产出教学素材，写入 ' + CH + '/explainer/explainer.json（已存在则增量 Edit 合并）；trace 存 ' + CH + '/explainer/traces/。\n' +
  '本章有 implementation/ 则跑它取 trace（trace_source="run"）；没有则 trace_source="manual" 并写 manual_reason。figure!=ok 的机制补 figure-spec（重绘错图的 spec 里写清旧图错在哪）。\n' +
  '完成后自跑 `python3 ' + REPO + '/scripts/lint_explainer.py ' + CH + '` 无 BLOCKING。返回 status/note。' + ESC,
  { schema: STATUS_SCHEMA, label: 'explain', phase: 'Explain', agentType: 'general-purpose', model: MODELS.explain }
)
if (expl && expl.status === 'BLOCKED') return { escalated: 'explain', stage: 'Explain', reason: expl.blocker_reason }

// ---------- Phase 3: Illustrate（补图/换图，视觉自查 + 盲审） ----------
let blindV = null
let blindLedger = []
let blindHistory = []
for (let b = 1; b <= 3; b++) {
  phase('Illustrate')
  const ill = await agent(
    head('illustrator') +
    '任务：按 ' + CH + '/explainer/explainer.json 的 figure_specs 补缺图/重绘错图到 ' + CH + '/diagrams/（gen_<figure_id>.py + svg + png，登记/更新 figure-manifest.json）。被替换的旧图：其 svg/png/gen 一并删除，正文引用由 PatchWrite 阶段更新。retrofit-plan 里 action=rebuild-gen 的既有图：重建其 gen 脚本（输出须与现图一致，Read PNG 对照）。\n' +
    '每张新图强制：渲染 → Read PNG 亲眼看 → 六项自查全真才登记。\n' +
    (blindLedger.length ? '上一轮盲审 FAIL，必须修复：\n' + blindLedger.join('\n') + '\n' : '') +
    '完成后自跑 `python3 ' + REPO + '/scripts/lint_diagram_geometry.py ' + CH + '/diagrams/*.svg` 无问题。返回 status/note。' + ESC,
    { schema: STATUS_SCHEMA, label: 'illustrate r' + b, phase: 'Illustrate', agentType: 'general-purpose', model: MODELS.illustrate }
  )
  if (ill && ill.status === 'BLOCKED') return { escalated: 'illustrate', stage: 'Illustrate', round: b, reason: ill.blocker_reason }
  blindV = await agent(
    '你是插图盲审员。**只准看**：' + CH + '/diagrams/figure-manifest.json 列出的每张 PNG（用 Read 打开图片）+ ' + CH + '/explainer/explainer.json 对应 figure_spec。禁止看 gen 代码与正文。\n' +
    '逐张：① 只看图复述论点；② 对照 spec.claim——不符 = FAIL；③ 图上数字逐个核 spec.numbers——不符 = FAIL；④ 明显不可读 = FAIL。verdict/notes 用 Edit 回填 manifest 的 blind_review。\n' +
    '返回 all_pass 与 failures（figure_id + problem + suggested_fix）。',
    { schema: BLIND_SCHEMA, label: 'blind-review r' + b, phase: 'Illustrate', agentType: 'general-purpose', model: MODELS.blind }
  )
  blindHistory.push({ round: b, failures: (blindV && blindV.failures) || [] })
  if (blindV && blindV.all_pass) break
  blindLedger = ((blindV && blindV.failures) || []).map(function (f) { return '[' + f.figure_id + '] ' + f.problem + ' → ' + f.suggested_fix })
  log('盲审第 ' + b + ' 轮 FAIL：' + blindLedger.length + ' 张图打回')
}
if (!blindV || !blindV.all_pass) return { chapter: A.chapter_id, escalated: 'blind-review-exhausted', stage: 'Illustrate', failures: (blindV && blindV.failures) || [] }

// ---------- Phase 4/5: PatchWrite + 缩编 Review（有界回环 2 轮） ----------
const DIMS = [
  'algorithm-pedagogy（逐 flagged 机制对账：直觉/数值推演表带 trace 标记/不变量/量化；先跑 lint_trace_consistency）',
  'figure-integration（先跑 lint_diagrams；逐张 Read PNG：新图被正文引用且在机制附近/图注给结论/数字一致/被删旧图无残留引用）',
]
let reviewV = null
let issuesForWriter = []
let reviewRounds = 0
for (let r = 1; r <= 2; r++) {
  phase('PatchWrite')
  reviewRounds = r
  const pw = await agent(
    head('writer') +
    '任务：**外科手术式**修改 ' + CH + '/narrative/chapter.md——**只许 Edit 定点修改** flagged 机制的算法段与图引用处。\n' +
    '⛔ 禁止：整章重写 / 用 Write 覆盖 / 移动章节结构 / 改非算法叙事 / 删既有标题锚点。\n' +
    '要做：按 ' + CH + '/explainer/explainer.json 素材加深讲解（直觉→机制→源码三层，怎么衔接由你）；数值推演表进正文（表格前一行 `<!-- trace: <mechanism_id> -->`，数字不许改）；更新图引用（新图 ../diagrams/<id>.png，被替换旧图的引用与图注一并更新）。\n' +
    (issuesForWriter.length ? '上轮评审 issue（逐条采纳或带理由反驳）：\n' + JSON.stringify(issuesForWriter) + '\n' : '') +
    '完成后自跑 lint_trace_consistency / lint_anchors / lint_chapter_structure / lint_formulas / lint_punct 无 BLOCKING。返回 status/note。' + ESC,
    { schema: STATUS_SCHEMA, label: 'patch-write r' + r, phase: 'PatchWrite', agentType: 'general-purpose', model: MODELS.patch }
  )
  if (pw && pw.status === 'BLOCKED') return { escalated: 'patch-write', stage: 'PatchWrite', round: r, reason: pw.blocker_reason }
  phase('Review')
  const dims = await parallel(DIMS.map(function (dim) {
    return function () {
      return agent(
        head('reviewer') +
        '任务：**只**从「' + dim + '」维度评审 ' + CH + '/narrative/chapter.md（对照 retrofit-plan.json 与 explainer.json）。每条 issue 给 suggested_fix + rationale + evidence，标 negotiable/blocking。该维度无 blocking issue → pass=true。',
        { schema: DIM_SCHEMA, label: 'review:' + dim.slice(0, 9) + ' r' + r, phase: 'Review', agentType: 'general-purpose', model: MODELS.review }
      )
    }
  }))
  const ok = dims.filter(Boolean)
  if (ok.length < DIMS.length) return { chapter: A.chapter_id, escalated: 'review-agents-failed', stage: 'Review', round: r }
  const issues = ok.flatMap(function (d) { return d.issues || [] })
  if (ok.every(function (d) { return d.pass }) && issues.filter(function (i) { return i.blocking }).length === 0) {
    reviewV = { verdict: 'APPROVED', issues: issues }
    break
  }
  issuesForWriter = issues
  reviewV = { verdict: 'REVISE', issues: issues }
  log('retrofit 评审第 ' + r + ' 轮 REVISE，回 writer 定点修')
}
if (!reviewV || reviewV.verdict !== 'APPROVED') return { chapter: A.chapter_id, escalated: 'review-exhausted', stage: 'Review', issues: (reviewV && reviewV.issues) || [] }

// ---------- Phase 6: Archive ----------
phase('Archive')
const runLedger = JSON.stringify({
  chapter_id: A.chapter_id, kind: 'retrofit',
  flagged: diag.flagged_count,
  impl_test_rounds: 0, impl_test_ledger: [],
  write_review_rounds: reviewRounds,
  blind_rounds: blindHistory.length, blind_failures: blindHistory,
  escalated: null,
})
const arch = await agent(
  head('archivist') +
  '任务一：把这个 review 对象**原样**写入 ' + CH + '/reviews/retrofit-review.json：\n' + JSON.stringify(reviewV) + '\n' +
  '任务一b：把这个 run-ledger 对象**原样**写入 ' + CH + '/reviews/run-ledger.json：\n' + runLedger + '\n' +
  '任务二：在 bible 的 figures.json 登记本章新图（{mechanism_id, figure_id, chapter_id: "' + A.chapter_id + '", claim}，文件在 instances/' + INST + '/book/bible/figures.json，不存在则创建）。\n' +
  '任务三：`python3 ' + REPO + '/scripts/archivist.py record --type delivery` 记 retrofit 交付并更新 trace/state.json。返回一句话状态。',
  { label: 'archive', phase: 'Archive', agentType: 'general-purpose', model: MODELS.archive }
)

return { chapter: A.chapter_id, verdict: 'RETROFITTED', flagged: diag.flagged_count, review: reviewV, archive: arch }
