// node --test — revise 分流行为锁（SDD 2026-07-18-pipeline-revise-dimension-routing）
const { test } = require('node:test')
const assert = require('node:assert')
const fs = require('node:fs')
const path = require('node:path')
const R = require('./revise-routing')

const F = (over) => Object.assign({ problem: 'p', suggested_fix: 'f', blocking: true }, over)

// ---------- 用例 1：混合路由 ----------
test('mixed issues route by dimension; non-blocking not routed', () => {
  const issues = [
    F({ dimension: 'fidelity' }),
    F({ dimension: 'figure-integration', figure_id: 'fig-a' }),
    F({ dimension: 'reader-comprehension' }),
    F({ dimension: 'figure-integration', blocking: false }),   // non-blocking 图评→不路由
    F({ dimension: 'algorithm-pedagogy', blocking: false }),
  ]
  const r = R.routeIssues(issues)
  assert.strictEqual(r.figIssues.length, 1)
  assert.strictEqual(r.figIssues[0].figure_id, 'fig-a')
  assert.strictEqual(r.textIssues.length, 2)
  assert.strictEqual(r.nonBlocking.length, 2)
})

// ---------- 用例 2：纯图环——writer 组为空、计划不派 writer ----------
test('figure-only round dispatches no writer', () => {
  const r = R.routeIssues([F({ dimension: 'figure-integration' }), F({ dimension: 'figure-integration' })])
  assert.strictEqual(r.textIssues.length, 0)
  const plan = R.planRevise(r)
  const flat = plan.flat(2)
  assert.ok(!flat.includes('writer-fix'), '纯图环不该派 writer 白跑')
  assert.ok(flat.includes('fig-fix'))
})

// ---------- 用例 3：dimension 缺失 → writer 组（不崩、不丢） ----------
test('missing dimension falls back to writer group', () => {
  const r = R.routeIssues([F({}), null, F({ dimension: 'figure-integration' })])
  assert.strictEqual(r.textIssues.length, 1)
  assert.strictEqual(r.figIssues.length, 1)
})

// ---------- 用例 4：PENDING 时序——fig-blind 紧随 fig-fix 且在 caption/下一轮之前 ----------
test('fig-blind ordered after fig-fix within stage1; caption in later stage', () => {
  const r = R.routeIssues([F({ dimension: 'figure-integration' }), F({ dimension: 'fidelity' })])
  const plan = R.planRevise(r)
  // stage1 并行组里有 writer 轨 + 图轨；图轨内序 fig-fix → fig-blind
  const figTrack = plan[0].find((t) => t.includes('fig-fix'))
  assert.ok(figTrack, '图轨在 stage1')
  assert.ok(figTrack.indexOf('fig-blind') > figTrack.indexOf('fig-fix'), '盲审再验证必须在修图后、revise 步内闭合')
  // caption 收尾在 stage2（两条并行轨都完成后，避免双 writer 竞态）
  assert.deepStrictEqual(plan[1], [['fig-caption']])
  assert.ok(plan[0].find((t) => t.includes('writer-fix')), '混合环 writer 轨在 stage1')
})

// ---------- 用例 5：ch39 变体——终局复验为准 ----------
test('final reverify decides: cleared → APPROVED', () => {
  const d = R.finalReviewDecision({ all_cleared: true, uncleared: [] }, [F({})])
  assert.strictEqual(d.verdict, 'APPROVED')
})

test('final reverify decides: uncleared → exhausted with LATEST items', () => {
  const latest = [{ problem: 'still-broken-x' }]
  const d = R.finalReviewDecision({ all_cleared: false, uncleared: latest }, [F({ problem: 'old-snapshot' })])
  assert.strictEqual(d.verdict, 'review-exhausted')
  assert.deepStrictEqual(d.issues, latest, 'exhausted 的 issues 必须是最新未清项，不是旧轮快照')
})

test('final reverify agent failure → exhausted (no fake pass)', () => {
  const last = [F({ problem: 'unresolved' })]
  const d = R.finalReviewDecision(null, last)
  assert.strictEqual(d.verdict, 'review-exhausted')
  assert.deepStrictEqual(d.issues, last)
})

// ---------- 打标：4 真维度 issues 补 dimension，不覆写已有 ----------
test('tagDimIssues tags by DIMS order, keeps existing dimension', () => {
  const dims = ['fidelity（…linter…）', 'algorithm-pedagogy（…）']
  const tagged = R.tagDimIssues(
    [{ issues: [F({ dimension: undefined })] }, { issues: [F({ dimension: 'pre-tagged' })] }], dims)
  assert.strictEqual(tagged[0].dimension, 'fidelity')
  assert.strictEqual(tagged[1].dimension, 'pre-tagged')
})

test('toFigRequestItems shape matches figure-requests done entries', () => {
  const items = R.toFigRequestItems([F({ dimension: 'figure-integration', figure_id: 'fig-x' })])
  assert.deepStrictEqual(Object.keys(items[0]).sort(), ['figure_id', 'problem', 'suggested_fix'])
})

// ---------- 用例 6：缓存约束回归——DIMS/readerPrompt 逐字不动 ----------
test('chapter-pipeline.js DIMS block and readerPrompt are byte-identical to snapshot', () => {
  const src = fs.readFileSync(path.join(__dirname, '..', 'chapter-pipeline.js'), 'utf8')
  // DIMS 四维原文锚点（改一个字都会 fail——防实现顺手「优化」既有提示词破坏 resume 缓存）
  const anchors = [
    "  'algorithm-pedagogy（逐机制对账：对 dossier.mechanisms 每条填勾选表——直觉在场？数值推演表在场且带 trace 标记？不变量论证？量化落数字？core 三层齐？先跑 lint_trace_consistency 作客观依据；输出逐机制勾选表，不是整体印象）',",
    "  'figure-integration（先跑 lint_diagrams；然后逐张用 Read 打开 PNG 亲眼看：图在其机制讲解附近？图注给结论而非描述画面？正文数字与图上一致？图对读懂机制真有帮助？）',",
    "  'formula-structure（公式规则+Roadmap 开场+自包含+锚点/半角，跑 lint_formulas/lint_anchors/lint_punct/lint_chapter_structure）',",
    'const readerPrompt = PRIMER',
    "你是第一次读这篇论文的工程师（高级工程师，懂 Transformer 基础，但**没读过这篇论文**）",
  ]
  for (const a of anchors) {
    assert.ok(src.includes(a), 'DIMS/readerPrompt 快照锚点丢失或被改动：' + a.slice(0, 40) + '…')
  }
})
