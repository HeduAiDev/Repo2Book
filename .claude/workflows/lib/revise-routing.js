// revise-routing.js — chapter-pipeline Phase E revise 步分流纯函数（exp-0716-1）
//
// SDD: docs/superpowers/specs/2026-07-18-pipeline-revise-dimension-routing.md
// 背景：revise 步把全部 blocking（含 figure-integration 维）只派 writer，而 writer 只有权
// 改 narrative/——图侧矛盾在回环内永远修不掉，3 轮耗尽必 review-exhausted 升级 Lead
// （样本 6：ch05/ch07/ch27/ch29/ch33/ch39；ch29/ch39 还揭示「修好了仍逃逸」时序竞态——
// 终局判定基于修复前快照）。
//
// 本文件是**行为真相源**：workflow 里的同名函数须与此处逐字一致（workflow 无模块系统，
// 函数体内联进 chapter-pipeline.js；node --test 在这里锁行为 + 快照锁 DIMS 不被顺手改）。

// 维度串 'fidelity（…）' → 短名 'fidelity'（DIMS 数组本身不动，聚合后打标用）
function dimShortName(dimStr) {
  const s = String(dimStr)
  const i = s.indexOf('（')
  return i === -1 ? s : s.slice(0, i)
}

// 4 个真维度的 issues 按 DIMS 顺序补 dimension 标（reader/derivation 维在聚合处已打标）。
// 已带 dimension 的 issue 不覆写。
function tagDimIssues(dimResults, dims) {
  const out = []
  for (let idx = 0; idx < dimResults.length; idx++) {
    const d = dimResults[idx]
    const issues = (d && d.issues) || []
    for (const i of issues) {
      out.push(i && i.dimension ? i : Object.assign({}, i, { dimension: dimShortName(dims[idx]) }))
    }
  }
  return out
}

// 分流：blocking 且 dimension===figure-integration → 图组（illustrator 子回环）；
// 其余 blocking → writer 组（含 dimension 缺失的——不崩、不丢）；
// non-blocking 不路由，随附 writer 作参考。
function routeIssues(issues) {
  const figIssues = []
  const textIssues = []
  const nonBlocking = []
  for (const i of issues || []) {
    if (!i) continue
    if (!i.blocking) { nonBlocking.push(i); continue }
    if (i.dimension === 'figure-integration') figIssues.push(i)
    else textIssues.push(i)
  }
  return { figIssues, textIssues, nonBlocking }
}

// 图组 issues → 与 figure-requests done 条目同构的清单（喂给按需补图站形状的 illustrator）
function toFigRequestItems(figIssues) {
  return (figIssues || []).map(function (i) {
    return {
      figure_id: i.figure_id || i.figure || '',
      problem: i.problem || '',
      suggested_fix: i.suggested_fix || '',
    }
  })
}

// 行动计划（顺序即约束，测试锁死）：
// - stage1（并行组）：writer 修文（仅 textIssues 非空——纯图环不派 writer 白跑）
//   ∥ 图链 fig-fix → fig-blind（盲审再验证必须在 revise 步内闭合，进下一轮前 PENDING 已清）
// - stage2：fig-caption（writer 微任务收尾图引/图注——须在两条并行轨都完成后，
//   避免两个 writer 同时编 narrative/ 的竞态）
function planRevise(routed) {
  const stage1 = []
  if (routed.textIssues.length) stage1.push(['writer-fix'])
  if (routed.figIssues.length) stage1.push(['fig-fix', 'fig-blind'])
  const stages = stage1.length ? [stage1] : []
  if (routed.figIssues.length) stages.push([['fig-caption']])
  return stages
}

// 终局复验裁决（ch29/ch39 变体：第 3 轮修复完成后必须复核最新稿，不许拿修复前快照判死刑）：
// - reverify.all_cleared === true → APPROVED（以复验为准）
// - 未清 → review-exhausted，issues 取**最新**未清项（不是旧轮快照）
// - reverify 为 null/undefined（复验 agent 崩/限流）→ 不假通过，按 exhausted 处理，
//   issues 回退旧轮快照并注明复验未完成
function finalReviewDecision(reverify, lastBlocking) {
  if (reverify && reverify.all_cleared === true) {
    return { verdict: 'APPROVED' }
  }
  if (reverify) {
    return { verdict: 'review-exhausted', issues: reverify.uncleared || [] }
  }
  return { verdict: 'review-exhausted', issues: lastBlocking || [], note: '终局复验 agent 失败，按未清处理（不假通过）' }
}

module.exports = { dimShortName, tagDimIssues, routeIssues, toFigRequestItems, planRevise, finalReviewDecision }
