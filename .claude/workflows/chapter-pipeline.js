export const meta = {
  name: 'chapter-pipeline',
  description: '单章流水线：档案(真相源)→只做减法实现+测试→真源码解读叙事→多维协作评审→归档（含逃生舱：任一阶段发现路线错可拉闸升级）',
  phases: [
    { title: 'Dossier', detail: 'analyst 深读真实源码产出共享档案，并对抗性自核' },
    { title: 'Implement', detail: 'implementer 产出 subtract-only 精简版 (TDD)' },
    { title: 'Test', detail: 'tester 验证复现 vLLM 行为（反压闸门）' },
    { title: 'Write', detail: 'writer 以真实源码为主线写章节（内嵌源码+Roadmap）' },
    { title: 'Review', detail: '多维并行协作评审，有界回环' },
    { title: 'Archive', detail: 'archivist 归档 + 回写 Book Bible' },
  ],
}

// ⚠️ 本环境实测 Workflow 的 args 注入不可靠（args 未到达脚本）→ 用脚本内 CFG 作可靠配置；
// args 可用时优先 args。换章节时改 CFG（或修复 args 注入后直接传 args）。
const CFG = {
  chapter_id: 'ch21',
  slug: 'ch21-async-engine',
  focus: '异步通信与数据并行: AsyncIntermediateTensors 的惰性 PP 同步(isend/irecv 推迟到真正用时)、DPCoordinator 的 wave 同步(all-reduce 共识决定全体是否继续/空转)、DP 负载均衡(DPLBAsyncMPClient 按各 engine 负载分发请求)、gpu_worker 的 DP 维度。承接 ch20 分布式集合通信 + ch12 batch queue',
  highlight: 'async-engine',
  source_root: '/mnt/e/Laboratory/Repo2Book/instances/vllm/source',
  repo_root: '/mnt/e/Laboratory/Repo2Book',
  skip_dossier: false,
  paths: ['vllm/v1/engine/async_llm.py', 'vllm/v1/engine/core_client.py', 'vllm/v1/engine/coordinator.py', 'vllm/v1/engine/core.py'],
}
const A = (typeof args !== 'undefined' && args && args.chapter_id) ? args : CFG
const REPO = A.repo_root || '/mnt/e/Laboratory/Repo2Book'
const SRC = A.source_root || (REPO + '/instances/vllm/source')
const CH = REPO + '/instances/vllm/artifacts/' + A.slug
const HL = A.highlight || A.subsystem || ''
const PATHS = (A.paths || []).join(', ')

// 逃生舱：任何阶段发现路线/档案是错的，不许硬着头皮做错
const ESC = '\n\n**逃生舱（重要）**：如果你发现给定的路线/档案是错的——真实源码与计划不符、subtraction_plan 会破坏正确性、档案缺关键信息、无法产出忠实结果——**不要硬着头皮按错的做**。立即返回 status="BLOCKED"，blocker_reason 写清「哪里错 + 建议怎么改」。workflow 会**立刻中止**并把问题交给 Team Lead（我），我修正后从断点续跑。宁可拉闸，不要产出错误成果一路跑到底。'

function head(role) {
  return [
    '你的角色契约在 ' + REPO + '/.claude/agents/' + role + '.md —— **先读它**，严格遵守其中所有铁律。',
    '源码 pin f3fef123，根目录 ' + SRC + '（引用源码写规范 vllm/ 路径，**不带** instances/vllm/source/ 前缀）。',
    '本章目录（绝对路径）：' + CH,
    '本章：' + A.chapter_id + ' 《' + A.focus + '》',
    'vLLM 相关运行进容器：' + REPO + '/scripts/vllm_docker.sh ...（host 无 CUDA/vLLM）。',
    '',
  ].join('\n')
}

const STATUS_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['status', 'note'],
  properties: { status: { type: 'string', enum: ['OK', 'BLOCKED'] }, note: { type: 'string' }, blocker_reason: { type: 'string' } },
}
const VERIFY_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['sound', 'problems'],
  properties: { sound: { type: 'boolean' }, problems: { type: 'array', items: { type: 'string' } } },
}
const TEST_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['verdict', 'failures'],
  properties: { verdict: { type: 'string', enum: ['APPROVED', 'REJECTED'] }, failures: { type: 'string' } },
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

// ---------- Phase A: Dossier (真相源) + 对抗性自核 ----------
if (!A.skip_dossier) {
phase('Dossier')
const dossierV = await agent(
  head('analyst') +
  '任务：深读真实源码（重点：' + PATHS + '），产出本章**档案**并 Write 到 ' + CH + '/dossier/dossier.json。\n' +
  '先跑 `python3 ' + REPO + '/scripts/bible.py due ' + A.chapter_id + '`，结果放入 foreshadow_due。\n' +
  'dossier.json 字段：code_spine、embed_excerpts(逐字真实源码片段+可省略分支说明，带 vllm/...:Lxxx)、key_classes、data_flow、design_decisions、theory、subtraction_plan{delete:[{what,why_safe}], must_keep:[{symbol,why} 可检测符号]}、diagram_plan、foreshadow_due。\n' +
  'must_keep 要把"读者需理解、writer 需讲清"的符号都放进去（宁多留勿误删）。只描述真实源码，禁止杜撰。完成返回 status/note。' + ESC,
  { schema: STATUS_SCHEMA, label: 'dossier', phase: 'Dossier', agentType: 'general-purpose' }
)
if (dossierV && dossierV.status === 'BLOCKED') return { escalated: 'dossier', stage: 'Dossier', reason: dossierV.blocker_reason }

const dv = await agent(
  head('analyst') +
  '任务：**独立对抗性核对** ' + CH + '/dossier/dossier.json 是否忠于真实源码：路线是否正确？embed_excerpts 是否逐字、file:Lxxx 是否准确？subtraction_plan.delete 是否都安全、must_keep 是否完整（有无遗漏读者要学的关键符号）？\n' +
  '返回 sound（是否可放行）与 problems（具体问题列表）。',
  { schema: VERIFY_SCHEMA, label: 'dossier-verify', phase: 'Dossier', agentType: 'general-purpose' }
)
if (dv && dv.sound === false) return { escalated: 'dossier-verify', stage: 'Dossier', problems: dv.problems }
log('dossier 已通过对抗性核对')
} else {
  log('复用已人工审核的 dossier，跳过档案阶段')
}

// ---------- Phase B/C: Implement (TDD) + Test，有界回环 ----------
let ledger = []
let testV = null
for (let r = 1; r <= 3; r++) {
  phase('Implement')
  const impl = await agent(
    head('implementer') +
    '任务：读 ' + CH + '/dossier/dossier.json，按 subtraction_plan 产出 **subtract-only** 精简版到 ' + CH + '/implementation/，TDD 先写测试到 ' + CH + '/tests/。\n' +
    (ledger.length ? '上一轮测试失败，必须修复：\n' + ledger.join('\n') + '\n' : '') +
    '每 def/class 标 `# SOURCE: vllm/...:Lxxx`；删除标 `# SUBTRACTED:`。\n' +
    '**只可删除 subtraction_plan.delete 批准项；must_keep 符号必须保留；不得按己见删其他细节**（lint_fidelity 会校验 must_keep 都在）。\n' +
    '完成后自跑 `python3 ' + REPO + '/scripts/lint_fidelity.py ' + CH + '` 确保无 BLOCKING。返回 status/note。' + ESC,
    { schema: STATUS_SCHEMA, label: 'implement r' + r, phase: 'Implement', agentType: 'general-purpose' }
  )
  if (impl && impl.status === 'BLOCKED') return { escalated: 'implement', stage: 'Implement', round: r, reason: impl.blocker_reason }
  phase('Test')
  testV = await agent(
    head('tester') +
    '任务：验证 ' + CH + '/implementation/ 复现 dossier 记录的真实 vLLM 行为（非自洽）。\n' +
    '精简版纯测试：`python3 -m pytest ' + CH + '/tests -q`；若需 import vllm/CUDA：`' + REPO + '/scripts/vllm_docker.sh -m pytest /work/instances/vllm/artifacts/' + A.slug + '/tests`。\n' +
    '写 ' + CH + '/tests/test-report.json（含 verdict；若用容器记录 docker 命令+镜像 tag+vllm 版本）。\n' +
    '全过且 lint_fidelity 无 BLOCKING → verdict=APPROVED；否则 REJECTED 且 failures 写清失败摘要。',
    { schema: TEST_SCHEMA, label: 'test r' + r, phase: 'Test', agentType: 'general-purpose' }
  )
  if (testV && testV.verdict === 'APPROVED') break
  ledger.push('[round ' + r + '] ' + (testV ? testV.failures : 'tester error'))
  log('test 第 ' + r + ' 轮未过，回 implementer')
}

// ---------- Phase D: Write (真源码主线) ----------
phase('Write')
let writeV = null
for (let w = 1; w <= 2 && !writeV; w++) {
if (w > 1) log('write 上轮中断(API崩)，第 ' + w + ' 轮重试：chapter.md 已存在就用 Edit 续完/校验，否则新建')
writeV = await agent(
  head('writer') +
  '任务：以**真实 vLLM 源码为主线**写 ' + CH + '/narrative/chapter.md（你唯一有权写它）。\n' +
  '读 dossier、implementation、' + REPO + '/instances/vllm/book/bible/voice-guide.md，并跑 `python3 ' + REPO + '/scripts/bible.py due ' + A.chapter_id + '`。\n' +
  '开场 Roadmap：跑 `python3 ' + REPO + '/instances/vllm/book/assets/roadmap/roadmap.py --highlight ' + HL + ' --out ' + CH + '/diagrams/roadmap.svg`，用 rsvg-convert -z 2 转 PNG（**勿用 ImageMagick convert**，会丢中文/错位），正文引用该 PNG。\n' +
  '正文内嵌**真实源码片段**(裁剪无关分支用 `# … 省略 …`)，逐段解读设计决策；精简版只作"运行看数值"的交叉验证，不是主角。\n' +
  '若发现精简版缺了你要讲清的细节 → 用逃生舱拉闸（status=BLOCKED）让 implementer 补回，别将就。\n' +
  '埋伏笔、`python3 ' + REPO + '/scripts/bible.py payoff --resolve` 回收应回收项。\n' +
  '**零脚手架泄漏**：规范 vllm/ 路径、自然标题(无 Cell N)、不提内部文件。\n' +
  '完成后自跑五个 linter（chapter_structure/formulas/source_grounding/fidelity/diagrams）均无 BLOCKING。返回 status/note。' + ESC,
  { schema: STATUS_SCHEMA, label: 'write r' + w, phase: 'Write', agentType: 'general-purpose' }
)
}
if (!writeV) return { chapter: A.chapter_id, escalated: 'write-failed', stage: 'Write', note: 'writer 多轮失败(限流/崩溃)，无 chapter.md，不进评审' }
if (writeV && writeV.status === 'BLOCKED') return { escalated: 'write', stage: 'Write', reason: writeV.blocker_reason }

// ---------- Phase E: Review (多维并行 → 协作回环) ----------
let reviewV = null
const DIMS = ['fidelity（保真度+过度删减+零脚手架泄漏）', 'readability（可读/不枯燥/连贯）', 'algorithm（算法可理解性：图/数值/证明）', 'formula-structure-diagrams（公式可渲染+Roadmap+自包含+图示质量，跑 lint_formulas/chapter_structure/source_grounding/diagrams）']
for (let r = 1; r <= 3; r++) {
  phase('Review')
  const dims = await parallel(DIMS.map(function (dim) {
    return function () {
      return agent(
        head('reviewer') +
        '任务：**只**从「' + dim + '」维度评审 ' + CH + '/narrative/chapter.md（对照 ' + CH + '/dossier/dossier.json 与 Book Bible）。\n' +
        '机械维度先跑对应 linter（' + REPO + '/scripts/lint_*.py）。\n' +
        '协作式：每条 issue 必须给 suggested_fix + rationale，标 negotiable/blocking。该维度无 blocking issue → pass=true。',
        { schema: DIM_SCHEMA, label: 'review:' + dim.slice(0, 6) + ' r' + r, phase: 'Review', agentType: 'general-purpose' }
      )
    }
  }))
  const ok = dims.filter(Boolean)
  if (ok.length < DIMS.length) return { chapter: A.chapter_id, escalated: 'review-agents-failed', stage: 'Review', round: r, note: '部分评审 agent 失败(限流/崩溃)，评审未完成，不假通过' }
  const issues = ok.flatMap(function (d) { return d.issues || [] })
  const blocking = issues.filter(function (i) { return i.blocking })
  if (!ok.some(function (d) { return !d.pass }) && blocking.length === 0) {
    reviewV = { verdict: 'APPROVED', issues: issues }
    break
  }
  log('review 第 ' + r + ' 轮 REVISE：' + blocking.length + ' 个阻断项，回 writer')
  const rev = await agent(
    head('writer') +
    '评审 REVISE（第 ' + r + ' 轮）。用 receiving-code-review skill 逐条处理（采纳或带理由反驳），改 ' + CH + '/narrative/chapter.md：\n' +
    JSON.stringify(issues) + '\n完成后自跑四个 linter。返回 status/note。' + ESC,
    { schema: STATUS_SCHEMA, label: 'revise r' + r, phase: 'Review', agentType: 'general-purpose' }
  )
  if (rev && rev.status === 'BLOCKED') return { escalated: 'review-revise', stage: 'Review', round: r, reason: rev.blocker_reason }
  reviewV = { verdict: 'REVISE', issues: issues }
}

// 评审 3 轮仍未过 → 升级 Lead（兑现"同一问题 >3 轮自动升级"承诺），不静默归档 REVISE
if (reviewV && reviewV.verdict !== 'APPROVED') {
  return { chapter: A.chapter_id, test: testV, escalated: 'review-exhausted', stage: 'Review', issues: reviewV.issues }
}

// ---------- Phase F: Archive ----------
phase('Archive')
// 完整 review 对象注入提示词 → review-report.json 忠实落盘(含 verdict 与全部 issues)，
// 不让 archivist 凭记忆重建出有损版本。
const reviewJson = JSON.stringify(reviewV || { overall_verdict: 'UNKNOWN', issues: [] })
const archiveTask = head('archivist') +
  '任务一(务必先做)：把下面这个完整 review 对象**原样**写入 ' + CH + '/reviews/review-report.json（保留 verdict 与全部 issues，不要删改、不要自己重写摘要）：\n' +
  reviewJson + '\n' +
  '任务二：回写 Book Bible —— 登记本章精简版新接口（`python3 ' + REPO + '/scripts/bible.py iface --add ' + A.chapter_id + " '<sig>'`)，确认已回收伏笔。\n" +
  '任务三：在 ' + REPO + '/instances/vllm/trace/ 记 delivery 并更新 state.json。返回一句话状态。'
let archV = null
for (let a = 1; a <= 2 && !archV; a++) {
  if (a > 1) log('archive 上轮中断(API崩)，第 ' + a + ' 轮重试：已写的(review-report.json/bible 接口/trace)校验后跳过，只补未完成的')
  archV = await agent(
    archiveTask + (a > 1 ? '\n注意：这是重试。先检查 review-report.json 是否已存在且为合法完整 JSON、bible 是否已登记本章接口，已做的别重复，只补未完成的。' : ''),
    { label: 'archive r' + a, phase: 'Archive', agentType: 'general-purpose' }
  )
}

return { chapter: A.chapter_id, test: testV, review: reviewV }
