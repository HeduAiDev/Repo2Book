export const meta = {
  name: 'chapter-map-rollout',
  description: '存量章「本章地图」铺开：每章 illustrator 画源码剖面图(自检+盲审回环 ≤2)→ writer 插图引+选读指引 → lint_chapter_map --require 收口',
  whenToUse: '试点或全量给已成稿章节补开篇本章地图(新章由 chapter-pipeline Map 站保证)。args: {instance, chapters:[slug…], repo_root?}',
  phases: [
    { title: 'Draw', detail: '每章 illustrator 画图+登记+自检,盲审不过回炉一次' },
    { title: 'Insert', detail: 'writer 只许 Edit 插图引+指引,--require 收口' },
  ],
}

// ---- args 解析(N1 护栏:named workflow 的 args 可能以 JSON 字符串到达) ----
const CFG = { instance: 'vllm', chapters: [], repo_root: '/mnt/e/Laboratory/Repo2Book' }
let A = (typeof args !== 'undefined' && args) ? args : CFG
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = CFG } }
if (!A.instance || !Array.isArray(A.chapters) || A.chapters.length === 0) {
  return { error: 'args 须为 {instance, chapters:[slug…]}——resume 时也必须重传 args(CFG 回退无章可跑)' }
}
const REPO = A.repo_root || CFG.repo_root
const ARTS = REPO + '/instances/' + A.instance + '/artifacts'

const DRAW_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['status', 'blind_verdict', 'map_lint', 'geometry', 'rounds'],
  properties: {
    status: { type: 'string', enum: ['OK', 'BLOCKED'] },
    blind_verdict: { type: 'string', enum: ['PASS', 'FAIL'] },
    map_lint: { type: 'string' }, geometry: { type: 'string' },
    rounds: { type: 'number' }, note: { type: 'string' },
  },
}
const INSERT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['status', 'require_lint', 'structure_lint'],
  properties: {
    status: { type: 'string', enum: ['OK', 'BLOCKED'] },
    require_lint: { type: 'string' }, structure_lint: { type: 'string' }, note: { type: 'string' },
  },
}

function drawPrompt(slug) {
  const dir = ARTS + '/' + slug
  return '你的角色契约在 ' + REPO + '/.claude/agents/illustrator.md——先读其「本章地图」职责节,并读模板 ' +
    REPO + '/.claude/skills/svg-diagram/references/example-chapter-map.py(不可变项照办,数据表按本章重写)。\n' +
    '任务:为已成稿章 ' + dir + ' 画开篇「本章地图」(源码剖面图:入口→真实符号走线→出口,§徽标挂讲解站牌,底部阅读路线)。\n' +
    '输入:narrative/chapter.md(节结构)+ dossier/dossier.json 或 dossier.json(机制锚点;kind=primer 时按契约画论文结构、核论文包)。\n' +
    '硬规则:节点≤12(超长章聚合);符号必须真实可核;编号标题章徽标用 §N.M(N=目录号),自然标题章禁 §N.M、站牌用标题词;渲染后必须 Read PNG 亲眼看。\n' +
    '产出:diagrams/chapter-map.{py,svg,png};登记 figure-manifest.json(六项自查如实;blind_review 先置 PENDING)。\n' +
    '自检必须全过:python3 ' + REPO + '/scripts/lint_chapter_map.py ' + dir + ' && python3 ' +
    REPO + '/scripts/lint_diagram_geometry.py ' + dir + '/diagrams/chapter-map.svg\n' +
    '然后自任盲审(换视角):只看 PNG+正文标题列表,复述"从哪进、经过什么、从哪出、想跳读某机制看哪节"——复述不出或与正文不符即 FAIL,按问题改图重渲一轮(总共至多 2 轮),把最终 verdict 回填 manifest 的 blind_review。\n' +
    '两轮仍 FAIL 或遇根本障碍(如章结构无法成图)→ status=BLOCKED 并在 note 说明。\n' +
    '返回:status/blind_verdict/map_lint(linter 末行)/geometry(末行)/rounds/note。**不 git commit**。'
}

function insertPrompt(slug) {
  const dir = ARTS + '/' + slug
  return '你的角色契约在 ' + REPO + '/.claude/agents/writer.md——先读其条款 7(开篇「本章地图」)。\n' +
    '任务:在 ' + dir + '/narrative/chapter.md 开篇插入本章地图引用:位置=开篇导航(你在这里/Roadmap 标题,若有)与 hook 段之后、第一个内容分节标题之前;内容=`![本章地图:<一句图题>](../diagrams/chapter-map.png)` + 紧随 1–2 句自然措辞选读指引(看 diagrams/chapter-map.png 实图后写,指引里的 §号/站牌须与图一致)。\n' +
    '**只许 Edit 定点插入这一处**,不动其他任何内容;全角标点。\n' +
    '自检必须全过:python3 ' + REPO + '/scripts/lint_chapter_map.py ' + dir + ' --require && python3 ' +
    REPO + '/scripts/lint_chapter_structure.py ' + dir + '/narrative/chapter.md\n' +
    '返回:status/require_lint(末行)/structure_lint(末行)/note。**不 git commit**。'
}

const results = await pipeline(
  A.chapters,
  function (slug) {
    return agent(drawPrompt(slug), {
      label: 'draw:' + slug.slice(0, 16), phase: 'Draw',
      model: 'sonnet', agentType: 'general-purpose', schema: DRAW_SCHEMA,
    }).then(function (r) { return Object.assign({ slug: slug }, r || { status: 'BLOCKED', note: 'agent null(限流?)' }) })
  },
  function (draw, slug) {
    if (!draw || draw.status !== 'OK' || draw.blind_verdict !== 'PASS') {
      return Object.assign({ skipped_insert: true }, draw || { slug: slug, status: 'BLOCKED' })
    }
    return agent(insertPrompt(draw.slug), {
      label: 'insert:' + draw.slug.slice(0, 16), phase: 'Insert',
      model: 'sonnet', agentType: 'general-purpose', schema: INSERT_SCHEMA,
    }).then(function (r) { return Object.assign({}, draw, { insert: r || { status: 'BLOCKED', note: 'agent null(限流?)' } }) })
  }
)

const all = results.filter(Boolean)
const ok = all.filter(function (r) { return r.insert && r.insert.status === 'OK' })
const blocked = all.filter(function (r) { return !r.insert || r.insert.status !== 'OK' })
log('本章地图铺开:' + ok.length + '/' + all.length + ' 章全绿' + (blocked.length ? ';BLOCKED/未插:' + blocked.map(function (r) { return r.slug }).join(',') : ''))
return {
  ok: ok.map(function (r) { return { slug: r.slug, rounds: r.rounds } }),
  blocked: blocked.map(function (r) { return { slug: r.slug, note: (r.note || (r.insert && r.insert.note) || '') } }),
}
