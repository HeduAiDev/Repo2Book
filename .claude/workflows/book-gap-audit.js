export const meta = {
  name: 'book-gap-audit',
  description: '全书概念覆盖审计：每章术语/概念首现须「本章建立/前章已立(concepts.json)/有先修指路」三者居一，输出按严重度排序的 gap 清单',
  phases: [
    { title: 'Audit', detail: '每章一个审计 agent 并行扫' },
    { title: 'Merge', detail: '去重排序落盘报告' },
  ],
}

const CFG = { instance: 'vllm-ascend', chapters: null, date: 'undated', repo_root: '/mnt/e/Laboratory/Repo2Book' }
// lint-exp-N1：args 若被 host 注入为 JSON 字符串（而非已解析对象），`args.instance` 对
// 字符串取属性返回 undefined，旧写法会静默回退到 CFG——先 JSON.parse 再判断，回退时显式告警。
let _args = typeof args !== 'undefined' ? args : undefined
if (typeof _args === 'string') {
  try { _args = JSON.parse(_args) } catch (e) { _args = null }
}
const A = (_args && _args.instance) ? _args : CFG
if (_args && !_args.instance) {
  log('⚠️ args 已收到但缺 instance 字段或解析失败，回退到脚本内 CFG 默认值: ' + JSON.stringify(CFG))
}
const REPO = A.repo_root || '/mnt/e/Laboratory/Repo2Book'
const INST = A.instance
const BOOK = REPO + '/instances/' + INST + '/book'
const ARTS = REPO + '/instances/' + INST + '/artifacts'
const OUT = A.out || (BOOK + '/audits/gap-audit-' + (A.date || 'undated') + '.json')

const AUDIT_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['issues'],
  properties: { issues: { type: 'array', items: { type: 'object', additionalProperties: false,
    required: ['concept', 'severity', 'evidence', 'suggested_fix'],
    properties: { concept: { type: 'string' }, severity: { type: 'string', enum: ['cliff', 'bump'] },
      evidence: { type: 'string' }, suggested_fix: { type: 'string' } } } } },
}

phase('Audit')
// 章清单：args 给定则用之；否则让首个 agent 列目录（脚本无 fs）
let slugs = A.chapters
if (!slugs || !slugs.length) {
  const LIST_SCHEMA = { type: 'object', additionalProperties: false, required: ['slugs'],
    properties: { slugs: { type: 'array', items: { type: 'string' } } } }
  const ls = await agent('列出目录 ' + ARTS + ' 下所有形如 chNN-* 的子目录名（用 Bash ls），按章号排序返回 slugs。',
    { schema: LIST_SCHEMA, label: 'list-chapters', phase: 'Audit', model: 'haiku', agentType: 'general-purpose' })
  slugs = (ls && ls.slugs) || []
}
log('审计 ' + slugs.length + ' 章')

const perCh = await parallel(slugs.map(function (slug) {
  return function () {
    return agent(
      '你是概念覆盖审计员。只读：' + ARTS + '/' + slug + '/narrative/chapter.md、' + BOOK + '/bible/glossary.json、' + BOOK + '/bible/concepts.json（可能不存在）、' + BOOK + '/cartography/papers-map.json（可能不存在）。\n' +
      '任务：找出本章**首现即使用**的术语/概念中，不满足三者居一的：① 本章自己建立（有定义/推导/直觉）；② 前章已立（concepts.json 里登记且章号更早）；③ 有先修指路（正文链接到某原理章/前章锚点）。\n' +
      '判严重度：cliff=不读论文/外部资料跟不上正文主线；bump=一句话补丁即可。常见词（tensor/GPU/KV cache 这类全书公设）不算。\n' +
      '每条 evidence 引正文行号与原句片段。无问题返回 issues=[]。',
      { schema: AUDIT_SCHEMA, label: 'audit:' + slug.slice(0, 12), phase: 'Audit', model: 'sonnet', agentType: 'general-purpose' }
    ).then(function (r) { return { slug: slug, issues: (r && r.issues) || [] } })
  }
}))

phase('Merge')
const all = perCh.filter(Boolean)
const flat = all.flatMap(function (c) { return c.issues.map(function (i) { return Object.assign({ chapter: c.slug }, i) }) })
const cliffs = flat.filter(function (i) { return i.severity === 'cliff' })
const report = { date: A.date, instance: INST, chapters_audited: slugs.length,
  totals: { cliffs: cliffs.length, bumps: flat.length - cliffs.length }, issues: flat }
await agent(
  '把下面 JSON **原样** Write 到 ' + OUT + '（目录不存在则先建）。不要改写内容。写完返回 "written"。\n' + JSON.stringify(report),
  { label: 'write-report', phase: 'Merge', model: 'haiku', agentType: 'general-purpose' }
)
log('gap 审计完成：cliff ' + cliffs.length + ' / bump ' + (flat.length - cliffs.length) + ' → ' + OUT)
return { report: OUT, totals: report.totals,
  top_cliffs: cliffs.slice(0, 12).map(function (i) { return i.chapter + ': ' + i.concept }) }
