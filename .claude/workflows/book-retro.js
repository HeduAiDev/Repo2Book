export const meta = {
  name: 'book-retro',
  description: '复盘:从评审 issue/run-ledger/审计报告挖经验候选——≥2 章重复才算,产出含落点建议与 patch 草案的清单,不做任何修改',
  phases: [
    { title: 'Mine', detail: '每章一个挖掘 agent,只读 reviews/+audits' },
    { title: 'Cluster', detail: 'opus 聚类:≥2 章重复,对照台账标复发' },
  ],
}

const CFG = { instance: 'vllm-ascend', chapters: null, date: 'undated', repo_root: '/mnt/e/Laboratory/Repo2Book' }
const A = (typeof args !== 'undefined' && args && args.instance) ? args : CFG
const REPO = A.repo_root || '/mnt/e/Laboratory/Repo2Book'
const INST = A.instance
const ARTS = REPO + '/instances/' + INST + '/artifacts'
const BOOK = REPO + '/instances/' + INST + '/book'
const LEDGER = REPO + '/docs/superpowers/experience-ledger.md'
const OUT = BOOK + '/retro/retro-' + (A.date || 'undated') + '.json'

const MINE_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['signals'],
  properties: { signals: { type: 'array', items: { type: 'object', additionalProperties: false,
    required: ['signal', 'evidence', 'candidate_rule', 'source'],
    properties: { signal: { type: 'string' }, evidence: { type: 'string' },
      candidate_rule: { type: 'string' }, source: { type: 'string' } } } } },
}
const CLUSTER_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['candidates', 'stats_note'],
  properties: {
    candidates: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['id', 'pattern', 'occurrences', 'root_cause', 'target', 'draft_patch', 'expected_effect', 'recurrence'],
      properties: { id: { type: 'string' }, pattern: { type: 'string' },
        occurrences: { type: 'array', items: { type: 'object', additionalProperties: false,
          required: ['chapter', 'evidence'],
          properties: { chapter: { type: 'string' }, evidence: { type: 'string' } } } },
        root_cause: { type: 'string' }, target: { type: 'string' },
        draft_patch: { type: 'string' }, expected_effect: { type: 'string' },
        recurrence: { type: 'string' } } } },
    stats_note: { type: 'string' },
  },
}

phase('Mine')
let slugs = A.chapters
if (!slugs || !slugs.length) {
  const LIST_SCHEMA = { type: 'object', additionalProperties: false, required: ['slugs'],
    properties: { slugs: { type: 'array', items: { type: 'string' } } } }
  const ls = await agent('列出目录 ' + ARTS + ' 下所有 chNN-* 子目录名（Bash ls），按章号排序返回 slugs。',
    { schema: LIST_SCHEMA, label: 'list-chapters', phase: 'Mine', model: 'haiku', agentType: 'general-purpose' })
  slugs = (ls && ls.slugs) || []
}
log('挖掘 ' + slugs.length + ' 章的经验信号')

const mined = await parallel(slugs.map(function (slug) {
  return function () {
    return agent(
      '你是经验信号挖掘员。**只读**：' + ARTS + '/' + slug + '/reviews/ 下的 review-report.json、run-ledger.json、retrofit-review.json（存在哪个读哪个）+ ' + BOOK + '/audits/ 里涉及本章的条目（目录可能不存在）。\n' +
      '任务：挑出**指向流程/提示词/门禁缺陷**的信号——不是本章内容问题本身，而是「什么样的问题反复让 agent 犯错」。看：blocking issue 的成因类型、回环 >1 轮的原因、盲审 FAIL 的原因、escalated 记录。\n' +
      '每条返回 {signal(一句话概括问题类型), evidence(引用原文≤60字), candidate_rule(若要根除，一句话规则该写成什么), source(文件名+定位)}。\n' +
      '章级一次性内容错误（如某行号写错）不算信号，返回时跳过。没有信号返回 signals=[]。',
      { schema: MINE_SCHEMA, label: 'mine:' + slug.slice(0, 12), phase: 'Mine', model: 'sonnet', agentType: 'general-purpose' }
    ).then(function (r) { return { slug: slug, signals: (r && r.signals) || [] } })
  }
}))

phase('Cluster')
const all = mined.filter(Boolean)
const payload = JSON.stringify(all)
const clustered = await agent(
  '你是复盘聚类员。输入是各章挖掘出的经验信号（JSON 见文末）。另外 Read 台账 ' + LEDGER + '（已落地经验清单，可能只有表头）。\n' +
  '任务：按 root cause 聚类。**硬门槛：同类信号 ≥2 章出现才成候选；来源为 reader 顾问维度的信号需 ≥4 章**。\n' +
  '每个候选给：id（exp-' + (A.date || 'undated') + '-序号）、pattern、occurrences（逐章+证据）、root_cause、target（五选一：linter:<脚本名> | contract:<角色> | skill:<名> | runbook | instance）、draft_patch（可直接落笔的条款文字或 linter 规则描述，全角标点）、expected_effect（下批次哪个可数指标应下降）、recurrence（台账已有同 pattern 则写"复发:<台账id>,建议升级为<新落点>"，否则写 "new"）。\n' +
  'target 选择原则：可机检的优先 linter，角色行为进 contract，操作方法进 skill/runbook，仓库事实进 instance。\n' +
  'stats_note 写一句总量统计。**你不做任何文件修改**。\n\n' + payload,
  { schema: CLUSTER_SCHEMA, label: 'cluster', phase: 'Cluster', model: 'opus', agentType: 'general-purpose' }
)
const cands = (clustered && clustered.candidates) || []
const report = { date: A.date, instance: INST, chapters_mined: slugs.length,
  stats_note: (clustered && clustered.stats_note) || '', candidates: cands }
await agent('把下面 JSON **原样** Write 到 ' + OUT + '（目录不存在先建），写完返回 "written"。\n' + JSON.stringify(report),
  { label: 'write-report', phase: 'Cluster', model: 'haiku', agentType: 'general-purpose' })
log('复盘完成：' + cands.length + ' 个经验候选 → ' + OUT)
return { report: OUT, candidates: cands.map(function (c) { return { id: c.id, pattern: c.pattern, target: c.target, recurrence: c.recurrence } }) }
