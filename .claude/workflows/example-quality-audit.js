export const meta = {
  name: 'example-quality-audit',
  description: '全书举例/类比质量审计+修复:每章 reader 查「论断-证据失配/类比误导/玩具值混淆」→ 有命中的 writer 定点修(证据对齐,禁整章重写)',
  whenToUse: '用户要求排查全部章节的举例恰当性。args: {chapters:[{instance,slug}], repo_root?}',
  phases: [
    { title: 'Audit', detail: '每章 reader 读全章找 claim-evidence/类比/玩具值问题' },
    { title: 'Fix', detail: '仅有命中的章:writer 定点修证据对齐' },
  ],
}

const CFG = { chapters: [], repo_root: '/mnt/e/Laboratory/Repo2Book' }
let A = (typeof args !== 'undefined' && args) ? args : CFG
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = CFG } }
if (!Array.isArray(A.chapters) || A.chapters.length === 0) {
  return { error: 'args 须为 {chapters:[{instance,slug}]}——resume 也须重传' }
}
const REPO = A.repo_root || CFG.repo_root

const AUDIT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['findings'],
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['section', 'claim', 'why_confusing', 'severity', 'fix_hint'],
        properties: {
          section: { type: 'string' },
          claim: { type: 'string' },
          why_confusing: { type: 'string' },
          severity: { type: 'string', enum: ['claim-wrong', 'confusing-but-claim-true', 'minor'] },
          fix_hint: { type: 'string' },
        },
      },
    },
  },
}
const FIX_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['status', 'fixed', 'note'],
  properties: {
    status: { type: 'string', enum: ['OK', 'BLOCKED'] },
    fixed: { type: 'number' }, note: { type: 'string' },
  },
}

function auditPrompt(inst, slug) {
  const dir = REPO + '/instances/' + inst + '/artifacts/' + slug
  return '只读审计(不改文件)。逐段通读 ' + dir + '/narrative/chapter.md,专找三类**会让读者困惑**的举例/类比问题:\n' +
    '(a) 论断-证据失配:强论断(「根本没有/再也找不到/完全不/就是」)与紧邻展示的代码/数值/图**表面矛盾**——证据没就地展示让论断成立的关键值/条件,或露出一个同概念的通用参数与论断打架(样板:某章说「缓存没有 n_h」却配了签名里带 num_kv_heads 的通用透传函数)。\n' +
    '(b) 类比/比喻误导:映射关系不成立、把读者引向错误直觉(只挑真误导的,不是所有类比)。\n' +
    '(c) 玩具数值例与真实代码维度混淆:读者分不清哪个是教学简化、哪个是真实值,且无显式免责。\n' +
    '判据从严只报**真会困惑**的:引原文句+说清矛盾/误导在哪+判严重度(claim-wrong 论断本身错 / confusing-but-claim-true 论断真但证据选错 / minor)+给一句修法方向(fix_hint)。\n' +
    '有显式免责("精简版用玩具值 kv_lora_rank=4""本机制纯论文推导代码里没有"…)、类比映射准确、论断有精确证据支撑的,**不报**。\n' +
    '若全章无此病,返回 findings:[]。返回 {findings:[{section,claim,why_confusing,severity,fix_hint}]}。'
}

function fixPrompt(inst, slug, findings) {
  const dir = REPO + '/instances/' + inst + '/artifacts/' + slug
  const list = findings.map(function (f, i) {
    return (i + 1) + '. [' + f.severity + '] §' + f.section + ':论断「' + f.claim + '」——' + f.why_confusing + ' 修法:' + f.fix_hint
  }).join('\n')
  return '你的角色契约在 ' + REPO + '/.claude/agents/writer.md(只 Edit,定点修,禁整章重写)。修本章审计出的举例/类比困惑点(' + dir + '/narrative/chapter.md):\n' + list + '\n' +
    '原则:论断真但证据选错(confusing-but-claim-true)→换成就地展示关键特化值的真源码/数值(必要时读 ' + REPO + '/instances/' + inst + '/source 找);论断本身错(claim-wrong)→按源码事实改论断;类比误导→修映射或加免责;玩具/真实混淆→补一句显式免责。改证据不改真源码本身、不杜撰。公式禁中文、$ 两侧留空格。\n' +
    '自检全过:python3 ' + REPO + '/scripts/lint_source_grounding.py ' + dir + ' && python3 ' + REPO + '/scripts/lint_formulas.py ' + dir + '/narrative/chapter.md && python3 ' + REPO + '/scripts/lint_chapter_structure.py ' + dir + '/narrative/chapter.md(无 BLOCKING)。**不 git commit**。返回 {status,fixed,note}。'
}

phase('Audit')
const results = await pipeline(
  A.chapters,
  function (ch) {
    return agent(auditPrompt(ch.instance, ch.slug), {
      label: 'audit:' + ch.slug.slice(0, 18), phase: 'Audit',
      model: 'sonnet', agentType: 'general-purpose', schema: AUDIT_SCHEMA,
    }).then(function (r) { return Object.assign({}, ch, { findings: (r && r.findings) || [] }) })
  },
  function (aud, ch) {
    const real = (aud && aud.findings ? aud.findings : []).filter(function (f) { return f.severity !== 'minor' })
    if (real.length === 0) return Object.assign({}, ch, { findings: [], fixed: 0 })
    return agent(fixPrompt(ch.instance, ch.slug, real), {
      label: 'fix:' + ch.slug.slice(0, 20), phase: 'Fix',
      model: 'sonnet', agentType: 'general-purpose', schema: FIX_SCHEMA,
    }).then(function (r) { return Object.assign({}, ch, { findings: real, fix: r || { status: 'BLOCKED' } }) })
  }
)

const all = results.filter(Boolean)
const hit = all.filter(function (r) { return r.findings && r.findings.length > 0 })
const minorOnly = all.filter(function (r) { return r.findings && r.findings.length === 0 && r.fixed !== 0 })
log('举例质量审计:' + all.length + ' 章扫完,' + hit.length + ' 章有真命中并修')
return {
  audited: all.length,
  hits: hit.map(function (r) {
    return { slug: r.slug, findings: r.findings.map(function (f) { return f.severity + ':' + f.section }), fix: r.fix }
  }),
  clean: all.length - hit.length,
}
