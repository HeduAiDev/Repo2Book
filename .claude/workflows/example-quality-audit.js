export const meta = {
  name: 'example-quality-audit',
  description: '全书连贯性审计+修复:每章 opus reader 查「术语漂移/数学↔代码未搭桥/顺序颠倒/论断-证据失配/类比误导/玩具值混淆」→ 有命中的 opus writer 定点修(统一称呼+就地搭桥,禁整章重写)',
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
          severity: { type: 'string', enum: ['claim-wrong', 'confusing', 'minor'] },
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
  return '只读连贯性审计(不改文件)。逐段**从头到尾**通读 ' + dir + '/narrative/chapter.md,把全章的术语/符号/称呼在脑子里连成一张表,专找六类**会让读者困惑、造成割裂感**的问题:\n' +
    '(a) 论断-证据失配:强论断(「根本没有/再也找不到/完全不/就是」)与紧邻代码/数值/图**表面矛盾**——证据没就地展示让论断成立的关键值/条件,或露出一个同概念的通用参数与论断打架(样板:说「缓存没有 n_h」却配签名里带 num_kv_heads 的通用透传函数)。\n' +
    '(b) **术语漂移(重点)**:同一个量/概念全章换了名字却没打通——数学符号(如 $c^{KV}$)、代码标识符(如 decode_k_nope)、中文术语(如「解耦 key」「解耦 RoPE 分量」「rope 部分」)之间指同一物,却在不同小节各叫各的、首次换名处没就地点明「这就是前面的 X」。列出这个概念的**全部别名+各自出处**。\n' +
    '(c) **数学↔代码未搭桥(重点)**:源码块里的代码标识符(kv_lora_rank/decode_k_pe/exec_kv_decode…)出现时,没在**出现处**绑回它对应的数学符号/含义,读者要自己猜或翻到几节之后。\n' +
    '(d) **顺序颠倒**:某段源码/论断依赖的概念要到**后文**才解释(如源码块用了某命名,而解释这个命名的道理在更后的小节)。\n' +
    '(e) 类比/比喻误导:映射关系不成立、把读者引向错误直觉(只挑真误导的)。\n' +
    '(f) 玩具数值例与真实代码维度混淆:分不清哪个是教学简化、哪个是真实值,且无显式免责。\n' +
    '判据从严只报**真会困惑**的:引原文句(claim 字段填涉及的概念/论断及其别名或矛盾点)+ why_confusing 说清割裂在哪 + severity(claim-wrong 论断本身错 / confusing 真会困惑但可修 / minor)+ fix_hint 一句修法(如「统一称呼为 X 并在源码块出现处补『decode_k_nope 就是潜向量 $c^{KV}$』」)。\n' +
    '有显式免责/打通("decode_k_nope 即前面的 $c^{KV}$"、"玩具值 kv_lora_rank=4")、类比准确、论断有精确证据的,**不报**。\n' +
    '若全章无此病,返回 findings:[]。返回 {findings:[{section,claim,why_confusing,severity,fix_hint}]}。'
}

function fixPrompt(inst, slug, findings) {
  const dir = REPO + '/instances/' + inst + '/artifacts/' + slug
  const list = findings.map(function (f, i) {
    return (i + 1) + '. [' + f.severity + '] §' + f.section + ':论断「' + f.claim + '」——' + f.why_confusing + ' 修法:' + f.fix_hint
  }).join('\n')
  return '你的角色契约在 ' + REPO + '/.claude/agents/writer.md(只 Edit,定点修,禁整章重写)。修本章审计出的连贯性/割裂感困惑点(' + dir + '/narrative/chapter.md):\n' + list + '\n' +
    '按病种修:\n' +
    '• 术语漂移→**全章统一成一个称呼**(优先沿用最早/符号表里的那个),并在每处换名/代码标识符首现处补一句「就是前面的 X」把三层(数学符号/代码名/中文术语)打通;\n' +
    '• 数学↔代码未搭桥→在源码块出现处就地补一句绑定(如「decode_k_nope 就是潜向量 $c^{KV}$、decode_k_pe 就是解耦 RoPE 分量 $d_h^R$」);\n' +
    '• 顺序颠倒→补一句前指(「这个命名的道理见下一节权重吸收」)或把解释前移一句;\n' +
    '• 论断-证据失配→换成就地展示关键特化值的真源码/数值(必要时读 ' + REPO + '/instances/' + inst + '/source);论断本身错→按源码事实改论断;类比误导→修映射或加免责;玩具/真实混淆→补显式免责。\n' +
    '不改真源码本身、不杜撰。公式禁中文/CJK;行内数学的空格在 $ 的**外侧**(`压到 $d_c$ 维`),$ **内侧禁空格**(`$ d_c $` 不渲染);数学记号禁用反引号代替。\n' +
    '自检全过:python3 ' + REPO + '/scripts/lint_source_grounding.py ' + dir + ' && python3 ' + REPO + '/scripts/lint_formulas.py ' + dir + '/narrative/chapter.md && python3 ' + REPO + '/scripts/lint_chapter_structure.py ' + dir + '/narrative/chapter.md(无 BLOCKING)。**不 git commit**。返回 {status,fixed,note}。'
}

phase('Audit')
const results = await pipeline(
  A.chapters,
  function (ch) {
    return agent(auditPrompt(ch.instance, ch.slug), {
      label: 'audit:' + ch.slug.slice(0, 18), phase: 'Audit',
      model: 'opus', effort: 'high', agentType: 'general-purpose', schema: AUDIT_SCHEMA,
    }).then(function (r) { return Object.assign({}, ch, { findings: (r && r.findings) || [] }) })
  },
  function (aud, ch) {
    const real = (aud && aud.findings ? aud.findings : []).filter(function (f) { return f.severity !== 'minor' })
    if (real.length === 0) return Object.assign({}, ch, { findings: [], fixed: 0 })
    return agent(fixPrompt(ch.instance, ch.slug, real), {
      label: 'fix:' + ch.slug.slice(0, 20), phase: 'Fix',
      model: 'opus', effort: 'high', agentType: 'general-purpose', schema: FIX_SCHEMA,
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
