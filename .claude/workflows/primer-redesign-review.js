export const meta = {
  name: 'primer-redesign-review',
  description: '按「原理章=设计过的数学表达、非源码解读」新哲学评审全部 primer 章:逐章 opus 判 keep/modify/rewrite + 给理想版重设计草图。只评审不改。',
  whenToUse: '用户质疑原理章生硬讲源码、数学没讲清。args: {primers:[{instance,slug}], repo_root?}',
  phases: [{ title: 'Review', detail: '每章一 opus 评审员按新哲学判 keep/modify/rewrite' }],
}

const CFG = { primers: [], repo_root: '/mnt/e/Laboratory/Repo2Book' }
let A = (typeof args !== 'undefined' && args) ? args : CFG
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = CFG } }
if (!Array.isArray(A.primers) || A.primers.length === 0) {
  return { error: 'args 须为 {primers:[{instance,slug}]}——resume 也须重传' }
}
const REPO = A.repo_root || CFG.repo_root

const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['verdict', 'math_clarity', 'source_overweight', 'key_problems', 'redesign_sketch'],
  properties: {
    verdict: { type: 'string', enum: ['keep', 'modify', 'rewrite'] },
    math_clarity: { type: 'string' },
    source_overweight: { type: 'string' },
    key_problems: { type: 'array', items: { type: 'string' } },
    redesign_sketch: { type: 'string' },
  },
}

function reviewPrompt(inst, slug) {
  const dir = REPO + '/instances/' + inst + '/artifacts/' + slug
  const papers = REPO + '/instances/' + inst + '/book/papers/' + slug
  return '你是原理章(primer)重设计评审员。**新哲学(用户定,以此为唯一标准)**:原理章的天职不是「走一遍真实源码」,而是**把数学/原理设计成最清晰的表达,让读者一眼看清本质、降低理解门槛,但不失深度**。论文与 vLLM/vllm-ascend 真实实现未必一样——原理章不该硬塞实现细节(实现落地属于隔壁配对的实现章)。现在的写法常常「数学没讲懂、看代码又一头雾水」,两头不讨好。\n' +
    '读:' + dir + '/narrative/chapter.md(定稿正文,含图)+ ' + papers + '/(论文包 paper*.md、meta.json)。**读论文包是为判断:这套数学本可以被更清晰地『设计』出来吗?**\n' +
    '按四条严格判:\n' +
    '① **核心洞见是否被「设计」成一眼可见的结构**——而非让读者跟一长串带下标的稠密代数推一遍。样板:MLA 权重吸收,理想=「一条结合律直接点明哪个权重矩阵吸进哪个 + 一张权重架构图(吸收前 vs 吸收后)」,一眼看懂;而现状是 $q^C\\!\\cdot\\!k^C=\\dots=\\dots$ 的下标推导 + 数值相等表,还挂 float64 舍入讨论。判:每个 core 机制有没有「先亮洞见 → 用最简记号/一张结构图暴露本质 → 再补严谨」这种设计过的表达。\n' +
    '② **源码是否喧宾夺主 / 硬走与论文不符的实现**——大段真实源码解读(尤其命名/结构与论文数学对不上、制造困惑,如潜向量 $c^{KV}$ 在码里叫 decode_k_nope)是本章负担还是必要?理想:原理章里源码至多一句「落地见第 N 章」指路,版面还给把原理讲透。数一数本章源码块占了多少、其中多少与讲清原理无关。\n' +
    '③ **数学是否真讲懂**——有直觉、有设计过的记号与可视化,还是稠密/赶工?\n' +
    '④ **深度是否保住**——清晰 ≠ 浅。\n' +
    'verdict:keep=已达标(数学设计清晰、源码不喧宾夺主);modify=大体好、定点改(补一张架构图/某段源码降级为指路/先亮洞见);rewrite=根子上是「源码解读+赶推导」,需按新哲学重构。**无需考虑沉默成本,该 rewrite 就 rewrite,别手软。**\n' +
    'redesign_sketch 要具体:若 modify/rewrite,列出本章每个 core 机制**理想的「设计过的表达」是什么**(该用哪条律/哪种最简记号/画哪张架构图一眼看懂)、源码该降到什么程度。返回 {verdict, math_clarity, source_overweight, key_problems:[], redesign_sketch}。'
}

phase('Review')
const results = await pipeline(
  A.primers,
  function (p) {
    return agent(reviewPrompt(p.instance, p.slug), {
      label: 'review:' + p.slug.slice(0, 22), phase: 'Review',
      model: 'opus', effort: 'high', agentType: 'general-purpose', schema: SCHEMA,
    }).then(function (r) { return Object.assign({}, p, r || { verdict: 'error' }) })
  }
)
const all = results.filter(Boolean)
const byV = function (v) { return all.filter(function (r) { return r.verdict === v }).map(function (r) { return r.instance + '/' + r.slug }) }
log('原理章评审:rewrite=' + byV('rewrite').length + ' modify=' + byV('modify').length + ' keep=' + byV('keep').length)
return {
  rewrite: byV('rewrite'), modify: byV('modify'), keep: byV('keep'),
  detail: all.map(function (r) {
    return { ch: r.instance + '/' + r.slug, verdict: r.verdict, problems: r.key_problems, redesign: r.redesign_sketch }
  }),
}
