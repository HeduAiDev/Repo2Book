export const meta = {
  name: 'primer-redesign-rollout',
  description: '按 ch21 验收样板把「设计过的数学表达」新哲学滚到其余原理章:fable5 重写→定图权补图(顿悟图头图)→盲审含顿悟门→素材同步→推导审计→读者门',
  whenToUse: '用户指示「用当前的新流程处理重写所有原理篇」。args: {chapters:[{instance,slug,verdict,forward}]}',
  phases: [
    { title: 'Write', detail: 'fable5 按新哲学+密度纪律+本章草图重构,产 figure-requests', model: 'fable' },
    { title: 'Map', detail: '结构变更时重绘本章地图' },
    { title: 'Figures', detail: 'illustrator 消化 requests→盲审(头图过顿悟门)→writer 插引用' },
    { title: 'Sync', detail: 'explainer/dossier 素材账与新图集对齐+四联 linter' },
    { title: 'Derivation', detail: 'opus 亲手重推全章,≤2 轮回环', model: 'opus' },
    { title: 'Reader', detail: 'opus 台阶四问+一致性+顿悟第六问,≤2 轮回环', model: 'opus' },
  ],
}

const REPO = '/mnt/e/Laboratory/Repo2Book'
const SPEC = REPO + '/docs/superpowers/specs/2026-07-12-primer-redesign-design.md'
const SKETCHES = REPO + '/docs/superpowers/specs/primer-redesign-sketches-2026-07-12.json'
const SAMPLE = REPO + '/instances/vllm-ascend/artifacts/ch21-primer-mla/narrative/chapter.md'
const FWD_SPEC = REPO + '/docs/superpowers/specs/2026-07-11-forward-looking-primer-design.md'

let A = (typeof args !== 'undefined' && args) ? args : null
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = null } }
if (!A || !Array.isArray(A.chapters) || A.chapters.length === 0) {
  return { error: 'args 须为 {chapters:[{instance,slug,verdict,forward}]}——resume 也须重传' }
}

const ESC = '\n\n**逃生舱**:发现约束彼此冲突/素材缺关键信息/无法忠实完成——不要硬做,返回 status="BLOCKED" + blocker_reason(哪里卡+建议),该章中止交 Lead,不影响其他章。'

const W_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['status', 'note', 'figure_requests', 'structure_changed'],
  properties: { status: { type: 'string', enum: ['OK', 'BLOCKED'] }, note: { type: 'string' }, blocker_reason: { type: 'string' }, figure_requests: { type: 'number' }, structure_changed: { type: 'boolean' } },
}
const S_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['status', 'note'],
  properties: { status: { type: 'string', enum: ['OK', 'BLOCKED'] }, note: { type: 'string' }, blocker_reason: { type: 'string' } },
}
const BLIND_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['all_pass', 'failures'],
  properties: { all_pass: { type: 'boolean' }, failures: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['figure_id', 'problem', 'suggested_fix'], properties: { figure_id: { type: 'string' }, problem: { type: 'string' }, suggested_fix: { type: 'string' } } } } },
}
const GATE_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['pass', 'issues'],
  properties: { pass: { type: 'boolean' }, issues: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['problem', 'suggested_fix', 'blocking'], properties: { problem: { type: 'string' }, suggested_fix: { type: 'string' }, blocking: { type: 'boolean' } } } } },
}

function dirOf(c) { return REPO + '/instances/' + c.instance + '/artifacts/' + c.slug }
function papersOf(c) { return REPO + '/instances/' + c.instance + '/book/papers/' + c.slug }

function writePrompt(c) {
  const dir = dirOf(c)
  return '你是本书的 Writer。先读角色契约 ' + REPO + '/.claude/agents/writer.md(重点:primer 分支的信息密度纪律、公式渲染硬规则、必达物3 图集决策权),再读新哲学 spec ' + SPEC + '(§1 天职/§2 三段式/§2.4 密度/§2.5 顿悟图/§3 源码纪律)。\n' +
    '**黄金样板**:读 ' + SAMPLE + '(用户验收的 ch21-MLA 终稿)——你要写出的就是这个密度与深度水准。\n' +
    '任务:按新哲学**重构** ' + dir + '/narrative/chapter.md(评审判定 ' + c.verdict + ')。本章专属任务书:读 ' + SKETCHES + ' 中键「' + c.instance + '/' + c.slug + '」的 key_problems(病灶清单)与 redesign_sketch(每个 core 机制理想表达),逐条落实。\n' +
    '素材:' + dir + '/dossier/dossier.json + ' + dir + '/explainer/explainer.json + 论文包 ' + papersOf(c) + '/ + 现有 ' + dir + '/diagrams/(逐张 Read PNG)。\n' +
    '写作要求:\n' +
    '① **全章主线**:找到贯穿全章的那一条定理/命题(如 ch21 的「吸收的前提是中间块为常量」),开篇点破,每节挂在它上面收线于小结。\n' +
    '② **三段式每 core 机制**:先亮洞见(常常就是一行等价式/不变量)→ 最简记号+图暴露本质 → 完整推导进「> **严谨**」折叠框。数学是主角,推导链直接写,公式后紧跟 1-2 句解说;比喻预算每机制至多一短句。\n' +
    '③ **点透深度**:优先「等价视角/不变量/工程真义」类一句话换一个理解层次的洞见(参照 ch21 的 decode≡MQA-576、吸收=重排计算次序)。\n' +
    (c.forward
      ? '④ **前瞻 primer 纪律**(本章是前瞻章):读 ' + FWD_SPEC + ';内嵌的上游源码片段保留其溯源句式(repo/PR/merge-commit),数学仍是主角、源码只作论据不作主线。\n'
      : '④ **源码纪律**:源码至多一句「落地见第 N 章」指路(沿用现有跨章指向);可留的唯一例外是能直接对上某条公式的 2-3 行核心且就地绑回数学符号。\n') +
    '⑤ **图集由你定**(必达物3):按 spec §2.5 设计**顿悟图头图**(锚定一个反直觉洞见/落差揭示/削到本质/一图一顿悟)+ 需要的 zoom-in;现有好图是资产(保留/替换/弃用由你判断);变更写 ' + dir + '/diagrams/figure-requests.json(claim+numbers 带溯源+target_section+reason),返回值 figure_requests 填条数(无变更填 0);**不许自己画**。\n' +
    '硬约束(linter 会挡):现有 trace 数值表**数字一个不许改**、`<!-- trace: … -->` 标记保留(位置可随新结构挪);符号速查表在场(本章地图引用后、第一个公式前);paper-fig-* 的「重绘自 arXiv:xxxx Fig.N」固定句式图注原样;行内数学一律 $`…`$、公式内禁中文、** 外侧空格;跨章链接两层相对路径、禁裸文字章号;现有伏笔/回收句功能保留(措辞可融入新行文);开场 roadmap 引用+本章地图引用+选读指引在场;零脚手架泄漏。\n' +
    '若你改动了 `##` 章节骨架(增删改节标题),structure_changed 填 true(workflow 会重绘本章地图)。\n' +
    '收工自检(全部无 BLOCKING):lint_formulas / lint_chapter_structure / lint_source_grounding / lint_trace_consistency / lint_paper_grounding(scripts/ 下,章目录或 chapter.md 为参)。**不 git commit。**返回 status/note(重构摘要:主线定理是什么/砍了什么/深度点落哪)/figure_requests/structure_changed。' + ESC
}

function mapPrompt(c) {
  const dir = dirOf(c)
  return '你是本书的 Illustrator。先读契约 ' + REPO + '/.claude/agents/illustrator.md 的「本章地图」节。任务:writer 重构了 ' + dir + '/narrative/chapter.md 的章节骨架,按契约重绘 ' + dir + '/diagrams/chapter-map.{py,svg,png}(§徽标逐一对正文实际标题;自然标题章禁 §N.M 徽标;节点≤12、宽≤1500、比例≤2.6:1)。强制:渲染→Read PNG 亲眼看→自查→更新 manifest。自跑 python3 ' + REPO + '/scripts/lint_chapter_map.py ' + dir + ' --require 通过。返回 status/note。' + ESC
}

function figIllPrompt(c, ledger) {
  const dir = dirOf(c)
  return '你是本书的 Illustrator。先读契约 ' + REPO + '/.claude/agents/illustrator.md(「开工前」输入优先级 1)。任务:处理 ' + dir + '/diagrams/figure-requests.json(writer 定的图集变更)。add/replace 逐张强制流程:gen_<id>.py(坐标循环/常量计算,文本全 esc())→渲染→**Read 打开 PNG 亲眼看**→六项自查全真→登记 figure-manifest.json(blind_review 置 PENDING);drop 删文件+移除 manifest 条目。**头图(顿悟图)按 spec ' + SPEC + ' §2.5 五步法画**:视觉主轴=落差对比,削到只剩「那一下」,量化落差做成视觉尺度。数字全部来自条目 numbers(带溯源),缺溯源→BLOCKED。处理完条目挪 done、requests 清空。先调 Skill(skill="svg-diagram")。\n' +
    (ledger.length ? '上一轮盲审 FAIL,先修:\n' + ledger.join('\n') + '\n' : '') +
    '自跑 python3 ' + REPO + '/scripts/lint_diagram_geometry.py ' + dir + '/diagrams/*.svg 无问题。返回 status/note。' + ESC
}

function figBlindPrompt(c) {
  const dir = dirOf(c)
  return '你是插图盲审员。**第一步(确定性,先做)**:Read ' + dir + '/diagrams/figure-requests.json,若 requests 数组非空 → 直接 all_pass=false,每条未处理请求记一条 failure(figure_id + problem=「illustrator 未处理即交差」+ suggested_fix=「回炉处理该条 request」)——不许只审已画的就放行。然后**只准看**:' + dir + '/diagrams/figure-requests.json 的 done 条目 + figure-manifest.json + 每张 PENDING PNG(用 Read 打开)。**禁止**看 gen_*.py 与正文。逐张四步:①只看图复述论点;②与 done 条目 claim 对照,对不上=FAIL;③图上每个数字与 numbers 逐个核对,对不上/多出无溯源数字=FAIL;④明显不可读=FAIL。**头图(顿悟图)加做顿悟门**:看图 5 秒,核心洞见有没有「啪」地击中?没击中=FAIL(判顿没顿,不是画得对不对;塞两个洞见互相稀释也算 FAIL)。verdict/notes 回填 manifest 的 blind_review。返回 all_pass 与 failures。'
}

function figInsertPrompt(c) {
  const dir = dirOf(c)
  return '你是本书的 Writer(先读 ' + REPO + '/.claude/agents/writer.md)。微任务:你提的图集变更已完成并过盲审(' + dir + '/diagrams/figure-requests.json 的 done 条目)。只许 Edit 定点收尾 ' + dir + '/narrative/chapter.md:新增/替换图在其 target_section 附近插引用(**先 Read PNG 亲眼看再写图注,图注给结论**;顿悟图头图插在开篇 hook 段附近);drop 图删除其引用;必要时对齐被替换图的图注。禁其他改动。自跑 lint_chapter_structure + lint_formulas 无 BLOCKING。返回 status/note。' + ESC
}

function syncPrompt(c) {
  const dir = dirOf(c)
  return '素材账同步员(机械任务,禁创作)。writer 重构了 ' + dir + ' 章并变更了图集(见 diagrams/figure-requests.json 的 done 条目与 figure-manifest.json 终态)。把素材真相源对齐到终态:\n' +
    '① explainer/explainer.json:mechanisms[].figure_specs 里指向已删除图的条目——若有取代图则改 figure_id 并在 claim 尾注一句「20xx-xx-xx 重设计:XX 取代 XX」;若该机制决定不配图则清空该 figure_specs 并加 figure_note 说明。新增图若对应某机制,不强求补 spec(figure-requests done 条目已是其真相源)。\n' +
    '② dossier/dossier.json:mechanisms[].needs_figure 与终态一致(降为不配图的机制置 false + figure_note 记决策)。\n' +
    '③ 跑四联:python3 ' + REPO + '/scripts/lint_dossier.py ' + dir + ' && python3 ' + REPO + '/scripts/lint_explainer.py ' + dir + ' && python3 ' + REPO + '/scripts/lint_trace_consistency.py ' + dir + ' && python3 ' + REPO + '/scripts/lint_diagrams.py ' + dir + ' 全部 exit 0。\n' +
    '只许为对账而改这两个 json 的 figure 相关字段,**禁改 trace 数字/机制定义/正文**。返回 status/note(改了几处)。' + ESC
}

function derivPrompt(c) {
  const dir = dirOf(c)
  return '你是推导审计员。对 ' + dir + '/narrative/chapter.md 每条 $$ 推导链**亲手重推**:从假设/定义独立推到结论再对照正文;矩阵运算逐步核形状;数值例逐个数字重算;能写 numpy/sympy 断言的写脚本实跑(scratchpad 下)。对照论文包 ' + papersOf(c) + '/paper*.md 为真相源;引用的测试/断言/论文原句须真实存在。发现推导错/形状不合法/数字对不上/符号冲突/杜撰引用 → issues(blocking=true);风格建议不记。返回 pass/issues。'
}

function gateFixPrompt(c, gate, issues) {
  const dir = dirOf(c)
  return '你是本书的 Writer(先读 ' + REPO + '/.claude/agents/writer.md)。' + gate + '门禁打回,**只许 Edit 定点修复**以下 blocking issues(逐条采纳或带理由反驳,不许整章重写):\n' + JSON.stringify(issues) + '\n改动涉及数值须亲手重算;改完自跑 lint_formulas + lint_chapter_structure + lint_trace_consistency 无 BLOCKING。返回 status/note。' + ESC
}

function readerPrompt(c) {
  const dir = dirOf(c)
  return '你是第一次读这篇论文的工程师(懂 Transformer 基础,**没读过这篇论文**)。只读 ' + dir + '/narrative/chapter.md 及其引用的图(Read PNG),前面章节当已读背景,不准看论文原文/源码/上网。\n' +
    '逐关键公式台阶四问:①符号都认识吗(前文/符号表解释过)?②公式前有直觉铺垫吗?③上一步到这一步跳步了吗?④需要先读别的论文吗?\n' +
    '第五问·全章一致性:同一个量自始至终同名吗?数学符号/代码名/中文术语换称呼处有没有就地打通?有没有段落依赖后文才讲的概念?\n' +
    '第六问·顿悟门:开篇头图 5 秒内让你一眼看清本章核心洞见了吗(而非稠密推导/仅数据流)?全章主线一句话是什么——你能答出来吗?\n' +
    '①–⑥ 任一真卡住 → 该条 blocking=true + problem + suggested_fix;风格建议 blocking=false。全过 → pass=true。返回 pass/issues。'
}

async function runChapter(c) {
  const tag = c.instance + '/' + c.slug
  const out = { chapter: tag, verdict_in: c.verdict, stages: {} }

  const wr = await agent(writePrompt(c), { schema: W_SCHEMA, label: 'write:' + c.slug.slice(0, 20), phase: 'Write', model: 'fable', agentType: 'general-purpose' })
  if (!wr || wr.status !== 'OK') { out.failed = 'write'; out.reason = (wr && wr.blocker_reason) || 'write agent 失败'; return out }
  out.stages.write = wr.note
  out.figure_requests = wr.figure_requests

  if (wr.structure_changed) {
    const mp = await agent(mapPrompt(c), { schema: S_SCHEMA, label: 'map:' + c.slug.slice(0, 20), phase: 'Map', model: 'sonnet', agentType: 'general-purpose' })
    if (!mp || mp.status !== 'OK') { out.failed = 'map'; out.reason = (mp && mp.blocker_reason) || 'map agent 失败'; return out }
    out.stages.map = 'redrawn'
  }

  if (wr.figure_requests > 0) {
    let blind = null
    let ledger = []
    for (let f = 1; f <= 3; f++) {
      const ill = await agent(figIllPrompt(c, ledger), { schema: S_SCHEMA, label: 'fig:' + c.slug.slice(0, 20) + ' r' + f, phase: 'Figures', model: 'opus', agentType: 'general-purpose' })
      if (!ill || ill.status !== 'OK') { out.failed = 'figures'; out.reason = (ill && ill.blocker_reason) || 'illustrator 失败'; return out }
      blind = await agent(figBlindPrompt(c), { schema: BLIND_SCHEMA, label: 'blind:' + c.slug.slice(0, 20) + ' r' + f, phase: 'Figures', model: 'sonnet', agentType: 'general-purpose' })
      if (blind && blind.all_pass) break
      ledger = ((blind && blind.failures) || []).map(function (x) { return '[' + x.figure_id + '] ' + x.problem + ' → ' + x.suggested_fix })
      log(tag + ' 图盲审 r' + f + ' FAIL:' + ledger.length + ' 张打回')
    }
    if (!blind || !blind.all_pass) { out.failed = 'fig-blind'; out.reason = JSON.stringify((blind && blind.failures) || []); return out }
    const ins = await agent(figInsertPrompt(c), { schema: S_SCHEMA, label: 'fig-ins:' + c.slug.slice(0, 20), phase: 'Figures', model: 'fable', agentType: 'general-purpose' })
    if (!ins || ins.status !== 'OK') { out.failed = 'fig-insert'; out.reason = (ins && ins.blocker_reason) || 'fig-insert 失败'; return out }
    out.stages.figures = wr.figure_requests + ' 条变更落地'
  }

  const sync = await agent(syncPrompt(c), { schema: S_SCHEMA, label: 'sync:' + c.slug.slice(0, 20), phase: 'Sync', model: 'sonnet', agentType: 'general-purpose' })
  if (!sync || sync.status !== 'OK') { out.failed = 'sync'; out.reason = (sync && sync.blocker_reason) || 'sync 失败'; return out }
  out.stages.sync = sync.note

  for (let g = 1; g <= 2; g++) {
    const dv = await agent(derivPrompt(c), { schema: GATE_SCHEMA, label: 'deriv:' + c.slug.slice(0, 20) + ' r' + g, phase: 'Derivation', model: 'opus', effort: 'high', agentType: 'general-purpose' })
    if (!dv) { out.failed = 'derivation'; out.reason = '审计 agent 失败——primer 不得免审'; return out }
    if (dv.pass) { out.stages.derivation = 'PASS r' + g; break }
    const blocking = dv.issues.filter(function (i) { return i.blocking })
    if (g === 2) { out.failed = 'derivation'; out.reason = JSON.stringify(blocking); return out }
    log(tag + ' 推导审计 r' + g + ' FAIL:' + blocking.length + ' 条,回 writer')
    const fx = await agent(gateFixPrompt(c, '推导审计', blocking), { schema: S_SCHEMA, label: 'deriv-fix:' + c.slug.slice(0, 18), phase: 'Derivation', model: 'fable', agentType: 'general-purpose' })
    if (!fx || fx.status !== 'OK') { out.failed = 'derivation-fix'; out.reason = (fx && fx.blocker_reason) || 'fix 失败'; return out }
  }

  for (let g = 1; g <= 2; g++) {
    const rd = await agent(readerPrompt(c), { schema: GATE_SCHEMA, label: 'reader:' + c.slug.slice(0, 20) + ' r' + g, phase: 'Reader', model: 'opus', agentType: 'general-purpose' })
    if (!rd) { out.failed = 'reader'; out.reason = 'reader agent 失败——primer 不得免审'; return out }
    const blocking = (rd.issues || []).filter(function (i) { return i.blocking })
    if (rd.pass && blocking.length === 0) { out.stages.reader = 'PASS r' + g; break }
    if (g === 2) { out.failed = 'reader'; out.reason = JSON.stringify(blocking); return out }
    log(tag + ' 读者门 r' + g + ' FAIL:' + blocking.length + ' 条,回 writer')
    const fx = await agent(gateFixPrompt(c, '读者', blocking), { schema: S_SCHEMA, label: 'reader-fix:' + c.slug.slice(0, 18), phase: 'Reader', model: 'fable', agentType: 'general-purpose' })
    if (!fx || fx.status !== 'OK') { out.failed = 'reader-fix'; out.reason = (fx && fx.blocker_reason) || 'fix 失败'; return out }
  }

  out.done = true
  return out
}

phase('Write')
const results = await pipeline(A.chapters, function (c) { return runChapter(c) })
const all = results.filter(Boolean)
const ok = all.filter(function (r) { return r.done })
const failed = all.filter(function (r) { return !r.done })
log('primer 重构批:' + ok.length + ' 章收口 / ' + failed.length + ' 章卡住')
return { ok: ok, failed: failed }