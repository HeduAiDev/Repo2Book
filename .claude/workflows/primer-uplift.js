export const meta = {
  name: 'primer-uplift',
  description: 'primer 原理章降认知台阶两段式：diagnose(只读体检产四清单)→ Lead/用户批 → apply(素材扩容→论文图重绘→writer 定点降台阶→reader 台阶四问硬门禁回环)',
  whenToUse: '9 章 primer 原理章降台阶铺开（试点或批量均用它）。先跑 phase="diagnose" 只读体检，汇总交用户批 key_figures 与先修分级；批准后带 args.approvals 再跑 phase="apply" 施工。args: {instance, chapters:[slug…], phase:"diagnose"|"apply", approvals?, repo_root?}',
  phases: [
    { title: 'Diagnose', detail: '每章一 sonnet agent 并行体检：符号覆盖/论文精髓图候选/认知悬崖点/先修分级——只读不改正文' },
    { title: 'Materials', detail: 'explainer 补 symbol_table；论文包 meta.json 写批准 key_figures；heavy 先修子论文核心片段扩容进包（# PAPER 锚）' },
    { title: 'Illustrate', detail: '逐批准 key_figure 取原图亲眼看→忠实重绘→登记 manifest→自跑 paper_grounding+geometry' },
    { title: 'Write', detail: 'writer 定点 Edit：符号速查表/首现解释/直觉垫(cliff_points)/light 先修框/图嵌入 target_section' },
    { title: 'DerivationCheck', detail: 'opus 推导审计员对全章 $$ 推导链亲手重推（假设→结论/矩阵形状/数值重算）；fail 则 writer 微修再判，回环 ≤2 轮，竭尽标 BLOCKED' },
    { title: 'ReaderGate', detail: 'opus 读者台阶五问(含一致性)硬门禁；fail 则 writer 微修 issues 再判，回环 ≤2 轮，竭尽标 BLOCKED' },
  ],
}

// ---- args 解析(N1 护栏:named workflow 的 args 可能以 JSON 字符串到达；resume 时也须重传) ----
const CFG = { instance: 'vllm-ascend', chapters: [], phase: 'diagnose', approvals: {}, repo_root: '/mnt/e/Laboratory/Repo2Book' }
let A = (typeof args !== 'undefined' && args) ? args : CFG
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = CFG } }
if (!A.instance || !Array.isArray(A.chapters) || A.chapters.length === 0) {
  return { error: 'args 须为 {instance, chapters:[slug…], phase}——resume 时也必须重传 args(CFG 回退无章可跑)' }
}
if (A.phase !== 'diagnose' && A.phase !== 'apply') {
  return { error: 'args.phase 须为 "diagnose" 或 "apply"，收到: ' + JSON.stringify(A.phase) }
}
if (A.phase === 'apply' && (!A.approvals || !A.approvals.chapters || typeof A.approvals.chapters !== 'object')) {
  return { error: 'apply 阶段缺 args.approvals——须为 {chapters:{slug:{key_figures:[…], prerequisites:[…已批级别]}}}；先跑 phase="diagnose" 交用户批准，批准结果原样带回来' }
}
if (A.phase === 'apply') {
  // 覆盖性护栏：漏批的章会静默空跑(空批准=素材/重绘无事可做)，宁可显式报错
  const missing = A.chapters.filter(function (s) { return !A.approvals.chapters[s] })
  if (missing.length) {
    return { error: 'approvals.chapters 未覆盖以下章(漏批会静默空跑)：' + missing.join(', ') }
  }
}
const REPO = A.repo_root || CFG.repo_root
const ARTS = REPO + '/instances/' + A.instance + '/artifacts'
const PAPERS_ROOT = REPO + '/instances/' + A.instance + '/book/papers'
const MODELS = Object.assign({ diagnose: 'sonnet', materials: 'sonnet', illustrate: 'sonnet', write: 'fable' /* 2026-07-13 用户定:原理章写作用 fable5 */, derivation: 'opus' /* 推导审计升档 */, reader: 'opus' /* exp-0712-3:一致性检测需能力,haiku 对割裂给假通过 */, fix: 'sonnet' }, A.models || {})

const ESC = '\n\n**逃生舱（重要）**：如果你发现给定的批准/素材/路线是错的——真实情况与批准清单不符、施工会破坏正文正确性、缺关键前置信息——**不要硬着头皮做**。立即返回 status="BLOCKED"，blocker_reason 写清「哪里错 + 建议怎么改」。workflow 会中止该章后续阶段（不影响其他章节），把问题交给 Team Lead。'

function approvalsFor(slug) {
  const c = (A.approvals && A.approvals.chapters && A.approvals.chapters[slug]) || {}
  return { key_figures: c.key_figures || [], prerequisites: c.prerequisites || [] }
}
function approvalCounts(slug) {
  const appr = approvalsFor(slug)
  return {
    key_figures_approved: appr.key_figures.length,
    prerequisites_light: appr.prerequisites.filter(function (p) { return p.load === 'light' }).length,
    prerequisites_heavy: appr.prerequisites.filter(function (p) { return p.load === 'heavy' }).length,
  }
}

// =====================================================================
// Phase A: Diagnose（只读体检，四清单，禁改正文，逐章 parallel 跑）
// =====================================================================
const DIAGNOSIS_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['symbols_uncovered', 'key_figures_candidates', 'cliff_points', 'prerequisites'],
  properties: {
    symbols_uncovered: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['symbol', 'first_use', 'suggested_meaning'],
      properties: { symbol: { type: 'string' }, first_use: { type: 'string' }, suggested_meaning: { type: 'string' } } } },
    key_figures_candidates: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['fig', 'arxiv', 'shows', 'why_essential', 'target_section'],
      properties: { fig: { type: 'string' }, arxiv: { type: 'string' }, shows: { type: 'string' }, why_essential: { type: 'string' }, target_section: { type: 'string' } } } },
    cliff_points: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['section', 'formula_hint', 'which_of_four_questions', 'suggested_fix'],
      properties: { section: { type: 'string' }, formula_hint: { type: 'string' }, which_of_four_questions: { type: 'string' }, suggested_fix: { type: 'string' } } } },
    prerequisites: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['concept', 'cited_paper', 'where_used', 'load', 'proposal'],
      properties: { concept: { type: 'string' }, cited_paper: { type: 'string' }, where_used: { type: 'string' },
        load: { type: 'string', enum: ['light', 'heavy', 'critical'] }, proposal: { type: 'string' } } } },
  },
}

function diagnosePrompt(slug) {
  const dir = ARTS + '/' + slug
  const papers = PAPERS_ROOT + '/' + slug
  return '你是 primer 原理章降认知台阶诊断员。**只读体检，禁止修改任何文件**，除了 Write 你的诊断报告本身。\n' +
    '读：' + dir + '/narrative/chapter.md（定稿正文）+ ' + papers + '/（论文包 paper.md 与 meta.json）+ ' + dir + '/explainer/explainer.json（数值轨迹/直觉素材，若存在）。\n' +
    '产出四份清单，Write 到 ' + dir + '/reviews/uplift-diagnosis.json：\n' +
    '① symbols_uncovered：正文 `$$` 公式块里出现、但全文找不到解释（既无符号速查表也无首现 ±3 行人话说明）的符号——{symbol(LaTeX 原文，如 k_j^{C}), first_use(首现的小节标题), suggested_meaning(据论文推断的一句人话含义)}。\n' +
    '② key_figures_candidates：论文原图里"降低阅读难度的精髓图"（是论文里的图，不是本章已画的自产机制图）——{fig(如 "Fig.2"), arxiv, shows(图展示什么，一句话), why_essential(为何是精髓、不可替代), target_section(嵌入本章哪个小节最合适)}。meta.json 已登记过的仍要盘点复核（可能有遗漏）。\n' +
    '③ cliff_points：正文里读者会卡住的认知悬崖——逐公式/推导步骤过台阶四问(①符号都认识吗②公式前有直觉铺垫吗③是否跳步④是否需要先读别的论文才能懂)，命中任一问记一条 {section(小节标题), formula_hint(是哪个公式/推导步骤), which_of_four_questions(命中第几问，可写多个), suggested_fix(具体怎么补——插一句直觉/插一步中间推导/插先修框)}。\n' +
    '④ prerequisites：正文引用支撑主线推导、但本章未展开的子论文/外部概念——{concept, cited_paper(arXiv 号), where_used(本章哪处用到), load("light"=一句直觉即可跟上|"heavy"=需要该子论文核心构造/定理|"critical"=子论文本身值一章), proposal(处置建议：light 给一句先修框直觉草稿；heavy 给建议扩容进论文包的核心片段来源；critical 给理由说明为何值得升级决策)}。\n' +
    '找不到问题的清单留空数组，不要为凑数硬造；只 Write 这一个文件，不改 narrative/chapter.md、不改论文包、不改 explainer.json。\n' +
    '返回你写入文件的同一份四清单 JSON（workflow 据此汇总计数与 critical 清单，供 Lead 呈用户批准）。'
}

if (A.phase === 'diagnose') {
  phase('Diagnose')
  const perCh = await parallel(A.chapters.map(function (slug) {
    return function () {
      return agent(diagnosePrompt(slug), {
        schema: DIAGNOSIS_SCHEMA, label: 'diagnose:' + slug.slice(0, 16), phase: 'Diagnose',
        model: MODELS.diagnose, agentType: 'general-purpose',
      }).then(function (r) { return { slug: slug, diag: r } })
    }
  }))
  const rows = perCh.filter(Boolean)
  const totals = { symbols_uncovered: 0, key_figures_candidates: 0, cliff_points: 0, prerequisites: 0 }
  const critical = []
  rows.forEach(function (r) {
    if (!r.diag) return
    totals.symbols_uncovered += r.diag.symbols_uncovered.length
    totals.key_figures_candidates += r.diag.key_figures_candidates.length
    totals.cliff_points += r.diag.cliff_points.length
    totals.prerequisites += r.diag.prerequisites.length
    r.diag.prerequisites.forEach(function (p) {
      if (p.load === 'critical') critical.push(Object.assign({ slug: r.slug }, p))
    })
  })
  const missing = rows.filter(function (r) { return !r.diag }).map(function (r) { return r.slug })
  log('诊断完成：' + (rows.length - missing.length) + '/' + A.chapters.length + ' 章产出诊断' +
    (missing.length ? '；agent 失败(限流/崩溃)：' + missing.join(',') : '') +
    '；symbols=' + totals.symbols_uncovered + ' figures=' + totals.key_figures_candidates +
    ' cliffs=' + totals.cliff_points + ' prereqs=' + totals.prerequisites + '(critical ' + critical.length + ')')
  return {
    phase: 'diagnose',
    totals: totals,
    critical: critical,
    chapters: rows.map(function (r) {
      return {
        slug: r.slug, ok: !!r.diag,
        counts: r.diag ? {
          symbols_uncovered: r.diag.symbols_uncovered.length,
          key_figures_candidates: r.diag.key_figures_candidates.length,
          cliff_points: r.diag.cliff_points.length,
          prerequisites: r.diag.prerequisites.length,
        } : null,
        diag: r.diag,
      }
    }),
    missing: missing,
  }
}

// =====================================================================
// Phase B: Apply（批后施工：素材→重绘→写作→reader 硬门禁，四段 pipeline）
// =====================================================================
const STATUS_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['status', 'note'],
  properties: { status: { type: 'string', enum: ['OK', 'BLOCKED'] }, note: { type: 'string' }, blocker_reason: { type: 'string' } },
}
const WRITE_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['status', 'note', 'counts'],
  properties: {
    status: { type: 'string', enum: ['OK', 'BLOCKED'] }, note: { type: 'string' }, blocker_reason: { type: 'string' },
    counts: { type: 'object', additionalProperties: false,
      required: ['symbols_addressed', 'cliff_points_addressed', 'key_figures_embedded', 'figure_requests'],
      properties: {
        symbols_addressed: { type: 'number' }, cliff_points_addressed: { type: 'number' }, key_figures_embedded: { type: 'number' },
        figure_requests: { type: 'number' },
      } },
  },
}
const READER_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['pass', 'issues'],
  properties: {
    pass: { type: 'boolean' },
    issues: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['problem', 'suggested_fix'],
      properties: { problem: { type: 'string' }, suggested_fix: { type: 'string' } } } },
  },
}
const DERIVATION_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['pass', 'issues'],
  properties: {
    pass: { type: 'boolean' },
    issues: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['location', 'problem', 'suggested_fix'],
      properties: { location: { type: 'string' }, problem: { type: 'string' }, suggested_fix: { type: 'string' } } } },
  },
}

function materialsPrompt(slug) {
  const dir = ARTS + '/' + slug
  const papers = PAPERS_ROOT + '/' + slug
  const appr = approvalsFor(slug)
  const heavy = appr.prerequisites.filter(function (p) { return p.load === 'heavy' })
  return '你的角色契约在 ' + REPO + '/.claude/agents/explainer.md——先读其 primer 节（symbol_table 产出职责）。\n' +
    '任务：为已批准的降台阶素材扩容，**不动正文** ' + dir + '/narrative/chapter.md 一个字。\n' +
    '① symbol_table：读定稿正文与论文包 ' + papers + '/paper.md，读诊断报告 ' + dir + '/reviews/uplift-diagnosis.json 的 symbols_uncovered，逐条产出 {symbol, meaning, first_use, source} 合并进 ' + dir + '/explainer/explainer.json 的 symbol_table 字段（已有则增量合并、不覆盖已核实字段；没有 uplift-diagnosis.json 就自行盘点公式块里缺解释的符号）。\n' +
    '② key_figures：把下列已批准 key_figures 写入 ' + papers + '/meta.json 的 key_figures 字段（追加去重，不删已有项）：\n' + JSON.stringify(appr.key_figures) + '\n' +
    '③ heavy 先修包扩容：下列已批准 load=heavy 的先修项，把其子论文核心片段（关键公式/论证段）整理成独立 md 写入 ' + papers + '/（文件名自定，如 prereq-<concept-slug>.md），每个引用片段标 `# PAPER: §x Eq.y` 锚（对齐既有 # PAPER 锚体系）：\n' + JSON.stringify(heavy) + '\n' +
    '（load=light 的先修项本步不处理，交给 writer 阶段直接写先修框；load=critical 已属用户升级决策范围，本步不处理。）\n' +
    '完成后自跑 `python3 ' + REPO + '/scripts/lint_explainer.py ' + dir + '` 确保无 BLOCKING。返回 status/note。' + ESC
}

function illustratePrompt(slug) {
  const dir = ARTS + '/' + slug
  const papers = PAPERS_ROOT + '/' + slug
  const appr = approvalsFor(slug)
  return '你的角色契约在 ' + REPO + '/.claude/agents/illustrator.md——先读其「论文精髓图重绘」节，严格照做；**不许碰** ' + dir + '/narrative/chapter.md（正文插图引用归 writer 阶段做）。\n' +
    '任务：逐张重绘下列已批准 key_figures（论文原图，非本章已有的自产机制图）：\n' + JSON.stringify(appr.key_figures) + '\n' +
    '每张流程：① 先取原图——从 arXiv HTML/ar5iv 抓图 URL 后 curl 下载，用 Read **亲眼看**原图长什么样（抓不到则降级：按论文包 ' + papers + '/paper.md 的文字描述重绘，图注相应改用降级句式）。② 忠实重绘 SVG（信息结构对齐原图，配色/字体套本书视觉语言，文字译中）到 ' + dir + '/diagrams/paper-fig-<N>.{py,svg,png}，图注固定句式「重绘自 arXiv:xxxx Fig.N:<一句话>」（降级时改「按 arXiv:xxxx Fig.N（§y）描述重绘」）。③ 渲染后 Read PNG 六项自查全真才登记进 ' + dir + '/diagrams/figure-manifest.json（blind_review 初写 PENDING）。\n' +
    '完成后自跑 `python3 ' + REPO + '/scripts/lint_paper_grounding.py ' + dir + ' --expect-primer` 与 `python3 ' + REPO + '/scripts/lint_diagram_geometry.py ' + dir + '/diagrams/*.svg`。**注意**：此时正文尚未插入图引用，lint_paper_grounding 报的 key_figure_missing（正文缺对应图注）在本阶段是预期的——writer 下一阶段插入引用后会消，不必现在解决；若报的是其他问题（fig 号错、孤儿图注、symbol_context 等）才需要你自己修。返回 status/note（note 里如实记录 key_figure_missing 是否只是预期的"待 writer 插入"）。' + ESC
}

function writePrompt(slug) {
  const dir = ARTS + '/' + slug
  const appr = approvalsFor(slug)
  const light = appr.prerequisites.filter(function (p) { return p.load === 'light' })
  return '你的角色契约在 ' + REPO + '/.claude/agents/writer.md——先读其 primer 节（符号速查表/首现解释/直觉垫/先修框写法）。本任务**禁整章重写，只许 Edit 定点修改** ' + dir + '/narrative/chapter.md。\n' +
    '读 ' + dir + '/reviews/uplift-diagnosis.json（本章诊断：symbols_uncovered/cliff_points）+ 本次批准 ' + JSON.stringify(appr) + '（已批 key_figures 与先修分级）+ ' + dir + '/explainer/explainer.json（已补 symbol_table）+ ' + dir + '/diagrams/（已重绘的 paper-fig-* PNG，先 Read 看图）。\n' +
    '定点插入以下几类修改（互不冲突，各自定位，不改其余正文、不动章节结构）：\n' +
    '① **符号速查表**：本章地图引用之后、第一个公式之前，用 explainer.symbol_table 数据插 markdown 表格（自然措辞，呈现方式你定）。\n' +
    '② **首现解释**：diagnosis.symbols_uncovered 逐条——在其 first_use 处紧邻一句人话解释（直觉优先于形式定义）。\n' +
    '③ **直觉垫**：diagnosis.cliff_points 逐条——按 suggested_fix 在其 section 补一句直觉铺垫或一步中间推导。\n' +
    '④ **light 先修框**：以下已批准 load=light 的先修项——3–5 句直觉+出处 arXiv 号，blockquote 样式，自然措辞，插在 where_used 附近：\n' + JSON.stringify(light) + '\n' +
    '⑤ **图嵌入**：已重绘的 key_figures 逐张嵌入其 target_section（`![重绘自 arXiv:xxxx Fig.N:一句话](../diagrams/paper-fig-N.png)`，图注句式须与 illustrator 图上一致）。\n' +
    '⑥ **图集由你定**(契约必达物3，2026-07-13)：本次修改若让某段落值得配新图、或某旧图不再贴合叙事，写 ' + dir + '/diagrams/figure-requests.json（add/replace/drop，claim+numbers 带溯源），counts.figure_requests 填条数（无变更填 0）——workflow 会派 illustrator 处理后再让你插/删引用；**不许自己画**，也不为加而加。\n' +
    '完成后自跑 `python3 ' + REPO + '/scripts/lint_paper_grounding.py ' + dir + ' --expect-primer` + `python3 ' + REPO + '/scripts/lint_formulas.py ' + dir + '/narrative/chapter.md` + `python3 ' + REPO + '/scripts/lint_chapter_structure.py ' + dir + '/narrative/chapter.md` 确保均无 BLOCKING。\n' +
    '返回 status/note/counts（counts.symbols_addressed=你实际处理的 symbols_uncovered 条数；counts.cliff_points_addressed=你实际处理的 cliff_points 条数；counts.key_figures_embedded=你实际嵌入的 key_figures 张数）。' + ESC
}

function derivationCheckPrompt(slug) {
  const dir = ARTS + '/' + slug
  const papers = PAPERS_ROOT + '/' + slug
  return '你是推导审计员。对每条 $$ 推导链**亲手重推**：从假设/定义独立推到结论，再对照正文；矩阵乘法逐步核形状；数值例逐个数字重算；凡能写成 numpy/sympy 可执行断言的写脚本实跑（scratchpad 下）。发现推导错误/形状不合法/数字对不上/符号用法与定义冲突 → 记入 issues；纯风格性建议不必记录（本门禁只关注推导正确性）。\n' +
    '只审本次改动触及的推导链与数值例可从 `cd ' + dir + ' && git diff -- narrative/chapter.md` 看（本次 uplift 新插入的符号速查表/直觉垫/先修框/图嵌入），但为防连带错误，**全章 $$ 链都过一遍**，不能只挑改动处。\n' +
    '读 ' + dir + '/narrative/chapter.md（定稿正文）+ 论文包 ' + papers + '/paper.md（推导对照的真相源）。\n' +
    '逐条推导过：①假设/定义是否足以独立推出结论（无缺失步骤）？②矩阵/张量运算各步形状是否合法（写下形状标注核对）？③数值例每个数字能否用脚本复算对上？④符号用法是否与全文定义/符号速查表一致（无冲突/无静默改名）？\n' +
    '返回 pass（全部推导链核验通过则 true）与 issues（每条 {location(哪个小节/哪条公式), problem(具体错在哪，需可复现), suggested_fix(具体怎么改)}）；pass=true 时 issues 为空数组。'
}
function derivationFixPrompt(slug, issues) {
  const dir = ARTS + '/' + slug
  return '你的角色契约在 ' + REPO + '/.claude/agents/writer.md。任务：推导审计门禁 FAIL，**只修**下列具体 issue（定点 Edit，不许整章重写、不改其余正文）：\n' + JSON.stringify(issues) + '\n' +
    '完成后自跑 `python3 ' + REPO + '/scripts/lint_paper_grounding.py ' + dir + ' --expect-primer` + `python3 ' + REPO + '/scripts/lint_formulas.py ' + dir + '/narrative/chapter.md` + `python3 ' + REPO + '/scripts/lint_chapter_structure.py ' + dir + '/narrative/chapter.md` 确保均无 BLOCKING。返回 status/note。' + ESC
}

function readerPrompt(slug) {
  const dir = ARTS + '/' + slug
  return '你是第一次读这篇论文的工程师（高级工程师，懂 Transformer 基础，但没读过这篇论文、没看过源码）。只用 Read 打开 ' + dir + '/narrative/chapter.md（含它引用的图，图也用 Read 打开看）。**不准看论文包/dossier/explainer，不准上网。**\n' +
    '逐个公式/推导步骤过台阶四问：①符号都认识吗（前文解释过，或有符号表）？②公式前有直觉铺垫吗？③从上一步到这一步是否跳步（缺中间推导）？④是否需要先读别的论文才能懂？\n' +
    '再做第五问·全章一致性（跨段落、非单公式）：⑤同一个量/概念是否自始至终同名？若在数学符号（如 $c^{KV}$）、代码标识符（如 decode_k_nope）、中文术语（如「解耦 key」）之间换了称呼，换名处有没有就地点明「这就是前面的 X」？源码块里的标识符是否在出现处就绑回它的数学符号（而非几节后才解释）？有没有某段源码/论断依赖了要到后文才解释的概念（顺序颠倒）？\n' +
    '①–⑤ 任一命中"读不懂/卡住/换名没打通/顺序颠倒" → 记一条 issue：{problem(卡在哪、命中第几问，引用附近原文一句), suggested_fix(具体该补一句什么样的直觉/中间推导/称呼打通)}。通读一遍后 issues 为空 → pass=true；否则 pass=false。'
}
function readerFixPrompt(slug, issues) {
  const dir = ARTS + '/' + slug
  return '你的角色契约在 ' + REPO + '/.claude/agents/writer.md。任务：读者台阶四问门禁 FAIL，**只修**下列具体 issue（定点 Edit，不许整章重写、不改其余正文）：\n' + JSON.stringify(issues) + '\n' +
    '完成后自跑 `python3 ' + REPO + '/scripts/lint_paper_grounding.py ' + dir + ' --expect-primer` + `python3 ' + REPO + '/scripts/lint_formulas.py ' + dir + '/narrative/chapter.md` + `python3 ' + REPO + '/scripts/lint_chapter_structure.py ' + dir + '/narrative/chapter.md` 确保均无 BLOCKING。返回 status/note。' + ESC
}

// Write 之后、ReaderGate 之前的推导审计门禁。回环形状**直接复用 readerGateStage 的 ≤2 轮预算**
// （而非另设独立 ≤1 轮修+复审变体）——两道门禁体量相当，复用同一形状实现最简单、行为也最好预测。
// 2026-07-13 定图权归 writer:write 站返回 counts.figure_requests>0 时,派 illustrator 处理
// figure-requests.json(画/删+盲审口径由其契约兜底),再让 writer 插/删引用。失败不掐死整章:
// 记 figure_note 放行到后续门禁(requests 文件留痕,Review/Lead 兜底),避免图之败株连推导审计。
async function figureRequestStage(wr, slug) {
  if (!wr || !wr.write || wr.write.status !== 'OK') return wr
  const n = wr.write.counts && wr.write.counts.figure_requests
  if (!n) return wr
  const dir = ARTS + '/' + slug
  phase('Illustrate')
  const ill = await agent(
    '你的角色契约在 ' + REPO + '/.claude/agents/illustrator.md——先读它。任务：处理 ' + dir + '/diagrams/figure-requests.json（writer 定的图集变更，你契约「开工前」输入优先级 1）：add/replace 逐张强制流程（渲染→Read PNG 亲眼看→六项自查→登记 manifest）；drop 删文件+移除 manifest 条目；数字溯源缺失→BLOCKED。处理完条目挪 done。自跑 lint_diagram_geometry 无问题。返回 status/note。' + ESC,
    { schema: STATUS_SCHEMA, label: 'fig-request:' + slug.slice(0, 14), phase: 'Illustrate', model: MODELS.illustrate, agentType: 'general-purpose' }
  )
  if (!ill || ill.status !== 'OK') {
    log('figure-requests 处理失败（' + slug + '）：' + ((ill && (ill.blocker_reason || ill.note)) || 'agent 失败') + '——requests 留痕，继续后续门禁')
    return Object.assign({}, wr, { figure_note: 'figure-requests 未完成:' + ((ill && (ill.blocker_reason || ill.note)) || 'agent 失败') })
  }
  const ins = await agent(
    '你的角色契约在 ' + REPO + '/.claude/agents/writer.md——先读它。微任务：你此前提的图集变更已由 illustrator 完成（' + dir + '/diagrams/figure-requests.json 的 done 条目）。用 Edit 定点收尾 ' + dir + '/narrative/chapter.md：新增/替换图在其 target_section 附近插引用（先 Read PNG 看图再写图注，图注给结论）；drop 图删除其引用。**禁其他改动。**自跑 lint_chapter_structure + lint_formulas 无 BLOCKING。返回 status/note。' + ESC,
    { schema: STATUS_SCHEMA, label: 'fig-insert:' + slug.slice(0, 14), phase: 'Illustrate', model: MODELS.fix, agentType: 'general-purpose' }
  )
  if (!ins || ins.status !== 'OK') {
    return Object.assign({}, wr, { figure_note: 'fig-insert 未完成:' + ((ins && (ins.blocker_reason || ins.note)) || 'agent 失败') })
  }
  log('按需补图完成（' + slug + '）：' + n + ' 条变更落地')
  return Object.assign({}, wr, { figure_note: 'ok:' + n })
}

async function derivationCheckStage(wr, slug) {
  if (!wr || !wr.write || wr.write.status !== 'OK') {
    return Object.assign({ slug: slug, skipped: 'derivation-check' }, wr || { status: 'BLOCKED', note: 'write agent 失败(限流/崩溃)' })
  }
  let verdict = null
  let rounds = 0
  let issuesLedger = []
  for (let g = 1; g <= 2; g++) {
    phase('DerivationCheck')
    rounds = g
    verdict = await agent(derivationCheckPrompt(slug), {
      schema: DERIVATION_SCHEMA, label: 'derivation:' + slug.slice(0, 12) + ' r' + g, phase: 'DerivationCheck',
      model: MODELS.derivation, agentType: 'general-purpose',
    })
    if (!verdict) { issuesLedger = ['derivation agent 失败(限流/崩溃)']; break }
    if (verdict.pass) { issuesLedger = []; break }
    issuesLedger = verdict.issues
    if (g === 2) break // 竭尽：不再修，直接判 BLOCKED
    const fix = await agent(derivationFixPrompt(slug, issuesLedger), {
      schema: STATUS_SCHEMA, label: 'derivation-fix:' + slug.slice(0, 12) + ' r' + g, phase: 'DerivationCheck',
      model: MODELS.fix, agentType: 'general-purpose',
    })
    if (fix && fix.status === 'BLOCKED') {
      return Object.assign({}, wr, { derivation: { status: 'BLOCKED', verdict: 'FIX-BLOCKED', rounds: g, note: fix.blocker_reason, issues: issuesLedger } })
    }
    if (!fix) { log('derivation-fix agent 失败(限流/崩溃)，第 ' + g + ' 轮，仍按原 issues 再判') }
  }
  const passed = !!(verdict && verdict.pass)
  log((passed ? '推导审计 PASS' : '推导审计 FAIL（' + rounds + ' 轮竭尽）') + '：' + slug)
  return Object.assign({}, wr, {
    derivation: {
      status: passed ? 'OK' : 'BLOCKED',
      verdict: passed ? 'PASS' : 'FAIL',
      rounds: rounds,
      issues: issuesLedger,
    },
  })
}

async function readerGateStage(wr, slug) {
  if (!wr || !wr.write || wr.write.status !== 'OK') {
    return Object.assign({ slug: slug, skipped: 'reader-gate' }, wr || { status: 'BLOCKED', note: 'write agent 失败(限流/崩溃)' })
  }
  // ⚠️ 短路不变式（同上）：DerivationCheck genuinely FAIL（非上游 skip 传导）时到这里 wr.derivation.status
  // 是 'BLOCKED' 且 wr 本身无 skipped 标签——补一道 reader-gate 自己的 skip，标签属于"被跳过的这一站"。
  if (!wr.derivation || wr.derivation.status !== 'OK') {
    return Object.assign({ slug: slug, skipped: 'reader-gate' }, wr)
  }
  let verdict = null
  let rounds = 0
  let issuesLedger = []
  for (let g = 1; g <= 2; g++) {
    phase('ReaderGate')
    rounds = g
    verdict = await agent(readerPrompt(slug), {
      schema: READER_SCHEMA, label: 'reader:' + slug.slice(0, 12) + ' r' + g, phase: 'ReaderGate',
      model: MODELS.reader, agentType: 'general-purpose',
    })
    if (!verdict) { issuesLedger = ['reader agent 失败(限流/崩溃)']; break }
    if (verdict.pass) { issuesLedger = []; break }
    issuesLedger = verdict.issues
    if (g === 2) break // 竭尽：不再修，直接判 BLOCKED
    const fix = await agent(readerFixPrompt(slug, issuesLedger), {
      schema: STATUS_SCHEMA, label: 'reader-fix:' + slug.slice(0, 12) + ' r' + g, phase: 'ReaderGate',
      model: MODELS.fix, agentType: 'general-purpose',
    })
    if (fix && fix.status === 'BLOCKED') {
      return Object.assign({}, wr, { gate: { status: 'BLOCKED', reader_verdict: 'FIX-BLOCKED', rounds: g, note: fix.blocker_reason, counts: approvalCounts(slug) } })
    }
    if (!fix) { log('reader-fix agent 失败(限流/崩溃)，第 ' + g + ' 轮，仍按原 issues 再判') }
  }
  const passed = !!(verdict && verdict.pass)
  const counts = Object.assign({}, approvalCounts(slug), wr.write.counts || {})
  log((passed ? '读者门禁 PASS' : '读者门禁 FAIL（' + rounds + ' 轮竭尽）') + '：' + slug)
  return Object.assign({}, wr, {
    gate: {
      status: passed ? 'OK' : 'BLOCKED',
      reader_verdict: passed ? 'PASS' : 'FAIL',
      rounds: rounds,
      issues: issuesLedger,
      counts: counts,
    },
  })
}

if (A.phase === 'apply') {
  const results = await pipeline(
    A.chapters,
    function (slug) {
      phase('Materials')
      return agent(materialsPrompt(slug), {
        schema: STATUS_SCHEMA, label: 'materials:' + slug.slice(0, 14), phase: 'Materials',
        model: MODELS.materials, agentType: 'general-purpose',
      }).then(function (r) { return Object.assign({ slug: slug }, r || { status: 'BLOCKED', note: 'materials agent 失败(限流/崩溃)' }) })
    },
    function (mat, slug) {
      // ⚠️ 短路不变式：Object.assign 的 source(上游对象)在后——上游若已带 skipped 标签必须赢，
      // 才能把「第一个失败 stage」的标签透传到底。别把参数顺序"修"反。
      if (!mat || mat.status !== 'OK') return Object.assign({ slug: slug, skipped: 'illustrate' }, mat || { status: 'BLOCKED', note: 'materials 阶段失败' })
      phase('Illustrate')
      return agent(illustratePrompt(slug), {
        schema: STATUS_SCHEMA, label: 'illustrate:' + slug.slice(0, 14), phase: 'Illustrate',
        model: MODELS.illustrate, agentType: 'general-purpose',
      }).then(function (r) { return Object.assign({}, mat, { illustrate: r || { status: 'BLOCKED', note: 'illustrate agent 失败(限流/崩溃)' } }) })
    },
    function (ill, slug) {
      if (!ill || !ill.illustrate || ill.illustrate.status !== 'OK') return Object.assign({ slug: slug, skipped: 'write' }, ill || { status: 'BLOCKED', note: 'illustrate 阶段失败' })
      phase('Write')
      return agent(writePrompt(slug), {
        schema: WRITE_SCHEMA, label: 'write:' + slug.slice(0, 14), phase: 'Write',
        model: MODELS.write, agentType: 'general-purpose',
      }).then(function (r) { return Object.assign({}, ill, { write: r || { status: 'BLOCKED', note: 'write agent 失败(限流/崩溃)', counts: null } }) })
    },
    function (wr, slug) {
      return figureRequestStage(wr, slug)
    },
    function (fr, slug) {
      return derivationCheckStage(fr, slug)
    },
    function (dc, slug) {
      return readerGateStage(dc, slug)
    }
  )

  const all = results.filter(Boolean)
  const ok = all.filter(function (r) { return r.gate && r.gate.status === 'OK' })
  const blocked = all.filter(function (r) { return !r.gate || r.gate.status !== 'OK' })
  function blockedNote(r) {
    if (r.skipped === 'illustrate') return r.note || ''
    if (r.skipped === 'write') return (r.illustrate && r.illustrate.note) || ''
    if (r.skipped === 'derivation-check') return (r.write && r.write.note) || ''
    if (r.skipped === 'reader-gate') return (r.derivation && (r.derivation.note || JSON.stringify(r.derivation.issues || []))) || ''
    return (r.gate && r.gate.note) || (r.gate && r.gate.issues && JSON.stringify(r.gate.issues)) || ''
  }
  log('primer 施工:' + ok.length + '/' + all.length + ' 章全绿' + (blocked.length ? '；BLOCKED:' + blocked.map(function (r) { return r.slug }).join(',') : ''))
  return {
    phase: 'apply',
    ok: ok.map(function (r) { return { slug: r.slug, counts: r.gate.counts, reader_verdict: r.gate.reader_verdict, rounds: r.gate.rounds, derivation_rounds: r.derivation && r.derivation.rounds } }),
    blocked: blocked.map(function (r) { return { slug: r.slug, stage: r.skipped || 'reader-gate', note: blockedNote(r) } }),
  }
}
