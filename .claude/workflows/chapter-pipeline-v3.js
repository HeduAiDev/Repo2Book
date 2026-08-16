export const meta = {
  name: 'chapter-pipeline-v3',
  description: 'v3 逐章流水线（vllm v0.27.1 重写专用）：章条目从 pedagogy-plan.json 解析 → Dossier(吃 deepread 域卡+L2-spec) → Research(introduces 驱动) → Implement/Test → Explain(deepread data_flow 优先) → Illustrate(gen_L2 渲染+视觉自查+盲审 / 机制图) → Write(注入 WRITING-CONTRACT-v3) → Review(+认知阶梯维) → Archive（Map 站退役；含逃生舱）',
  phases: [
    { title: 'Resolve', detail: '从 pedagogy-plan.json 解析本章：slug/Part/hook/introduces/伏笔/v2 映射章/deepread 域卡/primer 论文包' },
    { title: 'Dossier', detail: 'analyst 吃 deepread 域卡 + v2 映射章 dossier(参考,行号 v0.21.0 仅线索) + 章条目，产 v3 dossier + L2-spec(l2-specs/chNN.json，schema l2-spec/1)；行号一律对 v0.27.1 现核；对抗性自核' },
    { title: 'Research', detail: 'researcher 概念查透：本章 introduces 清单驱动；v2 research/concepts.json 可复用，缺的补研' },
    { title: 'Implement', detail: 'implementer 产出 subtract-only 精简版 (TDD)；primer 章为论文忠实参考实现' },
    { title: 'Test', detail: 'tester 验证复现行为（反压闸门）' },
    { title: 'Explain', detail: 'explainer 教学素材：deepread 卡 data_flow 优先（不重复挖），精简版只补卡上没有的数' },
    { title: 'Illustrate', detail: '①跑 gen_L2 渲染本章 L2 图 → Read PNG 自查 → 独立盲审；②机制图照 v2 模式（自检+盲审门禁）。开篇图拷贝进本章 diagrams/' },
    { title: 'Write', detail: 'writer 注入 WRITING-CONTRACT-v3 全文写作：hook 开篇/先地图后细节/阶梯门禁/why 链四要素/伏笔埋收/v2 源码段迁移规则' },
    { title: 'Review', detail: '多维并行评审（v2 四维 + 新增「认知阶梯」读者代言人维），有界回环 + 终局复验' },
    { title: 'Archive', detail: 'archivist 归档 + Book Bible 记账（含伏笔埋/收状态对账 pedagogy-plan；v3 侧车账本防 v2 混淆）' },
  ],
}

// ⚠️ 与 v2 chapter-pipeline.js 的关系：本 workflow 是 vllm v3 重写专用（cartography/deepread/
// pedagogy-plan/l2-specs 全部 vllm 实例专属），v2 版继续服务其他实例，两者并存互不改。
// ⚠️ args 注入不可靠的历史坑同 v2：args 传了但解析不出章号 → 直接终止拒绝 CFG 回退；
// 完全无 args 的手工调试场景才用脚本内 CFG。
const CFG = { chapter_no: 9, instance: 'vllm' }   // 手工调试默认（Resolve 会从 plan 读全部章信息）
let A = (typeof args !== 'undefined' && args) ? args : null
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = null } }   // named 调用 args 可能字符串化(N1)
if (A && !A.chapter_no && !A.chapter_id) A = null
if (!A) {
  if (typeof args !== 'undefined' && args) {
    return { escalated: 'bad-args', note: 'args 存在但无法解析出 chapter_no/chapter_id(字符串化/字段缺失)——拒绝 CFG 回退,请检查发车参数' }
  }
  A = CFG   // 仅在完全未传 args 的手工调试场景才允许 CFG
}
// 章号归一：接受 chapter_no(9/'9') 或 chapter_id('ch9'/'ch09')
let CHNO = A.chapter_no
if (CHNO === undefined && A.chapter_id) CHNO = parseInt(String(A.chapter_id).replace(/^ch/i, ''), 10)
if (typeof CHNO === 'string') CHNO = parseInt(CHNO, 10)
A.chapter_no = CHNO
if (!Number.isInteger(CHNO) || CHNO < 1 || CHNO > 99) {
  return { escalated: 'bad-args', note: 'chapter_no 无法解析为 1-99 的整数(收到 ' + JSON.stringify(A.chapter_no) + ')' }
}
// instance 护栏：本 workflow 硬编码 vllm v3 路径（cartography/deepread/pedagogy-plan），跑别的实例必错车
if (A.instance !== 'vllm') {
  return { escalated: 'bad-args', note: 'chapter-pipeline-v3 为 vllm v3 重写专用,instance 必须显式传 "vllm"(收到 ' + JSON.stringify(A.instance) + ');其他实例请用 chapter-pipeline' }
}
const REPO = A.repo_root || 'E:/Laboratory/Repo2Book'
const SRC = A.source_root || (REPO + '/instances/vllm/source')
const CART = REPO + '/instances/vllm/book/cartography'
const BIBLE = REPO + '/instances/vllm/book/bible'

// 模型分配（2026-08-15 用户定）：全站**不指定 model = 继承主模型**；fable 只给纯视觉
// 验收 agent（读图自查/盲审），其余一切（写作/分析/评审/研究/编排）不用 fable。
// A.models.<role> 可显式覆盖（含把视觉 agent 拉回主模型）。
function mo(base, role, visual) {
  const o = Object.assign({}, base)
  const m = (A.models && A.models[role]) || (visual ? 'fable' : null)
  if (m) o.model = m
  return o
}

// 逃生舱：任何阶段发现路线/档案是错的，不许硬着头皮做错
const ESC = '\n\n**逃生舱（重要）**：如果你发现给定的路线/档案是错的——真实源码与计划不符、subtraction_plan 会破坏正确性、档案缺关键信息、无法产出忠实结果——**不要硬着头皮按错的做**。立即返回 status="BLOCKED"，blocker_reason 写清「哪里错 + 建议怎么改」。workflow 会**立刻中止**并把问题交给 Team Lead（我），我修正后从断点续跑。宁可拉闸，不要产出错误成果一路跑到底。'

const STATUS_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['status', 'note'],
  properties: { status: { type: 'string', enum: ['OK', 'BLOCKED'] }, note: { type: 'string' }, blocker_reason: { type: 'string' } },
}
const WRITE_STATUS_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['status', 'note', 'figure_requests'],
  properties: { status: { type: 'string', enum: ['OK', 'BLOCKED'] }, note: { type: 'string' }, blocker_reason: { type: 'string' }, figure_requests: { type: 'number' } },
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
const BLIND_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['all_pass', 'failures'],
  properties: {
    all_pass: { type: 'boolean' },
    failures: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['figure_id', 'problem', 'suggested_fix'],
      properties: { figure_id: { type: 'string' }, problem: { type: 'string' }, suggested_fix: { type: 'string' } } } },
  },
}

// ---------- Phase 0: Resolve（章条目解析——v3 新增：slug/伏笔/v2 映射/deepread 卡全部从 pedagogy-plan 派生） ----------
// workflow 沙箱无文件系统 API（同 v2 落盘断言的结论），章解析借一个只读 agent 带回结构化结果。
phase('Resolve')
const RESOLVE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['status', 'note', 'chapter_no', 'chapter_id', 'slug', 'l2_key', 'kind', 'part', 'part_title', 'part_hook', 'title', 'l0_zoom', 'depends_on', 'introduces', 'notes', 'is_part_opener', 'needs_l2', 'suggest_skip_impl', 'foreshadow_due', 'deepread_cards', 'v2_refs', 'paper_dir', 'key_paths'],
  properties: {
    status: { type: 'string', enum: ['OK', 'BLOCKED'] }, note: { type: 'string' }, blocker_reason: { type: 'string' },
    chapter_no: { type: 'number' }, chapter_id: { type: 'string' }, slug: { type: 'string' }, l2_key: { type: 'string' },
    kind: { type: 'string', enum: ['code', 'primer'] },
    part: { type: 'string' }, part_title: { type: 'string' }, part_hook: { type: 'string' },
    title: { type: 'string' }, l0_zoom: { type: 'string' },
    depends_on: { type: 'array', items: { type: 'number' } },
    introduces: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
    is_part_opener: { type: 'boolean' }, needs_l2: { type: 'boolean' }, suggest_skip_impl: { type: 'boolean' },
    foreshadow_due: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['id', 'text', 'action'],
      properties: { id: { type: 'string' }, text: { type: 'string' }, action: { type: 'string', enum: ['plant', 'payoff'] } } } },
    deepread_cards: { type: 'array', items: { type: 'string' } },
    v2_refs: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['chapter', 'slug', 'has_dossier', 'has_research'],
      properties: { chapter: { type: 'string' }, slug: { type: 'string' }, has_dossier: { type: 'boolean' }, has_research: { type: 'boolean' } } } },
    paper_dir: { type: 'string' },
    key_paths: { type: 'array', items: { type: 'string' } },
  },
}
const resolveV = await agent(
  '你是流水线的「章解析员」。**只做解析与核对，不修改任何文件**（wc/ls/grep/read 均可）。仓库根 ' + REPO + '。\n' +
  '解析 v3 第 ' + CHNO + ' 章。按顺序读：\n' +
  '1. ' + CART + '/pedagogy-plan.json —— 找 no=' + CHNO + ' 的章条目（title/l0_zoom/depends_on/introduces/notes）与所属 Part 条目（id/title/hook/chapters）；再扫 foreshadows：planted==' + CHNO + ' → action=plant；paid 含 ' + CHNO + ' → action=payoff。**章号不存在 → status=BLOCKED**。\n' +
  '2. ' + CART + '/outline-v3-draft.md —— 本章表格行的「v2 映射」末列（如 "ch11+ch12 重组"），只作线索。\n' +
  '3. ls ' + REPO + '/instances/vllm/artifacts/ —— 把映射提到的每个 chNN 前缀对到实际目录名（chNN-*），核对 dossier/dossier.json 与 research/concepts.json 是否存在；映射含糊（如"v2 ch26 前半"）就抽查候选章 dossier 的主题确认相关性，候选都列入 v2_refs，has_* 如实标。\n' +
  '4. ' + CART + '/deepread/ 六张域卡 —— 每张只读 domain 与 code_spine/key_files 前几条，判断哪几张覆盖本章主题（按 title/l0_zoom/notes）。**ch1 特例：六张全要**（v0→v1 演进贯穿全章）。\n' +
  '5. （仅标题含【primer】时）' + REPO + '/instances/vllm/book/papers/ 与 ' + CART + '/papers-map.json —— 按主题找本章论文包目录（**含 paper.md 才算**；注意 v3 ch32 的 DSpark 素材在 papers/ch41* 下）。找不到 → paper_dir 填空串并在 note 说明。\n' +
  '派生字段规则：\n' +
  '- chapter_id = "ch" + 两位零填章号（ch09）；slug = chapter_id + "-" + 标题主题的 ASCII kebab（小写连字符，如 ch09-engine-core-step；标题中文时按主题取英文词，短而准）；本章产物目录 = ' + REPO + '/instances/vllm/artifacts-v3/<slug>/。\n' +
  '- l2_key = "ch" + 不补零章号（ch9）——与 l2-specs/ 现有命名一致（ch9.json）。\n' +
  '- kind：标题含【primer】→ primer，否则 code。\n' +
  '- is_part_opener：本章号 == 所属 Part chapters[0]。\n' +
  '- needs_l2：默认 true；**ch1=false**（L0 全图首次给出，无 L2）；**ch40=false**（L0 点亮版属 Lead 特制）；**primer=false**（FIGURE-SYSTEM §2：primer 的 L2 形态待定，开篇用 L1）。\n' +
  '- suggest_skip_impl：ch1/ch40 必 true；primer 必 false（走论文参考实现）；其余看 v2 映射章有无 implementation/ 与本章性质——走读/鸟瞰/实战/终章倾向 true，模块深读章 false。\n' +
  '- key_paths：从所选 deepread 卡的 key_files/code_spine 挑与本章最相关的 ≤12 条规范路径。\n' +
  '返回 schema 全字段（status=OK）。',
  mo({ schema: RESOLVE_SCHEMA, label: 'resolve', phase: 'Resolve', agentType: 'general-purpose' }, 'resolve', false)
)
if (!resolveV) return { chapter: 'ch' + CHNO, escalated: 'resolve-failed', stage: 'Resolve', note: '章解析 agent 失败（限流/崩溃）——无章条目不得继续' }
if (resolveV.status === 'BLOCKED') return { escalated: 'resolve', stage: 'Resolve', reason: resolveV.blocker_reason }
const R = resolveV
log('章解析完成：' + R.chapter_id + ' ' + R.slug + '（Part ' + R.part + ' · ' + R.kind + ' · L2:' + (R.needs_l2 ? 'yes' : 'no') + ' · skip_impl:' + R.suggest_skip_impl + '）')

const CHID = R.chapter_id
const CH = REPO + '/instances/vllm/artifacts-v3/' + R.slug
const L2KEY = R.l2_key
const PRIMER = R.kind === 'primer'
const SKIP_IMPL = (A.skip_impl === undefined) ? R.suggest_skip_impl : !!A.skip_impl
const V2ROOT = REPO + '/instances/vllm/artifacts'
const V2DOSS = R.v2_refs.filter(function (x) { return x.has_dossier }).map(function (x) { return V2ROOT + '/' + x.slug + '/dossier/dossier.json' })
const V2RES = R.v2_refs.filter(function (x) { return x.has_research }).map(function (x) { return V2ROOT + '/' + x.slug + '/research/concepts.json' })
const PAPERS = R.paper_dir || (REPO + '/instances/vllm/book/papers/' + R.slug)
const DEEPCARDS = R.deepread_cards.map(function (c) { return CART + '/deepread/' + c })
const PATHS = (R.key_paths || []).join(', ')
const FORE_DUE = (R.foreshadow_due || []).map(function (f) { return f.id + '(' + f.action + ')：' + f.text }).join('；') || '（无）'
// primer 论文包前置闸（RUNBOOK §原理章发车：paper.md 须先落盘）
if (PRIMER && !R.paper_dir) {
  return { chapter: CHID, escalated: 'primer-no-paper', stage: 'Resolve', note: 'primer 章但未找到论文包（paper.md）——Lead 先落盘论文包再发车（' + R.note + '）' }
}

// 开篇图集（v3 三层缩放体系；v2 的 arch-model/roadmap/chapter-map 已退役，勿再生成）
let OPENING_DESC = ''
if (R.chapter_no === 1) {
  OPENING_DESC = '本章 = L0 全图**首次给出**：开篇放 L0-architecture.png（+ 可用 L1-part' + R.part + ' 作 Part 导览）；无 L2、无站号。'
} else if (R.needs_l2) {
  OPENING_DESC = '开篇放本章 L2 章图（L2-' + L2KEY + '.png）' + (R.is_part_opener ? '；本章是 Part ' + R.part + ' 首章——先放 L1-part' + R.part + '.png 再放 L2' : '') + '。'
} else if (PRIMER) {
  OPENING_DESC = 'primer 章（无 L2/无站号，FIGURE-SYSTEM §2 形态待定）：开篇「你在这里」用 L1-part' + R.part + '.png（Part 图）作定位，顿悟头图按 ch21 样板另行走 explainer figure-spec。'
} else {
  OPENING_DESC = '本章无 L2（' + (R.chapter_no === 40 ? '终章：L0 点亮版属 Lead 特制；若点亮版尚不存在，用 L0 原图+文字交代，勿自造新画法' : 'Lead 特批') + '）：开篇用 L0-architecture.png / L1-part' + R.part + '.png。'
}

function head(role) {
  return [
    '你的角色契约在 ' + REPO + '/.claude/agents/' + role + '.md —— **先读它**，严格遵守其中铁律；契约与本次任务的 v3 指令冲突处（图系/开篇图/写作契约等 v3 新制），以 v3 指令为准。',
    '目标源码根目录 ' + SRC + '（**行号基线 vLLM v0.27.1（6e448d0ea）**——v2 资产里的行号是 v0.21.0 的、只作线索。引用源码写规范路径 vllm/…，**绝不带** instances/vllm/source/ 前缀）。',
    '本章目录（绝对路径）：' + CH,
    '本章：v3 ' + CHID + '（Part ' + R.part + ' ' + R.part_title + '）《' + R.title + '》 · L0 缩放：' + R.l0_zoom,
    'v3 真相源目录：' + CART + '/（pedagogy-plan.json · ARCHITECTURE.md · FIGURE-SYSTEM.md · WRITING-CONTRACT-v3.md · deepread/ 六域卡 · l2-specs/）',
    PRIMER ? '本章为 **primer 原理章**：论文包在 ' + PAPERS + '/paper.md（先读它）。硬规则 2 豁免仅限本章 kind——实现是**论文忠实的小型参考实现**（非 subtract-only），替代门禁为 lint_paper_grounding。' : '',
    'vLLM 相关运行进容器：' + REPO + '/scripts/vllm_docker.sh ...（host 无 CUDA/vLLM）。',
    '',
  ].join('\n')
}

// ---------- Phase A: Dossier（真相源 = deepread 域卡 + 章条目 + v2 参考；顺产 L2-spec） + 对抗性自核 ----------
if (!A.skip_dossier) {
phase('Dossier')
const dossierV = await agent(
  head('analyst') +
  '任务：为 v3 重写产本章**档案**并**顺产 L2-spec**（图系真相源）。\n' +
  '先读（按序）：\n' +
  '1. 章条目与伏笔：' + CART + '/pedagogy-plan.json 的 no=' + CHNO + ' 条目（title/l0_zoom/depends_on/introduces/notes）+ 所属 Part（hook）+ 本章伏笔：' + FORE_DUE + '（v3 伏笔真相源 = pedagogy-plan.foreshadows，**不跑 bible.py due**——bible arc-map 还是 v2 章号的旧账，取回来必错）。\n' +
  '2. deepread 域卡（**行号已对 v0.27.1 逐行核验**，why 链/契约/数据流都在里面）：\n  ' + DEEPCARDS.join('\n  ') + '\n' +
  '3. v2 参考档案（行号 = v0.21.0 **旧线索，禁照抄**；只作迁移候选与讲法参考）：\n  ' + (V2DOSS.join('\n  ') || '（无）') + '\n' +
  (PRIMER ? '4. 论文包 ' + PAPERS + '/paper.md（先读；机制必填 paper_origin{paper,sections}，dossier 顶层写 "kind":"primer"，subtraction_plan 留空对象）。\n' : '') +
  '产出一：' + CH + '/dossier/dossier.json —— 字段：code_spine、stations（**本章站号账本** [{n,where,what}]，= 请求流经代码的顺序，喂 L2 与正文「你在这里」）、key_classes、data_flow（引用 deepread 卡 data_flow 相关段即可，不重挖）、why_chains（**精选** deepread 卡中本章相关的决策，四要素 old_design/pain/solution/cost 原样带锚点并注明来源卡）、embed_excerpts（逐字 v0.27.1 源码片段+可省略分支说明，带 vllm/...:Lxxx——**每个锚点对当前 pin 现核**，从 v2 迁移的段落行号必须重核）、theory、mechanisms[{id,name,kind:algorithm|dataflow|layout|protocol|config,source_anchors,needs_figure,needs_worked_example,difficulty:core|supporting}]（宁多登记勿漏）、subtraction_plan{delete:[{what,why_safe}], must_keep:[{symbol,why}]}（' + (SKIP_IMPL ? '本章无精简版，留空对象' : 'must_keep 把读者要学的关键符号都放进去') + '）、foreshadow_due（抄上面伏笔清单）。只描述真实源码，禁止杜撰。\n' +
  '产出二（' + (R.needs_l2 ? '本章需要' : '本章**不需要** L2-spec（' + OPENING_DESC + '），跳过此项') + '）：' + CART + '/l2-specs/' + L2KEY + '.json —— schema **l2-spec/1**（契约 ' + CART + '/FIGURE-SYSTEM.md §2，样板 ' + CART + '/l2-specs/ch9.json）：chapter/part/title/hook（本章 hook 一句话——从 Part hook + 章主题 + deepread why 链提炼成**真问题**，不许陈述句）/l0_zoom/depends_on 直拷 pedagogy-plan；l0_region 用锚名词表（full/api_band/zmq_band/engine_band/loop_box/kv_column/gpu_column/sample_column）；frame/center/components（zone/role/kind 合法值，方法签名+file:line 全部 v0.27.1 现核）/flows/loop；stations 与 dossier.stations **同一份账本**。写完跑 `python3 ' + CART + '/gen_L2.py ' + L2KEY + '`——渲染器内置校验（zone/role/kind/flows 引用/站号徽标 ⊆ 账本且账本有挂点）+ overflow 警告；不干净就改 spec 重跑到干净（**PNG 视觉验收归 Illustrate 站，你不用看图**；只许改布局性字段/精简文字，组件与站号事实不许动）。\n' +
  '完成后自跑 `python3 ' + REPO + '/scripts/lint_dossier.py ' + CH + '` 确保无 BLOCKING。**注意**：lint_dossier 暂不识别 artifacts-v3 的 source 定位、embed 逐字校验会被静默跳过——所以你必须自己逐字对 ' + SRC + ' 核 embed_excerpts（这是本章保真底线）。返回 status/note。' + ESC,
  mo({ schema: STATUS_SCHEMA, label: 'dossier', phase: 'Dossier', agentType: 'general-purpose' }, 'analyst', false)
)
if (!dossierV) return { chapter: CHID, escalated: 'dossier-failed', stage: 'Dossier', note: 'dossier agent 失败（限流/崩溃），无档案不得继续' }
if (dossierV.status === 'BLOCKED') return { escalated: 'dossier', stage: 'Dossier', reason: dossierV.blocker_reason }

const dv = await agent(
  head('analyst') +
  '任务：**独立对抗性核对** ' + CH + '/dossier/dossier.json' + (R.needs_l2 ? ' 与 ' + CART + '/l2-specs/' + L2KEY + '.json' : '') + ' 是否忠于 v0.27.1 真实源码（' + SRC + '）：\n' +
  '① embed_excerpts **逐字比对**（lint_dossier 对 artifacts-v3 抓不到 source 根会静默跳过这步——你逐段打开源文件人工核对，这是兜底）；\n' +
  '② 行号漂移：有没有把 v2 参考档案（v0.21.0 行号）的锚点照抄进来（v0.21→v0.27 核心文件 diff 巨大：gpu_model_runner/scheduler/xgrammar 面/elastic EP）？\n' +
  (R.needs_l2 ? '③ L2-spec：组件/方法名/file:line/站号账本对源码与 dossier 双向对账；l0_region 锚在词表内；hook 是真问题非陈述句；渲染器校验已过（不信可重跑 gen_L2）。\n' : '') +
  '④ mechanisms 完整性（有无漏掉读者必须懂的机制）、needs_figure/needs_worked_example/difficulty 标得对吗？\n' +
  '⑤ subtraction_plan.delete 都安全吗、must_keep 完整吗（有无遗漏读者要学的关键符号）？\n' +
  (PRIMER ? '⑥（PRIMER）确认 dossier.json 顶层有 "kind":"primer"——没有则 sound=false。\n' : '') +
  '返回 sound（是否可放行）与 problems（具体问题列表）。',
  mo({ schema: VERIFY_SCHEMA, label: 'dossier-verify', phase: 'Dossier', agentType: 'general-purpose' }, 'verify', false)
)
if (!dv) return { chapter: CHID, escalated: 'dossier-verify-failed', stage: 'Dossier', note: 'dossier 对抗性自核 agent 失败（限流/崩溃），未核对不得放行' }
if (dv.sound === false) return { escalated: 'dossier-verify', stage: 'Dossier', problems: dv.problems }
log('dossier 已通过对抗性核对')
if (A.stop_after === 'Dossier') return { chapter: CHID, stopped_after: 'Dossier', resolve: { slug: R.slug, kind: R.kind, needs_l2: R.needs_l2, deepread_cards: R.deepread_cards, v2_refs: R.v2_refs }, note: '分段发车：dossier+L2-spec 已产并过对抗自核，待 Lead 审后继续（resumeFromRunId 续跑）' }
} else {
  log('复用已有 dossier，跳过档案阶段（L2-spec 也应已在 l2-specs/）')
}

// ---------- Phase A2: Research（introduces 驱动 + v2 存量复用） ----------
if (!A.skip_research) {
  phase('Research')
  const research = await agent(
    head('researcher') +
    '任务：为本章做**概念深度研究**，产 ' + CH + '/research/concepts.json（结构见你的契约）。\n' +
    '**本章 introduces 清单（pedagogy-plan 概念首现表——阶梯门禁的门表，这些概念在本章首现处必须讲透）**：' + (R.introduces.join('、') || '（无）') + '\n' +
    '章备注：' + (R.notes || '（无）') + '\n' +
    '流程：① 逐个判断 introduces 条目与正文将出现的「初学者看描述还是不懂、需例子或背景」的非常识名词/标准记法/项目自定义模式/竞争性外部项目，哪些需要读者定向外部背景；② **先查 v2 存量**（v2 映射章已查透的可复用——拷条目、保留 sources/writer_note，版本敏感处改锚 v0.27.1 现状）：\n  ' + (V2RES.join('\n  ') || '（无）') + '\n' +
    '③ 缺的补研（WebSearch/WebFetch 真去查：如 Mooncake / DSpark 落地 / v0.27.1 新机制——每条断言给出处+版本/日期，不靠陈旧记忆）。\n' +
    '**只做读者定向的外部/常识背景，不解读本仓 pin 源码**；notation/custom_pattern 必给具体可核的例子；竞争性外部项目给各自独特特征+何时选它+一条权威链接。版本敏感的锚定本章 pin。深入浅出、刨根问底；无外部 gap 就如实少产、别硬凑（多为纯内部机制的章可能只有 0-2 项）。若无法联网立刻 status=BLOCKED 报告。' + ESC,
    mo({ schema: STATUS_SCHEMA, label: 'research', phase: 'Research', agentType: 'researcher' }, 'research', false)
  )
  if (!research) return { chapter: CHID, escalated: 'research-failed', stage: 'Research', note: 'researcher agent 失败（限流/崩溃）——无背景素材,writer 会退回薄括注,不放行' }
  if (research.status === 'BLOCKED') return { escalated: 'research', stage: 'Research', reason: research.blocker_reason }
} else { log('skip_research: 本章跳过深度研究（无需外部概念背景）') }

// ---------- Phase B/C: Implement (TDD) + Test，有界回环 ----------
let ledger = []
let implTestRounds = 0
let testV = null
if (!SKIP_IMPL) {
for (let r = 1; r <= 3; r++) {
  implTestRounds = r
  phase('Implement')
  const impl = await agent(
    head('implementer') +
    (PRIMER
      ? '任务：读 ' + CH + '/dossier/dossier.json 与 ' + PAPERS + '/paper.md，产出**论文忠实的小型参考实现**到 ' + CH + '/implementation/（NumPy/纯 CPU torch，小参数可跑），TDD 先写测试到 ' + CH + '/tests/。\n' +
        (ledger.length ? '上一轮测试失败，必须修复：\n' + ledger.join('\n') + '\n' : '') +
        '每个 def/class 标 `# PAPER: §x Eq.y`。**不发明论文没有的机制**；实现规模以「explainer 能跑出可示教轨迹」为度。\n完成后自跑 `python3 ' + REPO + '/scripts/lint_paper_grounding.py ' + CH + ' --expect-primer` 确保无 BLOCKING。返回 status/note。' + ESC
      : '任务：读 ' + CH + '/dossier/dossier.json，按 subtraction_plan 产出 **subtract-only** 精简版到 ' + CH + '/implementation/，TDD 先写测试到 ' + CH + '/tests/。\n' +
        (ledger.length ? '上一轮测试失败，必须修复：\n' + ledger.join('\n') + '\n' : '') +
        '每 def/class 标 `# SOURCE: vllm/...:Lxxx`（**行号对 v0.27.1 现核**——别照抄 v2 资产的旧行号）；删除标 `# SUBTRACTED:`。\n' +
        '**只可删除 subtraction_plan.delete 批准项；must_keep 符号必须保留；不得按己见删其他细节**（lint_fidelity 会校验 must_keep 都在）。\n' +
        '完成后自跑 `python3 ' + REPO + '/scripts/lint_fidelity.py ' + CH + '` 确保无 BLOCKING。返回 status/note。' + ESC),
    mo({ schema: STATUS_SCHEMA, label: 'implement r' + r, phase: 'Implement', agentType: 'general-purpose' }, 'implement', false)
  )
  if (!impl) { ledger.push('[round ' + r + '] implementer error（限流/崩溃）'); testV = null; continue }
  if (impl.status === 'BLOCKED') return { escalated: 'implement', stage: 'Implement', round: r, reason: impl.blocker_reason }
  phase('Test')
  testV = await agent(
    head('tester') +
    (PRIMER
      ? '任务：验证 ' + CH + '/implementation/ **忠实复现论文断言**（非复现仓库行为）：对 dossier 各机制的论文性质设计测试——分布保持类跑统计检验（固定随机种子、宽松阈值防 flaky）、恒等类做数值对照、优化类验证目标量改善。host `python3 -m pytest ' + CH + '/tests -q`。\n写 ' + CH + '/tests/test-report.json（含 verdict 与每个性质对应的论文锚 §/Eq）。全过且 lint_paper_grounding --expect-primer 无 BLOCKING → APPROVED；否则 REJECTED 且 failures 写清。'
      : '任务：验证 ' + CH + '/implementation/ 复现 dossier 记录的真实 vLLM 行为（非自洽）。\n' +
        '精简版纯测试：`python3 -m pytest ' + CH + '/tests -q`（纯控制流，无需加速器）。若精简版 import 了目标仓/加速器运行时而 host 跑不动：按 ' + REPO + '/instances/vllm/INSTANCE.md 的运行约束处理——只验可读控制流、行为以源码为准（可用 ' + REPO + '/scripts/vllm_docker.sh；测试镜像 repo2book/vllm-test:latest）。\n' +
        '写 ' + CH + '/tests/test-report.json（含 verdict；若用容器记录 docker 命令+镜像 tag+vllm 版本）。\n' +
        '全过且 lint_fidelity 无 BLOCKING → verdict=APPROVED；否则 REJECTED 且 failures 写清失败摘要。'),
    mo({ schema: TEST_SCHEMA, label: 'test r' + r, phase: 'Test', agentType: 'general-purpose' }, 'test', false)
  )
  if (testV && testV.verdict === 'APPROVED') break
  ledger.push('[round ' + r + '] ' + (testV ? testV.failures : 'tester error'))
  log('test 第 ' + r + ' 轮未过，回 implementer')
}
} else { log('skip_impl: 本章无精简版（' + (R.chapter_no === 1 ? '骨架/meta 章' : 'Resolve 判定') + '），跳过 Implement+Test') }

// 实现↔测试 3 轮仍 REJECTED → 升级 Lead，不让 explainer 用被拒实现取数
if (!SKIP_IMPL && (!testV || testV.verdict !== 'APPROVED')) return { chapter: CHID, escalated: 'test-exhausted', stage: 'Test', ledger: ledger }

// ---------- Phase C2: Explain（素材真相源：deepread data_flow 优先，不重复挖） ----------
phase('Explain')
const expl = await agent(
  head('explainer') +
  '任务：读 ' + CH + '/dossier/dossier.json（mechanisms 账本）+ 本章 deepread 卡（data_flow 已含**行号核验过的数值轨迹**——素材**优先从这里取，不重复挖**）：\n  ' + DEEPCARDS.join('\n  ') + '\n' +
  (SKIP_IMPL
    ? '本章无精简版：trace_source="manual"，manual_reason 写清（deepread 卡 data_flow 来的数字就写「deepread 卡 data_flow（行号已对 v0.27.1 核验）」；引用源码常量的数字标 file:Lxxx）。\n'
    : '+ ' + CH + '/implementation/：**只补 deepread 卡没有的数**——优先写驱动脚本跑精简版取 trace（trace_source="run"），表格每个数字必须能在 trace 或 deepread 卡里找到出处。\n') +
  '对每个 needs_worked_example 机制产出教学素材，Write 到 ' + CH + '/explainer/explainer.json；trace 原始输出与驱动脚本存 ' + CH + '/explainer/traces/。\n' +
  '每个 needs_figure 机制至少 1 个 figure-spec（claim 一句话、numbers 全带 provenance、caption_draft 给结论）。**新图须能回答「它是 L0 哪一块的放大」**（FIGURE-SYSTEM §0 一张图原则），答不出的别提 figure-spec。\n' +
  '完成后自跑 `python3 ' + REPO + '/scripts/lint_explainer.py ' + CH + '` 确保无 BLOCKING。返回 status/note。' + ESC,
  mo({ schema: STATUS_SCHEMA, label: 'explain', phase: 'Explain', agentType: 'general-purpose' }, 'explain', false)
)
if (!expl) return { chapter: CHID, escalated: 'explain-failed', stage: 'Explain', note: 'explainer agent 失败（限流/崩溃），无素材不得继续' }
if (expl.status === 'BLOCKED') return { escalated: 'explain', stage: 'Explain', reason: expl.blocker_reason }

// ---------- Phase C3a: Illustrate ①——开篇「你在这里」图（v3 新制：每章必有） ----------
// 每章开篇必有图（一张图原则）：ch1=L0 首次给出(+L1)；needs_l2=L2(+L1 若 Part 首章)；
// primer/无 L2 章回退 L1(/L0)。站内 = 渲染(或拷贝) → Read PNG 视觉自查(fable) → 独立盲审(fable)。
const NEEDS_OPENING = true
let l2Rounds = 0
let l2Ledger = []
if (NEEDS_OPENING) {
  for (let li = 1; li <= 3; li++) {
    l2Rounds = li
    phase('Illustrate')
    const l2r = await agent(
      head('illustrator') +
      '任务：本章**开篇「你在这里」图**（v3 三层缩放图系——v2 的 arch-model/roadmap/chapter-map **已退役，一律不要再生成**）。\n' +
      '本章开篇方案：' + OPENING_DESC + '\n' +
      (R.needs_l2
        ? '1. 渲染 L2：spec 已由 Dossier 站产出（' + CART + '/l2-specs/' + L2KEY + '.json）。跑 `python3 ' + CART + '/gen_L2.py ' + L2KEY + '`（渲染器输出 ' + CART + '/L2-' + L2KEY + '.svg/png）。有 overflow 警告或渲染前置校验不过 → 改 spec（只许改布局性字段/精简文字；组件/方法/站号是 dossier 真相源，不许动）→ 重跑到干净。再跑 `python3 ' + REPO + '/scripts/lint_diagram_geometry.py ' + CART + '/L2-' + L2KEY + '.svg`（cartography 路径自动 strict）确保无问题。\n'
        : '1. 本章无 L2：不跑 gen_L2、不产 spec。\n') +
      '2. 开篇图拷贝进本章：把开篇要用的图拷到 ' + CH + '/diagrams/（保留原文件名：' + (R.needs_l2 ? 'L2-' + L2KEY + '.{svg,png}' : '') + (R.chapter_no === 1 || !R.needs_l2 ? ' L0-architecture.{svg,png}' : '') + (R.is_part_opener || R.chapter_no === 1 || !R.needs_l2 ? ' L1-part' + R.part + '.{svg,png}' : '') + '——源在 ' + CART + '/）。**只拷贝，不改造**（一张图原则：L0/L1/L2 只有一个权威产出点）。\n' +
      '3. 登记 ' + CH + '/diagrams/figure-manifest.json：每张开篇图一条（figure_id 如 L2-' + L2KEY + ' / L0-architecture / L1-part' + R.part + '，claim=本章 hook 一句话（L2 用 spec.hook），numbers=[每站 where·what]（L2 用 spec.stations 账本，无站号图留空数组），blind_review 初写 PENDING）。跑 `python3 ' + REPO + '/scripts/lint_diagram_scaffolding.py ' + CH + '` 无问题。\n' +
      (l2Ledger.length ? '上一轮视觉自查/盲审问题，必须先修（改 spec 或报告 Lead——若问题出在组件/站号事实本身，用逃生舱）：\n' + l2Ledger.join('\n') + '\n' : '') +
      '返回 status/note。' + ESC,
      mo({ schema: STATUS_SCHEMA, label: 'l2-render r' + li, phase: 'Illustrate', agentType: 'general-purpose' }, 'illustrate', false)
    )
    if (!l2r) return { chapter: CHID, escalated: 'l2-render-failed', stage: 'Illustrate', round: li, note: 'L2 渲染 agent 失败（限流/崩溃）' }
    if (l2r.status === 'BLOCKED') return { escalated: 'l2-render', stage: 'Illustrate', round: li, reason: l2r.blocker_reason }
    // 视觉自查（纯读图作业 → fable）
    const l2sc = await agent(
      '你是 v3 开篇图**视觉验收员**（纯视觉作业——只看图，不改文件）。用 Read 逐张打开 ' + CH + '/diagrams/ 下的开篇图 PNG（L2-' + L2KEY + '.png / L0-architecture.png / L1-part' + R.part + '.png，存在的都看）。逐张六项自查：① 文字越界/被裁 ② 文字相撞 ③ 压框（rect-rect 盲区 linter 补不上，就靠你的眼睛）④ 箭头悬空/不贴框边 ⑤ 同源感（配色语义与整套图一致：API 蓝/ZMQ 紫/引擎橙/GPU 绿/KV 青；同一组件在不同层长得像）⑥ 站号徽标与方法名/文件名标注清晰可读。\n任何一项不过 = 该图 fail。返回 all_pass 与 failures（figure_id + problem + suggested_fix）。',
      mo({ schema: BLIND_SCHEMA, label: 'l2-selfcheck r' + li, phase: 'Illustrate', agentType: 'general-purpose' }, 'l2-selfcheck', true)
    )
    if (l2sc && !l2sc.all_pass) {
      l2Ledger = (l2sc.failures || []).map(function (f) { return '[' + f.figure_id + '] ' + f.problem + ' → ' + f.suggested_fix })
      log('开篇图视觉自查第 ' + li + ' 轮 FAIL：' + l2Ledger.length + ' 处，回渲染修复')
      continue
    }
    // 独立盲审（插画者 ≠ 审图者；纯视觉 → fable）
    const l2b = await agent(
      '你是 v3 L2 章图**盲审员**（独立于作图者——自审看不见自己的自证话术）。**只准看**：' + CH + '/diagrams/ 下开篇图 PNG（用 Read 打开）' + (R.needs_l2 ? '+ ' + CART + '/l2-specs/' + L2KEY + '.json（spec 数据）' : '') + '。**禁止**看 gen_L2.py / dossier / 正文。\n' +
      (R.needs_l2
        ? '对 L2-' + L2KEY + '.png 四步：① 只看图，用自己的话复述本章讲解路线（第 1 站 → … → 第 N 站）；② 复述路线与 spec.stations 账本逐站对照——顺序/where 对不上 = FAIL；③ 图上组件与方法名与 spec.components 逐个核对——对不上 = FAIL；④ 明显不可读 = FAIL。\n'
        : '') +
      '对 L0/L1 拷贝图只做可读性核对（④）。verdict（PASS/FAIL）与一句话用 Edit 回填 ' + CH + '/diagrams/figure-manifest.json 对应条目的 blind_review 字段。返回 all_pass 与 failures。',
      mo({ schema: BLIND_SCHEMA, label: 'l2-blind r' + li, phase: 'Illustrate', agentType: 'general-purpose' }, 'l2-blind', true)
    )
    if (l2b && l2b.all_pass) { l2Ledger = []; break }
    l2Ledger = ((l2b && l2b.failures) || [{ figure_id: 'l2-blind', problem: '盲审 agent 失败（限流/崩溃）', suggested_fix: '重试' }]).map(function (f) { return '[' + f.figure_id + '] ' + f.problem + ' → ' + f.suggested_fix })
    log('开篇图盲审第 ' + li + ' 轮 FAIL，回渲染修复')
  }
  if (l2Ledger.length) return { chapter: CHID, escalated: 'l2-exhausted', stage: 'Illustrate', failures: l2Ledger }
  log('开篇图（' + (R.needs_l2 ? 'L2 + ' : '') + (R.is_part_opener || R.chapter_no === 1 || !R.needs_l2 ? 'L1/L0 + ' : '') + '）通过视觉自查 + 独立盲审')
}

// ---------- Phase C3b: Illustrate ②——机制图（照 v2 模式：自检 + 盲审门禁，有界回环） ----------
let blindV = null
let blindLedger = []
let blindHistory = []
for (let b = 1; b <= 3; b++) {
  phase('Illustrate')
  const ill = await agent(
    head('illustrator') +
    '任务：按 ' + CH + '/explainer/explainer.json 的全部 figure_specs 绘**机制图**到 ' + CH + '/diagrams/（gen_<figure_id>.py + svg + png + figure-manifest.json 登记；manifest 里已有的开篇图条目别动）。每张图强制流程：渲染 → 用 Read 打开 PNG **亲眼看** → 六项自查全真才登记 manifest（blind_review 初写 PENDING）。\n' +
    'v3 图系铁律（FIGURE-SYSTEM §3）：机制图允许存在，但**架构性内容必须回指 L0/L1/L2**（用文字/局部指北，不许另立第二种架构画法）；图上禁止杜撰类名/方法名/站号；数字须可溯源 explainer。开篇图已由上一步产出，**不要生成 arch-model/roadmap/chapter-map**（已退役）。\n' +
    (blindLedger.length ? '上一轮盲审 FAIL，必须修复后重渲重看：\n' + blindLedger.join('\n') + '\n' : '') +
    '完成后自跑 `python3 ' + REPO + '/scripts/lint_diagram_geometry.py ' + CH + '/diagrams/*.svg` 与 `python3 ' + REPO + '/scripts/lint_diagram_scaffolding.py ' + CH + '` 确保无问题。返回 status/note。' + ESC,
    mo({ schema: STATUS_SCHEMA, label: 'illustrate r' + b, phase: 'Illustrate', agentType: 'general-purpose' }, 'illustrate', false)
  )
  if (!ill) return { chapter: CHID, escalated: 'illustrate-failed', stage: 'Illustrate', round: b, note: 'illustrator agent 失败（限流/崩溃）' }
  if (ill.status === 'BLOCKED') return { escalated: 'illustrate', stage: 'Illustrate', round: b, reason: ill.blocker_reason }
  blindV = await agent(
    '你是插图盲审员。**只准看**：' + CH + '/diagrams/figure-manifest.json 里**机制图条目**的 PNG（用 Read 打开）+ ' + CH + '/explainer/explainer.json 里对应的 figure_spec（开篇图 L2/L1/L0 条目已有独立盲审，跳过）。**禁止**看 gen_*.py 生成代码、禁止看正文章节。\n' +
    '逐张图做四步：① 只看图，用自己的话复述这张图的论点；② 与 spec.claim 对照——复述对不上 = FAIL；③ 图上每个数字与 spec.numbers 逐个核对——对不上 = FAIL；④ 明显不可读（文字重叠/箭头悬空/不知从哪看起）= FAIL。\n' +
    '把每张图的 verdict（PASS/FAIL）与 notes 用 Edit 回填 figure-manifest.json 的 blind_review 字段。\n' +
    '返回 all_pass 与 failures（每条 figure_id + problem + suggested_fix）。',
    mo({ schema: BLIND_SCHEMA, label: 'blind-review r' + b, phase: 'Illustrate', agentType: 'general-purpose' }, 'blind', true)
  )
  blindHistory.push({ round: b, failures: (blindV && blindV.failures) || [] })
  if (blindV && blindV.all_pass) break
  blindLedger = ((blindV && blindV.failures) || []).map(function (f) { return '[' + f.figure_id + '] ' + f.problem + ' → ' + f.suggested_fix })
  log('盲审第 ' + b + ' 轮 FAIL：' + blindLedger.length + ' 张图打回 illustrator')
}
if (!blindV || !blindV.all_pass) return { chapter: CHID, escalated: 'blind-review-exhausted', stage: 'Illustrate', failures: (blindV && blindV.failures) || [] }
if (A.stop_after === 'Illustrate') return { chapter: CHID, stopped_after: 'Illustrate', note: '分段发车：全部图已过自检+盲审，待 Lead 看图后继续（resumeFromRunId 续跑）' }

// ---------- Phase D: Write（注入 WRITING-CONTRACT-v3 全文快照） ----------
const CONTRACT_V3 = [
  '====== WRITING-CONTRACT-v3.md 全文（快照注入；正本 ' + CART + '/WRITING-CONTRACT-v3.md，出入以正本为准并在 note 报出） ======',
  '# v3 写作契约（vllm 重写版，Write 站注入）',
  '',
  '> 依据：spec 四要素 + 用户裁决（hook/先原理后代码/阶梯认知/不淹没细节）。',
  '> 本契约由 chapter-pipeline-v3 的 Write 站注入 writer；违反任一条 = reviewer 打回。',
  '',
  '## 1. hook 开篇（替代「本章介绍 X」）',
  '',
  '每章第一段必须是**一个真问题**（pedagogy-plan 的 hook 字段为基准，可润色但不许降格成陈述句）。',
  '读者带着问题读——问题本身要能戳中困惑（好：「俩字凭啥跑两趟跨进程快递？」坏：「本章介绍 IPC 机制」）。',
  '开篇三件套顺序：**hook → 本章地图（L2 图）→ 进入正文**。',
  '',
  '## 2. 先地图后细节',
  '',
  '- 开篇 `## 你在这里` 段放本章 **L2 章图**（Part 首章先放 L1 再 L2）。',
  '- 图注三要素（缺一 reviewer 打回）：①这块在 L0 全局图的哪里（认得感——「它就是第 1 章那张图里的 XX 块」）；',
  '  ②本章打开什么、接在哪几块已读结构上；③站号 = 请求流经代码的顺序，正文按讲解需要编排、不必照站号读。',
  '- 正文任何时刻进入新组件，先一句「现在走到 L0 图的哪一段」再展开细节。',
  '',
  '## 3. 阶梯不跳级（概念首现门禁）',
  '',
  '- 本章 `introduces` 清单（pedagogy-plan）里的概念：**在本章首次出现处讲透**（好奇专家声线，writer.md 必达物 #9）。',
  '- 清单外的非常识概念若前章已立：直接用+规范跨章链接，**不许重讲**（重讲=节奏塌）。',
  '- 若发现正文需要某概念但其前置未立（本章与前章都没讲透）：拉逃生舱 BLOCKED 报 Lead 调整大纲，**不许硬写**。',
  '',
  '## 4. why 链四要素（叙事骨架）',
  '',
  '每个设计决策的讲述必须含：**旧设计是什么 → 痛点（哪个指标被卡，有数字更好）→ v1 方案（file:line）→ 代价（诚实！）**。',
  '素材唯一来源：本章对应的 `deepread/*.json` 卡（行号已对 v0.27.1 核验）+ 本章 dossier。',
  '四要素不许砍成「v1 是怎么做的」单向陈述——没有旧设计与代价的对照，读者建立不了判断力。',
  '',
  '## 5. 伏笔埋设/回收（pedagogy-plan 驱动）',
  '',
  '- 本章若是某伏笔的 `planted` 章：在自然位置埋（不刻意），措辞留钩子（「这个机制后面还会回来」级即可）。',
  '- 本章若是 `paid` 章：回收时显式回指（「第 N 章埋的 X，现在看清了」），跨章链接规范照 lint_anchors。',
  '',
  '## 6. v2 源码段迁移规则',
  '',
  '- 允许迁移 v2 章的源码解读段（v2 映射见大纲末列），**但每一处须对 v0.27.1 现核**：',
  '  行号漂移/符号改名/机制重构（§0.5 演进速览 13 条是高发区）一律以新 pin 为准。',
  '  - v2 的行号是 v0.21.0 的——**逐段对源码确认后才可落笔**，禁止照抄行号。',
  '  - 被重构的机制（elastic EP/xgrammar 面/partial CoW 等）：按 deepread 新卡重写，不迁移旧叙述。',
  '- 迁移的是**源码事实与走读结构**，叙事骨架（hook/地图/阶梯/why 链）全部按本契约重写。',
  '',
  '## 7. 收尾回指',
  '',
  '每章末节（收尾前）：一段「L0 图点亮了哪块」+下一块预告（下一章 hook 的种子）。',
  '终章（ch40）例外：全图点亮复盘。',
  '',
  '## 8. 硬边界（继承不变）',
  '',
  '- 只做减法/零脚手架泄漏/公式四规则/lint 全绿——writer.md 与 CLAUDE.md 原样生效。',
  '- primer 章（4 个）：走 writer.md primer 分支（ch21 样板哲学：设计过的数学表达+顿悟图），无站号无 code_spine。',
  '====== 快照结束 ======',
].join('\n')

phase('Write')
let writeV = null
for (let w = 1; w <= 2 && !writeV; w++) {
if (w > 1) log('write 上轮中断(API崩)，第 ' + w + ' 轮重试：chapter.md 已存在就用 Edit 续完/校验，否则新建')
writeV = await agent(
  head('writer') +
  '任务：以**真实 v0.27.1 源码为主线**写 ' + CH + '/narrative/chapter.md（你唯一有权写它）。这是 **v3 重写**——先读 ' + CART + '/WRITING-CONTRACT-v3.md（写作契约正本），其全文快照如下（防不读；与正本有出入以正本为准并在 note 报出）：\n\n' + CONTRACT_V3 + '\n\n' +
  '素材与真相源（读）：\n' +
  '- ' + CH + '/dossier/dossier.json（why_chains/stations/embed_excerpts/mechanisms/foreshadow_due）\n' +
  '- ' + CH + '/research/concepts.json（若有）：介绍「初学者看描述还是不懂」的非常识名词/自定义模式/竞争项目/标准记法时，把查透的背景按各条 writer_note 融进正文——好奇专家声线、刨根问底、深入浅出；notation/custom_pattern 必给具体例子，竞争项目给差异+如何选+链接。例子/记法明标「说明性/外部」（非 `# SOURCE:`）；版本敏感的锚定 pin；每条外部断言按其 confidence 保守写。不能只甩术语给薄括注。\n' +
  '- ' + CH + '/explainer/explainer.json（数值轨迹/直觉/不变量）与 ' + CH + '/diagrams/（开篇图+已过盲审机制图——**先 Read 几张 PNG 亲眼看再落笔**）\n' +
  '- ' + REPO + '/instances/vllm/book/bible/voice-guide.md（声线）\n' +
  '- v2 参考档案（**只许按契约 §6 迁移源码段**，逐段对 v0.27.1 现核后才可落笔、禁照抄行号；被重构的机制按 deepread 卡重写）：\n  ' + (V2DOSS.join('\n  ') || '（无）') + '\n' +
  '结构骨架（契约 §1/2/7）：hook 真问题开篇 → `## 你在这里`（' + OPENING_DESC + '图注**三要素**齐）→ 阶梯正文（每引入一个概念先答「为什么需要它」；本章 introduces：' + (R.introduces.join('、') || '（无）') + '——首现处讲透；前章已立的不重讲）→ 收尾「L0 点亮哪块 + 下一块预告」。\n' +
  '本章伏笔（契约 §5）：' + FORE_DUE + '——按契约埋/收，**只写正文，不碰 arc-map 状态**（回写由 archivist 归档时统一做）。\n' +
  'explainer 的数值推演表进正文，表格前一行放 `<!-- trace: <mechanism_id> -->` 标记，数字一个不许改（排版随意）；difficulty=core 机制三层递进（直觉→机制→源码）。**怎么讲由你**：结构/顺序/风格/篇幅自由，**必达物要在场**。\n' +
  '**图集由你定**（定图权）：已备图不贴合可 drop、新叙事需要新图就写 ' + CH + '/diagrams/figure-requests.json（add/replace/drop，数字带溯源），并在返回值 figure_requests 填条数（无变更填 0）——workflow 会派 illustrator 处理后再让你插/删引用；**不许自己画**。新图必须答「它是 L0 哪一块的放大」，架构性内容回指 L0/L1/L2——**不许另立第二种架构画法**。\n' +
  '正文内嵌**真实源码片段**（裁剪无关分支用 `# … 省略 …`），逐段解读设计决策（why 链四要素）。\n' +
  (SKIP_IMPL
    ? '本章无精简版——以真实源码 + 开篇图为主线，不要提"精简版"。\n'
    : '精简版只作"运行看数值"的交叉验证，不是主角。\n若发现精简版缺了你要讲清的细节 → 用逃生舱拉闸（status=BLOCKED）让 implementer 补回，别将就。\n') +
  (PRIMER ? '本章四段式必达物：动机 → 数学推导（每个关键公式给论文锚 §/Eq + arXiv id）→ 小参数数值推演（explainer 素材）→ 落地（v0.27.1 真实代码锚点 + 链接对应码章）。开篇地图用 L1（Part 图），顿悟头图按 ch21 样板。\n' : '') +
  '**零脚手架泄漏**：规范 vllm/ 路径、自然标题（无 Cell N）、不提内部文件（dossier/impl-notes/L2-spec 等）。\n' +
  '完成后自跑' + (PRIMER ? '五个 linter（chapter_structure/formulas/source_grounding/trace_consistency/paper_grounding --expect-primer，primer 章不跑 fidelity）' : (SKIP_IMPL ? '四个 linter（chapter_structure/formulas/source_grounding/trace_consistency，本章无精简版故不跑 fidelity）' : '五个 linter（chapter_structure/formulas/source_grounding/fidelity/trace_consistency）')) + '均无 BLOCKING（图的 linter 归 illustrator，不用你跑）。返回 status/note/figure_requests。' + ESC,
  mo({ schema: WRITE_STATUS_SCHEMA, label: 'write r' + w, phase: 'Write', agentType: 'general-purpose' }, 'write', false)
)
}
if (!writeV) return { chapter: CHID, escalated: 'write-failed', stage: 'Write', note: 'writer 多轮失败(限流/崩溃)，无 chapter.md，不进评审' }
if (writeV.status === 'BLOCKED') return { escalated: 'write', stage: 'Write', reason: writeV.blocker_reason }

// ---------- 落盘断言（exp-2026-07-21-01：writer 报 OK ≠ chapter.md 真在盘上） ----------
const landed = await agent(
  '只做一件事，不要读文件内容、不要评价、不要修改任何东西：\n' +
  '运行 `wc -c < ' + CH + '/narrative/chapter.md` 并把结果带回。\n' +
  '- 文件不存在、或字节数 < 2000（正文不可能这么短）→ status=BLOCKED，blocker_reason 写实际情况。\n' +
  '- 否则 status=OK，note 写字节数。',
  mo({ schema: STATUS_SCHEMA, label: 'write-landed', phase: 'Write', agentType: 'general-purpose' }, 'landed', false)
)
if (!landed || landed.status === 'BLOCKED') {
  return {
    chapter: CHID, escalated: 'write-not-landed', stage: 'Write',
    reason: (landed && landed.blocker_reason) || 'chapter.md 落盘断言 agent 失败',
    note: 'writer 报 OK 但 chapter.md 未落盘/过短——不进评审，Lead 需确认 writer 是否被隔离守卫拦到 tmp 中转区'
  }
}
log('chapter.md 已落盘：' + (landed.note || ''))

// ---------- 按需补图（定图权归 writer：requests → illustrator 画/删 → 盲审 → writer 插引用） ----------
if (writeV.figure_requests > 0) {
  log('writer 提出 ' + writeV.figure_requests + ' 条图集变更，进入按需补图')
  let figBlind = null
  let figLedger = []
  for (let f = 1; f <= 3; f++) {
    phase('Write')
    const figIll = await agent(
      head('illustrator') +
      '任务：处理 ' + CH + '/diagrams/figure-requests.json（writer 定的图集变更——你契约「开工前」输入优先级 1）。add/replace 逐张走强制流程：渲染 → Read 打开 PNG 亲眼看 → 六项自查全真 → 登记 figure-manifest.json（blind_review 初写 PENDING）；drop 删图文件并移除 manifest 条目。**数字溯源缺失 → status=BLOCKED 打回，不许脑补。**v3 图系铁律照旧：架构性内容回指 L0/L1/L2、不许第二种架构画法。处理完把条目挪进 done、requests 清空。\n' +
      (figLedger.length ? '上一轮盲审 FAIL，先修复：\n' + figLedger.join('\n') + '\n' : '') +
      '完成后自跑 `python3 ' + REPO + '/scripts/lint_diagram_geometry.py ' + CH + '/diagrams/*.svg` 无问题。返回 status/note。' + ESC,
      mo({ schema: STATUS_SCHEMA, label: 'fig-request r' + f, phase: 'Write', agentType: 'general-purpose' }, 'illustrate', false)
    )
    if (!figIll) return { chapter: CHID, escalated: 'fig-request-failed', stage: 'Write', round: f, note: 'illustrator agent 失败（限流/崩溃）' }
    if (figIll.status === 'BLOCKED') return { escalated: 'fig-request', stage: 'Write', round: f, reason: figIll.blocker_reason }
    figBlind = await agent(
      '你是插图盲审员。**只准看**：' + CH + '/diagrams/figure-requests.json 的 done 条目（本轮新增/替换的图）+ figure-manifest.json 对应条目 + 每张对应 PNG（用 Read 打开）。**禁止**看 gen_*.py、禁止看正文。\n' +
      '逐张四步：① 只看图复述论点；② 与 done 条目的 claim 对照——对不上 = FAIL；③ 图上每个数字与 done 条目的 numbers 逐个核对——对不上 = FAIL；④ 明显不可读 = FAIL。verdict 回填 manifest 的 blind_review。返回 all_pass 与 failures。',
      mo({ schema: BLIND_SCHEMA, label: 'fig-blind r' + f, phase: 'Write', agentType: 'general-purpose' }, 'fig-blind', true)
    )
    if (figBlind && figBlind.all_pass) break
    figLedger = ((figBlind && figBlind.failures) || []).map(function (x) { return '[' + x.figure_id + '] ' + x.problem + ' → ' + x.suggested_fix })
    log('按需补图盲审第 ' + f + ' 轮 FAIL：' + figLedger.length + ' 张打回')
  }
  if (!figBlind || !figBlind.all_pass) return { chapter: CHID, escalated: 'fig-request-blind-exhausted', stage: 'Write', failures: (figBlind && figBlind.failures) || [] }
  const figInsert = await agent(
    head('writer') +
    '微任务：你此前对 ' + CH + '/narrative/chapter.md 提的图集变更已由 illustrator 完成并过盲审（见 ' + CH + '/diagrams/figure-requests.json 的 done 条目）。用 Edit 定点收尾：新增/替换的图在其 target_section 附近插引用（`![图注给结论](../diagrams/<id>.png)`，先 Read PNG 看图再写图注——架构性图注含三要素）；drop 的图删除其正文引用。**禁其他改动。**自跑 lint_chapter_structure + lint_formulas 无 BLOCKING。返回 status/note。' + ESC,
    mo({ schema: STATUS_SCHEMA, label: 'fig-insert', phase: 'Write', agentType: 'general-purpose' }, 'write', false)
  )
  if (!figInsert || figInsert.status === 'BLOCKED') return { chapter: CHID, escalated: 'fig-insert', stage: 'Write', reason: (figInsert && figInsert.blocker_reason) || 'fig-insert agent 失败' }
  log('按需补图完成：画/删 + 盲审 + 引用收尾')
}

// ---------- Phase E: Review（v2 四维 + 新增认知阶梯维；多维并行 → 协作回环 → 终局复验） ----------
let reviewV = null
const DIMS = [
  PRIMER
    ? 'paper-fidelity（对照 ' + PAPERS + '/paper.md 逐公式核对：推导忠实于论文？符号一致？引用锚完备？跑 lint_paper_grounding --expect-primer；evidence 必须引论文小节）'
    : 'fidelity（保真度+过度删减+零脚手架泄漏，跑 lint_fidelity/lint_source_grounding/lint_chapter_structure；行号对 v0.27.1 抽查现核）',
  'algorithm-pedagogy（逐机制对账：对 dossier.mechanisms 每条填勾选表——直觉在场？数值推演表在场且带 trace 标记？不变量论证？量化落数字？core 三层齐？先跑 lint_trace_consistency 作客观依据；输出逐机制勾选表，不是整体印象）',
  'figure-integration（先跑 lint_diagrams；然后逐张用 Read 打开 PNG 亲眼看：开篇 L2/L1/L0 图注**三要素**齐吗（L0 位置/本章打开什么/站号读法）？机制图在其机制讲解附近？图注给结论而非描述画面？正文数字与图上一致？**架构性内容是否都回指 L0/L1/L2、无第二种架构画法**？）',
  'formula-structure（公式规则+`## 你在这里` 开场+自包含+锚点/半角+IR 算子名两段点分，跑 lint_formulas/lint_anchors/lint_punct/lint_chapter_structure/lint_ir_opname）',
  'cognitive-ladder（**认知阶梯·读者代言人**——先读 ' + CART + '/WRITING-CONTRACT-v3.md 与 ' + CART + '/pedagogy-plan.json 本章条目再审：① hook 在开头吗、读完正文被回答了吗（不许开篇提问后正文忘了答）② 有没有跳级——introduces 清单概念首现处讲透了吗？用到的概念其前置在前章/本章已立吗（对照 pedagogy-plan 概念首现与 depends_on）③ 有没有淹没细节——该 zoom-in 的地方没展开、不该展开的一次倾倒 ④ 开篇图注三要素齐吗 ⑤ why 链四要素（旧设计→痛点→方案→代价）齐吗——砍成单向陈述 = 违约 ⑥ **方位词核图**（exp-2026-08-16 用户抓 ch1 左右写反）：正文/图注一切「左/右/上/下/内/外」方位描述必须与图面一致——**Read 本章开篇图逐处核对**，方位错 = blocking。违约契约条目 = blocking ⑦ **说人话扫描**（WRITING-CONTRACT-v3 §8）：grep 端着词汇（分野/擘画/赋能/抓手/纵深/护城河/打法/沉淀/拉通/颗粒度/勾勒等）+ 文学腔词（墓碑/纪念碑/故居/遗址/余晖/挽歌/落幕/谢幕/登场/尘封/活化石/丰碑/绝唱）+ 比喻堆叠句与指代不明句（主语指什么代码/机制一眼不清即 blocking 定点改））'',
]
// ---- revise 分流纯函数（exp-0716-1；行为与 lib/revise-routing.js 同构，workflow 无模块系统只能内联） ----
function dimShortName(dimStr) {
  const s = String(dimStr)
  const i = s.indexOf('（')
  return i === -1 ? s : s.slice(0, i)
}
function tagDimIssues(dimResults, dims) {
  const out = []
  for (let idx = 0; idx < dimResults.length; idx++) {
    const d = dimResults[idx]
    const dIssues = (d && d.issues) || []
    for (const i of dIssues) {
      out.push(i && i.dimension ? i : Object.assign({}, i, { dimension: dimShortName(dims[idx]) }))
    }
  }
  return out
}
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
function toFigRequestItems(figIssues) {
  return (figIssues || []).map(function (i) {
    return {
      figure_id: i.figure_id || i.figure || '',
      problem: i.problem || '',
      suggested_fix: i.suggested_fix || '',
    }
  })
}
function finalReviewDecision(reverify, lastB) {
  if (reverify && reverify.all_cleared === true) {
    return { verdict: 'APPROVED' }
  }
  if (reverify) {
    return { verdict: 'review-exhausted', issues: reverify.uncleared || [] }
  }
  return { verdict: 'review-exhausted', issues: lastB || [], note: '终局复验 agent 失败，按未清处理（不假通过）' }
}
const FINAL_VERIFY_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['all_cleared', 'uncleared'],
  properties: {
    all_cleared: { type: 'boolean' },
    uncleared: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['problem', 'evidence'], properties: { problem: { type: 'string' }, evidence: { type: 'string' }, dimension: { type: 'string' } } } },
  },
}
let lastBlocking = []
let reviewRounds = 0
for (let r = 1; r <= 3; r++) {
  reviewRounds = r
  phase('Review')
  const dimThunks = DIMS.map(function (dim) {
    return function () {
      return agent(
        head('reviewer') +
        '任务：**只**从「' + dim + '」维度评审 ' + CH + '/narrative/chapter.md（对照 ' + CH + '/dossier/dossier.json 与 v3 真相源 ' + CART + '/）。\n' +
        '机械维度先跑对应 linter（' + REPO + '/scripts/lint_*.py）。\n' +
        '协作式：每条 issue 必须给 suggested_fix + rationale，标 negotiable/blocking。该维度无 blocking issue → pass=true。',
        mo({ schema: DIM_SCHEMA, label: 'review:' + dim.slice(0, 6) + ' r' + r, phase: 'Review', agentType: 'general-purpose' }, 'review', false)
      )
    }
  })
  // 读者视角理解检查：顾问性（不门控）；primer 换「第一次读论文的工程师」人格 + 台阶四问 + 一致性第五问（可 blocking）
  const readerPrompt = PRIMER
    ? ('你是第一次读这篇论文的工程师（高级工程师，懂 Transformer 基础，但**没读过这篇论文**）。只读 ' + CH + '/narrative/chapter.md（含它引用的图），把前面章节当已读背景，**不准看论文原文、不准看源码、不准上网**。\n' +
       '逐个关键公式做台阶四问：①符号都认识吗——前文/符号表解释过？②公式前有没有直觉铺垫？③从上一步到这一步是否跳步（缺推导环节）？④是否需要先读别的论文才能看懂？\n' +
       '再做第五问·全章一致性（跨段落、非单公式）：⑤同一个量/概念是否自始至终同名？若在数学符号、代码标识符、中文术语之间换了称呼，换名处有没有就地点明「这就是前面的 X」？源码块里的标识符，是否在出现处就绑回它对应的数学符号？有没有某段源码/论断依赖了要到后文才解释的概念（顺序颠倒）？\n' +
       '①–⑤ 任一真卡住 → blocking=true（卡回 writer），并给 problem + suggested_fix + rationale；其余风格性建议 negotiable=true、blocking=false。全部台阶都过 → pass=true、issues=[]。')
    : ('你是这本书的目标读者（高级工程师，但**没读过这个仓库的源码**）。只读 ' + CH + '/narrative/chapter.md（含它引用的图），把前面章节当已读背景，**不准看源码、不准上网**。\n' +
       '站读者视角挑"读不懂/卡住"处：① 术语/缩写首现未解释；② 逻辑跳跃、缺中间步骤；③ 引入了本章没建立的概念；④ 只有结论无直觉/例子；⑤ 全章一致性：同一概念多个叫法未打通、代码标识符没就地绑回其含义/数学符号、某段依赖后文才讲的概念（顺序颠倒）。\n' +
       '每条给 problem + suggested_fix + rationale；全部 negotiable=true、blocking=false（可读性不卡章）。读得顺则 pass=true、issues=[]。')
  const readerThunk = function () {
    return agent(readerPrompt, mo({ schema: DIM_SCHEMA, label: 'review:reader r' + r, phase: 'Review', agentType: 'general-purpose' }, 'reader', false))
  }
  // PRIMER 专属：推导审计维
  const derivationThunk = function () {
    return agent(
      head('reviewer') +
      '任务：**只**从「推导审计」维度评审 ' + CH + '/narrative/chapter.md（对照 ' + CH + '/dossier/dossier.json 与论文包 ' + PAPERS + '/paper.md）。\n' +
      '你是推导审计员。对每条 $$ 推导链**亲手重推**：从假设/定义独立推到结论，再对照正文；矩阵乘法逐步核形状；数值例逐个数字重算；凡能写成 numpy/sympy 可执行断言的写脚本实跑（scratchpad 下）。\n' +
      '发现推导错误/形状不合法/数字对不上/符号用法与定义冲突 → blocking=true；风格性建议 negotiable=true、blocking=false。该维度无 blocking issue → pass=true。',
      mo({ schema: DIM_SCHEMA, label: 'review:derivation r' + r, phase: 'Review', agentType: 'general-purpose' }, 'review', false)
    )
  }
  const all = await parallel(dimThunks.concat([readerThunk]).concat(PRIMER ? [derivationThunk] : []))
  const dims = all.slice(0, DIMS.length)        // 门控只看 DIMS 个真维度（v3 = 5：v2 四维 + cognitive-ladder）
  const reader = all[DIMS.length]               // 读者检查失败(限流)不门控
  const derivation = PRIMER ? all[DIMS.length + 1] : null
  const ok = dims.filter(Boolean)
  if (ok.length < DIMS.length) return { chapter: CHID, escalated: 'review-agents-failed', stage: 'Review', round: r, note: '部分评审 agent 失败(限流/崩溃)，评审未完成，不假通过' }
  if (PRIMER && !derivation) return { chapter: CHID, escalated: 'review-agents-failed', stage: 'Review', round: r, note: '推导审计 agent 失败(限流/崩溃)，primer 章不得免审通过' }
  const readerIssues = ((reader && reader.issues) || []).map(function (i) { return Object.assign({}, i, { dimension: 'reader-comprehension' }, PRIMER ? {} : { blocking: false, negotiable: true }) })
  const derivationIssues = ((derivation && derivation.issues) || []).map(function (i) { return Object.assign({}, i, { dimension: 'derivation-audit' }) })
  const issues = ok.flatMap(function (d) { return d.issues || [] }).concat(readerIssues).concat(derivationIssues)
  const blocking = issues.filter(function (i) { return i.blocking })
  const taggedIssues = tagDimIssues(ok, DIMS).concat(readerIssues).concat(derivationIssues)
  lastBlocking = taggedIssues.filter(function (i) { return i.blocking })
  if (!ok.some(function (d) { return !d.pass }) && blocking.length === 0) {
    reviewV = { verdict: 'APPROVED', issues: issues }
    break
  }
  // ---- revise 分流：figure-integration blocking → illustrator 子回环；其余（含认知阶梯）→ writer ----
  const routed = routeIssues(taggedIssues)
  const writerPayload = routed.textIssues.concat(routed.nonBlocking)
  log('review 第 ' + r + ' 轮 REVISE：' + blocking.length + ' 个阻断项（文 ' + routed.textIssues.length + ' / 图 ' + routed.figIssues.length + '）')
  const reviseThunks = []
  if (routed.textIssues.length) reviseThunks.push(function () {
    return agent(
      head('writer') +
      '评审 REVISE（第 ' + r + ' 轮）。用 receiving-code-review skill 逐条处理（采纳或带理由反驳），改 ' + CH + '/narrative/chapter.md。**v3 契约相关 issue（认知阶梯/图注三要素/why 链/伏笔）对照 ' + CART + '/WRITING-CONTRACT-v3.md 修**：\n' +
      JSON.stringify(writerPayload) + '\n完成后自跑' + (PRIMER ? '五个 linter（chapter_structure/formulas/source_grounding/trace_consistency/paper_grounding --expect-primer，primer 章不跑 fidelity）' : (SKIP_IMPL ? '四个 linter（chapter_structure/formulas/source_grounding/trace_consistency）' : '五个 linter（chapter_structure/formulas/source_grounding/fidelity/trace_consistency）')) + '均无 BLOCKING。返回 status/note。' + ESC,
      mo({ schema: STATUS_SCHEMA, label: 'revise r' + r, phase: 'Review', agentType: 'general-purpose' }, 'write', false)
    )
  })
  if (routed.figIssues.length) reviseThunks.push(async function () {
    const figFix = await agent(
      head('illustrator') +
      '任务：修复评审 figure-integration 维的阻断项（writer 无权动图，这些只有你能修）。清单（与 figure-requests done 条目同构）：\n' +
      JSON.stringify(toFigRequestItems(routed.figIssues)) +
      '\n逐张强制流程：改 ' + CH + '/diagrams/ 下 gen 脚本 → 重渲染 → 转 PNG → **用 Read 打开 PNG 亲眼看** → 六项自查全真 → 更新 figure-manifest.json 对应条目（blind_review 回写 PENDING 待重盲审）。**禁止即兴加示意数字**（数字须可溯源 explainer/正文）。v3 图系铁律照旧（回指 L0/L1/L2、无第二种架构画法）。完成后自跑 `python3 ' + REPO + '/scripts/lint_diagram_geometry.py ' + CH + '/diagrams/*.svg` 与 `python3 ' + REPO + '/scripts/lint_diagram_scaffolding.py ' + CH + '` 无问题。返回 status/note。' + ESC,
      mo({ schema: STATUS_SCHEMA, label: 'revise-fig r' + r, phase: 'Review', agentType: 'general-purpose' }, 'illustrate', false)
    )
    if (!figFix || figFix.status === 'BLOCKED') return { escalate: { escalated: 'review-revise-fig', stage: 'Review', round: r, reason: (figFix && figFix.blocker_reason) || 'revise-fig agent 失败（限流/崩溃）' } }
    const figBlind2 = await agent(
      '你是插图盲审员（revise 步内闭合的再验证——修图后 manifest 为 PENDING，进入下一轮评审前必须在此清掉）。**只准看**：本轮被修各图的 PNG（用 Read 打开）+ 对应评审 issue 清单：\n' +
      JSON.stringify(toFigRequestItems(routed.figIssues)) +
      '\n**禁止**看 gen_*.py、禁止看正文。逐张核对 issue 是否真被修复（图上内容/数字与 suggested_fix 相符），verdict 回填 ' + CH + '/diagrams/figure-manifest.json 的 blind_review。返回 all_pass 与 failures。',
      mo({ schema: BLIND_SCHEMA, label: 'revise-fig-blind r' + r, phase: 'Review', agentType: 'general-purpose' }, 'fig-blind', true)
    )
    return { blind: figBlind2 }
  })
  const reviseOut = await parallel(reviseThunks)
  let oi = 0
  let rev = null
  if (routed.textIssues.length) { rev = reviseOut[oi]; oi++ }
  const figOut = routed.figIssues.length ? reviseOut[oi] : null
  if (figOut && figOut.escalate) return Object.assign({ chapter: CHID }, figOut.escalate)
  if (rev && rev.status === 'BLOCKED') return { escalated: 'review-revise', stage: 'Review', round: r, reason: rev.blocker_reason }
  if (routed.figIssues.length && figOut && figOut.blind && figOut.blind.all_pass) {
    const cap = await agent(
      head('writer') +
      '微任务：本轮评审的图侧阻断项已由 illustrator 修复并过盲审再验证。用 Edit 定点核对 ' + CH + '/narrative/chapter.md 中这些图的引用与图注是否需同步（图注数字/结论与新图一致，架构性图注保三要素；无需改动则不改）。清单：\n' +
      JSON.stringify(toFigRequestItems(routed.figIssues)) +
      '\n**禁其他改动。**自跑 lint_chapter_structure + lint_formulas 无 BLOCKING。返回 status/note。' + ESC,
      mo({ schema: STATUS_SCHEMA, label: 'revise-fig-caption r' + r, phase: 'Review', agentType: 'general-purpose' }, 'write', false)
    )
    if (cap && cap.status === 'BLOCKED') return { escalated: 'review-revise', stage: 'Review', round: r, reason: cap.blocker_reason }
  } else if (routed.figIssues.length) {
    log('revise 图轨第 ' + r + ' 轮盲审未全过，留给下一轮评审复检')
  }
  reviewV = { verdict: 'REVISE', issues: taggedIssues }
}

// 评审 3 轮仍未过 → 终局复验（只核上轮 blocking 清单对照最新稿，全清 → APPROVED 免逃逸）
if (reviewV && reviewV.verdict !== 'APPROVED') {
  phase('Review')
  const reverify = await agent(
    '你是终局复验员（轻量：只核清单、不开新维度全审）。对下面每条上轮 blocking 项，逐条对照**当前最新**文件核实是否已解决：正文 ' + CH + '/narrative/chapter.md、图 ' + CH + '/diagrams/（PNG 用 Read 打开亲眼看、manifest 的 blind_review 状态），必要时跑对应 linter（' + REPO + '/scripts/lint_*.py）。清单：\n' +
    JSON.stringify(lastBlocking) +
    '\n已解决=从清单去掉；未解决=进 uncleared（problem + evidence 引最新稿/最新图证据）。全部解决 → all_cleared=true。宁严勿宽：拿不准的算未解决。',
    mo({ schema: FINAL_VERIFY_SCHEMA, label: 'review-final-verify', phase: 'Review', agentType: 'general-purpose' }, 'review', false)
  )
  const fdec = finalReviewDecision(reverify, lastBlocking)
  if (fdec.verdict === 'APPROVED') {
    log('终局复验：上轮阻断项已全部在最新稿解决 → APPROVED')
    reviewV = { verdict: 'APPROVED', issues: reviewV.issues }
  } else {
    return { chapter: CHID, test: testV, escalated: 'review-exhausted', stage: 'Review', issues: fdec.issues, note: fdec.note }
  }
}
if (A.stop_after === 'Review') return { chapter: CHID, stopped_after: 'Review', review: reviewV, note: '分段发车：评审已 APPROVED，待 Lead 终审后归档（resumeFromRunId 续跑）' }

// ---------- Phase F: Archive（v3：+ 伏笔埋/收对账 pedagogy-plan + v3 侧车账本防 v2 混淆） ----------
// （v2 的 Map 站已退役：本章地图=L2 章图，已由 Illustrate 站产出并过盲审。）
const reviewJson = JSON.stringify(reviewV || { overall_verdict: 'UNKNOWN', issues: [] })
const runLedgerObj = {
  chapter_id: CHID, book: 'v3', slug: R.slug, part: R.part,
  kind: PRIMER ? 'primer' : (SKIP_IMPL ? 'meta' : 'code'),
  impl_test_rounds: implTestRounds, impl_test_ledger: ledger,
  write_review_rounds: reviewRounds,
  l2_rounds: l2Rounds,
  blind_rounds: blindHistory.length, blind_failures: blindHistory,
  foreshadow_due: R.foreshadow_due,
  escalated: null,
}
if (A.skip_archive) {
  return { chapter: CHID, needs_archive: true, review_verdict: (reviewV && reviewV.verdict) || 'UNKNOWN', review_report: reviewV, run_ledger: runLedgerObj, note: '并行模式:Review 已过,L2 已过盲审,Bible/trace 待 Lead 串行归档' }
}
phase('Archive')
const runLedger = JSON.stringify(runLedgerObj)
const archiveTask = head('archivist') +
  '任务一(务必先做)：把下面这个完整 review 对象**原样**写入 ' + CH + '/reviews/review-report.json（保留 verdict 与全部 issues，不要删改）：\n' +
  reviewJson + '\n' +
  '任务一b：把这个 run-ledger 对象**原样**写入 ' + CH + '/reviews/run-ledger.json：\n' + runLedger + '\n' +
  '任务二（Book Bible——**v3 侧车账本，防 v2 混淆**：v2 已封版但同 bible 目录，v2 条目章号是旧编号；v3 登记一律写 v3 侧车文件，不动 v2 同名文件）：\n' +
  '(a) 新术语 → ' + BIBLE + '/glossary-v3.json（结构同 glossary.json；{term, 中文译名, 一句释义, 首现章=' + CHID + '}）。\n' +
  '(b) 本章新建立的核心概念 → ' + BIBLE + '/concepts-v3.json（{"<术语>":"' + CHID + '"}；gap 审计判「前章已立」用）。\n' +
  '(c) 精简版新接口 → ' + BIBLE + '/interfaces-v3.json（结构同 v2 interfaces；**不要用 bible.py iface**——它写 v2 账本；无精简版跳过）。\n' +
  '(d) **伏笔埋/收对账（v3 新增）**：对照 ' + CART + '/pedagogy-plan.json 的 foreshadows——本章应埋：' + ((R.foreshadow_due || []).filter(function (f) { return f.action === 'plant' }).map(function (f) { return f.id + ' ' + f.text }).join('；') || '无') + '；本章应收：' + ((R.foreshadow_due || []).filter(function (f) { return f.action === 'payoff' }).map(function (f) { return f.id + ' ' + f.text }).join('；') || '无') + '。逐组核对正文实际埋/收（review 已把关），状态写 ' + BIBLE + '/foreshadow-v3.json（{id,text,planted:{ch,done},paid:[{ch,done}]}，与 pedagogy-plan 对账：正文没埋到/没收到的**如实标 done:false 并在返回 note 上报**，不许硬改）。\n' +
  '(e) 本章图登记 → ' + BIBLE + '/figures.json（**同文件追加**——figure_id 章内唯一且 v3 新 id（如 L2-' + L2KEY + '）不与 v2 撞；把 ' + CH + '/diagrams/figure-manifest.json 每张图登记，含开篇图与机制图）。自跑 `python3 ' + REPO + '/scripts/lint_figures_registered.py ' + CH + '`（**显式传 chapter_dir**——无参 --all 模式经 instance.py 只扫 artifacts/，看不见 artifacts-v3）。\n' +
  '任务三：在 ' + REPO + '/instances/vllm/trace/ 记 delivery（文件名带 v3 前缀如 v3-' + CHID + '-delivery.md，不覆盖 v2 记录）并在 state.json 增量记 v3 进度（新增 "v3" 子对象，不动 v2 字段）。返回一句话状态。'
let archV = null
for (let a = 1; a <= 2 && !archV; a++) {
  if (a > 1) log('archive 上轮中断(API崩)，第 ' + a + ' 轮重试：已写的(review-report.json/v3 账本/trace)校验后跳过，只补未完成的')
  archV = await agent(
    archiveTask + (a > 1 ? '\n注意：这是重试。先检查 review-report.json 是否已存在且为合法完整 JSON、v3 账本是否已登记本章条目，已做的别重复，只补未完成的。' : ''),
    mo({ label: 'archive r' + a, phase: 'Archive', agentType: 'general-purpose' }, 'archive', false)
  )
}

return { chapter: CHID, slug: R.slug, test: testV, review: reviewV, l2_rounds: l2Rounds }
