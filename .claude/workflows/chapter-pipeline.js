export const meta = {
  name: 'chapter-pipeline',
  description: '单章流水线：档案(真相源)→只做减法实现+测试→真源码解读叙事→多维协作评审→归档（含逃生舱：任一阶段发现路线错可拉闸升级）',
  phases: [
    { title: 'Dossier', detail: 'analyst 深读真实源码产出共享档案，并对抗性自核' },
    { title: 'Implement', detail: 'implementer 产出 subtract-only 精简版 (TDD)' },
    { title: 'Test', detail: 'tester 验证复现 vLLM 行为（反压闸门）' },
    { title: 'Explain', detail: 'explainer 跑精简版取数值轨迹，产教学素材+figure-spec' },
    { title: 'Illustrate', detail: 'illustrator 绘图：视觉自查回环+盲审门禁' },
    { title: 'Write', detail: 'writer 以真实源码为主线写章节（内嵌源码+Roadmap）' },
    { title: 'Review', detail: '多维并行协作评审，有界回环' },
    { title: 'Map', detail: '评审收敛后，illustrator 产出「本章地图」源码剖面图（自检+盲审回环 ≤2 轮）+ writer 微任务插图引与选读指引' },
    { title: 'Archive', detail: 'archivist 归档 + 回写 Book Bible' },
  ],
}

// ⚠️ 本环境实测 Workflow 的 args 注入不可靠（args 未到达脚本）→ 用脚本内 CFG 作可靠配置；
// args 可用时优先 args。换章节时改 CFG（或修复 args 注入后直接传 args）。
const CFG = {
  chapter_id: 'ch01',
  slug: 'ch01-birdseye-oot-plugin',
  instance: 'vllm-ascend',
  focus: '【元/鸟瞰章·全书开篇，最后写】鸟瞰：一个不改 vLLM 却接管整条执行路径的 OOT 插件——建立全书心智模型 + 三支柱总览 + 每章 zoom-in 地图。**这是 meta 概览章、不是逐行源码解读**：只挑三支柱各一个最小源码锚点内嵌（其余各章展开），主线是「把读者的心智模型搭起来」。**核心心智模型**：vllm-ascend **不 fork、不改 vLLM 源码一行**，而是作为一个 OOT（树外）插件，靠三支柱把昇腾 NPU 顶替进 vLLM 执行路径的每一站。**三支柱**：**(1) 安装期挂入（entry points）**——setup.py 的 entry_points 声明 `vllm.platform_plugins`（ascend=vllm_ascend:register）+ `vllm.general_plugins`（register_connector/register_model_loader/register_model 等）；pip install 后 vLLM 启动自动发现并调这些 register，昇腾无需改 vLLM 就被挂进去。**(2) 运行期分发（NPUPlatform）**——register() 只返回一个字符串 "vllm_ascend.platform.NPUPlatform"；NPUPlatform(Platform) 覆写一堆 get_*_cls 工厂钩子（get_attn_backend_cls/get_communicator_cls/get_compiler_cls…），vLLM 每次要某个组件就问 current_platform，于是拿到的都是昇腾版——一个平台类接管所有分发。**(3) 两段式 monkey-patch（adapt_patch）**——patch/__init__.py 分 platform 段（worker 启动前、NPUPlatform.pre_register_and_update 里 adapt_patch(is_global_patch=True)）+ worker 段（每个 worker __init__ 里 adapt_patch(is_global_patch=False)）；对那些「没留工厂钩子、改不动」的地方，靠 import 副作用打补丁改写。**全书地图**：三支柱之上，vLLM 处处留扩展点，昇腾往每个扩展点登记实现——后面 29 章就是这套机制在每一站（入口/平台/配置 P1、通信/并行 P1-2、KV/PD/调度 P3-4、worker/runner/单步前向 P4、注意力 P5、算子/编译 P6、量化/采样/投机/加载 P7）的 zoom-in；每章开头 Roadmap 的「你在这里」就挂在这张全书地图上。**姊妹篇约定**：本书是「昇腾如何改」，与讲 vLLM 原版的姊妹书配对——正文对照基座 vLLM v0.21.0 讲「同一处 vLLM 原版长什么样、昇腾改成什么样」。**核心立意**：读完本章，读者应握住一句话——「OOT 插件 = 装上就被发现（entry points）+ 一个平台类接管分发（NPUPlatform）+ 改不动的地方两段式打补丁（monkey-patch）+ 往每个扩展点登记昇腾实现（注册/薄壳继承/必要时特化）」，然后带着这张地图去读后面每一章。【姊妹篇：对照基座 vLLM v0.21.0 在 instances/vllm/source，pairs vllm/platforms/interface.py（Platform 基类——NPUPlatform 覆写它的工厂钩子）+ vllm/plugins（entry point 发现机制）+ vllm/platforms/__init__.py（current_platform 懒加载分发）；正文写规范 vllm_ascend/… 与 vllm/… 路径（含 setup.py），绝不带 instances/.../source/ 前缀；这是 meta 概览章、**skip_impl（无精简版）**——只内嵌三支柱的最小真源码锚点（setup.py entry_points / __init__.py register+adapt_patch / platform.py NPUPlatform 类头）作自包含，控制流 host 可读；不要求跑精简版】',
  highlight: 'ch01',
  source_root: '/mnt/e/Laboratory/Repo2Book/instances/vllm-ascend/source',
  repo_root: '/mnt/e/Laboratory/Repo2Book',
  skip_dossier: true,
  skip_impl: true,
  paths: ['setup.py', 'vllm_ascend/__init__.py', 'vllm_ascend/platform.py', 'vllm_ascend/patch/__init__.py'],
}
let A = (typeof args !== 'undefined' && args) ? args : null
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = null } }   // named 调用 args 可能字符串化(N1)
if (A && !A.chapter_id) A = null
if (!A) {
  if (typeof args !== 'undefined' && args) {
    // args 传了但解析不出 chapter_id——拒绝 CFG 回退(曾静默错车生产别章烧 692k tokens),直接终止
    return { escalated: 'bad-args', note: 'args 存在但无法解析出 chapter_id(字符串化/字段缺失)——拒绝 CFG 回退,请检查发车参数' }
  }
  A = CFG   // 仅在完全未传 args 的手工调试场景才允许 CFG
}
const REPO = A.repo_root || '/mnt/e/Laboratory/Repo2Book'
// instance 必传护栏(与 chapter_id 同级):曾因 args 漏传 instance 静默默认 'vllm',使
// CH/PAPERS/roadmap/bible/trace 全指向错实例——ch06-08 均踩,仅靠子 agent 每次自纠 CH 才没错车
// (脆弱且污染每个 agent 提示词)。凡 args 发车必须显式带 instance,否则拉闸;仅手工无参调试(A===CFG)用 CFG.instance。
if (A !== CFG && !A.instance) return { escalated: 'bad-args', note: 'args 发车缺 instance 字段——拒绝静默默认 vllm(曾致错实例路径),请显式传 instance(如 "triton")' }
const INST = A.instance || CFG.instance
const SRC = A.source_root || (REPO + '/instances/' + INST + '/source')
const CH = REPO + '/instances/' + INST + '/artifacts/' + A.slug
const HL = A.highlight || A.subsystem || ''
// 模型分配(spec §7:全流水线 opus/sonnet,不继承主会话模型;args.models 可覆盖)
const MODELS = Object.assign(
  { analyst: 'opus', verify: 'opus', implement: 'sonnet', test: 'sonnet', explain: 'opus', illustrate: 'sonnet', blind: 'sonnet', write: 'opus', review: 'sonnet', archive: 'sonnet' },
  A.models || {})
const PRIMER = A.kind === 'primer'
// 2026-07-13 用户定:原理章 writer 能用 fable5 就用 fable5(ch21 对比验证:主线定理/悬崖诊断/事实修正均更优)。args.models.write 可覆盖。
if (PRIMER && !(A.models && A.models.write)) MODELS.write = 'fable'
const PAPERS = REPO + '/instances/' + INST + '/book/papers/' + A.slug
const PATHS = (A.paths || []).join(', ')

// 逃生舱：任何阶段发现路线/档案是错的，不许硬着头皮做错
const ESC = '\n\n**逃生舱（重要）**：如果你发现给定的路线/档案是错的——真实源码与计划不符、subtraction_plan 会破坏正确性、档案缺关键信息、无法产出忠实结果——**不要硬着头皮按错的做**。立即返回 status="BLOCKED"，blocker_reason 写清「哪里错 + 建议怎么改」。workflow 会**立刻中止**并把问题交给 Team Lead（我），我修正后从断点续跑。宁可拉闸，不要产出错误成果一路跑到底。'

function head(role) {
  return [
    '你的角色契约在 ' + REPO + '/.claude/agents/' + role + '.md —— **先读它**，严格遵守其中所有铁律。',
    '目标源码根目录 ' + SRC + '（引用源码写**规范路径**，如 ' + INST + ' 实例写 vllm_ascend/… / 对照基座写 vllm/…，**绝不带** instances/' + INST + '/source/ 前缀）。',
    '本章目录（绝对路径）：' + CH,
    '本章：' + A.chapter_id + ' 《' + A.focus + '》',
    PRIMER ? '本章为 **primer 原理章**：论文包在 ' + PAPERS + '/paper.md（先读它）。硬规则 2 豁免仅限本章 kind——实现是**论文忠实的小型参考实现**（非 subtract-only），替代门禁为 lint_paper_grounding。' : '',
    'vLLM 相关运行进容器：' + REPO + '/scripts/vllm_docker.sh ...（host 无 CUDA/vLLM）。',
    '',
  ].join('\n')
}

const STATUS_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['status', 'note'],
  properties: { status: { type: 'string', enum: ['OK', 'BLOCKED'] }, note: { type: 'string' }, blocker_reason: { type: 'string' } },
}
// 2026-07-13 定图权归 writer:Write 站专用 schema,figure_requests=writer 写进 diagrams/figure-requests.json 的变更条数(0=图集无变动)
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

// ---------- Phase A: Dossier (真相源) + 对抗性自核 ----------
if (!A.skip_dossier) {
phase('Dossier')
const dossierV = await agent(
  head('analyst') +
  '任务：深读真实源码（重点：' + PATHS + '），产出本章**档案**并 Write 到 ' + CH + '/dossier/dossier.json。\n' +
  '先跑 `python3 ' + REPO + '/scripts/bible.py due ' + A.chapter_id + '`，结果放入 foreshadow_due。\n' +
  'dossier.json 字段：code_spine、embed_excerpts(逐字真实源码片段+可省略分支说明，带 vllm/...:Lxxx)、key_classes、data_flow、design_decisions、theory、subtraction_plan{delete:[{what,why_safe}], must_keep:[{symbol,why} 可检测符号]}、mechanisms[{id,name,kind:algorithm|dataflow|layout|protocol|config,source_anchors,needs_figure,needs_worked_example,difficulty:core|supporting}](v3 账本：宁多登记勿漏)、foreshadow_due。\n' +
  (PRIMER ? '本章是 primer 原理章：深读论文包 ' + PAPERS + '/paper.md 与落地代码（' + PATHS + '）。dossier.json 顶层写 "kind":"primer"；每个机制**必填** paper_origin{paper,sections}；embed_excerpts 可含论文公式（带 §/Eq 锚）与代码双源；subtraction_plan 留空对象（primer 不做减法）。\n' : '') +
  'must_keep 要把"读者需理解、writer 需讲清"的符号都放进去（宁多留勿误删）。只描述真实源码，禁止杜撰。完成后自跑 `python3 ' + REPO + '/scripts/lint_dossier.py ' + CH + '` 确保无 BLOCKING。返回 status/note。' + ESC,
  { schema: STATUS_SCHEMA, label: 'dossier', phase: 'Dossier', agentType: 'general-purpose', model: MODELS.analyst }
)
if (!dossierV) return { chapter: A.chapter_id, escalated: 'dossier-failed', stage: 'Dossier', note: 'dossier agent 失败（限流/崩溃），无档案不得继续' }
if (dossierV.status === 'BLOCKED') return { escalated: 'dossier', stage: 'Dossier', reason: dossierV.blocker_reason }

const dv = await agent(
  head('analyst') +
  '任务：**独立对抗性核对** ' + CH + '/dossier/dossier.json 是否忠于真实源码：路线是否正确？embed_excerpts 是否逐字、file:Lxxx 是否准确？subtraction_plan.delete 是否都安全、must_keep 是否完整（有无遗漏读者要学的关键符号）？\n' +
  'mechanisms 是否完整——有无漏掉读者必须懂的机制？needs_figure/needs_worked_example/difficulty 标得对吗？\n' +
  (PRIMER ? '（PRIMER）确认 dossier.json 顶层有 "kind":"primer"——没有则 sound=false。\n' : '') +
  '返回 sound（是否可放行）与 problems（具体问题列表）。',
  { schema: VERIFY_SCHEMA, label: 'dossier-verify', phase: 'Dossier', agentType: 'general-purpose', model: MODELS.verify }
)
if (!dv) return { chapter: A.chapter_id, escalated: 'dossier-verify-failed', stage: 'Dossier', note: 'dossier 对抗性自核 agent 失败（限流/崩溃），未核对不得放行' }
if (dv.sound === false) return { escalated: 'dossier-verify', stage: 'Dossier', problems: dv.problems }
log('dossier 已通过对抗性核对')
} else {
  log('复用已人工审核的 dossier，跳过档案阶段')
}

// ---------- Phase B/C: Implement (TDD) + Test，有界回环 ----------
let ledger = []
let implTestRounds = 0
let testV = null
if (!A.skip_impl) {
for (let r = 1; r <= 3; r++) {
  implTestRounds = r
  phase('Implement')
  const impl = await agent(
    head('implementer') +
    (PRIMER
      ? '任务：读 ' + CH + '/dossier/dossier.json 与 ' + PAPERS + '/paper.md，产出**论文忠实的小型参考实现**到 ' + CH + '/implementation/（NumPy/纯 CPU torch，小参数可跑），TDD 先写测试到 ' + CH + '/tests/。\n' +
        (ledger.length ? '上一轮测试失败，必须修复：\n' + ledger.join('\n') + '\n' : '') +
        '每个 def/class 标 `# PAPER: §x Eq.y`（对标码章的 # SOURCE）。**不发明论文没有的机制**；实现规模以「explainer 能跑出可示教轨迹」为度。\n完成后自跑 `python3 ' + REPO + '/scripts/lint_paper_grounding.py ' + CH + ' --expect-primer` 确保无 BLOCKING。返回 status/note。' + ESC
      : '任务：读 ' + CH + '/dossier/dossier.json，按 subtraction_plan 产出 **subtract-only** 精简版到 ' + CH + '/implementation/，TDD 先写测试到 ' + CH + '/tests/。\n' +
        (ledger.length ? '上一轮测试失败，必须修复：\n' + ledger.join('\n') + '\n' : '') +
        '每 def/class 标 `# SOURCE: vllm/...:Lxxx`；删除标 `# SUBTRACTED:`。\n' +
        '**只可删除 subtraction_plan.delete 批准项；must_keep 符号必须保留；不得按己见删其他细节**（lint_fidelity 会校验 must_keep 都在）。\n' +
        '完成后自跑 `python3 ' + REPO + '/scripts/lint_fidelity.py ' + CH + '` 确保无 BLOCKING。返回 status/note。' + ESC),
    { schema: STATUS_SCHEMA, label: 'implement r' + r, phase: 'Implement', agentType: 'general-purpose', model: MODELS.implement }
  )
  if (!impl) { ledger.push('[round ' + r + '] implementer error（限流/崩溃）'); testV = null; continue }
  if (impl.status === 'BLOCKED') return { escalated: 'implement', stage: 'Implement', round: r, reason: impl.blocker_reason }
  phase('Test')
  testV = await agent(
    head('tester') +
    (PRIMER
      ? '任务：验证 ' + CH + '/implementation/ **忠实复现论文断言**（非复现仓库行为）：对 dossier 各机制的论文性质设计测试——分布保持类跑统计检验（固定随机种子、宽松阈值防 flaky）、恒等类做数值对照、优化类验证目标量改善。host `python3 -m pytest ' + CH + '/tests -q`。\n写 ' + CH + '/tests/test-report.json（含 verdict 与每个性质对应的论文锚 §/Eq）。全过且 lint_paper_grounding --expect-primer 无 BLOCKING → APPROVED；否则 REJECTED 且 failures 写清。'
      : '任务：验证 ' + CH + '/implementation/ 复现 dossier 记录的真实 vLLM 行为（非自洽）。\n' +
        '精简版纯测试：`python3 -m pytest ' + CH + '/tests -q`（纯控制流，无需加速器）。若精简版 import 了目标仓/加速器运行时而 host 跑不动：按 ' + REPO + '/instances/' + INST + '/INSTANCE.md 的运行约束处理——只验可读控制流、行为以源码为准（vLLM 实例可用 ' + REPO + '/scripts/vllm_docker.sh）。\n' +
        '写 ' + CH + '/tests/test-report.json（含 verdict；若用容器记录 docker 命令+镜像 tag+vllm 版本）。\n' +
        '全过且 lint_fidelity 无 BLOCKING → verdict=APPROVED；否则 REJECTED 且 failures 写清失败摘要。'),
    { schema: TEST_SCHEMA, label: 'test r' + r, phase: 'Test', agentType: 'general-purpose', model: MODELS.test }
  )
  if (testV && testV.verdict === 'APPROVED') break
  ledger.push('[round ' + r + '] ' + (testV ? testV.failures : 'tester error'))
  log('test 第 ' + r + ' 轮未过，回 implementer')
}
} else { log('skip_impl: 本章无精简版（方法论/概览章），跳过 Implement+Test') }

// 实现↔测试 3 轮仍 REJECTED → 升级 Lead，不让 explainer 用被拒实现取数
if (!A.skip_impl && (!testV || testV.verdict !== 'APPROVED')) return { chapter: A.chapter_id, escalated: 'test-exhausted', stage: 'Test', ledger: ledger }

// ---------- Phase C2: Explain（素材真相源：数值轨迹 + figure-spec） ----------
phase('Explain')
const expl = await agent(
  head('explainer') +
  '任务：读 ' + CH + '/dossier/dossier.json（mechanisms 账本）与 ' + CH + '/implementation/（若有），对每个 needs_worked_example 机制产出教学素材，Write 到 ' + CH + '/explainer/explainer.json；trace 原始输出与驱动脚本存 ' + CH + '/explainer/traces/。\n' +
  (A.skip_impl
    ? '本章无精简版：trace_source="manual"，manual_reason 写清；引用源码常量的数字标 file:Lxxx。\n'
    : '优先写驱动脚本跑精简版取 trace（trace_source="run"）——表格每个数字必须能在 trace 里找到。\n') +
  '每个 needs_figure 机制至少 1 个 figure-spec（claim 一句话、numbers 全带 provenance、caption_draft 给结论）。\n' +
  '完成后自跑 `python3 ' + REPO + '/scripts/lint_explainer.py ' + CH + '` 确保无 BLOCKING。返回 status/note。' + ESC,
  { schema: STATUS_SCHEMA, label: 'explain', phase: 'Explain', agentType: 'general-purpose', model: MODELS.explain }
)
if (!expl) return { chapter: A.chapter_id, escalated: 'explain-failed', stage: 'Explain', note: 'explainer agent 失败（限流/崩溃），无素材不得继续' }
if (expl.status === 'BLOCKED') return { escalated: 'explain', stage: 'Explain', reason: expl.blocker_reason }

// ---------- Phase C3: Illustrate（绘图 → 视觉自查 → 盲审门禁，有界回环） ----------
const BLIND_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['all_pass', 'failures'],
  properties: {
    all_pass: { type: 'boolean' },
    failures: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['figure_id', 'problem', 'suggested_fix'],
      properties: { figure_id: { type: 'string' }, problem: { type: 'string' }, suggested_fix: { type: 'string' } } } },
  },
}
let blindV = null
let blindLedger = []
let blindHistory = []
for (let b = 1; b <= 3; b++) {
  phase('Illustrate')
  const ill = await agent(
    head('illustrator') +
    '任务：按 ' + CH + '/explainer/explainer.json 的全部 figure_specs 绘图到 ' + CH + '/diagrams/（gen_<figure_id>.py + svg + png + figure-manifest.json）。每张图强制流程：渲染 → 用 Read 打开 PNG **亲眼看** → 六项自查全真才登记 manifest（blind_review 初写 PENDING）。\n' +
    '并生成本章 roadmap：`python3 ' + REPO + '/instances/' + INST + '/book/assets/roadmap/roadmap.py --highlight "' + HL + '" --out ' + CH + '/diagrams/roadmap.svg`（开篇「你在这里」窄长条横幅），rsvg-convert -z 2 转 PNG（**勿用 ImageMagick convert**）。\n' +
    (blindLedger.length ? '上一轮盲审 FAIL，必须修复后重渲重看：\n' + blindLedger.join('\n') + '\n' : '') +
    '完成后自跑 `python3 ' + REPO + '/scripts/lint_diagram_geometry.py ' + CH + '/diagrams/*.svg` 确保无问题。返回 status/note。' + ESC,
    { schema: STATUS_SCHEMA, label: 'illustrate r' + b, phase: 'Illustrate', agentType: 'general-purpose', model: MODELS.illustrate }
  )
  if (!ill) return { chapter: A.chapter_id, escalated: 'illustrate-failed', stage: 'Illustrate', round: b, note: 'illustrator agent 失败（限流/崩溃）' }
  if (ill.status === 'BLOCKED') return { escalated: 'illustrate', stage: 'Illustrate', round: b, reason: ill.blocker_reason }
  blindV = await agent(
    '你是插图盲审员。**只准看**：' + CH + '/diagrams/figure-manifest.json 列出的每张 PNG（用 Read 打开图片文件）+ ' + CH + '/explainer/explainer.json 里对应的 figure_spec。**禁止**看 gen_*.py 生成代码、禁止看正文章节。\n' +
    '逐张图做四步：① 只看图，用自己的话复述这张图的论点；② 与 spec.claim 对照——复述对不上 = FAIL；③ 图上每个数字与 spec.numbers 逐个核对——对不上 = FAIL；④ 明显不可读（文字重叠/箭头悬空/不知从哪看起）= FAIL。\n' +
    '把每张图的 verdict（PASS/FAIL）与 notes 用 Edit 回填 figure-manifest.json 的 blind_review 字段。\n' +
    '返回 all_pass 与 failures（每条 figure_id + problem + suggested_fix）。',
    { schema: BLIND_SCHEMA, label: 'blind-review r' + b, phase: 'Illustrate', agentType: 'general-purpose', model: MODELS.blind }
  )
  blindHistory.push({ round: b, failures: (blindV && blindV.failures) || [] })
  if (blindV && blindV.all_pass) break
  blindLedger = ((blindV && blindV.failures) || []).map(function (f) { return '[' + f.figure_id + '] ' + f.problem + ' → ' + f.suggested_fix })
  log('盲审第 ' + b + ' 轮 FAIL：' + blindLedger.length + ' 张图打回 illustrator')
}
if (!blindV || !blindV.all_pass) return { chapter: A.chapter_id, escalated: 'blind-review-exhausted', stage: 'Illustrate', failures: (blindV && blindV.failures) || [] }
log('插图全部通过视觉自查 + 盲审')

// ---------- Phase D: Write (真源码主线) ----------
phase('Write')
let writeV = null
for (let w = 1; w <= 2 && !writeV; w++) {
if (w > 1) log('write 上轮中断(API崩)，第 ' + w + ' 轮重试：chapter.md 已存在就用 Edit 续完/校验，否则新建')
writeV = await agent(
  head('writer') +
  '任务：以**真实目标源码为主线**写 ' + CH + '/narrative/chapter.md（你唯一有权写它）。\n' +
  '读 dossier、implementation、' + REPO + '/instances/' + INST + '/book/bible/voice-guide.md，并跑 `python3 ' + REPO + '/scripts/bible.py due ' + A.chapter_id + '`。\n' +
  '素材已备好：读 ' + CH + '/explainer/explainer.json（数值轨迹/直觉/不变量）与 ' + CH + '/diagrams/（已过盲审的图 + roadmap.png——先 Read 几张 PNG 看图长什么样再落笔）。**怎么讲由你**：结构/顺序/风格/篇幅自由。**必达物要在场**：difficulty=core 机制三层递进（直觉→机制→源码）；explainer 的数值推演表进正文，表格前一行放 `<!-- trace: <mechanism_id> -->` 标记，数字一个不许改（排版随意）；每张仍贴合的已验收图被引用且在其机制讲解附近；开场引用 roadmap.png（开篇「你在这里」窄长条横幅）；**ch01/鸟瞰章**另需 book-map.png（详细全书地图:各 Part×各章+primer 徽标）。**图集由你定**(契约必达物3)：已备图不贴合可 drop、新叙事需要新图就写 ' + CH + '/diagrams/figure-requests.json(add/replace/drop,数字带溯源)，并在返回值 figure_requests 填条数(无变更填 0)——workflow 会派 illustrator 处理后再让你插/删引用；**不许自己画**。\n' +
  (PRIMER ? '本章四段式必达物：动机 → 数学推导（**每个关键公式给论文锚 §/Eq + arXiv id**）→ 小参数数值推演（explainer 素材）→ 落地（vllm_ascend 真实代码锚点 + 链接对应码章）。\n' : '') +
  '正文内嵌**真实源码片段**(裁剪无关分支用 `# … 省略 …`)，逐段解读设计决策。' +
  (A.skip_impl
    ? '本章无精简版（方法论/概览章）——以真实源码 + 架构图为主线，不要提"精简版"。\n'
    : '精简版只作"运行看数值"的交叉验证，不是主角。\n若发现精简版缺了你要讲清的细节 → 用逃生舱拉闸（status=BLOCKED）让 implementer 补回，别将就。\n') +
  '按 dossier.foreshadow_due 在正文写好应埋伏笔的铺垫、应回收伏笔的回指兑现——**只写正文,不碰 arc-map 状态**(不要跑 `bible.py payoff --resolve`)。伏笔的 resolved 回写由 archivist 归档时统一做:writer 在 Write 阶段改 arc-map,会在本章尚未提交时就把 resolved 泄漏进并行发车的别章 commit(exp-0717-2)。\n' +
  '**零脚手架泄漏**：规范 vllm/ 路径、自然标题(无 Cell N)、不提内部文件。\n' +
  '完成后自跑' + (PRIMER ? '五个 linter（chapter_structure/formulas/source_grounding/trace_consistency/paper_grounding --expect-primer，primer 章不跑 fidelity）' : (A.skip_impl ? '四个 linter（chapter_structure/formulas/source_grounding/trace_consistency，本章无精简版故不跑 fidelity）' : '五个 linter（chapter_structure/formulas/source_grounding/fidelity/trace_consistency）')) + '均无 BLOCKING（图的 linter 归 illustrator，不用你跑）。返回 status/note。' + ESC,
  { schema: WRITE_STATUS_SCHEMA, label: 'write r' + w, phase: 'Write', agentType: 'general-purpose', model: MODELS.write }
)
}
if (!writeV) return { chapter: A.chapter_id, escalated: 'write-failed', stage: 'Write', note: 'writer 多轮失败(限流/崩溃)，无 chapter.md，不进评审' }
if (writeV && writeV.status === 'BLOCKED') return { escalated: 'write', stage: 'Write', reason: writeV.blocker_reason }

// ---------- Phase D2: 按需补图（2026-07-13 定图权归 writer：writer 提 requests → illustrator 画/删 → 盲审 → writer 插引用） ----------
if (writeV.figure_requests > 0) {
  log('writer 提出 ' + writeV.figure_requests + ' 条图集变更，进入按需补图')
  let figBlind = null
  let figLedger = []
  for (let f = 1; f <= 3; f++) {
    phase('Write')
    const figIll = await agent(
      head('illustrator') +
      '任务：处理 ' + CH + '/diagrams/figure-requests.json（writer 定的图集变更——你契约「开工前」输入优先级 1）。add/replace 逐张走强制流程：渲染 → Read 打开 PNG 亲眼看 → 六项自查全真 → 登记 figure-manifest.json（blind_review 初写 PENDING）；drop 删图文件并移除 manifest 条目。**数字溯源缺失 → status=BLOCKED 打回，不许脑补。**处理完把条目挪进 done、requests 清空。\n' +
      (figLedger.length ? '上一轮盲审 FAIL，先修复：\n' + figLedger.join('\n') + '\n' : '') +
      '完成后自跑 `python3 ' + REPO + '/scripts/lint_diagram_geometry.py ' + CH + '/diagrams/*.svg` 无问题。返回 status/note。' + ESC,
      { schema: STATUS_SCHEMA, label: 'fig-request r' + f, phase: 'Write', agentType: 'general-purpose', model: MODELS.illustrate }
    )
    if (!figIll) return { chapter: A.chapter_id, escalated: 'fig-request-failed', stage: 'Write', round: f, note: 'illustrator agent 失败（限流/崩溃）' }
    if (figIll.status === 'BLOCKED') return { escalated: 'fig-request', stage: 'Write', round: f, reason: figIll.blocker_reason }
    figBlind = await agent(
      '你是插图盲审员。**只准看**：' + CH + '/diagrams/figure-requests.json 的 done 条目（本轮新增/替换的图）+ figure-manifest.json 对应条目 + 每张对应 PNG（用 Read 打开）。**禁止**看 gen_*.py、禁止看正文。\n' +
      '逐张四步：① 只看图复述论点；② 与 done 条目的 claim 对照——对不上 = FAIL；③ 图上每个数字与 done 条目的 numbers 逐个核对——对不上 = FAIL；④ 明显不可读 = FAIL。verdict 回填 manifest 的 blind_review。返回 all_pass 与 failures。',
      { schema: BLIND_SCHEMA, label: 'fig-blind r' + f, phase: 'Write', agentType: 'general-purpose', model: MODELS.blind }
    )
    if (figBlind && figBlind.all_pass) break
    figLedger = ((figBlind && figBlind.failures) || []).map(function (x) { return '[' + x.figure_id + '] ' + x.problem + ' → ' + x.suggested_fix })
    log('按需补图盲审第 ' + f + ' 轮 FAIL：' + figLedger.length + ' 张打回')
  }
  if (!figBlind || !figBlind.all_pass) return { chapter: A.chapter_id, escalated: 'fig-request-blind-exhausted', stage: 'Write', failures: (figBlind && figBlind.failures) || [] }
  const figInsert = await agent(
    head('writer') +
    '微任务：你此前对 ' + CH + '/narrative/chapter.md 提的图集变更已由 illustrator 完成并过盲审（见 ' + CH + '/diagrams/figure-requests.json 的 done 条目）。用 Edit 定点收尾：新增/替换的图在其 target_section 附近插引用（`![图注给结论](../diagrams/<id>.png)`，先 Read PNG 看图再写图注）；drop 的图删除其正文引用。**禁其他改动。**自跑 lint_chapter_structure + lint_formulas 无 BLOCKING。返回 status/note。' + ESC,
    { schema: STATUS_SCHEMA, label: 'fig-insert', phase: 'Write', agentType: 'general-purpose', model: MODELS.write }
  )
  if (!figInsert || figInsert.status === 'BLOCKED') return { chapter: A.chapter_id, escalated: 'fig-insert', stage: 'Write', reason: (figInsert && figInsert.blocker_reason) || 'fig-insert agent 失败' }
  log('按需补图完成：画/删 + 盲审 + 引用收尾')
}

// ---------- Phase E: Review (多维并行 → 协作回环) ----------
let reviewV = null
const DIMS = [
  PRIMER
    ? 'paper-fidelity（对照 ' + PAPERS + '/paper.md 逐公式核对：推导忠实于论文？符号一致？引用锚完备？跑 lint_paper_grounding --expect-primer；evidence 必须引论文小节）'
    : 'fidelity（保真度+过度删减+零脚手架泄漏，跑 lint_fidelity/lint_source_grounding/lint_chapter_structure）',
  'algorithm-pedagogy（逐机制对账：对 dossier.mechanisms 每条填勾选表——直觉在场？数值推演表在场且带 trace 标记？不变量论证？量化落数字？core 三层齐？先跑 lint_trace_consistency 作客观依据；输出逐机制勾选表，不是整体印象）',
  'figure-integration（先跑 lint_diagrams；然后逐张用 Read 打开 PNG 亲眼看：图在其机制讲解附近？图注给结论而非描述画面？正文数字与图上一致？图对读懂机制真有帮助？）',
  'formula-structure（公式规则+Roadmap 开场+自包含+锚点/半角，跑 lint_formulas/lint_anchors/lint_punct/lint_chapter_structure）',
]
let reviewRounds = 0
for (let r = 1; r <= 3; r++) {
  reviewRounds = r
  phase('Review')
  const dimThunks = DIMS.map(function (dim) {
    return function () {
      return agent(
        head('reviewer') +
        '任务：**只**从「' + dim + '」维度评审 ' + CH + '/narrative/chapter.md（对照 ' + CH + '/dossier/dossier.json 与 Book Bible）。\n' +
        '机械维度先跑对应 linter（' + REPO + '/scripts/lint_*.py）。\n' +
        '协作式：每条 issue 必须给 suggested_fix + rationale，标 negotiable/blocking。该维度无 blocking issue → pass=true。',
        { schema: DIM_SCHEMA, label: 'review:' + dim.slice(0, 6) + ' r' + r, phase: 'Review', agentType: 'general-purpose', model: MODELS.review }
      )
    }
  })
  // 读者视角理解检查：非 primer 为 book-only 顾问性（不门控）；
  // primer 原理章换「第一次读论文的工程师」人格 + 逐公式台阶四问 + 全章一致性第五问，可 blocking=true 硬门禁。
  // 模型：primer=opus / 非 primer=sonnet（exp-0712-3：一致性检测需先「察觉多名共指」，haiku 会对割裂给假通过——
  // 不是「真读者认了」，是「弱模型没察觉有东西可认」，故不再用 haiku）。
  const readerPrompt = PRIMER
    ? ('你是第一次读这篇论文的工程师（高级工程师，懂 Transformer 基础，但**没读过这篇论文**）。只读 ' + CH + '/narrative/chapter.md（含它引用的图），把前面章节当已读背景，**不准看论文原文、不准看源码、不准上网**。\n' +
       '逐个关键公式做台阶四问：①符号都认识吗——前文/符号表解释过？②公式前有没有直觉铺垫？③从上一步到这一步是否跳步（缺推导环节）？④是否需要先读别的论文才能看懂？\n' +
       '再做第五问·全章一致性（跨段落、非单公式）：⑤同一个量/概念是否自始至终同名？若在数学符号（如 $c^{KV}$）、代码标识符（如 decode_k_nope）、中文术语（如「解耦 key」）之间换了称呼，换名处有没有就地点明「这就是前面的 X」？源码块里的标识符，是否在出现处就绑回它对应的数学符号（而非几节后才解释）？有没有某段源码/论断依赖了要到后文才解释的概念（顺序颠倒）？\n' +
       '①–⑤ 任一真卡住（答案是"没有/是/需要/换了名却没打通/顺序颠倒"）→ blocking=true（卡回 writer），并给 problem + suggested_fix + rationale；其余风格性建议 negotiable=true、blocking=false。全部台阶都过 → pass=true、issues=[]。')
    : ('你是这本书的目标读者（高级工程师，但**没读过这个仓库的源码**）。只读 ' + CH + '/narrative/chapter.md（含它引用的图），把前面章节当已读背景，**不准看源码、不准上网**。\n' +
       '站读者视角挑"读不懂/卡住"处：① 术语/缩写首现未解释；② 逻辑跳跃、缺中间步骤；③ 引入了本章没建立的概念（如某测试设施/外部机制）；④ 只有结论无直觉/例子；⑤ 全章一致性：同一概念多个叫法未打通、代码标识符没就地绑回其含义/数学符号、某段依赖后文才讲的概念（顺序颠倒）。\n' +
       '每条给 problem + suggested_fix（补一句话/一个例子让读者跟上）+ rationale；全部 negotiable=true、blocking=false（可读性不卡章）。读得顺则 pass=true、issues=[]。')
  const readerThunk = function () {
    return agent(
      readerPrompt,
      { model: PRIMER ? 'opus' : 'sonnet', schema: DIM_SCHEMA, label: 'review:reader r' + r, phase: 'Review', agentType: 'general-purpose' }
    )
  }
  // PRIMER 专属：推导审计维（仿 reader 维之形状——独立 thunk、非 PRIMER 完全不跑；
  // 现有 DIMS/dimThunks/readerPrompt 逐字不动，保 resume 缓存）。
  const derivationThunk = function () {
    return agent(
      head('reviewer') +
      '任务：**只**从「推导审计」维度评审 ' + CH + '/narrative/chapter.md（对照 ' + CH + '/dossier/dossier.json 与论文包 ' + PAPERS + '/paper.md）。\n' +
      '你是推导审计员。对每条 $$ 推导链**亲手重推**：从假设/定义独立推到结论，再对照正文；矩阵乘法逐步核形状；数值例逐个数字重算；凡能写成 numpy/sympy 可执行断言的写脚本实跑（scratchpad 下）。\n' +
      '发现推导错误/形状不合法/数字对不上/符号用法与定义冲突 → blocking=true；风格性建议 negotiable=true、blocking=false。该维度无 blocking issue → pass=true。',
      { schema: DIM_SCHEMA, label: 'review:derivation r' + r, phase: 'Review', agentType: 'general-purpose', model: MODELS.review }
    )
  }
  const all = await parallel(dimThunks.concat([readerThunk]).concat(PRIMER ? [derivationThunk] : []))
  const dims = all.slice(0, DIMS.length)        // 门控只看 4 个真维度
  const reader = all[DIMS.length]               // 读者检查失败(限流)不门控
  const derivation = PRIMER ? all[DIMS.length + 1] : null   // 仅 PRIMER 时跑；非 PRIMER 恒为 null
  const ok = dims.filter(Boolean)
  if (ok.length < DIMS.length) return { chapter: A.chapter_id, escalated: 'review-agents-failed', stage: 'Review', round: r, note: '部分评审 agent 失败(限流/崩溃)，评审未完成，不假通过' }
  // 推导审计是 primer 硬门禁：审计 agent 崩了不许静默放行，与其他维同等对待
  if (PRIMER && !derivation) return { chapter: A.chapter_id, escalated: 'review-agents-failed', stage: 'Review', round: r, note: '推导审计 agent 失败(限流/崩溃)，primer 章不得免审通过' }
  // 非 primer：强制 blocking:false/negotiable:true（顾问性，行为与此前一致）。
  // primer：保留 reader 自报的 blocking/negotiable（台阶四问的硬卡点纳入下面的 blocking 聚合)。
  const readerIssues = ((reader && reader.issues) || []).map(function (i) { return Object.assign({}, i, { dimension: 'reader-comprehension' }, PRIMER ? {} : { blocking: false, negotiable: true }) })
  // derivation 维只在 PRIMER 存在；blocking/negotiable 原样保留 agent 自报（不盖 blocking:false）。
  const derivationIssues = ((derivation && derivation.issues) || []).map(function (i) { return Object.assign({}, i, { dimension: 'derivation-audit' }) })
  const issues = ok.flatMap(function (d) { return d.issues || [] }).concat(readerIssues).concat(derivationIssues)
  const blocking = issues.filter(function (i) { return i.blocking })
  if (!ok.some(function (d) { return !d.pass }) && blocking.length === 0) {
    reviewV = { verdict: 'APPROVED', issues: issues }
    break
  }
  log('review 第 ' + r + ' 轮 REVISE：' + blocking.length + ' 个阻断项，回 writer')
  const rev = await agent(
    head('writer') +
    '评审 REVISE（第 ' + r + ' 轮）。用 receiving-code-review skill 逐条处理（采纳或带理由反驳），改 ' + CH + '/narrative/chapter.md：\n' +
    JSON.stringify(issues) + '\n完成后自跑' + (PRIMER ? '五个 linter（chapter_structure/formulas/source_grounding/trace_consistency/paper_grounding --expect-primer，primer 章不跑 fidelity）' : (A.skip_impl ? '四个 linter（chapter_structure/formulas/source_grounding/trace_consistency）' : '五个 linter（chapter_structure/formulas/source_grounding/fidelity/trace_consistency）')) + '均无 BLOCKING。返回 status/note。' + ESC,
    { schema: STATUS_SCHEMA, label: 'revise r' + r, phase: 'Review', agentType: 'general-purpose', model: MODELS.write }
  )
  if (rev && rev.status === 'BLOCKED') return { escalated: 'review-revise', stage: 'Review', round: r, reason: rev.blocker_reason }
  reviewV = { verdict: 'REVISE', issues: issues }
}

// 评审 3 轮仍未过 → 升级 Lead（兑现"同一问题 >3 轮自动升级"承诺），不静默归档 REVISE
if (reviewV && reviewV.verdict !== 'APPROVED') {
  return { chapter: A.chapter_id, test: testV, escalated: 'review-exhausted', stage: 'Review', issues: reviewV.issues }
}

// ---------- Phase F: Map（评审收敛后产出「本章地图」，站内自检+盲审回环 ≤2 轮） ----------
const MAP_BLIND_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['pass', 'problem', 'suggested_fix'],
  properties: { pass: { type: 'boolean' }, problem: { type: 'string' }, suggested_fix: { type: 'string' } },
}
let mapBlindV = null
let mapLedger = []
let mapHistory = []
for (let m = 1; m <= 2; m++) {
  phase('Map')
  const mapV = await agent(
    head('illustrator') +
    '任务：为本章画「本章地图」（源码剖面图）。先读你的契约 ' + REPO + '/.claude/agents/illustrator.md 的「本章地图」节（每章一次，Map 站移交给你）——严格照其输入/模板/节点预算/自然标题章规则/自查项执行，不要自己发明格式。模板：' + REPO + '/.claude/skills/svg-diagram/references/example-chapter-map.py。\n' +
    '输入：**定稿** ' + CH + '/narrative/chapter.md（真实节结构，站牌与徽标须对得上）+ ' + CH + '/dossier/dossier.json（mechanisms 锚点，代码符号须对得上）' + (PRIMER ? '+ ' + PAPERS + '/（论文包，primer 章节点=论文概念）' : '') + '。\n' +
    (mapLedger.length ? '上一轮盲审 FAIL，必须修复：\n' + mapLedger.join('\n') + '\n' : '') +
    '产出 ' + CH + '/diagrams/chapter-map.py（生成脚本，坐标全由循环/常量算，零手写魔数）+ chapter-map.svg + chapter-map.png（`rsvg-convert -z 2`，勿用 ImageMagick convert）。渲染后**用 Read 打开 PNG 亲眼看**，六项自查全真才登记进 ' + CH + '/diagrams/figure-manifest.json（`blind_review` 初写 PENDING）。\n' +
    '完成后自跑 `python3 ' + REPO + '/scripts/lint_chapter_map.py ' + CH + '`（无 --require，试点期）与 `python3 ' + REPO + '/scripts/lint_diagram_geometry.py ' + CH + '/diagrams/chapter-map.svg`，均确保无问题（不过就自己改图重渲，别指望盲审替你挑）。返回 status/note。' + ESC,
    { schema: STATUS_SCHEMA, label: 'map r' + m, phase: 'Map', agentType: 'general-purpose', model: MODELS.illustrate }
  )
  if (!mapV) return { chapter: A.chapter_id, escalated: 'map-failed', stage: 'Map', round: m, note: 'illustrator agent 失败（限流/崩溃），无法产出本章地图' }
  if (mapV.status === 'BLOCKED') return { escalated: 'map', stage: 'Map', round: m, reason: mapV.blocker_reason }
  mapBlindV = await agent(
    '你是「本章地图」的盲审员，形状同插图盲审：**只准看** ' + CH + '/diagrams/chapter-map.png（用 Read 打开）与 ' + CH + '/narrative/chapter.md 的标题结构（只扫标题定位，不需通读全文）。**禁止**看 gen 生成代码、禁止看 dossier.json。\n' +
    '四步核对：① 只看图，用自己的话复述这张「源码剖面图」讲的是哪条路线（入口→…→出口）；② 图上每个 §N.M 徽标（或自然标题站牌）逐一核对能在正文找到对应标题——对不上 = FAIL；③ 图上代码符号是否像本章会讲的真实符号（不是看着编出来的）——明显杜撰 = FAIL；④ 明显不可读（文字重叠/箭头悬空/不知从哪看起）= FAIL。\n' +
    '把 verdict（PASS/FAIL）与一句话回填 ' + CH + '/diagrams/figure-manifest.json 中 chapter-map 条目的 blind_review 字段（用 Edit）。\n' +
    '返回 pass（是否放行）；FAIL 时 problem+suggested_fix 必须具体；PASS 时两者留空字符串。',
    { schema: MAP_BLIND_SCHEMA, label: 'map-blind r' + m, phase: 'Map', agentType: 'general-purpose', model: MODELS.blind }
  )
  mapHistory.push({ round: m, pass: !!(mapBlindV && mapBlindV.pass), problem: (mapBlindV && mapBlindV.problem) || '', suggested_fix: (mapBlindV && mapBlindV.suggested_fix) || '' })
  if (mapBlindV && mapBlindV.pass) break
  mapLedger = [(mapBlindV ? (mapBlindV.problem + ' → ' + mapBlindV.suggested_fix) : 'map-blind agent error（限流/崩溃）')]
  log('本章地图第 ' + m + ' 轮盲审 FAIL，回 illustrator')
}
if (!mapBlindV || !mapBlindV.pass) return { chapter: A.chapter_id, escalated: 'map-exhausted', stage: 'Map', history: mapHistory }
log('本章地图通过自检 + 盲审')

// Map 站第二步：writer 微任务——插图引 + 选读指引（不改其余正文）
phase('Map')
let mapInsertV = null
for (let wi = 1; wi <= 2 && !mapInsertV; wi++) {
  if (wi > 1) log('map-insert 上轮中断(API崩)，第 ' + wi + ' 轮重试：图引已插入就跳过，只补未完成的')
  mapInsertV = await agent(
    head('writer') +
    '任务：本章地图已产出并过盲审（' + CH + '/diagrams/chapter-map.png）。读你的契约 ' + REPO + '/.claude/agents/writer.md 「必达物」第 7 条「开篇『本章地图』」（插图引用位置与选读指引写法，先读它、严格照做，不要自己发明格式）。\n' +
    '在 ' + CH + '/narrative/chapter.md 用 **Edit 定点修改**（不许 Write 整文件覆盖）：hook 段之后、第一个 `## ` 标题之前插入图引 + 1–2 句自然措辞的选读指引。只做这一处插入，不改其余正文。\n' +
    '完成后自跑' + (PRIMER ? '五个 linter（chapter_structure/formulas/source_grounding/trace_consistency/paper_grounding --expect-primer，primer 章不跑 fidelity）' : (A.skip_impl ? '四个 linter（chapter_structure/formulas/source_grounding/trace_consistency，本章无精简版故不跑 fidelity）' : '五个 linter（chapter_structure/formulas/source_grounding/fidelity/trace_consistency）')) + '，加上 `python3 ' + REPO + '/scripts/lint_chapter_map.py ' + CH + ' --require`，均无 BLOCKING。返回 status/note。' + ESC,
    { schema: STATUS_SCHEMA, label: 'map-insert r' + wi, phase: 'Map', agentType: 'general-purpose', model: 'sonnet' /* 插图微任务降档 */ }
  )
}
if (!mapInsertV) return { chapter: A.chapter_id, escalated: 'map-insert-failed', stage: 'Map', note: 'writer 微任务多轮失败(限流/崩溃)，图引未插入' }
if (mapInsertV.status === 'BLOCKED') return { escalated: 'map-insert', stage: 'Map', reason: mapInsertV.blocker_reason }

// ---------- Phase G: Archive ----------
// 完整 review 对象注入提示词 → review-report.json 忠实落盘(含 verdict 与全部 issues)，
// 不让 archivist 凭记忆重建出有损版本。
const reviewJson = JSON.stringify(reviewV || { overall_verdict: 'UNKNOWN', issues: [] })
const runLedgerObj = {
  chapter_id: A.chapter_id,
  kind: PRIMER ? 'primer' : (A.skip_impl ? 'meta' : 'code'),
  impl_test_rounds: implTestRounds, impl_test_ledger: ledger,
  write_review_rounds: reviewRounds,
  blind_rounds: blindHistory.length, blind_failures: blindHistory,
  map_rounds: mapHistory.length, map_history: mapHistory,
  escalated: null,
}
// skip_archive(并行发车模式):Review+Map 都过了,但**跳过写共享 Book Bible/trace**——多章
// 并行时各自的 Archive agent 会对 glossary/concepts/interfaces/arc-map/state.json 做并发
// read-modify-write,竞争必丢条目(静默损坏)。故把 review/run-ledger 交还 Lead,由 Lead **串行**
// 补齐归档(按章序,保伏笔/术语累积顺序)。per-chapter 隔离产物(章内 diagrams/narrative 等)已落盘。
if (A.skip_archive) {
  return { chapter: A.chapter_id, needs_archive: true, review_verdict: (reviewV && reviewV.verdict) || 'UNKNOWN', review_report: reviewV, run_ledger: runLedgerObj, note: '并行模式:Review+Map 已过,Bible/trace 待 Lead 串行归档' }
}
phase('Archive')
const runLedger = JSON.stringify(runLedgerObj)
const archiveTask = head('archivist') +
  '任务一(务必先做)：把下面这个完整 review 对象**原样**写入 ' + CH + '/reviews/review-report.json（保留 verdict 与全部 issues，不要删改、不要自己重写摘要）：\n' +
  reviewJson + '\n' +
  '任务一b：把这个 run-ledger 对象**原样**写入 ' + CH + '/reviews/run-ledger.json（经验回流的信号源，不要改写）：\n' + runLedger + '\n' +
  '任务二：回写 Book Bible —— (a) **登记本章新术语进 glossary.json**：本章首现并需全书统一译名的专业缩写/硬件型号/框架 API/自造记号(如 occupancy/SM/coalescing/warp/AOT/dtype 等)，逐条写 {term, 中文译名, 一句释义, 首现章}；不存在则读契约 glossary 结构建。(b) 登记本章新建立的核心概念进 concepts.json：`{"<术语>":"' + A.chapter_id + '"}`(gap 审计据此判「前章已立」)。(c) 登记本章精简版新接口(`python3 ' + REPO + '/scripts/bible.py iface --add ' + A.chapter_id + " '<sig>'`，无精简版可跳)。(d) 确认已回收伏笔、登记新埋伏笔。**glossary/concepts 不回写=术语跨章漂移的源头,勿漏。**\n" +
  '任务三：在 ' + REPO + '/instances/' + INST + '/trace/ 记 delivery 并更新 state.json。返回一句话状态。'
let archV = null
for (let a = 1; a <= 2 && !archV; a++) {
  if (a > 1) log('archive 上轮中断(API崩)，第 ' + a + ' 轮重试：已写的(review-report.json/bible 接口/trace)校验后跳过，只补未完成的')
  archV = await agent(
    archiveTask + (a > 1 ? '\n注意：这是重试。先检查 review-report.json 是否已存在且为合法完整 JSON、bible 是否已登记本章接口，已做的别重复，只补未完成的。' : ''),
    { label: 'archive r' + a, phase: 'Archive', agentType: 'general-purpose', model: MODELS.archive }
  )
}

return { chapter: A.chapter_id, test: testV, review: reviewV }
