export const meta = {
  name: 'arch-model-rollout',
  description: '渐进式架构模型图铺开：逐章生成「你在这里」架构模型图（取代 roadmap 窄长条）→ illustrator 渲染自查 → writer 换图引+图注 → 独立盲审 → 门禁。ch31 已试点定样。',
  phases: [
    { title: 'Render', detail: 'illustrator 生成 arch-model 图 + Read PNG 亲眼看自查 + 登记 manifest' },
    { title: 'Prose', detail: 'writer 把开篇 roadmap 换成架构模型图，写图注 + 若干接缝回指（定图权在 writer）' },
    { title: 'Blind', detail: '独立盲审（未参与作图者只看 PNG + spec 复核）+ 门禁收口' },
  ],
}

// ⚠️ args 注入不可靠 → CFG 为准（换批改 CFG.chapters）。
const CFG = {
  instance: 'vllm',
  // 批次2(vllm 剩 32 章的前 16 章)
  chapters: [
    'ch01-config-and-wiring',
    'ch02-entrypoints',
    'ch04-async-llm',
    'ch05-input-processing',
    'ch06-input-processor',
    'ch08-output-processor',
    'ch09-detokenization',
    'ch10-logprobs',
    'ch11-engine-core',
    'ch12-engine-core',
    'ch14-scheduler',
    'ch16-kv-cache',
    'ch17-worker-and-executor',
    'ch18-model-runner',
    'ch19-model-runner',
    'ch20-distributed-parallelism',
  ],
}
const A = (typeof args !== 'undefined' && args && args.chapters && args.chapters.length) ? args : CFG
const REPO = '/mnt/e/Laboratory/Repo2Book'
const INST = A.instance || CFG.instance
const CHS = A.chapters || []

const S_RENDER = {
  type: 'object', additionalProperties: false, required: ['status'],
  properties: {
    status: { type: 'string', enum: ['OK', 'BLOCKED'] },
    shape: { type: 'string', description: '本章展开呈现为哪种结构：contract/containment/flat' },
    selfcheck_pass: { type: 'boolean', description: '六项视觉自查是否全 true（亲眼 Read PNG 后据实填）' },
    note: { type: 'string' }, blocker_reason: { type: 'string' },
  },
}
const S_PROSE = {
  type: 'object', additionalProperties: false, required: ['status'],
  properties: {
    status: { type: 'string', enum: ['OK', 'BLOCKED'] },
    callbacks: { type: 'integer', description: '正文里做了几处接缝回指' },
    roadmap_removed: { type: 'boolean', description: '是否已删除本章 roadmap.png/svg 并改掉所有引用' },
    note: { type: 'string' }, blocker_reason: { type: 'string' },
  },
}
const S_BLIND = {
  type: 'object', additionalProperties: false, required: ['pass'],
  properties: { pass: { type: 'boolean' }, note: { type: 'string' } },
}

if (!CHS.length) { log('未配置 chapters，空跑'); return { instance: INST, done: 0 } }
log(INST + '：架构模型图铺开 ' + CHS.length + ' 章（Render → Prose → Blind）')

const results = await pipeline(
  CHS,
  // ---- Stage 1: 生成 + 视觉自查 ----
  async (ch) => {
    const CH = REPO + '/instances/' + INST + '/artifacts/' + ch
    const cid = (ch.match(/^(ch\d+)/) || [])[1] || ch
    const r = await agent(
      '你的角色契约在 ' + REPO + '/.claude/agents/illustrator.md —— **先读它**，尤其「开篇「你在这里」= 架构模型图」一节（roadmap 已退役）。\n' +
      '任务：为 ' + INST + ' 实例 ' + ch + ' 生成开篇**架构模型图**。\n' +
      '```\npython3 ' + REPO + '/scripts/arch_model.py build --instance ' + INST + '\n' +
      'python3 ' + REPO + '/scripts/arch_model_figure.py --chapter ' + cid + ' --instance ' + INST +
      ' --out ' + CH + '/diagrams/arch-model.svg\nrsvg-convert -z 2 ' + CH + '/diagrams/arch-model.svg -o ' + CH + '/diagrams/arch-model.png\n```\n' +
      '再写一个薄封装 ' + CH + '/diagrams/gen_arch-model.py（照 ch31 的同名文件抄，只改 CHAPTER/INSTANCE 两个常量），使本章图可一键重生成。\n' +
      '**然后必须 Read 那张 PNG 亲眼看**，逐项如实判定六项自查（claim_readable_10s / numbers_match_spec / '
      + 'no_overlap / arrows_attached / cjk_rendered / reading_order_clear），结果写进 ' + CH + '/diagrams/figure-manifest.json '
      + '的 arch-model 条目（figure_id=arch-model, gen=gen_arch-model.py, spec 里写清 claim 与数字出处，blind_review 置 PENDING）。\n' +
      '**纪律**：图上的层/组件/类名/站号/关系全部由 arch_model.py 从源码 AST 与档案抽取——**你不许手改图上的任何名字或站号**。'
      + '几何问题（文字越界/相撞/截断/中文豆腐/箭头悬空）是**渲染器**的问题：改 ' + REPO + '/scripts/arch_model_figure.py 并在 note 里说明改了什么；'
      + '若发现的是**数据**问题（关系抽错/站号归错），返回 BLOCKED 说清，别在图上打补丁。\n' +
      '参考已定样的 ch31：' + REPO + '/instances/vllm/artifacts/ch31-structured-output/diagrams/（三种形态：契约容器/盒套盒组合/诚实平铺）。\n' +
      '返回：status / shape（contract|containment|flat）/ selfcheck_pass / note。',
      { schema: S_RENDER, label: ch + ':render', phase: 'Render', agentType: 'illustrator' }
    )
    return { chapter: ch, render: r }
  },
  // ---- Stage 2: writer 换图引 + 图注 + 接缝回指 ----
  async (prev, ch) => {
    if (!prev || !prev.render || prev.render.status !== 'OK') return Object.assign({}, prev, { prose: null })
    const CH = REPO + '/instances/' + INST + '/artifacts/' + ch
    const p = await agent(
      '你的角色契约在 ' + REPO + '/.claude/agents/writer.md —— **先读它**。\n' +
      '任务：把 ' + INST + ' 实例 ' + ch + ' 开篇的 roadmap 窄长条**换成**新的架构模型图，并写图注。\n' +
      '图：' + CH + '/diagrams/arch-model.png —— **先 Read 亲眼看**（本章展开形态：' + (prev.render.shape || '?') + '）。\n' +
      '**样板（已定稿，照它的分工与语气写，不要照抄文字）**：' + REPO + '/instances/vllm/artifacts/ch31-structured-output/narrative/chapter.md 开头 30 行。\n' +
      '\n**这张图是什么**：第 1 章那张「一个请求的端到端旅程」长大后的样子，**整本书共用同一副骨架**'
      + '（入口→Stage 1→IPC 边界→EngineCore 大框→Stage 3）；蓝框＝前面章节已读（带章号），虚线＝后续才讲，橙＝本章新增；'
      + '本章那块**就地展开**成源码里的真实组织关系，本章站点标在这些组件上。\n' +
      '\n**要做的事**：\n' +
      '① 开篇 `## 你在这里` 段：把 roadmap.png 的图引换成 `../diagrams/arch-model.png`，重写 alt 与图注。'
      + '图注**必须**含三点：(a) 这张图整本书共用、从开篇起逐章生长，蓝/橙/虚线三色含义；'
      + '(b) 点出**认得感**——它就是第 1 章那张端到端旅程图长大后的样子；'
      + '(c) 「站号是请求流经代码的顺序；正文按讲解需要编排，不必照站号顺序读」，以及末句预告「跨模块的几个大接缝处，正文会随手报一句『现在走到哪一段』」。\n' +
      '**最值钱的一句**（若图上确有）：本章这块新结构**接在哪几块读者已经读过的结构上** —— 这是全局源码地图能立起来的关键，请点出来。\n' +
      '② **删除**本章 `diagrams/roadmap.png` 与 `roadmap.svg`，并确保正文再无 roadmap.png 引用'
      + '（否则 lint_diagrams 判孤儿图）。`lint_chapter_structure` 只要求开头 60 行内有「你在这里」字样，保留 H2 即可。\n' +
      '③ 正文里做 **少量**（约 2–5 处，接缝天然少的章可以更少甚至 0）跨模块**接缝回指**：在正文从一个组件讲到另一个组件时，'
      + '一句话点明「现在走到哪一段、属于哪块」。**克制**——别把正文变成图的解说词。'
      + '站号断言必须与图上一致（图是唯一真相源，别自己编站号）。\n' +
      '\n**纪律**：不重写整章；pin 源码解读的正确性一个字不许动；'
      + '全文不得出现内部工件名（arch-model.json / dossier / cartography / key_classes / code_spine / instances/…），对读者只说「架构模型图」。\n' +
      '**门禁**（自跑，均不得有 BLOCKING）：lint_chapter_structure / lint_formulas / lint_source_grounding / '
      + 'lint_fidelity（primer 章跳过）/ lint_punct / lint_chapter_map --require / lint_diagrams。\n' +
      '返回：status / callbacks（回指处数）/ roadmap_removed / note。',
      { schema: S_PROSE, label: ch + ':prose', phase: 'Prose', agentType: 'writer' }  // fable 曾不可用,暂继承主模型(opus);恢复后加 model:'fable'
    )
    return Object.assign({}, prev, { prose: p })
  },
  // ---- Stage 3: 独立盲审 + 门禁收口 ----
  async (prev, ch) => {
    if (!prev || !prev.prose || prev.prose.status !== 'OK') return Object.assign({}, prev, { blind: null })
    const CH = REPO + '/instances/' + INST + '/artifacts/' + ch
    const b = await agent(
      '你的角色契约在 ' + REPO + '/.claude/agents/illustrator.md —— **先读它**。\n' +
      '**独立盲审**（你没参与作图）：' + INST + ' 实例 ' + ch + ' 新增了开篇架构模型图。只盲审、不改图。\n' +
      '图：' + CH + '/diagrams/arch-model.png ｜ manifest：' + CH + '/diagrams/figure-manifest.json（arch-model 现 PENDING）｜ 对照正文 ' + CH + '/narrative/chapter.md。\n' +
      '核：①**不看正文**先复述这张图的论点，再对照图注看是否一致；②图上组件名/类名/站号是否在正文或源码真实出现（**杜撰即 FAIL**）；'
      + '③蓝框的「第 N 章已读」章号是否合理（该组件确实是那章讲的）；④本章展开块是否与本章主题相符；'
      + '⑤几何：文字越界/相撞/截断/箭头悬空/中文豆腐；⑥零脚手架泄漏（图上不得有 instances/ 内部路径或内部产物名）。\n' +
      '**另核两项收口**：本章 diagrams/ 下**不应再有 roadmap.png/svg**；`python3 ' + REPO + '/scripts/lint_diagrams.py ' + CH + '/` 必须通过（无孤儿图）。\n' +
      '若 PASS：把 manifest 里 arch-model 的 blind_review 置 **PASS**（notes 写清核了哪几点）。若 FAIL：保持 PENDING，note 写清哪里错、该怎么改。\n' +
      '返回：pass / note。',
      { schema: S_BLIND, label: ch + ':blind', phase: 'Blind', agentType: 'illustrator' }
    )
    return Object.assign({}, prev, { blind: b })
  }
)

const ok = results.filter((r) => r && r.blind && r.blind.pass)
const blocked = results.filter((r) => r && ((r.render && r.render.status === 'BLOCKED') || (r.prose && r.prose.status === 'BLOCKED')))
const blindFail = results.filter((r) => r && r.blind && r.blind.pass === false)
const failed = results.filter((r) => !r || !r.render)
log('铺开完：全绿 ' + ok.length + ' / BLOCKED ' + blocked.length + ' / 盲审未过 ' + blindFail.length + ' / 失败 ' + failed.length)
return {
  instance: INST,
  total: CHS.length,
  ok: ok.map((r) => ({ chapter: r.chapter, shape: r.render.shape, callbacks: r.prose.callbacks })),
  blocked: blocked.map((r) => ({ chapter: r.chapter, reason: (r.render && r.render.blocker_reason) || (r.prose && r.prose.blocker_reason) })),
  blind_failed: blindFail.map((r) => ({ chapter: r.chapter, note: r.blind.note })),
  failed: failed.map((r) => (r && r.chapter) || '?'),
  roadmap_left: results.filter((r) => r && r.prose && r.prose.roadmap_removed === false).map((r) => r.chapter),
}
