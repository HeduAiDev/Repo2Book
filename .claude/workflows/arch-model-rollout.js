export const meta = {
  name: 'arch-model-rollout',
  description: '渐进式架构模型图铺开：每章生成架构剖面图并写图注（writer），同步退役旧 roadmap 窄条。试点 ch31 已定稿，本 workflow 推到其余章节。',
  phases: [
    { title: 'Narrate', detail: 'writer 为每章写架构模型图的图注、删 roadmap 段、视接缝回指' },
    { title: 'Blind', detail: '独立盲审只看 PNG+spec 复核图与图注论点' },
  ],
}

// ⚠️ args 注入不可靠 → CFG 为准（换批改 CFG.chapters）。试点 ch31 已由人工走通,不在此列。
const CFG = {
  instance: 'vllm',
  // 首批 3 个不同类型章:普通码章/KV/模型定义,验证泛化再放量
  chapters: ['ch03-config-and-wiring', 'ch15-kv-cache', 'ch22-model-definitions'],
}
const A = (typeof args !== 'undefined' && args && args.chapters && args.chapters.length) ? args : CFG
const REPO = '/mnt/e/Laboratory/Repo2Book'
const INST = A.instance || CFG.instance
const CHS = A.chapters || []

const NARR_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['status'],
  properties: {
    status: { type: 'string', enum: ['OK', 'BLOCKED'] },
    note: { type: 'string' },
    station_refs: { type: 'integer', description: '正文里引用的站号数量（用于核 lint_arch_model_stations）' },
    blocker_reason: { type: 'string' },
  },
}
const BLIND_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['pass'],
  properties: { pass: { type: 'boolean' }, note: { type: 'string' } },
}

if (!CHS.length) { log('未配置 chapters，空跑'); return { instance: INST, done: 0, results: [] } }
log(INST + '：架构模型图铺开 ' + CHS.length + ' 章')

const results = await pipeline(
  CHS,
  // ---- Stage 1: writer 写图注 + 删 roadmap 段 + 接缝回指 ----
  async (ch) => {
    const CH = REPO + '/instances/' + INST + '/artifacts/' + ch
    const r = await agent(
      '你的角色契约在 ' + REPO + '/.claude/agents/writer.md —— **先读它**。\n' +
      '任务：把「渐进式架构模型图」接进 ' + INST + ' 实例 ' + ch + ' 的正文（ch31 已定稿，照它的范式做）。\n' +
      '章目录 ' + CH + '；图 ' + CH + '/diagrams/arch-model.png 已生成（**先用 Read 亲眼看**）。\n\n' +
      '**这张图是什么**：第 1 章那张「一个请求的端到端旅程」长大后的样子，全书共用——入口 → Stage 1 输入 → IPC 边界 → EngineCore 大框（内分四组：调度与显存/执行与并行/模型与算子/解码策略）→ Stage 3 输出；蓝=前面章节已读(带章号)、橙=本章展开、虚线=后续才讲。本章的组件就地展开成源码里的真实组织关系（契约+实现+持有），站点标在组件上。\n\n' +
      '**照 ch31 的范式（可 Read ' + REPO + '/instances/vllm/artifacts/ch31-structured-output/narrative/chapter.md 的开头学）**：\n' +
      '1. 在「你在这里」段放上架构模型图 `../diagrams/arch-model.png`（若该段还在用旧 roadmap 窄条，**用它顶替**，并在图注里写清「这张图就是第 1 章那张端到端旅程长大后的样子」）；\n' +
      '2. 图注三要点：①全书共用、逐章生长，蓝/橙/虚线的含义；②本章这块新结构**接在哪些前面章节已读的结构上**（这是读者建立全局源码地图的关键，务必点出）；③「站号是请求流经代码的顺序，正文按讲解需要编排，不必照站号顺序读」，末尾预告「跨模块的几个大接缝处，正文会随手报一句『现在走到哪一段』」；\n' +
      '3. **删除旧 roadmap 段**（若存在），并删除 ' + CH + '/diagrams/roadmap.png 与 roadmap.svg（不删会成孤儿图，lint_diagrams 判红）；\n' +
      '4. **跨模块接缝处做少量回指**（克制，全章 0–5 处，全程单目录的章可以 0 处）：一句话点明「现在走到走线的哪一段、属于哪个模块/组件」。**绝不**写「按站号顺序往下读」。\n\n' +
      '**硬约束**：pin 源码解读的正确性一个字不许动；不重写整章；公式规则照常；全文不得出现内部工件名（arch-model.json/dossier/cartography/code_spine/instances/…）；图注给结论不描述画面。\n' +
      '**门禁（改完自跑，cd 到 ' + REPO + '）**：lint_chapter_structure、lint_formulas、lint_source_grounding、lint_fidelity、lint_punct、lint_chapter_map --require、lint_diagrams、**lint_arch_model_stations**（正文引用的站号不得超本章走线总数）、lint_anchors --all —— 均不得有 BLOCKING。\n' +
      '返回：status / note / station_refs（正文里引用的站号数量）。',
      { schema: NARR_SCHEMA, label: ch + ':narrate', phase: 'Narrate', agentType: 'writer', model: 'fable' }
    )
    return { chapter: ch, narrate: r }
  },
  // ---- Stage 2: 独立盲审 ----
  async (prev, ch) => {
    if (!prev || !prev.narrate || prev.narrate.status !== 'OK') return Object.assign({}, prev, { blind: null })
    const CH = REPO + '/instances/' + INST + '/artifacts/' + ch
    const b = await agent(
      '你的角色契约在 ' + REPO + '/.claude/agents/illustrator.md —— **先读它**。\n' +
      '**独立盲审**（你没参与作图/写作）：' + INST + ' 实例 ' + ch + ' 的架构模型图刚接进正文。只盲审、不改。\n' +
      '图：' + CH + '/diagrams/arch-model.png ；正文：' + CH + '/narrative/chapter.md。\n' +
      '核：①图注论点是否与图一致（层/组件/蓝橙虚线/本章这块接在哪些已读结构上）；②站号数字是否与正文一致、且都在本章走线总数内；③「它是第 1 章那张端到端旅程长大后的样子」是否讲清；④几何：越界/相撞/压框/中文豆腐；⑤零脚手架泄漏（arch-model.json/dossier/cartography/instances/ 不得出现在正文）。\n' +
      '若 PASS：把 ' + CH + '/diagrams/figure-manifest.json 里 arch-model 的 blind_review 置 PASS。若 FAIL：保持 PENDING 并在 note 写清怎么改。返回 pass/note。',
      { schema: BLIND_SCHEMA, label: ch + ':blind', phase: 'Blind', agentType: 'illustrator', model: 'sonnet' }
    )
    return Object.assign({}, prev, { blind: b })
  }
)

const ok = results.filter((r) => r && r.narrate && r.narrate.status === 'OK')
const blocked = results.filter((r) => r && r.narrate && r.narrate.status === 'BLOCKED')
const failed = results.filter((r) => !r || !r.narrate)
const blindFail = results.filter((r) => r && r.blind && r.blind.pass === false)
log('铺开完：OK ' + ok.length + ' / BLOCKED ' + blocked.length + ' / 失败 ' + failed.length + '；盲审未过 ' + blindFail.length)
return {
  instance: INST, total: CHS.length,
  ok: ok.map((r) => ({ chapter: r.chapter, station_refs: r.narrate.station_refs, note: r.narrate.note })),
  blocked: blocked.map((r) => ({ chapter: r.chapter, reason: r.narrate.blocker_reason })),
  failed: failed.map((r) => (r && r.chapter) || '?'),
  blind_failed: blindFail.map((r) => ({ chapter: r.chapter, note: r.blind.note })),
}
