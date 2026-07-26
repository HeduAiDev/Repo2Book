export const meta = {
  name: 'retrofit-deep-revise',
  description: '深入浅出回修·第二步：按 research/concepts.json 逐章把非常识概念讲透（writer, 好奇专家声线）→ 本章地图同步判定/更新 → 改图则独立盲审 → 门禁验证。禁整章重写。',
  phases: [
    { title: 'Revise', detail: 'writer 按 research 把非常识概念讲透（背景/例子/竞争对比+链接），定点小修' },
    { title: 'MapSync', detail: 'illustrator 判本章地图是否 under-represent，需改才改+重渲，改了则重置 PENDING' },
    { title: 'MapBlind', detail: '地图改过的章走独立盲审复核' },
  ],
}

// ⚠️ args 注入不可靠 → CFG 为准（换批改 CFG.chapters）。
const CFG = {
  instance: 'vllm',
  // vllm 批次3(周限额 Jul 29 11am ET 重置后发车):剩余 24 章里 gap 最高的 8 章
  chapters: [
    'ch12-engine-core',
    'ch17-worker-and-executor',
    'ch20-distributed-parallelism',
    'ch30-sampling',
    'ch33-primer-eagle',
    'ch34-spec-decode',
    'ch21-async-engine',
    'ch22-model-definitions',
  ],
}
const A = (typeof args !== 'undefined' && args && args.chapters && args.chapters.length) ? args : CFG
const REPO = '/mnt/e/Laboratory/Repo2Book'
const INST = A.instance || CFG.instance
const CHS = A.chapters || []

const REVISE_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['status'],
  properties: {
    status: { type: 'string', enum: ['OK', 'BLOCKED'], description: 'OK=已按 research 补讲透并过门禁' },
    revised: { type: 'integer', description: '实际补讲透的概念处数' },
    new_h2: { type: 'boolean', description: '是否新增了 H2 主节（决定地图是否必须重生成）' },
    note: { type: 'string' },
    blocker_reason: { type: 'string' },
  },
}
const MAP_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['changed'],
  properties: {
    changed: { type: 'boolean', description: '是否更新了本章地图（不需要改则 false）' },
    note: { type: 'string', description: '一句：为何改/为何不改，改了哪个节点' },
    alt_text_suggestion: { type: 'string', description: '若建议同时调 alt-text，写建议文字（归 writer，你不改正文）' },
  },
}
const BLIND_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['pass'],
  properties: { pass: { type: 'boolean' }, note: { type: 'string' } },
}

if (!CHS.length) { log('未配置 chapters，空跑'); return { instance: INST, revised: 0, results: [] } }
log(INST + '：深入浅出回修 ' + CHS.length + ' 章（pipeline：Revise → MapSync → MapBlind）')

const results = await pipeline(
  CHS,
  // ---- Stage 1: writer 按 research 讲透 ----
  async (ch) => {
    const CH = REPO + '/instances/' + INST + '/artifacts/' + ch
    const r = await agent(
      '你的角色契约在 ' + REPO + '/.claude/agents/writer.md —— **先读它**，尤其**必达物 #9（非常识概念讲透 + 充满好奇的专家声线）**。\n' +
      '任务：回修 ' + INST + ' 实例 ' + ch + ' 的正文 ' + CH + '/narrative/chapter.md，把「初学者看描述还是不懂」的非常识名词/标准记法/项目自定义模式/竞争性外部项目**讲透**。\n' +
      '**背景真相源**：' + CH + '/research/concepts.json（researcher 真查、带 sources/links/confidence/writer_note）——**先通读，按每条 writer_note 指示的深度落笔**，别自己凭记忆现编。\n' +
      '声线：**充满好奇的专家**——刨根问底、查透了才讲、**深入浅出**（把深的讲得初学者也懂），不是甩术语。有发展背景的讲清来龙去脉；notation/自定义模式**必给具体例子**；竞争性外部项目给**各自特征+取舍+何时选谁**再放**权威链接**、点到即止别喧宾夺主。\n' +
      '**纪律（硬）**：①只补读者定向背景，**不动 pin 源码解读的正确性**；新增背景与源码解读**自然分层**、别堆成信息倾倒。②说明性示例/外部记法用普通代码围栏 + 明标「说明性/外部」，**不标 `# SOURCE:`**。③**proportional**：讲透≠拉长凑字，补到「初学者能懂」为止。④版本敏感的锚定本章 pin 版本；立场性口径（如竞品博客数字）带出处立场；按各条 confidence 保守写。⑤**若 research 指出正文现有事实错**（如归属写错、默认值写错），**一并改正**并在返回 note 里说明。\n' +
      '**门禁**：改完自跑并确保无 BLOCKING：`lint_chapter_structure`（narrative/chapter.md）、`lint_formulas`（同）、`lint_source_grounding`（章目录）、`lint_fidelity`（章目录，primer 章跳过）、`lint_punct`（narrative/chapter.md）。公式规则：行内一律 `` $`…`$ ``、块级 ```math、公式内禁 CJK、**粗体**定界符外侧留半角空格。\n' +
      '返回：status / revised（补讲透处数）/ **new_h2（是否新增了 H2 主节——决定本章地图是否必须重生成，据实填）** / note。',
      // 用户 2026-07-25 明确指定:Revise 站的 writer 用 fable5(叙事写作交给 fable),
      // 其余站(illustrator 判图/盲审、researcher 查证)保持 opus5。
      { schema: REVISE_SCHEMA, label: ch + ':revise', phase: 'Revise', agentType: 'writer', model: 'fable' }
    )
    return { chapter: ch, revise: r }
  },
  // ---- Stage 2: 本章地图同步判定 ----
  async (prev, ch) => {
    if (!prev || !prev.revise || prev.revise.status !== 'OK') return Object.assign({}, prev, { map: null })
    const CH = REPO + '/instances/' + INST + '/artifacts/' + ch
    const m = await agent(
      '你的角色契约在 ' + REPO + '/.claude/agents/illustrator.md —— **先读它**。\n' +
      INST + ' 实例 ' + ch + ' 的正文刚做过「深入浅出」回修（按 research 补讲透非常识概念）。请核对 **本章地图（' + CH + '/diagrams/chapter-map.*）是否仍忠实代表现在的正文**，**该改才改**。\n' +
      '**判据（本章地图是「源码走线剖面」，不是正文摘要）**：\n' +
      '① 若回修**新增了 H2 主节**（上一步报 new_h2=' + (prev.revise.new_h2 ? 'true' : 'false') + '）→ §徽标↔标题映射会坏，**必须**更新地图使徽标与 H2 标题逐一对应。\n' +
      '② 若某节点**自身讲的对象被换轴/显著加厚**（如某节把「四家能力矩阵」重组成「四条路线取舍」）→ 该节点概括已 under-represent，**更新那个节点的概括文字**（源码符号一般不动）。\n' +
      '③ 若新增内容是**另一个话题的读者背景**（如「什么是 EBNF」「Triton 来历」「页锁定内存」）→ 它讲的不是代码路径，**不进源码走线图，节点不动**。\n' +
      '④ 只加 H3 子节而源码符号/调用链未变 → **不动**。\n' +
      '核完若判定不需改：直接返回 changed=false + 一句理由（**这是常见且正确的结果，别为改而改**）。\n' +
      '若需改：改 `chapter-map.py` → 重渲染 svg+png → **Read PNG 亲眼看**（文字对不对/越界/相撞/箭头悬空/中文豆腐）→ 把 figure-manifest.json 里 chapter-map 的 blind_review 重置 **PENDING**（交独立盲审）。跑 `python3 ' + REPO + '/scripts/lint_chapter_map.py ' + CH + '/ --require` 与 `lint_diagram_geometry.py` 单文件确认。\n' +
      '**alt-text 归 writer（HARD RULE 1）**：你**不许改正文**；若认为开篇 alt-text 也该跟着调，把建议文字写进 alt_text_suggestion。\n' +
      '返回：changed / note / alt_text_suggestion（无则空）。',
      { schema: MAP_SCHEMA, label: ch + ':mapsync', phase: 'MapSync', agentType: 'illustrator' }
    )
    return Object.assign({}, prev, { map: m })
  },
  // ---- Stage 3: 地图改过的才走独立盲审 ----
  async (prev, ch) => {
    if (!prev || !prev.map || !prev.map.changed) return Object.assign({}, prev, { blind: null })
    const CH = REPO + '/instances/' + INST + '/artifacts/' + ch
    const b = await agent(
      '你的角色契约在 ' + REPO + '/.claude/agents/illustrator.md —— **先读它**。\n' +
      '**独立盲审**（你没参与作图）：' + INST + ' 实例 ' + ch + ' 的本章地图刚被更新（' + (prev.map.note || '') + '）。只盲审、不改图。\n' +
      '图：' + CH + '/diagrams/chapter-map.png ；manifest：' + CH + '/diagrams/figure-manifest.json（该图 blind_review 现 PENDING）；对照 ' + CH + '/narrative/chapter.md。\n' +
      '核：①改动是否忠实反映正文（§徽标↔各 H2 节标题逐一对应；被改节点的概括与该节现在的正文一致）；②源码符号未被误改/杜撰（图上符号须在正文或源码真实出现）；③其余节点/泳道/阅读路线未误伤；④几何：越界/相撞/压框/箭头悬空/中文豆腐；⑤零脚手架泄漏。\n' +
      '若 PASS：把 manifest 该图 blind_review 置 **PASS**（注明核的要点），并跑 `python3 ' + REPO + '/scripts/lint_diagrams.py ' + CH + '/` 确认转绿。若 FAIL：保持 PENDING，note 写清哪里错、该怎么改。\n' +
      '返回：pass / note。',
      { schema: BLIND_SCHEMA, label: ch + ':mapblind', phase: 'MapBlind', agentType: 'illustrator' }
    )
    return Object.assign({}, prev, { blind: b })
  }
)

const ok = results.filter((r) => r && r.revise && r.revise.status === 'OK')
const blocked = results.filter((r) => r && r.revise && r.revise.status === 'BLOCKED')
const failed = results.filter((r) => !r || !r.revise)
const mapChanged = results.filter((r) => r && r.map && r.map.changed)
const blindFail = results.filter((r) => r && r.blind && r.blind.pass === false)
const altSuggest = results.filter((r) => r && r.map && r.map.alt_text_suggestion)
log('回修完：OK ' + ok.length + ' / BLOCKED ' + blocked.length + ' / 失败 ' + failed.length +
    '；地图改了 ' + mapChanged.length + ' 章，盲审未过 ' + blindFail.length + ' 章')
return {
  instance: INST,
  total: CHS.length,
  ok: ok.map((r) => ({ chapter: r.chapter, revised: r.revise.revised, new_h2: r.revise.new_h2, note: r.revise.note })),
  blocked: blocked.map((r) => ({ chapter: r.chapter, reason: r.revise.blocker_reason })),
  failed: failed.map((r) => (r && r.chapter) || '?'),
  map_changed: mapChanged.map((r) => ({ chapter: r.chapter, note: r.map.note })),
  blind_failed: blindFail.map((r) => ({ chapter: r.chapter, note: r.blind.note })),
  alt_text_todo: altSuggest.map((r) => ({ chapter: r.chapter, suggestion: r.map.alt_text_suggestion })),
}
