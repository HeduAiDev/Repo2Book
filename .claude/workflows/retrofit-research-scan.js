export const meta = {
  name: 'retrofit-research-scan',
  description: '深入浅出回修·第一步：对一本书的每一章跑 researcher 查非常识概念 gap，产 research/concepts.json + 逐章 gap 计数。无 gap 的章几乎零成本；有 gap 的进后续 revise。',
  phases: [
    { title: 'Scan', detail: 'researcher 逐章查非常识名词/竞争项目/自定义模式/标准记法，产 concepts.json + gap 数' },
  ],
}

// ⚠️ 本环境 Workflow 的 args 注入不可靠(实测 args.chapters 到不了脚本)→ 用脚本内 CFG 作可靠配置,
// args 有值时优先。换书时改 CFG（chapters 列该书待扫章目录名）。
const CFG = {
  instance: 'vllm-ascend',
  source_root: 'instances/vllm-ascend/source',
  chapters: ['ch10-eplb-expert-load-balancing','ch11-pd-disaggregation-mooncake','ch12-kv-pooling-ascend-store','ch13-kv-offloading-host-cpu','ch14-npuworker-execution-control','ch15-npumodelrunner-cuda-monkeypatch','ch16-single-step-forward-context-dp-sync','ch17-kv-cache-allocation-reshape-bind','ch18-310p-inference-chip-specialization','ch19-attention-backend-selection','ch20-ascend-attention-mha','ch21-primer-mla','ch22-mla-on-npu','ch23-primer-sparse-attention','ch24-sparse-attention-sfa-dsa','ch25-kv-manager-and-schedulers','ch26-primer-v4-csa-hca','ch27-customop-oot-replacement','ch28-torch-library-and-meta','ch29-ascend-compiler-aclgraph','ch30-fusedmoe-batch-invariant','ch31-primer-quantization','ch32-ascend-quantization-framework','ch33-sampling-npu-adaptation','ch34-primer-speculative-sampling','ch35-primer-dflash','ch36-speculative-decode-npu','ch37-primer-dspark','ch38-model-lora-netloader-registration'],
}
const A = (typeof args !== 'undefined' && args && args.chapters && args.chapters.length) ? args : CFG
const REPO = '/mnt/e/Laboratory/Repo2Book'
const INST = A.instance || CFG.instance
const CHS = A.chapters || []
const SRC = A.source_root || ('instances/' + INST + '/source')

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['gaps', 'blocked'],
  properties: {
    gaps: { type: 'integer', description: '本章真正需要外部深度研究的非常识概念数（无则 0，别硬凑）' },
    blocked: { type: 'boolean', description: '是否无法完成（如无法联网）' },
    note: { type: 'string', description: '一句：研究了哪几项/为何 0/有无需 Lead 决断' },
  },
}

if (!CHS.length) { log('未传 chapters，空跑'); return { instance: INST, scanned: 0, results: [] } }
log(INST + '：对 ' + CHS.length + ' 章跑深度研究扫描')
phase('Scan')

const results = await parallel(CHS.map((ch) => async () => {
  const CH = REPO + '/instances/' + INST + '/artifacts/' + ch
  const r = await agent(
    '你的角色契约在 ' + REPO + '/.claude/agents/researcher.md —— **先读它**，严格遵守。\n' +
    '目标源码根目录 ' + REPO + '/' + SRC + '（引用写规范路径，绝不带 instances/.../source/ 前缀）。\n' +
    '任务：为 ' + INST + ' 实例的 ' + ch + ' 做**概念深度研究**，产 ' + CH + '/research/concepts.json（结构见契约）。\n' +
    '读该章 ' + CH + '/dossier/dossier.json 与 ' + CH + '/narrative/chapter.md，找出正文里**初学者看描述还是不懂、需例子或背景**的非常识名词 / 标准记法 / 项目自定义模式 / 竞争性外部项目，逐一 WebSearch/WebFetch 查透。\n' +
    '**真去查、每条给出处 + 版本/日期**；**只做读者定向外部背景、不解读 pin 源码**；notation/custom_pattern **必给具体可核例子**；竞争项目给差异 + 何时选 + 权威链接（点到即止）；深入浅出、刨根问底；版本敏感的锚定本章 pin 版本。\n' +
    '**本章聚焦真会用到的点——无外部 gap 就如实产一份 `{"concepts": []}`、gaps 返 0，别硬凑**（很多纯内部机制章确实没什么可查）。\n' +
    '返回：gaps（真需研究的概念数）/blocked/note。无法联网则 blocked=true。',
    // 用户 2026-07-24 定:不是非常难的问题优先 opus5,不用 fable5。research 是全书事实真相源、
    // 质量向下游所有写作复利,故用 opus(实测 sonnet 研究员出过归属/数字失实,需二次核实)。
    { schema: SCHEMA, label: ch + ':research', phase: 'Scan', agentType: 'researcher', model: 'opus' }
  )
  return { chapter: ch, gaps: r ? r.gaps : null, blocked: r ? r.blocked : true, note: r ? r.note : 'agent 失败' }
}))

const withGaps = results.filter((x) => x && x.gaps > 0).sort((a, b) => b.gaps - a.gaps)
const noGap = results.filter((x) => x && x.gaps === 0)
const failed = results.filter((x) => !x || x.gaps === null || x.blocked)
log('扫描完：有 gap ' + withGaps.length + ' 章 / 无 gap ' + noGap.length + ' 章 / 失败 ' + failed.length + ' 章')
return {
  instance: INST,
  scanned: CHS.length,
  with_gaps: withGaps,       // 进 revise 的工作单(按 gap 数降序)
  no_gap: noGap.map((x) => x.chapter),
  failed: failed.map((x) => (x && x.chapter) || '?'),
}
