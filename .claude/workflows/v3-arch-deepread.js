export const meta = {
  name: 'v3-arch-deepread',
  description: 'v3 Phase 0 架构深读：analyst 团队以「理解系统」为目标重读 vllm 源码——每个子系统的 why 链（旧设计→痛点→方案→代价）+ 上下游契约 + 深读卡片。不是素材收集（dossier 已有），是设计者视角的完全理解。产出喂给 Lead 综合 ARCHITECTURE.md + L0 图。',
  phases: [
    { title: 'DeepRead', detail: '6 个 analyst 并行深读各自子系统（why 链 + 契约 + 首读章建议）' },
    { title: 'CrossCheck', detail: '两两交叉核对面（进程边界/异步边界/数据所有权）——相邻子系统的契约必须对得上' },
  ],
}

const REPO = 'E:/Laboratory/Repo2Book'
const SRC = REPO + '/instances/vllm/source'
const ART = REPO + '/instances/vllm/artifacts'

// 六个子系统族（按数据流分段，不是按目录）——深读分工
const DOMAINS = [
  {
    id: 'api-entry',
    name: 'API 进程与两个使用面',
    focus: 'LLM/AsyncLLM 双使用面、OpenAI server、add_request 进 AsyncLLM 之后发生了什么、RequestOutputCollector 为什么存在（v0 直接返回 vs v1 SSE 流）。为什么 API 进程绝不能碰 GPU。',
    files: 'vllm/entrypoints/、vllm/v1/engine/async_llm.py、vllm/v1/engine/output_processor.py',
    chapters: 'ch01 ch02 ch37 ch38',
  },
  {
    id: 'ipc-stage',
    name: '三段式与 IPC 边界',
    focus: 'EngineCoreRequest 的诞生（token 化在哪发生）、ZMQ PUSH/DEALER 为什么是这两个 socket 类型、msgpack + 零拷贝 tensor、EngineCoreOutput 回程。v0 单进程同步的痛点到 v1 三段式的完整 why 链。',
    files: 'vllm/v1/engine/core_client.py、vllm/v1/engine/core.py（IPC 部分）、vllm/v1/request.py',
    chapters: 'ch04 ch05 ch06 ch07',
  },
  {
    id: 'engine-loop',
    name: '引擎心跳：调度循环',
    focus: 'EngineCore 逐拍循环 schedule→execute→sample→update 为什么是这个顺序、调度器只认 token 数不认请求数的连续批处理、抢占（recompute vs swap 的取舍）、请求生命周期状态机。为什么循环体里不能有任何慢操作。',
    files: 'vllm/v1/engine/core.py（循环）、vllm/v1/core/sched/scheduler.py',
    chapters: 'ch11 ch12 ch13 ch14',
  },
  {
    id: 'memory-kv',
    name: '显存主角：分页 KV 与前缀缓存',
    focus: 'KV cache 的显存账本（为什么利用率曾只有 20-38%）、PagedAttention 的虚拟内存类比、BlockPool/逻辑块→物理块、前缀缓存 radix 树、分配失败如何触发抢占。KVConnector 的 P/D 分离契约。',
    files: 'vllm/v1/core/kv_cache_manager.py、vllm/v1/core/kv_cache_utils.py、vllm/v1/core/block*',
    chapters: 'ch15 ch16 ch35 ch36',
  },
  {
    id: 'gpu-exec',
    name: 'GPU 执行管线',
    focus: 'Worker/Executor/ModelRunner 的层级（为什么三层）、execute_model 的两阶段、CustomOp 与 torch.compile 分片、注意力后端选择、CUDA Graph 捕获回放（为什么形状必须精确匹配）、Triton kernel 与 block_table。GPU 为什么不能等 Python。',
    files: 'vllm/v1/worker/、vllm/v1/executor/、vllm/v1/worker/gpu_model_runner.py',
    chapters: 'ch17 ch18 ch19 ch20 ch22 ch23',
  },
  {
    id: 'model-sample',
    name: '模型层与采样出口',
    focus: '模型定义层（ForCausalLM 统一入口/DecoderLayer/注意力变体 MLA/GQA）、DeepSeek-V4 这类新架构进来要实现什么、采样管线 9 步 logits 处理器、结构化输出 bitmask 如何约束采样、spec decode 的 draft/verify。为什么采样不在 GPU kernel 里一步做完。',
    files: 'vllm/model_executor/、vllm/v1/sample/、vllm/v1/structured_output/',
    chapters: 'ch24 ch25 ch26 ch27 ch28 ch29 ch30 ch31 ch32 ch33 ch34',
  },
]

const S_CARD = {
  type: 'object', additionalProperties: false, required: ['status'],
  properties: {
    status: { type: 'string', enum: ['OK', 'BLOCKED'] },
    card: {
      type: 'object', additionalProperties: false,
      properties: {
        domain: { type: 'string' },
        why_chains: {
          type: 'array', description: 'why 链数组，每条 = {decision, old_design, pain, solution, cost}',
          items: {
            type: 'object', additionalProperties: false,
            properties: {
              decision: { type: 'string', description: '设计决策名（如「三段式进程解耦」）' },
              old_design: { type: 'string', description: '之前怎么做（v0 或朴素做法）+ 源码证据' },
              pain: { type: 'string', description: '痛点：哪个指标被卡（吞吐/延迟/显存/并发），最好有数字' },
              solution: { type: 'string', description: 'v1 的方案 + 关键源码位置（file:line）' },
              cost: { type: 'string', description: '这个方案的代价（诚实记录，如延迟+1次IPC）' },
            },
          },
        },
        contracts: { type: 'array', items: { type: 'string' }, description: '与相邻子系统的契约：输入从谁来/输出给谁/数据所有权（谁写谁读）' },
        data_flow: { type: 'string', description: '本域内一个请求的数据流（file:line 级）' },
        first_read_suggestion: { type: 'string', description: 'pedagogy 建议：本域概念最早应在第几章引入、依赖哪些前置' },
        key_files: { type: 'array', items: { type: 'string' } },
      },
    },
    note: { type: 'string' }, blocker_reason: { type: 'string' },
  },
}

const S_CROSS = {
  type: 'object', additionalProperties: false, required: ['status'],
  properties: {
    status: { type: 'string', enum: ['OK', 'MISMATCH'] },
    mismatches: { type: 'array', items: { type: 'string' }, description: '契约对不上的点：A 说 X 给 B，B 说收的是 Y' },
    note: { type: 'string' },
  },
}

log('v3 Phase 0 架构深读：6 域并行 → 交叉核对面')

const cards = await parallel(DOMAINS.map((d) => () =>
  agent(
    '你的角色契约在 ' + REPO + '/.claude/agents/analyst.md —— **先读它**。\n' +
    '但这不是常规 dossier 任务——**这是 v3 重写的 Phase 0 架构深读**（背景读 ' + REPO + '/docs/superpowers/specs/2026-08-15-v3-pedagogy-rewrite.md 的 Phase 0 节）。\n\n' +
    '**目标不是收集素材，是理解系统**：从设计者视角，把「' + d.name + '」这块读透。\n' +
    '域：' + d.name + '\n焦点：' + d.focus + '\n主要文件（可超出）：' + d.files + '\nv2 对应章（参考其 dossier 但别被它的组织方式绑架）：' + d.chapters + '\n\n' +
    '**产出深读卡片**（结构见 schema）：\n' +
    '1. why_chains：本域每个重要设计决策一条链——旧设计是什么（v0/朴素做法，找源码或论文证据）→ 痛点（哪个指标被卡，有数字更好）→ v1 方案（file:line）→ **代价**（诚实！每个设计都有代价，如三段式的代价是每请求 2 次 IPC）。\n' +
    '2. contracts：与相邻域的契约——输入从谁来、输出给谁、**数据所有权**（这块内存谁写谁读、跨界时怎么传）。措辞要精确到类名/方法名。\n' +
    '3. data_flow：file:line 级的一个请求数据流。\n' +
    '4. first_read_suggestion：站在「不懂 AI 的后端工程师」读者视角，本域概念最早何时引入合适、依赖什么前置。\n\n' +
    '**纪律**：每条断言带源码锚点（file:line）；读不懂的就标 BLOCKED 别编；**特别警惕 v2 叙事的惯性**——你是在重新理解，不是在复述 v2 的讲法。\n' +
    '参考素材（可用可挑战）：旧 pin 深读卡（' + ART + '/../book/cartography/deepread-v021/）与 v2 各章 dossier（' + ART + '/ch*/dossier/dossier.json，行号属 v0.21.0 仅作历史线索）里的机制与行号——dossier 的源码事实可信，但它**没有 why 链**，why 链要你自己从源码+设计文档+commit history 里挖。\n' +
    '深读产物写到 ' + ART + '/../book/cartography/deepread/' + d.id + '.json（目录不存在则建）。',
    { schema: S_CARD, label: 'deepread:' + d.id, phase: 'DeepRead', agentType: 'analyst' }
  )
))

const ok = cards.filter((c) => c && c.status === 'OK' && c.card)
log('深读完成：' + ok.length + '/' + DOMAINS.length + ' 域')

// 交叉核对面：相邻域的契约必须对得上（A 说的输出 = B 说的输入）
const PAIRS = [
  ['api-entry', 'ipc-stage', 'API 进程交出去的请求数据形态 vs IPC 层收到时的形态'],
  ['ipc-stage', 'engine-loop', 'IPC 投递进 input_queue 的消息 vs 引擎循环取出的消息'],
  ['engine-loop', 'memory-kv', '调度器要的显存账本 vs KV 管理器给的分配结果'],
  ['engine-loop', 'gpu-exec', '调度产出的 SchedulerOutput vs ModelRunner 消费的输入'],
  ['memory-kv', 'gpu-exec', 'block_table/slot_mapping 的生产者 vs 消费者'],
  ['gpu-exec', 'model-sample', '前向产出的 logits vs 采样管线消费的 logits'],
]
const crosses = await parallel(PAIRS.map(([a, b, seam]) => () =>
  agent(
    '你的角色契约在 ' + REPO + '/.claude/agents/analyst.md —— **先读它**。\n\n' +
    '**v3 Phase 0 交叉核对面**：两个深读卡片对同一接缝的描述必须一致。\n' +
    '读 ' + ART + '/../book/cartography/deepread/' + a + '.json 与 ' + b + '.json 的 contracts 字段。\n' +
    '接缝：' + seam + '\n' +
    '核：①A 声称的输出（类名/方法名/数据形态）与 B 声称的输入是否同一个东西；②数据所有权描述是否冲突；③若有一方没提这个接缝，标出缺口。\n' +
    '不一致处需回源码裁决（' + SRC + '），MISMATCH 时把正确答案写进 mismatches（带 file:line）。',
    { schema: S_CROSS, label: 'cross:' + a + '-' + b, phase: 'CrossCheck', agentType: 'analyst' }
  )
))

const mismatches = crosses.filter((c) => c && c.status === 'MISMATCH').flatMap((c) => c.mismatches || [])
log('交叉核对完：' + (PAIRS.length - mismatches.length) + '/' + PAIRS.length + ' 面一致，' + mismatches.length + ' 处冲突待 Lead 裁决')
return {
  domains: ok.map((c) => c.card.domain),
  cards: ok.map((c) => ({ domain: c.card.domain, card_file: 'book/cartography/deepread/' + c.card.domain + '.json' })),
  blocked: cards.filter((c) => c && c.status === 'BLOCKED').map((c) => c.blocker_reason),
  cross_ok: PAIRS.length - mismatches.length,
  mismatches,
}
