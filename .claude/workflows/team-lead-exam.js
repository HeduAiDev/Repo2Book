export const meta = {
  name: 'team-lead-exam',
  description: '冷启动 Team Lead 上岗考（v2 加难）：并行考生只靠仓库文档自我 onboard 并答 30 题 + 判卷官按标准答案打分、定位文档缺口',
  phases: [
    { title: 'Examine', detail: 'N 个失忆的 Team Lead 只靠文档作答 30 题' },
    { title: 'Grade', detail: '判卷官按标准答案打分、定位文档缺口与修复清单' },
  ],
}

const A = args || {}
const REPO = A.repo_root || '/mnt/e/Laboratory/Repo2Book'
const N = A.n || 3

// 题目直接嵌入脚本（不依赖 args 注入，避免上一轮 2/3 考生收到空题）
const QUESTIONS = A.questions || [
  { id: 'Q1', text: '项目现在做什么书？相比之前发生了什么转向、为什么？' },
  { id: 'Q2', text: '当前构建到哪一步？磁盘实况(state.json/artifacts/trace)与"完成度"是否一致？下一步紧要动作？' },
  { id: 'Q3', text: '★用户说"写第4章"。给出确切调用方式与全部参数（可直接照抄执行）。' },
  { id: 'Q4', text: '一章的 focus/paths/highlight 三个参数分别从哪个文件、哪个字段取？' },
  { id: 'Q5', text: '★怎样不重做已完成阶段地续跑一条流水线？发车与续跑用的 Workflow 入参(name vs scriptPath)有何不同？' },
  { id: 'Q6', text: '怎么实时看进度、必要时急停一条正在跑的流水线？' },
  { id: 'Q7', text: '★有哪些质量闸门？各自是独立脚本还是并入某脚本？分别怎么跑？注意 linter 的真实数量。' },
  { id: 'Q8', text: '6 个角色分别是谁？各自 .md 在哪？workflow 用什么机制调用？为什么每个 workflow agent 提示词第一行要"先读自己的 .md"？' },
  { id: 'Q9', text: 'analyst 产出的 dossier.json 必含哪些字段？为什么 embed_excerpts 必须是逐字真实源码而非摘要？' },
  { id: 'Q10', text: 'implementer.md 里具体哪条契约防止"过度删减/误删"？analyst 的 subtraction_plan 用 delete 与 must_keep 各起什么作用、形态要求？' },
  { id: 'Q11', text: 'writer.md 的五条强制契约是什么？writer 发现精简版缺了要讲清的细节，提示词要求它怎么做？' },
  { id: 'Q12', text: 'reviewer.md 的结构化反馈 schema 是什么？为什么每条 issue 必须带 suggested_fix？哪些维度是 auto-REJECT？' },
  { id: 'Q13', text: '★tester 验证的是"精简版自洽"还是"复现真实 vLLM 行为"？为什么？判定二元还是打分？' },
  { id: 'Q14', text: '★本体系的"持久化"分哪两层？什么持久、什么不持久？为什么这样切？' },
  { id: 'Q15', text: 'workflow(无状态、agent 一次性)什么时机切换到持久/命名 agent？触发条件+机制(谁编排活体双向对话)？' },
  { id: 'Q16', text: '★系统在哪里埋了逃生通道？一个 workflow agent 跑到一半发现路线错，能否中止并交给 Lead？机制是什么(它不能/能做什么)？' },
  { id: 'Q17', text: '逃生舱共有哪几个触发 stage？各返回什么字段(reason vs problems)？收到后按 stage 分别怎么处理？' },
  { id: 'Q18', text: 'dossier 阶段除了产档案还有什么对抗性步骤？它在何时、拦住什么？' },
  { id: 'Q19', text: '多次重做时如何保证 fresh/重启的 agent 不失忆、不重蹈覆辙？列出具体机制。' },
  { id: 'Q20', text: '旧体系(ch04 实测)怎么"脱节"的？为什么"档案即唯一真相源"能从结构上根除它？' },
  { id: 'Q21', text: '★为什么跨章一致性/伏笔/回收不能赌一个"全程常驻 writer"的对话记忆？正确做法是什么？' },
  { id: 'Q22', text: '为什么 implementer 必须"只做减法"且不能自行决定删什么？这对 writer 有什么直接好处？' },
  { id: 'Q23', text: '伏笔是"碰运气涌现"还是"设计出来的"？靠什么在写作前就把跨章弧线定好？' },
  { id: 'Q24', text: '★implementer 把 writer 要讲的关键符号删了。系统有哪几道防线会发现并纠正？尽量列全。' },
  { id: 'Q25', text: 'writer 写出的章节大段在讲精简版而非真实源码。哪个 linter/哪个评审维度会拦？依据是什么？' },
  { id: 'Q26', text: '正文出现 instances/vllm/source/... 路径或 "Cell 3" 标题会怎样？哪个闸门拦？为什么是问题？' },
  { id: 'Q27', text: '★ch20 要回收 ch04 埋的伏笔，但写 ch20 的 writer 是全新进程、没见过 ch04。它怎么知道要回收什么？' },
  { id: 'Q28', text: '想发 ch04 但 cartography/map.json 里这章 paths 不全/不确定。怎么补齐参数再发车？' },
  { id: 'Q29', text: '某章在 reviewer↔writer 之间反复 >3 轮还过不了。系统/你怎么处理(自动+人工两条路)？' },
  { id: 'Q30', text: 'vLLM 相关代码必须在哪运行、为什么？容器 vllm 版本与源码 pin 不一致，引用行号以谁为准、容器用来干嘛？' },
]
if (!QUESTIONS.length) throw new Error('exam aborted: no questions')
const Qtext = QUESTIONS.map(function (q) { return q.id + '：' + q.text }).join('\n')

const EXAM_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['answers', 'self_grade', 'blockers'],
  properties: {
    answers: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['id', 'answer', 'doc_cite', 'status'],
      properties: { id: { type: 'string' }, answer: { type: 'string' }, doc_cite: { type: 'string' }, status: { type: 'string', enum: ['answered', 'GAP'] } } } },
    self_grade: { type: 'number' },
    blockers: { type: 'array', items: { type: 'string' } },
  },
}
const GRADE_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['per_question', 'total', 'verdict', 'top_fixes'],
  properties: {
    per_question: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['id', 'avg', 'is_doc_gap', 'fix'],
      properties: { id: { type: 'string' }, avg: { type: 'number' }, is_doc_gap: { type: 'boolean' }, fix: { type: 'string' } } } },
    total: { type: 'number' },
    verdict: { type: 'string', enum: ['PASS', 'FAIL'] },
    top_fixes: { type: 'array', items: { type: 'string' } },
  },
}

function examineePrompt() {
  return [
    '你是一个**全新、刚打开**的 Claude Code 主 session（Team Lead），上下文被完全压缩——你对本项目和任何此前对话**一无所知**。',
    '工作目录 ' + REPO + '。**只能靠仓库自身的文档自我 onboard**（从 CLAUDE.md 开始，按它的指针走读 RUNBOOK / spec / cartography 等）。',
    '然后回答下面 30 道上岗考（题目已在下方，无需找别处）。规则：',
    '- 只根据**仓库文档实际写了什么**作答；每题在 doc_cite 注明你在哪个文件/小节找到依据。',
    '- 找不到依据 → status="GAP"，answer 写清"缺什么/哪里含糊"，**绝不猜测**。',
    '- 这些题不少是设计原理/场景/提示词细节题，要答出**机制与为什么**，不只是名词。',
    '- **不得打开** docs/superpowers/team-lead-exam-KEY.md（答案，作弊）。**只读**：不得改文件/启动 workflow/提交。',
    '',
    '题目（共 ' + QUESTIONS.length + ' 题）：',
    Qtext,
    '',
    'self_grade：给自己 0-10（仅靠文档你觉得能多大程度胜任 Team Lead）。blockers：列出阻碍你的文档问题。',
  ].join('\n')
}

phase('Examine')
const exams = (await parallel(Array.from({ length: N }, function (_, i) { return i }).map(function (i) {
  return function () {
    return agent(examineePrompt(), { schema: EXAM_SCHEMA, label: 'lead-' + (i + 1), phase: 'Examine', agentType: 'general-purpose' })
  }
}))).filter(Boolean)
log('收到 ' + exams.length + ' 份答卷')

phase('Grade')
const grade = await agent([
  '你是 Team Lead 上岗考(v2 加难)判卷官。先读标准答案与评分要点：' + REPO + '/docs/superpowers/team-lead-exam-KEY.md。',
  '下面是 ' + exams.length + ' 名考生（冷启动 Team Lead，只靠仓库文档作答）的答卷 JSON：',
  JSON.stringify(exams),
  '',
  '逐题给每名考生打 0/1/2（2=与标准答案一致且引对出处/能落地；1=部分对；0=错或 GAP）。深层题(Tier3-5)要求答出机制与为什么才给 2。',
  'per_question：每题 avg、is_doc_gap（avg<1.5 ⇒ 多数考生答不出=文档缺口）、fix（要改哪个文档、怎么补，具体）。',
  'total=各题 avg 之和（满分 60）。verdict=PASS 当 total>=50 且 ★PASS-gate 题(Q3 Q5 Q7 Q13 Q14 Q16 Q21 Q24 Q27)的 avg 均>=1，否则 FAIL。',
  'top_fixes：按"最能补洞/提分"排序的文档修改清单。',
].join('\n'), { schema: GRADE_SCHEMA, label: 'grader', phase: 'Grade', agentType: 'general-purpose' })

return { exams: exams, grade: grade }
