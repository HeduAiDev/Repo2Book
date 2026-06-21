# Team Lead 上岗考（标准答案 / 评分要点 · v2 加难版）

> ⚠️ 考生（冷启动 Lead）**不得阅读本文件**。仅供判卷官与人工迭代用。每题 0/1/2。

## Tier 1 运营
- **Q1**：源码解读型技术书，直接解读真实 vLLM v1 源码（按真实模块，async 三段式旗舰，zh-CN 高级）。2026-06-21 转向：旧"从零简化重写+理论推导"太抽象/脱离代码，全部废弃。〔CLAUDE.md 顶部〕
- **Q2**：系统重建完成（4 linter+bible+vllm_docker 12/12 测试、6 角色、roadmap、chapter-pipeline+逃生舱、架构师文档）。磁盘：trace/state.json 已存在（默认）、trace/decisions/ 有记录、**artifacts/ 为空（尚无章节）**。下一步：发 ch04 试点。〔CLAUDE.md 当前状态 / RUNBOOK §9 / `archivist.py state`〕
- **Q3★**：`Workflow({name:"chapter-pipeline", args:{chapter_id:"ch04", slug:"ch04-async-llm", focus:"AsyncLLM 三段式异步解耦", highlight:"async-engine", source_root:".../instances/vllm/source", paths:["vllm/v1/engine/async_llm.py","vllm/v1/engine/__init__.py"]}})`。〔RUNBOOK §3〕
- **Q4**：`focus`+`paths` 来自 `cartography/map.json` 对应子系统 + `outline-final.json` 本章 subsystem/scope；`highlight` = `roadmap.py` 的 STAGES 键。〔RUNBOOK §3 / outline-final.json 每章 subsystem 字段〕
- **Q5★**：续跑 `Workflow({scriptPath:".claude/workflows/chapter-pipeline.js", resumeFromRunId:"<runId>"})`，已完成阶段命中缓存。发车可用 `name`（已注册）或 `scriptPath`；**续跑必须 scriptPath + resumeFromRunId**。〔RUNBOOK §3/§5〕
- **Q6**：`/workflows` 看实时进度；`TaskStop` 急停；`TaskOutput` 读阶段结果。〔RUNBOOK §4〕
- **Q7★**：**4 个 linter 脚本**——`lint_fidelity`、`lint_chapter_structure`（**roadmap + 自包含 + 零脚手架泄漏三项都在此一个脚本里**，非独立）、`lint_formulas`、`lint_source_grounding`；+ tester verdict + review verdict。命令见 RUNBOOK §6。〔CLAUDE.md 质量闸门 / RUNBOOK §6〕

## Tier 2 角色提示词契约
- **Q8**：analyst/implementer/tester/writer/reviewer/archivist；`.claude/agents/*.md`；workflow 用 `agentType:'general-purpose'` 调用，但每个 agent 提示词第一行强制"先读 `.claude/agents/{role}.md`"——确保角色契约加载、不赌 agentType 解析。〔CLAUDE.md / chapter-pipeline.js head()〕
- **Q9**：dossier 字段：code_spine、embed_excerpts、key_classes、data_flow、design_decisions、theory、subtraction_plan{delete,must_keep}、diagram_plan、foreshadow_due。embed_excerpts 逐字真源码因为书**自包含**（读者不开源码也要能懂）。〔analyst.md / chapter-pipeline.js Dossier〕
- **Q10**：implementer **只删 subtraction_plan.delete 批准项**、must_keep 符号必保留、**不得按己见删其他细节**；analyst 的 `delete=[{what,why_safe}]`（唯一批准删除清单）、`must_keep=[{symbol,why} 可检测符号]`；`lint_fidelity` 校验 must_keep 都在（缺=BLOCKING）。〔implementer.md / analyst.md / spec §3 / lint_fidelity.py over_subtraction〕
- **Q11**：writer 五契约：①主线=真实源码 ②自包含内嵌真源码 ③每章 Roadmap ④bible 读写(埋/回收) ⑤零脚手架泄漏。缺料→逃生舱 status=BLOCKED 让 implementer 补回，不将就。〔writer.md〕
- **Q12**：issue schema `{problem, suggested_fix, rationale, negotiable, blocking}`；必带 suggested_fix → 协作不死卡、给改法。auto-REJECT 维度：保真度(含过度删减+零脚手架泄漏)、算法可理解性、源码走读/公式可渲染。〔reviewer.md〕
- **Q13★**：验证**复现真实 vLLM 行为**（对照 dossier），非精简版自洽；因为精简版的价值在于忠实镜像 vLLM。判定**二元**（APPROVED/REJECTED），反压闸门。〔tester.md / chapter-pipeline.js Test〕

## Tier 3 编排/持久化/逃生通道
- **Q14★**：两层——【提示词+经验=持久（文件，版本控制）】vs【进程=按任务 spawn（非持久）】；archivist 全书持久，impl/writer/reviewer **章节级**（迭代期 SendMessage 续接，章末释放）。这样切是因为进程级常驻（旧 tmux）脆弱/会被压缩，改为"文件持久身份 + 外部持久记忆 + 按需进程"。〔CLAUDE.md / spec §5〕
- **Q15**：workflow 里 agent 一次性、不能活体来回、不能续接同一 agent；**切换时机**：硬骨头协作修订、同一问题 >3 轮、或任一阶段 BLOCKED 升级 → 由 **Lead（我）+ 命名 agent + SendMessage** 编排活体双向迭代。workflow 只管并行+确定性+结构化交接。〔spec §7.7 / RUNBOOK §5〕
- **Q16★**：逃生通道 = 每阶段 agent 返回 `status=OK|BLOCKED`；BLOCKED 时 workflow **早停**并 `return {escalated,reason}` → 我被通知。agent **不能**自己给我发消息或杀 workflow，但**能拉闸**（返回 BLOCKED）；我还能随时 `TaskStop`。〔chapter-pipeline.js ESC 常量+各阶段 check / spec §7.5 第4层〕
- **Q17**：5 个 stage：`dossier`(reason)、`dossier-verify`(**problems 数组**，非 reason)、`implement`(reason)、`write`(reason)、`review-revise`(reason)。处理：dossier/dossier-verify→修档案或 analyst；implement→修 subtraction_plan/implementer 提示词；write/review-revise→implementer 补 must_keep 或 Lead 活体迭代。〔RUNBOOK §5〕
- **Q18**：dossier 阶段有**对抗性自核**——第二个 analyst 独立核对档案（路线对不对？excerpts 逐字？must_keep 完整？），在 **implementer 动工前**拦住错误路线。〔chapter-pipeline.js dossier-verify / spec §7.5〕
- **Q19**：不失忆：①revision ledger（dossier+每次产出+反馈+diff，每次重做全量注入）②章节级命名 agent SendMessage 续接 ③有界(3)+升级 ④resumeFromRunId 缓存。〔spec §7.5〕

## Tier 4 设计原理
- **Q20**：旧体系**串联传工件**——implementer 杜撰的玩具代码成了 writer 的"教材"，writer 讲玩具不讲 vLLM。档案为真相源：implementer/writer 都吃同一份真实源码分析、谁都不以对方产物为准 → 结构性根除。〔spec §1/§3 / CLAUDE.md〕
- **Q21★**：不能赌常驻 writer 记忆——上下文 ~50 轮压缩，写到 ch30 早忘了 ch04 的伏笔。正解：外化为 **Book Bible**（arc-map 伏笔/回收、`bible.py due`）+ archivist + 每 Part 连贯性审计。〔spec §7.6 / CLAUDE.md〕
- **Q22**：只做减法 + 不能自行决定删什么 → 精简版始终是 vLLM 忠实子集 → 讲精简版≈讲 vLLM → writer 不必浪费篇幅解释杜撰代码（writer 聚焦真源码）。〔spec §3 / implementer.md〕
- **Q23**：**设计出来的**，非碰运气。大纲+依赖图+生命周期开局即全知 → 架构师写作前设计 arc-map（plant/payoff）注入每章 dossier。〔spec §7.6 B〕

## Tier 5 失败模式/场景
- **Q24★**：防过度删减多道防线：①analyst must_keep 声明必保留符号 ②implementer 只删 delete 清单 ③lint_fidelity over_subtraction 校验 must_keep 都在(BLOCKING) ④dossier-verify 完整性核对 ⑤reviewer 保真度维度 ⑥writer 逃生舱 BLOCKED 要求补回。〔spec §3 / lint_fidelity.py / chapter-pipeline.js / reviewer.md〕
- **Q25**：`lint_fidelity` 的 narrative_grounding（vllm 引用数 ≥ 精简版引用数，否则喧宾夺主）+ reviewer 保真度维度（叙事是否解读真源码而非精简版）。〔lint_fidelity.py / reviewer.md〕
- **Q26**：`lint_chapter_structure` 的 scaffold_leak 检测（BLOCKING）拦 instances/vllm/source 路径 / Cell N / 内部文件引用。因为正文是正式出版物，读者完全不懂脚手架。〔lint_chapter_structure.py / CLAUDE.md HARD RULE 3〕
- **Q27★**：ch20 的 writer 跑 `python3 scripts/bible.py due ch20` → arc-map 列出"应在 ch20 回收"的伏笔（埋于 ch04）+ archivist 再水化简报；读 Book Bible。**不靠它自己记得**。〔bible.py / CLAUDE.md 跨章连贯性 / writer.md〕
- **Q28**：读 `cartography/map.json` 的 async-engine 条目（key_files/citations）+ `outline-final.json` ch04 的 subsystem/scope 补全 paths/focus；必要时 spot-read 真源码（analyst 后续也会深读）。〔RUNBOOK §3 / cartography〕
- **Q29**：>3 轮：自动 loop-detection 升级 Lead；Lead 介入——命名 agent + SendMessage 活体迭代，或修提示词/档案，或拉用户。〔reviewer.md / spec §7.5 第3层 / RUNBOOK §5〕
- **Q30**：`scripts/vllm_docker.sh`（容器 vllm/vllm-openai:latest，vllm 0.15.1，CUDA）；精简版纯测试可 host；行号以源码 pin `f3fef123` 为准，容器仅用于观察/验证行为。〔CLAUDE.md HARD RULE 4 / spec §9 运行环境〕
