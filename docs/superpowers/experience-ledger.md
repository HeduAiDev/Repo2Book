# 经验台账(experience ledger)

> 经验回流系统的生效验证账本:每条 = 一次"发现→批准→落笔"。retro 复盘时对照本表——
> pattern 复发 = 沉淀无效 → 升级落点(契约→linter);连续两次复盘未复发 → 标 proven。
> 详见 docs/superpowers/specs/2026-07-04-experience-backflow-design.md。

| id | 日期 | pattern | 落点(文件) | 针对指标 | 状态 |
|---|---|---|---|---|---|
| exp-0705-1 | 2026-07-05 | primer 章含 engineering-only 机制时 lint_dossier 强制 paper_origin 误伤(ch32 实例) | scripts/lint_dossier.py(paper_origin_note 豁免) | lint_dossier 对 primer 章误报数 | active |
| exp-0705-2 | 2026-07-05 | lint_paper_grounding 只读 paper.md,多论文包(paper-mtp.md 等)小节核对误报(ch33 评审发现) | scripts/lint_paper_grounding.py(Check 3 glob 全部 *.md 拼接再 grep——该修复原以未提交工作区改动形态存在,本轮正式落盘)+scripts/tests/test_lint_paper_grounding.py 回归测试 | paper_ref 误报数 | active |
| exp-0705-3 | 2026-07-05 | 术语/缩写/硬件型号/内部 flag/自定义 API 首现处不给一句人话释义,靠 reviewer 逐条枚举打回(合并 exp-wisdom-4) | .claude/agents/writer.md | reviewer reader-comprehension 维度 issue 条数 | active |
| exp-0705-4 | 2026-07-05 | 多维并行评审的维度划分有重叠,同一处内容偏差被两个维度/两轮各写一遍近乎逐字重复的 issue | .claude/agents/reviewer.md | review-report.json 重复/近重复 issue 条数 | active |
| exp-0705-5 | 2026-07-05 | explainer 选的 worked example 参数落在退化/巧合分支(整除嵌套/位移0/前后相等/经验=理论),机制效果数字不可见 | .claude/agents/explainer.md | reviewer algorithm 维度"示例落退化值"issue 条数 | active |
| exp-0705-6 | 2026-07-05 | 配图与正文最终稿数值/计数/示例数据不一致,或图未画出正文强调的核心现象,盲审只核标签数字不核性质 | .claude/agents/illustrator.md + .claude/skills/svg-diagram/SKILL.md | figure-integration/盲审维度"图-正文数值不一致/未体现现象"issue 条数 | active |
| exp-0705-7 | 2026-07-05 | writer 动笔前不强制核对 Book Bible glossary,为已登记 canonical 词另造新译名/用不一致译名 | .claude/agents/writer.md | coherence 维度"术语译名漂移/另造新词"issue 条数 | active |
| exp-0705-8 | 2026-07-05 | writer 用数字承诺流程步数/对象个数,写完不回读核对,与正文实际小标题数/枚举项数不一致 | .claude/agents/writer.md | coherence 维度"承诺 N 步/N 个与实际不符"issue 条数 | active |
| exp-0705-9 | 2026-07-05 | dossier 锚点行号系统性偏向紧邻空行/装饰器/global 语句而非符号定义行,阶段语义(训练/推理)与机制描述文不对题 | .claude/agents/analyst.md | lint_dossier/reviewer"锚点标错/阶段语义不符"blocking 条数 | active |
| exp-0705-10 | 2026-07-05 | F.linear(x, weight) 的 weight 形状是[out,in],用错形状不报错但数值静默错 | .claude/agents/implementer.md | tester REJECTED"形状对但数值错"类原因数 | active |
| exp-0705-11 | 2026-07-05 | 验证条件触发型行为(抢占/驱逐等)的测试须先证明触发条件本身成立,否则只是"没崩溃"非"验证了行为" | .claude/agents/tester.md | reviewer"测试通过但未覆盖目标分支"issue 数 | active |
| exp-0705-12 | 2026-07-05 | 主编排者不得直接写 chapter.md 目前只是 prompt 层社会约定,无技术闸门拦截误覆盖事故 | docs/superpowers/ARCHITECT-RUNBOOK.md | narrative 被整体覆盖/清空事故次数 | active |
| exp-0705-13 | 2026-07-05 | 衍生仓类实例(如 vllm-ascend)每章应钉住一个对位基座仓章节对照解读,现仅原则性表述未落成可检查动作 | .claude/agents/writer.md | 衍生仓连贯性审计"读者不知对应基座部分"issue 数 | active |
| exp-0705-14 | 2026-07-05 | 内嵌源码拼接非相邻区间/静默抽行却无省略标记,标注行号区间与展示内容存在隐性缺口 | scripts/lint_fidelity.py(新增 elision_gap/non_adjacent_splice,按 dossier.embed_excerpts 核对;实测全书语料后降级为非阻断——dangling_reference 悬空引用检查因误报风险更高而暂缓,未实现) | 保真度"省略标记漏标/行号区间错位/悬空引用"issue 数 | active |
| exp-0705-15 | 2026-07-05 | lint_source_grounding 源文件计数读内部 impl-notes.md 而非发布正文,对 OOT 插件类实例系统性误报 | scripts/lint_source_grounding.py(Check 4 改读 narrative,新增 impl_notes_incomplete 提示) | vllm_files_listed 非阻断误报数 | active |
| exp-0705-16 | 2026-07-05 | difficulty=core 机制缺内嵌源码/三段式子标题,评审与结构 linter 核验粒度是全章级,个别机制漏内嵌拦不住 | scripts/lint_chapter_structure.py(新增 core_mechanism_missing_source:逐 core 机制核 source_anchors 与正文代码块区间相交;ch31/ch32 逗号列表式 marker 漏认→按防回归硬规则降级 warn,解析补全后再升 blocking;三段式子标题检查为第二迭代未做) | core 机制"缺内嵌源码/三段式不完整"blocking 条数 | active |
| exp-0705-17 | 2026-07-05 | lint_formulas 密度启发式把单符号/简单变量 inline 也计入密度计数,数学密集章节噪音偏高 | scripts/lint_formulas.py(_is_simple_inline_formula,Check 6) | 数学密集章节 too_many_inline_formulas 告警数 | active |
| exp-0705-18 | 2026-07-05 | book-retro/book-gap-audit 的 args 若被注入为 JSON 字符串,args.instance 判断为假,静默回退到脚本内 CFG 默认值 | .claude/workflows/book-retro.js,.claude/workflows/book-gap-audit.js(内联 JSON.parse 护栏+log 告警);.claude/workflows/lib/resolve-cfg.js(纯函数单测参照,node --test) | CFG 静默回退导致报告文不对题的事故次数 | active |
| exp-0705-19 | 2026-07-06 | 全书 228 跨章链接相对路径差一层(../ 应 ../../),写手契约示例即错源且无 lint | scripts/lint_anchors.py 三规+writer 契约+引擎迁移一并修正 | anchors 三规 BLOCKING 数 | active |
| exp-0705-20 | 2026-07-06 | agent 做机械批量重映射易套错映射表(节号修复者 23 章错位)——机械迁移应确定性脚本+机器裁判,不交推理 | RUNBOOK 补章 SOP(引擎+validate 为准) | 迁移后节号错位数 | active |
| exp-0708-1 | 2026-07-08 | agent 转写长 JSON 落盘必碎转义(三份审计报告全损,journal 救回) | book-gap-audit/book-retro 去 write-report agent,RUNBOOK 处方 Lead python 落盘 | 报告 JSON 可解析率 | active |
| exp-0709-1 | 2026-07-09 | GitHub cmark-gfm 行内公式紧邻 CJK/全角即整段不渲染(858 处/29 章) | lint_formulas check9+公式规则入册+盘古之白批修 | check9 违例数=0 | active |
| exp-0709-2 | 2026-07-09 | 图类横幅超宽(6.9:1)排版后不可读——节点预算挡不住画布失控 | lint_chapter_map 画布预算(宽≤1500 比例≤2.6:1)+illustrator 契约 | oversize_canvas=0 | active |
| exp-0709-3 | 2026-07-09 | lint_diagram_geometry 容差设计漏报图例贴边重叠(单例:chapter-map 模板) | 暂不动阈值,复发≥2 再收紧;illustrator"Read PNG 亲眼看"仍是第一道 | 复发计数 1 | watching |
| exp-0709-4 | 2026-07-09 | 引擎不改 ## N.M 标题——中段插章后图徽标必与标题错位撞门禁(终审 Medium) | SOP §4 补同批改号句;理想解=引擎顺带改移动章标题(下次 --insert 前评估) | 下次 --insert 零门禁意外 | active |
| exp-0709-5 | 2026-07-09 | lint-appeasement 反模式:为清 symbol warn 把裸 d 与 d_h 配对入表,制造事实错误(d=模型维≠头维);门禁时序洞:reader PASS 后的微修只过机械 lint | RUNBOOK 后置微修纪律(语义改动重过 reader 抽查);linter 下标基名合并歧义待评估 | 同类符号并行错误 0 复发 | active |
| exp-0709-6 | 2026-07-09 | 行文推导路径与数值见证路径不一致(W^Q vs W^{UQ})——reader/盲审/paper-fidelity 均未抓到,唯亲手重推抓出 | derivation 硬维度(pipeline PRIMER 维+uplift DerivationCheck)+writer 落笔纪律 | 推导审计 Medium+ 0 漏 | active |
| exp-0709-7 | 2026-07-09 | 杜撰证据引用:正文引用不存在的测试断言/把容差校验说成逐位保证(ch22/ch27,判修 sweep 抓出) | 落笔纪律已含"等式须可执行见证";derivation 维 prompt 补"引用的测试/断言须真实存在"(下轮铺开验证) | 引用核真 0 杜撰 | active |
| exp-0710-1 | 2026-07-10 | named-workflow args 字符串化撞上 CFG 残留调试配置→静默错车重写已出版章(烧 692k;评审逃生舱救回未归档) | chapter-pipeline args 护栏:字符串解析+无 chapter_id 拒绝 CFG 回退直接终止;CFG 仅限手工无参调试 | 错车零复发 | active |
| exp-0711-2 | 2026-07-11 | 前瞻 primer(external-source)首用即触发 lint_chapter_structure 系统性假阳性(dossier 锚点带 ../book/external-source/ 前缀 vs 正文规范路径,4/4 core 机制误报缺源码层) | lint_chapter_structure 加 _norm_anchor_path 剥前缀再比+测试;前瞻 spec 已预言"复发即固化",首用即固化(掩盖真信号故不等 ≥2) | 外部快照章零误报 | active |
| exp-0712-2 | 2026-07-12 | 论断-证据失配:强论断(「根本没有 n_h」)配了不支撑它的证据(通用透传函数签名里露着 num_kv_heads)——论断真但视觉打架致读者困惑(ch21 用户抓;审计 148 论断+36 类比仅此 1 例,孤例) | reviewer algorithm-pedagogy 维加「论断-证据对齐」问法;保真度门禁只验"真源码/不杜撰"、管不到"证据是否展示论断依赖的特化值" | 复发计数(现 1,<2 不立硬 linter) | watch |
| exp-0712-3 | 2026-07-12 | reader 门禁对"割裂感"(术语漂移/数学↔代码未搭桥/源码块早于解释)盲视——用户抓 ch21 潜向量 c^{KV} 在码里叫 decode_k_nope、"解耦 key"=解耦 RoPE 分量=rope 部分 一物四名 | 双因:①rubric 是逐公式局部台阶四问、从不问跨段一致性;②跑 haiku——一致性需先"察觉多名共指"才能判困惑,弱模型对割裂给假通过(非"真读者认了"是"没察觉有东西可认")。修:reader→opus(primer)/sonnet(码章)+第五问全章一致性(pipeline+uplift,primer blocking) | 一致性类零漏发 | active |
