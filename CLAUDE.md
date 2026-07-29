# repo2book — 把代码仓变成「源码解读型」技术书（v2 工厂）

把任意真实代码仓库变成**源码解读型**技术书：直接解读真实源码、按真实模块组织、正文自包含内嵌真源码逐段讲。

> 本文件 = **通用操作手册（仓库无关）**。**当前在写哪本书**由 `repo2book.json` 的 `active_instance` 决定 → 看 `instances/<active>/INSTANCE.md`（该实例的源码版本 / 当前状态 / 专属硬规则）。
> 新开一本：`python3 scripts/new_instance.py <name> --repo <git-url> [--title …] [--activate]`，再按 RUNBOOK §0 出地图/大纲。

## 📍 先读这些（架构师/编排者的持久文档）

你（主 session）是 **Team Lead / 架构师**，也会被上下文压缩。**不要靠记忆运转工厂**——靠这些持久文档：

- **操作手册**：`docs/superpowers/ARCHITECT-RUNBOOK.md` ← 怎么发车/监控/处理逃生舱/续跑（**运转工厂先读它**）
- **设计依据**：`docs/superpowers/specs/2026-06-21-vllm-source-reading-book-system.md`（方法论/角色/workflow/连贯性/协同/逃生舱——以 vLLM 为首例，方法论通用）
- **当前实例**：`repo2book.json.active_instance` → 实例配置 `instances/<active>/repo2book.json`、状态与专属规则 `instances/<active>/INSTANCE.md`、架构地图 `…/book/cartography/`、Book Bible `…/book/bible/`。
- **实例解析**：脚本统一经 `scripts/instance.py` 定位活动实例（或环境变量 `REPO2BOOK_INSTANCE` 覆盖）；linter 的 `--all` 自动扫活动实例。

## ⛔ HARD RULES

1. **叙事守护**：主编排者**不得**直接写/编 `artifacts/*/narrative/chapter.md`——只有 Writer 角色可写。质量不对就**改提示词，不改章节内容**。
2. **只做减法不做加法**：implementer 产出的精简版与源码**同名/同结构/同控制流，只删不增**，不杜撰源码没有的东西。（**唯一豁免**：kind=primer 原理章——论文忠实参考实现，成对门禁 lint_paper_grounding，见 RUNBOOK）
3. **零脚手架泄漏**：正文是正式出版物——规范源码路径（如 `vllm/...`，**绝不** `instances/<x>/source/`）、自然标题（**绝不** "Cell N"）、不提内部文件（dossier/impl-notes.md）。
4. **实例专属硬规则**见 `instances/<active>/INSTANCE.md`（如某栈调试须进容器、运行环境约束、源码版本/行号基线）。
- 可改：`.claude/agents/`、`.claude/workflows/`、`scripts/`、`docs/`、`CLAUDE.md`、`repo2book.json`、`instances/<active>/`。

## 核心方法论（修复"脱离代码"的脱节）

三支柱 + 写作规则：
- **A 档案即唯一真相源**：analyst 深读真实源码产出 `dossier.json`（含**要内嵌的真实源码片段** + 减法计划）。implementer 和 writer 都吃这份，**不以对方产物为准** → 结构性根除"writer 花篇幅讲 implementer 杜撰代码"的脱节。
- **B 只做减法的可运行精简版**：忠实子集，`# SOURCE:`/`# SUBTRACTED:` 全标注。**防过度删减**：只删 dossier `delete` 批准项、`must_keep` 符号必保留（`lint_fidelity` 校验）。
- **C 自包含、内嵌真源码**：不指望读者开着源码——正文直接内嵌真实源码片段（裁剪无关分支），逐段解读。精简版作"跑起来看数值"的交叉验证，不是主角。
- **D 素材先行**：图与数值轨迹先于写作、经运行验证产出（explainer.json 素材真相源 + figure-spec）；illustrator 强制"渲染→Read PNG 亲眼看→自查→盲审"；writer 拿素材自由叙事——**写作自由、门禁从严**。
- **E 概念查透先行（背景真相源）**：介绍初学者不懂的**非常识名词/标准记法/项目自定义模式/竞争性外部项目**，不许只给压缩括注——要带**充满好奇的专家**声线、刨根问底、**深入浅出**：讲清来龙去脉、记法/模式给具体例子、竞争项目给差异+如何选+权威链接。背景由 **researcher** 角色真去 Web 查透产 `research/concepts.json`（带 sources/confidence/writer_note），是 pipeline 的 Research 站产物、writer 消费——它是**读者定向的外部背景**，与 A(pin 源码档案) **自然分层**（例子明标「说明性/外部」、不标 `# SOURCE:`；版本敏感锚定 pin；立场性口径带出处）。首例 = vllm ch31/ch32 结构化输出回修（xgrammar/guidance/outlines 生态、structural_tag/EBNF 给例子）。
- **每章开场 Roadmap**：复用 `instances/<active>/book/assets/roadmap/roadmap.py` 出"你在这里"图 + 前后衔接。

## 运转工厂：混合编排（Workflow + 少量持久角色）

- **per-chapter workflow** `.claude/workflows/chapter-pipeline.js`：9 阶段 `Dossier→Research→Implement→Test→Explain→Illustrate→Write→Review→Map→Archive`，含 impl↔test / write↔review 有界回环、多维并行评审、插图盲审门禁(只看 PNG+spec 核论点/数字)、**逃生舱**（任一阶段返回 `status=BLOCKED` → 立即中止升级 Lead）、dossier 对抗性自核、Research 站(researcher 查透非常识概念产背景真相源,`skip_research` 可跳)。
- **发车**（详见 RUNBOOK）：`Workflow({name:"chapter-pipeline", args:{chapter_id, slug, source_root, focus, highlight, paths}})`。后台跑，完成/逃生舱触发会通知我；可 `/workflows` 看进度、`TaskStop` 急停、`resumeFromRunId` 续跑。
- **8 角色 = 持久提示词**（`.claude/agents/{analyst,implementer,tester,explainer,illustrator,writer,reviewer,archivist}.md`，已去 vLLM 化、仓库无关），workflow 经 agentType 调用 / 或 agent 自读契约。**持久分两层**：提示词+经验持久（文件），进程按任务 spawn（迭代靠 dossier+ledger+SendMessage 续接）。
- **存量回修** `.claude/workflows/chapter-retrofit.js`：外科式——逐机制体检(免修早退)→增量素材→补图/换错图→定点 Edit 算法段(禁整章重写)。
- **活体双向迭代 / 升级**：workflow 做不到的，由 Lead（我）+ 命名 agent + SendMessage 编排。

## 质量闸门（确定性 linter，前置于 LLM 评审）

```
python3 scripts/lint_fidelity.py {chapter_dir}                            # 保真度：# SOURCE 全覆盖/无杜撰/不喧宾夺主/无过度删减(must_keep)(primer 章不跑——换 lint_paper_grounding)
python3 scripts/lint_chapter_structure.py {chapter}/narrative/chapter.md  # Roadmap + 内嵌真源码 + 零脚手架泄漏
python3 scripts/lint_formulas.py {chapter}/narrative/chapter.md           # 公式可渲染
python3 scripts/lint_source_grounding.py {chapter}/                        # 源码根基
python3 scripts/lint_dossier.py {chapter_dir}             # v3:mechanisms 机制账本(锚点行号核真)
python3 scripts/lint_explainer.py {chapter_dir}           # v3:素材真相源(表格数字可溯源到 trace)
python3 scripts/lint_trace_consistency.py {chapter_dir}   # v3:正文数值表不漂移+机制覆盖
python3 scripts/lint_paper_grounding.py {chapter_dir}    # primer 原理章:# PAPER 全覆盖/正文有出处(码章空跑)
python3 scripts/lint_chapter_map.py {chapter_dir} --require   # 本章地图:§徽标↔标题/符号防杜撰/画布预算/开篇位置+选读指引
python3 scripts/lint_anchors.py --all   # 章内锚点+跨章三规(目标存在/文字号一致/../../ 深度)    python3 scripts/lint_punct.py --all   # 半角标点
python3 scripts/lint_diagram_geometry.py --all   # 图：文字越界/相撞/压框/箭头悬空（--all 走活动实例）
python3 scripts/lint_diagram_scaffolding.py --all   # 图面脚手架泄漏（内部路径 + 内部产物名/机制编号的裸用法）
#   ⚠️ 两个 --all 都只扫**活动实例**：换实例前先对旧实例显式跑一遍（exp-2026-07-21-14：
#   vllm-ascend ch37 三处图面泄漏就是这样长期照不到的，其中两处旧正则本就能抓）。
#   ⚠️ 图上「压框/相撞」有 rect-rect 盲区且**补不上**（exp-2026-07-21-13 负结果）——
#   靠 illustrator 契约的「渲染→Read PNG 亲眼看→自查」兜，别指望 linter。
#   ⚠️ 盲审必须**独立**：作图者自审天然看不见自己写的自证话术（ch09 就这么漏了一条泄漏）。
python3 scripts/lint_chapter_map.py --all   # 全书章图徽标↔标题一致（重编号后必跑：exp-2026-07-20-02 曾 7 章图印旧节号而 --all 全绿）
python3 scripts/lint_figures_registered.py   # 每章 manifest 图都登记进 bible/figures.json（archivist 反复漏登 chapter-map：ch23/25/26/27/32/33 六次；前缀容忍归一化比对，替掉 Lead 手工 grep）
python3 scripts/lint_arch_model_stations.py {chapter_dir}   # 架构模型图：正文引用的站号不超本章 code_spine 总站数（防插章/走线变化导致站号漂移；图上刻意不标的"其他章组件"站不误报）
```
机械问题让 writer 定点小修，不退整章。

## 跨章连贯性（不赌易失记忆 → 显式持久工件）

- **Book Bible**（`instances/<active>/book/bible/`，Archivist 持有）：术语译名、精简版接口注册表、**伏笔/回收登记**、声线指南。
- `python3 scripts/bible.py due {chapter_id}` → 本章应埋/应回收项；写后回写。（bible.py 经 `instance.py` 自动定位活动实例的 bible。）
- **伏笔是自顶向下设计的**（大纲+依赖图开局即全知），注入每章 dossier。
- **连贯性审计**：每完成一个 Part 跑并行审计（未回收伏笔/术语漂移/接口不符）。

## 记忆体系
- **Archivist**(唯一全书持久角色):trace 长期记忆 + Book Bible + concepts.json。`scripts/archivist.py`。
- **经验回流**(替代已退役的 wisdom/knowledge):每章落盘 `reviews/run-ledger.json`(回环轮数/盲审史);批次收尾跑 `book-retro` workflow 挖经验候选(≥2 章重复才算)→ Lead 批准 → curator 落笔进 linter/契约/skill/RUNBOOK/INSTANCE → 台账 `docs/superpowers/experience-ledger.md` 记录并在下次复盘验证生效(复发即升级落点)。
- **架构师自身连贯性**:本 CLAUDE.md + RUNBOOK + 实例 INSTANCE.md + trace 决策记录。

## 公式规则（NON-NEGOTIABLE，auto-REJECT）

**① 行内数学一律写 GitHub 转义式 `` $`…`$ ``**（`` 压到 $`d_c`$ 维 ``），**不要写朴素 `$…$`**。
朴素 `$…$` 在 GitHub(cmark-gfm) 上有**六**种静默失效方式，踩中任一整段吐裸源码——`压到$d_c$维`(紧贴 CJK)、`$ d_c $`(内侧带空格)、`一半;$N$ 涨`(前接半角标点)、`$\mathbf{q}_{t,j}$`(`}_{` 的 `_` 被 markdown 吃成 `<em>`)、`*图 1 同一 $L$*`(被单星号斜体包住)、以及被 `**` flanking 失败连累。**`` $`…`$ `` 对以上全部免疫**（正文/表格/粗体/斜体/列表/标题/紧贴 CJK 实测均渲染），LaTeX 源码逐字不变。一条规则替掉六条易错规则。
存量转换：`python3 scripts/fix_inline_math_escape.py <file.md>`（幂等）。

**② 块级数学一律 ` ```math ` 围栏,禁用 `$$…$$`**。`$$` 块会先过 CommonMark 反斜杠转义——`\,`→字面逗号、`\;`→分号、`\!`→`!`、`\{`→`{`（`\left\{` 变非法 `\left{`，GitHub 报 Missing delimiter）、`\\`→`\`（aligned/矩阵换行被砍）。` ```math ` 是代码围栏语义、逐字节免疫（引用块内 `> ```math` 同样成立）。存量转换：`python3 scripts/fix_display_math_fence.py <file.md>`（幂等）。与①同一条总原则：**数学内容永远放在不做 markdown 转义的容器里**（行内=code span，块级=code fence）。

**③ `**粗体**` 定界符外侧留半角空格**（内侧紧邻全角标点时必须）：❌ `是**「编译」…**` / ❌ `**…怎么读：**第一个` → ✅ `是 **「编译」…**` / ✅ `**…怎么读：** 第一个`。汉字既非空白也非标点，CommonMark flanking 不成立时 ** 原样显示、**并连累其中的数学**。修法同口诀：**空格永远在定界符外侧**。存量：`python3 scripts/fix_emphasis_flanking.py <file.md>`。

**④ 其余**：`\text{}`→`\mathrm{}`；`\boxed{}`→markdown 粗体标题；`\tag*{}`→`$$` 外；inline `\frac`→提升为 `$$` 块。inline 仅限单符号/简单式。**公式内禁中文/CJK**（strict KaTeX 报 unicodeTextInMathMode）。**禁把 LaTeX 命令塞进普通反引号**（`` `\mathbb{R}` `` 显示裸源码——注意 `` $`\mathbb{R}`$ `` 是数学、不是 code span）。

以上全部由 `lint_formulas.py` 阻断式校验。**真值口径**：GitHub 自家 markdown API——`python3 scripts/check_github_render.py <file.md>`（需网络+gh 鉴权，回归时跑；linter 是它的离线近似，已对齐到零漏报）。

## 当前实例 → 看 INSTANCE.md
- 活动实例、源码版本/行号基线、当前进度、实例专属坑：**`instances/<active>/INSTANCE.md`**（`<active>` = `repo2book.json` 的 `active_instance`，当前为 `vllm`）。
- **新建一本书**：`python3 scripts/new_instance.py <name> --repo <git-url> --title "…" --prefix <规范路径前缀> --activate` → scaffold 实例骨架 + blobless clone 源仓 → 按 RUNBOOK §0 出架构地图 + 大纲 → 逐章发车。

## 常见坑（通用）
1. 别写脱离代码的抽象——正文以真实源码为主线、自包含内嵌。
2. implementer 别过度删减/误删——只删 `delete` 批准项，`must_keep` 必保留。
3. 标记完成前跑全部 linter（含 `--all` 锚点/半角/图几何）。
4. 别赌自己的上下文——决策/状态写进 trace、Bible、本文档 / `INSTANCE.md`。
5. **提交后自动推送**（2026-07-14 用户定）：任务节点照常提交，提交完即推送、不必等用户指示。推送**须前台**跑且走 **gh HTTPS**（`git push https://github.com/HeduAiDev/Repo2Book.git <branch>`——SSH 22 端口被墙必超时，`gh auth setup-git` 已配好凭据）。
