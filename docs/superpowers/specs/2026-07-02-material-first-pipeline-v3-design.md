# repo2book v3:素材先行流水线(Material-First Pipeline)设计

日期:2026-07-02
状态:待用户评审
适用:改工厂 + 外科式回修存量 63 章(vllm 33 + vllm-ascend 30)
执行模型约束:**全流水线跑 opus/sonnet**——所有环节的提示词必须自解释、程序化、每步有确定性验收,不依赖模型自行补齐隐含要求。

---

## 1. 问题诊断(2026-07-02 实证审计结论)

对两本书的机械审计(图覆盖/lint/抽样读图)全绿,但人类读者仍判定"算法讲解粗糙、难懂、插图错误多"。落差的结构性原因:

1. **门禁只测机械项,不测语义与教学效果**。`lint_diagrams`/`lint_diagram_geometry` 查 SVG 合法、文字不重叠;从不查"图讲的机制对不对、与源码/正文数值一致吗、读者 10 秒能看懂吗"。
2. **图覆盖按"章"统计,不按"机制"统计**。每章 2–5 张图,但没人负责"每个核心机制配一张准确的图"——机制级缺图无人度量。
3. **算法深度全靠 writer 提示词劝诫**("算法章另需…三件套")。没有独立环节把 worked example 产出为**经运行验证的工件**。强模型自觉遵守、弱模型随机丢——导致"最难的章好、中位章粗糙"的质量分布。
4. **全流程没有一步"渲染后亲眼看图"**。writer 写整章顺手画图,注意力被稀释;reviewer 只读 markdown,从不 Read PNG。
5. **同族模型评审倾向放行人类不满意的产物**——评审必须换形态:从"整体打分"改为"逐项对照可判定清单 + 盲审"。

## 2. 设计原则

- **P1 素材先行**:图与数值轨迹**先于写作**产出,且数字**来自运行精简版**(或源码常量锚点),不是想象。
- **P2 写作自由、门禁从严**(用户明确要求):不限制 writer 的行文/结构/风格;约束全部**前移到素材生产**(explainer/illustrator)和**后置到验收**(盲审/一致性 diff)。reviewer 不得要求特定行文风格,只查对错、缺漏、可懂性。
- **P3 opus/sonnet 可执行**:每个环节 = 一份 ≤60 行的程序化契约 + 一个类型化工件(JSON schema)+ 一条可自跑的确定性验收命令。质量判断拆成二元可勾选项(强制输出 schema 逐项填),不给"综合判断"留空间。
- **P4 有眼睛的验收**:凡产出图像的环节,强制"render → Read PNG(视觉)→ 逐项自查 → 修"回环;独立盲审 agent 只看 PNG + figure-spec(不看生成代码)。看图判错比凭想象防错容易得多——这是弱模型友好的关键设计。
- **P5 机制为单位**:覆盖度、配图、深度全部以 dossier 的 `mechanisms[]` 清单为账本,一图讲一机制、一例讲一算法。

## 3. 角色变化

| 角色 | v2 → v3 |
|---|---|
| analyst | 保留;dossier 新增 `mechanisms[]` 机制清单(见 §4.1) |
| implementer / tester | 保留,契约不变 |
| **explainer(新增)** | 教学设计师:跑精简版拿真实数值轨迹,产出每机制的"直觉+逐轮状态表+不变量论证+量化+figure-spec" |
| **illustrator(新增)** | 插图师:按 figure-spec 绘图,强制视觉自查回环 + 盲审;接管 roadmap 生成 |
| writer | **减负**:不再画图、不再生成 roadmap;拿经验证的素材库自由叙事 |
| reviewer | 维度重组(见 §4.5);新增"看图"义务(Read PNG) |
| archivist | 保留;bible 新增机制→图注册表 |

无删除角色:tester 作为独立反压闸门保留(防橡皮图章);archivist 是连贯性唯一持久载体。

## 4. 单章流水线 v3

```
Dossier → Implement → Test → Explain → Illustrate → Write → Review → Archive
                               (新)       (新)
```

### 4.1 Dossier(analyst,增量修改)

dossier.json 新增字段:

```
"mechanisms": [{
  "id": "m1",                       // 章内唯一
  "name": "抢占回退循环",
  "kind": "algorithm|dataflow|layout|protocol|config",
  "source_anchors": ["vllm/v1/core/sched/scheduler.py:L210-L260"],
  "needs_figure": true,             // 一图讲一机制的账本
  "needs_worked_example": true,     // 数值推演账本(algorithm 类默认 true)
  "difficulty": "core|supporting"   // core 必须三层递进,supporting 可只讲机制层
}]
```

`theory` 字段升级为结构化:`[{mechanism_id, complexity_claim, quantified:"O(...) 且代入典型参数后的具体数字"}]`。

确定性验收:`lint_dossier.py`(新)——schema 合法;每个 `kind=algorithm` 的机制 `needs_worked_example=true`;`source_anchors` 的 file:Lxxx 真实存在(读文件核行号)。

### 4.2 Implement / Test(不变)

skip_impl 章(meta/概览章)照旧跳过;此时 explainer 的数值轨迹退化为"手工推演 + 源码常量锚点"(见 4.3)。

### 4.3 Explain(新环节,explainer 角色)

输入:dossier + implementation(若有)。产物:`explainer/explainer.json` + `explainer/traces/*.json`(原始运行输出)。

对每个 `needs_worked_example` 的机制产出:

```
{
  "mechanism_id": "m1",
  "intuition": "一句类比/直觉(如:图书馆按页借书,还书只还整页)",
  "worked_example": {
    "params": {"blocks": 4, "block_size": 16, ...},        // 小而具体
    "trace_source": "run|manual",                           // run = 跑精简版所得
    "trace_ref": "traces/m1-preempt.json",                  // 原始输出存档
    "table": {"columns": ["轮次","动作","队列长","预算","判定","返回"],
              "rows": [[...], [...], ...]}                  // ≥2 轮
  },
  "invariant": {"claim": "...", "argument": "单调量或基例+归纳步,一句话骨架"},
  "quantified": "复杂度代入数字后的可比较量级",
  "figure_specs": [ <见 §5 figure-spec 格式> ]
}
```

**关键契约**:
- `trace_source=run` 时,table 的每个数字必须能在 `trace_ref` 原始输出中找到;explainer 写一个小驱动脚本跑精简版生成 trace(存进 `explainer/traces/`,含运行命令)。
- skip_impl 章允许 `trace_source=manual`,但每个数字必须标源码常量出处(file:Lxxx)。
- 逃生舱保留:发现 dossier 机制清单错/精简版跑不出可示教的轨迹 → BLOCKED。

确定性验收:`lint_explainer.py`(新)——schema 合法;每个 needs_worked_example 机制的 intuition/table(≥2 行)/invariant/quantified 非空;`trace_source=run` 的每个 table 数字在 trace_ref 里可 grep 到;每个 `needs_figure` 机制至少 1 个 figure-spec;figure-spec 里每个数字有 provenance(见 §5)。

### 4.4 Illustrate(新环节,illustrator 角色)

输入:explainer.json 的 figure_specs + roadmap 参数。产物:`diagrams/gen_*.py + *.svg + *.png` + `diagrams/figure-manifest.json`(逐图登记:figure_id → 文件三件套 → 自查 checklist 结果 → 盲审 verdict;lint_diagrams v2 校验 manifest 完整)。

流程(每张图,强制顺序):
1. 按 figure-spec 选模板(§5 模板库)→ 写 gen 脚本(全坐标计算,零手写 x/y)→ 渲染 SVG→PNG(rsvg-convert -z 2)。
2. **Read 渲染出的 PNG**(视觉),逐项填自查 schema:
   `{claim_readable_10s, numbers_match_spec:[{value, in_figure:bool}], no_overlap, arrows_attached, cjk_rendered, reading_order_clear}` ——任一 false → 修 → 重渲 → 重看,回环 ≤3 轮。
3. 跑 `lint_diagrams.py` + `lint_diagram_geometry.py`。
4. roadmap:illustrator 统一生成(从 writer 契约移出)。

**盲审(独立 agent,门禁)**:输入只有 PNG + figure-spec(**不给生成代码**),回答:
`{claim_understood: "用自己的话复述图的论点", matches_spec: bool, numbers_verified:[...], readable: bool, issues:[{problem, suggested_fix}]}` ——复述论点与 spec claim 不符或数字对不上 → 打回 illustrator,回环 ≤3 轮,超限升级 Lead。

### 4.5 Write(writer,减负 + 放权)

输入:dossier + implementation + **explainer.json + 已验收的 diagrams/**。契约改为:

- **自由**:章节结构、行文风格、叙事顺序、篇幅分配完全自主;voice-guide 是参考不是枷锁。
- **必达物**(不是"怎么写"而是"必须在场"):
  1. 每个 core 机制按"直觉→机制→源码"三层在场(顺序/篇幅自便);
  2. explainer 的逐轮状态表进正文(可改排版,**数字不可改**——`lint_trace_consistency.py` 会 diff);
  3. 每张已验收图被引用且出现在其机制讲解附近,图注给结论(可重写图注文案);
  4. 内嵌真实源码逐段解读、Roadmap 开场、bible 埋/收、零脚手架泄漏(原契约保留)。
- **逃生舱升级为双向**:writer 觉得某图不贴合叙事/想要新图 → SendMessage illustrator 提需求(带 figure-spec 草稿),不许自己画,也不许硬塞不合适的图。
- 不再画图、不再跑 roadmap.py、linter 自查从 5 个减到 4 个(diagrams 由 illustrator 负责)。

新 linter:`lint_trace_consistency.py`(新)——正文中标记为数值推演的表格数字 ⊆ explainer.json 对应 table 的数字集合(允许排版差异,不允许数字漂移)。writer 收工前自跑;reviewer 的 algorithm-pedagogy 维度复跑作门禁。

### 4.6 Review(reviewer,维度重组)

并行维度(门控 4 + 顾问 1,回环 ≤3 轮不变):
1. **fidelity**(原样保留:保真/过度删减/零泄漏)。
2. **algorithm-pedagogy**(改造):逐机制对账——core 机制三层在场?状态表与 explainer 一致(跑 lint_trace_consistency)?不变量论证在场?量化落数字?输出为**逐机制勾选表**,不是整体打分。
3. **figure-integration**(新):**必须 Read 每张 PNG**,查:图在其机制附近?图注给结论?正文提到的数字与图一致?
4. **formula-structure**(原维度去掉 diagrams 职责,只剩公式/Roadmap/自包含)。
5. reader(haiku 读者视角,顾问性不门控,保留)。
- **评审纪律(写进 reviewer 契约)**:不得以风格偏好要求重写;所有 issue 必须 `{problem, suggested_fix, evidence(引用原文行/图), blocking}`;无 evidence 的 issue 无效。

### 4.7 Archive(archivist,增量)

bible 新增 `figures.json`:机制 → 图 → 章的注册表(跨章引用同机制时可复用/指路);其余不变。

## 5. "一张好图"方案(svg-diagram skill v2)

图 = **论点 + 数据 + 版式 + 双重验收**。写进 `.claude/skills/svg-diagram/SKILL.md` v2:

### 5.1 figure-spec(先有 spec 再动笔,无 spec 不绘图)

```
{
  "figure_id": "fig-m1-preempt",
  "claim": "一句话:这张图让读者看懂什么(如:抢占按 LIFO 弹出 running 尾部,每轮队列长严格减 1)",
  "template": "state-table|swimlane|layout|tensor-flow|before-after|flow",
  "numbers": [{"value": "512", "provenance": "traces/m1.json#rows[0][2] 或 vllm/...:L123"}],
  "elements": ["图中出现的每个视觉组及其含义"],
  "caption_draft": "图注草稿(给结论,不描述画面)"
}
```

### 5.2 设计规则(可判定,写成清单)

- **一图一论点**:claim 写不成一句话 → 拆成两张图。
- **元素预算**:每个视觉组 ≤7 个元素;超了必须分组加留白或拆图。
- **颜色即语义**:颜色只编码状态/类别,不做装饰;>2 种语义色必须有图例。
- **数字皆有出处**:图中每个数字来自 trace 或源码常量(spec.numbers 列全)——杜绝"示意数字"。
- **阅读顺序显式**:从哪看起要么符合左上→右下,要么用 ①②③ 编号。
- **图注给结论**:图注 = claim 的读者版,不写"本图展示了…的结构"。
- 模板库扩充(各配参数化 gen 脚本示例):现有 tiling/state-table/flow + 新增 **swimlane 泳道时序**(跨组件协议)、**layout 内存/张量布局**(块表/KV 页)、**before-after 双态对比**(优化前后)、**state-machine 状态机**。

### 5.3 双重验收(P4 的落地)

1. 机械:xmllint + validate_svg + lint_diagrams + lint_diagram_geometry(现有,保留)。
2. **视觉自查**:绘图 agent 自己 Read PNG 填 checklist(§4.4 第 2 步)——skill 里写明"没看过渲染结果的图 = 未完成"。
3. **盲审**:第二个 agent 只看 PNG+spec 复述论点、核数字(§4.4)。

## 6. 存量回修:chapter-retrofit workflow(新)

外科式,不重写章节主体。按章跑(`args:{chapter_id, slug, instance}`),阶段:

1. **Diagnose**:agent 读 chapter.md + dossier(缺 mechanisms 则现场补一份轻量清单),**Read 全部 PNG**,产出 `retrofit/retrofit-plan.json`:逐机制 `{depth: ok|shallow, figure: ok|missing|wrong, actions:[...]}`(每条 action 带 evidence)。plan 为空 → 本章免修,流水线直接结束。
2. **Explain(增量)**:只对 flagged 机制跑 explainer(有精简版的章 trace_source=run)。
3. **Illustrate(增量)**:补缺图/重绘错图,走完整视觉自查+盲审;修复 ascend ch02 无 gen 脚本问题(重建生成器)。
4. **Write(定点)**:writer **只许 Edit 算法段与图引用处**,契约明写"禁止整章重写、禁止动已 APPROVED 的非算法叙事、保持既有锚点标题不变"(lint_anchors 校验)。
5. **Review(缩编)**:只跑 algorithm-pedagogy + figure-integration 两个门控维度 + 全部机械 linter。
6. **Archive**:trace 记 retrofit delivery,bible figures.json 登记。

批量驱动:先用 sonnet 对 63 章各跑 Diagnose(便宜、只读),按 flagged 机制数排序,算法重章优先,Lead 分批发车。

## 7. 模型分配建议(可被 workflow opts.model 覆盖)

| 环节 | 模型 | 理由 |
|---|---|---|
| analyst / explainer / writer | opus | 深读源码、教学设计、叙事是推理最重的三处 |
| implementer / tester / illustrator / reviewer 各维度 | sonnet | 程序化契约 + 确定性验收兜底 |
| reader 顾问 | haiku | 现状保留 |

即使全降 sonnet,P3/P4 的结构(类型化工件 + 确定性 linter + 视觉回环 + 盲审)仍兜底质量下限——这是"opus/sonnet 能执行"的核心保证:**质量不系于单个模型的自觉,系于工件与门禁**。

## 8. 交付清单(实施范围)

新文件:
- `.claude/agents/explainer.md`、`.claude/agents/illustrator.md`(各 ≤60 行程序化契约)
- `scripts/lint_dossier.py`、`scripts/lint_explainer.py`、`scripts/lint_trace_consistency.py`
- `.claude/workflows/chapter-retrofit.js`
- svg-diagram skill v2(SKILL.md 重写 + 4 个新模板示例脚本)

修改:
- `.claude/agents/{analyst,writer,reviewer,archivist}.md`(按 §4 增删职责)
- `.claude/workflows/chapter-pipeline.js`(插入 Explain/Illustrate 两阶段 + 新 schema)
- `CLAUDE.md` / `docs/superpowers/ARCHITECT-RUNBOOK.md`(流程图与发车说明同步)

不动:implementer/tester 契约、bible/trace 机制、现有 5 个 linter、两本书已 APPROVED 的非算法叙事。

## 9. 风险与对策

- **explainer 跑不出可示教 trace**(精简版接口不便驱动)→ 逃生舱 BLOCKED 升级 Lead;允许降级 manual+源码锚点,但 lint_explainer 强制标注降级原因。
- **盲审假阳性卡死**(图没错被打回)→ 回环 ≤3 + 升级 Lead;盲审 issue 必须给 suggested_fix,机械分歧以 linter 为准。
- **retrofit 误伤既有叙事** → writer 契约禁整章重写 + lint_anchors 保锚点 + git diff 审计(Lead 抽查 diff 行数异常的章)。
- **成本上升**(每章 +2 环节 + 盲审)→ illustrator/盲审用 sonnet;retrofit 免修章在 Diagnose 后即终止。
