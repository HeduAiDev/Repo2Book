# 每章开篇「本章地图」:源码剖面图 + 选读路线

日期:2026-07-08
状态:设计已获用户口头确认(形态=源码鸟瞰由用户提出;生成=illustrator 逐章手绘;铺开=试点 4 章→验收→全量)
动机:用户反馈"每章叙事不够连贯,开篇应有一张图介绍本章地图,让读者理解叙事逻辑、选择性跳读"。现有开篇只有书级 roadmap(你在这章)+三行导语,章内怎么走全靠顺序读。

## 1. 图的定义:三层合一的源码剖面图

**不画目录,画代码**(用户核心思路):图主体是本章解读的真实源码架构鸟瞰,小节号只是挂在代码上的"讲解站牌"。

- **主体·源码走线**:本章代码从哪进(入口=调用方/上一章交出的接口)、经过哪些真实模块/类/函数(节点=真实符号名,泳道/分组=代码层次或执行阶段)、从哪出(出口=交给谁/下一章接什么)。
- **挂牌·§徽标**:每个代码站点挂 §N.M 徽标="这段在哪节讲"。一个站点可挂多节,一节可跨站点;不必每节都上图(纯过渡/收束节可不挂),但图上出现的 §号必须真实存在。
- **底部·阅读路线**:1–2 条,如「快通道(只要结论):§N.2→§N.8」「全通道:顺序读」。措辞按内容,不写死跨章号。
- **primer 原理章变体**:主体画**论文结构**(问题→关键构造→定理/算法→与落地章的接口),节点=论文概念/公式号(# PAPER 锚可核),其余同构。
- **与书级 roadmap 分工**:roadmap 答"本章在全书哪"(章级、全书统一图),本章地图答"本章内部怎么走"(节级+代码结构、逐章定制)。互补,均保留。

**位置**:开篇导航(你在这里/Roadmap 标题,若有)与 hook 段之后、第一个内容分节标题之前:`![本章地图:…](../diagrams/chapter-map.png)` + 紧随一段 1–2 句选读指引(自然措辞,禁"Cell N"式脚手架)。

## 2. 生产机制:illustrator 手绘 + 模板定调(用户选定)

- svg-diagram skill 新增参考实现 `references/example-chapter-map.py`,定视觉语言:横向泳道(代码层)/圆角节点(真实符号)/§徽标(角标胶囊)/底部路线条/入口出口箭头进出画布边缘。illustrator 按各章形态自由变体(线性章可退化为单泳道;分叉章画分流)。
- **输入**:成稿 `narrative/chapter.md`(节结构+叙事逻辑)+ `dossier.json`(mechanisms 真实锚点)+(primer 章)论文包。产出 `diagrams/chapter-map.{py,svg,png}`,登记进 `figure-manifest.json`(六项自查+盲审,流程照旧;盲审员核"入口→出口走线与 §挂牌能否复述本章逻辑")。
- **节点预算**:≤12 个代码节点;超长章聚合(如「§N.4–N.6 双算子路径」一站)。图宽 ≤1200,几何 lint 照旧。

## 3. 门禁(写作自由、门禁从严,成对落地)

新 linter `scripts/lint_chapter_map.py {chapter_dir}`,确定性检查:
1. **§徽标真实性**:解析 chapter-map.svg 文本节点,凡 `§N.M`/`N.M` 必须与正文实际 `## N.M` 标题集合匹配(N=目录号);
2. **符号真实性**:图上代码符号(反引号风格 token)须出现在 dossier.mechanisms 锚点符号或正文文本中(防杜撰);primer 章改核论文包文本;
3. **存在性与位置**(`--require` 时 blocking,供 pipeline/铺开后启用):第一个 `##` 前须引用 `chapter-map.png`,图后 300 字内有选读指引段;
4. 几何:并入既有 `lint_diagram_geometry`(无新逻辑)。
配套防复发(今日 36 处旧节号残留的病根):`lint_anchors` 增 warn——行文节号 `N.M` 的 N 非本章目录号、且同行无指向 chNN 的链接。

## 4. 流水线与契约集成

- **chapter-pipeline.js**:write↔review 回环**收敛(APPROVED)后、Archive 前**加轻量 **Map 站**(单 illustrator agent;图依赖定稿节结构——评审回环可能改节标题,故必须后置于收敛)。站内自检回环 ≤2 轮,BLOCKED 走逃生舱;末尾门禁加 `lint_chapter_map --require`。Review 阶段维度不变。
- **chapter-retrofit.js**:不改(地图缺失不算回修触发项;全量铺开由专项 rollout 完成,此后新章由 pipeline 保证)。
- **契约**:illustrator.md 增 chapter-map 职责节(输入/预算/自查项);writer.md 增开篇结构条款(hook 段后地图+选读指引);ARCHITECT-RUNBOOK 增发车说明。
- **铺开 workflow** `chapter-map-rollout.js`:pipeline(章列表, illustrator 画图+登记 → writer 插引用+指引句 → lint_chapter_map+geometry),逐章独立无 Bible 争用。

## 5. 铺开策略(用户选定:试点→验收→全量)

试点 4 章覆盖形态谱:ascend ch20(五态机分叉章)、ascend ch03(线性 patch 章)、vllm ch24-primer-flash-attention(原理章)、vllm ch36(超长收束章)。PNG 交用户验收、模板/预算调参定稿 → rollout 全量 72 章(试点 4 章豁免)→ 全绿后 lint_chapter_map `--require` 在 pipeline 内启用。

## 6. 风险与对策

- **地图与正文漂移**(改章后节号变):lint_chapter_map 进 linter 电池,批量跑;renumber 引擎已重写 SVG 内文字?——否,引擎只改 md/json;**补**:renumber 引擎 `_rewrite_targets` 纳入 `diagrams/*.py`(gen 脚本内 §号),SVG/PNG 由重跑 gen 脚本再生(rollout 后把「重编号后重生成 chapter-map」写进补章 SOP §4)。
- **超长章画不下**:节点预算+聚合规则(§2);盲审核"聚合是否掩盖主线"。
- **primer 章符号核对源不同**:lint 按 dossier `kind:primer` 分支(复用既有判定)。
- **72 章风格漂移**:模板参考实现+试点定稿的「变体边界」写进 illustrator 契约(可变:泳道数/分组;不可变:§徽标样式/路线条/配色语义)。
