# primer 原理章降台阶体系:符号纪律 + 论文精髓图重绘 + 先修分级 + 读者硬门禁

日期:2026-07-09
状态:设计已获用户确认(四项关键取舍均按推荐通过:忠实重绘+标出处/先修分级 Lead 批/reader 转硬门禁+台阶四问/试点 ch21→验收→8 章)
动机:用户反馈 primer 章「难度陡峭、认知台阶大;公式符号大多没有上下文解释;论文中最精髓的图没有贴出来(那些图是降低论文阅读难度的精髓);有些还需要递归读更多子论文」。
现状摸底:9 章 primer(vllm ch24/26/30,ascend ch09/21/23/26/31/34)公式块 7–19 个/章;图 7–10 张/章但全是自产机制图、**零论文原图重绘**;符号约定仅零散 1–4 处;论文包(book/papers/<slug>/)有 Figure 的文字引用但无图资产;haiku reader 评审维**当时跑了但是顾问性不门控**(blocking=false)——台阶问题正是从这里漏掉的。

## 1. 符号上下文纪律

- **素材**:explainer.json 增 `symbol_table: [{symbol, meaning, first_use, source}]`(symbol=LaTeX 原文如 `k_j^{C}`;meaning=一句人话;first_use=首现节;source=论文出处 §)。retrofit 期由诊断+素材 agent 从论文包与正文盘点;新章由 explainer 随 trace 一并产出。
- **正文**:writer 定点插「符号速查」表(位置:本章地图引用之后、第一个公式之前,markdown 表格,自然措辞);每个公式块**首现符号**在紧邻正文有一句人话解释(直觉优先于形式定义)。写作自由:表的详略、行文措辞不限;门禁只查覆盖。
- **门禁**(lint_paper_grounding 增,primer 章专属,**warn 级**):从 `$$` 块提取符号候选(独立单字母、带上下标的基名、`\mathrm{...}` 词;豁免数字/运算符/常见函数名 max/min/exp/softmax 等白名单),每个候选须出现在符号表或其首现公式 ±3 行正文中。启发式定 warn——判断力交给读者硬门禁(§4)。

## 2. 论文精髓图忠实重绘

- **盘点**:论文包 `meta.json` 增 `key_figures: [{fig: "Fig.2", arxiv, shows, why_essential, target_section}]`——「哪几张图是这篇论文降低阅读难度的精髓」。retrofit 期由诊断 agent 提候选;新论文包由 analyst 建包时登记(RUNBOOK primer 发车节补此要求)。
- **重绘**:illustrator **先取原图亲眼看**——从 arXiv HTML/ar5iv 抓原图文件下载后 Read;然后忠实重绘 SVG(布局与信息结构对齐原图,配色/字体套本书视觉语言,文字译中),图注固定句式「重绘自 arXiv:xxxx Fig.N:<一句话>」。抓不到原图(网络/版式)时按论文包文字描述重绘,图注改「按 arXiv:xxxx Fig.N（§y）描述重绘」如实标注。产物 `diagrams/paper-fig-*.{py,svg,png}`,登记 figure-manifest+盲审照旧(盲审员对照论文包描述核信息结构)。
- **版权口径**:重绘(非复制)+明确出处标注,出版惯例合规。
- **门禁**(lint_paper_grounding 增,确定性,blocking):meta.key_figures 每条须有章内图注含「重绘自 …Fig.N」(或「按 … 描述重绘」)的对应图;反向:章内「重绘自」图注的 Fig 号须在 key_figures 登记(防孤儿)。meta 无 key_figures 字段 → warn(包策展缺口)。

## 3. 子论文先修分级治理

- **诊断**:每章产出承重引用清单 `prerequisite ledger: [{concept, cited_paper(arXiv), where_used, load: light|heavy|critical, proposal}]`——判据:不懂该引用概念是否读不动本章推导(轻=一句直觉即可跟上;重=需要该子论文的核心构造/定理;极重=子论文本身值一章)。
- **处置**(Lead/用户批清单后执行):
  - light → writer 正文**先修框**(3–5 句直觉+出处 arXiv 号,blockquote 样式,自然措辞);
  - heavy → 论文包**扩容**:子论文核心片段入包(独立 md,# PAPER 锚;lint_paper_grounding 已 glob *.md 支持多文件),正文可引;
  - critical → 升级用户决策(扩章/立新 primer/接受)。
- **登记**:papers-map 对应条目增 `prerequisite_papers: [arxiv…]`;concepts.json 照常记账(gap-audit 判据②可覆盖)。

## 4. 认知坡度硬门禁(reader 升级)

- **chapter-pipeline 评审阶段**:reader 维对 **primer 章转 blocking**(码章保持顾问性):人格改「第一次读这篇论文的工程师(高级工程师,懂 Transformer 基础,没读过该论文)」,**逐公式过台阶四问**——①符号都认识吗(前文解释过/符号表有);②公式前有直觉铺垫吗;③从上一步到这一步跳步了吗;④需要先读别的论文才能懂吗。发现即 blocking=true 卡回 writer,并入既有 write↔review 有界回环(总轮数上限不变)。
- **写作模式**(writer 契约 primer 节补):每个公式块直觉句在前、数值例在后(explainer trace 即数值锚);这是模式要求非模板——写法自由,reader 门禁按效果验收。
- primer 判定沿用 dossier 顶层 `kind:"primer"`。

## 5. 执行:primer-uplift workflow + 试点

- **新 workflow** `.claude/workflows/primer-uplift.js`,两段式(Lead 批隔断):
  - **Phase A 诊断**(只读):每章一 agent 产 `reviews/uplift-diagnosis.json`——符号盘点(缺解释清单)/key_figures 候选/台阶点清单(逐公式四问预扫)/先修分级 ledger → 汇总返回,Lead 交用户批先修分级与 key_figures;
  - **Phase B 施工**(批后带 args 发):素材(explainer 补 symbol_table;heavy 先修的包扩容)→ illustrator 重绘(原图亲眼看→重绘→盲审)→ writer 定点(符号速查表/首现解释/直觉垫/先修框/图嵌入;**禁整章重写**)→ reader 硬门禁(台阶四问,≤2 轮)+ paper-fidelity 复核 → lint 电池(paper_grounding 新检查/trace 一致/structure/map/anchors/punct)。
- **铺开**:试点 ch21-primer-mla 全套 → 用户验收降台阶效果 → 其余 8 章批量(诊断批量→一次批→施工批量)。
- **固化**:analyst(建包记 key_figures)/explainer(symbol_table)/illustrator(重绘职责)/writer(三段式+符号表+先修框)/reviewer(reader 门控)契约;chapter-pipeline reader 门控开关;lint_paper_grounding 两组新检查;RUNBOOK primer 发车节同步。

## 6. 风险与对策

- **符号提取启发式误报**:定 warn 级+白名单;硬判断交 reader 门禁。
- **原图抓取失败**:降级为按包描述重绘+如实图注;不阻断。
- **重绘侵权边界**:忠实重绘指信息结构对齐,非像素复制;图注必带出处;更保守的「简化示意」留作个案选项。
- **reader 门禁误卡**(读者人格吹毛求疵):blocking 发现须给 suggested_fix 且 negotiable 标注;writer 可在回环里 rebut,评审收敛逻辑沿用现行 write↔review 机制;≤2 轮竭尽走逃生舱升级。
- **9 章工期**:试点先行;施工阶段逐章独立可并行,无 Bible 争用(符号表/先修框均章内工件)。
