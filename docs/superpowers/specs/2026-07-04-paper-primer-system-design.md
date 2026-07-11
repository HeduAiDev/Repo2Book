# 论文算法原理篇 + 系统级 gap 治理(primer kind)设计

日期:2026-07-04
状态:待用户评审
动机:MLA/DSA/V4 特色算法等"有论文出处的算法"在书中名词裸奔,读者撞认知悬崖(2026-07-04 实证盘点:ascend 书 6 处高严重度悬崖——解耦 RoPE 为什么/拒绝采样定理/EPLB 均衡算法/量化数学/DSA 论文谱系/MTP 铺垫)。
用户两条系统性要求:**① gap 问题在系统流程层面解决**(工厂能力,不是单书补丁);**② 质量靠流程保证**(成对门禁,不靠人工盘点)。

---

## 1. 总体形态

两层交付:
- **A 工厂能力(仓库无关,永久)**:primer 章 kind + 论文根基门禁 + gap 审计 workflow + 账本字段——之后每本书自动继承。
- **B 首次应用(vllm-ascend 书)**:Part VIII「DeepSeek 算法原理篇」6 章 + 存量 5 章先修指路框。

## 2. 工厂能力(系统级 gap 治理)

### 2.1 入口防漏:papers-map.json(cartography 阶段)
新书出架构地图时(RUNBOOK §0)同步产出 `instances/<x>/book/cartography/papers-map.json`:
```
{"papers": [{"algorithm", "paper": {"title","arxiv","sections_core_math"},
  "mechanisms_in_repo": ["file:Lxxx"], "primer_chapter": "chNN|inline|none",
  "rationale": "为何需要/不需要独立原理章"}]}
```
开局就规划 primer 章,不等成书后人工盘点。存量书补做(本次 ascend 即是)。

### 2.2 账本字段:dossier.mechanisms 扩展(逐章防漏)
`mechanisms[]` 新增可选字段:
- `paper_origin: {"paper": "arXiv:xxxx.xxxxx", "sections": ["§3.1","Eq.4"]}` ——凡"理解该机制需要论文"必须登记;
- `prereq: "ch31"` ——该机制的原理铺垫章(读者先修指路的数据源)。
`lint_dossier` 增量校验:字段存在时 schema 合法(arXiv id 格式、prereq 章目录存在);`kind=algorithm` 且无 `paper_origin` 时 WARNING(提示 analyst 确认是否确无论文出处,不阻断)。

### 2.3 全书 gap 审计 workflow:book-gap-audit.js(存量防漏,可反复跑)
并行扫指定章集(一个 Part 或全书),每章一个审计 agent(sonnet):
- 输入:chapter.md + bible glossary + papers-map;
- 判定:每个术语/概念**首现**是否三者居一——本章建立 / 前章已立(glossary 有 `defined_in` 且早于本章)/ 有先修指路链接;
- 输出:`book/audits/gap-audit-<date>.json`,逐条 {chapter, concept, severity: cliff|bump, evidence, suggested_fix};
- 汇总 agent 按严重度排序去重 → Lead 决定 retrofit / 立 primer 章 / 接受。
接入现有"每 Part 连贯性审计"位;bible `glossary.json` 条目增 `defined_in: "chNN"` 字段(archivist 回写时登记)。

### 2.4 primer 章 kind(流水线支持,chapter-pipeline.js 增 `kind:'primer'`)
8 阶段不变,四处契约切换(args.kind==='primer' 时提示词换文案):

| 阶段 | 码章(现状) | primer 章 |
|---|---|---|
| Dossier | 深读源码 | 深读**论文包**(§2.5)+ 落地代码;analyst 在 dossier.json 顶层写 `"kind":"primer"`(下游 linter/评审据此分流);embed_excerpts 可含论文公式(带 §/Eq 锚)与代码双源;mechanisms 必填 paper_origin |
| Implement | subtract-only 精简版 | **论文忠实小型参考实现**(NumPy/纯 torch-CPU,小参数可跑);每个函数 `# PAPER: §x Eq.y` 锚注。**硬规则 2 豁免仅限本 kind**,写进 CLAUDE.md/INSTANCE.md |
| Test | 复现仓库行为 | **验证论文性质**:拒绝采样保分布(统计检验)、吸收前后数值恒等、EPLB 重排后负载方差下降等——"复现论文断言"替代"复现仓库行为" |
| 门禁 | lint_fidelity | **lint_paper_grounding(新)**——豁免与替代门禁成对出现,不留真空 |

Explain/Illustrate/Write/Review/Archive **原样复用**:explainer 跑参考实现取 trace(数字可溯源)、插图盲审、trace 标记锁数字、锚点/标点/几何 linter 全套生效。
Review 维度对 primer 章:fidelity 维度替换为 **paper-fidelity**(评审拿离线论文包逐公式对账推导,evidence 必须引论文小节;auto-REJECT);其余 3+1 维不变。

### 2.5 论文包(发车前置物)
`instances/<x>/book/papers/<chNN-slug>/paper.md`(arXiv HTML 转 markdown,保公式)+ `meta.json`(title/arxiv/version/取回日期)。由 Lead 发车前 WebFetch 取回落盘——**不赌 workflow 内网络**。版权纪律:论文包仅内部参考;正文只引公式+出处,不整段转写。

### 2.6 lint_paper_grounding(新 linter,确定性)
输入 chapter_dir,仅当 dossier 标记 kind=primer(dossier.json 顶层 `"kind":"primer"`)时启用:
- implementation/*.py 每个 def/class 有 `# PAPER:` 锚(对标 lint_fidelity 的 # SOURCE 全覆盖);
- narrative 含 arXiv id;每个 `$$` 块 ±10 行内有 §/Eq/论文锚或"推导自上式"衔接(启发式,超限 WARNING、零引用 BLOCKING);
- dossier.mechanisms 的 paper_origin.sections 在 paper.md 中可 grep 到小节号。
码章不受影响(无 kind 标记即跳过)。

## 3. 首次应用:ascend 书 Part VIII「DeepSeek 算法原理篇」

### 3.1 章清单(依赖序;v4 调研 2026-07-04,V4 已发布,arXiv 2606.19348)
| 章 | 内容 | 论文 | 服务的码章 | 落地锚点 |
|---|---|---|---|---|
| ch31 | MLA:低秩 KV 压缩·解耦 RoPE(为何不可吸收)·权重吸收·q_lora | DeepSeek-V2 | ch20 | mla_v1.py |
| ch32 | 稀疏注意力谱系:NSA→DSA·Lightning Indexer 打分公式·top-k 训练协同 | NSA / V3.2 | ch21 | sfa_v1.py, dsa_v1.py |
| ch33 | 投机采样:拒绝采样定理(保分布证明+期望接受长度)·MTP·DSpark 前瞻节(ascend 仅 RFC #11126,无代码,节末对照) | Leviathan 2211.17192 + V3 | ch29 | deepseek_v4_mtp.py, spec_decode |
| ch34 | EPLB:专家负载均衡算法本体(层内/层间重排、方差目标) | DeepSeek EPLB | ch09 | eplb 模块 |
| ch35 | 量化数学:scale/zero-point→per-channel→GPTQ/AWQ/SmoothQuant | 各原论文 | ch27 | quantization 框架 |
| ch36 | V4 CSA/HCA 两级压缩混合注意力:m=4 softmax 压缩+top-k 稀疏(CSA)⊕ m′=128 重压缩稠密(HCA);1M 上下文 FLOPs 27%/KV 10% | arXiv 2606.19348 | ch21/ch22(演进线 V3.2→V4) | **kvcomp_utils.py, models/deepseek_v4.py**(pin v0.21.0rc1 已含) |

每章固定四段式:动机 → 数学推导(锚论文公式号)→ 小参数数值推演(跑参考实现,v3 trace 门禁)→ 落地(vllm_ascend 真实代码锚点,与码章双向链接)。
写作顺序:ch31→ch32→ch36 串行(符号/概念递进,bible glossary 衔接);ch33/34/35 相互独立可并行。

### 3.2 存量整合
- **先修指路框**:ch20/21/29/09/27 各一处定点 Edit(一句话+链接到对应 primer 章),bible arc-map 登记为伏笔/回收对(primer 章"落地"段回指即回收)。
- **roadmap**:roadmap.py 增 primer 高亮键(`primer-mla` 等→挂到所属子系统 spine 阶段,callout 文案"原理篇");
- **outline-final.json** 增 Part VIII;**INSTANCE.md** 记 kind=primer 豁免与 6 章状态;姊妹书对照约定沿用(V2/V3 论文机制在 vllm 基座的对应物照常指路)。

### 3.3 验收(闭环)
Part VIII 全部 APPROVED 后,**重跑 book-gap-audit 全书**:2026-07-04 盘点的 6 处悬崖必须全部消解(变为"有指路/有建立"),审计报告存档——用系统自己的审计证明 gap 已填,而非口头宣称。

## 4. 交付清单
- 改:`.claude/workflows/chapter-pipeline.js`(kind=primer 契约切换)、`scripts/lint_dossier.py`(paper_origin/prereq 校验)、`.claude/agents/{analyst,implementer,tester,reviewer}.md`(primer 分支段)、`instances/vllm-ascend/book/assets/roadmap/roadmap.py`、CLAUDE.md/RUNBOOK/INSTANCE.md;
- 新:`scripts/lint_paper_grounding.py` + tests、`.claude/workflows/book-gap-audit.js`、`instances/vllm-ascend/book/cartography/papers-map.json`、6 个论文包、Part VIII 6 章(走流水线产出)、5 个先修指路框;
- bible:glossary 增 defined_in 字段(archivist.md 一句话)。
- 模型分配沿用 spec §7(opus:analyst/explainer/writer;sonnet:其余)。

## 5. 风险与对策
- **论文包质量**(HTML 转 md 公式丢失)→ 取回后 Lead 抽查核心公式段;缺公式则补 PDF 页截图作 illustrator 参考。
- **V4 论文过新、二手解读少** → ch36 推导只依据论文原文+官方 vLLM 博客;单源断言在正文标注"据技术报告"。
- **primer 参考实现走样**(实现了但不是论文的算法)→ tester 论文性质验证 + paper-fidelity 评审双闸;拒绝采样类另跑统计检验(固定种子,阈值宽松防 flaky)。
- **DSpark 无代码可锚** → 仅作 ch33 前瞻节,明示"pin 版本未含,见 RFC",不做正文级推导。
- **先修指路框把码章改坏** → retrofit 同款纪律:定点 Edit+lint_anchors,禁动主体。
