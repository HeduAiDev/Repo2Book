# v3 ch24《【primer】注意力变体数学》交付归档（APPROVED）

- **Type**: delivery（v3 Archive 站收口，chapter-pipeline-v3 第二十四章、Part VI「模型的形状」第二章——kind=primer，全书四篇原理章的第二篇）
- **Chapter**: v3 ch24 · Part VI · kind=primer（L0 缩放：（原理章）——无 L2/无站号，开篇以 L1-partVI 代位；配对 ch25《MLA 的两种展开》）
- **Pin**: vLLM v0.27.1（6e448d0ea）；行号基线即此版
- **Date**: 2026-09-12 · **Agents**: pipeline 各站（analyst→researcher→implementer→tester→explainer→illustrator→writer→reviewer）+ Lead + archivist（本记录与 bible/trace/state 回写）
- **Verdict**: APPROVED（六维评审 15 issue——3 blocking 全部修复在稿——figure-integration 一并收口）。终局报告 `artifacts-v3/ch24-primer-attn-variants/reviews/review-report.json`；分维存档：review-report-{algorithm-pedagogy, cognitive-ladder, formula-structure, paper-fidelity, figure-integration}.json + run-ledger.json

## What happened

- **论文包（primer 真相源）**：`book/papers/ch24b-primer-attn-variants/` 双文件——paper-gqa.md（arXiv:2305.13245 全文）+ paper-mla.md（arXiv:2405.04434 §2.1+§3.1.2+App B.1/C/D，附 DSV3 arXiv:2412.19437 §2.1.1 重述注记）。发车指令写的 `paper.md` 不存在（dossier.papers 为准，impl-notes 已核记）；lint_paper_grounding 的「论文包目录名 ch24- vs ch24b-」启发式 WARN 与 ch20 同款、不改名（key_figures 策展在 ch24b- 侧）。
- **Implement 站为完成中断运行**：3 个实现文件已在，修一处 einsum 头下标 bug（`"hts,sid->tid"` 把 probs 与 v 的头下标当两个自由指标、头被求和掉——n_h=2 时输出恰放大 2 倍，3 条一致性测试红）、补 12 处 `# PAPER:` 短锚（长注释块超 linter ±3 行窗口）、补 m08/m01 两个 worked example、补写 impl-notes.md。36 passed（host 纯 CPU NumPy float64，无需容器）。hard rule 2 豁免仅本章 kind：实现是论文忠实参考实现（非 subtract-only），门禁 lint_paper_grounding 无 BLOCKING。
- **write↔review 3 轮（run-ledger: write_review_rounds=3）**，六维 + 图系评审 15 issue，其中 3 条 blocking 全部修复在稿、终局 APPROVED：
  1. **cognitive-ladder blocking**——「与第 20 章 kernel 走读用的形状同构」为假前置（ch20 全文无任何 T=3/H=2/d_h=4 带头数玩具形状），按 suggested_fix ② 改为锚定 ch20 真正立过的 `(total, nheads, headdim)` 排布约定（L128 现核在稿）；
  2. **reader-comprehension blocking**——「位置平移只改旁路，缓存集合逐位不变」与自己引用的表④（k^R diff=1.9473，缓存里的 k^R 实测变了）及全章主账 576=512+64 矛盾，改为只声明真不变量「潜向量 c^KV 逐位不变；k^R 本就是位置的函数，随位重旋正是它的职责」（L411 现核在稿）；
  3. **derivation-audit blocking**——「总量 1/32 vs 每卡 1/4」口径矛盾（TP=8 只复制时全机合计同样只省 8/32=1/4）+「GQA 论文引 Pope 引的正是 1/4」过度归因，一句收紧为「MQA 模型账面（单份、不分片）省到 1/32，但 TP=8 复制下无论按每卡还是全机合计都只省到 1/4；GQA 论文引 Pope 引的正是『复制造成浪费』这一点（§2.2 原句只作定性论证）」（L191 现核在稿）。
  其余 12 条非阻断（repeat_kv 归属收回到 HF 一侧、L1 图注「标题带→底部」、MLAAttention 类限定符、S_q/S_kv 字义、d_h^R/k^R 首现括注、吸收主语、V3 式号注、lint_formulas 29 条机械告警滚动保留等）均已在稿或按契约存档。
- **图与盲审（blind_rounds=1 零失败）**：5 张机制图（mha-wiring/gqa-spectrum/mla-latent-compression/decoupled-rope-two-lanes/table1-account）盲审全 PASS；mla-latent-compression 经 revise r1 定点修（扇出区逐头标注 16384×512→128×512，gen 脚本 L79 单标签替换→重渲→Read PNG 亲眼看）后独立重盲审 PASS；L1-partVI 只拷贝不改造，但其权威产出点 gen_L1 本批新增裁切护栏连带修复三缺陷（spec decode 卡副行 2 被裁 80%/差量标签贴框/锥形虚线下根悬空）后盲审 PASS（护栏同批修复 Part III/IV/V/VII 同类潜在切字）；L0-architecture 拷贝图盲审 PASS。数值全部溯源 traces/ch24_m01..m09（explainer 素材真相源）。
- **bible 登记（v3 侧车）**：glossary-v3 **+11**（10 条首现 ch24：解耦 RoPE、潜向量、吸收、写头、连续分组、mean-pool 转换、uptraining、等效 GQA 组数、砍头路／降维路、蓝框；1 条跨章补账首现如实：RoPE→ch14——字面首现在 584B 布局的 128B 旋转段、ch21/ch23 黑盒接线，数学 ch24 展开；全仓逐章 grep 复核，惯例同 ch23 的补账四条）。concepts-v3 **+5**（MHA/GQA/MQA 数学谱系、MLA 低秩联合压缩、离线吸收、解耦 RoPE、四机制元素账与口径纪律——pedagogy-plan introduces 两项「MHA/GQA/MQA」「MLA 低秩压缩」被前两条覆盖）。interfaces-v3 **+ch24 3 模块**（mha_gqa/mla/kv_cache_table，primer 参考实现格式——kind=primer/# PAPER: 锚/lint_paper_grounding 标记，同 ch20/ch27 先例）。figures.json **+7**（L0-architecture 按 chapter_id=ch24 登记与 ch01 同 mechanism_id l0-panorama；L1-partVI 第三户登记——ch27 首登、ch23 第二户；5 张机制图 mech=ch24-m02/m04/m05/m08/m09），`lint_figures_registered` 显式传参章目录 **exit 0**。已立术语不重复登记：MHA→ch13、GQA/MQA→ch21、MLA→ch21、TP 切头与复制乘数→ch23。
- **伏笔对账**：本章**不埋不收**（pedagogy-plan F1-F10 planted/paid 集合均不含 24；foreshadow-v3.json ch24 零命中、零改动；dossier.foreshadow_due 与 run-ledger.foreshadow_due=[] 三方一致）。章尾给 ch25 的 Sq/Skv 两路分流引子是「配对章」结构（pedagogy-plan notes: 配对 ch25），非伏笔账目。

## Why it matters

Part VI 第二块、全书第二篇 primer：把 ch23 拼好的「Attention = 插座」积木翻开看插座内部的数学。主线一句话——**注意力变体是「KV cache 每 token 每层付多少元素」的账**：先立 MHA 基线 $`2 n_h d_h`$（式(1)-(8) 四拍与 llama.py 五行逐列同构），再走砍头路（MQA 砍到 1、GQA 沿头数轴插值 idx//(H/G)，端点与 MHA 合拢；mean-pool+α=0.05 uptraining 迁移配方），最后走降维路（MLA 低秩压缩 + 解耦 RoPE + 离线吸收），总账 576=512+64（≈1.75%、等效 2.25 组）。三条分水岭论断：**GQA 共享现成向量（离散头选择）vs MLA 共享生成基底（连续低秩子空间）**——App D.1 质量罚 vs D.2 反超的数学根因；**RoPE 逐位置矩阵非参数、夹在中间破坏结合律**（F(0,0)≠F(0,1) 数值见证）——解耦旁路 k^R 是「买回吸收权」的 64 元素价格，YaRN 只动这根；**口径纪律**——93.3/1.75/14-4 三个百分比比较基不同不可混用。落点：前三代变体= `total_num_kv_heads` 一个整数参数，MLA= 一套投影骨架（fused_qkv_a_proj [1536,576]/kv_b_proj 512→32768/process_weights_after_loading 拆 W_UK/W_UV 转置副本）+ 无 2 因子的 MLAAttentionSpec 页字节公式，经 get_kv_cache_spec（num_kv_heads=1、head_size=576）接 ch14 账本。下游接口全部预埋：ch25 吃 Sq/Skv 分流引子与 bmm 消费、ch26 吃 indexer 分支挂点（L205-L206）、ch27 吃 6-bit KV 量化口径、ch28 吃 fp8_ds_mla 584B 布局与 compress_ratio。评审侧沉淀：「论断真但证据口径错位」三连（repeat_kv 归属可被源码证伪、1/4 归因过度、假前置）——exp-0712-2 同型在本章的集中体现，primer 章「论文说没说过这句话」与「前章立没立过这个东西」要逐句抠。

## What to remember

1. **【primer 章接口登记先例】** 「无精简版跳过」不适用于 primer：ch20/ch27/ch24 均登记参考实现模块（本章 3 模块，格式带 kind=primer/# PAPER: 锚/lint_paper_grounding 标记）——参考实现就是 primer 章的交付物，后续 primer 章（ch27 已按此、ch32 投机解码数学等）照登。
2. **【RoPE 跨章补账】** 字面首现 ch14（584B 布局「128 B RoPE」）、ch19/ch21/ch23 均为接线级提及，glossary 一直空缺——本次归档补登（首现章=ch14、数学展开注 ch24）。跨章补账惯例（ch23 四条之后）第 5 例：归档时术语入账要全仓 grep 首现，不能只看本章。
3. **【L1-partVI 三户登记 + gen_L1 裁切护栏】** L1-partVI 由 ch27 首登、ch23 第二户、ch24 第三户（ch27 归档预告「ch23-26 归档时按各自 chapter_id 登记」兑现中）；ch25/ch26/ch28 照此。本批 L1 重渲连带修复 Part III/IV/V/VII 的同类潜在切字（gen_L1 运行台账），后续章开篇若见旧 L1 PNG 与新 SVG 不同源以 cartography 为准。
4. **【中断续跑的 einsum 头下标坑】** `"hts,sid->tid"` 型错误（头下标两名、静默求和掉）形状全对、数值恰放大 n_h 倍——兜底是论文三重循环直译对拍（rtol=1e-12）。同族经验：权重方向 W∈R^{out×in} 统一 `x @ W.T`，转置手滑也是静默数值错。
5. **【review 三 blocking 的共性】** 三条 blocking 全是「论断-证据失配」而非数学错：假前置（ch20 没有该形状）、口径扩写（「缓存集合」≠「压缩段」）、归因过度（GQA 论文没说 1/4）。primer 章引用密度天然高（论文+前章+源码三向），reviewer 的 derivation-audit/reader-comprehension 两维在本章抓的全是这类——契约「论断-证据对齐」比「数学对」更靠前。
6. **【python3 坏桩复现】** 本机 python3 仍是 WindowsApps stub（exit 49 静默），全部脚本走 Miniconda python——ch09 起各 delivery 已记，本章 lint_figures_registered 同此。
