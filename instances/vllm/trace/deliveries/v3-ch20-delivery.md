# v3 ch20《【primer】Flash-Attention 数学》交付归档（APPROVED）

- **Type**: delivery（v3 Archive 站，chapter-pipeline-v3 第二十章、Part V「GPU 不等 Python：执行管线」的原理章）
- **Chapter**: v3 ch20 · Part V · kind=primer（硬规则 2 豁免：论文忠实参考实现替代 subtract-only，门禁 lint_paper_grounding；全书四篇 primer 之一、v3 第二个 primer 章）
- **Pin**: vLLM v0.27.1（6e448d0ea）；行号基线即此版（v2 ch24 资产行号仅作线索）
- **Date**: 2026-09-05 · **Agents**: pipeline 各站（analyst→researcher→implementer→tester→explainer→illustrator→writer→reviewer）+ archivist（本记录）
- **Verdict**: APPROVED，18 条 issue 全 non-blocking（17 negotiable + 1 条 negotiable:false 的数字口径漂移；0 blocking），全文见 `artifacts-v3/ch20-flash-attention-math/reviews/review-report.json`

## What happened

- **回环**（`reviews/run-ledger.json`）：impl↔test 1 轮（46 passed host 纯 CPU NumPy 参考实现，事实出自 impl-notes，impl_test_ledger=[]）；write↔review 2 轮；L2 1 轮；盲审 1 轮零失败（13 图全 PASS）。foreshadow_due=[]、escalated=null。
- **归档时抽查（issue 兑现状态）**：negotiable 词级定点小修**均未在稿**（L3「6700 万」取整、L11 图注「底部」方位词、L221「走楼梯式」标签、L301「M/d² 约 25」、L305 图注 380px、L321 19.5 漏 FP32 限定、L342「跳上三角块」歧义、L580「上千行」未落数字、符号表密度/softmax_scale 首现/200704 构成/M 单位口径等）——以 APPROVED 归档（ch02/ch07/ch08/ch09/ch10/ch12 同先例），writer 定点小修清单留 review-report.json，最优先 issue-7（M/d²≈25 vs 图上 M=51200 自算 12.5，negotiable:false 的正文-图数字口径漂移）与 issue-2（19.5 加 FP32 限定）。台账侧三类本归档处置：issue-10（13 图未登记）**已收口**（见下）；issue-1/6（lint_paper_grounding 7+8 条 paper_ref/sections 假阴性——跨行子串 grep 抓不到、所引小节论文包逐一核实在）为 linter 归一化匹配 or dossier sections 写法的 curator 候选，留 Lead；issue-3（paper-fig-1/4 柱值提取 provenance 建议登记进论文包 meta.json key_figures）留 Lead。
- **bible 登记（v3 侧车）**：glossary-v3 **+16** = 11 条新登首现 ch20（HBM/SRAM/SM（streaming multiprocessor）/memory-bound·compute-bound/GEMM/safe softmax/⊕ 合并算子/softmax_scale/因果掩码与右下对齐/varlen 打平/split-KV（FlashDecoding））+ 5 条新登**首现如实记先现章**（online softmax、tiling→ch19 章尾钩子一词预告「online softmax 与 tiling 的数学，下一篇展开」；logsumexp（LSE）→ch08 log 域数值稳定技巧处已用已算；cascade attention→ch19 查表三出口「禁 FULL 的特性」一名带过并给中文名；warp/occupancy→ch17 SPMD 对照 lockstep 一词——各注明立义正主 ch20）+ **2 条存量勘正 ch22→ch20**（TMA：ch20 FA-3 版本族段给全称展开、书序在前；flash_attn_varlen_func：ch20 调用面整节先现并拆形参契约）——LoRA/fork/PAD 双哨兵勘正先例的延续。concepts-v3 **+13**（带宽墙/online-softmax 单遍递推/⊕/tiling 免物化/IO 账/FA-2 三改/FA 版本族/LSE 可合并/merge_attn_states 落地/cascade/split-KV/varlen 打平与右下对齐/穿页表读——对齐 pedagogy-plan introduces 三项 online softmax·tiling·LSE 并拆细）。interfaces-v3 **+ch20 三模块**（online_softmax/flash_attention/lse_merge，primer 论文参考实现，出自 impl-notes 1:1 Paper Map；含 FA-2 line 10 缩放方向按论文不变式修正的登记）。figures.json **追加 13 张**（book:v3、mechanism_id 对齐 dossier ch20-mNN 账本 + 论文四图按 key_figures_note 映射 m04/m05/m06）。
- **伏笔对账**：本章**应埋无、应收无**（pedagogy-plan F1-F10 的 planted/paid 集合均不含 20，与 run-ledger foreshadow_due=[] 及 dossier foreshadow_due should_plant/payoff 均空三方一致），**foreshadow-v3.json 零改动**。F7（planted 13→paid 22 已清）本章以 m10「kernel 沿 block_table 读分页 KV」形态再次路过——**路过加固非收款**（dossier note 同口径）；ch13 notes「伏笔埋：block_table → ch20/22」与 F7 paid=22 口径一致。ch21 depends_on=[19,20]：本章 introduces（online softmax/tiling/LSE）是 ch21「消费 ch20 数学」的直接前置（pedagogy notes：配对 ch21、ch21 标样板哲学）。
- **图登记门禁**：`REPO2BOOK_INSTANCE=vllm python scripts/lint_figures_registered.py instances/vllm/artifacts-v3/ch20-flash-attention-math` 显式传章目录 **exit 0**；manifest 13 图与 bible v3-ch20 条目**逐 id 集合相等**（本记录内程序核对 True）。注意：L1-partV 为**第二位持有章登记**（ch17 首登、ch20 按 chapter_id 再登记）——ch21 归档时其 manifest 若含 L1-partV（或自身 L2）须照此按各自 chapter_id 登记。

## Why it matters

Part V 的原理地基章：ch19 把 attention 留成 CUDA graph 外的不透明算子，本章把这个算子内部拆透明。全章压在一条主线上——**softmax 的归一化统计量在合并算子 ⊕ 下满足结合律与交换律，注意力可任意切块、任意顺序归并而结果精确不变**（多重集规范摘要的一行换元证明），动机是另一笔账（注意力慢在 HBM 往返不在算力：Alg.0 四次整表搬运、Θ(Nd+N²)→Θ(N²d²/M)、Prop.3 全域最优下界）。承重骨架全是可心算的 worked example（x=[1,3,2,5,4] 五轮递推、N=4 切 2×2 四步「每步都是至今为止的正确答案」、N=1024 元素级 IO 账 4456448 vs 933888），并四面落到 v0.27.1 源码：flash_attn.py:L1041-L1066 主路径一次调用零物化、merge_attn_states 六步（triton_merge_attn_states.py:L257-L322，FA2 inf/FA3 −inf 归一+双空护栏+NOTE(woosuk) 先算 scale 再乘）、cascade attention 两段调用与启发式（L1521-L1690）、varlen 打平/右下对齐/三断言与穿页表读（两种「块」撞名主动澄清）。FA-2 三改（循环序对调/推迟归一化/只存 L=m+log ℓ，line 10 印刷方向有误按不变式修正）与 FA 版本族（fa_utils.py:L163-L171 SM 代际决议、三代数学骨架不变）把「今天在跑的是什么」交代齐。primer 章范式第二次跑通（ch27 量化之后）：论文包三文（arXiv:2205.14135/1805.02867/2307.08691）+ 参考实现 + 论文图四张重绘（30 根柱值照录源图矢量文本层）+ lint_paper_grounding 门禁。

## What to remember

1. **【writer 定点小修清单】** 18 条 issue 的词级修补均未在稿（verdict 已 APPROVED、全 non-blocking），清单在 review-report.json；最优先两条：issue-7「M/d²（这里约 25）」与本章自己的 M=51200 口径算出的 12.5 打架（negotiable:false）、issue-2「非矩阵乘只有 19.5」补 FP32 限定。
2. **【lint_paper_grounding 假阴性两族，curator 候选】** 7 条 mechanism paper_ref + 8 条 dossier paper_origin.sections 告警全部为「跨行子串」假阴性（ar5iv 转换后小节标题与 Algorithm/Theorem/Eq 名不同行，连续子串 grep 抓不到）——review 给了两条修法（linter 第 3) 步把 key 匹配压平换行归一化，或 dossier sections 写成可连续命中的形式）；全假阳性告警每轮出现会训练「paper_ref 可忽略」习惯，与 ch24/ch33/ch27 的同族告警合并回流。
3. **【论文包路径沿用 v2 章位第二次】** 论文包在 `book/papers/ch24-primer-flash-attention/`（dossier.papers 明确指向），linter 按章目录名推导 `ch20-flash-attention-math` 报 WARN 不阻断——与 ch27（ch26-primer-quantization）同款既有约定；不改名论文包（meta.json 的 key_figures 登记都在那里）。
4. **【L1 Part 导览图第二位持有章登记先例】** ch17 首登 L1-partV、ch20 按 chapter_id 再登记同 figure_id——linter 按 chapter 分桶不冲突；ch21（Part V 末章）归档时若其 manifest 含 L1-partV/L2-ch21 须照此办理（ch27 的 L1-partVI 已给 ch23-26 同款预告）。
5. **【跨章首现勘正第三次/第四次使用 + 先现章如实记先例群】** 存量勘正 2 条（TMA、flash_attn_varlen_func ch22→ch20）、新登先现章如实记 5 条（online softmax/tiling→ch19、logsumexp→ch08、cascade→ch19、warp→ch17）——「首现如实、正主注明」惯例已稳定；另观察到 ch19 立义的 kernel/grid/thread block 从未入 glossary（通用 CUDA 词、译名无歧义），留 Lead 决定是否补账。
6. **【ch21 发车前读这条】** pedagogy-plan ch21 notes「消费 ch20 数学；回收 block_table 伏笔」——F7 账本 paid=22 已清，ch21 的块表消费属信息性回指（ch18/ch20 同款），**勿重复收款**；ch22 正文对「前两章」的前向链接链条至此只剩 ch21 未上线。
