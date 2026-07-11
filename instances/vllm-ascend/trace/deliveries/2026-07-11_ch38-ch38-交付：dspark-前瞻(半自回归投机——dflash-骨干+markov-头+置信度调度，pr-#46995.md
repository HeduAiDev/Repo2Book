# ch38 交付：DSpark 前瞻(半自回归投机——DFlash 骨干+Markov 头+置信度调度，PR #46995 前瞻解读)

- **Type**: delivery
- **Chapter**: 38
- **Date**: 2026-07-11
- **Timestamp**: 2026-07-11T09:50:11Z
- **Agents involved**: archivist
- **User present**: False
- **Tags**: ch38,  delivery,  APPROVED,  primer,  forward-looking,  dspark,  dflash-markov,  skip_impl

## What happened

reviewer 判定 APPROVED（9 条 issue 全 negotiable/non-blocking：2 条 lint_paper_grounding 因 dossier paper_origin 字符串与 paper.md 标题标点不一致、及符号下标裸词匹配限制导致的机械误报，逐条人工核对论文小节归属与符号提及均属实；1 条 draft/target 中英显式对应的可读性小修建议；1 条逐机制对账勾选表核验(dossier 十机制中 m1/m4/m5/m6 四个 core 机制直觉/推演表/不变量/量化四层证据齐全，lint_trace_consistency 客观核验通过)；1 条 lint_chapter_structure core_mechanism_missing_source 告警核实为 external-source 前瞻章特有的路径前缀假阳性(dossier source_anchors 带 ../book/external-source/dspark-pr46995/ 前缀、正文按硬规则 3 只标规范路径，二者字符串永不相等，与内容无关，逐一核对后确认四机制源码均已在正文内嵌)；2 条图注/跨章链接风格小修(3 张图图注未含结论、line 281 两处「第 34 章」裸文字未加链接)；1 条 derivation-audit 维度发现的因果连接词跳跃(『P_k 单调非增』不必然推出『Θ 必在 ≤N 步内触发早停』，二者是不同的终止原因，玩具数值例里恰好吻合，但通用情形依赖未言明的 SPS(B) 单调性假设)，建议拆句润色但不影响任何具体数字，全部玩具数值(softmax/Θ/低秩线性一致性/TV-D_LK 恒等式与反例)均已用 numpy 逐位复核为真)。本章是本书首个前瞻/外部快照 primer 章：开篇 Roadmap 与首段显著声明 DSpark 尚未合入 vllm-ascend(RFC #11126)，代码取自 vLLM 主线 PR #46995 @f5a8d73；正文如实交代『论文全貌』与『代码到哪』的落差——confidence_head 与 Algorithm 1 硬件感知调度器均『not wired into inference yet』(load_weights 显式跳过 confidence_head.*)，只有并行骨干+Markov 头是已落地代码。本章 skip_impl(V4/昇腾环境跑不起)——不配精简版，无 bible 接口新增；bible.py due ch38 为空，无待埋/待回收伏笔。论文包 book/papers/ch38-primer-dspark/paper.md。

## Why it matters

本章是 ch35(NPU 落地)之后的前瞻 capstone，衔接 ch34(投机采样定理)/ch37(DFlash 并行骨干)——把 DSpark 论文的半自回归 Markov 头机制讲透，同时给读者一份诚实的『代码现状快照』：论文机制≠已落地代码，这是本系列前瞻章的方法论示范(先读保真 spec docs/superpowers/specs/2026-07-11-forward-looking-primer-design.md)。9 条 issue 均非阻断，多数为跨章通用的 linter 已知局限(路径前缀假阳性、下标裸词匹配、区间记法字符串不匹配)，已记录建议供 curator 后续批量修复 linter 而非逐章补丁。

## What to remember

APPROVED，9 条 issue 全 negotiable/non-blocking。skip_impl 前瞻章：论文机制(confidence_head/Algorithm 1 调度器)与已落地代码(并行骨干+Markov 头)的落差已在正文讲清。derivation-audit 抓到一处因果连接词跳跃(早停非 P_k 单调性的必然推论)，建议拆句但不阻断，数值全部复核为真。
