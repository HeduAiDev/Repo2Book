# 前瞻 primer:解读 pin 之外/未合入上游代码的保真纪律

日期:2026-07-11
状态:随 DSpark 章首用落地
动机:用户要为 DSpark 立原理章,但 DSpark 未合入本书 pin 版(vllm-ascend v0.21.0rc1),只在 vLLM 主线(PR #46995)。要"读真源码"又不能伪装成 pin 树内容——需一套外部来源保真纪律,避免违反 HARD RULE 2(只删不增,以 pin 为基线)/HARD RULE 3(零脚手架)。

## 1. 何时算前瞻 primer
机制重要、有论文/上游实现,但**本书 pin 版源码树里没有**(尚未合入,或 pin 之后才进上游)。dossier 顶层加 `"kind":"primer"` + `"forward_looking":true`。

## 2. 外部来源快照(唯一真相源,可溯源)
- 落盘到 `instances/<inst>/book/external-source/<slug>/`:真实上游文件(gh api contents@ref 拉取,文件名 `a__b__c.py` = 上游 `a/b/c.py`)+ `PROVENANCE.md`(来源仓/PR/merge commit/拉取日)。
- 章内嵌这些片段时**每处必须标注**「来自 <上游仓> PR #NN @<commit7>,尚未合入本书 pin 的 <版本>——前瞻解读」。这不是脚手架泄漏(泄漏指内部路径/Cell N),而是**诚实的来源声明**,反而是保真所必需。
- 不得把外部片段伪装成 pin 树真源码;不得杜撰 pin 树里不存在的行号基线。

## 3. 与常规 primer 的差异(成对)
| 环节 | 常规 primer | 前瞻 primer |
|---|---|---|
| 源码来源 | pin 树 `source/` | `book/external-source/<slug>/` + 溯源 |
| 减法精简版 | 可配(源码能跑) | **skip_impl**(上游需特定硬件/模型跑不起) |
| 内嵌片段标注 | 规范路径 `vllm_ascend/…` | 规范路径 + 「PR #NN @commit 前瞻」溯源句 |
| lint_source_grounding | 对 pin 树 | 对 external-source 目录(片段仍是真代码) |
| 数值见证 | 精简版跑 | 论文数字(标来源+未独立复现) |
| paper_grounding | # PAPER 锚 | # PAPER 锚(照旧) |

## 4. 发车
`chapter-pipeline` args:`kind:"primer"`、`skip_impl:true`、`source_root` 指向 external-source 目录、`paths` 为该目录下文件、`focus` 内写明"每处上游片段标溯源句"。analyst/writer 读 external-source;reviewer 的 paper-fidelity 维加一项核"溯源句在场且 commit 号一致"。

## 5. 风险
- **上游漂移**:external-source 是快照,上游会变——PROVENANCE 锁 commit,正文措辞用"截至 PR #NN 的实现",不写死会随上游变的细节承诺。
- **读者误当已合入**:开篇 Roadmap 与首段必须显著声明"本章为前瞻,ascend 尚未合入,代码来自上游"。
- **复发即固化**:若 ≥2 章用前瞻模式,把"溯源句在场"提升为确定性 lint(现暂靠 review 维)。
