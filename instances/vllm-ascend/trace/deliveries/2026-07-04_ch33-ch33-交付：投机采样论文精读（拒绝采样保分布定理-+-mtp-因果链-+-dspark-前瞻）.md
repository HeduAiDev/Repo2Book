# ch33 交付：投机采样论文精读（拒绝采样保分布定理 + MTP 因果链 + DSpark 前瞻）

- **Type**: delivery
- **Chapter**: 33
- **Date**: 2026-07-04
- **Timestamp**: 2026-07-04T09:10:34Z
- **Agents involved**: archivist
- **User present**: False
- **Tags**: ch33, delivery, APPROVED, primer, speculative-sampling, mtp, paper-fidelity

## What happened

reviewer 判定 APPROVED（22 条 issue 全 negotiable/non-blocking，无阻断）。本章四段式:动机(自回归串行受限)→推导(拒绝采样保分布定理完整证明:接受准则 min(1,p/q)、残差分布 p'=norm(max(0,p-q))、期望接受长度 E[L]=(1-alpha^(gamma+1))/(1-alpha))→数值推演(参考实现 speculative_sampling.py 复现论文 Table 1: alpha=0.8,gamma=5→3.689x)→落地(deepseek_v4_mtp.py 锚点，回指第 29 章 proposer 工厂)。末节前瞻 DSpark 明示 pin 版本无代码，指向 vllm-ascend RFC #11126，不做正文级推导。论文包在 book/papers/ch33-primer-speculative-sampling/(paper.md + paper-mtp.md 双包)。已登记 6 条精简版接口签名到 bible；本章无待埋/待回收伏笔(bible.py due ch33 为空，dossier.foreshadow_due 已确认)。

## Why it matters

全书第一个'原理篇 primer'系列章节之一(与 ch31 MLA primer 同属论文精读支线)，为 ch28/29 昇腾投机解码实现章提供理论根基——把 rejection_sampler.py 的向量化实现与 Leviathan 论文定理、DeepSeek-V3 MTP 论文的因果链结构对应起来，读者读完能看懂'为什么 min(1,p/q) 保分布'而不止'代码在做什么'。

## What to remember

reviewer APPROVED，22 条 issue 全非阻断(6 条 paper-fidelity/公式排版打磨 + 15 条 reader-comprehension 润色 + 1 条 linter 已知限制记录)。四段式:动机→拒绝采样保分布定理证明→数值复现论文 Table1→deepseek_v4_mtp.py 落地锚点。末节 DSpark 前瞻不做正文级推导。6 接口已登记 bible，无伏笔缺口。
