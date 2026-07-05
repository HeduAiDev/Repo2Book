# ch35-交付：primer 量化数学（GPTQ/AWQ/SmoothQuant，回指第 27 章）

- **Type**: delivery
- **Chapter**: 35
- **Date**: 2026-07-05
- **Timestamp**: 2026-07-05T06:39:19Z
- **Agents involved**: archivist
- **User present**: False
- **Tags**: primer, quantization, GPTQ, AWQ, SmoothQuant, ch27-callback

## What happened

原理篇 P8 论文精读：均匀量化 scale/zero-point/粒度基础 -> GPTQ 二阶补偿+三大工程优化 -> AWQ 激活感知缩放 -> SmoothQuant 迁移难度 s 因子，四段式(动机/推导/数值推演/落地)，9 个机制(M1-M9)全部通过逐机制勾选核验，回指第 27 章 vllm_ascend/quantization 框架。3 轮 write-review + 1 轮 blind review 全过，全部 linter 复绿。

## Why it matters

全书量化主题的数学根基，串起第 27 章昇腾量化框架的落地实现与三篇经典论文的算法推导；review 中 15 条 issue 全部 negotiable/non-blocking（2 条 lint_paper_grounding 字面格式误报、1 条图注引用措辞、9 条机制勾选表核验记录、若干 reader-comprehension 润色建议），无需返工。

## What to remember

原理篇 P8 论文精读：均匀量化 scale/zero-point/粒度基础 -> GPTQ 二阶补偿+三大工程优化 -> AWQ 激活感知缩放 -> SmoothQuant 迁移难度 s 因子，四段式(动机/推导/数值推演/落地)，9 个机制(M1-M9)全部通过逐机制勾选核验，回指第 27 章 vllm_ascend/quantization 框架。3 轮 write-review + 1 轮...
