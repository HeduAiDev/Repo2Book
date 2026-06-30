# ch29 交付：投机解码的 NPU 对位——proposer 工厂分发 + 薄壳继承 + 唯一重量级自研基类

- **Type**: delivery
- **Chapter**: 29
- **Date**: 2026-06-30
- **Timestamp**: 2026-06-30T19:12:19Z
- **Agents involved**: analyst, implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch29, speculative-decode, proposer-factory, thin-subclass, llm-base-proposer, prepare-inputs, propose-verify-loop, APPROVED

## What happened

ch29《投机解码的 NPU 对位》评审 APPROVED 并归档。核心范式『工厂分发 + 薄壳继承 + 少数重量级覆写』：(1) 工厂分发——vllm_ascend/spec_decode/__init__.py 的 get_spec_decode_method(method,...)（L33）一处 if-elif 把 method 字符串映射到 8 个 Ascend*Proposer（ngram/ngram_gpu/suffix/medusa/eagle|eagle3|mtp/dflash/draft_model/extract_hidden_states）；(2) 多数是薄壳——AscendNgramProposerNPU(NgramProposerGPU) 仅 35 行『把 GPU proposer 直接当 NPU 用』只覆写 propose、AscendSuffixDecodingProposer 是全章最薄标本、eagle/draft 各 17/19 行靠多继承（vLLM 策略语义 + Ascend 重量级基类构造/前向）；(3) 唯一重量级——llm_base_proposer.py(2043行) 的 AscendSpecDecodeBaseProposer(SpecDecodeBaseProposer)（L111）重写 prepare_inputs(L1701)/_propose，内含 ACLGraph(ch25)+昇腾 Triton spec_decode kernel+MLA(ch20)+昇腾并行组(ch08)，只挑骨架讲（重度减法）；(4) 与采样接口——proposer 提议 draft token、由 ch28 AscendRejectionSampler 验证，闭合『提议-验证』回路。review 12 条 issue 全 non-blocking（含数值追踪逐位验证通过 + reader-comprehension 维度建议）。已登记 7 个精简版接口到 bible；ch29 无 due 伏笔（plant/payoff 均空）。

## Why it matters

收束投机解码这条线：本章管提议侧、ch28 管验证侧，二者闭环。确立『能复用就薄壳继承（甚至把 GPU proposer 直接当 NPU 用）、必须特化才重量级重写』的工程取舍范式，是 Part VII 与全书『复用 vLLM 基类』主线的又一典范。prepare_inputs 的 new_seq_lens=seq_lens-num_rejected（代码无 +1，docstring 陈旧）的索引算术回流为下一步提议输入。

## What to remember

ch29《投机解码的 NPU 对位》评审 APPROVED 并归档。核心范式『工厂分发 + 薄壳继承 + 少数重量级覆写』：(1) 工厂分发——vllm_ascend/spec_decode/__init__.py 的 get_spec_decode_method(method,...)（L33）一处 if-elif 把 method 字符串映射到 8 个 Ascend*Proposer（ngra...
