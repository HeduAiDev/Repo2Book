# ch36 交付：DeepSeek-V4 CSA/HCA 两级压缩混合注意力（论文精读，MLA→DSA 演进收束章）

- **Type**: delivery
- **Chapter**: 36
- **Date**: 2026-07-05
- **Timestamp**: 2026-07-05T04:56:57Z
- **Agents involved**: archivist
- **User present**: False
- **Tags**: ch36, delivery, APPROVED, primer, csa, hca, mhc, kvcomp, paper-fidelity

## What happened

reviewer 判定 APPROVED（13 条 issue 全 negotiable/non-blocking，无阻断：1 条 lint_paper_grounding --expect-primer 对 6 个 mechanism 的 paper_ref 警告核实为 dossier.json 区间记号(如 Eq.(9)-(12))与 linter 字面子串匹配不兼容的机械误报、逐条对照论文原文确认公式编号/符号/推导均吻合无杜撰；1 条 lint_formulas.py 公式密度警告核实为论文精读章推导密集段的固有特性、无 NON-NEGOTIABLE 违规；11 条 reader-comprehension 润色建议(核注意力定义前置、CSA 盲区根源前置、CSA/HCA 交叠取舍动机、ReLU 打分理由、汉明距离直觉、混合精度切分理由、注意力 sink 应用场景、部分 RoPE 记号歧义澄清、KVComp 必要性量化、compress_ratios 示意声明)。本章是 ch31 MLA→ch32 DSA(稀疏注意力)演进线的收束章，四段式：谱系回顾(MLA 压维度→DSA 稀疏选块→CSA/HCA 合流)→动机(1M 上下文下 KV 与 FLOPs 的账)→推导(CSA 每 m=4 token 重叠 softmax 压缩+lightning indexer top-k 稀疏；HCA 每 m'=128 token 不重叠重压+稠密 MQA；两者层间交错互补)→数值推演(FLOPs 27%/KV 10% 逐项账本)→落地(kvcomp_utils.py KVCompConfig/HashEncoder/KVCompMetaData 与 models/deepseek_v4.py Compressor/Indexer/DeepseekV4Attention/DeepseekV2DecoderLayer)。论文包 book/papers/ch36-primer-v4-csa-hca/paper.md(arXiv:2606.19348)。已登记 6 条精简版接口签名 + 8 条机制→图注册到 bible；本章无待埋/待回收伏笔(bible.py due ch36 为空，arc-map.json 无 ch36/csa/hca 引用)；新增 4 个核心概念(CSA/HCA/mHC/KVComp)登记到 concepts.json 与 glossary.json。

## Why it matters

全书「原理篇 primer」系列的收束章——把 ch31(MLA，压 KV 维度)与 ch32(DSA，稀疏选块)两条独立演进轴在 DeepSeek-V4 里合流为 CSA(压序列长度+复用 DSA 稀疏)/HCA(更狠压缩+稠密兜底全局)的混合注意力，并交代运行期 KVComp 哈希选块工程近似与 mHC 流形约束超连接的落地。读者读完能看懂 vllm_ascend/models/deepseek_v4.py 里 compress_ratios 开关表驱动的层间交错设计，为什么两种压缩策略要交替堆叠而不是二选一。

## What to remember

reviewer APPROVED，13 条 issue 全非阻断(1 条 dossier paper_origin 区间记法与 linter 字符串匹配不兼容的机械误报核实 + 1 条公式密度启发式误报核实 + 11 条 reader-comprehension 润色，集中在核注意力定义前置/CSA 盲区根源/CSA-HCA 交叠取舍/ReLU 打分/汉明距离直觉/混合精度切分/sink 应用场景/部分 RoPE 记号歧义/KVComp 必要性量化)。6 接口 + 8 图已登记 bible，4 概念(CSA/HCA/mHC/KVComp)已登记 concepts.json+glossary.json，无伏笔缺口。
