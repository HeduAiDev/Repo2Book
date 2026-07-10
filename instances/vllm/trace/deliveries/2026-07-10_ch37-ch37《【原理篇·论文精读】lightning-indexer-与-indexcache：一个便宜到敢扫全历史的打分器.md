# ch37《【原理篇·论文精读】Lightning Indexer 与 IndexCache：一个便宜到敢扫全历史的打分器》交付-approved

- **Type**: delivery
- **Chapter**: 37
- **Date**: 2026-07-10
- **Timestamp**: 2026-07-10T16:58:51Z
- **Agents involved**: archivist, writer, reviewer, illustrator
- **User present**: False
- **Tags**: primer, dsa, lightning-indexer, indexcache, mxfp4, paper-grounding, ch27-link

## What happened

ch37-primer-lightning-indexer（原理章，DeepSeek Sparse Attention 论文包 arXiv:2512.02556 + CSA/V4 论文包 arXiv:2606.19348 精读）四层结构：①打分数学——轻量 q·k 打分函数 Eq.(1)-(2) 与 top-k 选块 Eq.(4)，独立小头架构与训练期 KL 对齐理由，O(L^2)→O(Lk) 复杂度诚实账；②缓存工程——索引器自己的 KV 缓存（K^IComp 与 C^Comp 并行构建，V4 报告 Eq.(9)-(19)），IndexCache 张量布局与主 KV cache 的独立性，vLLM 里 DeepseekV32/V4IndexerCache 的真实实现落地；③量化变体——use_fp4_indexer_cache（MXFP4 QAT，2x top-k 提速/99.7% 召回）；④接线——SparseAttnIndexer CustomOp 与 v1/attention/backends/mla/indexer.py 后端如何被 deepseek_v2.py 模型消费。参考实现 7 个文件（lightning_indexer.py/csa.py/index_cache.py/kl_alignment.py/mxfp4_quant.py/wiring.py/complexity.py）忠实复现论文算法，39 个测试全过（host 纯 NumPy/CPU，验证对象为论文断言，非目标代码仓行为复现——kind=primer 豁免 subtract-only，改用 lint_paper_grounding 门禁）。reviewer verdict=APPROVED，6 条 issue 全 non-blocking/negotiable：1 条 lint_paper_grounding --expect-primer 的 6 条 paper_ref 警告经人工核对为 dossier 小节标签措辞与论文包原文断句不完全一致的机械误报，非事实错误；1 条 algorithm-pedagogy 维度逐机制勾选表（9 个机制全部核对，6 个 core 三层齐全，3 个 supporting 合理精简）；1 条 kl-alignment-separate-optimization 机制的 source_anchor（_try_load_fp8_indexer_wk L740-777）未字面嵌入、改用旁证注释推断（诚实非杜撰，可协商）；1 条 fig-mxfp4-before-after 图注缺 "(右)" 后缀与 meta.json/图内标注不一致；1 条行内公式密度/复杂度超出 CLAUDE.md 字面尺度但 lint_formulas 判 warn 非 blocking（本书 primer 章一贯数值推演文风，同 ch24 先例）；1 条 reader-comprehension 维度符号表缺 n_h^I 条目建议。run-ledger：impl_test_rounds=1、write_review_rounds=2、blind_rounds=1（0 failures）、map_rounds=1（pass）、无升级。bible.py due ch37 为空（无应埋/应回收伏笔）。Book Bible 登记 7 条精简参考实现接口签名（index_score/topk_select、IndexCache、csa_index_score+core_attention_sparse、dense_warmup_loss+sparse_training_loss、MXFP4 量化三件套、TopkIndicesBuffer 接线）。

## Why it matters

补齐 DeepSeek 稀疏注意力谱系的最后一块理论拼图：ch21 稀疏注意力后端/ch25 注意力后端/ch26 量化数学之后、DeepSeek-V4 整读之前，把 lightning indexer 打分数学、独立缓存工程与量化变体钉死成可引用的理论基座（终位插入 ch27 之前，行文衔接按内容措辞不写死章号）。避免后续任何涉及 DSA/IndexCache/MXFP4 索引器量化的章节重复推导。

## What to remember

ch37-primer-lightning-indexer：DSA 打分公式(Eq.1-4)+复杂度账+IndexCache 独立缓存工程(Eq.9-19)+MXFP4 量化变体+CustomOp 接线四层，39 测试全过，reviewer APPROVED 6 条 non-blocking issue（含 1 条 paper_ref 机械误报+1 条逐机制勾选表+1 条源锚证据链弱化+1 条图注后缀+1 条公式密度+1 条符号表缺项）。bible 登记 7 接口，无伏笔登记。
