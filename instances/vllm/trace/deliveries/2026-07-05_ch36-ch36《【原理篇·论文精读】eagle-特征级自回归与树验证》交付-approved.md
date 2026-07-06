# ch36《【原理篇·论文精读】EAGLE 特征级自回归与树验证》交付-approved

- **Type**: delivery
- **Chapter**: 36
- **Date**: 2026-07-05
- **Timestamp**: 2026-07-05T23:06:58Z
- **Agents involved**: archivist, writer, reviewer, illustrator
- **User present**: False
- **Tags**: primer, eagle, spec-decode, paper-grounding, ch28-link

## What happened

ch36-primer-eagle（原理章，EAGLE arXiv:2401.15077 + EAGLE-2 arXiv:2406.16858 论文精读）四段式：动机(投机加速取决于草稿分布 q 与目标 p 的贴合；token 层自回归不确定性来源) → 推导(特征层回归为何更准——不确定性分解；EAGLE-2 动态草稿树/树注意力掩码/树验证) → 数值推演(小树手算接受路径与期望接受长度，跑参考实现 traces) → 落地(vllm/v1/spec_decode 的 eagle proposer 调用面，回指 ch28 投机解码章；拒绝采样保分布定理该章已完整推导，本章直接引用不重复证明)。参考实现 4 个文件（feature_autoregression.py/chain_drafting.py/draft_tree.py/speculative_sampling.py）忠实复现论文算法（本章 kind=primer，豁免 subtract-only 硬规则，改用 lint_paper_grounding 门禁）。reviewer verdict=APPROVED，9 条 issue 全 non-blocking/negotiable：1 条数值溯源披露（EAGLE-2 校准曲线 5 分桶数字实为本书参考实现模拟，非论文 Fig.6 原始数据，建议补一句归属说明）；1 条 lint_paper_grounding --expect-primer 的 7 条 paper_ref 警告经人工核对均为字面缩写(Fig./Appendix)对字面全称(Figure/Algorithm)的 grep 失配，非事实错误；1 条图注(fig36-7)步数措辞（“随后四步”应为“随后三步”）与图不符；其余 6 条为 reader-comprehension 维度的可读性建议（loss 公式索引/token-feature 记号/层选择理由/噪声增强因果链/greedy 采样理由/EAGLE-1 vs EAGLE-3 范围澄清）。run-ledger：impl_test_rounds=1、write_review_rounds=1、blind_rounds=1（0 failures）、无升级。bible.py due ch36 为空（无应埋/应回收伏笔）。Book Bible 登记 4 条精简参考实现接口签名（ToyTargetLLM/AutoregressionHead+combined_loss、propose_chain、draft_tree 系列 API、speculative_sampling 系列 API）。

## Why it matters

巩固 primer 系列（ch34 flash-attention/ch35 quantization/ch36 EAGLE）论文精读方法论的第三个实例：素材先行(explainer.json 模拟轨迹) + lint_paper_grounding 引用锚门禁 + 拒绝复制论文图表数字而不标注来源。为后续任何涉及投机解码草稿质量/树验证/特征级 vs token 级不确定性的章节提供可链接的理论基座，避免重复证明拒绝采样保分布定理（该定理归属 ch28，本章仅引用）。

## What to remember

ch36-primer-eagle（原理章，EAGLE arXiv:2401.15077 + EAGLE-2 arXiv:2406.16858 论文精读）四段式：动机(投机加速取决于草稿分布 q 与目标 p 的贴合；token 层自回归不确定性来源) → 推导(特征层回归为何更准——不确定性分解；EAGLE-2 动态草稿树/树注意力掩码/树验证) → 数值推演(小树手算接受路径与期望接受长度，跑参...
