# ch34《【原理篇·论文精读】FlashAttention：从 online-softmax 到 IO-aware 注意力》交付-approved

- **Type**: delivery
- **Chapter**: 34
- **Date**: 2026-07-05
- **Timestamp**: 2026-07-05T23:32:16Z
- **Agents involved**: archivist, writer, reviewer, illustrator
- **User present**: False
- **Tags**: primer, flash-attention, online-softmax, io-aware, lse-merge, cascade-attention, paper-grounding, ch24-link

## What happened

ch34-primer-flash-attention（原理章，FlashAttention arXiv:2205.14135 + online-softmax arXiv:1805.02867 + FlashAttention-2 arXiv:2307.08691 三篇论文精读）四段式：动机(N×N 物化的显存带宽墙——朴素注意力慢在 HBM 读写而非算力) → 推导(online-softmax 单遍 running (m,d) 递推与 ⊕ 结合律算子；FlashAttention 分块 tiling 免物化前向；§3.2 Theorem 2 的 IO 复杂度账 Θ(N²d+N²) vs Θ(N²d²/M)；FA-2 循环序对调/warp 分工改进一节带过) → 数值推演(N=4 小分块手算 (m,l,O) 递推，跑参考实现对照朴素/safe softmax 恒等) → 落地(vllm/v1/attention 后端对 flash_attn_varlen_func 的调用面与形参语义、cascade_attention 两段合并、merge_attn_states 的 Triton kernel，回指 ch24 注意力后端章)。附 LSE 合并小节(log-sum-exp 两段注意力合并——cascade attention 的数学地基)。参考实现 3 个文件（online_softmax.py/flash_attention.py/lse_merge.py）忠实复现论文算法（本章 kind=primer，豁免 subtract-only 硬规则，改用 lint_paper_grounding 门禁）。8 张图（内存墙/online-softmax 递推表/tiling 递推/IO 账/FA-1 vs FA-2/LSE 合并/varlen 调用面/cascade）全部盲审 PASS。reviewer verdict=APPROVED，12 条 issue 全 non-blocking/negotiable：1 条 lint_paper_grounding 字面子串匹配对 6 条 mechanism 的 paper_ref 误报(倒装标题失配，人工核对引用真实存在)；1 条 dossier lse-merge 的 paper_origin.sections 漏登第二篇论文出处(arXiv:2307.08691 §2.3)建议补全；2 条 worked example 量化粒度补数字(online-softmax 本例 N=4 具体读次数；lse-merge 合并后 lse 数值+成本对比)；1 条 fig34-8 图注收尾风格不统一(描述性而非结论性)；1 条 lint_formulas 17 处 inline 公式非阻断告警(均为单符号简单式，符合规则)；其余 3 条为 reader-comprehension 维度的可读性建议(Br/Bc 记号显式定义、block_table 切片语义注释、is_fa_version_supported 用途注释)。run-ledger：impl_test_rounds=1、write_review_rounds=3、blind_rounds=1（0 failures）、无升级。bible.py due ch34 为空（无应埋/应回收伏笔）。Book Bible 登记 3 条精简参考实现接口签名（online_softmax_merge ⊕ 算子、flash_attention_forward 分块前向、attention_with_lse+merge_lse_states LSE 合并）；glossary 补充 6 条 FlashAttention 系列译名（在线 softmax/IO 感知/分块/对数和指数合并/级联注意力/变长注意力调用面）；figures.json 登记 8 条 mechanism→figure 映射。

## Why it matters

全书此前把 flash_attn_varlen_func 当黑盒调用（ch24 注意力后端章仅点名未掀开算法内幕），本章是全书对读者最大的认知悬崖补课——完整推导 online-softmax 数值稳定性到 FlashAttention IO-aware tiling 再到 LSE 合并，为 cascade attention（vLLM 共享前缀优化）与后续任何涉及 split-KV/prefix caching 合并的机制提供可复用的数学地基，避免后续章节重复证明 ⊕ 算子的结合律。

## What to remember

ch34-primer-flash-attention（原理章，FlashAttention arXiv:2205.14135 + online-softmax arXiv:1805.02867 + FlashAttention-2 arXiv:2307.08691 论文精读）四段式：动机(HBM 带宽墙) → 推导(online-softmax 递推/⊕ 算子/tiling/IO 复杂度/FA-2 改进一节带过) → 数值推演(N=4 手算 + 参考实现验证) → 落地(flash_attn_varlen_func 调用面 + cascade_attention + merge_attn_states，回指 ch24)。reviewer APPROVED，12 条 issue 全 negotiable/non-blocking，无需回修。三个精简参考实现接口与 8 条图注册已回写 Book Bible；无伏笔待埋/待收。
