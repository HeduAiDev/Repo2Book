# ch32 五维补审 REVISE — 处理记录

来源：`reviews/pending-issues.json`（Lead 手动五维补审 2026-07-05）。态度：receiving-code-review，逐条核实后处理。

## 5 条 blocking（全部已修）

### B1 · formula-structure · 4 处半角标点
- **改法**：改标点，不改语义。
- **位置**：
  - 「…注意力配置**；**它是 GQA…」（承接 MLA 记号那段，原 L15）
  - 「…取自落地维度配置**；**它取决于…」（§六 表注算量口径，原 L330）
  - 「…核心参数**；**其余是配置旗标或旁支**：**`sparse_mode=3`…」（§七 算子旁支说明，原 L396，两处一并改）
- **验证**：`lint_punct --all` ch32 零命中。

### B2 · algorithm-pedagogy · attn-quadratic-tax 缺源码内嵌层
- **核实**：原文 §一「落到真实规模」仅一句提及 forward 主链，无代码块。属实。
- **改法**：在该小节内嵌 `AscendDSAImpl.forward` 主链 2 条分派（`# vllm_ascend/attention/dsa_v1.py:L1619`，prefill/decode 两路），先 Read 源文件 L1574-L1649 核实后摘录（args 重排、内容忠实，无杜撰）。点明 `_forward_prefill`/`_forward_decode` 内那条对全部前驱算稠密 $q\cdot k$ 的注意力正是要被替换的对象，并前指 §七。

### B3 · dsa-training-coadapt · 训练锚点文不对题
- **核实**：dossier mechanisms[5] 锚点 `dsa_v1.py:L2735` 是推理期 `weights_proj` 缩放，与本机制训练期 KL 对齐文不对题。属实（Read L2100-2144 佐证 L2735 一类为推理侧 weights）。
- **改法（正文）**：§五 KL 段后新增一段，明确「两阶段续训代码不在本推理仓库、本机制是纯论文推导」，并把 §三 `weights_proj`、§七 `npu_quant_lightning_indexer` 的 `weights` 挑明为「训练调好的 indexer 权重在推理侧的落点」，作对照。
- **改法（dossier，仅此一处）**：mechanisms[5] `source_anchors` 加注「推理侧对照…非训练代码」，并新增 `anchor_note`：「纯论文推导 + 推理侧对照」。

### B4 · dsa-cost-model · 成本模型未回指算子
- **核实**：§六 通篇数学，无代码回指。属实。
- **改法**：§六「机制」节末补一段，把两笔 MAC 账各自落到 §七 的算子——indexer 的 $O(L^2)$ 对应 `npu_quant_lightning_indexer`，主注意力被砍到 $O(L\cdot k)$ 对应把 top-k 喂进 `cmp_sparse_indices` 后的稀疏注意力算子。（采用「回指」而非再内嵌，避免与 §七 重复贴同一算子。）

## 非阻断 4 条

| # | 维度 | 处理 | 说明 |
|---|---|---|---|
| NB1 | algorithm-pedagogy · nsa-three-branch §七 未回指§二 | 采纳 | §七 `cmp_sparse_indices` 段补一句「这正是第二节 $N_t\ll t$ 框架简化后的落地，三支路收敛成单一 $k$」 |
| NB2 | paper-fidelity · Eq 标注 | 采纳 | §一 标注改「Eq.(1)-(2)」并注明下式合并了 Eq.(1) 输出与 Eq.(2) softmax 展开；已比对 paper.md L46/L52 确认 |
| NB3 | figure-integration · 图注 0.03 | 采纳 | 图 32-2 图注 0.03→0.031（与正文/图内 0.0312 取整一致） |
| NB4 | formula-structure · 内联密度高 | 部分采纳 | §六「机制」把两笔单-query 成本 $O(L\cdot d_{\mathrm{idx}})$/$O(k\cdot d)$ 提为 $$ 块，缓解最密处；L202/L242 未动——那两处内联为记号装配/校验值列举，拆块反而割裂句意（风格自主权范围内保留） |

## reader 顾问 6 条（半行垫脚石，全部采纳）

| # | 顾问点 | 落点 |
|---|---|---|
| R1 | l/d/l′ 几何关系 | §二「免费打分」补几何直觉：分块对齐时同段 token 在两路落同一块位，分数一一重合可搬用 |
| R2 | p^(h) 从哪来 | §二 p′ 定义处补注：每个 $\mathbf{p}_t^{\mathrm{slc},(h)}$ 是压缩支路头 h 的 softmax 注意力权重 |
| R3 | k=L「最大差 0.0」像两路相同 | §四 补一句：k=L 是验证正确性的退化基例，真稀疏在 k<L，勿误读 |
| R4 | α=0 召回 0.386 看着低 | §五 补一句：0.386 是 k≪L 约束下的召回上界，余量由 sparse stage 全参微调吸收 |
| R5 | 8.7x 无口径/未提带宽 | §六「直觉」补一句：MAC 为算力视角，真实延迟另受内存带宽影响（论文口径） |
| R6 | top-k 选 latent KV 还是 token / MQA 对应 | §四 Eq.(2) 后补注：MLA-MQA 下选出的是一组跨头一致的 latent KV 条目，token 粒度选、全头统一生效 |

## 反驳清单

- **NB4 的 L202/L242 两处未提块**：非拒绝，属「部分采纳」。已在最密的 §六 机制提一块；L202（indexer 参数装配一一对应）与 L242（k=8/k=3 校验值列举）的内联是逐项点名式记号，改成 $$ 块会打断「参数↔符号」的对读节奏。写作自主权范围内保留，`lint_formulas` 该两段仅非阻断 warning。

## 收工自检（均无 BLOCKING）

- `lint_chapter_structure`：✓ 结构检查通过（Roadmap + 自包含源码 + 零脚手架泄漏）
- `lint_formulas`：🟢 No blocking issues（Total 32 非阻断密度 warning）
- `lint_source_grounding`：exit=0，无 BLOCKING（`vllm_files_listed` 为非阻断项，本 primer 真实只读 dsa_v1.py/sfa_v1.py 两文件）
- `lint_trace_consistency`：✓ 正文数值推演表与素材一致
- `lint_paper_grounding --expect-primer`：✓ 无 BLOCKING
- `lint_punct --all`：✓ 无半角标点问题（ch32 零命中）
