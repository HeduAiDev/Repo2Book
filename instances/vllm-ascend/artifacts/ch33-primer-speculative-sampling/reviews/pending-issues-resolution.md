# ch33 pending-issues 处理记录（receiving-code-review r3）

处理日期：2026-07-04
处理人：writer
方法：先 Read 源码 `vllm_ascend/sample/rejection_sampler.py`（L289-343 `rejection_sample`、
L770-800 `sample_recovered_tokens`、L1225-1260 残差重采样）逐条核实再落笔，逐条采纳或带技术理由反驳。

> 说明：本章目录为未跟踪状态（无 git 基线）；进入本轮时 chapter.md 已含两条 blocking 的
> 修复。本记录逐条把当前正文与 issue 原文/suggested_fix + **源码事实** 对齐核验，确认修复忠实、
> 无残留，并跑五个 linter 全部无 BLOCKING。

## Blocking 两条

### #1 q 符号撞名 + 机制讲错（原报 chapter.md:428 附近）— 采纳，已在场且经源码核实
- **源码核实**：`sample_recovered_tokens()` L787-792 `q = torch.empty(...); q.exponential_()`
  确为每次调用现采的 i.i.d. Exponential(1) 噪声；L1243 `q_values = q[token_to_batch]`；
  L1249 `prob_over_q = prob / q_values_safe`；L1259 `argmax(prob_over_q)`。分母那个 `q`
  与残差 `torch.maximum(target_probs - draft_probs, 0)`（L1238）里的草稿概率毫无语义关系。
- **正文落点**：narrative/chapter.md L459（代码注释点明「q：调用方现采的 i.i.d. Exp(1) 噪声，
  不是草稿概率 q(x)」）+ L468-474。正文已：
  (a) 明确 `q_values`(L1243) 来自 `sample_recovered_tokens` 的 `q.exponential_()`，改记号为 ε；
  (b) 机制改为「残差权重 max(0,p−q) 除以独立 Exp(1) 噪声再 argmax = 指数竞速/Gumbel-max 除法变体」，
      配块公式 `argmax_i w_i/ε_i ~ Categorical(w)`（L470-472）；
  (c) 删去旧「祖先采样/逐维递归」类比、删去旧「对数概率加 Gumbel(0,1)」加性变体话术。
- 结论：与 suggested_fix 逐点一致，符号一致、推导忠实。**采纳。**

### #3 两个 core 机制缺源码层（expected-accepted-length / walltime-speedup）— 采纳，已在场且经源码核实
- **源码核实**：`rejection_sample(...)` L289-329 函数签名含 `max_spec_len: int`（L294），
  docstring `max_spec_len: Maximum speculative length`（L316）、返回 `[batch_size, max_spec_len + 1]`（L328）。
- **正文落点**：L312-333 内嵌 `rejection_sample` 签名/docstring 真实片段（无关入参以
  `# … 省略 …` 裁剪），L333 一段解读点明 `max_spec_len` 即正文通篇的 γ，两条 core 机制
  （饱和曲线横轴 / 先升后降曲线自变量）共享这一个代码块；并额外解读返回宽度 `+1` 兜底位。
- 结论：与 suggested_fix 一致，两条 core 机制第三层（内嵌真源码 + 逐段解读）到位。**采纳。**

## 非阻断条目（逐条）

- **#2 lint_paper_grounding 多论文盲点** — 反驳（超出 writer 定点改章范围）。这是 factory 级
  linter 小修（读 dossier.sources 的 supplementary_paper.pack），不属正文缺陷；reviewer 已
  确认引用忠实。本轮 `lint_paper_grounding --expect-primer` 输出「✓ 无 BLOCKING」，仅一条
  L470 公式锚点 ±10 行提示（非阻断）。留作 factory 跟进，正文不改。
- **#4 mtp-as-speculative-proposer 第二锚点(DeepSeekV4MTP.forward)未内嵌** — 反驳（reviewer
  标 optional/锦上添花）。该 core 机制三层已由 L166 的 `DeepSeekMultiTokenPredictor.forward`
  代码块满足；顶层 proposer 于 L428 prose 带过即可，避免堆叠冗余代码块。**不改。**
- **#5 两处 invariant 缺显式小标题** — 采纳（已在场）。L34「**不变量（下界保证）**」、
  L361「**不变量（窗口随深度收缩）**」均已用与全章一致的加粗小标题标出。
- **#6 42 段 inline 公式过密** — 部分采纳/部分保留。linter 判 🟢 No blocking；本章逐符号讲
  MTP 递推是数学+代码对照的自然密度。关键式已提升为 $$ 块（如 L470 竞速式、Eq.21-23、加速比式）；
  L468/L474 两条 inline ε/p' 记号引用为支撑性符号，保留 inline 更贴读感。非阻断，不逐条清空。
- **#7 Algorithm 1 未以伪代码展示** — 采纳（已在场）。L46-58 已整段摆出 Algorithm 1 伪代码
  （草稿连猜/并行验证/逐个判定/残差重采/bonus），后三小节逐行拆解。
- **#8 成本系数 c=0.05 无出处** — 采纳（已在场）。L306 已加「c 取决于硬件与草稿相对主模型大小…
  实践中常落在 c∈[0.02,0.08]」的说明。
- **#9 β 先用后定义** — 采纳（已在场）。L76 在首次出现 β 的表格前给了预览定义「β=Σmin(p,q)，
  单点接受率，本小节末正式定义并证明 β∈[0,1]」。
- **#10 内存带宽动机未机制化** — 采纳（已在场）。L24 已解释「一次前向要把几百亿参数从显存搬进
  计算单元，延迟被搬运时间主导…把 γ 个草稿摊进同一趟参数搬运」。
- **#11 Lemma3.3/Thm3.5 归一化=1−β 缺证明梗概** — 采纳（已在场）。L142-148「**不变量（残差
  质量=1−β）**」给出 Σmax(0,p−q)=Σ(p−min)=1−Σmin=1−β 的一行推导。
- **#12 hc_mult/hc_head 未定义** — 采纳（已在场）。L409 已解释二者为 V4「头合并」旋钮
  （hc_mult reshape 路数、hc_head 在 compute_logits 汇合多路送共享 OutHead）。
- **#13 自回归解码未正式定义** — 采纳（已在场）。L22「第 t 个字必须等前 t−1 个字落纸…解码 K 个
  token 就是 K 次串行前向」已给出定义。
- **#14 Gumbel-max 当常识未解释** — 反驳（suggested_fix 本身有误）。该条 suggested_fix 复用了
  「对数概率加 Gumbel(0,1) 噪声 + 祖先采样」的加性话术——正是 #1 判定为错、并已删除的表述。
  正文 L468-472 已改用与源码一致的**除法变体（指数竞速）**正确解释该技巧及其省归一化的目的。
  按 #1 的修正采纳、按 #14 的错误措辞**反驳**。
- **#15 proposer / proposer factory 未定义** — 采纳（已在场）。L428 已内联定义
  「proposer=生成待验证候选 token 的模块，MTP 是其一；proposer 工厂按模型类型选 EAGLE/MTP
  并调度验证」，兼顾自包含与对第 29 章的引用。

## Linter 复核（均无 BLOCKING，退出码 0）

- lint_chapter_structure：✓ 结构检查通过（Roadmap + 自包含源码 + 零脚手架泄漏）
- lint_formulas：🟢 No blocking issues（L468/L474 两条 inline 提示为非阻断）
- lint_source_grounding：exit 0；仅 `vllm_files_listed` 建议项（本 primer 章正当地只读
  rejection_sampler.py 与 deepseek_v4_mtp.py 两文件，路径规范，非阻断——不臆造第三个文件）
- lint_trace_consistency：✓ 正文数值推演表与 explainer 素材一致
- lint_paper_grounding --expect-primer：✓ 无 BLOCKING（L470 公式锚点 ±10 行提示为非阻断）
