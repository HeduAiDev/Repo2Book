# v3 ch29《Sampler 9 步管线》交付归档（APPROVED）

- **Type**: delivery（v3 Archive 站，chapter-pipeline-v3 第二十九章、Part VII「选一个 token 出门」首章入账）
- **Chapter**: v3 ch29 · Part VII · kind=code（subtract-only 精简版，`# SOURCE:`/`# SUBTRACTED:` 全标注）
- **Pin**: vLLM v0.27.1（6e448d0ea）· L0 缩放：采样出口列
- **Date**: 2026-09-06 · **Agents**: pipeline 各站（analyst→researcher→implementer→tester→explainer→illustrator→writer→reviewer）+ archivist（本记录）
- **Verdict**: APPROVED，16 条 issue（3 blocking：lint_explainer 素材侧陈旧行号 L159→L166 / bad_words 节「实测」括注无 trace 可溯源 / 契约 §8 破折号 62>60 且两段一段两处；13 non-blocking——fidelity×2（fp64 动机判等定性、logprob_token_ids_tensors 未引入）+ algorithm-pedagogy×3 + figure-integration×3（含 figures.json 漏登六图——归档站本记录收口）+ formula-structure×1 + cognitive-ladder×3 + reader-comprehension×4），全文见 `artifacts-v3/ch29-sampler-pipeline/reviews/review-report.json`

## What happened

- **回环**（`reviews/run-ledger.json`）：impl↔test 1 轮（impl_test_ledger 空数组；host 82 passed + 1 容器专属 skip，GPU 容器 83 passed 全绿——flashinfer 三 API 分支真跑）；write↔review 3 轮；L2 1 轮；盲审 1 轮零失败（blind_failures=[{round:1,failures:[]}]，6 图全 PASS）。foreshadow_due=[]、escalated=null。
- **归档时抽查（issue 兑现状态）**：逐条 grep 现稿，**3 条 blocking 修复全部在稿**——explainer.json L316「L159 条件」→「L166」、L324 invariant→「L166-L173」与 trace 对齐（本记录复跑 `lint_explainer` exit 0）；bad_words 节删「本章取证容器 torch 2.11 实测」括注、换有据表述「源码注释『Assign to slice to avoid cpu->gpu sync』针对的正是这个差别」，且与 reader-comprehension 写路径 why 项同点双收（「退化成 CPU 侧立即执行的小搬运，不是一条能排进队列的张量操作；slice 赋值…能像普通算子一样入队、后台异步执行」）；破折号 62→50（≤60 且无一段两处，grep 复核）。**其余 negotiable 定点修复亦几乎全在稿**：fp64 接受判定改「不等式 uniform 数 ≤ 目标概率÷草稿概率，逐 token 在边界上比大小…连零概率目标都翻成接受」；第 9 步节选以 `logprob_token_ids_tensors = None   # 稀疏旁路的产出，默认 None  # L113` 引入；「互相不贴对方」删（统计等价证据换「随机源不同、逐次不可比对」）；L1 图注改逐字一致「此处之后 tokens 离开 GPU 出 EngineCore（D2H 归第 8 章）」；你在这里列表补「ch32 投机解码数学+DSpark（原理章）」；L1214 复杂度记法统一 `` $`O(V)`$ ``/`` $`O(V\log V)`$ `` 数学仓；「九拍」→「九步」（grep=0）；min_p 写入顺序句补前向指针「机制在第 7 步 c 展开」；`grammar_output` 进可见签名（L32 第二行）+ L55 就地绑回；「出口的账」→「出口的 D2H 与判停」（L19/L1267）、「账面证据」→「源码证据」（L522）。**唯一未修 negotiable**：L686「间谍快照/哨兵替身」自造雅词句（cognitive-ladder，指代无歧义、与 ch20 同 precedent 留清单不强制）。
- **bible 登记（v3 侧车）**：glossary-v3 +11（argmax 不变性二分/Gumbel 技巧·指数噪声变体/拒绝采样/top-k·top-p·min_p 截断三代/惩罚三件套/Sampler·9 步管线编排者/TopKTopPSampler·构造期后端绑定/Qrita·pivot 截断核/allowed_token_ids/bad_words/思考预算）；已立术语未重复登记（logprob·logprobs_mode·SamplingMetadata 归 ch08、min_tokens 归 ch07、FlashInfer 归 ch03、冻结快照语义随 SamplingMetadata）。concepts-v3 +6（argmax 不变性二分/Gumbel 技巧/multinomial 同步之死/9 步管线骨架/构造期后端绑定/截断不动 argmax）；**plan introduces 第三条「raw logprobs」不重复登记**——concepts-v3 已有 ch08 条目「raw logprobs 语义（…）」，正文站 3 亦自认「语义全景 ch8 已立」，ch29 立的是 9 步骨架第一步的引用位而非概念本体。interfaces-v3 +ch29 16 条（出自 impl-notes 1:1 Source Map：Sampler.forward/apply_logits_processors/sample、TopKTopPSampler.__init__/apply_top_k_top_p/_pytorch、random_sample+sample_with_exponential_noise、bad_words、penalties、layers.utils 惩罚真算式、LogitsProcessors、builtin 三件套、build_logitsprocs、SamplingMetadata 17 字段、topk_topp_triton 整文件 must_keep、outputs 载体面）。figures.json 追加 6 张（book:v3，claim 自 manifest 程序化拷贝）：**L1-partVII（Part VII 导览 v3 首登）** + L2-ch29（l2-sampler-nine-steps）+ 机制图 nine-steps-gating→m1/argmax-dichotomy→m2/topk-topp-sort→m9/backend-binding→m11。
- **伏笔对账**：本章应埋**无**、应收**无**（pedagogy-plan F1-F10 无一条 planted=29 或 paid=29，与 dossier foreshadow_due、run-ledger 三方一致）——foreshadow-v3.json 无 ch29 条目、零改动（F6 paid=ch30 done:false 等待收不变）。邻章衔接非账本（正文点名即可）：站 1 `apply_grammar_bitmask` 提前回收 F6 的一半窗口（「同一行 logits：先被掩码改写、再采样」ch9 窗口的采样侧入口，F6 正式回收仍在 ch30）；站 2 RejectionSampler 换轨与 `build_logitsprocs` 的「min_p/logit_bias 与 spec decode 互斥」警告前指 ch32/33；站 3 raw 留底回指 ch8；async 占位与 token 留 GPU 回指 ch12/ch18。
- **图登记门禁**：`REPO2BOOK_INSTANCE=vllm python scripts/lint_figures_registered.py <章目录>` 显式传参 **exit 0**（review 期六图漏登 FAIL → 本站补登后复跑收口）；manifest 6 图与 bible ch29 条目逐 id 集合相等（本记录内程序核对）。
- **其余门禁（本站复跑全绿）**：lint_fidelity / lint_chapter_structure / lint_formulas / lint_source_grounding / lint_dossier（m16 kind=algorithm 无 paper_origin 一条 WARN 属码章正常）/ lint_explainer / lint_trace_consistency / lint_figures_registered 全 exit 0。

## Why it matters

Part VII 首章入账、v3 全书第 23 章交付（ch01-20、ch22、ch27、ch29）。本章把「13 万个 logits 选 1 个 token」讲成一条 9 道关卡的门控管线：multinomial 因读回同步被逐出热路径（两台计算机/队列/排空的范式级样板）→ Gumbel 指数噪声 argmax 掷骰 → argmax 不变性二分决定处理器挂点（会改第一名的 greedy 前、只砍尾的温度后）→ 构造期后端绑定（运行期零平台分支）→ 截断不动 argmax（严格 < 保并列、最末位恒保）。9 步骨架/二分/统计等价的数值证据全部实跑（host 8 组纯数学 + GPU 容器 Triton/FlashInfer 两路）。ch30/31 的语法位掩码（F6 正式回收）、ch32 的拒绝采样数学（fp64 支路的边界动机已在此埋好语义入口）都从本章交接点出发。

## What to remember

1. **【L1-partVII 由本章首登】** Part VII 导览图首次登记在 ch29 名下（Part VII 第一个入账章）——ch30/31/32/33 归档时其 manifest 若也含 L1-partVII，须按各自 chapter_id 再登记一条（lint 按 chapter_id 索引、同 figure_id 不同章不冲突）；与 ch27→ch23-26 同款提醒。
2. **【素材侧行号漂移由 lint_explainer 兜底（正向样板）】** m13 worked_example 表格里 L159 陈旧行号正文已改 L166 而 explainer 未同步——lint_trace_consistency 只对账「正文↔素材」故绿灯，lint_explainer 的「表格数字须在 trace 里找得到」抓住了素材侧不符。机制：素材/正文/trace 三方行号以 trace 实跑锚为准，归档前 explainer 门禁不可跳。
3. **【契约 §8 破折号纪律首例执行】** 本章是 WRITING-CONTRACT-v3 §8（2026-09-05 用户裁定）后第一个过「超 60 必须减法」红线的章：62→50 定点小修（语气性补充改冒号/括注/拆句），不退整章——后续章 writer 契约注入时直接带数字红线即可复用此路径。
4. **【Part VII 进度】** ch29 已归档；ch30（语法编译）/ch31（bitmask 落地）/ch32（【primer】投机解码数学+DSpark）/ch33（vLLM 落地）未成稿。F6 grammar bitmask 窗口正式回收在 ch30（本章站 1 只开了采样侧入口）。
