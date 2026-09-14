# v3 ch28《实战：DeepSeek-V4 拼装》交付归档（APPROVED）

- **Type**: delivery（v3 Archive 站，chapter-pipeline-v3 第二十八章、Part VI「模型的形状」capstone 实战章）
- **Chapter**: v3 ch28 · Part VI · kind=meta（L0 缩放：模型层全景；**无 subtract-only 精简版、无 impl/test 站**——impl_test_rounds=0，走读与手算全在 pin 树参考实现/树内对拍基准上，kind=meta 先例）
- **Pin**: vLLM v0.27.1（6e448d0ea）；mHC 数值出自树内对拍基准 tests/kernels/test_mhc_kernels.py、路由数值出自 XPU/CPU 纯 torch 回退实现（四点诚实边界在稿 L17）
- **Date**: 2026-09-15 · **Agents**: pipeline 各站（analyst→researcher→explainer→illustrator→writer→reviewer）+ archivist（本记录）
- **Verdict**: APPROVED，17 条 issue（3 blocking：algorithm-pedagogy m09「四个案例/分最低」与自家 trace 表直接矛盾的两处事实滑步、cognitive-ladder×2 图注方位词与图面不符（fp4-byte-layout 左右→实为上下堆叠、mhc-residual-journey「横向三轨」→实为上通路+三核塔+底挂三件套）；14 non-blocking：fidelity×2（disable_tp 括注引错 TP 子情形→已改列切+all-gather 正机制、三处裸文件名补全路径）/figure-integration×1（9 图欠登，本站办结）/formula-structure×1（L1302 片段头补 GPUModelRunner.propose_draft_token_ids）/cognitive-ladder×4（four-kv 热点注位置、前向指路错一站→幕四第 8 站、流形/谱范数就地括注、wo_a·wo_b why 链补代价句）/reader-comprehension×6（3·4·5 计数对齐、T 记号首现、fused 管线不展开标注、__all__ 断言软化、llama_4_scaling 占位括注、finalize 两级方法名映射）），全文见 `artifacts-v3/ch28-deepseek-v4-assembly/reviews/review-report.json`

## What happened

- **回环**（`reviews/run-ledger.json`，原样落盘）：impl_test_rounds=0（meta 章无 impl）、write_review_rounds=3、l2_rounds=3、盲审 2 轮（round 1 判 mhc-residual-journey 1 处④不可读——_mtp_hidden_buffer 框内橙色虚线箭头两端脱开+压字，修复后 round 2 零失败、8 张机制图全 PASS）；foreshadow_due=[]、escalated=null。
- **归档抽查（issue 兑现状态）**：逐条 grep 现稿，**17/17 修复全部在稿——零 writer 小修债**。三 blocking 逐字复核：L1215 已改「表中五行（A 两行、B 一行、C 两行）…恒等于 1.5（scaling 因子）」+「进不了分数 top-2 的 e0」；L681 图注已改「上段 H/2…下段 H/32」；L927 图注首句已按实际布局重写（上层 2D x 通路+三座核塔纵向贯通+占位子层正下挂三件套）。**素材源头同步核真**：explainer.json ch28-m09「四案例」措辞已改（「五行」×3 在稿、「分最低」0 命中）。L269 括注已改输出维列切+all-gather 正机制；gate_linear/moe_runner/tilelang_kernels 三处全路径在稿；L1302 片段头、L148 前向指路（幕四第 8 站展开）、L723 流形/谱范数括注、L717 T 记号、L1258 fused 不展开标注、L98 __all__ 软化、L933 llama_4_scaling 括注、L683 finalize 两级名映射全部在稿。
- **bible 登记（v3 侧车）**：glossary-v3 +11（mHC 多流残差/Sinkhorn 归一化/TileLang/DeepSeekMoE 切法/sqrtsoftplus 路由/noaux_tc·e_score_correction_bias/hash MoE tid2eid/MegaMoE deep_gemm_mega_moe/swiglu_limit/EAGLE-3/Quark）**+1 勘正**：MoE（mixture of experts）首现章 ch34→ch28——v3 非线性写作顺序（ch34 先归档）所致，ch28 才是全书按序正式立 MoE 之章（起源 Shazeer 2017+DeepSeekMoE 切法），释义已补「GShard 分布式骨架与 EP 全景归 ch34」，与 ch20 delivery「TMA/flash_attn_varlen_func 首现勘正」同款台账卫生。已立术语未重复登记：attention sink 归 ch21（ch28 只接讲逐头可学参数版）、EPLB 归 ch34（ch16/19/23 已先路过）、IndexCache/压缩机/V4 三类层归 ch26、UE8M0/MXFP4/NVFP4/四重门归 ch27、breakable cudagraph 归 ch19、SwiGLU/RMSNorm/RoPE/残差总线归 ch23。concepts-v3 +15（MoE 拼装/FP4 落地/code→diagram 三步法/mHC 多流残差/融合单核与首层核内展开/三代 MLA 装配/一层四本 KV 账/sqrtsoftplus 三分离/双后端二岔/MegaMoE 三守卫/FP4 专家词典与 regex 顺序/EPLB 多槽装载活口/多流 overlap 与 eager break/MTP 钩子三段接力/全干形状主线）。interfaces-v3 **+0**（kind=meta 无精简版，跳过）。figures.json 追加 9 张（L2-ch28（l2-deepseek-v4-assembly）+ mhc-residual-journey→m04/mla-gen3-projections→m05/four-kv-ledgers→m06/moe-dual-backend→m08/fp4-byte-layout→m10/multistream-timeline→m13/mtp-hook-fork→m14/trunk-shape-journey→m16，book:v3、claim 自 manifest 程序化拷贝）。
- **伏笔对账**：本章应埋**无**、应收**无**（pedagogy-plan F1-F10 的 planted/paid 集合均不含 28，与 run-ledger foreshadow_due=[] 一致）；逐组核对正文实际埋收——L1389 总结收口的六问全部回指本章开篇问题链（章内闭环非跨章伏笔）、ch19 可断 cudagraph/ch27 四重门/ch26 三类层均为回指先章非新埋；foreshadow-v3.json 无 ch28 条目、无需改动（台账仍为已清 F1/F2/F3/F4/F6/F7/F10 七组，未到期 F8→ch36/37、F5→ch38、F9→ch40）。
- **图登记门禁**：`python scripts/lint_figures_registered.py <ch28 章目录>`（active_instance=vllm 下显式传参）exit 0；manifest 9 图与 bible v3-ch28 条目逐 id 集合相等（本记录内程序核对）。
- **归档站门禁复跑**（显式传 v3 章路径，全 exit 0）：lint_chapter_structure / lint_formulas / lint_source_grounding / lint_dossier（m08·m09 无 paper_origin 两条 benign warn） / lint_explainer / lint_trace_consistency / lint_paper_grounding（非 primer 跳过） / lint_punct / lint_anchors（章内锚点全可解析，1 处裸文字章号+5 处节号前缀 warn 不计退出码）/ lint_figures_registered。

## Why it matters

Part VI capstone 入账：ch23 立的「拼层契约」在旗舰上期末考——五件套全换（注意力低秩两头/FFN 换 MoE/残差多流/输出头加 draft/数值全量化）而装配契约一件没变，全书「模型层」叙事线在此合龙（L0 图 GPU 执行臂「模型层 forward+编译」盒最后一块拼图）。本章把 mHC（Manifold-Constrained Hyper-Connections，arXiv:2512.24880）从 v2 的低置信推测升级为一手档案级讲述（双随机流形/Sinkhorn 投影/六组 fp32 参数与论文 §3.3 的 24 行互证、「论文数字在 config 里活着」hc_sinkhorn_iters=20 现场），并立全书最大的自造方法论概念「code→diagram 三步法」（registry 查表→__init__ 树画框→forward 走线）作为读者带走的能力。kind=meta 是 v3 第一个无精简版交付的 code 系章——走读+对拍基准手算（五行路由账/2×2 Sinkhorn 两轮收敛/字节解剖 3.7647 倍压缩）替代 impl/test 站，为同类 capstone 章立先例。盲审回环 1 修 1 过（mhc 图悬空箭头删除后零信息损失），验证「渲染→亲眼看→盲审」门禁在 8 图同批下仍抓得住单处压字。

## What to remember

1. **【MoE 首现勘正 ch34→ch28 已办】** v3 非线性写作顺序再添一笔：ch34 归档时 ch28 未成稿、MoE/EPLB/SP-MoE 等条目按当时唯一深讲章记了 ch34；ch28 按书序先立 MoE，本次已把 MoE 条目首现改回 ch28（释义注明 EP 归 ch34）。EPLB 因 ch16/19/23 已先路过、ch34 才深讲，维持 ch34 不动；attention sink 正主 ch21（ch28 只接讲 attn_sink=-inf 参数版，已在 concepts「三代 MLA 装配」条内）。
2. **【kind=meta 无精简版先例】** ch28 impl_test_rounds=0、interfaces-v3 +0——capstone/实战章可声明 kind=meta 免 impl/test 站（dossier 声明、run-ledger 如实记 0）；后续同类章（如 ch35 部署实战）照此办理，勿因缺 implementation/ 目录判缺陷。
3. **【ch25 待归档的两笔既有勘正仍然有效】** ch26 glossary 已把 CSA/HCA/topk_indices_buffer 字面首现如实记 ch25 名下（ch25 归档时勿重复登记）；ue8m0 合并条目首现 ch27 宜随 ch25 归档勘正（ch27 delivery 已记）。Part VI 现仅剩 ch25 待归档。
4. **【盲审回环素材同步纪律复验】** m09 计数词「四案例」在 explainer.json 与正文同源滑步——review 抓到后 writer 两边同步改（本记录复核 explainer「四案例」0 命中）；后续章 explainer invariant 措辞与正文表口径的对齐应在 Review 站即核，勿留到归档。
5. **【python3 坏桩复现】** 本机 python3 仍是 WindowsApps stub（exit 49 静默），全部脚本走 Miniconda python——各 delivery 已多次记录，续跑者直接用 python。
