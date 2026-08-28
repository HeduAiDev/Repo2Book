# v3 ch14《显存账本》交付归档（APPROVED，评审补完后归档）

- **Type**: delivery（v3 Archive 站，chapter-pipeline-v3 第十四章、Part IV「显存是主角：分页 KV」第二块拼图——**评审补完 + 遗留 negotiable 修补落定后由 archivist 归档**；ch14 全目录（含 narrative/dossier 本轮修订）尚未提交，归档落盘后由 Lead 统一定稿提交）
- **Chapter**: v3 ch14 · Part IV · kind=code（L0 缩放：KV 账本列上半+启动装配带——池多大·谁定的·门多紧：测量式分配三步定账 / 混合注意力组化 / 准入门与水位）
- **Pin**: vLLM v0.27.1（6e448d0ea）；行号基线即此版
- **Date**: 评审完成 2026-08-29T00:21+08:00 · 归档 2026-08-29 · **Agents**: pipeline 各站（analyst→researcher→implementer→tester→explainer→illustrator→writer→reviewer）+ ch14-review-completer（六维评审补完）+ archivist（本记录）
- **Verdict**: APPROVED（round 2 六维全 PASS、0 blocking、6 条全 negotiable——1 条评审时已解决 + 5 条遗留由 writer 后续修毕、正文稳定），全文见 `artifacts-v3/ch14-memory-ledger/reviews/review-report.json`

## What happened

- **回环**（`reviews/run-ledger.json`）：round 1 = 流水线原跑（2026-08-28，仅 algorithm-pedagogy 维返回即中止，writer 当日 19:59 按 issue 修订 m14 协商口径）；round 2 = 评审完成者补跑其余五维（2026-08-29，对修订后现稿全新评审）全维 PASS → APPROVED。impl↔test 按 test-report 终态记 1 轮：host 81 passed + **容器差分电池 47 CHECK 场景（钉版真源码 vs 精简版 stdout 逐行 diff 零差异）**，lint_fidelity 0 BLOCKING（must_keep 74 项全数核在）；L2 1 轮；盲审 1 轮零失败（9 图 2026-08-28 独立盲审全 PASS，manifest 可查）。foreshadow_due=[]、escalated=null。
- **遗留五条 negotiable 的修补核实（归档时 grep 复验，全部在稿）**：TRT-LLM 身份交代（L799「NVIDIA 的 TensorRT-LLM 推理引擎」）、KVConnector 首现就地立义（L747「跨机搬运 KV cache 的接入组件…Part IV 末章正面拆」，不再隔 80 行）、L355 三层括号嵌套长句拆解、三处无标记删行（lint_fidelity 按设计盲区类，与 ch18 L388 同族）、wrapper 裸用（CUDA 图包装对象就地交代）。
- **bible 登记（v3 侧车，2026-08-29 归档回写）**：glossary-v3 +20 条 + 1 条 ch10 追记（ch14 正主 12：三本显存账/测量式分配/CUDA 图显存估计/护栏四道/页统一/张量共享布局/两把尺/窗外回收/回收感知准入上限/max_in_flight_tokens/map_to_kernel_blocks/KVCacheConfig 一份账喂两侧；首现章如实记先现章 8：gpu_memory_utilization→ch03、KVCacheSpec/KVCacheConfig/混合注意力组化/KV cache group→ch13、SWA→ch11、KVConnector→ch09、TRT-LLM→ch10、Mamba·SSM→ch10，各注明正主 ch14——同 LoRA/fork 勘正先例；追记：ch10 整序列准入门条目补第一幕史 #37307 与封顶论证）。**抢注核查（Lead 指示）**：grep 确认 ch13 归档未抢注任何 ch14 域概念（hash_block_size/KV cache group/水位预算/组化均无 ch13 首现条目）；ch15/ch18 先前留账亦如约未抢注。concepts-v3 +16（对齐 pedagogy-plan introduces 三项 profile 三步定账/准入门/混合注意力组化+拆细：三本账/CUDA 图估计/护栏四道/一份账喂两侧/页统一/张量共享/两把尺/两幕史与封顶论证/窗外回收 null 占位/单源铁律/水位抖动环/kernel 细分多组表/并发核算/util 语义）；interfaces-v3 ch14 +33（定账管线 request_memory→memory_profiling→determine_available_memory→get_kv_cache_configs 全链 + 组化/页统一/布局/两把尺 + spec 家族四件 + 准入上限两法 + manager 夹取/回收三件 + coordinator/manager 门段/scheduler/engine 装配序 + block_table 细分/后端协商，出自 impl-notes 1:1 Source Map）；figures.json 追加 9 张（L2-ch14(l2-memory-ledger) + boot-three-steps(m1)/binary-search-len(m4)/hybrid-groups(m5)/tensor-sharing(m7)/one-ledger-two-sides(m9)/swa-cap-plateau(m11)/watermark-gate(m12)/swa-null-swap(m13)，book:v3；needs_figure 3/3、m2/m3/m6/m8/m10/m14/m15/m16 无独立图由正文表承载）。
- **伏笔对账**：本章不埋不收（pedagogy-plan foreshadows 无 planted=14/paid 含 14，与 run-ledger foreshadow_due=[] 一致），foreshadow-v3.json 零改动。ch13 三次点名「→ ch14 显存账本」（池多大/水位/null 占位语义/kernel 细分/多组块表/packed 别名）本章全部接住；F8（KVConnector，planted=16）本章只在 reserved_blocks/异步预约处留「→ ch16」一句、不算埋点；F7（ch13 埋→ch22 收）路过 map_to_kernel_blocks 顺笔对齐块号换算、未展开 kernel 内景无踩线。
- **图登记门禁**：`REPO2BOOK_INSTANCE=vllm python scripts/lint_figures_registered.py <章目录>` 显式传参 exit 0；manifest 9 图与 bible v3-ch14 条目逐 id 集合相等。

## Why it matters

Part IV 第二块拼图落位：ch10-12 三章当参数用的 `num_gpu_blocks` 有了完整出生证明，L0「调度 · 显存账本」列 KV 半区上半点亮——与 ch13（下半块池）合起来该列从上到下全通。本章给全书立了两条可迁移的工程句型：**「门与账本若用两套公式，向松漂是抢占循环、向紧漂是队头死锁」**（#37307/#39734 两幕史 + Single source of truth 单源铁律——结构上让漂移不可能，比校验便宜也可靠）与**「测量式分配是一次性契约」**（profile 是快照不是保证，防御全在启动侧）。ch14 的评审补完也验证了「流水线中止 → 补跑评审 → 修补落定 → 归档」这条恢复路径走得通（round 2 对修订后现稿全新评审，非沿用旧结论）。

## What to remember

1. **【本次提交范围】** ch14 全目录（含 narrative/dossier/reviews 本轮修订）从未入库——本 delivery 与 bible/state 回写一并落盘后由 Lead 统一定稿提交；提交前 git status 应见 ch14 目录整体新增。
2. **【修订要点】** m14 协商口径是 round 1 唯一 issue 且评审时已解决：kernel 块大小是**逐 KV 组**协商（该组账本块大小 × 该组后端支持集的最大公因子块），不是全引擎一把尺——后续章（ch21 后端协商）引用时按此口径。
3. **【impl 口径】** host 无 CUDA：显存快照读数与 cudagraph 估计为注入示教值（算术路径与源码逐字一致、涉设备读数的表已标明）；容器差分电池 47 场景零差异是忠实性的主证据；enable_prefix_caching=False 支（定账与门不依赖它）；dossier 两处行号勘误已在 impl-notes 记档（m3 第 2 锚→profile_cudagraph_memory L6645-L6811；_check_enough 双层结构）。
4. **【邻章衔接】** hash_block_size（两把尺）与 KV cache group（组化）本章立住后，ch15（粒度视图/mamba 对齐/不动点）与 ch16（connector/reserved_blocks）的留账概念已可引用；ch17 的显存三锚点条目（determine_available_memory 账本归 ch14）与本章互为正反两半，gap 审计两侧均不再悬空。
