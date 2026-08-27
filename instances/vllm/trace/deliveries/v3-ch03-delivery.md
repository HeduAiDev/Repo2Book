# v3 ch03《从 EngineArgs 到 VllmConfig》交付归档（APPROVED）

- **Type**: delivery（v3 Archive 站，chapter-pipeline-v3 第三章、Part I 全景与读法收官）
- **Chapter**: v3 ch03 · Part I · kind=code（L0 缩放：L0 启动视角）
- **Pin**: vLLM v0.27.1（6e448d0ea）；行号基线即此版
- **Date**: 2026-08-27 · **Agents**: pipeline 各站（analyst→researcher→implementer→tester→explainer→illustrator→writer→reviewer）+ archivist（本记录）
- **Verdict**: APPROVED，8 条 issue（0 blocking + 8 negotiable/non-blocking，后 4 条 reader-comprehension 维），全文见 `artifacts-v3/ch03-engineargs-to-vllmconfig/reviews/review-report.json`

## What happened

- **回环**（`reviews/run-ledger.json`）：impl↔test 1 轮（32 passed host，`implementation/config_wiring.py` 单模块无 torch/vllm/CUDA 依赖）；write↔review 3 轮；L2/图 1 轮；盲审 1 轮零失败（3 图全 PASS——L2-ch3 经 2026-08-27 REVISE 轮 spec 链式速记两处措辞修正重渲后内容级结论承继，两机制图独立盲审 PASS）。
- **bible 登记（v3 侧车）**：glossary-v3 +20 条新术语（两级映射/VllmConfig/EngineArgs/dataclass·__post_init__/三态开关/usage_context/TP·PP·DP/集合通信/world_size/执行后端/优化级/torch.compile/enforce_eager/谓词默认/配置指纹/工厂三连/safetensors/A100 反例 #17885/performance_mode/FlashInfer）+ 3 条首现章勘正（pickle、fork/spawn 原记 ch04、LoRA 原记 ch06——ch03 后补出序归档、却在书中先现，勘正后首现章如实=ch03，LoRA 深讲仍归 ch06）；concepts-v3 +14（两级映射/唯一上下文容器/__post_init__ 槽位/单一真相源/三态纪律/批默认查表/TP·PP·DP/uni-mp-ray/O 级两种保证/谓词默认/指纹因子表/工厂只选类不实例化/三工厂不同进程/两使用面同一装配线）；interfaces-v3 +24 条（EngineArgs 全家→五大子 Config→VllmConfig.__post_init__ 主线/compute_hash/预设应用→三工厂→EngineCore.__init__ 装配主干→两使用面入口，1:1 Source Map 同步）；figures.json 追加 3 张（L2-ch3(l2-engineargs-to-vllmconfig) + ch03-fig-async-tri-state + ch03-fig-optimization-levels，book:v3）。
- **伏笔对账**：本章应埋无、应收无（pedagogy-plan F1-F10 的 planted/paid 集合均不含 3，与 run-ledger foreshadow_due:[] 一致）——foreshadow-v3.json 零改动；正文 Part IV/V/VI/VII/VIII 门牌与 ch9/ch12/ch14/ch19 前向指针均为信息性指路、非登记伏笔。注：figure-manifest 早期 selfcheck 曾有「底部伏笔行 埋 F17/F18/F19」描述，2026-08-21 档案回修后 pedagogy-plan 已删 ch3 伏笔条目（manifest 同步勘误）。
- **图登记门禁**：`python scripts/lint_figures_registered.py <章目录>` 显式传参 + REPO2BOOK_INSTANCE=vllm exit 0（3/3 登记；active_instance=triton-ascend，无环境变量时 bible 路径解析到错实例——ch01 已记同坑）。

## Why it matters

Part I 收官章，兑现 ch1 开篇承诺「每个旋钮拨下去系统哪里变，启动视角展开」：两级映射把 227 个扁平旋钮 → 一份 VllmConfig（dataclass `__post_init__`＝构造即校验的固定槽位、跨子配置约束收口在聚合瞬间）→ 三个工厂按同一份配置选出执行器/客户端/调度器（只选类不实例化——类过线、实例化留给引擎进程）。全书此后每个「Part N 打开」门牌的出厂配置侧在此立账：批默认显存×场景查表（H100 16384/8192、A100 反例 #17885——ch10 预算三档地形的上半段已在本文现核）、async_scheduling 三态决策定出 v0.27.1 默认心跳（ch12 主场）、-O0..-O3 一个数字换一桌旋钮与「预设垫底只填 None、enforce_eager/env 无条件改写」的两种保证（Part V 编译捕获的地基）、compute_hash 10 位指纹（改哪个参数触发重编译的判定表）。出序归档说明：ch04-ch10 先于此章交付（ch03 曾长期「在产线」），bible 侧车因此含 3 条首现章勘正——后续 Part 审计以勘正后账本为准。interfaces-v3 累计 8 章 187 条。

## What to remember

1. **【writer 定点小修清单待用】** 8 条 negotiable 全部未在稿归档（APPROVED 不阻断合规）。最值得修的三条：① 站 8 节末指路句指错一节（chapter.md L438「怎么定的，下一节」——下一节实为站 9-10 聚合节，async 答案在再下一节站 11；一处定点改写即愈，8 条中唯一事实性小错）；② 「三个工厂的产物在引擎核的构造函数里碰头」（L11 图注 + L892）与所示 EngineCore.__init__ 代码不符——代码块里没有客户端的任何痕迹，节末 L965「①③在前端进程跑」自相澄清，半句收窄即可（「执行器与调度器两个产物在构造函数里落地；客户端在前端进程提前选出」）；③ 行话零假设四处（量化 L190/L478、FP4 L641、rank L293、NCCL L438）——各 ~20 字括注，量化与 rank/NCCL 在 Part VI/VIII 还会高频回来。其余：四步钳制 200+ 字长句编号化（L404，与章内「四步读①②③④」习惯一致）、max_concurrent_batches 表格先用 15 行后才立义（L594-601 vs L612，一词 gloss）、Part V 装载/编译两副面孔未打通（L161 vs L251/L782/L1076，半句绑定或更正 Part 号）、docstring 裸「…」省略标记（L714，全章唯一非「# … 省略：」式）。
2. **【出序归档的首现章勘正模式】** ch03 在 ch04-ch10 之后归档造成 3 条术语首现章回拨（pickle/fork·spawn←ch04、LoRA←ch06）。ch11 起恢复顺序交付后不应再出现；但若未来再出序（如 retrofit 重写），archivist 归档时应对「本章先现、他章已登」的词做同款勘正（首现章如实改、释义尾附勘正注、深讲章不动）。
3. **【explainer 数字与正文表已三方锁定】** 批默认七场景/async 三态五场景/O 级五场景/指纹六场景四张数值表均出自本章精简版运行轨迹（trace 注释锚在正文表上）；指纹表「绝对值不与真实 vLLM 逐位对齐、只消费作用域语义」的 caveat 在稿（L1046）——后续章引用 ch03 指纹数字时勿当真机值引用。
4. **【L2-ch3 的 REVISE 轮字节史】** figure-manifest selfcheck ⑩ 记录了 2026-08-27 spec 两处「用户显式>env>预设」链式速记勘正（真实源码是 enforce_eager/env 无条件写、压过包括用户显式值在内的一切）后重渲——PNG 字节有变、内容级盲审结论承继。若流程要求字节级盲审，可对当前字节独立复验（manifest blind_review.notes 已如实披露）。
