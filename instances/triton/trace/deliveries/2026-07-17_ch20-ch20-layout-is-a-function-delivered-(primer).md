# ch20-layout-is-a-function-delivered-(primer)

- **Type**: delivery
- **Chapter**: 20
- **Date**: 2026-07-17
- **Timestamp**: 2026-07-17T00:00:00Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch20, part-5, primer, layout-function, distributed, shared, blocked-triple, broadcast, wrap-around, f14-payoff, f17-plant, linear-layout-forward

## What happened

第二十章《布局即函数：GPU 张量凭什么和普通张量不同》交付（Part V「IR 与布局」第二站，primer 原理章；并行发车 skip_archive 模式，Review+Map 已 APPROVED/PASS，本次由 archivist 串行补归档）。全章只立一个定义、正面回收 ch19 悬念：encoding 属性的正式定义是一个函数 `𝓛: ℤ^d → 𝒫({0,...,n-1})`（`TritonGPUAttrDefs.td:L36-L38` 逐字），把张量索引映到「允许访问该格数据的线程集合」。教法主线：把张量画成格子、每格填线程号，抽象函数变成可逐格核对的座位表。展开顺序：①正式定义+顿悟例(`𝓛(0,0)={0,4}`)；②两大类分野——distributed(分散寄存器，四级层次算出)vs shared(共享内存全员可见，`{0,...,32·num_warps-1}`)；③distributed 的四级计算层次 CTA→Warp→Thread→Value(与硬件同构，不物化大表)；④Blocked 三元组(`sizePerThread`/`threadsPerWarp`/`warpsPerCTA`)把座位表压缩成三组小数字，16×16/2warp/64线程逐格核对；⑤broadcast(一格多号)与 wrap-around(一号多格)——同一条取模公式 `𝓛(T)[i_d]=L[(i_d+k_d·T.shape[d]) mod L.shape[d]]` 的两个对称分支；⑥模块契约(`num-warps` 强制/`threads-per-warp` 缺省 32/`num-ctas` 缺省 1)锁定线程总数 n；⑦前瞻框一句：𝓛 其实是 GF(2) 线性映射(LinearLayout，arXiv:2505.23819)，深化留 ch23。6 图(chapter-map+5 机制图：layout-as-function-table/distributed-vs-shared/four-level-hierarchy/blocked-triple-table/broadcast-wraparound)全 blind PASS(round1 zero failures)。write-review 3 轮、无 escalation。review APPROVED(issues 均 negotiable/non-blocking：m03 即时量化建议、m04 双射对账句风格补齐、fig-four-level-hierarchy 最内层 Value 文字与背景同色的渲染小瑕疵、broadcast/wrap-around 组合规则未显式陈述)。

## Why it matters

本章是全书布局系统(ch20-24)与后续性能 pass(ch25 起)的公共语言起点——「布局=函数」这一句定义此后被反复引用：看 TTGIR dump 判断合并访存/Tensor Core 命中，第一步就是读懂 `#triton_gpu.blocked<…>` 这类 encoding 属性，而它不过是这个函数的参数化写法。distributed/shared 分野为 ch21(Distributed 布局家族)/ch22(Shared 编码与 swizzle)立框架；模块契约(num-warps 强制)解释了「改 num_warps 为何要求编译器重新选布局参数」这一常见调优现象。

## What to remember

ch20 done（kind=primer，Part V 第二站，6 图全 PASS，write-review 3 轮/blind 1 轮/无 escalation）。**glossary.json 204→215**（新增 11 条：`布局函数 𝓛(layout as a function)`/`distributed 布局`/`shared 布局`/`四级计算层次`/`CTALayoutAttr`/`Blocked 三元组`/`broadcast(布局语境)`/`wrap-around`/`LinearLayout`/`threads-per-warp`/`num-ctas`；同时扩充已有 `num_warps` 词条补上模块契约强制性与 n=num_warps×threads_per_warp 公式）。**concepts.json 154→159**（新增 5 条→ch20：布局即函数正式定义、distributed vs shared 分野、四级计算层次生成 distributed 布局、broadcast/wrap-around 同一取模公式对称语义、模块契约锁定线程总数）。**interfaces.json 未改**——primer 章理论记号(𝓛/Blocked 三元组等)不入接口注册表（与 ch02/ch15 等既往 primer 章处理一致）。

**arc-map.json**：**f14 已回收**——`plant:ch19→payoff:ch20`，`status: open→resolved`，`resolved_in: "ch20"`（ch20 §1-2 正面回答「encoding 到底填什么」= 布局函数 𝓛）。**一致性核验**：全部 resolved 伏笔现为 f4/f5/f7/f11/f12/f13/**f14**，均 payoff==resolved_in 且 payoff≤已交付章节，无异常；f15(plant ch19→payoff ch24)、f16(plant ch17→payoff ch30)仍 open，未被误动。**新开正式伏笔 f17**（plant ch20→payoff ch23）：本章前瞻框「𝓛 是 GF(2) 线性映射(LinearLayout)，代数展开留 ch23」构成一条明确的 plant→payoff 钩子，已按 Lead 指示登记。distributed/shared 分野→ch21/ch22 的自然延续**未**额外登记为伏笔——dossier.json 的 `foreshadow_due.note` 已明确将其归为「primer 前瞻性铺垫（非 bible 登记，供 writer 收尾点名）」而非需回收的技术性悬念钩子，登记会造成 arc-map 对每章「下一章见」式过渡语句的过度登记（bloat），故按判断从简。figures.json **未更新**——沿用 ch19 先例（ch19 同样有 5 图但未登记 figures.json，该表当前仅覆盖 ch01/ch03/ch04/ch05/ch09，非强制逐章维护项）。

trace：本条 delivery 已建；`state.json` 已加 `ch20` 条目（figures=6，chapter-map+5 机制图）并刷新 `updated` 时间戳；`trace/INDEX.md` 已刷新（保留最近 10 条，自检确认 ch20 在列）。`reviews/review-report.json`、`reviews/run-ledger.json`、`narrative/chapter.md`、`diagrams/`、`dossier/dossier.json` 均未触碰（按指示，writer/illustrator 并行修订、Lead 已预写评审文件）。
