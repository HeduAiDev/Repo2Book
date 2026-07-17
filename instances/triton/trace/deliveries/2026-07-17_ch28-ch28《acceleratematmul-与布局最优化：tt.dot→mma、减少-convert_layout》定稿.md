# ch28《AccelerateMatmul 与布局最优化：tt.dot→MMA、减少 convert_layout》定稿

- **Type**: delivery
- **Chapter**: ch28
- **Date**: 2026-07-17
- **Timestamp**: 2026-07-17T11:01:31Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, archivist
- **User present**: False
- **Tags**: triton, part-vi, skip_impl, optimization-pass, AccelerateMatmul, RemoveLayoutConversions, OptimizeDotOperands, MMA, convert_layout, APPROVED

## What happened

Part VI 优化 pass 落地施工篇。三个挂在 make_ttgir 的 pass：AccelerateMatmul(BlockedToMMA 主 pattern——getMMAVersionSafe 按 compute capability 选 MMA v1/v2/v3+supportMMA 验票退档、warpsPerTile 平铺、换 dot-operand/mma 编码;ScaledBlockedToMMAv2+UpcastMXFP 兜 mxfp/fp8;decomposeMixedModeDotOp 混合精度升位兜底)、RemoveLayoutConversions(锚点→传播染色→消冲突→支配序重写四阶段消冗余 convert,前三纯分析;backwardRematerialization/hoistConvert 消残余但故意跳过 dot-operand)、OptimizeDotOperands(挪不删,SwizzleShmemConvert 把 tt.trans 融进 swizzled 共享编码/HoistLayoutConversion 省 shmem 往返)。三 pass 皆 ch25 analysis→transform 母范式实例、硬绑 NVIDIA。配对脊柱重灾区(回指 ch27 MMA 原理/ch24 convert_layout/ch25 母范式/ch22 swizzle/ch21 布局)。skip_impl(C++ pass,headless make_ttgir dump 观察),逐机制 15/15,盲审 1 轮过,APPROVED 全非阻断。Lead 修 lint_chapter_structure _BLOCK_MARKER_RE 接受 C++ // 锚点(exp-0717-7),清 7 条 core_mechanism_missing_source 假报、取代 ch24 的 //→# workaround。writer 修 3 处逐字块省略标记/2 must_keep 符号名/§2.4 直觉/§2.1 固定变量声明/§4 保语义论证。本章无伏笔埋/回收(bible due ch28 空)。

## Why it matters



## What to remember

Part VI 优化 pass 落地施工篇。三个挂在 make_ttgir 的 pass：AccelerateMatmul(BlockedToMMA 主 pattern——getMMAVersionSafe 按 compute capability 选 MMA v1/v2/v3+supportMMA 验票退档、warpsPerTile 平铺、换 dot-operand/mma 编码;ScaledB...
