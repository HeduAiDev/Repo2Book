# ch22-shared-encoding-swizzle-delivered-(skip_impl)

- **Type**: delivery
- **Chapter**: 22
- **Date**: 2026-07-17
- **Timestamp**: 2026-07-17T04:43:37Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch22, part-5, skip_impl, shared-encoding, xor-swizzle, bank-conflict, mma-driven-derivation

## What happened

第二十二章《Shared 编码与 swizzle：共享内存里如何避开 bank 冲突》交付（Part V「IR 与布局」第四站，skip_impl 章；并行发车 skip_archive 模式，Review+Map 已 APPROVED/PASS，本次由 archivist 串行补归档）。与 ch21 distributed 布局并列的另一半布局拼图：①`SharedEncodingAttr` 六字段 schema(`vec`/`perPhase`/`maxPhase`/`order`/`CTALayout`/`hasLeadingOffset`)，与 distributed 的分野由 `getElemsPerThread` 对 shared 直接 `llvm_unreachable` 坐实(回指 ch21)；②xor swizzle 命门`out[r][c]=in[r][c^phase]`，.td 文档五个逐步加料例子(basic→perPhase→maxPhase→合并→vec)手算逐位复现(Python 核验与源码完全一致)，归纳出统一公式 `phase(r)=⌊r/perPhase⌋ mod maxPhase` 作用于 `⌊c/vec⌋` 组；③全章第三论点——swizzle 参数不是拍脑袋，由目标 mma 指令的访问模式反推钉死：AMD MFMA 分支(numBanks/bankBitWidth/SIMDWidth 硬件常量→perPhase/vec/maxPhase)、Ampere 分支(`matShape={8,8,4*kWidth}` 直接钉死 vec 与 maxPhase)逐段内嵌真源码；④Hopper/MMAv3 走独立 by-eltTy builder(128B/64B/32B 三档+`hasLeadingOffset=true`)，因为 dotOperand 反推路径对 Hopper `llvm_unreachable`；⑤`CTALayoutAttr` 跨 CTA 切分/multicast 作次要论点点到；⑥`SharedEncodingAttr::parse/print` 闭环读者在 IR dump 里能看到的 `#shared<{...}>` 文本形态。全部由 TritonGPUAttrDefs.td/Dialect.cpp 真实源码逐段内嵌驱动；无精简版(kind=skip_impl，MLIR TableGen .td 与 C++ 无法做同名同结构精简版，交叉验证由 explainer 走 pin 精确编译+五例手算逐位复现承担)。11 机制(5 core+6 supporting)；全 blind PASS；review APPROVED。

## Why it matters

本章与 ch21 合拢成 ch23(LinearLayout)的两块直接前置拼图——distributed 与 shared 是布局函数𝓛的两套心智模型(寄存器分散 vs 共享内存全员可见)，`toLinearLayout` 要同时吃这两种 encoding。`SharedEncodingAttr` 的六字段是 ch27(mma 布局深化)、ch34(共享内存降级：`storeDistributedToShared` 按 order 换相、`LocalLoadOpConversion` 的 ldmatrix 加载器)真正把 swizzle 编译成物理偏移算术的输入——本章只给数学定义与参数来源，物理 lowering 留给 ch34（新开伏笔 f18）。`xor swizzle` 消 bank 冲突的机理也回应了 ch02 内存延迟金字塔留下的悬念（本章用结论、不重讲硬件原理）。

## What to remember

ch22 done（kind=skip_impl，Part V 第四站，全 blind PASS，review APPROVED，无 escalation）。**glossary.json 226→234**（新增 8 条：`SharedEncodingAttr`/`vec`/`perPhase`/`maxPhase`/`hasLeadingOffset`/`xor swizzle`/`bank 冲突`/`needTrans`；`order`/`CTALayoutAttr`/`DotOperandEncodingAttr`/`getElemsPerThread` 已在 ch20/ch21 登记，本章复用未改动既有词条）。**concepts.json 163→167**（新增 4 条→ch22：SharedEncoding 六字段描述共享内存排布、xor swizzle 消 bank 冲突机理、swizzle 参数被目标 mma 反推钉死、shared vs distributed 两套心智模型分野）。**interfaces.json 新增 ch22 键**（`SharedEncodingAttr`+两条 builder(dotOperand 反推/Hopper by-eltTy)+`CTALayoutAttr`(复用注明)+`parse/print`+`getElemsPerThread`+swizzle 物理 lowering 消费点占位，供 ch27/ch34 回指）。

**arc-map.json**：新开正式伏笔 **f18**(plant ch22→payoff ch34：swizzle 六字段如何在物理 lowering 时真正驱动地址计算，`storeDistributedToShared`/`LocalLoadOpConversion` 是消费点)——`bible.py due ch22` 应埋/应回收本为空，判断 shared 布局参数确有具体、非泛化的下游消费者(ch34 focus 原文点名 `storeDistributedToShared` 按 SharedEncoding order 换相)，构成值得登记的独立技术悬念，不同于 ch20/ch21 从简处理的泛化过渡句。**一致性核验**：全部 resolved 伏笔(f4→ch16/f5→ch13/f7→ch06/f11→ch12/f12→ch14/f13→ch17/f14→ch20)均 payoff==resolved_in 且 payoff≤已交付章节，无异常；f15(plant ch19→payoff ch24)/f16(plant ch17→payoff ch30)/f17(plant ch20→payoff ch23)仍 open 未被误动；新开 f18(plant ch22→payoff ch34)status=open，payoff>ch22，无异常。

trace：本条 delivery 已建；`state.json` 已加 `ch22` 条目；`trace/INDEX.md` 已刷新（保留最近 10 条，自检确认 ch22 在列，无 untitled 垃圾——本次全程用 `bible.py foreshadow --add`（非 `archivist.py record` 无参探针）登记新伏笔，未触发该已知陷阱）。`reviews/review-report.json`、`reviews/run-ledger.json`、`narrative/chapter.md`、`diagrams/`、`dossier/dossier.json` 均未触碰。
