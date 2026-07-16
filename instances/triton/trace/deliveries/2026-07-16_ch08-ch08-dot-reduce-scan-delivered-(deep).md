# ch08-dot-reduce-scan-delivered-(deep)

- **Type**: delivery
- **Chapter**: 08
- **Date**: 2026-07-16
- **Timestamp**: 2026-07-16T16:59:05Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch08, deep, tl.dot, tensor-core, min_dot_size, input_precision, combine_fn, reduce, scan, argmax, histogram

## What happened

第八章《块级计算的两大主题:tl.dot 命不命中 Tensor Core;combine_fn 变 IR region》交付(kind=deep,pin triton==3.2.0 精确编译取证,skip_impl 无精简版)。上半程(§1-§5,性能落点):semantic.dot 校验闸门(dtype 白名单+同型+阶数+K相容)->min_dot_size 后端能力钩子(codegen_fns 声明形状锁,NVIDIA 非int8(16,16,16)/int8(16,32,16))->input_precision 三层优先级(显式参数>环境变量TRITON_F32_DEFAULT>后端默认,tf32 vs ieee精度锁)->acc/out_dtype 累加器类型反推(dtype锁)->三把锁合流命中判据自查表(fig-tc-hit-criterion)。dot_scaled/microscaling 点到即止。下半程(§6-§9,概念机理):combine_fn 经 make_combine_region+call_JitFunction 把函数体AST再编译进 reduce_op/scan_op 的 IR region(非Python调用求值,回指第1章 visit_Call/CodeGenerator 机制不重讲)->为何combine_fn只能写tl.*可追踪操作(tl.tensor无__bool__,须用core.where)->combine_fn双入参协议(function_type(in,in*2))-> _reduce_with_indices 支撑 argmax/argmin(值索引成对归约,fig-argmax-tree交叉验证value=4/index=2)->histogram 语义封闭无combine_fn作对照。13机制(8 core+5 supporting)。5图(chapter-map/fig-min-dot-size/fig-tc-hit-criterion/fig-combine-fn-to-region/fig-argmax-tree)。review APPROVED,8条issue全negotiable/non-blocking(1条lint误报存档+1条逐机制勾选表+2条可选加强+1条chapter-map门禁越权提醒+3条reader-comprehension:uint8白名单与int8窄assert矛盾未搭桥/argmax_combine_tie_break_left命名跳跃未绑定/MMA缩写未展开)。blind round1 PASS(0 failures)。chapter-map round1 PASS。write_review_rounds=1、blind_rounds=1、无escalation。

## Why it matters

本章解锁全书性能主线的关键杠杆:读者能据此判断自己的tl.dot写法命不命中Tensor Core(三把锁模型可直接套用到任意dot调用)。同时把combine_fn->IR region这一编译机理讲透,解释了归约/扫描原语的核心易错点(为什么combine_fn里不能写原生if/调外部库)。min_dot_size的codegen_fns钩子模式(后端声明能力、语言层照查)是后续硬件后端章节的一条主线伏笔,呼应ch01 f1/ch05 f6已埋的后端接缝伏笔(payoff ch36),本章不新开伏笔条目。

## What to remember

ch08 done(kind=deep,无精简版)。dossier foreshadow_due 为空(should_plant/should_recover 均空),本章无新埋伏笔,亦无待回收项;min_dot_size/codegen_fns 天然呼应已开放的 f1(ch01->ch36)/f6(ch05->ch36)后端接缝伏笔,未重复登记。glossary 新增14术语(Tensor Core/MMA/min_dot_size/input_precision/tf32与ieee/codegen_fns/acc与out_dtype/max_num_imprecise_acc/combine_fn/IR region/call_JitFunction/_reduce_with_indices/associative_scan/histogram)。concepts 新增9条->ch08。review-report.json 8条issue均negotiable/non-blocking,3条reader-comprehension小卡点(uint8/int8矛盾、argmax_combine命名跳跃、MMA缩写未展开)供writer后续顺手打磨,不影响归档。无精简版接口(interfaces.json未新增)。
