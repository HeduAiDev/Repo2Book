# ch25《AxisInfo 静态分析与 Coalesce 改写》定稿——Part VI 开篇 + 全书第一个 analysis→transform 最短闭环

- **Type**: delivery
- **Chapter**: 25
- **Date**: 2026-07-17
- **Timestamp**: 2026-07-17T06:36:59Z
- **Agents involved**: analyst, implementer, tester, explainer, illustrator, writer, reviewer, archivist
- **User present**: False
- **Tags**: part-vi, foreshadow-payoff, f3, f9, analysis-transform, axisinfo, coalesce, skip_impl, skip_archive

## What happened

ch25 APPROVED 定稿归档,Part VI(优化 pass 部分)开篇。本章讲透两半场:AxisInfo 只读静态分析(lib/Analysis/AxisInfo.cpp,三元组 contiguity/divisibility/constancy + 悲观初值 + 稀疏前向数据流 + 格上 gcd join)与 Coalesce 改写 pass(lib/Dialect/TritonGPU/Transforms/Coalesce.cpp,setCoalescedEncoding/argSort 定 order + getNumElementsPerThread 三道 min 定向量宽)。回收两笔核心伏笔:f3(ch01 constexpr 性能主线)在 §3『constexpr→精确 divisibility』兑现——只有编译期具体数字 highestPowOf2Divisor 才算得出大 2 幂因子;f9(ch07 coalescing 判据 N_txn)兑现——ch09 multiple_of/max_contiguous、ch16 set_arg_attr 打的 tt.divisibility 标记经 initPessimisticStateFromFunc 成为悲观初值,getNumElementsPerThread 把人工 N_txn 判据算成每线程向量宽,Coalesce 按 argSort(contiguity) 产合并最优布局。§8 显式『两笔账在这里一并结清』复述。meta/skip_impl 章无精简版接口,按契约跳过 interfaces 登记。归档动作:f3/f9 arc-map 改 resolved(--in ch25);glossary 新增 12 词;concepts 新增 6 概念。

## Why it matters

analysis→transform 是 Part VI-VII(ch26-35)后续所有优化 pass 的母范式:先建一个只读分析、再据其结果保语义局部改写。这条前向依赖须显式登记——ch26-35 每章都是它的变体(先分析后改写),后续章开工前简报应回指本章作范式锚。f3/f9 是全书性能主线的两个关键兑现点,回收后 constexpr 主线(f3)与 coalescing 自动化(f9)在编译器内部闭环。

## What to remember

ch25=analysis→transform 母范式建立章,ch26-35 皆其变体(前向依赖,后续章简报回指此章)。f3(constexpr 精度)、f9(coalescing 自动化)已在此 resolved。术语口径与 ch07(coalescing/N_txn)、ch09(multiple_of/max_contiguous 打标)、ch16(set_arg_attr/tt.divisibility 落 IR)一致:标记→seed→传播→join→改写是一条链,标记进不了悲观初值则全链退 1。归档含 transient 逃逸→resume 恢复注记,产物完整无损。
