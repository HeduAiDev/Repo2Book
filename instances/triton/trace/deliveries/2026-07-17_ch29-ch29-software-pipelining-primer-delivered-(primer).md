# ch29 software-pipelining-primer delivered (primer)

- **Type**: delivery
- **Chapter**: ch29
- **Date**: 2026-07-17
- **Timestamp**: 2026-07-17T11:37:20Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, archivist
- **User present**: False
- **Tags**: part-vi, primer, software-pipelining, modulo-scheduling, num_stages, triton

## What happened

ch29《软件流水线与模调度:num_stages 到底调度了什么》定稿(Part VI,primer 原理章,kind=primer 无精简版接口)。核心:软件流水线=跨迭代重叠(切 stage 让不同迭代的不同 stage 同一时间片并行,盖住访存延迟,不改语义);两半架构=建模(MatmulLoopPipeline 的 scheduleLoads→createAsyncCopy→createFinalSchedule 定 op→(stage,cluster))+ 展开(PipelineExpander pipelineForLoop 五步发 prologue 填/steady state 稳态/epilogue 排空三段);num_stages=流水线深度=预取深度=buffer 份数(<=1 关流水线,createAlloc 扩 distance 维成环形缓冲 numBuffers=max(distToUse));落地权衡 num_stages 调大藏延迟 vs 爆共享内存/寄存器(回指 ch26 预算/occupancy)。诚实边界:经典模调度的 initiation interval(II,发起间隔)与 steady state(稳态)两词 grep 整个 Pipeliner/ 未命中 Triton 源码——Triton 用 stage-based 非 II-based,正文只用教科书级重叠直觉,II 定义/下界公式/模调度伪码一律标待核·回指 DOI:10.1145/53990.54022(Lam 1988)。归档流程:pipeline 经 review-exhausted 逃逸(exp-0716-1 第 4 例,figure-only)——但图实为已被 pipeline 补图站修好、review 记账滞后;Lead 核 PNG 确认 prologue-steady-epilogue 图已正确((末,wait)+(末-1,dot))、清 figure-requests、补 Map 站、writer 定点修 §3.3 peelEpilogue 前提-结论错位(Triton 设 peelEpilogue=false)。linter 全 green,review APPROVED。无伏笔埋/回收(bible.py due ch29 确为空)。

## Why it matters

num_stages 是 GEMM 类访存受限循环最直接的一档提速旋钮,读者从 ch04/ch17 只知它挂成 tt.num_stages 属性、不知谁消费;本章把'到底调度了什么'讲透,是 ch30(MatmulLoopPipeline 建模+PipelineExpander 落地完整源码走查)的前置原理章(前向依赖 ch30)。诚实边界(II/steady state 非源码符号)确立了 primer 章'教科书概念与真实源码机制分层标注'的口径,防术语漂移。

## What to remember

ch29=软件流水线 primer;num_stages=深度=预取深度=buffer 份数(numBuffers=max(distToUse),回指 ch26 共享内存预算);两半=MatmulLoopPipeline 建模(op→(stage,cluster) CoarseSchedule)+PipelineExpander 展开(prologue/steady/epilogue);Triton peelEpilogue=false 靠谓词化处理边界;II/steady state 是经典理论背景、Triton 用 stage-based 非 II-based,待核回指 Lam 1988 DOI:10.1145/53990.54022;是 ch30 前置原理章(前向依赖);无伏笔埋/回收(due 空)。
