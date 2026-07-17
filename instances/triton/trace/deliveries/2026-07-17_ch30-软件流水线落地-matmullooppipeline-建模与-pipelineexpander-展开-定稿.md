# 软件流水线落地 MatmulLoopPipeline 建模与 PipelineExpander 展开 定稿

- **Type**: delivery
- **Chapter**: ch30
- **Date**: 2026-07-17
- **Timestamp**: 2026-07-17T13:43:28Z
- **Agents involved**: analyst, implementer, tester, explainer, illustrator, writer, reviewer, archivist
- **User present**: False
- **Tags**: triton, part-vi, software-pipelining, payoff-f16, skip_impl, APPROVED

## What happened

ch30(Part VI)定稿并 APPROVED。本章是 ch29 软件流水线/模调度原理的源码落地 pass:分建模半(preProcessLoopAndGetSchedule 识别喂 dot 的 load→createAsyncCopy 换成 cp.async 三件套写进多 buffer 环形缓冲→scheduleLoads/scheduleDependencies/scheduleRemainingToLastStage 给每个 op 打 stage 标签→createFinalSchedule 拍平,产物封进 PipeliningOption)与展开半(LoopPipelinerInternal 五步:initializeLoopInfo→emitPrologue 灌流水→analyzeCrossStageValues 量活跃跨度→createKernelLoop 模变量扩展→createKernel 稳态改写,展开成 prologue+稳态 kernel loop+epilogue)。回收伏笔 f16:getNumStagesOrDefault 读 ch17 埋下的 tt.num_stages 属性驱动流水线深度(第一读点 L63),且循环显式标 tt.num_stages 还放宽哪些 load 值得预取(第二读点 L277)——已 resolve。Hopper 尾声 asyncLaunchDots+dotCanBeProperlyAsync 让 wgmma 异步(回指 ch24)。两类代价:SRAM(numBuffers 随 num_stages 涨,回指 ch26)+iter_arg 膨胀(3 撑到 7)。skip_impl 章,无精简版接口。Lead 派 writer 补 7 处、illustrator 修 fig-m11(术语提前)。

## Why it matters

f16 是 ch17→ch30 的跨 13 章长伏笔,本章兑现「谁读 tt.num_stages、怎么用」;ch30 让 ch29 抽象原理(最优深度 s-star/模调度)落到真实源码行,PipeliningOption 后端无关接缝为姊妹篇(triton-ascend)可移植性埋接口。

## What to remember

ch30 done、APPROVED、Part VI;f16(ch17→ch30)已 resolve;skip_impl 无接口;新增 10 glossary+10 concepts;依赖 ch29(原理)/ch24(async_copy/warp_group_dot)/ch26(共享内存预算)/ch17(tt.num_stages)。
