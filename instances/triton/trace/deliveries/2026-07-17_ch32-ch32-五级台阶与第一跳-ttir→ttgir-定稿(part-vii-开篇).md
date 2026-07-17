# ch32 五级台阶与第一跳 TTIR→TTGIR 定稿(Part VII 开篇)

- **Type**: delivery
- **Chapter**: ch32
- **Date**: 2026-07-17
- **Timestamp**: 2026-07-17T12:25:36Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, archivist
- **User present**: False
- **Tags**: triton, ch32, part-7, lowering, ttir-ttgir, dialect-conversion, blocked-encoding, dot-operand, foreshadow-payoff, f2

## What happened

ch32《五级台阶与第一跳 TTIR→TTGIR:给每个张量贴上布局》定稿,APPROVED,Part VII「降级」开篇章(skip_impl,无精简版接口)。两条主线:(1)五级降级阶梯全貌 TTIR→TTGIR→LLIR→PTX→cubin——CUDABackend.add_stages 把五段注册进 stages 字典(前四段进程内 MLIR pass_manager 做 IR→IR、第五段 make_cubin shell 调 ptxas),回收 ch01 埋下的伏笔 f2。(2)正式走第一跳:make_ttir 先在 TTIR 级清理(RewriteTensorPointer 把 block pointer 降解成显式 splat/expand_dims/broadcast/addptr、Combine 窥孔合并),make_ttgir 第一个 pass add_convert_to_ttgpuir 即第一跳——TritonGPUTypeConverter 给无布局张量默认贴 Blocked(getDefaultBlockedEncoding:order 行主序/sizePerThread 全 1)、TritonGPUConversionTarget 用 addDynamicallyLegalOp 声明 tt.dot 两操作数皆 DotOperand 才合法、框架 applyPartialConversion 不动点驱动 TritonDotPattern 焊上三条 convert_layout 胶水(A/B 转 opIdx=0/1 DotOperand、C 转结果布局)。实测 16x16 fp16 matmul_bp 取证 #blocked1=blocked<sizePerThread=[1,1],threadsPerWarp=[2,16],warpsPerCTA=[4,1],order=[1,0]> + 三条 convert_layout SSA(%19/%20/%21)。归档:Lead 派 writer 补 6 处覆盖/可读性定点小修 + illustrator 修 2 图行号,归档时 chapter.md/diagrams 仍在小修中,archivist 只写 bible/trace 无冲突。

## Why it matters

f2 是全书降级主线的总纲伏笔(ch01 埋),ch32 兑现全貌+第一跳,后续 ch33/34/35 顺台阶逐级展开到 PTX 出口——回收登记确保降级主线闭合、后续章能引用第一跳产物(默认 #blocked、tt.dot 胶水)而不重述。Part VII 开篇为整部降级篇立骨架。

## What to remember

f2 已 resolved(ch32)。第一跳三样东西:TypeConverter 贴默认 Blocked(基线非最优)/ConversionTarget 声明 tt.dot 须 DotOperand 逼非法/TritonDotPattern 焊 convert_layout。dialect-conversion=声明合法性+提供 pattern+applyPartialConversion 不动点收敛。block pointer 只活到 TTIR(RewriteTensorPointer 降解)。glossary +11、concepts +5。后续 ch33+ 讲同机制(五级台阶/布局注入)复用本章,勿重画重述。skip_impl 无接口。
