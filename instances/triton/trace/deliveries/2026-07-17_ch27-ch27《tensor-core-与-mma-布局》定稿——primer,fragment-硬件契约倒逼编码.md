# ch27《Tensor Core 与 MMA 布局》定稿——primer,fragment 硬件契约倒逼编码

- **Type**: delivery
- **Chapter**: 27
- **Date**: 2026-07-17
- **Timestamp**: 2026-07-17T09:07:26Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch27, primer, part-6, tensor-core, mma, fragment, dot-operand, layout, wgmma, APPROVED

## What happened

Part VI primer 原理章定稿(kind=primer,无精简版接口)。核心论证:MMA/dot-operand 编码字段不是自由设计,是被 warp 级 mma.sync 的 (lane,register) fragment 硬件契约倒逼出来的——opIdx 选 a/b 半张 fragment 表、instrShape 一砖尺寸、warpsPerCTA warp 分工、kWidth(=32/bitwidth)沿 K 每线程连续段长,各答一条 fragment 要求。黄金可核 worked example:m16n8(FP32)C accumulator 逐 lane 线程矩阵,32 lane 各持 4 fp32、坐标 (g,2h)/(g,2h+1)/(g+8,2h)/(g+8,2h+1) g=lane>>2 h=lane&3,逐格照抄 TritonGPUAttrDefs.td:L1105-L1126 源码注释、完全可核(fig-m16n8k16-fragment)。A/B 操作数每线程元素数由 getSizePerThreadForOperand(Dialect.cpp:L2144-L2159)算死可核(A=8/B=4 f16、守恒 8×32=256/4×32=128),但逐 lane 精确 (row,K)/(K,N) 坐标本仓/paper 未证——诚实标待核回指 PTX ISA #mma-16816-a-f16/-b-f16(exp-0715-1 硬规则:绝不编造硬件坐标)。versionMajor 按算力经 getMMAVersionSafe 自动选代(v2 Ampere 操作数在寄存器/warpsPerTile 贪心两维平铺 vs v3 Hopper 操作数搬共享内存/warpsPerTile 固定(4,1)发 WarpGroupDotOp)。5 图全盲审 PASS。全 linter 绿;review APPROVED(algorithm-pedagogy)。bible due ch27 确为空(无伏笔埋/回收)。

## Why it matters

primer 章质量的两大命门都在本章被压住:(1) 顿悟入口=可核黄金 worked example——C accumulator 逐 lane 坐标是全书唯一逐格照抄源码即可自证的 fragment 表,读者拿它就能反推『编码字段为什么长这样』;(2) 诚实边界=A/B 逐 lane 坐标能算的(每线程元素数)算死、不能核的(精确 PTX 坐标)标待核不编造,是 exp-0715-1『绝不编造硬件坐标』在 primer 章的样板执行。本章为后续 AccelerateMatmul pass 本体章的前置原理,布局侧动机(MMAv3 操作数搬 shared 发 WarpGroupDotOp)与 ch24 warp_group_dot 的异步流水 lowering 分工互指、不重讲。

## What to remember

primer 抓两点:黄金可核 worked example(逐格照抄源码的 C accumulator 座位表)+诚实边界(能算的算死、不能核的标待核回指 PTX,exp-0715-1)。归档注记:①dossier 对抗性自核抓出线程矩阵错读(C 座位表读法),Lead 修正复跑改对;②figure-integration 逃逸——fig-ab-operand-structure 首版沿 M/K 轴切 4 条『每带 4 threads』横带,暗示只 16 线程参与+伪造 M 轴切分(与 32 lane 守恒矛盾),Lead 派 illustrator 重绘为高亮一行/一列沿 K=16 由 t0..t3 分担+绿色守恒行,重渲染后盲审 PASS。glossary +10 词(mma.m16n8k16/fragment、opIdx、kWidth、instrShape、warpsPerTile、accumulator、MMAv2/MMAv3、WGMMA、getSizePerThreadForOperand);concepts +8;figures +6(含 chapter-map);Tensor Core/MMA/NvidiaMmaEncodingAttr/DotOperandEncodingAttr/warpsPerCTA/warp_group_dot 沿用 ch08/ch21/ch24 既有词条不重复。interfaces 按 primer 契约跳过。
