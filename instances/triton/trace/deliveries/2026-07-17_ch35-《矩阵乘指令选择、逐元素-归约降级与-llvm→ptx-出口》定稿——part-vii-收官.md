# 《矩阵乘指令选择、逐元素-归约降级与 LLVM→PTX 出口》定稿——Part VII 收官

- **Type**: delivery
- **Chapter**: ch35
- **Date**: 2026-07-17
- **Timestamp**: 2026-07-17T19:01:59Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, archivist, lead
- **User present**: False
- **Tags**: part-vii, part-vii-收官, skip_impl, per-op-lowering, dot, elementwise, fp8, reduce, scan, ptx-exit, 五级阶梯

## What happened

ch35(slug ch35-dot-elementwise-reduce-ptx-exit,kind=skip_impl)定稿,APPROVED(5/5 core 三层,write-review 1 轮/blind 1 轮/map 1 轮,all non-blocking)。Part VII(ch32-35)降级脊柱收官章:承 ch33 第二跳/ch34 共享内存+全局访存后,讲 per-op 降级尾段三支——(1)dot:DotOpToLLVM 按 NvidiaMma versionMajor 派单 Volta→mma.884/Turing→mma.1688/Ampere→mma.16816/Hopper→wgmma,布局非 mma 退 convertFMADot 标量兜底;MMAv2 ValueTableV2 凑操作数拼 mma.sync,kWidth=8 时 128bit 装不下沿 K stride-4 拆 4 条物理 mma(回指 ch27 fragment/ch08 dot 表面);(2)elementwise:ElementwiseOpConversionBase CRTP 模板拆 struct→unpackI32→per-element createDestOps→repack;fp8 转换查 srcMap 表+位宽镜像(fp16→fp8=32,16;fp8→fp16=16,32),RTNE/RTZ 进键(回指 ch05 fp8/ch06 类型转换/ch33 struct 塌缩);(3)reduce/scan:reduce warp 内 shfl.bfly 蝶形树+warp 间共享内存两轮,scan warp 内 Kogge-Stone,combine region 经 inlineCombineBlock 克隆内联(回指 ch08/ch34 shfl)。出口:所有硬件指令经 PTXBuilder→llvm.inline_asm 统一出关,NVGPUToLLVMPass 是第三方 dialect 挂载配对脊柱,末尾 BreakStructPhiNodes 拆 struct phi 收尾——五级阶梯 TTGIR→LLVM→PTX 走完(承 ch32 全貌)。逃生舱:Dossier 自核抓 fp8 Fp8ConversionDesc 位宽方向写反(Fp16_to_Fp8E4M3Nv 误记 16,32 应 32,16)→Lead 修 dossier 两方向+skip_dossier 复跑(首次 blind replace-all 误翻本对的 fp8→fp16 那条、经 2-occurrences 告警核源码后 revert),review 复核两方向对 pin 一致;Lead 另派 writer 补 3 处可读性(§4 fp8 89/90 门槛调和、§2 builder 前向指路 §7、RTNE/RTZ 缩写补全称),reviewer 定点修 ch36 跨章悬空链接。7 图全 PASS(dot-dispatch/mma-operand-assembly/elementwise-template/reduce-butterfly/scan-kogge-stone/ptxbuilder-launch/break-phi-struct)+chapter-map。无精简版接口(skip_impl 按契约跳过 interfaces)。本章无伏笔埋/回收(bible.py due ch35 确为空)。归档:并行 skip_archive 波次里 Lead 串行调度的唯一 Bible 写入者,无竞态;glossary +18、concepts +12、figures +8;archivist 只写 bible/trace。本章交付=Part VII(ch32-35)全部完成。

## Why it matters

Part VII(降级脊柱四章 ch32-35)收官,五级阶梯 TTIR→TTGIR→LLVM→PTX→cubin 的进程内 MLIR pass 四段(前四跳)在正文彻底走完:ch32 第一跳 TTIR→TTGIR、ch33/ch34 第二跳 TTGIR→LLVM 的类型塌缩+搬运/访存、ch35 per-op 降级尾段+LLVM→PTX 出口。全书『一个 kernel 怎么从 Python 一路降到硬件指令』的降级主线闭合,后续只余 ch36 CUDABackend(已定稿)把五段串成编译驱动。fp8 位宽镜像是全书 fp8 主线(ch05 dtype→ch06 转换→ch36 后端能力清单)的物理落地终点,dossier 自核在此抓到方向写反、避免了错误数值表进正文。

## What to remember

ch35 done,Part VII(ch32-35)全部完成。per-op 降级尾段落点:dot=DotOpToLLVM 按 versionMajor 派 mma.884/1688/16816/wgmma+FMA 兜底、MMAv2 ValueTableV2 拼 mma.sync(kWidth=8 拆 4);elementwise=ElementwiseOpConversionBase CRTP 模板+fp8 srcMap 位宽镜像(fp16→fp8=32,16 / fp8→fp16=16,32,dossier 自核修正点);reduce=shfl.bfly 树、scan=Kogge-Stone、combine 经 inlineCombineBlock 内联;统一经 PTXBuilder→llvm.inline_asm 出关,NVGPUToLLVMPass 配对脊柱,BreakStructPhiNodes 收尾。glossary+18/concepts+12/figures+8。无伏笔。后续若讲 mma 指令/fp8 转换/reduce-scan 降级可复用本章图与词条,勿重画。
