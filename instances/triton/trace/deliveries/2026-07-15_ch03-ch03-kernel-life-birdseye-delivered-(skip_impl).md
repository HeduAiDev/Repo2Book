# ch03 kernel life birdseye delivered (skip_impl)

- **Type**: delivery
- **Chapter**: 03
- **Date**: 2026-07-15
- **Timestamp**: 2026-07-15T16:52:50Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch03, skip_impl, birdseye, compilation-pipeline, cache-key, five-stage-lowering, bilingual-seam, headless-gpu-fracture, perf-locator-map

## What happened

第三章《核的一生：鸟瞰》交付。kind=skip_impl(取景框/串场章,无精简版,只做减法条款不适用)。用最小 add_kernel 走通全链路低分辨率地图:fn[grid](...) 触发 JITFunction.run → cache key(签名+特化位+constexpr,m02 编译期特化旋钮)命中/未命中分岔 → compile() 五级降级阶梯 ttir(追踪期 vs make_ttir 之后,add_inliner 抹平 tt.call,ch01 教训①照搬)→ttgir(贴布局,m05 优化 pass 旋钮)→llir(MLIR→LLVM 边界)→ptx(NVPTX 后端)→cubin(ptxas 子进程)→CompiledKernel._init_handles 发射(m09 发射开销旋钮,真卡起点)。首次点名全部双语栈断点(m10)与无 GPU 断裂处地图(m11,headless 能看到哪几级 IR/哪级起须真卡)，TRITON_KERNEL_DUMP 逐级 dump 锚点(m12)配定位地图(m13:症状→层→章)。pin triton==3.2.0 headless 编译核实每级产物真实存在。5 张图 1 轮盲审 PASS,本章地图 1 轮 PASS,写作-评审 2 轮收口,APPROVED(8 条 negotiable/non-blocking issue,均判定为可选锦上添花,不影响交付)。

## Why it matters

本章是全书性能归因的定位地图——给读者「kernel 慢时该去哪一层拧旋钮」的心智框架,不重复 ch01 已讲透的机制细节(visit_Call 三岔/双语接缝),而是把 ch01 建立的概念低分辨率串成一条主线,并给后续每一章(编译期特化/优化 pass/IR 与布局/发射开销等)挂上「后面哪一章细讲」的路标。姊妹篇 Triton-Ascend 对位、五级降级/发射开销等多条伏笔在此重新激活并指向 part-4 起各章。

## What to remember

ch03 done(skip_impl,鸟瞰/串场章,无精简版接口可登记,bible.py due ch03 为空——无应埋/应回收伏笔,foreshadow_due.note 已在 dossier 中记录本章式的组织性路标非正式伏笔)。5图入figures.json候选、复用ch01术语(concepts.json未新增,本章为低分辨率重述而非新概念建立)。review APPROVED,8条issue全negotiable/non-blocking。
