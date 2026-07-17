# ch33《第二跳的地基:类型塌缩与 ConvertLayoutOp 的三条搬运路径》定稿

Date: 2026-07-17
Type: delivery
Chapter: ch33
Part: part-7 (skip_impl)
Verdict: APPROVED

## 交付要点
- TTGIR→LLVM 脊柱(第二跳地基,接 ch32 第一跳)。
- 类型塌缩 TritonGPUToLLVMTypeConverter:带布局张量→LLVM struct-of-N-elements;两阶段 applyPartialConversion 按 PatternBenefit 拼 pattern(TargetInfo 接缝)。
- ConvertLayoutOp 三条搬运路径(代价按跨线程流量严格递增):纯寄存器重排(零流量)< warp shuffle < 共享内存往返(transferWithinBlock/Impl,invertAndCompose 给精确 shmem 地址,stmatrix 快路径,padding 防 bank 冲突);legacy/LinearLayout 两路迁移期共存(benefit 11 vs 10)。
- perf:dump 里走 shmem 往返的 convert_layout = 优化目标(回指 ch24/ch28)。
- 无伏笔埋/回收。glossary/concepts 已登记(transferWithinBlock/struct-of-N/三路径等)。

## 归档过程注记
- workflow review-exhausted 逃逸(exp-0716-1 第 5 例,figure-only blocking:fig-ch33-pass-spine benefit=20 误标 convert_layout 实为 LocalAllocOpConversion(pin ConvertLayoutOpToLLVM.cpp:790);fig-ch33-cost-ordering 8-vs-4 store 与正文矛盾);另一 Claude session(reviewer 子 agent)提前私信 Lead。
- Lead 待 workflow 逃逸后派 illustrator 落地 3 条 figure-requests(Lead 亲 Read PNG 核实两 blocking 图已正确:local_alloc(优化)/4 store+4 load(共8元素)/两迭代合计)+ 补 Map 站。
- writer 第 3 轮文字修(transferWithinBlockImpl 点名 / shmem 与共享内存绑定)已在稿。
- archivist 写完 glossary/concepts 后 API stall(Response stalled mid-stream),Lead 补齐 state/delivery/INDEX。
