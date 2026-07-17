# ch24《ttg.* 与 ttng.* 算子:布局转换、异步拷贝与 Hopper 硬件方言》定稿——Part V 收官

- **Type**: delivery
- **Chapter**: ch24
- **Date**: 2026-07-17
- **Timestamp**: 2026-07-17T08:04:24Z
- **Agents involved**: archivist
- **User present**: False
- **Tags**: ch24, part-5, ttg, ttng, convert_layout, memdesc, cp.async, TMA, mbarrier, WGMMA, f10-payoff, f15-payoff, skip_impl, pairing-spine, APPROVED

## What happened

Part V 收官章(kind=skip_impl,无精简版,交叉验证走 pin 精确编译 + dump 认字)。读透 TritonGPU(ttg,后端无关)与 TritonNvidiaGPU(ttng,NVIDIA 硬件专属)两层方言。核心论断:convert_layout 是 ttg 层唯一真正在线程间搬数据的算子(SameShape+SameElementType+Pure,唯一可变的只有 encoding),trans/reshape 只给元素改名零搬运——数 dump 折叠后残留的 convert_layout 条数=数布局转换开销(hasCanonicalizer 单调递减到不动点证明每条都省不掉)。共享内存生命周期由 memdesc SSA 句柄界定(local_alloc/dealloc/load/store + memdesc_subview);cp.async 三件套异步流水(async_copy/async_commit_group/async_wait + async.token 串时序)。ttng 层收 sm90 杀手锏:warp_group_dot(WGMMA)/async_tma_copy+mbarrier 时序基元(init_barrier·barrier_expect·wait_barrier 相位翻转)/cluster 同步/warp 专化/upcast_mxfp。回收两条伏笔:f10(ch07 埋 TMA descriptor 化访存→ch24 async_tma_copy_global_to_local 是这条路径降级到硬件方言的落点,搬运粒度从线程级升到张量块级)、f15(ch19 埋 tt.trans 只是改名、真搬运在前后 convertLayout→ch24 convert_layout 段兑现)。10 机制(chapter.md 全覆盖,m3/m5 带 trace 数值推演表)。5 图(chapter-map + f24-1 改名vs搬运/f24-2 memdesc 生命周期/f24-4 cp.async token 链/f24-9 TMA+mbarrier 时序)全 blind PASS。战略价值:TritonNvidiaGPU 硬件专属方言=配对脊柱,姊妹篇 ascend NPU 硬件方言逐结构对位的样板。所有 linter green,review 双维度 APPROVED(fidelity+mechanism 均无 blocking)。并行 skip_archive 波次 Lead 串行归档,本轮唯一 Bible 写入者无竞态。

## Why it matters

Part V(布局与方言)收官,把前四章布局代数落到真正搬数据的算子层,并首次讲透 ttng 硬件专属方言——为姊妹篇 triton-ascend NPU 硬件方言逐结构对位立样板(配对脊柱)。回收 f10/f15 两条跨部伏笔,闭合 ch07 访存/ch19 tt 层转置两条前瞻线。

## What to remember

Part V 收官章(kind=skip_impl,无精简版,交叉验证走 pin 精确编译 + dump 认字)。读透 TritonGPU(ttg,后端无关)与 TritonNvidiaGPU(ttng,NVIDIA 硬件专属)两层方言。核心论断:convert_layout 是 ttg 层唯一真正在线程间搬数据的算子(SameShape+SameElementType+Pure,唯一可变的只有 ...
