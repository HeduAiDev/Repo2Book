# ch33 手工推演台账（trace_source=manual）

本章 kind=skip_impl，无 subtract-only 精简版可跑。可运行交叉验证由 pin==3.2.0 精确编译承担
（headless dump 一段带 convert_layout 的 kernel 的 make_llir），但本机 host 无 CUDA、且已装
triton 为 3.6.0 而非 pin 的 3.2.0（版本不逐字节同，dump 出的 IR 行号/结构可能偏离本章所引
源码），故所有 worked_example 走**手工推演**：每个引用源码常量/行号的数字标 `file:Lxxx`，
自选的小示例参数（张量形状、置换、迭代数）为构造值、内部自洽、不冒充实测。

真值口径 = 源码常量（已逐一 grep 核对于 `instances/triton/source`，规范路径去前缀）：

## 源码常量核对
- `patternBenefitDefault = 1`、`patternBenefitPrioritizeOverLLVMConversions = 10`、
  `patternBenefitConvertLayoutOptimizedPattern = 20`
  —— include/triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h:L25-L28
- struct-of-N 字段数 `N = getTotalElemsPerThread(type)`
  —— lib/Conversion/TritonGPUToLLVM/TypeConverter.cpp:L112-L114
- shared/memdesc 塌成 {ptr} + `rank*2` 个 i32（offsets+strides）
  —— lib/Conversion/TritonGPUToLLVM/TypeConverter.cpp:L97-L110, L117-L134
- fp8（E4M3FN/E4M3FNUZ/E5M2/E5M2FNUZ 四编码）→ i8
  —— lib/Conversion/TritonGPUToLLVM/TypeConverter.cpp:L34-L45
- 相除维序 `dims = {"block","warp","lane","register"}`
  —— lib/Analysis/Utility.cpp:L661
- 选路四分支：block(L295, NYI)/warp(L300→transferWithinBlock L307)/
  lane(L308→transferWithinBlock L318, warp-shuffle 专用实现 TODO 未做故落 shmem)/
  register(L319→transferWithinThread L322)
  —— lib/Conversion/TritonGPUToLLVM/ConvertLayoutOpToLLVM.cpp:L295-L322
- transferWithinThread：`outVals.resize(conversion.getInDimSize(kRegister))`，
  `outVals[i] = inVals[srcIdx]`，前置 `assert(!cvtNeedsSharedMemory(...))`
  —— lib/Conversion/TritonGPUToLLVM/ConvertLayoutOpToLLVM.cpp:L339-L345
- 往返 store/load 地址：`srcLayout.invertAndCompose(sharedLayout)`（store）/
  `dstLayout.invertAndCompose(sharedLayout)`（load）
  —— lib/Conversion/TritonGPUToLLVM/ConvertLayoutOpToLLVM.cpp:L514, L523
- 往返主循环 barrier：迭代开头 `if (i != 0) insertBarrier`（L608）+ store 后 `insertBarrier`（L632）
  → 迭代数 iters 时 barrier 总数 = 2*iters - 1
  —— lib/Conversion/TritonGPUToLLVM/ConvertLayoutOpToLLVM.cpp:L606-L644
- padding：`paddedSize = max(inVec, outVec)`，加在 `paddedRepShape[outOrd[0]]`
  —— lib/Analysis/Allocation.cpp:L164-L165
- 三判据（互斥完备分档）：cvtReordersRegisters(L672)/cvtNeedsWarpShuffle(L683)/
  cvtNeedsSharedMemory(L695-L705)
  —— lib/Analysis/Utility.cpp:L672-L705
- legacy 路径 knob `useLegacyMMAConversion = false`；LL 路径 benefit=20 优先、legacy benefit 低作后备
  —— lib/Conversion/TritonGPUToLLVM/ConvertLayoutOpToLLVM.cpp:L28-L30

## 各 worked_example 构造参数与推导

### type-collapse-tensor
张量 `tensor<16x16xf32, #blocked>`：总元素 16*16 = 256；numWarps=4、threadsPerWarp=32 →
线程数 4*32 = 128；`N = getTotalElemsPerThread = 256/128 = 2`（均匀 blocked，元素守恒地摊到线程）
→ `!llvm.struct<(f32, f32)>`。换 numWarps=8 → 256 线程 → N=1，总量 256*1... 即 256=8*32*1 恒等。

### cvt-path-selection
构造 comp = dstLayout.invertAndCompose(srcLayout) 在 block、warp 维为恒等、lane 维非恒等
（典型：同一 warp 内跨 lane 的转置/洗牌，不跨 warp、不跨 CTA）。相除循环按 [block,warp,lane,register]：
第 1 轮 block 恒等→quotient 成功消去；第 2 轮 warp 恒等→成功消去；第 3 轮 lane 非恒等→quotient
失败→break。剩余 dims={lane,register}，matchAndRewrite 首个命中 `is_contained(dims,"lane")`（L308）
→ transferWithinBlock（共享内存往返）。

### path-register-reorder
N=4 寄存器/线程，取一个"交换中间两张牌"的置换：apply({register:i}) = [0→0, 1→2, 2→1, 3→3]。
outVals[i]=inVals[srcIdx]，全在本线程寄存器内，0 跨线程流量。

### path-shmem-roundtrip
构造 inVals.size()=8、iterations=2、inVec=outVec=2：每迭代 store 循环跑 8/2=4 元素、步长 inVec=2
→ 2 次 store op；load 同理 2 次 load op。barrier 总数 = 2*2-1 = 3（iter0 只有 store 后 1 次；
iter1 有开头 1 次 + store 后 1 次）。padding paddedSize=max(2,2)=2。

### cost-ordering-perf
固定 T=8 元素/线程、32 lane/warp、f32(4 字节)。三档：
- register(reorder)：0 跨线程搬运、0 barrier、0 shmem 字节。
- lane(warp shuffle)：约 log2(32)=5 次 shfl（v3.2.0 该专用实现 TODO 未做→暂落 shmem）。
- warp/block(shmem 往返)：承接上例 8 store + 8 load + 3 barrier，往返 shmem 每线程 T*4=32 字节/iter。
