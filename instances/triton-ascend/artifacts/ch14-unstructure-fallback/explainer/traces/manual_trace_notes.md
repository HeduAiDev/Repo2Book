# ch14 manual trace notes（trace_source=manual）

**为何 manual**：本章是纯 C++ MLIR pass（TritonToUnstructure），skip_impl，无精简版可跑；宿主无 CANN，
不伪造编译器 dump。所有"数值轨迹"取自 pin 内 **lit 夹具的 RUN+CHECK 前后对照**（编译器
`triton-opt --triton-to-unstructure` 的确定性输出已由夹具 CHECK 钉死）+ **pin 源码常量**（标 file:Lxxx）。
下列每个数字都能在夹具的 CHECK 行或源码行找到。

---

## 夹具 A：unstructure_mix.mlir（部分标量化，头号 worked example）
`third_party/ascend/unittest/Conversion/General/TritonToUnstructure/unstructure_mix.mlir`
RUN: `triton-opt --triton-to-unstructure %s | FileCheck %s`

### 输入 SSA 链（求 %18 = 第二次 load 的指针 的逐维四态）
- `%2 = tt.make_range end=8`  → parseMakeRange → **[structured]**，scalarLike=false  (OffsetAnalysis.cpp:L585-592)
- `%5 = tt.make_range end=16` → **[structured]**，scalarLike=false
- `%7 = tt.addptr(splat %arg1, %5)`；`%8 = tt.load %7 : tensor<16xi64>` → parseLoad → **[unstructured]**，rank1  (OffsetAnalysis.cpp:L629-642，setUnstructured L641)
- `%9 = expand_dims %2 axis=0 : ->1x8` → parseExpandDims：axis0→scalar，其余透传 → **[scalar, structured]**  (OffsetAnalysis.cpp:L718-745)
- `%10 = tt.splat %arg3 : ->1x8` → parseSplat：dim size1→scalar / size8→scalarlike，scalarLike 标志=true → **[scalar, scalarlike]**，scalarLike=true  (OffsetAnalysis.cpp:L493-499)
- `%11 = arith.muli %9,%10` → parseMulI：rhs(%10) scalarLike → 逐维透传 lhs(%9) → **[scalar, structured]**，scalarLike=false  (OffsetAnalysis.cpp:L644-672；透传分支 L666-668)
- `%13 = arith.extsi %11` → 透传 → **[scalar, structured]**
- `%14 = tt.broadcast %13 : 1x8->16x8` → parseBroadcast：被广播维(dim0,1→16)→scalarlike，其余透传 → **[scalarlike, structured]**  (OffsetAnalysis.cpp:L674-711，L708-710)
- `%12 = expand_dims %8 axis=1 : ->16x1` → parseExpandDims：axis1→scalar，dim0 透传 unstructured → **[unstructured, scalar]**
- `%15 = tt.broadcast %12 : 16x1->16x8` → parseBroadcast：被广播维(dim1,1→8)→scalarlike，dim0 透传 unstructured → **[unstructured, scalarlike]**
- `%16 = arith.addi %14,%15` → parseAddI → combineInfo 逐维 min（enum 序 unstructured=0<structured=1<scalarlike=2<scalar=3）  (OffsetAnalysis.cpp:L525-538 → combineInfo L192-204，min L201-202)
  - dim0: min(scalarlike=2, unstructured=0) = **unstructured(0)**
  - dim1: min(structured=1, scalarlike=2) = **structured(1)**
  - scalarLike: false && false = false
  - → **%16 / %18 = [unstructured(dim0=16), structured(dim1=8)]**

### 判定与 codegen（CHECK 钉死）
- matchAndRewrite 早退：%18 isStructured()==false（含 unstructured 维）→ 不放行，继续兜底  (UnstructureConversionPass.cpp:L251-263)
- 对齐检查：结构化尾维 = dim1，size=8，f32=4 字节 → sizeInByte = 4×8 = **32**；32 % 32 == **0** → 对齐，dim1 **保留为向量**，不强制标量化  (UnstructureConversionPass.cpp:L334-342，%32 常量 L341)
- codegen（CHECK L62-70）：
  - dim0(unstructured) → `scf.for %arg = %c0 to %c16 step %c1`（**16 次循环**）  (L408-466 的 scf.for 分支)
  - dim1(structured=8) → `tensor.extract_slice %25[%28,0][1,8][1,1] {DiscreteMemAccess}`（**1×8 向量切片**）
  - 循环内：splat %arg2 → addptr → `tt.load {DiscreteMemAccess}` → `tensor.insert_slice ... [1,8]` 写回 iter_arg
- 量化：**16 次搬运，每次 1×8=8 个连续 f32** vs 结构化路径 **1 次** tensor<16x8> 连续搬运。

### store %28（对照：结构化不被标量化）
`%26 = addi(broadcast(muli %21,%cst), broadcast(%23=expand make_range16))` 全 structured → isStructured()==true 且非 scalarLike → matchAndRewrite `return failure()` 放行结构化路径，store 保持 `tt.store %28 : tensor<16x8x!ptr>` 整块向量。

---

## 夹具 B：nested_loop.mlir（完全标量化 + 循环携带指针 + while 变体）
`third_party/ascend/unittest/Conversion/General/TritonToUnstructure/nested_loop.mlir`
RUN: `triton-opt --triton-to-unstructure --split-input-file %s | FileCheck %s`

- `%10 = tt.load %9 : tensor<128x!ptr<i32>>`（%9 结构化）→ parseLoad → **[unstructured]** rank1，tensor<128xi64>（extsi 后）
- `%12 = tt.addptr(splat %arg3, %10)` → **完全 unstructured** tensor<128>，循环携带进 `scf.for`（iter_arg %arg7）
- `%21 = tt.load %arg11`（%arg11 = %12 携带）→ 兜底：fullyUnstructured（isUnstructuredOrScalarlike==true）
- codegen（CHECK L67-74，@test_kernel）：内层单层 `scf.for %VAL_43 = 0 to 128 step 1`（**128 次**）→ `tensor.extract %VAL_35[%VAL_43] {DiscreteMemAccess}` 逐元素取偏移 → `tt.addptr %arg3(标量), %ext : !ptr<i32>, i64` → `tt.load {DiscreteMemAccess} : !ptr<i32>`（**单元素 load**）→ `tensor.insert_slice ... [1][1]` 写回
- 对照：同体内 `%19 = tt.addptr(%5, %arg9)`（%arg9 结构化，源自 %4）的 `%20 = tt.load %19,%18` **不被标量化**（结构化路径放行，保持 tensor<128> 向量 masked load）
- @test_kernel2 同逻辑走 `scf.while`（CHECK 转换后循环体同样是 scf.for 0→128 + extract + splat i32→tensor<1xi32> + insert_slice）
- 量化：**128 次单元素 load** vs 结构化 **1 次** 连续 load（∏unstructured 维 = 128；本例 element 4 字节，fully unstructured 尾维无结构化维，连续带宽=0 保住）。

---

## 夹具 C：splat.mlir（splatAndLoadScenario，scalarLike ≠ gather）
`third_party/ascend/unittest/Conversion/General/TritonToUnstructure/splat.mlir`
RUN: `triton-opt %s --triton-to-unstructure | FileCheck %s`

- `%offset = const 10 : i64`；`%offset_tensor = tt.splat %offset : ->tensor<128xi64>` → scalarlike
- `%base_tensor = tt.splat %base`；`%ptr = tt.addptr(...)`；`%val = tt.load %ptr : tensor<128x!ptr<f32>>`
- %ptr isScalarLike==true → matchAndRewrite 走 splatAndLoadScenario  (UnstructureConversionPass.cpp:L206-226，触发 L275-280)
- codegen（CHECK L3-6）：`tensor.extract %..[%..] {DiscreteMemAccess} : tensor<128x!ptr<f32>>`（取**第 0 个**单指针）→ `tt.load %ext : !ptr<f32>`（**1 次** load）→ `tt.splat %val : f32 -> tensor<128xf32>`（广播）
- 量化：**1 次** load + 1 次 splat（O(1)）vs 真 gather 的 128 次。**不进循环**。

---

## 源码常量清单（figure/quantified 引用锚）
- 四态 enum：`unstructured, structured, scalarlike, scalar`（声明序=偏序 0/1/2/3）  OffsetAnalysis.h:L76-81
- meet = 逐维 `std::min`  OffsetAnalysis.cpp:L201-202
- parseMakeRange → setStructured(1)  OffsetAnalysis.cpp:L591
- parseLoad → setUnstructured(rank)  OffsetAnalysis.cpp:L641
- parseMulI 双非 scalarLike → unstructured  OffsetAnalysis.cpp:L669-670
- 对齐粒度常量 **32**（字节）  UnstructureConversionPass.cpp:L341
- 兜底触发：forceScalarize || scalarLike || fromTensorArg  UnstructureConversionPass.cpp:L303-306
