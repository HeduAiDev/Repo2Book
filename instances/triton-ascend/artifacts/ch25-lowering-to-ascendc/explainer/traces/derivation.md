# ch25 手工推演台账（trace_source=manual）

本章降的是编译器内部的一个 MLIR conversion pass（`ConvertHIVMToStandardPass`，C++），
无 host 可跑的精简版：库函数名 mangle 与 rank 拆循环都发生在编译期 pattern 重写里，
要观测其输出需构建 `bishengir-opt` 全套 LLVM/MLIR + Ascend 工具链（本环境无）。
故所有数字均**从源码常量直接推演**，逐项标 `file:Lxxx`。真值来源如下。

## 数据类型 → 库名映射（getTypeName）

file: `third_party/ascend/AscendNPU-IR/bishengir/lib/Dialect/HIVM/IR/HIVMImpl.cpp:L456-L496`

| MLIR 类型 | 库名字符串 | 行 |
|---|---|---|
| f16 | half | L477-L478 |
| bf16 | bfloat16_t | L479-L480 |
| f32 | float | L485-L486 |
| f64 | double | L487-L488 |
| i32（有符号）| int32_t | L471 |
| i8 | int8_t | L471 |
| i1 | bool | L461-L462 |

## 内存域 → 库名映射（kAddressSpace2LibraryName）

file: `.../LibraryFunctionOpInterface/LibraryFunctionOpInterfaceImpl.cpp:L86-L89`

- UB → `ubuf`
- GM → `gm`
- L1 → `cbuf`

## 库支持最大 rank（StaticMaxRankExternalModel 注册表）

file: `.../LibraryFunctionOpInterfaceImpl.cpp:L1103-L1177`

- VAddOp = 3 (L1122)，VMulOp = 3 (L1123)
- LoadOp = 3 (L1156)，StoreOp = 3 (L1157)，CopyOp = 3 (L1158)
- NZ2NDOp = 2 (L1159)
- VXorOp = 2 (L1129)，VCastOp = 2 (L1145)，VCumsumOp = 2 (L1142)
- MmadL1Op / ND2NZOp / FixpipeOp / MatmulOp = NoMaxRank（不设 rank，直接调库）(L1160-L1170)

## rank 钳制（getOpLibraryCallRankImpl）

file: `.../LibraryFunctionOpInterfaceImpl.cpp:L47-L53`
`rank' = std::min(rank, maxOpRank)`（L52）。

## 名字拼装规则

- 向量 op `concatVectorOpLibraryCallName`(L30-L37)：`<op>_<rank'>d_<elemType>`
- 搬运 op `getLibraryCallNameForCopyLikeOp`(L91-L113)：`<op>_<srcScope>_to_<dstScope>_<rank'>d_<elemType>`
- MmadL1（无 bias/transpose）`NoMaxRankExternalModel<MmadL1Op>::getOpLibraryCallName`(L876-L913)：
  `<op>_<srcType>_to_<dstType>`（transpose/hf32 时追加 `_ta`/`_tb`/`_hf32`）
- ND2NZ `NoMaxRankExternalModel<ND2NZOp>::getOpLibraryCallName`(L915-L933)：`<op>_<elemType>`（喂 bias 时插 `_forbias`）
- op 助记符（getOpName 默认返回 mnemonic，HIVMBase.td:L67）：
  vadd (HIVMVectorOps.td:L511)、vmul (L546)、load/store/copy (HIVMDMAOps.td:L62/L145/L199)、
  mmadL1 (HIVMMacroOps.td:L163)、nd2nz/nz2nd (HIVMDMAOps.td:L331/L372)

## m2 逐例推演（每步套上面规则）

1. `hivm.hir.vadd` 作用于 1D f16：min(1,3)=1，half → `vadd_1d_half`
2. `hivm.hir.vadd` 作用于 5D f16：min(5,3)=3，half → `vadd_3d_half`（高维靠 m3 拆循环补齐）
3. `hivm.hir.vmul` 作用于 2D f32：min(2,3)=2，float → `vmul_2d_float`
4. `hivm.hir.load` gm→UB 1D f16：srcScope=gm,dstScope=ubuf → `load_gm_to_ubuf_1d_half`
5. `hivm.hir.store` UB→gm 1D f16 → `store_ubuf_to_gm_1d_half`
6. `hivm.hir.mmadL1` src f16 → dst f32，无 transpose/bias → `mmadL1_half_to_float`
7. `hivm.hir.nd2nz` f16，无 bias → `nd2nz_half`

## m3 逐轮推演：vadd 作用于 memref<2x3x4x8x16xf16>（rank=5）

VectorOpToLibraryCallPattern：rank=5 > maxRank=3（VAddOp=3, L1122），走
`reduceMemrefsToNestedFor(..., 0, rank - maxOpRank)` = `(..., 0, 2)`（HIVMToStandard.cpp:L936-L940）。
→ reducedAxes = {0,1}，各生成一层 scf.for；循环体内对每个 memref 建降 rank 的 subview
（HIVMToStandard.cpp:L229-L338）。外 2 轴 size = 2 与 3，内 3 轴 = 4x8x16 保留。

- 循环次数 = 2 × 3 = 6，每次一条 `func.call @vadd_3d_half`
- 每个 subview：offset=[i0,i1,0,0,0]，size=[1,1,4,8,16]，rank-reduced 后 = memref<4x8x16xf16>

| 轮次 | i0∈[0,2) | i1∈[0,3) | subview offset | subview 类型 | 生成的调用 |
|---|---|---|---|---|---|
| 1 | 0 | 0 | [0,0,·] | memref<4x8x16xf16> | func.call @vadd_3d_half |
| 2 | 0 | 1 | [0,1,·] | memref<4x8x16xf16> | func.call @vadd_3d_half |
| 3 | 0 | 2 | [0,2,·] | memref<4x8x16xf16> | func.call @vadd_3d_half |
| 4 | 1 | 0 | [1,0,·] | memref<4x8x16xf16> | func.call @vadd_3d_half |
| 5 | 1 | 1 | [1,1,·] | memref<4x8x16xf16> | func.call @vadd_3d_half |
| 6 | 1 | 2 | [1,2,·] | memref<4x8x16xf16> | func.call @vadd_3d_half |

对照源码注释里的 Example（HIVMToStandard.cpp:L300-L323）：rank=6、reduce 轴 [1,4) → 3 层循环、
subview memref<1x64x128xi16>——同一机制、不同轴集，互为佐证。
