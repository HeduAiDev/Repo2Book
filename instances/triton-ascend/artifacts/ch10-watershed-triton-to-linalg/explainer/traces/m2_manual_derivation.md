# m2 手工推演：PtrAnalysis 还原 add_kernel 的指针算术（无精简版）

**trace_source = manual。** 本章 skip_impl（纯 C++ MLIR pass，无可运行精简版）；宿主无 CANN/NPU
工具链，`triton-opt --triton-to-structured` 无法在取证环境编译运行。故本轨迹是**按 pin @2badfc89e
源码规则逐算子手工推演**，每一步的还原规则与常量都标注到 `file:Lxxx`。参数取值 `BLOCK=4` 与真实
lit 夹具 `unittest/Conversion/General/TritonToStructured/parseMakeRange.mlir`（`tt.make_range {0,4}` →
`tensor<4xi32>`）、`unittest/Conversion/General/TritonToLinalg/legal_stride.mlir`（`sizes: [4, 1]`）一致；
`pid=2` 为选定的非退化标量（`pid=0` 会使 offset=0 退化，故避开）。

## 场景 IR（ttir，add_kernel 的 x 侧地址算术）

```mlir
%pid    = tt.get_program_id x : i32                       // pid = 2
%bs     = arith.muli %pid, %cBLOCK : i32                  // block_start = 2 * 4 = 8（scalar×scalar）
%range  = tt.make_range {start = 0 : i32, end = 4 : i32}  : tensor<4xi32>
%bs_t   = tt.splat %bs   : i32          -> tensor<4xi32>
%off    = arith.addi %bs_t, %range      : tensor<4xi32>   // offsets = [8,9,10,11]
%xptr_t = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
%addr   = tt.addptr %xptr_t, %off : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
```

PtrAnalysis 从最外层 `tt.addptr` 的 `visitOperandAddptr` 进入，后序递归下潜其操作数 DAG。
PtrState 三元组记法：`{source, offset, stateInfo=[(stride, size)...]}`。

## 逐算子还原（后序）

| 轮次 | 当前算子 | 还原规则(函数) | 得到的 PtrState | 源码依据 |
|---|---|---|---|---|
| 1 | `tt.make_range {0,4}` | visitOperandMakeRange | `{src=∅, off=0, [(stride=1, size=4)]}` | stride=(4-0+4-1)/4=1 @PtrAnalysis.cpp:L778,L786;size=shape[0]=4 @L787;off=start=0 @L788 |
| 2 | `tt.splat %bs`(标量 8) | visitOperandSplat | `{src=∅, off=8, [(stride=0, size=4)]}` | 标量 splat 各维 stride=0 @L908,L913;off=标量值 8（visitOperand(src) 先取得） |
| 3 | `arith.addi`(#2,#1) | visitOperandAdd→addState | `{src=∅, off=8, [(stride=1, size=4)]}` | 逐维 stride 相加 0+1=1 @L561;offset 相加 8+0=8 @L578;size min(4,4)=4 |
| 4 | `tt.splat %x_ptr`(指针) | visitOperandSplat | `{src=x_ptr, off=0, [(stride=0, size=4)]}` | 指针 splat：source=x_ptr、各维 stride=0 @L908,L913 |
| 5 | `tt.addptr`(#4,#3) | visitOperandAddptr→addState | `{src=x_ptr, off=8, [(stride=1, size=4)]}` | ptr 态+offset 态 addState @PtrAnalysis.cpp:L279;stride 0+1=1 @L561;off 0+8=8 @L578 |

## 落地（m3，交由 BlockPtrAnalysis 侧同构解析后物化）

最终结构化三元组 `(source=x_ptr, offset=8, sizes=[4], strides=[1])` 经 `createCastOp` 铸成：

```mlir
%v = memref.reinterpret_cast %x_ptr to offset: [8], sizes: [4], strides: [1]
       : memref<?xf32> to memref<4xf32, strided<[1], offset: 8>>
```

对照真实夹具 `legal_stride.mlir` 的 CHECK 行形态一致：
`memref.reinterpret_cast %arg2 to offset: [%arg13], sizes: [4, 1], strides: [%c4, %c1]`
（该夹具是 2D 广播嵌套场景，offset 为循环 iter-arg；本推演是 1D 常量 offset 的最小主线）。

## 关键标量核对

- pid = 2（选定非退化标量，避开 pid=0 的 offset=0 退化）
- block_start = pid × BLOCK = 2 × 4 = 8
- offsets 元素 = [8, 9, 10, 11]（off=8，stride=1，size=4）
- 最终：offset=8, size=4, stride=1
